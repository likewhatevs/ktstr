# Summary

[Overview](overview.md)
[ktstr in Action](features.md)

# Getting Started

- [Getting Started](getting-started.md)
- [Tutorial: Zero to ktstr](tutorial.md)

# Writing Tests

- [Writing Tests](writing-tests.md)
  - [The #\[ktstr_test\] Attribute](writing-tests/ktstr-test-macro.md)
  - [Scheduler Definitions](writing-tests/scheduler-definitions.md)
  - [Payloads and Included Files](writing-tests/payloads.md)
  - [Custom Scenarios](writing-tests/custom-scenarios.md)
  - [Snapshots](writing-tests/snapshots.md)
  - [Watch Snapshots](writing-tests/watch-snapshots.md)
  - [Periodic Capture](writing-tests/periodic-capture.md)
  - [Temporal Assertions](writing-tests/temporal-assertions.md)

# Running Tests

- [Running Tests](running-tests.md)
  - [cargo ktstr](running-tests/cargo-ktstr.md)
  - [Gauntlet](running-tests/gauntlet.md)
  - [BPF Verifier Sweep](running-tests/verifier.md)
  - [Reading Failure Output](running-tests/failures.md)
  - [Auto-Repro](running-tests/auto-repro.md)
  - [Runs and Regression Gates](running-tests/runs.md)

# Core Concepts

- [Core Concepts](concepts.md)
  - [Scenarios](concepts/scenarios.md)
  - [Ops, Steps, and Backdrop](concepts/ops.md)
  - [Work Types](concepts/work-types.md)
  - [Checking](concepts/checking.md)
  - [Topology](concepts/topology.md)
  - [MemPolicy](concepts/mem-policy.md)
  - [Performance Mode](concepts/performance-mode.md)
  - [Resource Budget](concepts/resource-budget.md)

# Recipes

- [Recipes](recipes.md)
  - [Test a New Scheduler](recipes/test-new-scheduler.md)
  - [Investigate a Crash](recipes/investigate-crash.md)
  - [A/B Compare Branches](recipes/ab-compare.md)
  - [Capture and Compare Host State](recipes/host-state.md)
  - [Diagnose a Slow Scheduler with ctprof](recipes/diagnose-slow-scheduler.md)
  - [Customize Checking](recipes/custom-checking.md)
  - [Benchmark Gates and Negative Tests](recipes/benchmarking-tests.md)
  - [Compare a Scheduler vs EEVDF](recipes/scheduler-vs-eevdf.md)

# Architecture

- [Architecture Overview](architecture.md)
  - [VMM](architecture/vmm.md)
  - [Monitor](architecture/monitor.md)
  - [Workers and Workloads](architecture/workers.md)
  - [CgroupManager](architecture/cgroup-manager.md)
  - [CgroupGroup](architecture/cgroup-group.md)

# Reference

- [CI](ci.md)
- [Troubleshooting](troubleshooting.md)
- [Environment Variables](reference/environment-variables.md)
- [ctprof](reference/ctprof.md)
- [ktstr (standalone)](running-tests/ktstr.md)
- [Assertable Metrics](reference/assertable-metrics.md)
- [API Reference](reference/api.md)
