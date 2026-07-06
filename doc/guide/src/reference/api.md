# API Reference

The guide and the rustdoc split the work: the guide explains why and
when to reach for an API; the rustdoc carries signatures, field
semantics, and edge cases. The complete rustdoc for every ktstr
workspace crate is published at
[ktstr.dev/rustdoc/ktstr](https://ktstr.dev/rustdoc/ktstr/), and
[`ktstr::prelude`](https://ktstr.dev/rustdoc/ktstr/prelude/index.html)
re-exports everything a test author needs.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Author tests</strong><p>Use the guide for flow and rustdoc for exact signatures.</p></div>
<div class="kt-doc-card"><strong>Find modules</strong><p>The table maps every common task to the crate API and relevant chapter.</p></div>
<div class="kt-doc-card"><strong>Stay current</strong><p>Pre-1.0 API details move quickly; rustdoc is the signature source of truth.</p></div>
</div>

| You want to | Reach for | Rustdoc | Guide chapter |
|---|---|---|---|
| Declare a test | `#[ktstr_test]` | [attr.ktstr_test](https://ktstr.dev/rustdoc/ktstr/attr.ktstr_test.html) | [The #\[ktstr_test\] Attribute](../writing-tests/ktstr-test-macro.md) |
| Declare a scheduler | `declare_scheduler!`, `Scheduler`, `SchedulerSpec` | [macro.declare_scheduler](https://ktstr.dev/rustdoc/ktstr/macro.declare_scheduler.html), [test_support](https://ktstr.dev/rustdoc/ktstr/test_support/index.html) | [Scheduler Definitions](../writing-tests/scheduler-definitions.md) |
| Drive a scenario | `Ctx`, `scenarios::*` | [scenario](https://ktstr.dev/rustdoc/ktstr/scenario/index.html) | [Scenarios](../concepts/scenarios.md), [Custom Scenarios](../writing-tests/custom-scenarios.md) |
| Compose steps and ops | `Step`, `Op`, `HoldSpec`, `Backdrop` | [scenario::ops](https://ktstr.dev/rustdoc/ktstr/scenario/ops/index.html) | [Ops, Steps, and Backdrop](../concepts/ops.md) |
| Shape cgroups and cpusets | `CgroupDef`, `CpusetSpec` | [scenario::ops](https://ktstr.dev/rustdoc/ktstr/scenario/ops/index.html) | [Topology](../concepts/topology.md) |
| Check results | `Assert`, `AssertResult`, `Verdict`, `claim!` | [assert](https://ktstr.dev/rustdoc/ktstr/assert/index.html) | [Checking](../concepts/checking.md), [Customize Checking](../recipes/custom-checking.md) |
| Gate performance regressions | `PerfDeltaAssertion` | [test_support](https://ktstr.dev/rustdoc/ktstr/test_support/index.html) | [Assertable Metrics](assertable-metrics.md) |
| Generate load | `WorkType`, `WorkloadConfig`, `WorkloadHandle` | [workload](https://ktstr.dev/rustdoc/ktstr/workload/index.html) | [Work Types](../concepts/work-types.md), [Workers and Workloads](../architecture/workers.md) |
| Run guest binaries | `#[derive(Payload)]`, `Payload` | [derive.Payload](https://ktstr.dev/rustdoc/ktstr/derive.Payload.html) | [Payloads and Included Files](../writing-tests/payloads.md) |
| Capture guest state | `Snapshot`, `SnapshotBridge`, `Sample`, `SampleSeries` | [scenario::snapshot](https://ktstr.dev/rustdoc/ktstr/scenario/snapshot/index.html), [scenario::sample](https://ktstr.dev/rustdoc/ktstr/scenario/sample/index.html) | [Snapshots](../writing-tests/snapshots.md), [Temporal Assertions](../writing-tests/temporal-assertions.md) |

The `#[ktstr_test]` attribute's arguments (topology dimensions,
thresholds, execution flags) are documented in the
[macro reference chapter](../writing-tests/ktstr-test-macro.md) and on
the attribute's own
[rustdoc page](https://ktstr.dev/rustdoc/ktstr/attr.ktstr_test.html);
the macro itself lives in the `ktstr-macros` crate and is re-exported
at the crate root.
