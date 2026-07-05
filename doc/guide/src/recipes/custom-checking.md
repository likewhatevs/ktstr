# Customize Checking

Override [checking](../concepts/checking.md) thresholds for
schedulers that tolerate higher imbalance, different gap
thresholds, or relaxed event rates — and opt in to the checks that
are off by default.

> [!WARNING]
> `Assert::default_checks()` is `Assert::NO_OVERRIDES` — every
> field `None`. Until a scheduler-level or per-test override sets a
> threshold, no worker assertions run. A green suite with no
> overrides proves only that the VM booted and the scheduler didn't
> crash.

## What a tripped gate looks like

Here a test set `min_iteration_rate` to a floor the workload could
never meet (deliberately, to force the failure). The report names
each worker that missed the gate, with the measured rate and the
floor it was compared against:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
    ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:
      worker 71 iteration rate 41903.3/s below floor 50000000.0/s
      worker 73 iteration rate 37834.5/s below floor 50000000.0/s

    --- stats ---
    2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
      cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
      cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
...
    --- monitor ---
    samples=41 max_imbalance=2.00 max_dsq_depth=0 stuck=0
    avg: imbalance=1.32 nr_running/cpu=1.2 dsq/cpu=0.0
    events: fallback=0 (0.0/s) keep_last=210 (52.5/s) offline=0
    events+: refill_slice_dfl=210
    schedstat: csw=586 (146/s) run_delay=381246314ns/s ttwu=204 goidle=1
    bpf: ktstr_select_cp cnt=189 145ns/call
    bpf: ktstr_enqueue cnt=373 34ns/call
    bpf: ktstr_dispatch cnt=584 237ns/call
    verdict: monitor OK
```

Note the two channels: the *worker* gate tripped (the two `below
floor` lines) while the *monitor* verdict is OK — worker checks and
host-side monitor checks are evaluated independently. See
[Checking](../concepts/checking.md) for the model. The fix is
whichever of these matches the intent: set a floor the scheduler
can actually meet, or fix the scheduler until it meets the floor
you wrote.

## Scheduler-level overrides

Declare a scheduler with assertion overrides that apply to every
test using it:

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(RELAXED, {
    name = "relaxed",
    binary = "scx_relaxed",
    assert = Assert::NO_OVERRIDES
        .max_imbalance_ratio(5.0)    // tolerate 5:1 imbalance
        .max_fallback_rate(500.0)    // higher fallback rate ok
        .fail_on_stall(false),       // don't fail on stall
});
```

These are the first layer that can carry an actual check — without
them (or a per-test override), nothing asserts.

## Per-test overrides

Attributes on `#[ktstr_test]` merge last and win:

```rust,ignore
#[ktstr_test(
    scheduler = RELAXED,
    not_starved = true,
    max_gap_ms = 5000,
    max_imbalance_ratio = 10.0,
    sustained_samples = 10,
)]
fn high_imbalance_test(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits topology from RELAXED
    Ok(AssertResult::pass())
}
```

`not_starved = true` enables the starvation, fairness-spread, and
scheduling-gap checks as a group; each threshold can still be
overridden independently. The full attribute list and default
thresholds live in the
[`#[ktstr_test]` reference](../writing-tests/ktstr-test-macro.md).

## Merge order

The runtime evaluates
`Assert::default_checks().merge(&scheduler.assert).merge(&test.assert)`
— three layers, last-`Some`-wins per field. Worked example:

- scheduler layer: `max_imbalance_ratio(5.0)`
- test layer: `max_imbalance_ratio = 10.0`
- effective: `10.0` — the test's `Some` wins; every field the test
  leaves `None` falls through to the scheduler layer, then to the
  (all-`None`) defaults.

To see the merged result for a registered test without reading
source:

```sh
cargo ktstr show-thresholds high_imbalance_test
```

It prints every threshold field of the exact `Assert` the runtime
will evaluate, with `none` for unset fields.

## Using Assert directly in ops scenarios

```rust,ignore
fn my_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_gap_ms(3000);

    let steps = vec![/* ... */];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

`execute_steps_with` applies the given `Assert` for worker checks,
overriding the merged config. `execute_steps` (without `_with`)
passes `None` and falls back to `ctx.assert` — the merged
three-layer config above. Reaching for `_with` when you meant to
*add* to the merged config is a classic trap: the explicit `Assert`
replaces `ctx.assert`, it does not compose with it.

See [Ops, Steps, and Backdrop](../concepts/ops.md) for the step
execution model.
