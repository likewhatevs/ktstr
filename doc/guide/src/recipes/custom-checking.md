# Customize Checking

Override default [checking](../concepts/checking.md) thresholds
for schedulers that tolerate higher imbalance, different gap thresholds,
or relaxed event rates.

## Scheduler-level overrides

Declare a scheduler with custom assertion overrides:

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(RELAXED, {
    name = "relaxed",
    binary = "scx_relaxed",
    assert = Assert::NO_OVERRIDES
        .max_imbalance_ratio(5.0)    // tolerate 5:1 imbalance
        .max_fallback_rate(500.0)     // higher fallback rate ok
        .fail_on_stall(false),        // don't fail on stall
});
```

These scheduler-level overrides are the FIRST layer that can carry
any actual check — `Assert::default_checks()` is `Self::NO_OVERRIDES`
(every field `None`), so without either a scheduler-level or a
per-test override, no assertions run. Per-test overrides on
`#[ktstr_test]` merge LAST and win.

For `post_vm` callbacks that synthesize their own `Assert` and
need to surface the verdict back through the test result, use
`v.into_anyhow_or_log()` on the resulting `Verdict` — it returns
`Ok(())` on pass and `Err(_)` (with full failure-detail rendering)
on fail, which is what a `post_vm` body wants. The older
`AssertResult::into_anyhow_or_log` does the same for callers that
already hold an `AssertResult` directly.

## Per-test overrides via #\[ktstr_test\]

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

## Understanding not_starved

`not_starved = true` enables starvation, fairness spread, and
scheduling gap checks. Each threshold can be overridden independently.
See [Checking: Worker checks](../concepts/checking.md#worker-checks)
for details and default thresholds.

## Merge order

Three-layer merge with last-`Some`-wins semantics. See
[Checking: Merge layers](../concepts/checking.md#merge-layers).

## Using Assert directly in ops scenarios

```rust,ignore
fn my_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let assertions = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_gap_ms(3000);

    let steps = vec![/* ... */];
    execute_steps_with(ctx, steps, Some(&assertions))
}
```

`execute_steps_with` applies the given `Assert` for worker checks.
`execute_steps` (without `_with`) passes `None`, falling back to
`ctx.assert` (the merged three-layer config: `default_checks` ->
scheduler -> per-test).

For post_vm callbacks that build assertions via `Verdict`, use
`v.into_anyhow_or_log()` to close the verdict — it returns
`Ok(())` on pass and `Err(_)` (with full failure-detail rendering)
on fail, which is exactly what a `post_vm` callback wants. The
older `AssertResult::into_anyhow_or_log` does the same for the
direct-AssertResult path.

See [Ops and Steps](../concepts/ops.md) for the full step execution
model.
