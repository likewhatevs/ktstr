# Benchmark Gates and Negative Tests

A performance gate you have never seen fail proves nothing. This
recipe builds gates in pairs: a positive test that holds the
scheduler to a floor, and a negative twin that degrades the
scheduler on purpose and asserts the same gate trips. (Extracting
metrics from benchmark payloads like schbench or fio is covered in
[Payloads and Included Files](../writing-tests/payloads.md);
cross-commit regression gates are
[A/B Compare Branches](ab-compare.md).)

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Positive</strong><p>Set a realistic floor or ceiling on the scheduler behavior you expect.</p></div>
<div class="kt-doc-card"><strong>Negative</strong><p>Deliberately degrade the fixture so the gate proves it can fail.</p></div>
<div class="kt-doc-card"><strong>Regression</strong><p>Reuse the same metric in sidecar-based comparisons against a baseline commit.</p></div>
</div>

## Positive: gate a scenario

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 1, 2, 1),
});

#[ktstr_test(
    scheduler = MY_SCHED,
    performance_mode = true,
    duration_s = 5,
    sustained_samples = 15,
)]
fn perf_positive(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step::with_defs(
        vec![CgroupDef::named("cg_0").workers(2)],
        HoldSpec::FULL,
    )];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

Key points:

- `performance_mode = true` pins vCPUs to reserved host cores so
  the measurement isn't host noise — see
  [Performance Mode](../concepts/performance-mode.md).
- `Assert::default_checks()` is all-`None`: every gate is opt-in,
  and nothing fails until you chain a setter. Threshold layering
  and merge order live in [Customize Checking](custom-checking.md).
- `execute_steps_with` applies the `Assert` during worker checks.

A passing gated run looks like any passing run:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/sched_basic_proportional)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
cargo ktstr: resolved kernel "7.0"
...
    Starting 1 test across 121 binaries (12531 tests skipped)
        PASS [  22.158s] (1/1) ktstr::ktstr_sched_tests ktstr/sched_basic_proportional
────────────
     Summary [  22.197s] 1 test run: 1 passed, 12531 skipped
```

## Choosing the floor

Don't guess thresholds. Run the scenario once with no gates, read
the observed per-cgroup iteration counts from the `--- stats ---`
block (or the run's stats sidecar), and set the floor at a healthy
margin below the observed rate — far enough down that run-to-run
noise never trips it, close enough up that a real regression does.
Tighten later once
[`perf-delta --noise-adjust`](ab-compare.md) has shown you the
actual run-to-run spread.

## Negative: prove the gate fires

`expect_err = true` inverts the harness: the test must fail. An
`Ok` return panics with `expected test to fail but it passed`, so a
gate that silently stopped firing turns the negative test red.
Skips are handled separately: a run that could not boot a kernel or
lost the host-resources race emits a `SKIP` banner and does not
count as the expected failure — an environment problem is never
mistaken for proof that the gate fired (the skip-vs-fail taxonomy
lives in [Troubleshooting](../troubleshooting.md)). Auto-repro is
disabled automatically for expected-error tests.

```rust,ignore
#[ktstr_test(
    scheduler = MY_SCHED,
    performance_mode = true,
    duration_s = 5,
    extra_sched_args = ["--degrade"],
    expect_err = true,
)]
fn perf_negative(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step::with_defs(
        vec![CgroupDef::named("cg_0").workers(2)],
        HoldSpec::FULL,
    )];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

`extra_sched_args` passes CLI args to the scheduler binary.
`--degrade` is a real knob on ktstr's fixture scheduler
(`scx-ktstr`) that deliberately worsens its scheduling; the fixture
also exposes `--slow`, `--stall-after`, and `--fail-verify` for
other failure classes. Substitute your own scheduler's equivalent —
a degradation flag your scheduler ships for exactly this purpose is
a feature, not a wart.

When the degraded run trips the gate, the failure the harness
expects looks like the real thing, because it is:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
    ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:
      worker 71 iteration rate 41903.3/s below floor 50000000.0/s
      worker 73 iteration rate 37834.5/s below floor 50000000.0/s
```

The in-repo pattern to copy is `tests/assert_gate_matrix.rs`: a
macro stamps out a positive and a negative variant for each of ten
worker gates (`max_p99_wake_latency_ns`, `max_wake_latency_cv`,
`min_iteration_rate`, `max_gap_ms`, …), with the negative variant
passing `--degrade` and setting `expect_err`. Each gate in the
matrix is proven to fire by its own negative test — hold your gates
to the same standard.

## Related

- [Payloads and Included Files](../writing-tests/payloads.md) —
  metric extraction from real benchmark binaries, declarative
  include files.
- [A/B Compare Branches](ab-compare.md) — the cross-commit
  complement: gates on the delta between two commits instead of an
  absolute floor.
- [Customize Checking](custom-checking.md) — threshold layers and
  merge order.
