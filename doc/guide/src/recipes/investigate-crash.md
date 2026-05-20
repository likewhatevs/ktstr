# Investigate a Crash

When a scheduler crashes during a test, the failure output and
auto-repro pipeline help identify the cause.

## First step: enable full diagnostics

Rerun the failing test with `RUST_BACKTRACE=1` before digging into
individual sections:

```sh
RUST_BACKTRACE=1 cargo ktstr test --kernel ../linux -- -E 'test(my_test)'
```

Setting `RUST_BACKTRACE=1` unconditionally appends the
`--- diagnostics ---` section (init stage, VM exit code, last lines
of kernel console) to every failure, not only when the scheduler
self-dies. It also enables verbose VM console output (equivalent to
`KTSTR_VERBOSE=1`).

## Reading failure output

A test failure message contains up to eight sections, each present
only when relevant:

| Section | Content |
|---|---|
| Error line | Test name, scheduler, failure reason. |
| `--- stats ---` | Per-cgroup worker count, CPU count, spread, gap, migrations, iterations. |
| `--- diagnostics ---` | Init stage classification, VM exit code, last 20 lines of kernel console. |
| `--- timeline ---` | Kernel version, topology, scheduler, scenario duration, phase breakdown with monitor samples. |
| `--- scheduler log ---` | Scheduler process stdout+stderr (cycle-collapsed). |
| `--- monitor ---` | Host-side monitor: sample count, max imbalance ratio, max local-DSQ depth, sustained-violation flag, SCX event counters (`select_cpu_fallback`, `dispatch_keep_last`, `enq_skip_exiting`, `enq_skip_migration_disabled`), per-sched_domain load-balance rates, per-BPF-program `verified_insns`, and the merged threshold verdict. |
| `--- sched_ext dump ---` | `sched_ext_dump` trace lines from the guest kernel. |
| `--- auto-repro ---` | BPF probe data from a second VM run, plus repro VM duration, scheduler log, sched_ext dump, and dmesg tails. |

`--- diagnostics ---` appears automatically when the scheduler died
or crashed, or when `RUST_BACKTRACE` is set to `1` or `full`.

## Auto-repro

`auto_repro` defaults to `true` in `#[ktstr_test]`. When the scheduler
crashes, ktstr automatically:

1. Captures the crash stack trace from the scenario output.
2. Boots a second VM with BPF kprobes (kernel functions) and fentry
   probes (BPF callbacks) on each function in the crash chain, plus
   a `tp_btf/sched_ext_exit` tracepoint trigger.
3. Reruns the scenario to capture function arguments at each crash
   point.

## Reading auto-repro output

The probe output shows each function in the crash chain with:

- Function signature and argument values during execution of the same workload
- Source file and line number
- Call chain context

After the probe data, the auto-repro section includes the repro VM
duration and the last 40 lines of the repro VM's scheduler log
(cycle-collapsed), sched_ext dump, and kernel console (dmesg). These
supplement probe data when the crash produces sparse or no probe events.
When probe data is absent, a crash reproduction status line replaces it.

See [Auto-Repro](../running-tests/auto-repro.md) for details on how
the two-VM repro cycle works.

## Pin a known error as a regression test

Once auto-repro identifies the crash, pin its signature as a
regression test so the same bug fails the next CI run instead of
silently regressing. Two `#[ktstr_test]` attributes attach matchers
to the captured `scx_bpf_error` text (the combined scheduler log and
`--- sched_ext dump ---` corpus):

- `expect_scx_bpf_error_contains = "literal"` — substring match.
  Use for the common case of pinning an exact error fragment
  without escaping regex metacharacters.
- `expect_scx_bpf_error_matches = "regex"` — full regex match
  via the `regex` crate. Use for anchored patterns, character
  classes, and wildcards.

Both attributes require `expect_err = true` (a reproducer matcher
narrows which failure counts as the expected bug and only applies
to expected-error tests). Both compose via AND semantics — set both
to require BOTH to match.

```rust,ignore
const APPLY_CELL_CONFIG_EINVAL: &str = r"apply_cell_config returned -E[A-Z]+";

#[ktstr_test(
    scheduler = MITOSIS,
    expect_err = true,
    expect_scx_bpf_error_contains = "apply_cell_config",
    expect_scx_bpf_error_matches = APPLY_CELL_CONFIG_EINVAL,
)]
fn mitosis_apply_cell_config_einval_regression(ctx: &Ctx) -> Result<AssertResult> {
    // Test body sets up the conditions that trigger the bug.
    Ok(AssertResult::pass())
}
```

The test fails if EITHER matcher misses. A passing run means the
scheduler still hits the pinned bug; a failure means the error text
drifted (rename the matcher to the new text) or the bug was fixed
(delete the regression test).

Regex anchors use string-boundary semantics by default: `^` matches
the start of the WHOLE captured corpus, `$` matches the end, and `.`
does NOT cross `\n`. For line-level anchoring inside the multi-line
corpus, opt in with the inline `(?m)` flag (e.g. `(?m)^apply_cell_config$`);
for `.` to span line breaks, use `(?s)`. Whitespace in the pattern
is matched byte-for-byte — no trim or normalization.

Empty patterns, invalid regex syntax, and any pattern satisfying
`is_match("")` all panic at construction. `is_match("")` catches
two no-op classes with one check: patterns that match every
position (e.g. `a?`, `.*`, `(?:)`) trivially pass against any
corpus, and patterns that match only the empty string (e.g. `^$`)
trivially fail against any non-empty corpus — every real captured
scheduler-output corpus is non-empty, so both are equally useless
pins. Bare `\b` (word boundary) slips this gate because the empty
string contains no word characters; use a substring of the
expected error text instead of a bare boundary assertion. See the
[`#[ktstr_test]` reference](../writing-tests/ktstr-test-macro.md#checking)
for the full attribute list.
