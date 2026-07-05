# Investigate a Crash

When the scheduler under test dies mid-run, the test fails with a
structured crash report: the scx exit reason, a kernel-side
backtrace, per-CPU scheduler state, and — by default — the results
of a second VM that ktstr boots automatically to replay the crash
with probes attached. This recipe walks a real crash from report to
regression test.

## First step: rerun with full diagnostics

```sh
RUST_BACKTRACE=1 cargo ktstr test --kernel 7.0 -- -E 'test(my_test)'
```

`RUST_BACKTRACE=1` (or `full`) does two things: it appends the
`--- diagnostics ---` section (init stage, VM exit code, kernel
console tail) to every failure — not only scheduler deaths — and it
boots the guest with a verbose console (the same switch
`KTSTR_VERBOSE=1` flips).

## The crash report

A failure prints as a header line plus sections, each present only
when relevant: `--- stats ---` (per-cgroup worker results),
`--- diagnostics ---`, `--- timeline ---` (phases with monitor
samples), `--- scheduler log ---` (scheduler stdout+stderr,
including the kernel's `DEBUG DUMP` when the scheduler died),
`--- monitor ---` (host-side observations and verdict),
`--- sched_ext dump ---` (the same dump as traced by the guest
kernel), and `--- auto-repro ---`. See
[Reading Failure Output](../running-tests/failures.md) for the full
anatomy; this recipe focuses on the crash workflow.

## A real crash, top to bottom

The crash below is real: ktstr's fixture scheduler was told to call
`scx_bpf_error()` via a host-written `.bss` trigger (the same
mechanism [step 7 of Test a New Scheduler](test-new-scheduler.md)
uses). Trimmed to the load-bearing sections:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration,wprof -E 'test(=ktstr/bpf_crash_auto_repro_e2e)' --no-capture | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel 7.0 -- -E 'test(=ktstr/bpf_crash_auto_repro_e2e)'</span></div>

<pre><span class="t-red">BUG SUMMARY: scx_bpf_error (src/bpf/main.bpf.c:424: ktstr: host-triggered crash)</span>
ktstr_test 'bpf_crash_auto_repro_e2e' [sched=scx-ktstr] [topo=1n1l4c1t] failed:
  scheduler process died unexpectedly during workload (2.2s into test)
...
--- scheduler log ---
...
DEBUG DUMP
================================================================================

swapper/3[0] triggered exit kind 1025:
  <span class="t-yel">scx_bpf_error (src/bpf/main.bpf.c:424: ktstr: host-triggered crash)</span>

Backtrace:
  scx_exit+0x50/0x70
  scx_bpf_error_bstr+0x78/0x90
  <span class="t-b">bpf_prog_1fed99378f3a8055_ktstr_dispatch+0x4d/0x1cb</span>
  bpf__sched_ext_ops_dispatch+0x4b/0xa7
  do_pick_task_scx+0x379/0x770
  __schedule+0x5ca/0xfc0
  schedule+0x44/0x1b0
  worker_thread+0xa2/0x2d0
  kthread+0xf3/0x130
  ret_from_fork+0x19b/0x260
  ret_from_fork_asm+0x1a/0x30

ktstr scheduler state:
  stall=0 <span class="t-b">crash=1</span> degrade_rt=0
  rodata: degrade=0 slow=0 scattershot=0 verify_loop=0 fail_verify=0
  ktstr_alloc_count=87 degrade_cnt=0 slow_cnt=0
...
Error: EXIT: scx_bpf_error (src/bpf/main.bpf.c:424: ktstr: host-triggered crash)</pre></div>

Three lines identify the cause:

- The `BUG SUMMARY` / exit line carries the message and C source
  line your scheduler passed to `scx_bpf_error()`.
- The backtrace names the BPF program that raised it
  (`…ktstr_dispatch`) and shows it fired from the kernel's pick
  path (`do_pick_task_scx` inside `__schedule`).
- The `ktstr scheduler state` block is the scheduler's own dump
  callback output — whatever your scheduler prints in its `.dump()`
  op appears here. In this case `crash=1` confirms the host-written
  trigger was read.

## Auto-repro

`auto_repro` defaults to `true` in `#[ktstr_test]`. When the
scheduler crashes, ktstr automatically:

1. Captures the crash stack trace from the scenario output.
2. Boots a second VM with kprobes (kernel functions) and fentry
   probes (BPF callbacks) on each function in the crash chain, plus
   a `tp_btf/sched_ext_exit` tracepoint trigger.
3. Reruns the scenario to capture function arguments at each crash
   point.

The `--- auto-repro ---` section starts with the probe pipeline's
own accounting, so you can always see how much of the chain
attached. On a kernel with the `sched_ext_exit` tracepoint, the same
crash replayed under probes:

<!-- captured: cargo ktstr test --kernel local-8cd2b47 (v7.1 + sched_ext_exit tracepoint) --features integration,wprof -E 'test(=ktstr/bpf_crash_auto_repro_e2e)' --no-capture; ktstr 0.23.0 + pending trigger-signature fix and probe-ship hold | full run: captures/autorepro-live.txt -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">--- auto-repro --- (repro VM, probes on the crash chain)</span></div>

<pre>--- probe pipeline ---
  extracted:   9 functions from crash backtrace
  traceable:   4 passed, 4 dropped: bpf_prog_9a11f2edaac0b52f_ktstr_dispatch, ...
  kprobes:     5 attached
  fentry:      11 attached
  <span class="t-grn">trigger:     fired (tp_btf)</span>
  events:      125 captured, 8 after stitch

=== AUTO-PROBE: scx_exit fired ===

  ktstr_select_cpu                                              main.bpf.c:380
    task_struct *p
      pid                 40
      <span class="t-grn">dsq_id              SCX_DSQ_INVALID  →  SCX_DSQ_LOCAL</span>
      <span class="t-grn">slice               19982063         →  20000000</span>
      scx_flags           RESET_RUNNABLE_AT|DEQD_FOR_SLEEP|ENABLED
  do_enqueue_task                                               kernel/sched/ext.c:1885
    rq *rq
      cpu                 0
    task_struct *p
      pid                 40
      dsq_id              SCX_DSQ_LOCAL
      scx_flags           QUEUED|DEQD_FOR_SLEEP|ENABLED</pre></div>

This is the chain to the error, not just the point of death: each
probed function shows the task state it saw (fexit arrows mark what
changed inside the call), so you can watch scheduling decisions
progress right up to the frame where `ktstr_dispatch` raised
`scx_bpf_error()`. Entry state that never got its exit pair (the
crash interrupted it) still prints — the last rows are the
scheduler's final acts.

Without the tracepoint the trigger cannot attach and the section
reports it plainly (`trigger: attach failed ... No such process`)
while still appending the repro VM's diagnostics — sched_ext dump,
dmesg tail, and the failure dump when produced. Cost either way:
one extra VM boot plus a scenario replay (~10-17 s here).
Auto-repro is skipped when `expect_err = true` — an expected
failure is not worth a repro VM — and can be turned off with
`auto_repro = false`. See
[Auto-Repro](../running-tests/auto-repro.md) for the two-VM cycle
and kernel requirements.

## Pin the bug as a regression test

Once the crash is understood, pin its signature so the same bug
fails the next CI run instead of silently regressing. Two
`#[ktstr_test]` attributes attach matchers to the captured
`scx_bpf_error` text (the combined scheduler log and
`--- sched_ext dump ---` corpus):

- `expect_scx_bpf_error_contains = "literal"` — substring match.
  Use for the common case of pinning an exact error fragment
  without escaping regex metacharacters.
- `expect_scx_bpf_error_matches = "regex"` — full regex match via
  the `regex` crate. Use for anchored patterns, character classes,
  and wildcards.

Both require `expect_err = true`, and both compose with AND
semantics — set both to require both to match. The test body must
actually set up the trigger; here the host-side `BpfMapWrite` from
the crash above does it:

```rust,ignore
use ktstr::prelude::*;

// Host writes 1 into the scheduler's `crash` .bss global after
// load; the scheduler's dispatch path reads it and calls
// scx_bpf_error(...) — the crash this test pins.
static BPF_CRASH: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);

#[ktstr_test(
    scheduler = MY_SCHED,
    bpf_map_write = BPF_CRASH,
    expect_err = true,
    expect_scx_bpf_error_contains = "host-triggered crash",
    expect_scx_bpf_error_matches = r"src/bpf/main\.bpf\.c:\d+",
)]
fn crash_regression(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::basic::custom_crash_light(ctx)
}
```

The test fails if either matcher misses. A passing run means the
scheduler still hits the pinned bug; a failure means the error text
drifted (update the matcher) or the bug was fixed (delete the
regression test).

Regex anchors use string-boundary semantics against the whole
captured corpus: `^`/`$` match its start and end, and `.` does not
cross `\n`. Opt in to line-level anchoring with `(?m)` and to
dotall with `(?s)`. Pattern-validity edge cases (empty patterns,
trivially-matching patterns, builder vs attribute validation) are
covered in the
[`#[ktstr_test]` reference](../writing-tests/ktstr-test-macro.md).
