# Auto-Repro

The crash log tells you where the scheduler died; auto-repro tells
you what the state was on the way there. When a test fails because
the scheduler crashed or exited, ktstr boots a second VM, reruns the
scenario with BPF probes attached to the functions from the crash
backtrace, and prints each probed call with decoded arguments and
struct fields — entry and exit values side by side. The trail
appears in the `--- auto-repro ---` section of the failure output
(see [Reading Failure Output](failures.md)); the end-to-end
debugging story is the
[Investigate a Crash](../recipes/investigate-crash.md) recipe.

## Example output

The probe dump shows each function with decoded fields and source
locations (DWARF for kernel functions, BPF line info for callbacks).
Where fexit captured post-mutation state, changed fields show an
arrow between entry and exit values:

<!-- captured: cargo ktstr test --kernel local-8cd2b47 (v7.1 + sched_ext_exit tracepoint) --features integration,wprof -E 'test(=ktstr/bpf_crash_auto_repro_e2e)' --no-capture; ktstr 0.23.0 + pending trigger-signature fix and probe-ship hold (scratchpad patches) | full run: captures/autorepro-live.txt -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test — auto-repro trail after a scheduler crash</span></div>

<pre>--- probe pipeline ---
  extracted:   9 functions from crash backtrace
  kprobes:     5 attached
  fentry:      11 attached
  <span class="t-grn">trigger:     fired (tp_btf)</span>
  events:      125 captured, 8 after stitch

=== AUTO-PROBE: scx_exit fired ===

  ktstr_select_cpu                                              main.bpf.c:380
    task_struct *p
      pid                 40
      cpus_ptr            0xf(0-3)
      <span class="t-grn">dsq_id              SCX_DSQ_INVALID  →  SCX_DSQ_LOCAL</span>
      enq_flags           NONE
      <span class="t-grn">slice               19982063         →  20000000</span>
      vtime               0
      weight              100
      scx_flags           RESET_RUNNABLE_AT|DEQD_FOR_SLEEP|ENABLED
  do_enqueue_task                                               kernel/sched/ext.c:1885
    rq *rq
      cpu                 0
    task_struct *p
      pid                 40
      dsq_id              SCX_DSQ_LOCAL
      slice               20000000
      scx_flags           QUEUED|DEQD_FOR_SLEEP|ENABLED
<span class="t-dim">  ...</span>
  scx_bpf_error_bstr+0x8a/0xd0
  <span class="t-red">bpf_prog_9a11f2edaac0b52f_ktstr_dispatch+0x57/0x1db</span>
  do_pick_task_scx+0x50e/0x940
  __schedule+0x5a3/0x10c0</pre></div>

Reading it as a progression: pid 40 entered `select_cpu` off any
dispatch queue with a nearly expired slice; by exit it sat on the
local DSQ with a refilled 20 ms slice (`SCX_DSQ_INVALID →
SCX_DSQ_LOCAL`, `19982063 → 20000000`); the kernel's
`do_enqueue_task` then queued it. Healthy scheduling decisions,
captured right up to the frame where `ktstr_dispatch` called
`scx_bpf_error()` — the chain shows what the scheduler did with its
last tasks on the way into the error, not just where it died.

After the probe data, the section appends the repro VM's wall time
and, when non-empty, the last lines of its scheduler log, sched_ext
dump, failure-dump JSON, and dmesg.

## Enabling it — and what it costs

Auto-repro is on by default for every `#[ktstr_test]` with a
scheduler. Opt out per test:

```rust,ignore
#[ktstr_test(scheduler = MY_SCHED, auto_repro = false)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> { ... }
```

It fires only when the primary run fails, and it is disabled
automatically when `expect_err = true` (no point probing a
deliberately failing test). The cost is a second VM boot plus a full
scenario rerun — in the captured demo below, the repro VM added
about 17 seconds.

## How it works

1. **Stack extraction** — function names are parsed from the crash
   trace in the scheduler log or kernel console. BPF program symbols
   (`bpf_prog_*`) are recognized and their short names extracted;
   generic frames (spinlocks, syscall entry, sched_ext exit
   machinery, trampolines) are filtered out.
2. **BPF discovery** — in the repro VM, loaded struct_ops programs
   are discovered and added to the probe list along with their
   kernel-side callers (e.g. `enqueue` → `do_enqueue_task`), so the
   pipeline still probes something when the crash produced no
   extractable stack.
3. **BTF resolution** — signatures come from vmlinux BTF and program
   BTF; known structs (`task_struct`, `rq`, dispatch queues) have
   curated fields resolved to offsets, and other struct pointers get
   scalar/enum/cpumask fields auto-discovered.
4. **Probed rerun** — the second VM reruns the scenario with kprobes
   on kernel entry, fentry/fexit on BPF callbacks and kernel exits,
   and a one-shot trigger on the `sched_ext_exit` tracepoint that
   fires at the moment the exit is claimed.
5. **Stitching** — events are filtered to the task that triggered the
   exit, sorted by timestamp, and rendered with decoded values.

If the primary VM failed before the scheduler ever attached and the
workload ever ran, the repro has nothing to reproduce — the framework
prepends a `PRIMARY DID NOT REACH WORKLOAD` label to the repro
verdict so you chase the primary's startup failure (see its
`--- diagnostics ---` and `--- timeline ---` sections) instead of
reading the repro as evidence.

## Kernel requirement

The probe *trigger* needs the `sched_ext_exit` tracepoint, which is
currently only in the sched_ext `for-7.2` development branch — no
released stable kernel has it. On a kernel without it, the rest of
the pipeline still runs — the crash call chain is extracted and
probes are prepared — but the trigger cannot attach and no events are
captured. The `--- probe pipeline ---` block says exactly that; this
is the shape to recognize:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration,wprof -E 'test(=ktstr/bpf_crash_auto_repro_e2e)' --no-capture (expect_auto_repro demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- auto-repro ---
--- probe pipeline ---
  extracted:   10 functions from crash backtrace
  traceable:   7 passed, 3 dropped: bpf_prog_1fed99378f3a8055_ktstr_dispatch, bpf__sched_ext_ops_dispatch, ret_from_fork_asm
  bpf_discover: 0 programs found
  after_expand: 7 total probe targets
  kprobes:     0 attached
  trigger:     attach failed (skeleton load (retry): No such process (os error 3); original error before retry: No such process (os error 3))
  probe_data:  0 keys, 0 unmatched IPs
  events:      0 captured, 0 after stitch

repro VM duration: 16.9s
```

The diagnostic tails (repro VM sched_ext dump, dmesg) are still
appended, so the repro run remains useful as a crash-reproduction
check even without probe events.

## Example test

`bpf_crash_auto_repro_e2e` in ktstr's `tests/scenario_coverage.rs`
drives the path end to end: a host-side BPF map write sets the
fixture scheduler's `crash` global, the scheduler calls
`scx_bpf_error`, and the auto-repro VM replays it.
