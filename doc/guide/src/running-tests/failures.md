# Reading Failure Output

When a test fails, everything ktstr knows lands in the test's stderr
as one bundle: the violated thresholds, the workload statistics, a
phase timeline, the scheduler's own log, the monitor's summary, and —
for scheduler crashes — the kernel's sched_ext dump and an
[auto-repro](auto-repro.md) trail. This page walks the sections in
the order they appear, using two real failures.

## A check-gate failure

This test set an iteration-rate floor its (deliberately slowed)
scheduler could not meet. The first line names the test, scheduler,
and topology; the indented lines under it are the violated checks:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' (docs demo test vs a deliberately slowed scheduler) | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)'</span></div>

<pre>  TRY 1 FAIL [  31.810s] (───) ktstr::docs_demo ktstr/throughput_gate
  stderr ───
...
    <span class="t-red">ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:</span>
      <span class="t-red">worker 71 iteration rate 41903.3/s below floor 50000000.0/s</span>
      worker 73 iteration rate 37834.5/s below floor 50000000.0/s

    --- stats ---
    2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
      cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
      cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252

    --- timeline ---
    topology: 1n1l2c1t (2 cpus)  scheduler: my_sched  scenario: throughput_gate  duration: 15.0s

    Phase 1: StepStart[0] ops=0 (4960ms, 0 samples):
      imbalance: avg=1.2 max=5.0 | dsq: avg=0 max=0 | nr_run: avg=1.0 | fallback: 0/s | keep_last: 38/s | throughput: 79697 iter/s (stimulus-derived)
      per-cgroup:
        cg_a: off-cpu avg=0.3% min=0.3% max=0.3% spread=0.0% | run-delay mean=915µs worst=915µs | iters=209600 migrations=1 | gap=10ms@cpu0
        <span class="t-yel">cg_b: off-cpu avg=9.0% min=9.0% max=9.0% spread=0.0% | run-delay mean=5654µs worst=5654µs</span> | iters=189252 migrations=1 | gap=21ms@cpu0
      &gt;&gt;&gt; StepStart[0]: ops=0 (2 cgroups, 2 workers)</pre></div>

Reading it:

- **Header + details** — the checks that failed, one line each, with
  the observed value and the threshold in the same line. The detail
  line is the verdict; everything below is context.
- **`--- stats ---`** — the per-run roll-up: worker count, distinct
  CPUs touched, migrations, worst per-cgroup spread and off-CPU gap.
  `cg{i}` is the positional index of the cgroup in this roll-up, not
  its name — it lines up with the per-cgroup rows underneath.
- **`--- timeline ---`** — one block per scenario phase with monitor
  averages and per-cgroup detail (named cgroups here). In this run,
  `cg_b`'s 9% off-CPU time and 5.6 ms mean run-delay against `cg_a`'s
  0.3%/0.9 ms is the asymmetry the failed rate gate traces back to.

Two more sections follow every failure when they have content:

<!-- captured: same run as above (throughput_gate, TRY 1) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
    --- scheduler log ---
    libbpf: struct_ops ktstr_ops: member sub_attach not found in kernel, skipping it as it's set to zero
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

The scheduler log is whatever the scheduler binary printed (libbpf
noise included). The monitor block is the host-side observer's
summary — see [Monitor](../architecture/monitor.md) for what each
line means. `verdict: monitor OK` here says the *monitor's* checks
passed; the test still failed on the worker-side rate gate. The two
channels are independent.

## A scheduler crash

When the scheduler itself dies, the trail grows: a `BUG SUMMARY`
line, a `--- diagnostics ---` section for the run stage, the kernel's
sched_ext debug dump, and an auto-repro section. From a real
`scx_bpf_error` crash (triggered on purpose by a demo test):

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration,wprof -E 'test(=ktstr/bpf_crash_auto_repro_e2e)' --no-capture (expect_auto_repro demo test; failure-trail excerpt) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
BUG SUMMARY: scx_bpf_error (src/bpf/main.bpf.c:424: ktstr: host-triggered crash)
ktstr_test 'bpf_crash_auto_repro_e2e' [sched=scx-ktstr] [topo=1n1l4c1t] failed:
  scheduler process died unexpectedly during workload (2.2s into test)

--- stats ---
4 workers, 0 cpus, 0 migrations, worst_spread=0.0%, worst_gap=0ms
  cg0: workers=4 cpus=0 spread=n/a gap=0ms migrations=0 iter=0

--- diagnostics ---
stage: payload started but produced no test result
exit_code=1
```

`BUG SUMMARY` is the one-line cause, extracted from the kernel's
`triggered exit kind` emission or the scheduler log. The
`--- diagnostics ---` stage line tells you how far the run got before
dying — here the workload started but never reported, because the
scheduler died under it.

The scheduler-log section carries the kernel's full debug dump for
scheduler exits — exit kind, backtrace, and (if the scheduler
implements `ops.dump`) its own state:

```text
--- scheduler log ---
...
DEBUG DUMP
================================================================================

swapper/3[0] triggered exit kind 1025:
  scx_bpf_error (src/bpf/main.bpf.c:424: ktstr: host-triggered crash)

Backtrace:
  scx_exit+0x50/0x70
  scx_bpf_error_bstr+0x78/0x90
  bpf_prog_1fed99378f3a8055_ktstr_dispatch+0x4d/0x1cb
  bpf__sched_ext_ops_dispatch+0x4b/0xa7
  do_pick_task_scx+0x379/0x770
  __schedule+0x5ca/0xfc0
...
ktstr scheduler state:
  stall=0 crash=1 degrade_rt=0
```

A `--- sched_ext dump ---` section repeats the same dump as captured
from the kernel trace channel, and `--- auto-repro ---` reports the
second VM's replay of the crash:

<!-- captured: same bpf_crash_auto_repro_e2e run | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- auto-repro ---
--- probe pipeline ---
  extracted:   10 functions from crash backtrace
  traceable:   7 passed, 3 dropped: bpf_prog_1fed99378f3a8055_ktstr_dispatch, bpf__sched_ext_ops_dispatch, ret_from_fork_asm
...
repro VM duration: 16.9s
```

See [Auto-Repro](auto-repro.md) for how to read the probe pipeline
and its output.

## Detail-line catalog

The worker-side checks emit a fixed set of detail-line shapes (each
format string is pinned by a unit test, so these stay accurate):

- `worker {N} iteration rate {R}/s below floor {F}/s` — a benchmark
  rate gate failed.
- `tid {N} made no progress (0 work units)` — a worker made no
  measured progress (`not_stuck`).
- `tid {N} stuck {X}ms on cpu{C} at +{T}ms (threshold {N}ms)` — a
  worker's longest off-CPU gap crossed `max_gap_ms`.
- `unfair cgroup: spread={P}% ({lo}-{hi}%) {N} workers on {N} cpus (threshold {P}%)`
  — per-cgroup fairness exceeded `max_spread_pct`.

See [Checking](../concepts/checking.md) for the model behind these
and the monitor-side violations.

## Artifacts on disk

<div class="kt-figure"><svg width="700" height="120" viewBox="0 0 700 120" role="img" aria-label="Failure dump pipeline: a scheduler error trips the freeze coordinator, vCPUs freeze, the monitor walks BTF-typed state, and the dump is written as JSON and rendered sections">
  <defs><marker id="kt-arr4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <g font-size="10.5" fill="var(--fg)">
    <rect x="8" y="26" width="130" height="56" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="22" y="49" font-weight="700" opacity=".85">scx exit fires</text>
    <text x="22" y="67" opacity=".7">error / watchdog</text>
    <path d="M140 54 L 172 54" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr4)"/>
    <rect x="176" y="26" width="130" height="56" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.2"/>
    <text x="190" y="49" font-weight="700" fill="var(--kt-accent)">vCPUs freeze</text>
    <text x="190" y="67" opacity=".7">state can't decay</text>
    <path d="M308 54 L 340 54" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr4)"/>
    <rect x="344" y="26" width="160" height="56" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.2"/>
    <text x="358" y="49" font-weight="700" fill="var(--kt-accent)">monitor walks state</text>
    <text x="358" y="67" opacity=".7">BTF-typed maps · regs · rq</text>
    <path d="M506 54 L 538 54" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr4)"/>
    <rect x="542" y="26" width="150" height="56" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="556" y="49" font-weight="700" opacity=".85">failure dump</text>
    <text x="556" y="67" opacity=".7" font-family="var(--mono-font)">.json + rendered</text>
  </g>
</svg></div>

After the run, `cargo ktstr` prints where everything landed:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' (tail of the same failing run) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
cargo ktstr: test outputs
  ~/ktstr/target/ktstr/7.0.14-73730e0-dirty
    FAILED  throughput_gate  [my_sched 1n1l2c1t]
      failure dump  ~/ktstr/target/ktstr/7.0.14-73730e0-dirty/throughput_gate-2ecd2624f3df7276.failure-dump.json
      stats         ~/ktstr/target/ktstr/7.0.14-73730e0-dirty/throughput_gate-2ecd2624f3df7276.ktstr.json
      replay        cargo ktstr replay --filter throughput_gate --exec
```

Every failed test writes a
`{test_name}-{variant_hash:016x}.failure-dump.json` next to its
result sidecar in the run directory (see [Runs](runs.md) for the
directory semantics). Auto-repro runs write a sibling
`.repro.failure-dump.json` with the repro VM's own snapshot. The
path is pre-cleared at each dispatch, so a passing rerun never
leaves a stale dump behind.

The dump comes in two shapes. When the scheduler attached and its
exit path triggered, it is a full post-mortem: BPF map contents with
BTF-typed field names, per-vCPU registers, per-program runtime stats
— the JSON form of what [Snapshots](../writing-tests/snapshots.md)
renders. Every BPF map is walked with BTF, so values come back as
named struct members, not hex:

<!-- captured: target/ktstr/7.1.0/failure_dump_renders_bss_fields-*.failure-dump.json (trimmed) | ktstr 0.23.0 | kernel v7.1-patched -->
```json
{
  "schema": "single",
  "maps": [
    { "name": "arena", "map_type": 33, "arena": { "pages": 13, "..." : "..." } },
    { "name": "bpf_bpf.bss", "map_type": 2, "value_size": 280,
      "value": { "kind": "struct", "type_name": ".bss", "members": [
        { "name": "chunk_pool", "value": { "kind": "struct", "type_name": "sdt_pool",
          "members": [
            { "name": "elem_size", "value": { "kind": "uint", "bits": 64, "value": 4096 } },
            "..."
        ] } },
        "..."
      ] } },
    "..."
  ]
}
```

When the failure happened before the capture path could adopt, a
placeholder is written instead, and says so:

<!-- captured: target/ktstr/7.0.14-73730e0-dirty/failure_dump_renders_bss_fields-69298e1249f472ef.failure-dump.json | ktstr 0.23.0 | kernel 7.0.14 -->
```json
{
    "schema": "single",
    "maps": [],
    "sdt_alloc_unavailable": "test failed at stage `payload started but produced no test result`; no BPF state captured (probe did not attach before failure)",
    ...
    "is_placeholder": true
}
```

The JSON is for tooling that walks the run directory; humans should
read the stderr — the actionable diagnostics are the `BUG SUMMARY`
line and the `--- sched_ext dump ---` section.

## Investigation workflow

1. Read the header and detail lines — they name the check and the
   margin by which it failed.
2. For check failures, correlate against `--- stats ---` and
   `--- timeline ---`: which cgroup, which phase, migrations or gaps?
3. For crashes, start from `BUG SUMMARY` and the backtrace in the
   debug dump, then read the [auto-repro](auto-repro.md) trail for
   the state on the way to the exit.
4. Re-run exactly the failing variant — including the gauntlet preset
   segment if it was a gauntlet case (see
   [test name shapes](../running-tests.md#test-name-shapes)):

   ```sh
   cargo ktstr test --kernel 7.0 -- -E 'test(=gauntlet/my_test/smt-2llc)'
   ```

   or re-run everything that failed last session with
   [`cargo ktstr replay`](cargo-ktstr.md#replay).
5. Poke at the same environment interactively:
   `cargo ktstr shell --test my_test` boots a VM with the test's
   topology, memory, and include files (see
   [ktstr shell](ktstr.md#shell)).

## Verbosity knobs

- `RUST_BACKTRACE=1` — Rust panic backtraces, plus ktstr's verbose
  mode: the full guest kernel console is appended to
  `--- diagnostics ---` on failure, the auto-repro VM's console is
  forwarded live, and the guest boots with `loglevel=7`.
- `RUST_LOG=ktstr=debug` — host-side tracing (probe attach reasons,
  libbpf errors).
- `--dmesg` — streams the guest kernel console in real time; it is a
  flag on `cargo ktstr shell` / `ktstr shell`, not on `test`.

Environment errors (kernel not found, cgroup controllers missing,
flock timeouts) are cataloged in
[Troubleshooting](../troubleshooting.md).
