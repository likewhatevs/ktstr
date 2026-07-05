<div class="kt-hero">
<h1>ktstr</h1>
<p class="kt-tagline">Test Linux schedulers like code. Every test boots a real kernel in a KVM micro-VM with the topology it declares — and ktstr watches what your scheduler does from the host, without touching the guest.</p>
<div class="kt-cta-row"><a class="kt-btn kt-btn-primary" href="getting-started.html">Get started</a><a class="kt-btn kt-btn-ghost" href="features.html">See it in action</a></div>
</div>

Scheduler bugs hide in topology: the fairness regression that only shows
up on an odd LLC count, the starvation that needs SMT siblings, the crash
that wants a NUMA crossing. Testing a
[sched_ext](https://github.com/sched-ext/scx) scheduler against those
shapes has meant scrounging hardware and hand-running repro scripts.
ktstr turns it into `cargo test`: declare the topology on the test, and
the VM actually has it.

## Quick taste

```rust,ignore
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_mine",
});

#[ktstr_test(scheduler = MY_SCHED, llcs = 1, cores = 2, threads = 1)]
fn steady_under_my_sched(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

Run it against any kernel — a released version, a local source tree, or
a git URL:

```sh
cargo ktstr test --kernel 7.0
```

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/failure_dump_renders_bss_fields)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
cargo ktstr: fetching latest 7.0.x kernel version
cargo ktstr: latest 7.0.x kernel: 7.0.14
cargo ktstr: resolved kernel "7.0"
...
 Nextest run ID 98581174-… with nextest profile: default
    Starting 1 test across 121 binaries (12531 tests skipped)
        PASS [  34.459s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields
...
     Summary [  34.498s] 1 test run: 1 passed, 12531 skipped
```

Without a `scheduler` attribute, tests run under the kernel's default
scheduler (EEVDF) — useful for baselines and A/B comparisons.

## When it breaks, you see why

A crash log tells you where the scheduler died. ktstr also tells you what
the state looked like on the way there: on a crash it boots a second VM,
attaches BPF probes along the crash path, and reruns the scenario. Each
probed function prints decoded struct fields; `→` marks fields that
changed between entry and exit:

<!-- captured: cargo ktstr test (ktstr/bpf_crash_auto_repro_e2e) — prior-run sample preserved from running-tests/auto-repro.md | ktstr 0.23.0 | kernel with the sched_ext_exit tracepoint -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test — auto-repro output after a scheduler crash</span></div>

<pre><span class="t-b">=== AUTO-PROBE: scx_exit fired ===</span>

  ktstr_enqueue                                                   main.bpf.c:21
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
      dsq_id      SCX_DSQ_INVALID
<span class="t-dim">      ...</span>
      scx_flags   QUEUED|ENABLED
  do_enqueue_task                                               kernel/sched/ext.c
    rq *rq
      cpu         1
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
<span class="t-grn">      dsq_id      SCX_DSQ_INVALID          →  SCX_DSQ_LOCAL</span>
<span class="t-dim">      ...</span>
<span class="t-grn">      scx_flags   QUEUED|DEQD_FOR_SLEEP    →  QUEUED</span></pre></div>

Auto-repro is on by default and needs a kernel with the `sched_ext_exit`
tracepoint — see [Auto-Repro](running-tests/auto-repro.md). For the
anatomy of ordinary failures (stats, timeline, monitor verdict), see
[Reading Failure Output](running-tests/failures.md).

<div class="kt-cards">
<div class="kt-card"><h3>Real kernels under KVM</h3><p>Each test gets a fresh micro-VM booting the exact kernel you target. Real cgroups, real BPF, no shared state.</p><a href="architecture.html">How it works →</a></div>
<div class="kt-card"><h3>Topology as code</h3><p>NUMA nodes, LLCs, cores, SMT — declared on the test attribute, realized in the guest down to the ACPI tables.</p><a href="concepts/topology.html">Topology →</a></div>
<div class="kt-card"><h3>Gauntlet</h3><p>One test declaration fans out across a matrix of topology presets — odd LLC counts, SMT, NUMA crossings — with budget-aware selection for CI.</p><a href="running-tests/gauntlet.html">Gauntlet →</a></div>
<div class="kt-card"><h3>Auto-repro</h3><p>Crashes rerun themselves in a probe VM that captures function arguments and struct state along the crash path.</p><a href="running-tests/auto-repro.html">Auto-Repro →</a></div>
</div>

## Design

**Fidelity without overhead.** Every test boots a real Linux kernel in a
KVM VM with real cgroups and real BPF programs — no mocking, no
containers, no state carried between tests. The VMM is purpose-built for
this job; see [VMM](architecture/vmm.md).

**Direct access over tooling layers.** The host-side monitor reads guest
memory through BTF-resolved struct offsets — runqueues, DSQ depths,
schedstat counters — loading nothing into the guest, so observation does
not perturb the scheduler under test. See
[Monitor](architecture/monitor.md).

## What it tests

- **Fair scheduling** — workers get CPU time without starvation or
  excessive scheduling gaps.
- **Cpuset isolation** — workers stay on assigned CPUs.
- **Dynamic operations** — cgroups created, destroyed, and resized
  mid-run.
- **Affinity** — the scheduler respects thread affinity constraints.
- **Stress** — many cgroups, many workers, rapid topology changes.
- **Stall detection** — the scheduler doesn't drop tasks.

> [!NOTE]
> ktstr is pre-release. 0.x APIs change between releases, so pin the
> exact version — [Getting Started](getting-started.md) shows how.

## Next steps

- [ktstr in Action](features.md) — the full feature tour, with real output.
- [Getting Started](getting-started.md) — install, build a kernel, first green test.
- [Tutorial: Zero to ktstr](tutorial.md) — build up a real test suite step by step.
- [Test a New Scheduler](recipes/test-new-scheduler.md) — already have a scheduler? Start here.
