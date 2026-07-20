<div class="kt-hero">
<h1>ktstr</h1>
<p class="kt-tagline">Test Linux schedulers like code. Declare the topology, boot a real kernel in KVM, run real workloads, and let host-side checks explain what happened.</p>
<div class="kt-hero-meta">
<div><strong>Real kernels</strong><span>released, local, or git-built</span></div>
<div><strong>Topology as code</strong><span>NUMA, LLCs, cores, SMT</span></div>
<div><strong>Actionable failures</strong><span>stats, dumps, auto-repro</span></div>
</div>
<div class="kt-cta-row"><a class="kt-btn kt-btn-primary" href="getting-started.html">Get started</a><a class="kt-btn kt-btn-ghost" href="features.html">See it in action</a></div>
</div>

Scheduler bugs hide in topology: the fairness regression that only shows
up on an odd LLC count, the no-progress case that needs SMT siblings, the crash
that wants a NUMA crossing. ktstr turns those cases into ordinary Rust
tests for [sched_ext](https://github.com/sched-ext/scx) schedulers:
the test declares the machine shape, the VM boots with that shape, and
the host watches without instrumenting the guest.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong><a href="getting-started.html">New to ktstr?</a></strong><p>Install the CLI, build a guest kernel, and run your first green test.</p></div>
<div class="kt-doc-card"><strong><a href="recipes/test-new-scheduler.html">Already have a scheduler?</a></strong><p>Register an <code>scx_*</code> binary and put it under test quickly.</p></div>
<div class="kt-doc-card"><strong><a href="running-tests/failures.html">Debugging a failure?</a></strong><p>Read the verdict, timeline, dumps, and auto-repro output in order.</p></div>
</div>

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

Run it against any target kernel. A path points at a local kernel tree;
a version such as `7.0` points at ktstr's kernel cache:

```sh
cargo ktstr test --kernel ../linux --no-perf-mode
```

<!-- captured: cargo ktstr test --kernel ../linux --no-perf-mode -- --features integration -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)' | ktstr 0.24.0-dev | kernel ../linux (b4dc42d2, sched_ext-for-7.2) | verified by independent rerun -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel ../linux --no-perf-mode</span></div>

<pre><span class="t-dim">cargo ktstr: resolved kernel "path_linux_b5562c"
cargo ktstr: BTF type anchor at target/ktstr_btf_anchor.h</span>
...
 Nextest run ID d79da4e4-… with nextest profile: default
    Starting 1 test across 120 binaries (13597 tests skipped)
        <span class="t-grn">PASS [   8.601s] (1/1) ktstr::scx_cleanup_test ktstr/scx_empty_run_exits_under_watchdog</span>
...
     <span class="t-b">Summary [   8.642s] 1 test run: 1 passed, 13597 skipped</span></pre></div>

Without a `scheduler` attribute, tests run under the kernel's default
scheduler (EEVDF) — useful for baselines and A/B comparisons.

<div class="kt-section-kicker">Mental model</div>

<div class="kt-steps">
<div class="kt-step" data-step="1"><strong>Describe the machine</strong><p><code>llcs</code>, <code>cores</code>, <code>threads</code>, NUMA, and gauntlet constraints become the VM's CPU topology.</p></div>
<div class="kt-step" data-step="2"><strong>Run real work</strong><p>Scenarios create cgroups, cpusets, workers, benchmarks, and mid-run operations inside the guest.</p></div>
<div class="kt-step" data-step="3"><strong>Check behavior</strong><p>Opt-in assertions judge worker progress, spread, stuck gaps, throughput, temporal patterns, and regressions.</p></div>
</div>

## When it breaks, you see why

A crash log tells you where the scheduler died. ktstr also tells you what
the state looked like on the way there: on a crash it boots a second VM,
attaches BPF probes along the crash path, and reruns the scenario. Each
probed function prints decoded struct fields; `→` marks fields that
changed between entry and exit:

<!-- captured: cargo ktstr test --kernel local-8cd2b47 (v7.1 + scheduler-exit probe trigger) -E 'test(=ktstr/bpf_crash_auto_repro_e2e)' --no-capture | ktstr 0.23.0 (with the probe trigger fix) | full run: captures/autorepro-live.txt -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test — auto-repro output after a scheduler crash</span></div>

<pre><span class="t-b">=== AUTO-PROBE: scx_exit fired ===</span>

  ktstr_select_cpu                                              main.bpf.c:380
    task_struct *p
      pid                 40
      cpus_ptr            0xf(0-3)
<span class="t-grn">      dsq_id              SCX_DSQ_INVALID  →  SCX_DSQ_LOCAL</span>
<span class="t-grn">      slice               19982063         →  20000000</span>
      weight              100
      scx_flags           RESET_RUNNABLE_AT|DEQD_FOR_SLEEP|ENABLED
  enqueue_task_scx                                              kernel/sched/ext.c
    rq *rq
      cpu                 0
    task_struct *p
      pid                 40
      dsq_id              SCX_DSQ_LOCAL
      scx_flags           QUEUED|DEQD_FOR_SLEEP|ENABLED
<span class="t-dim">  ...</span>
<span class="t-red">  bpf_prog_9a11f2edaac0b52f_ktstr_dispatch+0x57/0x1db</span></pre></div>

Auto-repro is on by default and selects the kernel's native scheduler-exit
hook (`sched_ext_exit` on the newest kernels, a raw `scx_vexit` entry/return
pair on the preceding generation, or `scx_dump_state` on global-era kernels) —
see [Auto-Repro](running-tests/auto-repro.md). For the
anatomy of ordinary failures (stats, timeline, monitor verdict), see
[Reading Failure Output](running-tests/failures.md).

<div class="kt-cards">
<div class="kt-card"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/><path d="M8 1v2M16 1v2M8 21v2M16 21v2M1 8h2M1 16h2M21 8h2M21 16h2"/></svg><h3>Real kernels under KVM</h3><p>Each test gets a fresh micro-VM booting the exact kernel you target. Real cgroups, real BPF, no shared state.</p><a href="architecture.html">How it works →</a></div>
<div class="kt-card"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="4" width="9" height="7" rx="1.5"/><rect x="13" y="4" width="9" height="7" rx="1.5"/><rect x="2" y="14" width="4" height="4" rx="1"/><rect x="7.5" y="14" width="4" height="4" rx="1"/><rect x="13" y="14" width="4" height="4" rx="1"/><rect x="18.5" y="14" width="3.5" height="4" rx="1"/></svg><h3>Topology as code</h3><p>NUMA nodes, LLCs, cores, SMT — declared on the test attribute, realized in the guest down to the ACPI tables.</p><a href="concepts/topology.html">Topology →</a></div>
<div class="kt-card"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16M4 9h16M4 14h16M4 19h16" opacity=".35"/><circle class="fill" cx="7" cy="9" r="1.6"/><circle class="fill" cx="12" cy="14" r="1.6"/><circle class="fill" cx="17" cy="4" r="1.6"/><circle class="fill" cx="9" cy="19" r="1.6"/></svg><h3>Gauntlet</h3><p>One test declaration fans out across a matrix of topology presets — odd LLC counts, SMT, NUMA crossings — with budget-aware selection for CI.</p><a href="running-tests/gauntlet.html">Gauntlet →</a></div>
<div class="kt-card"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.6"/><path d="M12 2v3.6M12 18.4V22M2 12h3.6M18.4 12H22"/></svg><h3>Auto-repro</h3><p>Crashes rerun themselves in a probe VM that captures function arguments and struct state along the crash path.</p><a href="running-tests/auto-repro.html">Auto-Repro →</a></div>
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

- **Fair scheduling** — workers get CPU time without zero-work outliers
  or excessive scheduling gaps.
- **Cpuset isolation** — workers stay on assigned CPUs.
- **Dynamic operations** — cgroups created, destroyed, and resized
  mid-run.
- **Affinity** — the scheduler respects thread affinity constraints.
- **Stress** — many cgroups, many workers, rapid topology changes.
- **Stuck-task detection** — the scheduler doesn't leave runnable
  tasks unrun.

> [!NOTE]
> ktstr is pre-release. 0.x APIs change between releases, so pin the
> exact version — [Getting Started](getting-started.md) shows how.

## Next steps

- [ktstr in Action](features.md) — the full feature tour, with real output.
- [Getting Started](getting-started.md) — install, build a kernel, first green test.
- [Tutorial: Zero to ktstr](tutorial.md) — build up a real test suite step by step.
- [Test a New Scheduler](recipes/test-new-scheduler.md) — already have a scheduler? Start here.
