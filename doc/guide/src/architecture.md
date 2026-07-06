# Architecture Overview

A scheduler bug rarely returns an error code — it wedges a CPU, strands
a runqueue, or panics the kernel. ktstr's architecture follows from
that: every test boots its own KVM microVM, so a crash takes down a
disposable guest instead of your machine; the CPU topology is whatever
the test declared; and the kernel is the exact build you targeted.

ktstr has three execution domains:

<div class="kt-steps">
<div class="kt-step" data-step="1"><strong>Host process</strong><p>The test binary on the host manages VM lifecycle, monitors guest memory, and evaluates results.</p></div>
<div class="kt-step" data-step="2"><strong>Guest process</strong><p>The same binary runs inside the VM as PID 1: mounts filesystems, starts the scheduler, creates cgroups, forks workers, and writes results back.</p></div>
<div class="kt-step" data-step="3"><strong><a href="architecture/monitor.html">Monitor thread</a></strong><p>A host thread reads guest VM memory directly, observing scheduler state without instrumenting the guest.</p></div>
</div>

## Execution flow

<div class="kt-sequence">
<div class="kt-seq-row"><span class="kt-seq-num">01</span><div class="kt-seq-card host"><strong data-lane="host">Package the guest</strong><p>The host builds an initramfs that contains the same test binary as <code>/init</code>, plus the scheduler binary, payloads, and files the test declared.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">02</span><div class="kt-seq-card host"><strong data-lane="host">Boot the target kernel</strong><p>KVM starts a disposable VM using the requested kernel, topology, memory, transports, and performance-mode settings.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">03</span><div class="kt-seq-card guest"><strong data-lane="guest">Re-enter as PID 1</strong><p>Inside the VM, the test binary takes the guest path: mount filesystems, read the host-provided test spec, and prepare cgroup state.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">04</span><div class="kt-seq-card monitor"><strong data-lane="monitor">Start observation</strong><p>A host thread begins reading guest memory through KVM mappings, resolving BTF offsets and watching scheduler state without injecting code into the guest.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">05</span><div class="kt-seq-card guest"><strong data-lane="guest">Run the scenario</strong><p>The guest starts the scheduler, creates cgroups and cpusets, forks workers, applies ops, and records worker reports.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">06</span><div class="kt-seq-card monitor"><strong data-lane="monitor">Capture violations and state</strong><p>The monitor records stuck observations, snapshots, event counters, dumps, and scheduler-exit context while the workload runs.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">07</span><div class="kt-seq-card guest"><strong data-lane="guest">Return the scenario verdict</strong><p>The guest evaluates worker checks and writes the structured result to virtio-console port 1.</p></div></div>
<div class="kt-seq-row"><span class="kt-seq-num">08</span><div class="kt-seq-card host"><strong data-lane="host">Merge and report</strong><p>The host combines guest results with monitor findings, writes artifacts, and returns the nextest case outcome.</p></div></div>
</div>

Results travel on virtio-console port 1; panics, crashes, and other
non-blockable diagnostics fall back to the COM2 serial port (see
[VMM — guest–host transports](architecture/vmm.md#transports)).


## Key design decisions

**Same binary, two roles.** The test binary serves as both host
controller and guest test runner. The initramfs embeds the binary as
`/init`; when the binary finds itself running as PID 1, it executes
the guest lifecycle (mounts, scheduler start, test dispatch, reboot)
instead of the host one. One `cargo build` produces everything needed
for both sides — there is no separate guest agent to version or ship.

**Forked workers (default), threads optional.** The default `Fork`
clone mode spawns each worker as its own process so cgroup placement
via `cgroup.procs` is tgid-granular. The `Thread` clone mode shares
the harness's tgid and routes placement through `cgroup.threads`
instead — useful when workers need a shared address space or when
measuring thread-only scheduler paths. See
[Workers and Workloads](architecture/workers.md).

**Host-side monitoring.** The monitor reads guest memory via KVM,
avoiding BPF instrumentation of the scheduler under test. This
eliminates observer effects on scheduling decisions.

## Where to go next

- [VMM](architecture/vmm.md) — how VMs boot, topology modeling,
  guest–host transports.
- [Monitor](architecture/monitor.md) — what is observed from the host
  and how violations become verdicts.
- [Workers and Workloads](architecture/workers.md) — worker lifecycle
  and the telemetry each worker reports.
- [CgroupManager and CgroupGroup](architecture/cgroup-manager.md) —
  cgroup plumbing and RAII cleanup inside the guest.
