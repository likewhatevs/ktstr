# Architecture Overview

A scheduler bug rarely returns an error code — it wedges a CPU, strands
a runqueue, or panics the kernel. ktstr's architecture follows from
that: every test boots its own KVM microVM, so a crash takes down a
disposable guest instead of your machine; the CPU topology is whatever
the test declared; and the kernel is the exact build you targeted.

ktstr has three execution domains:

1. **Host process** — the test binary running on the host. Manages
   [VM lifecycle](architecture/vmm.md), monitors guest memory,
   evaluates results.

2. **Guest process** — the same test binary running inside the VM
   as PID 1. Mounts filesystems, starts the scheduler, creates
   cgroups, forks [workers](architecture/workers.md), runs scenarios,
   and writes results back to the host.

3. **[Monitor](architecture/monitor.md) thread** — runs on the host
   while the guest executes. Reads guest VM memory directly to observe
   scheduler state without instrumenting it.

## Execution flow

```text
Host                          Guest
----                          -----
test binary                   
  |                           
  +-- build initramfs         
  |   (test binary as /init   
  |    + optional scheduler)  
  |                           
  +-- boot KVM VM             
  |                           test binary (PID 1 init)
  |                             |
  +-- start monitor thread      +-- mount filesystems
  |   (reads guest memory)      +-- start scheduler (if any)
  |                             +-- create cgroups
  |                             +-- fork workers
  |                             +-- move workers to cgroups
  |                             +-- signal workers to start
  |                             +-- poll scheduler liveness
  |                             +-- stop workers, collect reports
  |                             +-- evaluate results
  |                             +-- write result to virtio-console port 1
  |                           
  +-- read result from virtio-console port 1
  +-- evaluate monitor data   
  +-- report pass/fail        
```

Results travel on virtio-console port 1; panics, crashes, and other
non-blockable diagnostics fall back to the COM2 serial port (see
[VMM — guest–host transports](architecture/vmm.md#transports)).

From the host, a passing run looks like this:

<!-- captured: cargo ktstr test --kernel 7.0 -E 'test(failure_dump_renders_bss_fields)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
cargo ktstr: fetching latest 7.0.x kernel version
cargo ktstr: latest 7.0.x kernel: 7.0.14
cargo ktstr: resolved kernel "7.0"
...
    Starting 1 test across 121 binaries (12531 tests skipped)
        PASS [  34.451s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields
...
     Summary [  34.490s] 1 test run: 1 passed, 12531 skipped
```

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
- [CgroupManager](architecture/cgroup-manager.md) /
  [CgroupGroup](architecture/cgroup-group.md) — cgroup plumbing and
  RAII cleanup inside the guest.
