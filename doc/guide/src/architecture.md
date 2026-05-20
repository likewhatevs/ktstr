# Architecture Overview

ktstr has three execution domains:

1. **Host process** -- the test binary running on the host. Manages
   [VM lifecycle](architecture/vmm.md), monitors guest memory, evaluates
   results.

2. **Guest process** -- the same test binary running inside the VM
   as PID 1. Mounts filesystems, starts the scheduler, creates
   cgroups, forks [workers](architecture/workers.md), runs scenarios,
   writes results to virtio-console port 1 (with COM2 fallback for early-boot diagnostics).

3. **[Monitor](architecture/monitor.md) thread** -- runs on the host
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
  |                             +-- write result to virtio-console port 1 (with COM2 fallback for early-boot diagnostics)
  |                           
  +-- read result from virtio-console port 1 (with COM2 fallback for early-boot diagnostics)
  +-- evaluate monitor data   
  +-- report pass/fail        
```

## Key design decisions

**Same binary, two roles.** The test binary serves as both host
controller and guest test runner. The initramfs embeds the binary
as `/init`. When running as PID 1, the Rust init code
(`vmm::rust_init`) handles the full guest lifecycle: mounts,
scheduler start, test dispatch, and reboot.

**Forked workers (default), threads optional.** The default `Fork`
clone mode spawns each worker as its own process so cgroup placement
via `cgroup.procs` is tgid-granular. The `Thread` clone mode shares
the harness's tgid and routes placement through `cgroup.threads`
instead — useful when test bodies need to share address space or
when measuring thread-only scheduler paths. See
[WorkloadHandle](architecture/workload-handle.md) and
[Workers](architecture/workers.md) for the per-mode dispatch table.

**Host-side monitoring.** The monitor reads guest memory via KVM,
avoiding BPF instrumentation of the scheduler under test. This
eliminates observer effects on scheduling decisions.

**Typed flag declarations.** Flags use static references instead of
string matching, enabling compile-time dependency resolution.
