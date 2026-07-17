# Performance Mode

Without performance mode, a 50ms scheduling gap in a measurement could
be host noise; with it, the same gap indicates a scheduler problem.
Performance mode removes host-side variance — vCPU threads pinned to
dedicated cores, hugepage-backed guest memory, NUMA-local allocation,
real-time scheduling — so timing thresholds measure the scheduler
under test, not the host it happens to share.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Reserve</strong><p>Use host CPU locks and topology checks so noisy neighbors do not share the same cores.</p></div>
<div class="kt-doc-card"><strong>Pin</strong><p>Bind vCPU threads, memory policy, and hugepages to the reserved host resources.</p></div>
<div class="kt-doc-card"><strong>Measure</strong><p>Use tighter gap, latency, and iteration thresholds only when perf mode is actually available.</p></div>
</div>

## Usage

```rust,ignore
#[ktstr_test(
    llcs = 2,
    cores = 4,
    threads = 2,
    performance_mode = true,
)]
fn my_perf_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

The VM builder API takes the same switch:
`KtstrVm::builder().performance_mode(true)`.

## When to use

Performance mode is for tests where host-side scheduling noise affects
results — fairness spread measurements, scheduling gap detection,
imbalance ratio checks. It is not needed for correctness tests (cpuset
isolation, zero-work detection) where pass/fail is binary.

The gauntlet runs many VMs in parallel. Performance mode on parallel
VMs can oversubscribe the host if scheduled naively. Avoid
`performance_mode` unless the host has enough CPUs for the topology
matrix.

With stable measurements, tests can set tight thresholds
(`max_gap_ms`, `min_iteration_rate`, `max_p99_wake_latency_ns`) to
catch regressions against a fixed bar;
[`cargo ktstr perf-delta`](../running-tests/runs.md) builds on the
same tests to catch regressions against a previous commit. Perf-mode
results are comparable only against runs on the same host — guest-side
jitter from shared caches and memory bandwidth remains.

## What it does

On x86_64:

- **vCPU pinning** — each virtual LLC maps to a physical LLC group
  and vCPU threads are pinned to cores within it, so the host
  scheduler cannot migrate them across cache domains mid-measurement.
- **Hugepages** — guest memory is allocated from 2MB hugepages when
  enough are free, eliminating host-side TLB pressure.
- **NUMA mbind** — guest memory is bound (`MPOL_BIND`, strict — no
  silent fallback to remote nodes) to the NUMA nodes of the pinned
  vCPUs.
- **RT scheduling** — vCPU threads run `SCHED_FIFO` priority 1; the
  monitor and watchdog run at priority 2 on a dedicated host CPU no
  vCPU shares, so sampling and timeout enforcement can always
  preempt a vCPU thread.
- **PAUSE and HLT exit suppression** — guest spinlock `PAUSE` loops
  and idle `HLT` normally trap to the hypervisor so it can schedule
  other vCPUs; with dedicated cores that reschedule is pure overhead,
  so both exits are disabled. (HLT disable is skipped when the host's
  SMT-RSB mitigation forbids it; PAUSE alone is still disabled.)
- **KVM_HINTS_REALTIME** — a CPUID hint telling the guest kernel its
  vCPUs own dedicated cores; the guest drops paravirt yield paths and
  polls briefly before halting instead of paying wakeup latency.

On aarch64, the four host-side items apply (pinning, hugepages, NUMA
mbind, RT scheduling); the x86-specific exit suppression and CPUID
hint do not exist there.

## Prerequisites

**Sufficient host CPUs** — at least
`(llcs * cores * threads) + 1` online CPUs; the extra CPU hosts the
monitor and watchdog threads. The host also needs at least as many
physical LLC groups as the test declares virtual LLCs.

**2MB hugepages** (optional) — check
`/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages`. Without
them guest memory uses regular pages and a warning is printed.

**CAP_SYS_NICE or an rtprio limit** (optional) — `SCHED_FIFO`
requires root or `RLIMIT_RTPRIO` at or above the requested priority.
For non-root use:

```text
# /etc/security/limits.conf
username  -  rtprio  99
```

Log out and back in for the limit to take effect. Without it, RT
scheduling is skipped with a warning and results may be noisier.

## Sizing the host

A single perf-mode test needs `(llcs * cores * threads) + 1` online
CPUs and `llcs` free physical LLC groups — the test holds an
exclusive lock on one host LLC group per virtual LLC for the run's
duration. To run `K` perf-mode tests concurrently without contention
delays, the host needs `K * llcs` free LLC groups; with fewer, the
excess tests join the lock-dir acquisition queue and proceed as each
holder releases. The queue does not infer that a live holder is wedged
from elapsed wall time: the holder's VM watchdog owns guest progress,
nextest owns the final process-lifecycle rail, and a crash releases the
flock in the kernel. The
`vm-perf` test group in `.config/nextest.toml` caps how many run at
once.

## Failure modes

Performance mode never runs unisolated: if the host cannot honor the
guarantee, the build fails before boot and the test skips visibly
rather than shipping a measurement that does not match what was asked
for.

- **`PerfModeUnavailable`** — permanent host insufficiency: too few
  CPUs or LLC groups for the topology, no satisfiable pinning plan,
  or no free CPU left for service threads. Skips by default with a
  visible banner (`ktstr: SKIP: <reason>` on stderr, exit 0, skip
  recorded in the run sidecar); promoted to a hard fail under
  `KTSTR_NO_SKIP_MODE` for runs that demand execution.
- **`ResourceContention`** — transient host-resource pressure outside
  queued test acquisition (for example retryable KVM allocation
  errors), or a busy reservation in a one-shot non-waiting command.
  It produces a `FAIL: transient resource contention` banner that
  nextest retries — never a skip. Queued tests wait for a lock holder,
  so a busy host costs throughput, not coverage.
- **Warnings (non-fatal)** — insufficient free hugepages (regular
  pages used); high host load (`procs_running` above half the vCPU
  count — results may be noisy); unstable TSC (x86_64, common in
  nested virtualization — timing variance is higher).

The full skip-vs-fail model — which requester gets a skip, which gets
a hard error, and what the default path does instead — is in
[Resource Budget](resource-budget.md#too-small-hosts-who-asked-determines-the-verdict).

## Disabling performance mode

`--no-perf-mode` (or `KTSTR_NO_PERF_MODE=1`) forces
`performance_mode = false` and routes the run through the budgeted
coordination path: a shared LLC reservation sized to a CPU budget,
enforced by a cgroup cpuset instead of pinning — none of the
isolation features above apply. [Run Modes](run-modes.md) sets this mode
beside the default and performance modes; the CPU budget, the `--cpu-cap`
flag, and the coordination-mode comparison live in
[Resource Budget](resource-budget.md#the-three-coordination-modes).
