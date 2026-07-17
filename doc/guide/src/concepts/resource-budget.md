# Resource Budget

ktstr boots KVM VMs and builds kernels on hosts that are usually doing
other things at the same time — more tests, kernel builds, a developer
session. The resource budget is how concurrent ktstr processes share
host CPUs without silently corrupting each other's measurements: every
run reserves host LLCs through advisory file locks, and budgeted runs
are additionally confined to an exact CPU count by a cgroup v2 cpuset
sandbox.

## When to use it

- **Multi-tenant CI hosts** where unbounded parallelism overloads
  concurrent jobs but the full [performance-mode](performance-mode.md)
  contract (RT scheduling, hugepages, NUMA mbind) is too heavy.
- **Kernel builds beside perf-mode tests** — the build's shared lock
  coordinates with the perf-mode exclusive lock, so `make` never
  stomps a measurement in progress.
- **Concurrent no-perf-mode VMs** — a cap of `N` CPUs bounds how much
  capacity each run reserves. Shared-lock peers coexist; they wait only
  behind an incompatible performance-mode exclusive holder.

## The three coordination modes

Every VM run takes one of three coordination paths, selected by two
switches: `performance_mode` on the test or builder, and
`--no-perf-mode` / `KTSTR_NO_PERF_MODE` (any non-empty value).

| Mode | Selected by | LLC lockfiles | Per-CPU lockfiles | Enforcement |
|---|---|---|---|---|
| Performance mode | `performance_mode = true` | exclusive (`LOCK_EX`), one per virtual LLC — **or** shared (`LOCK_SH`) on a huge LLC (see below) | none — **or** exclusive (`LOCK_EX`) per pinned CPU on a huge LLC | vCPU pinning, RT scheduling, hugepages, NUMA mbind |
| Budgeted (no-perf-mode) | `--no-perf-mode` / `KTSTR_NO_PERF_MODE` | shared (`LOCK_SH`) on the planned LLC set | none — the cgroup cpuset is the enforcement layer | cgroup v2 cpuset sandbox + soft affinity mask |
| Default | neither | shared (`LOCK_SH`) on the 1:1 plan's LLCs | exclusive (`LOCK_EX`), one per assigned host CPU | none — reservation only |

### Performance mode on a huge LLC: per-CPU grain

Performance mode's default is whole-LLC exclusion — the cell owns its
entire cache domain, the strongest isolation. That is right when the
cell is a large fraction of its LLC (the validated hosts: the dev box's
16-CPU L3s, x86 CI's ~8-CPU L3s, native arm's 4-CPU L2s, all running
2-4-vCPU cells). It is pathological when the host presents ONE
monolithic L3 spanning scores of CPUs: the AWS Graviton exposes a single
96-CPU L3, so a 2-vCPU perf cell taking `LOCK_EX` on it exclusively locks
the *whole machine*, and every perf cell serializes globally (a 20-60×
makespan blowup versus a many-small-LLC x86 host).

So when a host LLC **dwarfs** the cell, performance mode reserves at
**per-CPU grain** instead: a *shared* (`LOCK_SH`) lock on the giant LLC
so cells coexist on it, plus *exclusive* (`LOCK_EX`) per-CPU locks over
exactly the cell's pinned cores and service CPU. Each cell is placed on a
disjoint, cache-coherent block of the L3 (`vcpus_per_llc + 1` contiguous
CPUs, service CPU included, so distinct blocks never overlap), and the
published claim names those actual CPUs — so peers see the freed capacity
and disjoint perf cells run in parallel. On a 96-CPU L3 a few-CPU
neighbour perturbs the cell's cache share negligibly, so the `D≈1`
isolation contract still effectively holds; the dilation witness and the
`D>1.5` perf-isolation gate catch any residual perturbation.

The switch is gated by two conditions, evaluated per occupied host LLC:

1. **Absolute floor** — the LLC has at least 32 CPUs. This is set
   comfortably above every validated host's LLC (≤ 16) and well below
   the Graviton's 96, so *below the floor the reservation is always
   whole-LLC exclusive* and the measured perf campaign is unchanged on
   every host it was validated on. The floor is what makes the ratio
   below safe: the real perf cells are small (2-4 vCPUs), so on a 16-CPU
   dev-box L3 a cell occupies as little as `2/16 = 0.125` — a pure
   occupancy ratio alone would misclassify the validated hosts.
2. **Occupancy ratio** — the cell occupies less than half of that LLC.
   A cell wanting most of even a huge L3 genuinely needs the cache to
   itself and keeps whole-LLC exclusion regardless of absolute size.

Both must hold on *every* LLC the plan touches; one modest or
cell-filling LLC keeps the whole plan exclusive. Per-CPU-grain perf cells
flow through the same acquisition queue and claim protocol as every other
reservation (see below) — a queued grain cell holds nothing, and its head
partials are the plan's exact per-CPU locks.

Lockfiles live at `{KTSTR_LOCK_DIR or /tmp}/ktstr-llc-{N}.lock` and
`{KTSTR_LOCK_DIR or /tmp}/ktstr-cpu-{C}.lock`. The modes compose:
shared holders coexist with each other, an exclusive holder blocks
every shared acquirer and vice versa. So any number of budgeted and
default runs share LLCs among themselves, a perf-mode run waits for
all of them to release, and while a perf-mode run holds its LLCs
nobody else touches those CPUs. Default runs additionally exclude
each other per CPU, so two default VMs never time-slice the same
host CPU. Kernel builds take the budgeted path, and so do
`cargo ktstr test` / `cargo ktstr verifier` harness compiles: each
dispatcher runs a reserved `cargo … --no-run` warm-up under the same
shared LLC locks and cpuset sandbox, releasing both before any cell
starts — a harness build on one runner never invades a peer runner's
exclusive reservation, and never contends with its own cells' locks.

<div class="kt-figure"><svg width="700" height="272" viewBox="0 0 700 272" role="img" aria-label="Host LLC lock coordination: a performance-mode run holds whole LLCs under exclusive flock (LOCK_EX) while budgeted no-perf-mode runs share the remaining LLCs under shared flock (LOCK_SH). An exclusive holder blocks every shared acquirer and vice versa; shared holders coexist. Below, when the process cpuset is narrower than the guest vCPU count the CPU budget collapses to it and the host time-slices the vCPU threads.">
  <defs><marker id="kt-arr8" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <text x="24" y="20" font-size="12.5" font-weight="700" fill="var(--kt-accent)">host — one flock per LLC</text>
  <rect x="25" y="30" width="310" height="30" rx="8" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.5"/>
  <text x="40" y="49" font-size="10.5" fill="var(--kt-accent)" font-weight="700">run A · performance_mode</text>
  <text x="322" y="49" font-size="9.5" fill="var(--kt-accent)" text-anchor="end">LOCK_EX</text>
  <rect x="345" y="26" width="300" height="19" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.2"/>
  <text x="358" y="39" font-size="9.5" fill="var(--fg)">run B · budgeted</text>
  <text x="632" y="39" font-size="9" fill="var(--fg)" text-anchor="end" opacity=".8">LOCK_SH</text>
  <rect x="355" y="46" width="300" height="19" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.2"/>
  <text x="368" y="59" font-size="9.5" fill="var(--fg)">run C · budgeted</text>
  <text x="642" y="59" font-size="9" fill="var(--fg)" text-anchor="end" opacity=".8">LOCK_SH</text>
  <path d="M100 60 L 100 106" stroke="var(--kt-accent)" stroke-width="1.3" marker-end="url(#kt-arr8)"/>
  <path d="M260 60 L 260 106" stroke="var(--kt-accent)" stroke-width="1.3" marker-end="url(#kt-arr8)"/>
  <path d="M420 65 L 420 106" stroke="var(--fg)" stroke-width="1.2" opacity=".55" marker-end="url(#kt-arr8)"/>
  <path d="M580 65 L 580 106" stroke="var(--fg)" stroke-width="1.2" opacity=".55" marker-end="url(#kt-arr8)"/>
  <g fill="var(--fg)">
    <rect x="25" y="110" width="150" height="52" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
    <text x="100" y="133" text-anchor="middle" font-size="11" font-weight="700">LLC 0</text>
    <text x="100" y="150" text-anchor="middle" font-size="9" fill="var(--kt-accent)">held · exclusive</text>
    <rect x="185" y="110" width="150" height="52" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
    <text x="260" y="133" text-anchor="middle" font-size="11" font-weight="700">LLC 1</text>
    <text x="260" y="150" text-anchor="middle" font-size="9" fill="var(--kt-accent)">held · exclusive</text>
    <rect x="345" y="110" width="150" height="52" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.3"/>
    <text x="420" y="133" text-anchor="middle" font-size="11" font-weight="700">LLC 2</text>
    <text x="420" y="150" text-anchor="middle" font-size="9" opacity=".8">shared · B + C</text>
    <rect x="505" y="110" width="150" height="52" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.3"/>
    <text x="580" y="133" text-anchor="middle" font-size="11" font-weight="700">LLC 3</text>
    <text x="580" y="150" text-anchor="middle" font-size="9" opacity=".8">shared · B + C</text>
  </g>
  <text x="24" y="184" font-size="9.5" fill="var(--fg)" opacity=".78">Exclusive blocks every shared acquirer and vice versa — shared holders coexist; a perf-mode run waits for release.</text>
  <line x1="24" y1="200" x2="676" y2="200" stroke="var(--kt-rule)" stroke-width="1"/>
  <g opacity=".62">
    <text x="24" y="222" font-size="10.5" font-weight="700" fill="var(--fg)">budget collapse — process cpuset narrower than the guest</text>
    <rect x="24" y="230" width="150" height="34" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="99" y="251" text-anchor="middle" font-size="10" fill="var(--fg)">guest · 16 vCPUs</text>
    <path d="M176 247 L 232 247" stroke="var(--fg)" stroke-width="1.2" marker-end="url(#kt-arr8)"/>
    <rect x="236" y="230" width="150" height="34" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="311" y="251" text-anchor="middle" font-size="10" fill="var(--fg)">8 allowed host CPUs</text>
    <text x="400" y="244" font-size="9.5" fill="var(--fg)">2× oversubscription — the host time-slices the vCPU</text>
    <text x="400" y="257" font-size="9.5" fill="var(--fg)">threads; guest-scheduler measurement is confounded.</text>
  </g>
</svg></div>

## The queue: a second-order scheduler under nextest

nextest is a pure spawner: it launches every test process at once and
retries failures. The lock dir is the actual scheduler — it decides
which cells *execute* against host capacity (with the guest scheduler
under test as the third layer down). The coordination protocol that
makes this work across disjoint invocations sharing one
`KTSTR_LOCK_DIR` (colocated runner units, concurrent nextest runs):

- **Fast path.** Every acquirer first tries its whole planned lock
  set non-blocking, all-or-nothing, in canonical lock order — cells
  satisfiable *right now* never queue, and no fast-path partial ever
  persists (everything is released on any bounce).
- **The queue.** A bounced acquirer joins a single queue flock in the
  lock dir (arrival order via the kernel's flock wait queue; tickets
  are flocks, so a crashed waiter vanishes with its process). Waiters
  are cheap: acquisition runs *before* VM setup, so a queued cell
  holds a ticket and an inotify fd — no guest memory, no vCPUs. That
  is what lets nextest admit hundreds of cells at once.
- **The head license.** The queue holder — the head — is the only
  agent anywhere allowed to hold a *partial* lock set while waiting
  for more. One incremental acquirer cannot deadlock (a cycle needs
  two holders-waiting) and cannot starve (it only accumulates), which
  is the protocol's core safety argument.
- **Re-plan on every wake.** The head sleeps on an inotify watch of
  the lock dir; any lockfile release wakes it, and it re-plans
  against *live* holder state — plans are never cached across waits,
  so a freed resource that satisfies a different plan than the one
  that was busy is taken immediately. Partials the fresh plan still
  needs are kept; only abandoned ones release.
- **Claims.** The head publishes its current target set; fast-path
  planners in every process subtract it from their view of free
  capacity, so nobody snipes the slots the head is accumulating —
  while any *other* free capacity stays fair game (work conservation:
  a small `LOCK_SH` cell never waits behind a hungry `LOCK_EX` head
  it doesn't overlap).
- **Lifecycle-owned bounds.** The lock layer waits for the
  authoritative flock release. It cannot infer that a live holder is
  wedged from elapsed wall time: coverage instrumentation, host
  preemption, and a legitimately long cell make that inference false.
  A crash releases flocks in the kernel, the holder's VM watchdog
  bounds guest progress, and nextest's slow-timeout is the final
  process-lifecycle rail. One slow healthy holder therefore cannot
  fail every waiter behind it.

When the default path cannot map its topology 1:1 onto the host at
all (no plan can exist — host too small), the run proceeds
*overcommitted* — every vCPU thread masked to the allowed CPUs — and
warns when that means oversubscription (see below).

## Too-small hosts: who asked determines the verdict

The outcome of an unsatisfiable request depends on where the request
came from — an explicit guarantee must never silently degrade, and an
operator typo must not look like a host limitation:

| Request | Error | Outcome |
|---|---|---|
| `performance_mode = true`, host can't honor isolation | `PerfModeUnavailable` | skip (fail under `KTSTR_NO_SKIP_MODE`) |
| Per-test `cpu_budget` above the allowed CPUs | `TopologyInsufficient` | skip (fail under `KTSTR_NO_SKIP_MODE`) |
| Operator `--cpu-cap` / `KTSTR_CPU_CAP` above the allowed CPUs | `CpuBudgetUnsatisfiable` | hard fail |
| Default mode, no 1:1 placement possible | — | runs overcommitted, warns |

A test attribute is a capability requirement a bigger host would
satisfy, so it skips. An operator-typed number that does not exist on
this host is a misconfiguration, so it fails. The over-cap error names
both numbers:

```text
--cpu-cap N = 96 exceeds the 64 CPUs this process is allowed on (from
sched_getaffinity / Cpus_allowed_list). Pick a value ≤ 64, release the
cgroup/taskset constraint restricting this process, or omit --cpu-cap
to use the auto-sized default (30% of the allowed set for kernel
builds; min(vCPUs + 1, allowed CPUs) for no-perf VMs).
```

No-perf mode records intentional sharing without prescribing a metric or
changing placement:

```text
ktstr: 16 guest vCPUs share 8 host CPUs (2.0x oversubscribed;
no-perf/cpu-budget mode)
```

The unrestricted fallback emits a warning when host capacity is below
guest width:

```text
ktstr: WARNING: 16 guest vCPUs share 8 host CPUs (2.0x oversubscribed;
host capacity is below guest width)
```

The stamped `cpu_budget` in the run's sidecar also drops below the
vCPU count, so an A/B comparison against an overcommitted run is
flagged rather than silently confounded.

## The CPU budget

The budget is resolved in precedence order:

1. `--cpu-cap N` on the command line.
2. `KTSTR_CPU_CAP=N` when the flag is absent (empty string = unset).
3. Neither: kernel builds get 30% of the allowed CPUs (rounded up,
   minimum 1); no-perf-mode VMs get `min(vcpus + 1, allowed)` (minimum
   1). The extra host CPU leaves room for the VMM/control threads without
   making a small VM reserve 30% of a large host. If the allowed cpuset
   is narrower than the guest, the budget clamps to that cpuset and the
   sharing note above is emitted. An explicit cap below the vCPU count is
   the deliberate opt-in to oversubscription.

`0` is rejected with `--cpu-cap must be ≥ 1 CPU (got 0)` — zero is a
scripting sentinel, not a silent "no cap".

The reference set is the calling process's *allowed* CPUs
(`sched_getaffinity`, with a `/proc/self/status` fallback), not the
host's online count — so the reservation stays valid under
cgroup-restricted CI runners. An empty allowed set is a hard error:
guessing on a misconfigured host is worse than failing visibly.

A per-test `cpu_budget` attribute on `#[ktstr_test]` overrides the
auto-size for that test; an operator `--cpu-cap` / `KTSTR_CPU_CAP`
wins over both.

## Flag availability

- `--no-perf-mode`: `cargo ktstr test` / `coverage` / `llvm-cov` /
  `shell`, and `ktstr shell`. `KTSTR_NO_PERF_MODE` (any non-empty
  value) works everywhere.
- `--cpu-cap N`: `ktstr shell`, `ktstr kernel build`,
  `cargo ktstr shell`, `cargo ktstr kernel build` — and it requires
  `--no-perf-mode` (perf mode already holds whole LLCs exclusively,
  so a cap would double-reserve). For `cargo ktstr test` /
  `coverage` / `llvm-cov` set `KTSTR_CPU_CAP=N` instead.

## How a reservation is planned

Budgeted acquisition runs three phases:

1. **Discover** — stat every LLC lockfile and read `/proc/locks` once
   to snapshot current holders. No locks taken.
2. **Plan** — rank LLCs: prefer LLCs that already have holders
   (consolidation packs shared runs together), seed on the
   best-ranked LLC's NUMA node, and greedily fill that node before
   spilling to nearest-by-distance neighbors. Accumulate LLCs until
   their allowed CPUs cover the budget.
3. **Acquire** — non-blocking shared locks on every selected LLC,
   all-or-nothing. If any lock is busy, every held lock is dropped
   and the whole cycle retries a few times with short ascending
   backoff (absorbing plan/acquire races). Test-run acquisition then
   joins the acquisition queue and completes as head — re-running
   DISCOVER → PLAN against live holder state on every lock-dir wake
   (see the queue section above); build-time and interactive
   acquisition stop at the short backoff and bail with a
   `ResourceContention` error naming the winning holders.

The lock granularity is per-LLC, but the reserved CPU list holds
exactly the budget — the last selected LLC typically contributes only
a prefix of its CPUs. When the plan spans more than one NUMA node,
stderr warns:

```text
ktstr: reserving LLCs [0 (node 0), 2 (node 1)] across 2 NUMA nodes
(preferred single-node contiguous unavailable). Build will run;
memory-access latency may be higher.
```

## Cgroup v2 cpuset sandbox

Budgeted runs write the reserved CPUs and their NUMA nodes into a
child cgroup — `cpuset.cpus`, then `cpuset.mems`, then the pid into
`cgroup.procs`, in that order because the kernel may kill a task
migrated into a cgroup whose `cpuset.mems` is still empty. After each
write the effective value is read back: narrowing by a parent cgroup
(a systemd slice, a container limit) is a fatal error under an
explicit `--cpu-cap` and a warning otherwise. Kernel builds inside
the sandbox also get their `make -j` width set to the reserved CPU
count — without that, `make -j$(nproc)` fans gcc children out to a
width the cpuset then has to time-slice, silently defeating the
budget in scheduling terms.

## Observing locks

`ktstr locks` (or `cargo ktstr locks`) prints every ktstr lock
currently held on the host — LLC, per-CPU, kernel-cache, and run-dir
locks — with each holder's PID and command line. It is read-only and
takes no locks itself. Use it when an acquire fails with
`ResourceContention`: the error names the busy LLCs, the snapshot
shows every contending peer at once. The full output and flags are in
[cargo ktstr locks](../running-tests/cargo-ktstr.md#locks).

## `KTSTR_BYPASS_LLC_LOCKS` — escape hatch

Setting `KTSTR_BYPASS_LLC_LOCKS=1` skips lock acquisition entirely:
the VM boots or the build starts immediately, with no coordination
against concurrent runs. Use it only when measurement noise is
acceptable — an isolated workstation, or a CI queue that already
serializes jobs at a higher layer. It is mutually exclusive with
`--cpu-cap` / `KTSTR_CPU_CAP` at every entry point; the rejection
message always contains `"resource contract"` so it is greppable.

## Filesystem requirement

Every lockfile path must live on a local filesystem — tmpfs, ext4,
xfs, btrfs, f2fs, and bcachefs are the accepted set. NFS, CIFS/SMB,
CephFS, AFS, and FUSE mounts are rejected at open time: `flock(2)`
coordination or `/proc/locks` holder enumeration is unreliable on
these configurations, and ktstr refuses to run on a lock it cannot
trust. The error names the offending filesystem and the fix: move the
lockfile path (`KTSTR_LOCK_DIR`, the cache root, or the runs root) to
a local filesystem. Unknown-but-local filesystems (zfs, erofs, ...)
pass through.

## Related

- [Run Modes](run-modes.md) — how this budgeted path sits alongside the
  default and performance modes, and the shared reliability machinery.
- [Performance Mode](performance-mode.md) — the full-isolation mode.
- [Environment Variables](../reference/environment-variables.md) —
  `KTSTR_CPU_CAP`, `KTSTR_LOCK_DIR`, `KTSTR_BYPASS_LLC_LOCKS`.
