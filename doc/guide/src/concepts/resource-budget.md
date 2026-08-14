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
| Performance mode | `performance_mode = true` | exclusive (`LOCK_EX`), one per virtual LLC — **or** shared (`LOCK_SH`) on a huge LLC (see below) | exclusive (`LOCK_EX`) on every CPU covered by a whole-LLC claim — or on the pinned CPUs plus service CPU at per-CPU grain | vCPU pinning, RT scheduling, hugepages, NUMA mbind |
| Budgeted (no-perf-mode) | `--no-perf-mode` / `KTSTR_NO_PERF_MODE` | shared (`LOCK_SH`) on the planned LLC set | shared (`LOCK_SH`) on the exact budgeted CPU pool; the cgroup cpuset enforces placement | cgroup v2 cpuset sandbox + soft affinity mask |
| Default | neither | shared (`LOCK_SH`) on an exact candidate or fallback pool | instantaneous exact `LOCK_EX` probe, converted to lifetime `LOCK_SH`; fallback is `LOCK_SH` on a `vcpus + 1` pool | exact 1:1 pin when initially unshared; otherwise soft affinity mask |

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
flow through the same admission registry and exact-claim protocol as every
other reservation (see below) — a waiting grain cell holds nothing, and
coordinator partials are the plan's exact per-CPU locks.

Lockfiles live at `{KTSTR_LOCK_DIR or /tmp}/ktstr-llc-{N}.lock` and
`{KTSTR_LOCK_DIR or /tmp}/ktstr-cpu-{C}.lock`. Shared holders coexist;
an exclusive holder blocks overlapping shared acquirers and vice versa.
Performance mode is the only hard non-interference contract. Default
uses CPU EX only as an instantaneous 1:1 availability probe. On success
it converts that same footprint to CPU SH before publishing the run; on
failure it uses the shared budget machinery. Exact-pinned defaults,
default fallbacks, no-perf runs, and builds may therefore overlap
CPU-SH pools, but none can enter CPUs held by performance EX. Kernel
builds take the budgeted path, and so do
`cargo ktstr test` / `cargo ktstr verifier` harness compiles: each
dispatcher runs a reserved `cargo … --no-run` warm-up under the same
shared LLC locks and cpuset sandbox, releasing both before any cell
starts — a harness build on one runner never invades a peer runner's
exclusive reservation, and never contends with its own cells' locks.

<div class="kt-figure"><svg width="700" height="272" viewBox="0 0 700 272" role="img" aria-label="Host resource coordination: a performance-mode run holds whole LLCs and their CPUs under exclusive claims while budgeted no-perf-mode runs share the remaining LLCs and exact CPU pools under shared claims. An exclusive holder blocks every overlapping shared acquirer and vice versa; shared holders coexist. Below, when the process cpuset is narrower than the guest vCPU count the CPU budget collapses to it and the host time-slices the guest's vCPU threads within that admitted shared pool.">
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

## Admission registry: a second-order scheduler under nextest

nextest is a pure spawner: it launches every test process at once and
retries failures. The lock dir is the actual scheduler — it decides
which cells *execute* against host capacity (with the guest scheduler
under test as the third layer down). The coordination protocol that
makes this work across disjoint invocations sharing one
`KTSTR_LOCK_DIR` (colocated runner units, concurrent nextest runs):

- **Fast path.** Every acquirer first tries its whole planned lock
  set non-blocking, all-or-nothing, in canonical lock order. No partial
  survives a bounce. A successful reservation publishes one HELD record
  for its actual lock modes until the physical flocks are dropped.
- **The v8 registry.** A bounced waiting acquirer publishes one monotonic
  fixed-size ticket containing its exact CPU/LLC claim. Each record also
  caches four predecessor-prefix bitsets (CPU-any, CPU-exclusive,
  LLC-any, LLC-exclusive), published epoch-last with the resource
  observation serial that issued the callback. A waiter can therefore
  test a complete alternative without walking every earlier ticket,
  while an improvement that races the callback invalidates that
  callback instead of being consumed unseen. Ordinary
  tickets sleep on targeted shared futex words; only one elected
  coordinator owns exact resource-release watches. Acquisition runs
  *before* VM setup, so a
  waiting cell holds no guest memory or vCPUs. Per-ticket liveness
  flocks make abrupt process death recoverable.
- **Work-conserving grants.** The coordinator scans tickets in arrival
  order. A ticket may run as soon as its exact claim is compatible
  with every earlier live claim, so disjoint work passes blocked work
  without weakening the requested CPU or LLC `LOCK_SH`/`LOCK_EX`
  modes.
- **The coordinator license.** The elected coordinator is the only
  agent allowed to retain a *partial* lock set while waiting for more.
  One incremental acquirer cannot form a deadlock cycle.
- **Ready-alternative re-plan.** The coordinator sleeps on filtered
  inotify events for resource releases and explicit registry
  notifications. A granted flexible ticket evaluates its bounded,
  deterministic alternatives against its cached predecessor prefix and
  the current aggregate availability snapshot. If another complete
  candidate is ready, it publishes that exact claim once for the next
  coordinator pass; it does not blindly rotate through busy plans or
  wake itself for a claim it just abandoned. Partials the selected plan
  still needs are kept; only abandoned ones release.
- **Exact claim fencing.** A shared registry fence spans the aggregate
  claim check and the real all-or-nothing flock probe. No ticket can
  publish in that gap, and a fast path cannot snipe an earlier exact
  reservation. Default's non-atomic CPU EX-to-SH flock conversion uses
  a short registry EX fence spanning the probe, conversion, and weaker
  HELD publication; normal probes require registry SH and therefore
  cannot enter that transition window.
- **Lifecycle-owned bounds.** The lock layer waits for the
  authoritative flock release. It cannot infer that a live holder is
  wedged from elapsed wall time: coverage instrumentation, host
  preemption, and a legitimately long cell make that inference false.
  A crash releases flocks in the kernel, the holder's VM watchdog
  bounds guest progress, and nextest's slow-timeout is the final
  process-lifecycle bound. One slow healthy holder therefore cannot
  fail every waiter behind it.

When default cannot immediately acquire a 1:1 placement, it requests a
shared `vcpus + 1` pool (clamped to the allowed set). It warns only when
that admitted pool is actually narrower than the guest vCPU count.

## Too-small hosts: who asked determines the verdict

The outcome of an unsatisfiable request depends on where the request
came from — an explicit guarantee must never silently degrade, and an
operator typo must not look like a host limitation:

| Request | Error | Outcome |
|---|---|---|
| `performance_mode = true`, host can't honor isolation | `PerfModeUnavailable` | skip (fail under `KTSTR_NO_SKIP_MODE`) |
| Per-test `cpu_budget` above the allowed CPUs | `TopologyInsufficient` | skip (fail under `KTSTR_NO_SKIP_MODE`) |
| Operator `--cpu-cap` / `KTSTR_CPU_CAP` above the allowed CPUs | `CpuBudgetUnsatisfiable` | hard fail |
| Default mode, no immediate 1:1 placement | — | uses shared pool; warns only if pool is narrower than vCPUs |

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
  `cargo ktstr test` / `coverage` / `llvm-cov` / `shell`, and
  `cargo ktstr kernel build` — and it requires `--no-perf-mode`
  (perf mode already holds whole LLCs exclusively, so a cap would
  double-reserve). `KTSTR_CPU_CAP=N` works everywhere; the flag
  wins over the env var. On the test-family commands the cap binds
  the harness prebuild and every test VM, but an on-miss auto
  kernel build stays uncapped by design.

## How a reservation is planned

Budgeted acquisition runs three phases:

1. **Discover** — stat every LLC lockfile and read `/proc/locks` once
   to snapshot current holders. No locks taken.
2. **Plan** — build a bounded, deterministic set of complete candidates
   from the currently available CPUs and LLCs. Prefer strict
   test-NUMA-to-host-NUMA mappings; when no strict mapping exists, fall
   back explicitly to the closest valid cross-node placement. Candidate
   rotation is process-stable and duplicate placements are removed, so
   heterogeneous LLC widths and sparse CPU IDs do not turn into an
   LCM-sized offset search.
3. **Acquire** — non-blocking shared locks on every selected LLC and on
   exactly the budgeted CPUs, all-or-nothing. If any lock is busy, every
   held lock is dropped
   and the whole cycle retries a few times with short ascending
   backoff (absorbing plan/acquire races). Test-run acquisition then
   joins the admission registry and completes as coordinator — choosing
   one ready alternative from the bounded set against live holder state
   on a relevant lock-dir wake
   (see the admission-registry section above); build-time and interactive
   acquisition stop at the short backoff and bail with a
   `ResourceContention` error naming the winning holders.

The LLC claim covers every selected cache domain, while the shared CPU
claim and cpuset hold exactly the budget — the last selected LLC
typically contributes only a subset of its CPUs. When the plan spans
more than one NUMA node,
stderr warns:

```text
ktstr: reserving LLCs [0 (node 0), 2 (node 1)] across 2 NUMA nodes
(preferred single-node contiguous unavailable). Work will proceed;
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
