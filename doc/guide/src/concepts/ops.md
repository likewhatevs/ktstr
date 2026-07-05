# Ops, Steps, and Backdrop

Dynamic scenarios used to mean hand-written scenario bodies that
create cgroups, sleep, poke sysfs, sleep again, and collect — with
every ordering rule and error path yours to get right:

```rust,ignore
// By hand: manual setup, manual timing, manual teardown.
fn shrink_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let (mgr, cgroups, handles) = setup_cgroups(ctx, /* defs */)?;
    std::thread::sleep(first_half);
    mgr.set_cpuset("cg_hot", &ctx.topo.llc_aligned_cpuset(0))?;
    std::thread::sleep(second_half);
    // ...collect every handle, run checks, tear down in order...
}
```

The ops system expresses the same scenario declaratively — the
framework owns timing, teardown, liveness checking, and report
collection:

```rust,ignore
execute_steps(ctx, vec![
    Step::with_defs(
        vec![CgroupDef::named("cg_hot").workers(4)],
        HoldSpec::frac(0.5),
    ),
    Step::with_op(
        Op::set_cpuset("cg_hot", CpusetSpec::llc(0)),
        HoldSpec::frac(0.5),
    ),
])
```

## Op

An `Op` is one atomic operation on the running scenario. The enum is
`#[non_exhaustive]` — pattern matches must end with `..`.

| Op | Effect |
|---|---|
| `AddCgroup` | Create an empty cgroup |
| `AddCgroupDef` | Create cgroup + cpuset + workers from a `CgroupDef`, mid-step |
| `RemoveCgroup` | Stop workers and remove a cgroup |
| `StopCgroup` | Stop a cgroup's workers, keep the cgroup |
| `SetCpuset` / `ClearCpuset` / `SwapCpusets` | Set, clear, or swap cgroup cpusets |
| `Spawn` | Spawn workers into a named cgroup or the runner's own cgroup |
| `SetAffinity` | Set worker affinity via `AffinityIntent` |
| `MoveAllTasks` | Move all tasks from one cgroup to another |
| `FreezeCgroup` / `UnfreezeCgroup` | Kernel-side cgroup freeze (not SIGSTOP); teardown auto-unfreezes |
| `SteerIrq` | Re-steer a hardware IRQ to one CPU (system-wide, not cpuset-scoped) |
| `CaptureSnapshot` | On-demand host-side snapshot of BPF maps, vCPU registers, per-CPU counters |
| `WatchSnapshot` | Snapshot every time the guest writes a named kernel symbol |
| `CaptureCgroupProcs` | Record a cgroup's `cgroup.procs` PIDs under a tag |
| `ReadKernelHot` / `ReadKernelCold` | Read kernel memory (symbol, KVA, per-CPU or task field) |
| `WriteKernelHot` / `WriteKernelCold` | Write kernel memory; adjacent cold writes are batched |
| `RunPayload` / `WaitPayload` / `KillPayload` | Launch, await, or kill a binary payload |
| `AttachScheduler` / `DetachScheduler` / `RestartScheduler` / `ReplaceScheduler` | Manage the live scheduler mid-scenario |
| `PinBpfMap` | Hold a BPF map fd open across a scheduler swap |

Constructors take string literals directly (no `.into()`):

```rust,ignore
Op::add_cgroup("cg_0")
Op::add_cgroup_def(CgroupDef::named("cg_1").workers(4))
Op::set_cpuset("cg_0", CpusetSpec::disjoint(0, 2))
Op::spawn_workers("cg_0", WorkSpec::default().workers(4))
Op::spawn_host(WorkSpec::default().workers(4))
Op::set_affinity("cg_0", AffinityIntent::random_subset([0, 1, 2, 3], 2))
Op::capture_snapshot("after_spawn")
Op::freeze_cgroup("cg_0")
```

`Op::spawn_host` puts workers in the test runner's own cgroup —
typically the guest root — to simulate host-level contention beside
managed cgroups.

### Snapshot and watch ops

`CaptureSnapshot` pauses every vCPU through the freeze coordinator,
reads BPF map state, vCPU registers, and per-CPU counters, then
resumes; the report is keyed by the op's name. With no snapshot
bridge installed it fails loudly rather than dropping the capture.
`WatchSnapshot` fires one capture per guest write to the named
symbol; the name must match the guest kernel's vmlinux symbol table
verbatim, and at most 3 watch ops fit in a scenario (hardware debug
slots; one is reserved for the error-exit trigger). Details and
failure modes: [Snapshots](../writing-tests/snapshots.md) and
[Watch Snapshots](../writing-tests/watch-snapshots.md).

### Kernel-memory ops

`Hot` variants read/write against the running vCPU; `Cold` variants
take a freeze rendezvous first. Targets are `KernelTarget`s: a
symbol, a kernel virtual address, a per-CPU field, or a task field.

### Payload ops

`RunPayload` spawns a binary-kind
[`Payload`](../writing-tests/payloads.md) in the background;
`WaitPayload` blocks until it exits naturally, then evaluates its
checks and records its metrics; `KillPayload` does the same after
SIGKILL. Payloads are addressed by `(name, cgroup)`; `cgroup: None`
resolves to the unique live copy. `WaitPayload` has no timeout — pair
it with a bounded hold or the payload's own runtime flag.
Scheduler-kind payloads are rejected: the scheduler slot is the
`#[ktstr_test(scheduler = ...)]` attribute.

### Scheduler ops

`ReplaceScheduler` swaps to a different staged scheduler binary
(declared via `#[ktstr_test(staged_schedulers = [...])]`).
`PinBpfMap` keeps a map fd alive so a same-binary swap window's
`.bss` survives the replacement.

### Permissive removal — a footgun

`Op::RemoveCgroup` and `Op::StopCgroup` are permitted against any
cgroup, including [Backdrop](#backdrop)-owned ones, and removing a
nonexistent cgroup silently succeeds (rmdir on a missing path is a
no-op). A typo'd name therefore surfaces later, as the kernel's
`No such file or directory` on the next op that references the real
name. If a later step fails with a missing-cgroup error, grep the
test for `Op::remove_cgroup` calls naming a similar identifier
first. `Op::MoveAllTasks` is the exception: it rejects moves that
would strand Backdrop workers in a step-local cgroup.

## CpusetSpec

`CpusetSpec` computes a cpuset from the topology at runtime. Build
via constructors — the enum is `#[non_exhaustive]`:

```rust,ignore
pub enum CpusetSpec {
    Llc(usize),                               // all CPUs in one LLC
    Numa(usize),                              // all CPUs in one NUMA node
    Range { start_frac: f64, end_frac: f64 }, // fraction of usable CPUs
    Disjoint { index: usize, of: usize },     // equal disjoint partitions
    Overlap { index: usize, of: usize, frac: f64 }, // overlapping partitions
    Exact(BTreeSet<usize>),                   // caller-supplied set
}
```

`CpusetSpec::llc(0)`, `CpusetSpec::numa(0)`, `CpusetSpec::range(0.0,
0.5)`, `CpusetSpec::disjoint(0, 2)`, `CpusetSpec::overlap(0, 2,
0.5)`, `CpusetSpec::exact([0, 1, 2])`. Fractional and partition
variants operate on
[`usable_cpus()`](topology.md#topology-queries); `Llc` and `Numa`
cover their full domain.

## CgroupDef

`CgroupDef` bundles the three ops that always travel together —
create cgroup, set cpuset, spawn workers — and is the primary way to
declare cgroups:

```rust,ignore
let def = CgroupDef::named("cg_0")
    .cpuset(CpusetSpec::disjoint(0, 2))
    .workers(4)
    .work_type(WorkType::SpinWait);
```

Builder methods:

- `.cpuset(CpusetSpec)` / `.cpuset_mems(set)` — CPU set, and an
  explicit `cpuset.mems` override (default derives from the cpuset's
  NUMA nodes).
- `.workers(n)` / `.workers_pct(p)` — worker count, absolute or as a
  fraction of the resolved cpuset (see below). Setting both is
  rejected with a diagnostic.
- `.work_type(WorkType)` — what workers do (default `SpinWait`); see
  [Work Types](work-types.md).
- `.work(WorkSpec)` — add another worker group; call repeatedly for
  concurrent groups.
- `.workload(&'static Payload)` — run a binary payload inside the
  cgroup alongside the workers. Panics on a scheduler-kind payload
  (there is no scenario-level recovery at build time; the step-level
  `Op::RunPayload` returns an error instead).
- `.sched_policy(SchedPolicy)` — Linux scheduling policy (default
  `Normal`); see [Scheduling policies](work-types.md#scheduling-policies).
- `.affinity(AffinityIntent)` — per-worker affinity (default
  `Inherit`).
- `.mem_policy(MemPolicy)` / `.mpol_flags(MpolFlags)` — NUMA memory
  placement; see [MemPolicy](mem-policy.md).
- `.nice(n)`, `.comm(name)`, `.pcomm(name)`, `.uid(u)` / `.gid(g)`,
  `.numa_node(node)` — per-worker identity defaults, merged into
  every `WorkSpec` that doesn't set its own.
- `.swappable(bool)` — opt into gauntlet work-type overrides (see
  below).

Cgroup-v2 controller knobs (default unconstrained):
`.cpu_quota_pct(pct)` / `.cpu_quota(quota, period)` /
`.cpu_unlimited()`, `.cpu_weight(w)`, `.memory_max(b)` /
`.memory_high(b)` / `.memory_low(b)` / `.memory_unlimited()`,
`.memory_swap_max(b)` / `.memory_swap_unlimited()`, `.io_weight(w)`,
`.pids_max(n)` / `.pids_unlimited()`.

### Cpuset-scaled worker counts

Tests that span topologies need worker counts that scale with the
cpuset. Hand-computing couples the test to a manual resolution step:

```rust,ignore
// Before: hand-computed via Ctx::cpuset_cpus.
let n = (ctx.cpuset_cpus(&CpusetSpec::Llc(0)) as f64 * 0.9).ceil() as usize;
let def = CgroupDef::named("cg_hot").cpuset(CpusetSpec::Llc(0)).workers(n);

// After: resolved from the cgroup's own cpuset at apply time.
let def = CgroupDef::named("cg_hot")
    .cpuset(CpusetSpec::Llc(0))
    .workers_pct(0.9);  // ceil(cpuset_cpus * 0.9)
```

Fractions above `1.0` are accepted as deliberate oversubscription.

## Work-type overrides and swappable {#workspec-type-overrides-and-swappable}

A gauntlet run can sweep work types (`--ktstr-work-type=NAME`,
surfaced as `Ctx.work_type_override`). The override replaces a
def's work type only when that `CgroupDef` is marked
`.swappable(true)` (default `false`), and is skipped when the
override is a grouped work type whose group size does not divide the
resolved worker count. Non-swappable defs keep their declared type;
`Op::Spawn` always uses the type as given. This is the single
override mechanism for both `#[ktstr_test]` and ops-based scenarios.

## Step

A `Step` is a list of ops plus a hold period:

```rust,ignore
pub struct Step {
    pub setup: Setup,   // CgroupDefs to create (after ops run)
    pub ops: Vec<Op>,   // operations to apply
    pub hold: HoldSpec, // how long to hold afterward
}
```

`Setup` is `Defs(Vec<CgroupDef>)` or a topology-dependent
`Setup::with_factory(fn(&Ctx) -> Vec<CgroupDef>)`.

Constructors:

- `Step::with_defs(defs, hold)` — the primary constructor: create
  cgroups with workers, hold.
- `Step::new(ops, hold)` — ops only, no cgroup setup.
- `Step::hold(hold)` — hold only; the canonical phase-A shape in an
  A/B scenario (`Step::hold(HoldSpec::frac(0.3))` then
  `Step::with_op(Op::replace_scheduler(&ALT), HoldSpec::frac(0.7))`).
- `Step::with_op(op, hold)` — one op, then hold.
- `Step::with_payload(payload, hold)` — run one binary payload for
  the hold; it is drained at step teardown, nothing blocks on it.

Builder methods `Step::set_ops` and `Step::set_hold` *replace* their
field. The verb prefixes are consistent across the API: `set_X`
replaces, `push_X` appends one, `extend_X` appends many — so
`Step::new(ops).set_ops(more)` drops `ops`, while
`Backdrop::new().extend_ops(a).extend_ops(b)` accumulates both.

## HoldSpec

| Variant | Meaning |
|---|---|
| `Frac(f64)` | Fraction of the scenario duration |
| `Fixed(Duration)` | Fixed time |
| `Loop { interval }` | Re-apply the step's ops at `interval` until time runs out |

Sugar: `HoldSpec::frac(0.5)`, `HoldSpec::fixed(d)`,
`HoldSpec::loop_at(d)`, and `HoldSpec::FULL` for `Frac(1.0)`.

A `Loop` step is the natural shape for "every N seconds, do X":

```rust,ignore
Step::new(
    vec![Op::capture_snapshot("periodic")],
    HoldSpec::loop_at(Duration::from_secs(2)),
)
```

The step's setup runs once at step entry; only the ops repeat. Prior
steps' step-local state was already torn down at their own
boundaries, so each loop iteration sees only Backdrop-owned state
and this step's own setup.

## Backdrop

Steps tear their state down at each step boundary. A `Backdrop` is
the scenario-wide layer for state that must persist across steps —
long-lived cgroups every step references, background payloads that
run for the whole scenario, setup ops that seed state once:

```rust,ignore
let backdrop = Backdrop::new()
    .push_cgroup(CgroupDef::named("bg_cell").cpuset(CpusetSpec::disjoint(0, 2)))
    .push_op(Op::add_cgroup("bg_overflow"))   // empty move-target cgroup
    .push_payload(&BG_LOAD);

execute_scenario(ctx, backdrop, steps)
```

- `Backdrop::from_cgroups([...])` builds one from `CgroupDef`s;
  `push_cgroup` / `extend_cgroups`, `push_op` / `extend_ops`, and
  `push_payload` / `extend_payloads` compose incrementally.
- Backdrop cgroups are created in declaration order, before the
  first step; every `CgroupDef` spawns at least one worker. Declare
  empty cgroups (move targets) via `push_op(Op::add_cgroup(...))`.
- Backdrop ops run after the cgroups, before the payloads, with full
  authority — they may remove or stop Backdrop cgroups where
  step-local ops are restricted.
- Payloads are spawned once and drained (killed, metrics preserved)
  at scenario teardown.

Any step can reference Backdrop cgroups by name (`Op::MoveAllTasks`,
`Op::SetCpuset`, ...). The Backdrop tears down after the last step.

## Executors

All in the prelude; each returns `Result<AssertResult>`:

- `execute_defs(ctx, defs)` — the one-shot path: create cgroups,
  run for the full duration, collect. Equivalent to
  `execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])`.
- `execute_steps(ctx, steps)` — run a step sequence: for each step,
  apply ops, then setup, then hold (`Loop` steps run setup once,
  then repeat ops); check scheduler liveness between steps; collect
  worker reports and run checks at the end.
- `execute_steps_with(ctx, steps, Some(&assert))` — same, with an
  explicit [`Assert`](checking.md) overriding `ctx.assert` for
  worker checks. `None` falls back to `ctx.assert` (the merged
  scheduler + per-test config).
- `execute_scenario(ctx, backdrop, steps)` /
  `execute_scenario_with(ctx, backdrop, steps, checks)` — the full
  composition: Backdrop setup, step sequence with per-step teardown,
  Backdrop teardown.

## Phases

Steps give a scenario its timeline. The framework publishes the
active phase as steps progress: captures (periodic samples, watch
trips, on-demand snapshots) stamp with the phase active at capture
time, and every assertion detail constructed during a step's hold
auto-stamps with that step's label. Labels render as `BASELINE` (the
settle window before step 0) and `Step[k]` everywhere — sidecar
JSON, the timeline diagnostic, and per-assertion `phase` fields.

Phase-bucketed metrics are queryable from the result:

```rust,ignore
let baseline = r.stats.phase(Phase::BASELINE).expect("always populated");
let step_0   = r.stats.phase(Phase::step(0)).expect("Step 0 ran");
let thr      = r.stats.phase_metric(Phase::step(0), "throughput");
```

Gate on `r.stats.has_steps()` before assuming step buckets exist —
a scenario that bailed in setup returns `None` from every
`phase(Phase::step(k))` lookup. `PhaseBucket::expect_metric` panics
with the bucket's label, sample count, and the metric keys actually
present, so a typo'd name and an empty phase are distinguishable at
a glance. [Temporal Assertions](../writing-tests/temporal-assertions.md)
builds per-phase pattern checks on top of this.

The per-phase timeline also renders in every failure report:

<!-- captured: cargo ktstr test --kernel 7.0 (throughput_gate demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
--- timeline ---
topology: 1n1l2c1t (2 cpus)  scheduler: my_sched  scenario: throughput_gate  duration: 15.0s

Phase 1: StepStart[0] ops=0 (4960ms, 0 samples):
  imbalance: avg=1.2 max=5.0 | dsq: avg=0 max=0 | nr_run: avg=1.0 | fallback: 0/s | keep_last: 38/s | throughput: 79697 iter/s (stimulus-derived)
  per-cgroup:
    cg_a: off-cpu avg=0.3% min=0.3% max=0.3% spread=0.0% | run-delay mean=915µs worst=915µs | iters=209600 migrations=1 | gap=10ms@cpu0
    cg_b: off-cpu avg=9.0% min=9.0% max=9.0% spread=0.0% | run-delay mean=5654µs worst=5654µs | iters=189252 migrations=1 | gap=21ms@cpu0
  >>> StepStart[0]: ops=0 (2 cgroups, 2 workers)
```
