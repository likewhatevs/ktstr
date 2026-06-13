# Ops and Steps

The ops system is a composable way to express dynamic cgroup topology
changes. It replaces hand-written `fn(&Ctx) -> Result<AssertResult>`
custom-scenario closures that manipulated cgroups directly for most
dynamic scenarios.

## Op

An `Op` is an atomic operation on the cgroup topology. The enum is
`#[non_exhaustive]`, so external pattern matches must end with `..` to
stay compatible across ktstr version bumps that add new variants:

| Op | Description |
|---|---|
| `AddCgroup` | Create a cgroup |
| `AddCgroupDef` | Create a cgroup, set cpuset, and spawn workers from a `CgroupDef` mid-step (the same three-op bundle that `Step::with_defs` runs at step entry, executable from any `ops` vec position) |
| `RemoveCgroup` | Stop workers and remove a cgroup (see [RemoveCgroup / StopCgroup against Backdrop targets](#removecgroup--stopcgroup-against-backdrop-targets) for the permissive-removal contract) |
| `SetCpuset` | Set a cgroup's cpuset via `CpusetSpec` |
| `ClearCpuset` | Remove cpuset constraints |
| `SwapCpusets` | Swap cpusets between two cgroups |
| `Spawn` | Spawn workers with a `SpawnPlacement`: `SpawnPlacement::Cgroup(name)` forks them into a named cgroup; `SpawnPlacement::RunnerCgroup` spawns them in the test runner's own cgroup (typically the guest root, cgid=1). Constructor sugar: `Op::spawn_workers` / `Op::spawn_host`. |
| `StopCgroup` | Stop a cgroup's workers (see [RemoveCgroup / StopCgroup against Backdrop targets](#removecgroup--stopcgroup-against-backdrop-targets) for the permissive-stop contract) |
| `SetAffinity` | Set worker affinity via `AffinityIntent` |
| `MoveAllTasks` | Move all tasks from one cgroup to another |
| `RunPayload` | Spawn a binary-kind [`Payload`](../writing-tests/scheduler-definitions.md#derive-payload) in the background and track its `PayloadHandle` under the step's payload set. Subsequent `WaitPayload` / `KillPayload` address it by `(payload.name, cgroup)`. Scheduler-kind payloads are rejected at apply time. |
| `WaitPayload` | Block until the named payload exits naturally, evaluate its checks, and record metrics to the per-test sidecar. Target lookup is by `(name, cgroup)` composite key; `cgroup: None` resolves to the unique live copy. No timeout — pair with a bounded `HoldSpec` or the payload's own `--runtime` for time-boxed runs. |
| `KillPayload` | SIGKILL the named payload, reap the child, evaluate checks, and record metrics. Same `(name, cgroup)` lookup rules as `WaitPayload`. Mirrors step-teardown drain for an explicitly-targeted payload. |
| `FreezeCgroup` | Freeze every task in the named cgroup via `cgroup.freeze` (kernel-side asynchronous freeze; not a SIGSTOP). Idempotent for already-frozen cgroups. Pair with `UnfreezeCgroup` to release; teardown auto-unfreezes. See [Snapshots](../writing-tests/snapshots.md) for the observer-cgroup deadlock warning. |
| `UnfreezeCgroup` | Unfreeze every task in the named cgroup via `cgroup.freeze`. Inverse of `FreezeCgroup`. Idempotent. |
| `CaptureSnapshot` | Capture a host-side diagnostic snapshot under `name` via the freeze coordinator: pauses every vCPU, reads BPF map state, vCPU registers, and per-CPU counters into a `FailureDumpReport`, then resumes. The report is keyed by `name` on the active `SnapshotBridge`. With no bridge installed it fails loudly (a guest run issues a host-coordinator snapshot request; a host-only context errors) per the no-silent-drops policy. See [Snapshots](../writing-tests/snapshots.md). |
| `WatchSnapshot` | Capture a snapshot whenever the guest writes to the named kernel symbol; one fire = one capture tagged with the symbol path. Symbol resolution at op execution time looks the name up by **verbatim vmlinux ELF symbol-table match** — the requested name must appear in the guest kernel's static symbol table exactly as written (no path expansion, no BTF descent). Maximum 3 watch ops per scenario (the KVM hardware-watchpoint plumbing exposes 4 debug slots; slot 0 is reserved for the error-class `exit_kind` trigger, leaving 3 user slots). See [Watch Snapshots](../writing-tests/watch-snapshots.md). |
| `ReadKernelHot` / `ReadKernelCold` | Read a kernel-memory location (symbol, KVA, per-CPU field, or task field) via the freeze coordinator. `Hot` runs live against the running vCPU; `Cold` requires a freeze rendezvous. See [`KernelTarget`](#kerneltarget--kernelvalue) for the target enum. |
| `WriteKernelHot` / `WriteKernelCold` | Write a kernel-memory location. Same target shape as the read ops. Cold-path writes are auto-merged when adjacent ops target the same address to amortize freeze cost. |
| `AttachScheduler` / `DetachScheduler` / `RestartScheduler` / `ReplaceScheduler` | Manage the live scheduler in the guest mid-scenario. `ReplaceScheduler` swaps to a different staged scheduler binary (declared via the `#[ktstr_test(staged_schedulers = [...])]` attribute). |
| `PinBpfMap` | Open a BPF map fd by name and hold a refcount for the scenario lifetime (keeps a same-binary-swap window's `.bss` alive across `ReplaceScheduler`) |
| `CaptureCgroupProcs` | Read a cgroup's `cgroup.procs` (thread-group-leader PIDs) and store them on the active `SnapshotBridge` under a tag |

Op constructors accept string literals directly (no `.into()` needed):

```rust,ignore
Op::add_cgroup("cg_0")
Op::add_cgroup_def(CgroupDef::named("cg_1").workers(4))
Op::set_cpuset("cg_0", CpusetSpec::disjoint(0, 2))
Op::stop_cgroup("cg_0")
Op::spawn_workers("cg_0", WorkSpec::default().workers(4))
Op::set_affinity("cg_0", AffinityIntent::random_subset([0, 1, 2, 3], 2))
Op::spawn_host(WorkSpec::default().workers(4))
Op::freeze_cgroup("cg_0")
Op::unfreeze_cgroup("cg_0")
Op::capture_snapshot("after_spawn")
Op::watch_snapshot("jiffies_64")
```

`Op::spawn_host` (sugar for `SpawnPlacement::RunnerCgroup`) creates
workers in the test runner's own cgroup — typically the guest root
(cgid=1) — not in a managed cgroup. Use this to simulate host-level
CPU contention alongside managed cgroups.

### RemoveCgroup / StopCgroup against Backdrop targets

`Op::RemoveCgroup` and `Op::StopCgroup` are permitted against
`Backdrop`-owned cgroups from inside a Step's ops. Removing a
Backdrop cgroup mid-scenario drops the
framework tracking entry so a later `Op::AddCgroup` with the same
name re-creates the cgroup cleanly. Stopping a Backdrop cgroup's
workers mid-scenario leaves the cgroup hierarchy alive but kills
the workers — subsequent `Op::WaitPayload` / `Op::KillPayload` ops
that expect those workers will fail to find them.

The framework intentionally trades the early-bail for permissive
removal. A typo'd cgroup name silently succeeds at the
`Op::RemoveCgroup` site (rmdir on a non-existent path is a no-op)
and surfaces later as the kernel's `No such file or directory`
error on the next op that references the name (`Op::SetCpuset`,
`Op::MoveAllTasks`, etc.). Debug with care: if a later Step fails
with a missing-cgroup error, grep the test for `Op::remove_cgroup`
calls naming a similar identifier first.

`Op::MoveAllTasks` retains a separate `Backdrop`→step-local
rejection that protects against stranding persistent workers in a
cgroup that gets torn down at step boundary — see the variant
docstring at `Op::MoveAllTasks` for the asymmetric-ownership table.

## OpKind

`OpKind` is a payload-free discriminant enum generated from `Op` via
`#[strum_discriminants]`. It carries the same variant set as `Op`
(`AddCgroup`, `AddCgroupDef`, `RemoveCgroup`, `SetCpuset`,
`ClearCpuset`, `SwapCpusets`, `Spawn`, `StopCgroup`, `SetAffinity`,
`MoveAllTasks`, `RunPayload`, `WaitPayload`, `KillPayload`,
`FreezeCgroup`, `UnfreezeCgroup`, `CaptureSnapshot`, `WatchSnapshot`,
`WriteKernelHot`, `WriteKernelCold`, `ReadKernelHot`, `ReadKernelCold`,
`AttachScheduler`, `DetachScheduler`, `RestartScheduler`,
`ReplaceScheduler`, `PinBpfMap`, `CaptureCgroupProcs`) with none of the
inner fields, so it is cheap to copy and use as a map key. Framework code uses `OpKind` when it
only cares WHICH operation ran (per-op statistics, stimulus-event
tagging, verifier/monitor bookkeeping) without the payload. Test
authors rarely spell `OpKind` directly — the `strum::EnumIter`
derive also lets tooling enumerate every `OpKind` variant for
coverage checks.

`OpKind` is NOT `#[non_exhaustive]` — `strum_discriminants` does not
propagate `Op`'s `#[non_exhaustive]` to the generated discriminant
enum, so exhaustive matches over `OpKind` compile. Adding a new `Op`
variant requires updating any local exhaustive `OpKind` match (e.g.
`OpKind::bit_index`).

## CpusetSpec

`CpusetSpec` computes a cpuset from the topology at runtime. The enum
is `#[non_exhaustive]`, so external callers should construct via the
associated constructor functions (see the list below this snippet)
rather than naming variant literals — a future field addition (e.g.
a stride on `Range`) can land behind a defaulted parameter without
breaking call sites. Pattern matches over `CpusetSpec` must also end
with `..`:

```rust,ignore
pub enum CpusetSpec {
    Llc(usize),                          // All CPUs in an LLC
    Numa(usize),                         // All CPUs in a NUMA node
    Range { start_frac: f64, end_frac: f64 }, // Fraction of usable CPUs
    Disjoint { index: usize, of: usize },     // Equal disjoint partitions
    Overlap { index: usize, of: usize, frac: f64 }, // Overlapping partitions
    Exact(BTreeSet<usize>),              // Exact CPU set
}
```

Convenience constructors accept parameters directly:
`CpusetSpec::disjoint(0, 2)`, `CpusetSpec::range(0.0, 0.5)`,
`CpusetSpec::exact([0, 1, 2])`, `CpusetSpec::llc(0)`,
`CpusetSpec::numa(0)`, `CpusetSpec::overlap(0, 2, 0.5)`.

All fractional specs operate on
[`usable_cpus()`](topology.md#topology-queries).

## CgroupDef

`CgroupDef` bundles three ops that always go together: create cgroup,
set cpuset, spawn workers. It is the primary way to define cgroups in
ops-based scenarios.

```rust,ignore
let def = CgroupDef::named("cg_0")
    .cpuset(CpusetSpec::disjoint(0, 2))
    .workers(4)
    .work_type(WorkType::SpinWait);
```

### Builder methods

- `.cpuset(CpusetSpec)` -- set the cpuset (CPU set the cgroup is
  pinned to).
- `.cpuset_mems(BTreeSet<usize>)` -- explicit `cpuset.mems`
  override (default derives from the resolved cpuset's NUMA nodes).
- `.workers(n)` -- set worker count.
- `.workers_pct(p)` -- set worker count as a fraction of the cgroup's
  resolved cpuset; framework computes `ceil(cpuset_cpus * p)` at
  apply-setup time. Fractions above `1.0` are accepted as deliberate
  oversubscription. Conflicts with `.workers(n)` — pick one. See
  `Ctx::cpuset_cpus` for hand-computing the same denominator outside
  the builder.
- `.work_type(WorkType)` -- set work type (default: `SpinWait`).
- `.sched_policy(SchedPolicy)` -- set Linux scheduling policy
  (default: `Normal`). See [WorkSpec Types](work-types.md#scheduling-policies).
- `.work(WorkSpec)` -- add a work group (multiple calls for concurrent groups).
- `.workload(&'static Payload)` -- attach a binary workload payload
  to run alongside the worker group; the framework launches it as a
  child process inside the cgroup. **Panics** when called with a
  scheduler-kind `Payload` (`PayloadKind::Scheduler(_)`); the
  scheduler slot is `#[ktstr_test(scheduler = ...)]` at the test
  level, not the cgroup-level `workload` slot. Step-level
  `Op::RunPayload` rejects scheduler-kind payloads with an
  `anyhow::Error` instead of panicking; the build-time `workload`
  call panics because there is no scenario-level recovery path.
- `.affinity(AffinityIntent)` -- set per-worker affinity (default: `Inherit`).
- `.mem_policy(MemPolicy)` -- set NUMA memory placement policy
  (default: `Default`). See [MemPolicy](mem-policy.md).
- `.mpol_flags(MpolFlags)` -- set mode flags for `set_mempolicy(2)`
  (default: `NONE`). See [MemPolicy](mem-policy.md#mpolflags).
- `.nice(n)` -- cgroup-level default per-worker nice value, merged
  into every WorkSpec whose own `nice` is unset. See
  [Tutorial: Step 11](../tutorial.md#step-11-name-and-prioritize-workers).
- `.comm(name)` -- cgroup-level default per-worker `task->comm` via
  `prctl(PR_SET_NAME)`. Merged into every WorkSpec whose own `comm`
  is unset.
- `.pcomm(name)` -- thread-group-leader `task->comm` for the
  fork-then-thread spawn path (workers run as threads under one
  forked leader). Stamps every existing WorkSpec in-place; not
  order-independent with `.work(...)`.
- `.uid(uid)` / `.gid(gid)` -- cgroup-level default per-worker
  effective UID / GID via `setresuid` / `setresgid`. Merged into
  every WorkSpec whose own `uid` / `gid` is unset.
- `.numa_node(node)` -- cgroup-level default NUMA-node affinity for
  every WorkSpec. Merged at apply-setup time.
- `.swappable(bool)` -- opt into gauntlet work type override.

#### Example: cpuset-scaled worker count

Tests that span topologies need worker counts that scale with the
cpuset. Hand-computing the count from `ctx.topo` works but couples
the test to the topology shape:

```rust,ignore
// Before: hand-computed via Ctx::cpuset_cpus
// (couples the test to a manual resolution step).
let n_workers =
    (ctx.cpuset_cpus(&CpusetSpec::Llc(0)) as f64 * 0.9).ceil() as usize;
let def = CgroupDef::named("cg_hot")
    .cpuset(CpusetSpec::Llc(0))
    .workers(n_workers);
```

`workers_pct(p)` expresses the same intent declaratively; the
framework resolves the count from the cgroup's cpuset at apply-setup
time using `ceil(cpuset_cpus * p)` and the test stays independent
of the underlying topology:

```rust,ignore
// After: framework computes the count from the resolved cpuset
let def = CgroupDef::named("cg_hot")
    .cpuset(CpusetSpec::Llc(0))
    .workers_pct(0.9);  // ceil(N_CPUs * 0.9) workers
```

Both `workers(n)` and `workers_pct(p)` cannot coexist on the same
`WorkSpec`; the apply-setup path rejects the dual-set with an
actionable diagnostic.

#### Cgroup controllers

The cgroup-v2 cpu / memory / io / pids controllers are exposed as
typed setters (default: unconstrained):

- `.cpu_quota_pct(pct)` / `.cpu_quota(quota, period)` /
  `.cpu_unlimited()` -- write `cpu.max` (`pct` is shorthand: `100`
  = one full CPU). `cpu_unlimited` resets to the kernel default.
- `.cpu_weight(weight)` -- write `cpu.weight` (`1..=10000`,
  default `100`).
- `.memory_max(bytes)` / `.memory_high(bytes)` /
  `.memory_low(bytes)` / `.memory_unlimited()` -- write
  `memory.max` / `memory.high` / `memory.low`. `memory_unlimited`
  resets `memory.max` to `max`.
- `.memory_swap_max(bytes)` / `.memory_swap_unlimited()` -- write
  `memory.swap.max`.
- `.io_weight(weight)` -- write `io.weight` (`1..=10000`,
  default `100`).
- `.pids_max(n)` / `.pids_unlimited()` -- write `pids.max`.

### MemPolicy-cpuset validation

When a cgroup has a cpuset, ktstr validates that the `MemPolicy`'s
node set is covered by the NUMA nodes reachable from that cpuset. A
`MemPolicy::Bind([1])` on a cgroup whose cpuset covers only NUMA
node 0 fails at setup time. Policies without a node set (`Default`,
`Local`) skip validation.

### WorkSpec type overrides and swappable

`CgroupDef` has a `swappable` flag (default: `false`). The gauntlet
work-type override (`Ctx.work_type_override`) is applied only to a
`CgroupDef` marked `.swappable(true)`. When active, it replaces the
def's work type with the override — any base type, not just
`SpinWait` — and is skipped when the override is a grouped work type
whose group size does not divide the resolved worker count
(`resolve_work_type` in `src/workload/config/mod.rs`). There is no
`SpinWait`-specific override path.

Work-type overrides apply only to `CgroupDef`-based setup. `Op::Spawn`
(and its sugar `Op::spawn_workers` / `Op::spawn_host`) always uses the
work type as given. Use `CgroupDef` with `.swappable(true)` when the
work type should participate in gauntlet overrides.

## Step

A `Step` is a sequence of ops with a hold period:

```rust,ignore
pub struct Step {
    pub setup: Setup,   // CgroupDefs to create after ops
    pub ops: Vec<Op>,   // Operations to apply
    pub hold: HoldSpec, // How long to wait after
}
```

`Setup` is either `Defs(Vec<CgroupDef>)` or
`Factory(fn(&Ctx) -> Vec<CgroupDef>)`. Construct via the variant
syntax or the named const-fn for the `Factory` arm:

- `Setup::Defs(defs)` -- variant constructor for the static-list case
  (`Setup::Defs(vec![CgroupDef::named("cg")])`).
- `Setup::with_factory(f)` -- named const-fn constructor for the
  `Factory` arm so the def list can depend on the resolved topology.

`Setup::default()` returns `Setup::Defs(Vec::new())` (the ops-only
path). `Vec<CgroupDef>` also implements `Into<Setup>` for the inline
`setup: vec![...].into()` form when a chain-builder context already
produces a `Vec`.

### Constructors

**`Step::new(ops, hold)`** -- creates a step with ops only (no
CgroupDef setup). Use when the step only applies dynamic operations
to an existing topology.

**`Step::hold(hold)`** -- shorthand for a hold-only step with no
setup and no ops. The canonical shape for phase A in an A/B-style
two-step scenario (`Step::hold(HoldSpec::frac(0.3))` then
`Step::with_op(Op::replace_scheduler(&ALT), HoldSpec::frac(0.7))`).

**`Step::with_op(op, hold)`** -- shorthand for a step that runs a
single op then holds. Pairs naturally with `Step::hold` for two-step
A/B scenarios.

**`Step::with_defs(defs, hold)`** -- creates a step with CgroupDef
setup and a hold period. The primary constructor for steps that
create cgroups with workers.

**`Step::with_payload(payload, hold)`** -- creates a step that runs
a single binary-kind `Payload` for `hold`. Sets up a single
`Op::RunPayload` and an empty `Setup`; the payload runs in the
background for `hold` and is drained at step teardown (no
`Op::WaitPayload` is emitted, so nothing blocks on the child exiting).
Use for inline payload-driven steps without the `CgroupDef` ceremony.

**`Step::set_ops(self, ops)`** -- REPLACES the ops on a step
(builder method). Chain after `with_defs` to add dynamic operations
to a step that also creates cgroups.

**`Step::set_hold(self, hold)`** -- REPLACES the hold on a step
(builder method). Use the `set_X` prefix family to mutate the step
in place; `with_X` is reserved for alternative constructors.

> **Naming convention:** `Step::set_ops` REPLACES; the sibling
> `Backdrop::extend_ops` APPENDS several, `Backdrop::push_op`
> APPENDS one. The verb prefix encodes the semantics: `set_X`
> replaces a field, `push_X` appends one element, `extend_X`
> appends an iterator. A `Step::new(ops).set_ops(more)` chain
> produces a step whose ops vec is exactly `more` (the original
> `ops` is dropped); a
> `Backdrop::new().extend_ops(ops_a).extend_ops(ops_b)` chain
> produces a backdrop whose ops vec is `ops_a + ops_b`. If you
> need to extend a step's ops vec, build the combined `Vec<Op>`
> at the call site and pass it to `set_ops`, or compose at the
> `Backdrop` layer instead.

## HoldSpec

How long to hold after a step completes:

| Variant | Description |
|---|---|
| `Frac(f64)` | Fraction of the total scenario duration |
| `Fixed(Duration)` | Fixed time |
| `Loop { interval }` | Repeat ops at interval until time runs out |

Construct via the const-fn sugar: `HoldSpec::frac(0.5)`,
`HoldSpec::fixed(Duration::from_secs(5))`, `HoldSpec::loop_at(
Duration::from_secs(2))`. `HoldSpec::FULL` is a constant for
`Frac(1.0)` (hold for the full scenario duration).

### Loop ops to drive periodic state changes

Use `HoldSpec::loop_at(interval)` when you want a Step's ops to
fire repeatedly until the scenario time budget is exhausted, not
just once. This is the natural shape for "every N seconds, do X"
patterns that drive periodic churn without writing a host-thread
loop. Example: snapshot host state at Step entry and then every
2 seconds thereafter, capturing periodic behavior across the
scenario:

```rust,ignore
Step::new(
    vec![Op::capture_snapshot("periodic")],
    HoldSpec::loop_at(Duration::from_secs(2)),
)
```

The Step's setup (Step::with_defs) runs once at Step entry; only
the ops repeat. A `Loop` Step that pairs setup with periodic ops
runs the setup ONCE, then applies the ops vec immediately and
again every interval until the scenario completes. The Backdrop
persists across the loop (and across the entire scenario); prior
Steps' step-local cgroups/handles/payloads were torn down at
their own step boundaries before this Step started, so only
Backdrop-owned state and this Step's own setup are live during
each loop iteration.

## execute_defs

`execute_defs(ctx, defs)` is a convenience wrapper for the common
pattern of creating cgroups and running them for the full duration:

```rust,ignore
execute_defs(ctx, vec![
    CgroupDef::named("cg_0").workers(4),
    CgroupDef::named("cg_1").workers(4),
])
```

Equivalent to `execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])`.

## execute_steps

`execute_steps(ctx, steps)` runs a step sequence:

1. For each step: apply ops, then apply setup (create cgroups from
   `CgroupDef`s), hold for the specified duration. Ops run first so
   parent cgroups can be created before children are spawned.
   `Loop` steps reverse this: setup runs once before the loop, then
   ops repeat at the specified interval.
2. Check scheduler liveness between steps.
3. After all steps: collect worker reports and run checks.
4. Writes stimulus events over the virtio-console port-1 bulk channel for timeline analysis.

## execute_steps_with

`execute_steps_with(ctx, steps, assertions)` is the same as
`execute_steps` but accepts an explicit
[`Assert`](checking.md#assert-struct) for worker checks.
`execute_steps` is a convenience wrapper that passes `None`.

```rust,ignore
use ktstr::prelude::*;

fn my_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let assertions = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_gap_ms(3000);

    let steps = vec![/* ... */];
    execute_steps_with(ctx, steps, Some(&assertions))
}
```

When `assertions` is `Some`, the provided `Assert` overrides `ctx.assert`
for worker checks. When `None`, uses `ctx.assert` (the merged
three-layer config: `default_checks` -> scheduler -> per-test).
