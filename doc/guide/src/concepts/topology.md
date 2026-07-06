# Topology

Schedulers make placement decisions across LLC and NUMA boundaries —
where to wake a task, when a migration is worth the cache cost. Each
ktstr test declares the topology those decisions should be tested
against, and the VM it runs in actually has it: the declared NUMA
nodes, cache domains, and SMT siblings are what the guest kernel
sees.

<div class="kt-figure"><svg width="680" height="250" viewBox="0 0 680 250" role="img" aria-label="Topology 1n2l4c2t: one NUMA node containing two LLCs, each with four cores of two threads">
  <rect x="8" y="8" width="664" height="234" rx="12" fill="none" stroke="var(--kt-accent)" stroke-width="1.6"/>
  <text x="24" y="32" font-size="13" fill="var(--kt-accent)" font-weight="700">NUMA node 0</text>
  <g font-size="10.5" fill="var(--fg)">
    <rect x="24" y="44" width="308" height="184" rx="10" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/>
    <text x="38" y="66" font-size="12" font-weight="700" opacity=".75">LLC 0</text>
    <rect x="38" y="78" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="50" y="96">core 0</text>
    <rect x="50" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="57" y="121" font-size="9.5">cpu 0</text>
    <rect x="104" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="110" y="121" font-size="9.5">cpu 1</text>
    <rect x="186" y="78" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="198" y="96">core 1</text>
    <rect x="198" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="205" y="121" font-size="9.5">cpu 2</text>
    <rect x="252" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="258" y="121" font-size="9.5">cpu 3</text>
    <rect x="38" y="152" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="50" y="170">core 2</text>
    <rect x="50" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="57" y="195" font-size="9.5">cpu 4</text>
    <rect x="104" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="110" y="195" font-size="9.5">cpu 5</text>
    <rect x="186" y="152" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="198" y="170">core 3</text>
    <rect x="198" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="205" y="195" font-size="9.5">cpu 6</text>
    <rect x="252" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="258" y="195" font-size="9.5">cpu 7</text>
  </g>
  <g font-size="10.5" fill="var(--fg)" opacity=".5">
    <rect x="348" y="44" width="308" height="184" rx="10" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/>
    <text x="362" y="66" font-size="12" font-weight="700">LLC 1</text>
    <rect x="362" y="78" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="374" y="96">core 4</text>
    <rect x="374" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="381" y="121" font-size="9.5">cpu 8</text>
    <rect x="428" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="434" y="121" font-size="9.5">cpu 9</text>
    <rect x="510" y="78" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="522" y="96">core 5</text>
    <rect x="522" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="528" y="121" font-size="9.5">cpu 10</text>
    <rect x="576" y="104" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="582" y="121" font-size="9.5">cpu 11</text>
    <rect x="362" y="152" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="374" y="170">core 6</text>
    <rect x="374" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="380" y="195" font-size="9.5">cpu 12</text>
    <rect x="428" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="434" y="195" font-size="9.5">cpu 13</text>
    <rect x="510" y="152" width="130" height="62" rx="7" fill="var(--bg)" stroke="var(--kt-rule)"/><text x="522" y="170">core 7</text>
    <rect x="522" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="528" y="195" font-size="9.5">cpu 14</text>
    <rect x="576" y="178" width="46" height="26" rx="4" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width=".7"/><text x="582" y="195" font-size="9.5">cpu 15</text>
  </g>
</svg></div>


## The notation

Topologies render as `{n}n{l}l{c}c{t}t` — NUMA nodes, LLCs, cores per
LLC, threads per core. One quirk to internalize:

> [!NOTE]
> The `l` count is the **total** LLC count across the VM, not
> per-node. `2n4l4c2t` is 2 NUMA nodes and 4 LLCs total (2 per
> node), 4 cores per LLC, 2 threads per core = 4 × 4 × 2 = 32 vCPUs.

Containment is strict — threads in a core, cores in an LLC, LLCs in
a NUMA node — and guest CPUs are numbered sequentially through it.
`1n2l4c2t` (16 vCPUs) lays out as:

```text
node 0
├─ LLC 0                      ├─ LLC 1
│  ├─ core 0: cpu 0, 1        │  ├─ core 4: cpu 8,  9
│  ├─ core 1: cpu 2, 3        │  ├─ core 5: cpu 10, 11
│  ├─ core 2: cpu 4, 5        │  ├─ core 6: cpu 12, 13
│  └─ core 3: cpu 6, 7        │  └─ core 7: cpu 14, 15
```

Most tests use one NUMA node; multi-NUMA topologies matter when the
scheduler weighs memory locality. The
[gauntlet](../running-tests/gauntlet.md) sweeps a test across a
whole preset matrix of these shapes.

## What a test declares — and what it gets

The `#[ktstr_test]` attributes `numa_nodes`, `llcs`, `cores`,
`threads` declare the shape (see the
[macro reference](../writing-tests/ktstr-test-macro.md) for defaults
and inheritance). The run output echoes the topology the guest
booted with — the `[topo=...]` tag in failure headers and the
timeline header:

<!-- captured: cargo ktstr test --kernel 7.0 (throughput_gate demo test) | ktstr 0.23.0 | kernel 7.0.14 -->
```text
ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:
...
topology: 1n1l2c1t (2 cpus)  scheduler: my_sched  scenario: throughput_gate  duration: 15.0s
```

To see a host's physical layout in the same vocabulary, `ktstr topo`:

<!-- captured: ktstr topo | ktstr 0.23.0 | host, no VM -->
```text
CPUs:       64
LLCs:       4
NUMA nodes: 1
  LLC 0 (node 0): [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
  LLC 1 (node 0): [8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47]
  LLC 2 (node 0): [16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55]
  LLC 3 (node 0): [24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63]
```

(Host CPU numbering differs from the guest's sequential scheme —
here SMT siblings sit 32 apart — which is exactly why tests declare
a topology instead of inheriting the host's.)

## Cpusets from topology

Scenarios don't hard-code CPU lists; a
[`CpusetSpec`](ops.md#cpusetspec) resolves against the test's
topology at runtime. On `1n2l4c2t`, `CpusetSpec::Llc(0)` resolves to
CPUs 0-7, so the cgroup's `cpuset.cpus` is written as `0-7`; `Llc`
and `Numa` cover their full domain, while the fractional and
partition variants (`Range`, `Disjoint`, `Overlap`) slice the
usable-CPU pool.

## Querying topology from a scenario {#topology-queries}

`Ctx.topo` is a `TestTopology`. The queries scenario authors
actually use:

- `total_cpus()`, `num_llcs()`, `num_numa_nodes()` — sizes, e.g. for
  skip guards (`if ctx.topo.num_llcs() < 2 { return
  Ok(AssertResult::skip(...)) }`).
- `usable_cpus()` / `usable_cpuset()` — CPUs available for workload
  placement. On topologies with more than 2 CPUs the last CPU is
  reserved for the root cgroup (on 8 CPUs: usable = 0-6). Built-in
  scenarios and fractional `CpusetSpec`s use this pool
  automatically.
- `llc_aligned_cpuset(idx)` / `numa_aligned_cpuset(node)` — the CPU
  set of one LLC or one node's LLCs.
- `numa_nodes_for_cpuset(cpus)` — which nodes a CPU set touches;
  this derives the expected-node set for
  [NUMA checks](checking.md#numa-checks).
- `numa_distance(from, to)` — kernel conventions: 10 local, higher
  is farther, 255 unreachable/unknown. VM topologies without
  explicit distances report 10 local / 20 remote.
- `node_meminfo(node)` / `is_memory_only(node)` — per-node memory
  and CXL-style memory-only node detection.

`Ctx::cpuset_cpus(&spec)` returns the CPU count a spec resolves to —
useful for sizing worker counts by hand. Its denominator is the
topology-level cpuset, not any cgroup's currently-effective one; for
cgroup-aware sizing prefer
[`CgroupDef::workers_pct`](ops.md#cpuset-scaled-worker-counts),
which resolves against the cgroup's own cpuset at apply time.

The full method catalog (construction, `LlcInfo`, CPU-list parsing)
is in the
[`TestTopology` rustdoc](https://ktstr.dev/rustdoc/ktstr/topology/struct.TestTopology.html).

## Memory policy

NUMA is the axis topology and memory placement share: a multi-node
topology gives the scheduler somewhere to get locality wrong, and
`MemPolicy` gives the test a way to measure it. It pins where each
worker's pages live — pages that verifiably sit on specific nodes, so
placement decisions show up as page counts instead of guesswork.
`MemPolicy` wraps `set_mempolicy(2)` per worker (applied after fork,
before the work loop), and the [NUMA checks](checking.md#numa-checks)
gate on where the pages actually landed. Pair it with multi-NUMA
[gauntlet presets](../running-tests/gauntlet.md) to sweep the same
test across node counts.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Bind</strong><p>Force pages onto one node set when locality must be strict.</p></div>
<div class="kt-doc-card"><strong>Prefer</strong><p>Bias allocation toward a node or set while allowing fallback.</p></div>
<div class="kt-doc-card"><strong>Interleave</strong><p>Spread pages across nodes for bandwidth or cross-node fairness tests.</p></div>
</div>

```rust,ignore
pub enum MemPolicy {
    Default,
    Bind(BTreeSet<usize>),
    Preferred(usize),
    Interleave(BTreeSet<usize>),
    Local,
    PreferredMany(BTreeSet<usize>),
    WeightedInterleave(BTreeSet<usize>),
}
```

- **`Default`** — inherit the parent's policy; no syscall made.
- **`Bind(nodes)`** (`MemPolicy::bind([0, 1])`) — allocate only from
  these nodes (`MPOL_BIND`); allocation fails with `ENOMEM` when
  they are exhausted.
- **`Preferred(node)`** (`::preferred(0)`) — prefer one node, fall
  back silently when it is full (`MPOL_PREFERRED`).
- **`Interleave(nodes)`** (`::interleave([0, 1])`) — round-robin
  allocations across the nodes (`MPOL_INTERLEAVE`).
- **`Local`** — nearest node to the allocating CPU (`MPOL_LOCAL`).
- **`PreferredMany(nodes)`** (`::preferred_many([0, 1])`) — prefer
  any of the nodes, fall back when all are full
  (`MPOL_PREFERRED_MANY`, kernel 5.15+).
- **`WeightedInterleave(nodes)`** (`::weighted_interleave([0, 1])`)
  — interleave proportional to the per-node weights in
  `/sys/kernel/mm/mempolicy/weighted_interleave/`
  (`MPOL_WEIGHTED_INTERLEAVE`, kernel 6.9+).

Node-set constructors accept any `IntoIterator<Item = usize>`.
`MemPolicy::node_set()` returns the referenced nodes (empty for
`Default` / `Local`).

### MpolFlags

Optional mode flags OR'd into the `set_mempolicy` mode:

| Flag | Meaning |
|---|---|
| `NONE` | No flags |
| `STATIC_NODES` | Nodemask is absolute — not remapped when the task's cpuset changes |
| `RELATIVE_NODES` | Nodemask is relative to the task's current cpuset |
| `NUMA_BALANCING` | Enable NUMA-balancing optimization for this policy |

Flags combine with `|`. `STATIC_NODES | RELATIVE_NODES` is rejected
at setup time (the kernel would return `EINVAL`), as is any unknown
bit. The kernel accepts `NUMA_BALANCING` only alongside `MPOL_BIND`
or `MPOL_PREFERRED_MANY` — ktstr does not pre-validate that pairing,
so other combinations surface as `EINVAL` from the worker's
`set_mempolicy` call.

### Usage

`WorkSpec` and `CgroupDef` both take `.mem_policy()` and
`.mpol_flags()`:

```rust,ignore
let def = CgroupDef::named("cg_0")
    .cpuset(CpusetSpec::numa(0))
    .workers(4)
    .mem_policy(MemPolicy::bind([0]));
```

### Cpuset validation

When a cgroup has a cpuset and no remapping flag is set, ktstr
validates at setup time that the policy's nodes are reachable from
that cpuset — `MemPolicy::Bind([1])` on a cgroup confined to node 0
fails before the run starts, not as a mystery `ENOMEM` mid-run.

The check is flag-aware: `STATIC_NODES` swaps it for a
node-exists-on-host check (the nodemask is absolute and deliberately
allowed outside the cpuset), and `RELATIVE_NODES` bypasses it (the
kernel remaps the ordinals internally). Policies without a node set
(`Default`, `Local`) skip validation.

### What gets checked

Locality results feed the [NUMA checking
thresholds](checking.md#numa-checks) — `min_page_locality`,
`max_cross_node_migration_ratio`, `max_slow_tier_ratio`. The
expected node set is derived from the cgroup's *cpuset* at
evaluation time, not from the worker's `MemPolicy`; in the common
case where memory is bound to the same nodes the cpuset pins, the
two coincide. A locality violation renders with the observed
fraction, the threshold, and the page counts (format from the
assertion source):

```text
page locality <observed> (<pct>%) below threshold <min> (<pct>%) (<local>/<total> pages local)
```

### Example: NUMA-aware locality test

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(
    numa_nodes = 2, llcs = 4, cores = 4, threads = 1,
    min_numa_nodes = 2, max_numa_nodes = 2,
    min_page_locality = 0.8,
)]
fn numa_locality(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("node0")
            .cpuset(CpusetSpec::numa(0))
            .workers(4)
            .mem_policy(MemPolicy::bind([0])),
        CgroupDef::named("node1")
            .cpuset(CpusetSpec::numa(1))
            .workers(4)
            .mem_policy(MemPolicy::bind([1])),
    ])
}
```

Each cgroup's workers are pinned to one NUMA node's CPUs via
`CpusetSpec::numa()` and their allocations bound to the same node
via `MemPolicy::bind()`; the test fails if less than 80% of pages
land where they were bound. The constraint pair
`min_numa_nodes = 2, max_numa_nodes = 2` keeps gauntlet expansion on
two-node presets — single-node presets are filtered out rather than
failing. Both bounds are needed: the default constraints cap at one
NUMA node, and an inverted pair (min above max) is rejected at
validation time.

## Related

- [Gauntlet](../running-tests/gauntlet.md) — preset topology
  matrices and the constraints that filter them.
- [Resource Budget](resource-budget.md) — how the host's topology is
  carved up when tests run concurrently.
