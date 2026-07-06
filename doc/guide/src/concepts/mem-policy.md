# MemPolicy

Testing whether a scheduler keeps tasks near their memory requires a
measurable locality signal — workers whose pages verifiably live on
specific NUMA nodes, so that placement decisions show up as page
counts instead of guesswork. `MemPolicy` creates that signal: it
wraps `set_mempolicy(2)` per worker (applied after fork, before the
work loop), and the [NUMA checks](checking.md#numa-checks) then gate
on where the pages actually landed. Pair it with multi-NUMA
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

## MpolFlags

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

## Usage

`WorkSpec` and `CgroupDef` both take `.mem_policy()` and
`.mpol_flags()`:

```rust,ignore
let def = CgroupDef::named("cg_0")
    .cpuset(CpusetSpec::numa(0))
    .workers(4)
    .mem_policy(MemPolicy::bind([0]));
```

## Cpuset validation

When a cgroup has a cpuset and no remapping flag is set, ktstr
validates at setup time that the policy's nodes are reachable from
that cpuset — `MemPolicy::Bind([1])` on a cgroup confined to node 0
fails before the run starts, not as a mystery `ENOMEM` mid-run.

The check is flag-aware: `STATIC_NODES` swaps it for a
node-exists-on-host check (the nodemask is absolute and deliberately
allowed outside the cpuset), and `RELATIVE_NODES` bypasses it (the
kernel remaps the ordinals internally). Policies without a node set
(`Default`, `Local`) skip validation.

## What gets checked

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

## Example: NUMA-aware locality test

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
land where they were bound.

Node-set policies only mean something on multi-NUMA topologies. The
constraint pair `min_numa_nodes = 2, max_numa_nodes = 2` keeps
gauntlet expansion on two-node presets — single-node presets are
filtered out rather than failing. Both bounds are needed: the
default constraints cap at one NUMA node, and an inverted pair
(min above max) is rejected at validation time. See
[Gauntlet](../running-tests/gauntlet.md) for the preset matrix.
