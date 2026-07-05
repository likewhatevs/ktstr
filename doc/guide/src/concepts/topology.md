# Topology

Schedulers make placement decisions across LLC and NUMA boundaries —
where to wake a task, when a migration is worth the cache cost. Each
ktstr test declares the topology those decisions should be tested
against, and the VM it runs in actually has it: the declared NUMA
nodes, cache domains, and SMT siblings are what the guest kernel
sees.

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

## Related

- [Gauntlet](../running-tests/gauntlet.md) — preset topology
  matrices and the constraints that filter them.
- [MemPolicy](mem-policy.md) — NUMA memory placement to pair with
  multi-node topologies.
- [Resource Budget](resource-budget.md) — how the host's topology is
  carved up when tests run concurrently.
