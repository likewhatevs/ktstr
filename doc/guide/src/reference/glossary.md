# Glossary

The words this guide leans on, in one place. Kernel terms carry their
kernel meaning; ktstr terms are defined by ktstr's own behavior.

## Kernel and sched_ext

| Term | Meaning |
|---|---|
| **sched_ext** | The kernel's BPF-extensible scheduler class: process schedulers written as BPF programs, attached via struct_ops. |
| **EEVDF** | The kernel's default scheduler. Tests without a `scheduler` attribute run under it — the baseline ktstr compares against. |
| **DSQ** | Dispatch queue — sched_ext's task queues. Schedulers move tasks between DSQs; ktstr reads their depths from guest memory. |
| **LLC** | Last-level cache. The cache domain schedulers optimize placement around, and one axis of every ktstr topology. |
| **stall** | The kernel scx watchdog's word, and reserved for it here: a runnable task the scheduler failed to run within the watchdog timeout fires `SCX_EXIT_ERROR_STALL` and evicts the scheduler. |
| **verifier** | The kernel's BPF verifier. The [verifier sweep](../running-tests/verifier.md) runs it — the real one, in the target kernel — against every declared scheduler. |
| **BTF** | The kernel's type metadata. ktstr resolves struct offsets through it to read guest memory, and BPF skeletons load against it. |
| **kfunc** | A kernel function exposed to BPF programs. Schedulers call `scx_bpf_*` kfuncs; kernels and headers must agree on their signatures. |

## Declaring tests

| Term | Meaning |
|---|---|
| **topology** | The machine shape a test declares — NUMA nodes, LLCs, cores, threads — rendered `1n2l4c2t` and realized in the guest. See [Topology](../concepts/topology.md). |
| **scenario** | What the test does inside the VM: cgroups, workers, operations, expressed as data. |
| **Step / Op / hold** | A scenario is Steps; each Step applies Ops (create, resize, destroy, write), builds its setup, then holds — the measured window. See [Ops, Steps, and Backdrop](../concepts/ops.md). |
| **phase** | The metrics bucket for one Step (plus a pre-run baseline). Per-phase metrics make before/after comparisons inside one run possible. |
| **Backdrop** | Scenario furniture that persists across Steps instead of being torn down with the Step that made it. |
| **work type** | What a worker burns its time on — spin, futex ping-pong, schbench, taobench — 45 variants in all. See [Work Types](../concepts/work-types.md). |
| **payload** | An external binary (fio, stress-ng, your tool) run as a workload, with metrics extracted from its output. See [Payloads](../writing-tests/payloads.md). |

## Checking results

| Term | Meaning |
|---|---|
| **verdict** | The four-state outcome lattice — Fail > Inconclusive > Pass > Skip; the most severe wins. See [Checking](../concepts/checking.md#verdicts-and-outcomes). |
| **no progress** | A worker that completed zero work units. Whether the scheduler denied it CPU or it wedged itself is more than the measurement claims. |
| **stuck** | A worker whose gap between progress checkpoints exceeded the threshold — ktstr's own liveness word (the kernel's is *stall*, above). |
| **unfair** | Off-CPU share spread across a cgroup's workers beyond the threshold. |
| **claim** | One labeled assertion inside a `Verdict` — the accumulator custom scenarios use. `claim_better` compares candidate vs baseline with registry polarity. |
| **projection** | Turning a per-sample value — an scx_stats field, a BPF global or map entry, a host per-CPU reading — into a typed `SeriesField` column for temporal patterns and claims. See [Projections](../writing-tests/temporal-assertions.md). |
| **sidecar** | The per-test JSON stats file written under `target/ktstr/{kernel}-{commit}/` — the raw material for `stats` and `perf-delta`. |

## Running at scale

| Term | Meaning |
|---|---|
| **gauntlet** | One test declaration fanned out across topology presets (and kernels). Variants are named `gauntlet/{test}/{preset}` and skipped unless asked for. |
| **preset** | A named topology in the gauntlet catalog — `tiny-1llc`, `odd-3llc`, `numa2-8llc`, 26 in all on x86_64. |
| **cell** | One (scheduler × kernel × preset) unit of the verifier sweep: verify, attach, dispatch. |
| **perf-delta** | The regression gate comparing performance metrics between HEAD and a baseline commit, with `--noise-adjust` running both sides fresh. See [Runs](../running-tests/runs.md#perf-delta). |
| **auto-repro** | On a crash, the failing scenario reruns in a second VM with BPF probes on the crash call chain, capturing argument and struct state on the way to the error. See [Auto-Repro](../running-tests/auto-repro.md). |
| **performance mode** | Exclusive host-LLC reservation for measurement-grade runs; budgeted (shared) mode is the default. See [Resource Budget](../concepts/resource-budget.md). |
