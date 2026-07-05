# Gauntlet

Some scheduler bugs only exist on topologies you don't develop on: a
per-LLC work-splitting heuristic that breaks on an odd LLC count, an
idle-core picker that lands both SMT siblings, a migration policy that
never crosses a NUMA boundary. The gauntlet expands every
`#[ktstr_test]` into one variant per topology preset — up to 24
presets (14 on aarch64) — so those bugs surface as a named, re-runnable
test case instead of a production report.

Gauntlet variants are prefixed `gauntlet/` and ignored by default:

```sh
# Run only base tests (default)
cargo ktstr test --kernel ../linux

# Run only gauntlet variants
cargo ktstr test --kernel ../linux -- --run-ignored ignored-only -E 'test(gauntlet/)'

# Run everything
cargo ktstr test --kernel ../linux -- --run-ignored all

# Run a single variant
cargo ktstr test --kernel ../linux -- --run-ignored ignored-only \
  -E 'test(=gauntlet/my_test/smt-2llc)'
```

This is what the expansion looks like when nextest lists a test with
`min_llcs = 1` and default constraints on this host:

<!-- captured: KTSTR_KERNEL=7.0 cargo nextest list --features integration -E 'test(gauntlet/) & binary(worktype_coverage_fork_gauntlet_e2e)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/medium-4llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/medium-4llc-nosmt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/odd-3llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/smt-2llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/smt-3llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/tiny-1llc
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/tiny-2llc
```

Under a multi-kernel run a `{kernel_label}` segment is appended —
`gauntlet/{name}/{preset}/{kernel}`. See
[Test names and variants](../running-tests.md#test-name-shapes) for
the label format.

## Topology presets

> [!NOTE]
> Multi-NUMA and scale-boundary presets are **opt-in**. The default
> constraints (`max_numa_nodes = 1`, `max_llcs = 12`,
> `max_cpus = 192`) exclude the five `numa*` presets plus
> `near-max-llc`, `max-cpu`, and their `-nosmt` variants — 15 of the
> 24 presets are active by default. Raise `max_numa_nodes`,
> `max_llcs`, or `max_cpus` on the test to opt in.

| Preset | Topology | CPUs | LLCs | NUMA | Description |
|---|---|---|---|---|---|
| `tiny-1llc` | 1n1l4c1t | 4 | 1 | 1 | Single LLC |
| `tiny-2llc` | 1n2l2c1t | 4 | 2 | 1 | Minimal multi-LLC |
| `odd-3llc` | 1n3l3c1t | 9 | 3 | 1 | Odd CPU count |
| `odd-5llc` | 1n5l3c1t | 15 | 5 | 1 | Prime LLC count |
| `odd-7llc` | 1n7l2c1t | 14 | 7 | 1 | Prime LLC count |
| `smt-2llc` | 1n2l2c2t | 8 | 2 | 1 | SMT enabled |
| `smt-3llc` | 1n3l2c2t | 12 | 3 | 1 | SMT, 3 LLCs |
| `medium-4llc` | 1n4l4c2t | 32 | 4 | 1 | Medium topology |
| `medium-8llc` | 1n8l4c2t | 64 | 8 | 1 | Medium, many LLCs |
| `large-4llc` | 1n4l16c2t | 128 | 4 | 1 | Large, few LLCs |
| `large-8llc` | 1n8l8c2t | 128 | 8 | 1 | Large, many LLCs |
| `near-max-llc` | 1n15l8c2t | 240 | 15 | 1 | Near maximum |
| `max-cpu` | 1n14l9c2t | 252 | 14 | 1 | Near KVM vCPU limit |
| `medium-4llc-nosmt` | 1n4l8c1t | 32 | 4 | 1 | Medium, no SMT |
| `medium-8llc-nosmt` | 1n8l8c1t | 64 | 8 | 1 | Medium, many LLCs, no SMT |
| `large-4llc-nosmt` | 1n4l32c1t | 128 | 4 | 1 | Large, no SMT |
| `large-8llc-nosmt` | 1n8l16c1t | 128 | 8 | 1 | Large, many LLCs, no SMT |
| `near-max-llc-nosmt` | 1n15l16c1t | 240 | 15 | 1 | Near maximum, no SMT |
| `max-cpu-nosmt` | 1n14l18c1t | 252 | 14 | 1 | Near KVM vCPU limit, no SMT |
| `numa2-4llc` | 2n4l4c1t | 16 | 4 | 2 | Multi-NUMA, 2 nodes |
| `numa2-8llc` | 2n8l8c2t | 128 | 8 | 2 | Multi-NUMA, 2 nodes, SMT |
| `numa2-8llc-nosmt` | 2n8l16c1t | 128 | 8 | 2 | Multi-NUMA, 2 nodes, no SMT |
| `numa4-8llc` | 4n8l4c1t | 32 | 8 | 4 | Multi-NUMA, 4 nodes |
| `numa4-12llc` | 4n12l8c2t | 192 | 12 | 4 | Multi-NUMA, 4 nodes, SMT |

Topology format: `{numa_nodes}n{llcs}l{cores_per_llc}c{threads_per_core}t`
— `1n2l4c2t` is 1 NUMA node, 2 LLCs, 4 cores per LLC, 2 threads per
core = 16 CPUs. Note that `llcs` is the total across the machine, not
per node.

**aarch64:** ARM64 CPUs do not have SMT. Presets with
`threads_per_core > 1` are excluded on aarch64, leaving 14 presets
(the 5 small presets, 6 `-nosmt` variants, and 3 non-SMT NUMA
presets).

## Constraint filtering

`#[ktstr_test]` topology constraints filter which presets a test runs
on. A preset is skipped when any constraint is not met:

- `num_numa_nodes() < min_numa_nodes`
- `max_numa_nodes` is set and `num_numa_nodes() > max_numa_nodes`
- `num_llcs() < min_llcs`
- `max_llcs` is set and `num_llcs() > max_llcs`
- `requires_smt` and `threads_per_core < 2`
- `total_cpus() < min_cpus`
- `max_cpus` is set and `total_cpus() > max_cpus`

See [The #\[ktstr_test\] Attribute](../writing-tests/ktstr-test-macro.md)
for the attribute table.

## Authoring gauntlet-ready tests {#authoring}

### Worked example

A test with `min_llcs = 2`, `requires_smt = true`, and default
`max_numa_nodes = 1` against the preset table above:

- `tiny-1llc` (1 LLC): excluded — below `min_llcs`
- All non-SMT presets (`tiny-2llc`, `odd-*`, `*-nosmt`):
  excluded — `requires_smt`
- `near-max-llc` (15 LLCs): excluded — above default `max_llcs = 12`
- `max-cpu` (252 CPUs, 14 LLCs): excluded — above default
  `max_cpus = 192` (also above default `max_llcs = 12`)
- All `numa*` presets: excluded — above default `max_numa_nodes = 1`

Result: 6 of 24 presets survive (`smt-2llc`, `smt-3llc`,
`medium-4llc`, `medium-8llc`, `large-4llc`, `large-8llc`). On
aarch64, none survive — all aarch64 presets lack SMT.

### Variant count

The total number of gauntlet variants for a test is
`valid_presets × resolved_kernels`: the 6 surviving presets above
produce 6 variants under a single kernel and 12 under
`--kernel A --kernel B`.

### Tests that skip gauntlet

Entries with `host_only = true` never produce gauntlet variants —
they run on the host without booting a VM, so topology variation
carries no signal. Tests whose names start with `demo_` are ignored
by default, gauntlet variants included.

## Operator notes

- **Wall time.** Each variant boots its own VM and runs the full
  scenario, so a sweep costs roughly (surviving presets × the per-run
  wall time you observe for the base test). nextest runs variants in
  parallel within your host's budget. For a coverage-per-second subset
  under a deadline, use
  [budget-based selection](../running-tests.md#budget-based-test-selection).
- **Memory.** Each gauntlet VM gets
  `max(cpus × 64 MiB, 256 MiB, entry.memory_mib)` of guest RAM (plus
  an initramfs-derived floor). For the 252-CPU `max-cpu` presets that
  is at least 16128 MiB — the host needs that much free memory to run
  the variant.
