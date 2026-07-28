# Gauntlet

Some scheduler bugs only exist on topologies you don't develop on: a
per-LLC work-splitting heuristic that breaks on an odd LLC count, an
idle-core picker that lands both SMT siblings, a migration policy that
never crosses a NUMA boundary. The gauntlet expands every
`#[ktstr_test]` into one variant per topology preset — up to 25
presets (14 on aarch64) — so those bugs surface as a named, re-runnable
test case instead of a production report.

<div class="kt-figure"><svg width="700" height="180" viewBox="0 0 700 180" role="img" aria-label="Gauntlet fan-out: one #[ktstr_test] declaration expands to one variant per topology preset; some presets are filtered out by constraints or host budget, and the whole matrix repeats for each --kernel">
  <defs><marker id="kt-arr5" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <rect x="8" y="70" width="150" height="62" rx="12" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
  <text x="83" y="97" font-size="11.5" font-weight="700" fill="var(--kt-accent)" text-anchor="middle">#[ktstr_test]</text>
  <text x="83" y="115" font-size="10" fill="var(--fg)" text-anchor="middle" opacity=".8">one declaration</text>
  <path d="M158 101 L 296 53" stroke="var(--fg)" stroke-width="1.2" fill="none" marker-end="url(#kt-arr5)"/>
  <path d="M158 101 L 296 91" stroke="var(--fg)" stroke-width="1.2" fill="none" marker-end="url(#kt-arr5)"/>
  <path d="M158 101 L 296 129" stroke="var(--fg)" stroke-width="1.2" fill="none" marker-end="url(#kt-arr5)"/>
  <text x="170" y="150" font-size="9.5" fill="var(--fg)" opacity=".7">one variant / preset</text>
  <g font-size="7.5">
    <rect x="300" y="40" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="350" y="57" fill="var(--fg)" text-anchor="middle">4cpu-1llc-nosmt</text>
    <rect x="410" y="40" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="460" y="57" fill="var(--fg)" text-anchor="middle">4cpu-2llc-nosmt</text>
    <rect x="520" y="40" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="570" y="57" fill="var(--fg)" text-anchor="middle">9cpu-3llc-nosmt</text>
    <rect x="300" y="78" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="350" y="95" fill="var(--fg)" text-anchor="middle">8cpu-2llc-smt</text>
    <rect x="410" y="78" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="460" y="95" fill="var(--fg)" text-anchor="middle">32cpu-4llc-smt</text>
    <rect x="520" y="78" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="570" y="95" fill="var(--fg)" text-anchor="middle">64cpu-8llc-smt</text>
    <rect x="300" y="116" width="100" height="26" rx="6" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)"/><text x="350" y="133" fill="var(--fg)" text-anchor="middle">128cpu-4llc-smt</text>
    <g opacity=".5">
      <rect x="410" y="116" width="100" height="26" rx="6" fill="none" stroke="var(--kt-rule)"/><text x="460" y="133" font-size="8.5" fill="var(--fg)" text-anchor="middle">240cpu-15llc-smt</text>
      <line x1="414" y1="140" x2="506" y2="118" stroke="var(--kt-rule)" stroke-width="1"/>
      <rect x="520" y="116" width="100" height="26" rx="6" fill="none" stroke="var(--kt-rule)"/><text x="570" y="133" fill="var(--fg)" text-anchor="middle">2numa-16cpu-4llc-nosmt</text>
      <line x1="524" y1="140" x2="616" y2="118" stroke="var(--kt-rule)" stroke-width="1"/>
    </g>
  </g>
  <text x="515" y="162" font-size="9.5" fill="var(--fg)" text-anchor="middle" opacity=".7">filtered: constraints / host budget</text>
  <path d="M640 40 L 646 40 L 646 142 L 640 142" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
  <text transform="rotate(90 666 91)" x="666" y="91" font-size="10.5" fill="var(--fg)" text-anchor="middle" opacity=".75">× each --kernel</text>
</svg></div>

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
  -E 'test(=gauntlet/my_test/8cpu-2llc-smt)'
```

This is what the expansion looks like when nextest lists a test with
`min_llcs = 1` and default constraints on this host:

<!-- captured: KTSTR_KERNEL=7.0 cargo nextest list --features integration -E 'test(gauntlet/) & binary(worktype_coverage_fork_gauntlet_e2e)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/12cpu-3llc-smt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/32cpu-4llc-nosmt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/32cpu-4llc-smt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/4cpu-1llc-nosmt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/4cpu-2llc-nosmt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/8cpu-2llc-smt
ktstr::worktype_coverage_fork_gauntlet_e2e gauntlet/worktype_fork_gauntlet_covers_all_arms/9cpu-3llc-nosmt
```

Under a multi-kernel run a `{kernel_label}` segment is appended —
`gauntlet/{name}/{preset}/{kernel}`. See
[Test names and variants](../running-tests.md#test-name-shapes) for
the label format.

## Topology presets

> [!NOTE]
> Multi-NUMA and scale-boundary presets are **opt-in**. The default
> constraints (`max_numa_nodes = 1`, `max_llcs = 12`,
> `max_cpus = 192`) exclude the six `*numa-*` presets plus
> `240cpu-15llc-smt`, `240cpu-15llc-nosmt`, `252cpu-14llc-smt`, and
> `252cpu-14llc-nosmt` — 15 of the 25 gauntlet presets are active by
> default. Raise `max_numa_nodes`, `max_llcs`, or `max_cpus` on the
> test to opt in.
>
> The catalog has one further preset, `192cpu-11llc-smt` (26 in all), that
> is **verifier-only**: its non-uniform LLC sizing cannot be expressed
> through the gauntlet execution path, so it never fans out as a
> `gauntlet/…` variant — only the [BPF Verifier
> Sweep](verifier.md) boots it.

| Preset | Topology | CPUs | LLCs | NUMA | Description |
|---|---|---|---|---|---|
| `4cpu-1llc-nosmt` | 1n1l4c1t | 4 | 1 | 1 | Single LLC |
| `4cpu-2llc-nosmt` | 1n2l2c1t | 4 | 2 | 1 | Minimal multi-LLC |
| `9cpu-3llc-nosmt` | 1n3l3c1t | 9 | 3 | 1 | Odd CPU count |
| `15cpu-5llc-nosmt` | 1n5l3c1t | 15 | 5 | 1 | Prime LLC count |
| `14cpu-7llc-nosmt` | 1n7l2c1t | 14 | 7 | 1 | Prime LLC count |
| `8cpu-2llc-smt` | 1n2l2c2t | 8 | 2 | 1 | SMT enabled |
| `12cpu-3llc-smt` | 1n3l2c2t | 12 | 3 | 1 | SMT, 3 LLCs |
| `32cpu-4llc-smt` | 1n4l4c2t | 32 | 4 | 1 | Medium topology |
| `64cpu-8llc-smt` | 1n8l4c2t | 64 | 8 | 1 | Medium, many LLCs |
| `128cpu-4llc-smt` | 1n4l16c2t | 128 | 4 | 1 | Large, few LLCs |
| `128cpu-8llc-smt` | 1n8l8c2t | 128 | 8 | 1 | Large, many LLCs |
| `240cpu-15llc-smt` | 1n15l8c2t | 240 | 15 | 1 | 15-LLC SMT topology |
| `252cpu-14llc-smt` | 1n14l9c2t | 252 | 14 | 1 | Near KVM vCPU limit |
| `192cpu-11llc-smt` | 1n11l\*c2t | 192 | 11 | 1 | **Verifier-only.** Non-uniform LLCs — ten of 18 CPUs + one of 12. Forces continuous overcommit (`forced_cpu_budget = 96`) so 192 vCPUs always time-slice. Targets schedulers assuming equal-sized caches. |
| `32cpu-4llc-nosmt` | 1n4l8c1t | 32 | 4 | 1 | Medium, no SMT |
| `64cpu-8llc-nosmt` | 1n8l8c1t | 64 | 8 | 1 | Medium, many LLCs, no SMT |
| `128cpu-4llc-nosmt` | 1n4l32c1t | 128 | 4 | 1 | Large, no SMT |
| `128cpu-8llc-nosmt` | 1n8l16c1t | 128 | 8 | 1 | Large, many LLCs, no SMT |
| `240cpu-15llc-nosmt` | 1n15l16c1t | 240 | 15 | 1 | 15-LLC topology, no SMT |
| `252cpu-14llc-nosmt` | 1n14l18c1t | 252 | 14 | 1 | Near KVM vCPU limit, no SMT |
| `2numa-32cpu-2llc-smt` | 2n2l8c2t | 32 | 2 | 2 | Multi-NUMA, 2 nodes, one LLC per node, SMT |
| `2numa-16cpu-4llc-nosmt` | 2n4l4c1t | 16 | 4 | 2 | Multi-NUMA, 2 nodes |
| `2numa-128cpu-8llc-smt` | 2n8l8c2t | 128 | 8 | 2 | Multi-NUMA, 2 nodes, SMT |
| `2numa-128cpu-8llc-nosmt` | 2n8l16c1t | 128 | 8 | 2 | Multi-NUMA, 2 nodes, no SMT |
| `4numa-32cpu-8llc-nosmt` | 4n8l4c1t | 32 | 8 | 4 | Multi-NUMA, 4 nodes |
| `4numa-192cpu-12llc-smt` | 4n12l8c2t | 192 | 12 | 4 | Multi-NUMA, 4 nodes, SMT |

Topology format: `{numa_nodes}n{llcs}l{cores_per_llc}c{threads_per_core}t`
— `1n2l4c2t` is 1 NUMA node, 2 LLCs, 4 cores per LLC, 2 threads per
core = 16 CPUs. Note that `llcs` is the total across the machine, not
per node. A `*` in the cores field (`1n11l*c2t`) marks a **non-uniform**
machine whose LLCs are not all the same size — `192cpu-11llc-smt` is ten
LLCs of 9 cores plus one of 6 (packing width 9); the guest observes the
uneven layout via fixed-width, partially-populated APIC-ID blocks.

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

- `4cpu-1llc-nosmt` (1 LLC): excluded — below `min_llcs`
- All non-SMT presets (`4cpu-2llc-nosmt`, `9cpu-3llc-nosmt`, `*-nosmt`):
  excluded — `requires_smt`
- `240cpu-15llc-smt` (15 LLCs): excluded — above default `max_llcs = 12`
- `252cpu-14llc-smt` (252 CPUs, 14 LLCs): excluded — above default
  `max_cpus = 192` (also above default `max_llcs = 12`)
- All `*numa-*` presets: excluded — above default `max_numa_nodes = 1`

Result: 6 of 25 gauntlet presets survive (`8cpu-2llc-smt`, `12cpu-3llc-smt`,
`32cpu-4llc-smt`, `64cpu-8llc-smt`, `128cpu-4llc-smt`, `128cpu-8llc-smt`).
(`192cpu-11llc-smt` is verifier-only and never in the gauntlet count.) On
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
  an initramfs-derived floor). For the `252cpu-14llc-*` presets that is
  at least 16128 MiB — the host needs that much free memory to run the
  variant.
