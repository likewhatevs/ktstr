# Scheduler Definitions

A `Scheduler` tells the framework how to find, configure, and launch
the scheduler under test. `declare_scheduler!` builds one and
registers it so both `#[ktstr_test]` and the
[verifier sweep](../running-tests/verifier.md) can see it:

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Find</strong><p>Discover a built <code>scx_*</code> binary, point at an absolute path, or use a kernel built-in scheduler.</p></div>
<div class="kt-doc-card"><strong>Configure</strong><p>Attach scheduler args, default topology, constraints, and scheduler-wide assertions.</p></div>
<div class="kt-doc-card"><strong>Reuse</strong><p>One declaration feeds tests, verifier cells, staged swaps, sidecars, and CLI labels.</p></div>
</div>

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    sched_args = ["--exit-dump-len", "1048576"],
    topology = (1, 2, 4, 1),
});

#[ktstr_test(scheduler = MY_SCHED)]
fn basic(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![ctx.cgroup_def("cg_0"), ctx.cgroup_def("cg_1")])
}
```

`MY_SCHED` is the Rust handle tests reference; `name = "my_sched"`
is the user-visible label in nextest output, sidecars, and the CLI.
Rename either independently. Once declared, the scheduler shows up
in the verifier sweep's cells with no further wiring:

<!-- captured: cargo ktstr verifier --kernel 7.0 -- --test kaslr_axis_e2e 4cpu-1llc-nosmt 4cpu-2llc-nosmt 9cpu-3llc-nosmt 8cpu-2llc-smt | ktstr 0.23.0 | kernel 7.0.14 -->
```text
 Nextest run ID 3522bea7-... with nextest profile: default
    Starting 4 tests across 1 binary (55 tests skipped)
        PASS [  12.406s] (1/4) ktstr::kaslr_axis_e2e verifier/ktstr_sched/kernel_7_0/9cpu-3llc-nosmt
        PASS [  12.432s] (2/4) ktstr::kaslr_axis_e2e verifier/ktstr_sched/kernel_7_0/8cpu-2llc-smt
...
```

## Defining a scheduler {#defining-a-scheduler}

`declare_scheduler!` emits a `pub static MY_SCHED: Scheduler` and
registers a reference to it in the `KTSTR_SCHEDULERS` distributed
slice, which is what `cargo ktstr verifier` enumerates.
`#[ktstr_test(scheduler = ...)]` expects the bare ident; the macro
takes the reference internally. The ident can carry a visibility
prefix (`pub`, `pub(crate)`).

### Accepted fields

`name` plus exactly one binary-source key (`binary`, `binary_path`,
or the `kernel_builtin_enable`/`kernel_builtin_disable` pair) are
required; every other key is optional.

- `name = "..."` — short human name (required).
- `binary = "scx_name"` — discover a binary by name. Resolution
  happens entirely on the host, before the VM boots, and the
  resolved binary is packed into the guest initramfs — nothing is
  resolved inside the guest. The order: the global `KTSTR_SCHEDULER`
  override, then a fresh build via `cargo build -p <name>` run in the
  workspace of the crate that *declared* the scheduler (the
  `declare_scheduler!` call site's `CARGO_MANIFEST_DIR`). Cargo owns
  freshness — it rebuilds when the scheduler's sources change,
  no-ops when they are up to date, and fails when it cannot build. A
  failed build is a hard error: there is no pre-built fallback,
  because a scheduler that will not build is a failed test. Test
  binaries run outside the cargo-ktstr pipeline
  (`KTSTR_CARGO_TEST_MODE=1`) consult the host `PATH` before the
  build, so an installed scheduler resolves without an in-tree build.

  Because the build runs in the declaring crate's own workspace, a
  scheduler living in a *separate* workspace just needs its
  `declare_scheduler!` to be compiled in that workspace — no env var
  is required. In CI, a prior `cargo build --release -p <name>` in
  the same workspace makes the in-test build a cache no-op.
- `binary_path = "/abs/path"` — explicit pre-built binary; must
  exist on the host, packed into the initramfs as-is.
- `kernel_builtin_enable = [...]` + `kernel_builtin_disable = [...]`
  — paired guest shell-command lists for a scheduler compiled into
  the kernel (no userspace binary). Both keys must appear together.
- `sched_args = ["--a", "--b"]` — scheduler CLI args applied to
  every test; per-test `extra_sched_args` append after them.
- `kargs = ["nosmt"]` — extra guest kernel command line (not the
  scheduler's CLI — that's `sched_args`). Do not override the kargs
  ktstr injects itself (`console=`, `loglevel=`, `rdinit=`); those
  break guest init.
- `sysctls = [Sysctl::new("kernel.foo", "1")]` — applied at guest
  boot, before the scheduler starts (see below).
- `topology = (numa_nodes, llcs, cores, threads)` — default VM
  topology tests inherit dimension-by-dimension.
- `constraints = TopologyConstraints { ... }` — gauntlet constraints
  tests inherit; see the
  [macro reference](ktstr-test-macro.md#topology-constraints).
- `cgroup_parent = "/path"` — cgroup subtree the guest creates for
  the scheduler before it starts (see below).
- `config_file = "configs/my.toml"` / `config_file_def = (...)` —
  the two config-file seams (see below).
- `assert = Assert::NO_OVERRIDES.max_imbalance_ratio(2.0)` —
  scheduler-level checking overrides, merged between the library
  defaults and per-test attributes. See
  [Customize Checking](../recipes/custom-checking.md).
- `kernels = ["6.14", "6.15..=7.0"]` — filters which kernel-list
  entries this scheduler verifies against in the verifier sweep.
  Entries use the same grammar as `cargo ktstr verifier --kernel`
  (exact versions, inclusive ranges, paths, git specs); empty means
  no filter. Match semantics live in
  [BPF Verifier Sweep](../running-tests/verifier.md).
- `verifier_exclude_topologies = ["240cpu-15llc-nosmt"]` — removes
  exact preset names only from this scheduler's verifier matrix when a
  topology is irrelevant or unsupported. It does not change ordinary
  tests or gauntlet variants.

### Manual definition {#manual-definition}

The const builder still works when the macro doesn't fit — e.g. a
programmatically composed scheduler, or a fixture that must stay out
of the verifier sweep:

```rust,ignore
use ktstr::prelude::*;

const MY_SCHED: Scheduler = Scheduler::named("my_sched")
    .binary_discover("scx_my_sched")
    .topology(1, 2, 4, 1)
    .sched_args(&["--exit-dump-len", "1048576"])
    .cgroup_parent("/ktstr")
    .assert(Assert::NO_OVERRIDES.max_imbalance_ratio(2.0));
```

`binary_discover("scx_my_sched")` is host-side binary discovery; the
argument is the binary name to discover, not the scheduler label. It is
shorthand for `.binary(SchedulerSpec::Discover("scx_my_sched"))`. A
manual const is not registered in `KTSTR_SCHEDULERS`, so the verifier
sweep does not see it; use `declare_scheduler!` for anything that
should participate in `cargo ktstr verifier`.

## SchedulerSpec

```rust,ignore
pub enum SchedulerSpec {
    Eevdf,                   // no sched_ext binary — kernel EEVDF
    Discover(&'static str),  // host-side discovery by name
    Path(&'static str),      // explicit host path
    KernelBuiltin {          // compiled into the kernel
        enable: &'static [&'static str],
        disable: &'static [&'static str],
    },
}
```

`Scheduler::EEVDF` (binary `SchedulerSpec::Eevdf`) runs tests under
the kernel's default scheduler and is what `#[ktstr_test]` uses when
`scheduler =` is omitted. It is not reachable via
`declare_scheduler!` — reference `Scheduler::EEVDF` directly.
`Eevdf` and `KernelBuiltin` are excluded from the verifier sweep:
neither has a userspace binary to load BPF programs from.

### Kernel-builtin example

```rust,ignore
declare_scheduler!(MINLAT, {
    name = "minlat",
    kernel_builtin_enable = ["echo minlat > /sys/kernel/debug/sched/ext/root/ops"],
    kernel_builtin_disable = ["echo none > /sys/kernel/debug/sched/ext/root/ops"],
});
```

The `enable` commands run in the guest before scenarios start;
`disable` runs after they complete.

## Sysctls

`sysctls` takes `Sysctl::new("key", "value")` pairs (dot-separated
keys; duplicates apply in order, last write wins). The framework
injects each as `sysctl.<key>=<value>` on the guest kernel command
line, so the kernel applies them at boot — each test gets a fresh
VM, so there is no apply/revert step. `Sysctl::new` is `const fn`,
so a shared tuning block can live in a `const` slice:

```rust,ignore
const RT_TUNING: &[Sysctl] = &[
    Sysctl::new("kernel.sched_rt_runtime_us", "950000"),
    Sysctl::new("kernel.numa_balancing", "0"),
];

declare_scheduler!(RT_TUNED, {
    name = "rt_tuned_scx",
    binary = "scx_rt_tuned",
    sysctls = RT_TUNING,
});
```

## Config files

**Pick one of `config_file` / `config_file_def` — they are
alternatives.**

- The config is the same file for every test →
  `config_file = "configs/my_sched.toml"`. The framework packs the
  host file into the guest at `/include-files/{filename}` and
  prepends `--config /include-files/{filename}` to the scheduler
  args. The `--config` flag name is fixed; a scheduler that uses a
  different flag can still take the packed path via `sched_args`,
  but must tolerate the extra `--config` argument.
- The config varies per test → `config_file_def = ("--config={file}",
  "/include-files/my.json")` declares the arg-template + guest-path
  pair, and each test supplies content via
  `#[ktstr_test(config = …)]`. The pairing is enforced both ways at
  compile time — see
  [Inline scheduler config](ktstr-test-macro.md#inline-scheduler-config).

Both fields may technically coexist (the `config_file` path is
always packed and its flag prepended; the inline config is written
when a test supplies `config = …`), but a two-config launch is
rarely what anyone wants — pick one.

## Cgroup parent

`cgroup_parent = "/ktstr"` makes guest init create
`/sys/fs/cgroup/ktstr` (enabling cpuset/cpu controllers on its
ancestors) before the scheduler starts. It does not pass
`--cell-parent-cgroup` to the scheduler — a cell-aware scheduler
that needs the flag must carry it in `sched_args` or per-test
`extra_sched_args`, and the guest then also creates the directory
named by the flag. Paths are validated at compile time by
`CgroupPath`: they must start with `/`, must not be `/` alone, and
must not contain `..`.

The same validation applies to any `--cell-parent-cgroup` value
found in `sched_args` / `extra_sched_args` at test setup: empty
values, bare `/`, relative paths, and a trailing flag with no value
all panic with an actionable message instead of resolving to (or
next to) the host cgroup root and corrupting host state.

## Default topology {#default-topology}

`topology = (numa_nodes, llcs, cores_per_llc, threads_per_core)`
sets the VM topology tests inherit. `Scheduler::named()` defaults to
`(1, 1, 2, 1)` — a minimal 2-CPU VM. Tests override individual
dimensions; unset ones still inherit:

```rust,ignore
// Inherits llcs=2, cores=4 from MITOSIS; overrides threads to 2.
#[ktstr_test(scheduler = MITOSIS, threads = 2)]
fn smt_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

## Related test-attribute slots

Two `#[ktstr_test]` attributes complement the scheduler definition:
`staged_schedulers = [PATH, …]` packs extra scheduler binaries for
runtime swaps via `Op::ReplaceScheduler` / `Op::AttachScheduler`,
and `workload_root_cgroup = "/path"` roots workload cgroups
independently of the scheduler's `cgroup_parent`. Both are
documented in [the macro reference](ktstr-test-macro.md#execution).

## Payloads {#derive-payload}

Payload authoring — `#[derive(Payload)]`, metric hints, include
files — lives on its own page:
[Payloads and Included Files](payloads.md).
