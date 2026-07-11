# BPF Verifier Sweep

A scheduler that loads on your machine can still be rejected — or
attach and then wedge — on a topology you never booted.
`verified_insns` varies with topology whenever topology-derived
config like `nr_cpus` is baked into `.rodata`: the verifier sees
different known constants, walks different branches, and can reach a
different verdict. The verifier sweep boots every declared scheduler
in a KVM VM across a range of topologies and checks three things
against the real kernel: the BPF programs verify, the scheduler
attaches as the active sched_ext scheduler, and it dispatches an
injected workload.

The verifier that runs is the real verifier in the real target
kernel — no host-side BPF loading, no version skew. And there is no
subprocess to `bpftool` or `veristat`: the host reads per-program
`verified_insns` directly from guest memory via `bpf_prog_aux`
introspection, and applies cycle collapse to verifier logs instead of
truncating them.

## Quick start

```sh
# Every declared scheduler, kernel discovered via KTSTR_KERNEL / cache
cargo ktstr verifier

# Pin the kernel
cargo ktstr verifier --kernel ../linux

# Sweep across kernels (each cell runs against its own)
cargo ktstr verifier --kernel 6.14.2 --kernel 7.0

# Every stable/longterm release in a range (add --include-eol for
# end-of-life series)
cargo ktstr verifier --kernel 6.12..6.14

# One scheduler across topologies
cargo ktstr verifier --scheduler scx-ktstr

# Raw verifier log, no cycle collapse
cargo ktstr verifier --raw
```

See [cargo-ktstr verifier](cargo-ktstr.md#verifier) for the flag
list.

## A real sweep

Three real schedulers from the [scx](https://github.com/sched-ext/scx)
tree, four topologies, one development kernel — each of the twelve
cells boots its own VM, loads the scheduler, and confirms
attach + dispatch. Declare the schedulers once (in any linked test
file), pointing at the prebuilt binaries:

```rust,ignore
ktstr::declare_scheduler!(BPFLAND, {
    name = "scx_bpfland",
    binary_path = "../scx/target/release/scx_bpfland",
});
// ... same for scx_lavd, scx_p2dq
```

<!-- captured: cargo ktstr verifier --kernel <local sched_ext dev tree> --test docs_real_scheds tiny-1llc tiny-2llc odd-3llc smt-2llc | ktstr 0.23.0 | kernel sched_ext-for-7.2 b4dc42d2 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr verifier --kernel ../linux --test my_schedulers tiny-1llc tiny-2llc odd-3llc smt-2llc</span></div>

<pre><span class="t-b">verifier verified_insns (per scheduler; rows: kernel, cols: BPF program, cell: range across topologies):</span>

scx_bpfland:
 kernel        bpfland_dispatc  bpfland_enable  bpfland_enqueue  bpfland_exit  bpfland_init  bpfland_init_ta  bpfland_runnabl  bpfland_running  bpfland_select_  bpfland_stoppin
 kernel_local  269..304         5               766              25            302           11               44               122              571..611         52

scx_p2dq:
 kernel        p2dq_dequeue  p2dq_dispatch  p2dq_enqueue  p2dq_exit  p2dq_exit_task  p2dq_init  p2dq_init_task  p2dq_running  p2dq_select_cpu  p2dq_set_cpumas  p2dq_stopping  p2dq_update_idl
 kernel_local  5             1159..2130     2026..5118    25         419             2121       27601           609           801..887         149              853            723

verifier summary: 8 ✅  <span class="t-red">4 ❌</span>  0 ❎
 topology   scx_bpfland  scx_lavd  scx_p2dq
 odd-3llc   ✅           <span class="t-red">❌</span>        ✅
 smt-2llc   ✅           <span class="t-red">❌</span>        ✅
 tiny-1llc  ✅           <span class="t-red">❌</span>        ✅
 tiny-2llc  ✅           <span class="t-red">❌</span>        ✅

failing combinations (scheduler / kernel / topology):
  scx_lavd / kernel_local / odd-3llc
  ...</pre></div>

That ❌ column is the sweep doing its job. This development kernel
removed the deprecated `scx_bpf_cpu_rq()` kfunc; `scx_lavd` still
requires it, so its BPF skeleton fails at load in every cell — caught
here, not on a user's machine. The failing cells' captured output
names the exact break:

<!-- captured: same sweep, scx_lavd cell scheduler log | ktstr 0.23.0 | kernel sched_ext-for-7.2 b4dc42d2 -->
```text
libbpf: extern (func ksym) 'scx_bpf_cpu_rq': not found in kernel or module BTFs
libbpf: failed to load BPF skeleton 'bpf_bpf': -EINVAL
Error: Failed to load BPF program (Invalid argument, os error 22)
```

A cell in the `verified_insns` table shows a single number when the
count is flat across topologies, `lo..hi` when it varies, and `-`
when that program reported no stats on that kernel. In the grid, ✅
means the scheduler verified, attached, and dispatched on every
kernel that ran the cell; ❌ means it failed on every kernel; ❎ means
mixed results across kernels (the failing-combinations list below the
grid is the authoritative record).

> [!NOTE]
> A scheduler in a *separate* workspace works with a bare
> `binary = "name"` (discovery) as long as its `declare_scheduler!` is
> compiled in that workspace: verifier cells are built with
> `cargo build -p <name>` run in the declaring crate's own workspace, and
> emission checks membership of that same workspace — so no env var is
> needed. `binary_path` (an explicit prebuilt binary) also works. In CI, a
> prior `cargo build --release -p <name>` in that workspace makes the
> in-test build a cache no-op. A `binary = "name"` whose package is not a
> member of the declaring crate's workspace has nothing to build, so its
> cells stay filtered.

## The kernel axis

`--kernel` takes the same grammar everywhere — repeatable flags, a
`START..END` range, versions, cache keys, paths, git refs (see
[kernel resolution](cargo-ktstr.md#test)). A range expands against
kernel.org's active releases, so end-of-life series silently drop
unless you ask for them:

<!-- captured: cargo ktstr kernel list --kernel 6.12..6.14 [--include-eol] | ktstr 0.23.0 -->
```text
$ cargo ktstr kernel list --kernel 6.12..6.14
kernel list: range expanded to 1 kernel(s): 6.12.95

$ cargo ktstr kernel list --kernel 6.12..6.14 --include-eol
kernel list: range expanded to 3 kernel(s): 6.12.95, 6.13.12, 6.14.11
```

With multiple kernels resolved, each cell runs against its own, the
`verified_insns` table grows one row per kernel, and the pass/fail
grid folds kernels into each cell (✅ passed on every kernel, ❌
failed on every kernel, ❎ mixed — the failing-combinations list is
authoritative):

<!-- captured: cargo ktstr verifier --kernel 7.0.14-tarball-x86_64-kcabd40422 --kernel local-8cd2b47-x86_64-kcabd40422 --scheduler ktstr_sched --test kaslr_axis_e2e tiny-1llc tiny-2llc | ktstr 0.23.0 | kernels 7.0.14 + v7.1-patched -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr verifier --kernel 7.0 --kernel ../linux --scheduler ktstr_sched tiny-1llc tiny-2llc</span></div>

<pre>ktstr_sched:
 kernel               ktstr_dispatch  ktstr_dump  ktstr_dump_cpu  ktstr_dump_task  ktstr_enqueue  ktstr_exit  ktstr_exit_task  ktstr_init  ktstr_init_task  ktstr_select_cp  ktstr_yield
 kernel_7_0_14        102             81          13              70               74             25          419              2296        29077            39               8
 kernel_local_8cd2b4  102             81          13              70               74             25          419              2296        29077            39               8

verifier summary: 2 ✅  0 ❌  0 ❎
 topology   ktstr_sched
 tiny-1llc  ✅
 tiny-2llc  ✅</pre></div>

Flat rows across kernels are the boring, reassuring case — the same
BPF verified identically on both. A kfunc or verifier change between
kernels shows up as diverging counts, or as a ❎ cell.

## What a cell checks

1. **Verify** — inside the VM the scheduler loads its BPF programs;
   the target kernel's verifier runs against them. The host reads
   per-program `verified_insns` from `bpf_prog_aux` via guest memory
   introspection. On load failure, libbpf's verifier log is forwarded
   to the host.
2. **Attach (positive confirmation)** — the guest confirms the
   scheduler process survived load and `/sys/kernel/sched_ext/state`
   reached `enabled`. The kernel sets `enabled` only after `ops.init`,
   per-task init, and switching eligible tasks to the sched_ext class,
   so this proves the scheduler is scheduling, not merely that its BPF
   loaded. Attach is confirmed only when the guest reaches its
   post-attach dispatch phase — a guest that vanishes early (e.g. a
   panic before any frame is emitted) fails rather than passing by
   default.
3. **Dispatch probe** — the verifier VM has no `#[ktstr_test]` body,
   so it injects a SpinWait workload sized to the guest's online CPUs,
   running as SCHED_EXT. A cell passes only when a worker makes
   forward progress after attach: a scheduler that attaches but never
   dispatches a runnable task is a distinct, worse failure the attach
   gate alone cannot catch.

Every cell boots with performance mode disabled
([no_perf_mode](../concepts/performance-mode.md)) — `verified_insns`
is perf-mode-independent, so cells share LLC reservations instead of
serializing on them.

## A real rejection

The fixture scheduler ships rejection knobs (see
[fixture knobs](#fixture-knobs)) precisely so this path stays
exercised. Here `--verify-loop` plants an unrolled loop ending in a
store through a null pointer — the verifier walks the loop, then
rejects the store. Note the collapse markers: the loop body is shown
once, not eight times:

<!-- captured: cargo ktstr verifier --kernel 7.0 --scheduler ktstr_broken --test verifier_pipeline tiny-1llc (scratch declare_scheduler! running scx-ktstr with --verify-loop) | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr verifier --kernel 7.0 --scheduler ktstr_broken --test verifier_pipeline tiny-1llc</span></div>

<pre>=== ktstr_broken | kernel kernel_7_0 | topology tiny-1llc ===

verifier
  <span class="t-red">scheduler: NOT ATTACHED — scheduler process exited during BPF load/startup</span>

verifier --- verifier stats ---
  processed=186  states=7/7

verifier --- scheduler log ---
Global function ktstr_dispatch() doesn't return scalar. Only those are supported.
0: R1=ctx() R10=fp0
; if (crash) @ main.bpf.c:423
0: (18) r1 = 0xff5d3bb3000f60dc       ; R1=map_value(map=bpf_bpf.bss,ks=4,vs=280,off=220)
...
; volatile u32 acc = 0; @ main.bpf.c:450
37: (63) *(u32 *)(r10 -8) = r1        ; R1=0 R10=fp0 fp-8=mmmm0
<span class="t-yel">--- 8x of the following 25 lines ---</span>
; u64 t = bpf_ktime_get_ns(); @ main.bpf.c:453
38: (85) call bpf_ktime_get_ns#5      ; R0=scalar()
; acc += (u32)t; @ main.bpf.c:454
39: (61) r1 = *(u32 *)(r10 -8)        ; R1=0 R10=fp0 fp-8=mmmm0
...
<span class="t-yel">--- 6 identical iterations omitted ---</span>
; u64 t = bpf_ktime_get_ns(); @ main.bpf.c:453
171: (85) call bpf_ktime_get_ns#5     ; R0=scalar()
...
--- end repeat ---
190: (b7) r1 = 0                      ; R1=0
; *p = (int)acc; @ main.bpf.c:464
...
192: (63) *(u32 *)(r1 +0) = r2
<span class="t-red">R1 invalid mem access 'scalar'</span>
processed 186 insns (limit 1000000) max_states_per_insn 0 total_states 7 peak_states 7 mark_read 0
...
verifier summary: 0 ✅  1 ❌  0 ❎
 topology   ktstr_broken
 tiny-1llc  ❌

failing combinations (scheduler / kernel / topology):
  <span class="t-red">ktstr_broken / kernel_7_0 / tiny-1llc</span>
error: cargo nextest run exited with 100</pre></div>

The interleaved `; source line @ file:line` comments name the C
statement each instruction group came from — the offending store is
`*p = (int)acc;` at `main.bpf.c:464`.

## Cycle collapse {#cycle-collapse-algorithm}

The kernel verifier unrolls loops, re-verifying each instruction with
updated register state. A bounded 8-instruction loop verified 100
times produces 800 near-identical lines that differ only in
register-state annotations; naive truncation loses the context you
came for. Cycle collapse keeps the structure: first iteration (what
the loop does), an omission count, last iteration (final state).

The algorithm normalizes lines by stripping register-state
annotations (source comments are preserved as anchors), finds the
most frequent normalized line to establish the cycle period (minimum
period 5 lines, minimum 3 repetitions), verifies consecutive blocks
match, and collapses — iterating up to 5 passes for nested loops.
`--raw` skips all of this and prints the full log.

## Matrix dimensions and filters {#matrix-dimensions--filters}

The sweep matrix is (declared scheduler × kernel × topology preset).
Schedulers come from the `declare_scheduler!` registry (`--scheduler
NAME` narrows to one; EEVDF and kernel-builtin declarations are
skipped — no userspace binary to verify). Kernels come from the
operator's `--kernel` set; with no flag, one auto-discovered kernel is
used. The topology axis is the set of
[gauntlet presets](gauntlet.md#topology-presets) each scheduler's
constraints accept.

Each scheduler's `kernels = [...]` declaration filters the
operator-supplied kernel set:

- `kernels = []` (or omitted) — accepts every kernel-list entry.
- Version specs (`"6.14.2"`) — match entries whose label equals the
  version (raw or sanitized form).
- Range specs (`"6.14..6.16"`, `"6.14..=6.16"`) — match entries whose
  version falls in the inclusive range. One asymmetry vs the CLI:
  `--kernel 6.14..6.16` widens a two-component end to the whole
  `6.16.x` series, but a declaration filter compares the end
  literally — `kernels = ["6.14..6.16"]` does not match a `6.16.5`
  entry.
- Path / cache-key / git specs — match by sanitized-label equality.

```sh
# Scheduler declares kernels = ["6.14..6.16"]
# Operator passes 6.14.2, 6.15.0, 6.17.0 — the third is filtered out.
# Cells emitted per accepted preset:
#   verifier/<sched>/kernel_6_14_2/<preset>
#   verifier/<sched>/kernel_6_15_0/<preset>
cargo ktstr verifier --kernel 6.14.2 --kernel 6.15.0 --kernel 6.17.0
```

A cell whose kernel label matches nothing in the resolved set errors
with a diagnostic naming the present labels — no silent fallback to
an unrelated kernel.

**Runtime**: total cost is one VM boot per cell — schedulers ×
kernels × accepted presets. Cells run in parallel under nextest; the
4-cell example above cost ~13 s.

## Fixture knobs {#fixture-knobs}

The `scx-ktstr` fixture scheduler ships two flags that make the
rejection path testable on demand:

- **`--fail-verify`** — sets a `.rodata` variable before
  `scx_ops_load!` that enables a store through a null pointer in
  `ktstr_dispatch` — the invalid access the verifier rejects.
- **`--verify-loop`** — same rejection, preceded by an unrolled
  8-iteration loop so the log exercises cycle collapse. It is
  deliberately not a `while(1)`: the verifier's infinite-loop
  analysis could keep `scx_ops_load` from returning within the host's
  scheduler-attach poll.

Pass them via `sched_args` on a scratch `declare_scheduler!` — that
is exactly how the rejection capture above was produced.
