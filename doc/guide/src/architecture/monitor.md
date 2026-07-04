# Monitor

The monitor observes scheduler state from the host side by reading
guest VM memory directly. It does not instrument the guest kernel or
the scheduler under test.

## What it reads

The monitor resolves kernel structure offsets via BTF (BPF Type Format)
from the guest kernel. Sources break into two hierarchy levels.

### Per-CPU runqueue

The monitor reads per-CPU runqueue structures to extract:

- `nr_running` -- number of runnable tasks on each CPU
- `scx_nr_running` -- tasks managed by the sched_ext scheduler
- `rq_clock` -- runqueue clock value
- `local_dsq_depth` -- scx local dispatch queue depth
- `scx_flags` -- sched_ext flags for each CPU
- scx event counters (fallback, keep-last, offline dispatch,
  skip-exiting, skip-migration-disabled, reenq-immed,
  reenq-local-repeat, refill-slice-dfl, bypass-duration,
  bypass-dispatch, bypass-activate, insert-not-owned,
  sub-bypass-dispatch)

When `CONFIG_SCHEDSTATS` is enabled, the monitor also reads per-CPU
`struct rq` schedstat fields (run_delay, pcount, sched_count,
ttwu_count, etc.).

### Per sched_domain level

The monitor walks the `struct sched_domain` tree whenever BTF
contains `rq->sd` and `struct sched_domain` — no `CONFIG_SCHEDSTATS`
required. Domain tree walking starts at `rq->sd` (lowest level) and
follows `sd->parent` pointers up to the root. Each domain level
provides:

- Topology metadata: `level`, `name`, `flags`, `span_weight`.
- Runtime fields: `balance_interval`, `nr_balance_failed`,
  `max_newidle_lb_cost`.
- Optional fields: `newidle_call`, `newidle_success`,
  `newidle_ratio` — added together in 7.0 (backported to 6.18.5+
  and 6.12.65+); resolved as optional and absent on 6.16–6.18.4.

When `CONFIG_SCHEDSTATS` is also enabled, each domain additionally
provides load-balancing stats: `lb_count`, `lb_failed`, `lb_balanced`,
`alb_pushed`, `ttwu_wake_remote`, and other counters indexed by idle
type (`CPU_NOT_IDLE`, `CPU_IDLE`, `CPU_NEWLY_IDLE`).

## Sampling

The monitor takes periodic snapshots (`MonitorSample`) of all per-CPU
state. Each sample captures a point-in-time view of every CPU.

`MonitorSummary` aggregates samples into peak values (max imbalance
ratio, max DSQ depth, stall detection), per-sample averages
(imbalance ratio, nr_running per CPU, DSQ depth per CPU), and event
counter deltas. Averages are computed over valid samples only
(excluding uninitialized guest memory).

## Threshold evaluation

`MonitorThresholds` defines pass/fail conditions:

```rust,ignore
pub struct MonitorThresholds {
    pub max_imbalance_ratio: f64,
    pub max_local_dsq_depth: u32,
    pub fail_on_stall: bool,
    pub sustained_samples: usize,
    pub max_fallback_rate: f64,
    pub max_keep_last_rate: f64,
    pub enforce: bool,
}
```

A violation must persist for `sustained_samples` consecutive samples
before triggering a failure. This filters transient spikes from cpuset
transitions and cgroup creation/destruction.

**`enforce` is the on/off gate for the threshold-violation path.** The
default (`MonitorThresholds::new()`) is `enforce: false` — monitor
evaluations record observations but do NOT fail the test on threshold
violations. Test authors opt in to enforcement by either setting
`enforce: true` on a custom `MonitorThresholds`, or using
`Assert::with_monitor_defaults()` (an Assert-builder method that fills
unset threshold fields and sets `enforce`, which propagates into the
produced `MonitorThresholds`). Setting
fields like `fail_on_stall: true` without flipping `enforce` is a
no-op for the violation path — every violation will appear in the
monitor report but the verdict will pass.

The no-signal arms (empty sample buffer, or `data_looks_valid` rejection
of uninitialized guest memory) are evaluator-internal and bypass
`enforce` — they always produce a `MonitorVerdict` with
`passed: false, inconclusive: true`, which folds into the test's
`AssertResult` as Inconclusive (exit code 2) regardless of `enforce`.
"Couldn't evaluate" is not the same as "evaluated and OK," so the
no-signal path always surfaces distinct from Pass. Only threshold
*violations* are gated by `enforce`.

### Stall detection

A stall is detected when a CPU's `rq_clock` does not advance between
consecutive samples. Three exemptions prevent false positives:

- **Idle CPUs**: when `nr_running == 0` in both the current and previous
  sample, the CPU has no runnable tasks. The kernel stops the tick
  (NOHZ) on idle CPUs, so `rq_clock` legitimately does not advance.
  These CPUs are excluded from stall checks.

- **Preempted vCPUs**: when the vCPU thread's CPU time did not advance
  past the preemption threshold between samples, the host preempted the
  vCPU. These samples are excluded from stall checks.

- **Sustained window**: stall detection uses per-CPU consecutive
  counters and the `sustained_samples` threshold, matching how
  imbalance and DSQ depth checks work. A single stuck sample does
  not trigger failure -- the stall must persist for `sustained_samples`
  consecutive samples on the same CPU.

## Uninitialized memory detection

Before the guest kernel initializes per-CPU structures, monitor reads
return uninitialized data. Two layers handle this:

- **Summary computation** (`MonitorSummary::from_samples`): skips
  individual samples where any CPU's `local_dsq_depth` exceeds
  `DSQ_PLAUSIBILITY_CEILING` (10,000) via `sample_looks_valid()`.

- **Threshold evaluation** (`MonitorThresholds::evaluate`): checks all
  samples globally for plausibility. If all `rq_clock` values are
  identical across every CPU and sample, or any sample exceeds the
  plausibility ceiling, the entire report is passed as "not yet
  initialized" — no per-threshold checks run.

## BPF map introspection

The monitor module also provides host-side BPF map discovery and
read/write access via the `GuestMemMapAccessor` (which implements the
`bpf_map::BpfMapAccessor` trait). The host reads and
writes guest BPF maps directly through the physical memory mapping
— no guest cooperation or BPF syscalls are needed.

### GuestMem

`GuestMem` wraps a host pointer to the start of guest DRAM and
provides bounds-checked volatile reads and writes for scalar types
(u8/u32/u64). Byte-slice reads (`read_bytes`) use
`copy_nonoverlapping`. It also implements x86-64 page table walks
(`translate_kva`) for both 4-level and 5-level paging, and
granule-agnostic aarch64 walks (4 KB / 16 KB / 64 KB; level count
derived from TCR_EL1's TG1 + T1SZ fields).

Scalar accesses use volatile semantics because the guest kernel
modifies memory concurrently.

### GuestKernel

`GuestKernel` builds on `GuestMem` by adding kernel symbol
resolution and address translation. It parses the vmlinux ELF
symbol table at construction and resolves paging configuration
(PAGE_OFFSET, CR3, 4-level vs 5-level) from guest memory.
Subsequent reads use cached state.

Three address translation modes are supported:

- **Text/data/bss**: `kva - __START_KERNEL_map`. For statically-linked
  kernel variables (`read_symbol_*`, `write_symbol_*`).
- **Direct mapping**: `kva - PAGE_OFFSET`. For SLAB allocations,
  per-CPU data, physically contiguous memory (`read_direct_*`).
- **Vmalloc/vmap**: Page table walk via CR3. For BPF maps, vmalloc'd
  memory, module text (`read_kva_*`, `write_kva_*`).

### GuestMemMapAccessor

`GuestMemMapAccessor` is the concrete guest-physical-memory accessor:
it resolves BTF offsets for BPF map kernel structures (`struct bpf_map`,
`struct bpf_array`, `struct xa_node`, `struct idr`), borrows a
`GuestKernel` for address translation, and implements the
`bpf_map::BpfMapAccessor` trait that provides map discovery and value
read/write.

`GuestMemMapAccessorOwned` is a convenience wrapper that owns the
`GuestKernel` internally. Use `GuestMemMapAccessor::from_guest_kernel`
when you already have a `GuestKernel`; use
`GuestMemMapAccessorOwned::new` when you want a self-contained accessor.

Map discovery walks the kernel's `map_idr` xarray:

1. Read `map_idr` (BSS symbol, text mapping translation)
2. Walk xa_node tree (SLAB-allocated, direct mapping translation)
3. Read `struct bpf_map` fields. The allocation may be kmalloc'd or
   vmalloc'd depending on size and flags, so the translation uses
   `translate_any_kva` which handles both paths rather than assuming
   either.

`find_map` searches by name suffix (e.g. `".bss"` matches
`"mitosis.bss"`) and returns the first name-matching map of any
type. The sibling `find_array_map` applies the same suffix match
but returns only `BPF_MAP_TYPE_ARRAY` maps — the ARRAY filter is the
intended value-region read/write target. (`value_kva` is populated for
both `BPF_MAP_TYPE_ARRAY` and `BPF_MAP_TYPE_STRUCT_OPS` maps, but the
inline-value read/write path narrows to ARRAY.) Use `maps()` to
enumerate all maps without filtering.

Value access for `BPF_MAP_TYPE_ARRAY` maps reads/writes the inline
`bpf_array.value` flex array at the BTF-resolved offset. The value
region is vmalloc'd, so each byte access goes through the page table
walker to handle page boundaries.

For `BPF_MAP_TYPE_PERCPU_ARRAY` maps, `bpf_array.pptrs[key]` holds
a `__percpu` pointer (at the same union offset as `value`). Adding
`__per_cpu_offset[cpu]` yields the per-CPU KVA in the direct mapping.
`read_percpu_array` returns one `Option<Vec<u8>>` per CPU: `Some`
when the per-CPU PA falls within guest memory, `None` when it does not.

### Program BTF

When a map carries program BTF (`btf_kva != 0`), the accessor loads
the guest's program BTF (`load_program_btf_kva`) so the dump renderer
can resolve the value struct's field types for rendering. There is no
typed `read_field` / `write_field` / `BpfValue` API; value access is by
byte offset (`read_value_*` / `write_value_*`).

### Usage example

Find a scheduler's `.bss` map and write a crash variable:

```rust,ignore
let offsets = BpfMapOffsets::from_vmlinux(vmlinux)?;
let accessor = GuestMemMapAccessor::from_guest_kernel(&kernel, &offsets)?;
let bss = accessor.find_array_map(".bss").expect(".bss map not found");
accessor.write_value_u32(&bss, crash_offset, 1);
```

### BpfMapWrite

`BpfMapWrite` specifies a host-side write to a BPF map during VM
execution. The test runner waits for the scheduler to load (map
becomes discoverable), writes the value, then signals the guest via
the virtio-console RX queue (`SIGNAL_BPF_WRITE_DONE`) to start the
scenario.

`BpfMapWrite::new(map_name_suffix, field, value)` takes a validated
map-name suffix (e.g. `".bss"`), the BPF global variable NAME within
that section (e.g. `"crash"`), and the `u32` to write. Its fields are
crate-private, so it is constructed only through this const constructor
(which const-asserts the suffix format) — direct struct-literal
construction is rejected.

Use with `#[ktstr_test]` via the `bpf_map_write` attribute:

```rust,ignore
const BPF_CRASH: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);

#[ktstr_test(bpf_map_write = BPF_CRASH, expect_err = true)]
fn crash_test(ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
```

The map is discovered by name suffix, and the field's byte offset and
width are resolved from the map's program BTF at write time (which
disambiguates same-suffix maps by picking the one whose BTF names the
field). Only `BPF_MAP_TYPE_ARRAY` maps are supported, and only 4-byte
scalar fields are written; the u32 lands at the field's resolved offset
within the map's value region.

### Prerequisites

- **vmlinux**: Required for ELF symbols and BTF. Must match the guest
  kernel. Symbols include `phys_base` so the runtime KASLR offset can
  be resolved via a page-table walk through the BSP's CR3, breaking
  the chicken-and-egg between text-symbol PA translation and KASLR.

### Cast analysis

BPF maps frequently store kernel pointers (`task_struct *`,
`cgroup *`, …) and arena pointers in `u64` fields because BTF cannot
express a pointer to a per-allocation type. Without intervention the
renderer treats them as integers and the failure dump shows raw
0xffff…ffff values with no further chase.

The cast analyzer (`monitor::cast_analysis::analyze_casts`) closes
that gap. The analyzer is lazy: nothing runs at scheduler-load time.
The first dump that needs cast metadata (any periodic capture,
on-demand snapshot, or stall dump) triggers `LazyCastMap::get_full`
from the dump-dispatch path; subsequent dumps in the same VM hit the
per-VM cache, and a process-wide content-hash cache dedupes across
VMs that share the same scheduler binary. The analyzer pipeline:

1. The host reads the scheduler binary and locates each embedded BPF
   object ELF in its `.bpf.objs` PROGBITS section.
2. Each program section is decoded through
   `cast_analysis::BpfInsn::from_le_bytes` into a flat `&[BpfInsn]`
   slab; relocations against `.bss` / `.data` / `.rodata` annotate
   the corresponding `BPF_LD_IMM64` PCs with their datasec target.
3. `analyze_casts` walks the slab forward, tracking register and
   stack-slot state for each instruction. Two detection paths feed
   the output: the arena pointer path (LDX through a previously
   loaded `u64` field) and the kernel kptr path (STX of a typed
   pointer register into a `u64` field). Function-entry seeding
   from `bpf_func_info` reseeds R1..R5 from the BTF FuncProto so
   typed parameters propagate correctly across subprogram joins.
4. The result is a `CastMap` (`BTreeMap<(source_struct_btf_id,
   field_byte_offset), CastHit>`) cached on the per-VM
   `KtstrVm.cast_map` (a `LazyCastMap` that runs the analyzer on
   first dump and caches the result process-wide by scheduler
   binary content hash). The freeze coordinator threads the cached
   `CastMap` through `DumpContext::cast_map` into every per-map
   render so the renderer can consult it at every dump site.
5. `render_cast_pointer` in `monitor::btf_render` consumes
   `CastHit` via `MemReader::cast_lookup`. When a `u64` field at a
   recorded `(struct, offset)` is rendered, the renderer chases
   the pointer through the address-space-appropriate reader (arena
   vs slab/vmalloc) and tags the result with a `cast_annotation`
   of `"cast→arena"` or `"cast→kernel"` (plus a `(sdt_alloc)`
   suffix when the bridge described below fired). Failure dumps
   show the annotation alongside the resolved struct fields, so
   cast-recovered pointers are visually distinct from BTF-typed
   ones.

The renderer also consults an `sdt_alloc` bridge whenever a chase
target peels to a `BTF_KIND_FWD` forward declaration (typical for
`struct sdt_data __arena *` fields whose body lives in the
sdt_alloc library's BTF rather than the scheduler's program BTF).
The dump-state pre-pass walks each live `scx_allocator` and
populates a `slot_start → ArenaSlotInfo` index — one entry
per live allocator slot, carrying `elem_size`, `header_size`, and
the resolved payload BTF type id — that
`MemReader::resolve_arena_type` (in
`dump::render_map::AccessorMemReader`) range-looks up during the
chase. The lookup finds the slot whose
`[slot_start, slot_start + elem_size)` range contains the chased
address and routes by `offset_in_slot`: a slot-start chase
(`offset == 0`, e.g. the `data` field of `scx_task_map_val`
storing the raw `sdt_alloc()` return) returns the payload type id
with `header_skip = header_size`; a payload-start chase
(`offset == header_size`, e.g. the return of `scx_task_data(p)`
cached in `cached_taskc_raw`) returns the same payload type id
with `header_skip = 0`. The renderer reads `header_skip + btf_size`
bytes from the chased address, slices off the leading
`header_skip` bytes, and renders the payload struct. The
resulting `Ptr` carries a `sdt_alloc`-flavoured annotation:
`"sdt_alloc"` on the BTF-typed `Type::Ptr` arm, and
`"cast→arena (sdt_alloc)"` / `"cast→kernel (sdt_alloc)"` on the
cast-analyzer-driven path. The `sdt_alloc` bridge fires only when
the BTF-only resolve has already exhausted same-name siblings;
false-positive risk on that arm is bounded by the arena-window
range check (`MemReader::resolve_arena_type` returns `None` for
addresses outside every known allocator slot).

A separate cross-BTF Fwd resolution path covers the case where a
`BTF_KIND_FWD` pointee's body lives in a sibling embedded BPF
object's BTF rather than an `sdt_alloc` slot — the typical
multi-`.bpf.objs` shape where one object declares
`struct cgx_target;` (forward) and a sibling object defines
`struct cgx_target { ... }` (full body). The cast-analysis
pre-pass (`vmm::cast_analysis_load::build_fwd_index`) walks every
parsed embedded program BTF and records a
`name -> (btfs index, type_id)` entry for every complete
(`!is_fwd`) `Type::Struct` / `Type::Union`. First-write-wins on
duplicate names: when the same name appears in multiple BTFs the
index keeps the first-seen entry. A named `typedef` over an
anonymous `struct` is also indexed (under its `_t`-stripped
base name, pointing at the anonymous struct's type id), to recover
a body that only carries a name via the typedef alias; types with no
usable name are not indexed. The index is threaded through
`DumpContext::cross_btf_fwd_index` and exposed to the renderer via
`MemReader::cross_btf_resolve_fwd`. When `chase_arena_pointer` /
`render_cast_pointer` peel a chase target through
`peel_modifiers_resolving_fwd` and the local same-BTF sibling
search came up empty, `try_cross_btf_fwd_resolve` consults the
cross-BTF index by the Fwd's name (and aggregate kind — `struct`
vs `union`); a hit returns a `CrossBtfRef { btf, type_id }` and
the chase recursion switches to the resolved sibling BTF for the
pointee render. Cross-BTF resolution does NOT introduce a new
annotation — the body is recovered transparently and the rendered
subtree carries the cast or BTF-typed annotation it would have
had if the same struct lived in the entry BTF. Unlike the
`sdt_alloc` bridge the cross-BTF index is consulted whenever a
Fwd terminal survives the local resolve — there is no
arena-window gate, since the lookup is purely a name-keyed BTF
table and a name miss simply leaves the chase on its existing
"forward declaration; body not in this BTF" skip path.

The analyzer is deliberately conservative: branch joins reset
register and stack state, conflicts drop the offending entry, and
self-stores are rejected. False negatives fall back to raw `u64`
(the prior behavior); false positives would chase garbage and are
avoided. The analysis is unconditional — no test-author
configuration, no opt-in flag — and the freeze coordinator wires
the resulting `CastMap` through every snapshot, periodic capture,
and failure dump.

## Probe pipeline

The probe pipeline captures function arguments and struct fields during
auto-repro. It operates inside the guest VM (not from the host), using
two BPF skeletons that share maps.

### Architecture

```text
crash stack -> extract functions -> BTF resolve -> load skeletons -> poll
                                                         |
                                    kprobe skeleton      |     fentry/fexit skeleton
                                    (kernel entry)       |     (BPF entry + kernel exit)
                                         |               |          |
                                         v               v          v
                                    func_meta_map  <--shared-->  probe_data
                                                         |        (entry + exit fields)
                                              trigger fires (ring buffer)
                                                         |
                                              read probe_data entries
                                                         |
                                              stitch by tptr
                                                         |
                                              format with entry→exit diffs
```

### Kprobe skeleton (`probe.bpf.c`)

Attaches to kernel functions via `attach_kprobe`. The BPF handler:
1. Gets the function IP via `bpf_get_func_ip`
2. Looks up `func_meta` from `func_meta_map` (keyed by IP)
3. Captures 6 raw args from `pt_regs`
4. Dereferences struct fields via BTF-resolved offsets
5. Reads char * string params if configured
6. Stores result in `probe_data` (keyed by `(func_ip, task_ptr)`)

The trigger fires via `tp_btf/sched_ext_exit` (inside
`scx_claim_exit()`) and sends an `EVENT_TRIGGER` via ring buffer
with the current task pointer and kernel stack.

### Fentry/fexit skeleton (`fentry_probe.bpf.c`)

Handles both BPF struct_ops callbacks and kernel function exit
capture. Each skeleton instance exposes exactly 4 indexed fentry
slots (`ktstr_fentry_0..3`) and 4 fexit slots (`ktstr_fexit_0..3`),
attached via `set_attach_target` before load. The Phase B polling
loop instantiates additional skeletons as needed once the scheduler
has loaded and the fentry/fexit attach targets become available.
Shares `probe_data` and `func_meta_map` with the kprobe skeleton via
`reuse_fd`.

**Phase A / Phase B split.** The kprobe skeleton + trigger fexit
attach during Phase A (before scheduler load). The fentry/fexit
skeletons attach during Phase B — a 100 ms polling loop that runs
AFTER the scheduler binary loads and the callback targets become
attachable. Operators debugging attach failures should know that the
fentry/fexit half always lags scheduler load.

A per-slot `is_kernel` rodata flag controls argument access:
- **BPF callbacks** (`is_kernel=0`): `ctx[0]` is a void pointer to
  the real callback arguments. The handler dereferences through it.
  Uses sentinel IPs (`func_idx | (1<<63)`) in `func_meta_map`.
- **Kernel functions** (`is_kernel=1`): args are directly in
  `ctx[0..5]`. Uses `bpf_get_func_ip(ctx)` for the real IP,
  matching the kprobe entry handler's key.

Fexit handlers look up the existing `probe_data` entry (written by
fentry or kprobe at function entry) and re-read struct fields into
`exit_fields`. This captures post-mutation state for paired display.

### BTF resolution

Two BTF sources:

- **vmlinux BTF** (`btf-rs`): resolves kernel struct offsets. Types in
  `STRUCT_FIELDS` (task_struct, rq, scx_dispatch_q, etc.) use curated
  field lists with chained pointer dereferences (e.g.
  `->cpus_ptr->bits[0]`). Other struct pointer params get scalar, enum,
  and cpumask pointer fields auto-discovered from vmlinux BTF.

- **Program BTF** (`libbpf-rs`): resolves BPF-local struct offsets for
  types not in vmlinux (e.g. scheduler-defined `task_ctx`).
  Auto-discovers scalar, enum, and cpumask pointer fields.

Callback signatures are resolved by:
1. `____name` inner function in program BTF (typed params)
2. `sched_ext_ops` member in vmlinux BTF (fallback)
3. Wrapper function (void *ctx, no useful params)

### Field decoding

The output formatter decodes field values based on their key name:
- `dsq_id` -> `SCX_DSQ_INVALID`, `SCX_DSQ_GLOBAL`, `SCX_DSQ_LOCAL`, `SCX_DSQ_BYPASS`, `SCX_DSQ_LOCAL_ON|{cpu}`, `BUILTIN({v})`, `DSQ(0x{hex})`
- `cpumask_0..3` -> coalesced into one `cpus_ptr` field rendered as
  `0x{hex}({cpu-list})` — the masked hex of the cpumask words
  (high-order word first; multi-word masks join with `_` between
  64-bit chunks) followed by the run-length-collapsed CPU range
  list (e.g. `0xf(0-3)`, `0x0000000000000001_00000000000000ff(0-7,64)`)
- `enq_flags` -> `WAKEUP|HEAD|PREEMPT`
- `exit_kind` -> `ERROR`, `ERROR_BPF`, `ERROR_STALL`, etc.
- `scx_flags` -> `QUEUED|ENABLED`
- `sticky_cpu` -> `-1` for 0xffffffff

### Event stitching

After the trigger fires, all `probe_data` entries are read, matched
to functions by IP, then filtered to a single task's scheduling
journey:

1. Read the task_struct pointer from the trigger event's
   `bpf_get_current_task()` value (`args[0]`)
2. For functions with a task_struct parameter: keep events where
   `args[param_idx] == tptr`
3. For functions without a task_struct parameter: keep events where
   `task_ptr == tptr` (matched via `bpf_get_current_task()` at
   probe time)

Events are sorted by timestamp for chronological output.
