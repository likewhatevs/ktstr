# Snapshots

Was the scheduler's per-task state right in the middle of the run?
A **snapshot** answers that: the freeze coordinator pauses every
vCPU long enough to walk the kernel's BPF maps, BTF-render every
captured value, and store the result under a name you choose. Test
code reads it back through a typed accessor whose errors carry the
available alternatives — a typo'd map or field name tells you what
was actually there.

Three capture triggers share this machinery:

| Capture | Trigger | The question it answers |
|---|---|---|
| `Op::capture_snapshot` (this page) | a chosen point in the scenario | what does state look like *right now*? |
| [Watch Snapshots](watch-snapshots.md) | a kernel write to a named symbol | what was state at the instant the kernel touched X? |
| [Periodic Capture](periodic-capture.md) | evenly spaced boundaries | how does state evolve across the run? |

In a `#[ktstr_test]` scenario the pipeline is wired automatically:
the op sends a request from the guest to the host coordinator, which
freezes, captures, and stores the report on the host-side
`SnapshotBridge`. The test reads captures after the VM exits, in a
`post_vm` callback. No bridge setup is needed — manual wiring exists
only for [host-side unit tests](#harness-internals-manual-bridge-wiring).

## Capturing and reading

```rust,ignore
use ktstr::prelude::*;

fn inspect_after_spawn(result: &VmResult) -> anyhow::Result<()> {
    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let entry = drained
        .iter()
        .find(|e| e.tag == "after_spawn")
        .ok_or_else(|| anyhow::anyhow!("snapshot 'after_spawn' missing"))?;
    let snap = Snapshot::new(&entry.report);

    let nr_dispatched = snap.var("nr_dispatched").as_u64()?;
    anyhow::ensure!(nr_dispatched > 0, "scheduler never dispatched");
    Ok(())
}

#[ktstr_test(scheduler = MY_SCHED, post_vm = inspect_after_spawn)]
fn snapshot_then_inspect(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("workers")].into(),
        ops: vec![Op::capture_snapshot("after_spawn")],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
```

A scenario may issue any number of `Op::capture_snapshot` ops with
distinct names; reusing a name overwrites the prior capture (with a
warning). If the capture pipeline is unavailable, the op fails
loudly — a snapshot that silently didn't happen would let
assertions that depend on it pass vacuously.

## The accessor surface

`Snapshot::new(report)` builds a borrowed view; accessors walk the
report in place.

### Maps and globals

```rust,ignore
let map = snap.map("scx_per_task")?;         // a captured map by name
let nr = snap.var("nr_cpus_onln").as_u64()?; // a top-level global
```

`var(name)` searches every `*.bss` / `*.data` / `*.rodata`
global-section map for a top-level member. When several schedulers'
sections carry the same name, `var` first tries to resolve the
active scheduler's copy automatically; `live_var(name)` opts into
that active-scheduler filter explicitly, and `map(name)` addresses
one scheduler's section directly. Note `var` does not split dotted
paths — to walk into a struct global, chain:
`snap.var("ctx").get("weight")`.

### Entries inside a map

```rust,ignore
let first   = map.at(0);                                               // by index
let busy    = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 1234);
let busiest = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
let active  = map.filter(|e| e.get("runtime_ns").as_u64().unwrap_or(0) > 0);
```

Per-CPU maps (`BPF_MAP_TYPE_PERCPU_*`) need narrowing before
reading: `map.cpu(1).at(0)`. Calling `get` on a per-CPU entry
without `.cpu(N)` first is an error, not a silent first-slot read.

### Dotted paths and terminal reads

`get(path)` walks struct members along a dotted path
(`entry.get("ctx.weight")` ≡ `entry.get("ctx").get("weight")`),
transparently following pointer dereferences up to 16 hops — you
write the path the BTF suggests, indirection is invisible. `get("")`
returns the current value, for terminal reads on scalar per-CPU
slots.

| Method | Returns | Accepts |
|---|---|---|
| `as_u64()` | `u64` | `Uint`, non-negative `Int`/`Enum`, `Bool`, `Char`, `Ptr` (raw pointer value) |
| `as_i64()` | `i64` | `Int`, `Uint` ≤ `i64::MAX`, `Bool`, `Char`, `Enum` |
| `as_bool()` | `bool` | `Bool`; non-zero scalar is `true` |
| `as_f64()` | `f64` | `Float`, `Int`, `Uint`, `Enum` |
| `as_str()` | `&str` | `Enum` with a resolved variant name |
| `raw()` | `Option<&RenderedValue>` | the underlying rendered value |

## Errors carry the fix

Every accessor returns `Result<_, SnapshotError>`, and each variant
carries what you need to correct the call site without re-running
the test. The rendered messages (quoted from the `Display` impl):

- `Snapshot::map` miss —
  `snapshot has no map '{requested}' (captured maps: {available:?})`
- `Snapshot::var` miss —
  `snapshot has no global variable '{requested}' in any
  *.bss/*.data/*.rodata map (available globals: {available:?})`
- ambiguous global —
  `snapshot global '{requested}' is ambiguous (found in
  {found_in:?}); use Snapshot::active().var(name) (or the shorthand
  Snapshot::live_var(name)) to pick the active scheduler's copy
  automatically, or Snapshot::map(name) to address a specific
  scheduler's bss explicitly`
- path-walk miss —
  `path '{requested}': component '{component}' (after walking
  '{walked}') not found (members at this depth: {available:?})`
- wrong terminal type —
  `path '{requested}': cannot read as {expected} — actual rendered
  variant is {actual}`
- predicate miss (`find` / `max_by`) — `map '{map}': {op} matched
  none of {len} entries (first {sampled}: {available_keys:?})`; an
  empty map instead renders `map '{map}': {op} matched no entries
  (map is empty)`, distinguishing it from a populated map whose every
  entry the predicate rejected. When every sampled key renders as raw
  hex (no BTF for the key type at capture time), the message appends
  a hint naming `CONFIG_DEBUG_INFO_BTF=y` as the fix.

Two variants matter for series-based assertions and are routed
specially by the [temporal patterns](temporal-assertions.md):
`PlaceholderSample` (the freeze rendezvous timed out, so the report
carries no real data — skipped, never counted as zero progress) and
`MissingStats` (the per-sample scx_stats request failed or no stats
client was wired — distinct from an in-JSON path miss so the
assertion site can branch on the cause).

`SnapshotError` implements `std::error::Error`, so it composes with
`?` and `anyhow`.

## Cast-recovered pointers

Schedulers stash kernel and arena pointers in fields whose BTF says
`u64`, because BTF cannot express a pointer to a per-allocation
type. The host-side [cast analyzer](../architecture/monitor.md)
recovers the real target type from the scheduler's instruction
stream, and the renderer chases the pointer into the right address
space. For the test author:

- `as_u64()` still returns the raw pointer value — existing tests
  keep working.
- Dotted-path walks follow the recovered chase transparently;
  nested fields appear under the same path a natively-typed pointer
  would give.
- Rendered dumps annotate recovered pointers so you can tell them
  from BTF-typed ones — no extra calls needed to consume them.

This is what the annotations look like in a real failure dump
(scx-ktstr's `.bss`, from the run on the
[macro reference page](ktstr-test-macro.md#what-a-failing-gate-looks-like)):

<!-- captured: cargo ktstr test --kernel 7.0 -- -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
map bpf_bpf.bss (type=array, value_size=448, max_entries=1)
.bss:
  scx_arena_verify_once=true   ktstr_alloc_count=76   nr_dispatched=907
  nr_enqueued=495              nr_select_cpu=372      stats_magic=6004496034161779060
...
  scx_task_allocator scx_allocator:
...
    root 0x100000006000 → sdt_desc:
      nr_free=512
      chunk 0x100000007000 (sdt_alloc) → ktstr_arena_ctx{}
  ktstr_bss_arena_holder ktstr_bss_arena_holder:
    bss_plain_counter=76
    arena_target 0x10000000aa80 (cast→arena) [chase: arena chase: STX-flow path tagged slot as Arena with deferred resolve; bridge had no entry for 0x10000000aa80]
```

`(cast→arena)` / `(cast→kernel)` mark analyzer-recovered pointers;
`(sdt_alloc)` marks a forward-declared arena type resolved through
the allocator bridge. The full annotation taxonomy lives in
[Monitor](../architecture/monitor.md).

## Composing reads with writes

Snapshots are the read half of host↔guest interaction. The write
half is the `#[ktstr_test]` attribute `bpf_map_write = CONST` — a
one-shot host-side poke at scheduler-load time:

```rust,ignore
use ktstr::prelude::*;

const TRIGGER_FAULT: BpfMapWrite = BpfMapWrite::new(".bss", "crash", 1);
// (map_name_suffix, BPF global variable name, u32 value). The
// variable's byte offset is resolved from the map's program BTF at
// write time.

#[ktstr_test(scheduler = MY_SCHED, bpf_map_write = TRIGGER_FAULT, expect_err = true)]
fn fault_then_inspect(ctx: &Ctx) -> Result<AssertResult> {
    // The host writes 1 into the scheduler's `crash` global before
    // workers start; the scheduler reads the flag and reacts.
    /* Op::capture_snapshot + post_vm read as above */
    Ok(AssertResult::pass())
}
```

The write waits for the scheduler's map to appear, resolves the
named variable to an offset via BTF, writes the value, and signals
completion to the guest before workers spawn. Only
`BPF_MAP_TYPE_ARRAY` maps are supported. A read+write test then
composes naturally: seed a flag with `bpf_map_write`, run the
scenario, capture with `Op::capture_snapshot`, assert on the
scheduler's reaction through the `Snapshot` accessors.

There is no op for runtime writes — mid-scenario mutation belongs to
interfaces the scheduler itself exports (sysfs, debugfs, a BPF map
command interface) driven from a workload process.

## Harness internals: manual bridge wiring

> [!WARNING]
> Do not install a thread-local bridge inside a `#[ktstr_test]`
> scenario that boots a VM — the host coordinator owns the bridge
> there, and a scenario-local one would shadow it. Read captures in
> `post_vm` from `VmResult::snapshot_bridge` instead.

Host-side unit tests that exercise the executor without booting a
guest install a fixture bridge:

```rust,ignore
let cb: CaptureCallback = std::sync::Arc::new(|_name: &str| {
    Some(FailureDumpReport::default())   // hand-crafted report
});
let bridge = SnapshotBridge::new(cb);
let handle = bridge.clone();
let _guard = bridge.set_thread_local();
// ... execute_steps(...) ... then handle.drain() ...
```

`set_thread_local` returns a guard that restores the prior bridge on
drop; bind it to `_guard`, not `let _ =` — the latter drops the
guard immediately and clears the bridge before any op runs.
`tests/snapshot_e2e.rs` exercises this pattern end-to-end.
