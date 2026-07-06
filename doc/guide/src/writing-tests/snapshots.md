# Snapshots and Live Capture

Was the scheduler's per-task state right in the middle of the run?
A **snapshot** answers that: the freeze coordinator pauses every
vCPU long enough to walk the kernel's BPF maps, BTF-render every
captured value, and store the result under a name you choose. Test
code reads it back through a typed accessor whose errors carry the
available alternatives — a typo'd map or field name tells you what
was actually there.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>One-shot</strong><p><code>Op::capture_snapshot</code> freezes at a scenario point and stores a named report.</p></div>
<div class="kt-doc-card"><strong>Watch</strong><p>Watch snapshots capture state when a kernel symbol is written.</p></div>
<div class="kt-doc-card"><strong>Periodic</strong><p>Periodic capture turns repeated snapshots into time-series assertions.</p></div>
</div>

The same freeze-and-render pipeline backs three triggers; they
differ only in what fires the capture:

| Capture | Trigger | The question it answers |
|---|---|---|
| `Op::capture_snapshot` (on demand) | a chosen point in the scenario | what does state look like *right now*? |
| [Watch snapshots](#watch-snapshots) | a kernel write to a named symbol | what was state at the instant the kernel touched X? |
| [Periodic capture](#periodic-capture) | evenly spaced boundaries | how does state evolve across the run? |

In a `#[ktstr_test]` scenario the pipeline is wired automatically:
the trigger sends a request from the guest to the host coordinator,
which freezes, captures, and stores the report on the host-side
`SnapshotBridge`. The test reads captures after the VM exits, in a
`post_vm` callback — every trigger lands on the same bridge and is
read back through the same accessors. No bridge setup is needed —
manual wiring exists only for
[host-side unit tests](#harness-internals-manual-bridge-wiring).

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
report in place. Every capture kind — on-demand, watch, periodic —
produces the same `Snapshot`, so the surface below reads any of them;
the same type also appears per-sample inside a `SampleSeries`, where
[projections](temporal-assertions.md) apply these accessors across
every capture at once.

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

Snapshots are the read half of host↔guest interaction. The write half
is the `#[ktstr_test]` attribute
[`bpf_map_write = CONST`](ktstr-test-macro.md#bpf_map_write) — a
one-shot host-side poke at a scheduler global at load time, before
workers spawn. A read+write test composes naturally: seed a flag with
`bpf_map_write`, run the scenario, capture with `Op::capture_snapshot`,
and assert on the scheduler's reaction through the `Snapshot`
accessors.

There is no op for runtime writes — mid-scenario mutation belongs to
interfaces the scheduler itself exports (sysfs, debugfs, a BPF map
command interface) driven from a workload process.

## Watch snapshots — live streaming variant {#watch-snapshots}

An on-demand capture fires where *your* scenario says so. A **watch
snapshot** fires where the *kernel* does:
`Op::watch_snapshot("symbol")` arms a hardware data-write watchpoint
on a named kernel symbol, and every guest write to it triggers a full
snapshot capture, tagged with the symbol name. That answers "what was
state the instant the kernel touched X" — a state field flipping, a
counter the scheduler bumps on a specific event.

Watch snapshots are supported on x86_64 and aarch64 KVM hosts; each
architecture's KVM plumbing maps the slots onto its native
hardware-watchpoint facility.

```rust,ignore
use ktstr::prelude::*;

fn read_watch_fires(result: &VmResult) -> anyhow::Result<()> {
    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    // Each fire is stored under the symbol name as its tag.
    let fires = drained.iter().filter(|e| e.tag == "scx_watchdog_timestamp");
    anyhow::ensure!(fires.count() > 0, "watchpoint never fired");
    Ok(())
}

#[ktstr_test(scheduler = MY_SCHED, post_vm = read_watch_fires)]
fn watch_watchdog_writes(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(vec![ctx.cgroup_def("workers")], HoldSpec::FULL)
            .set_ops(vec![Op::watch_snapshot("scx_watchdog_timestamp")]),
    ];
    execute_steps(ctx, steps)
}
```

The op registers the symbol with the host coordinator, which resolves
the address from the vmlinux ELF, arms a free hardware watchpoint slot
via `KVM_SET_GUEST_DEBUG`, and stores one capture per fire on the
host-side bridge. Read the fires in `post_vm` through the same
[accessors](#the-accessor-surface) every capture kind shares. When a
sidecar dump path is configured for the run, each fire's report is
also mirrored to a tagged JSON file for post-hoc inspection.

### Choosing a symbol

Production resolution is a verbatim, byte-for-byte match against the
vmlinux ELF symbol table — no prefix stripping, no BTF lookup, no
kallsyms walk. Use exactly the name `nm` prints:

```sh
nm vmlinux | grep -w scx_watchdog_timestamp
```

A string that matches nothing fails the step with
`symbol '<name>' not found in vmlinux symtab` (typo, symbol stripped
from the build, or a non-ELF kernel image).

> [!WARNING]
> High-frequency symbols soft-lock the guest. Watching a symbol the
> kernel writes every jiffy (e.g. `jiffies_64` at `HZ=1000`) fires
> 1000+ captures per second, and each capture freezes all vCPUs for
> the full dump pipeline. The guest spends almost all of its wall
> time paused — schedulers stall, watchdogs fire, and the test
> wedges before any meaningful work runs. Pick symbols the kernel
> writes at scenario-relevant cadence: a state field, a per-event
> counter.

### Three watches per scenario

The cap is 3, tied to the hardware watchpoint slots KVM exposes:
slot 0 is permanently reserved for the `*scx_root->exit_kind`
trigger that drives the failure-dump pipeline on `SCX_EXIT_ERROR`
(it always runs, whether or not a scenario declares watches), and
the remaining three user slots are yours. A fourth
`Op::watch_snapshot` fails the step with the pinned message:

```text
Op::WatchSnapshot cap exceeded: scenario already registered 3
watchpoints (3 user watchpoint slots occupied; slot 0 reserved for
the error-class exit_kind trigger). Drop a watch or use
Op::CaptureSnapshot for a time-driven capture instead.
```

A failed registration — cap exceeded, resolution failure, callback
error — does not consume a slot; the bridge rolls the count back so
the scenario can retry with a different symbol.

### Failure modes

Registration is the single point where the production pipeline can
fail. The callback returns an error when:

- The symbol does not match any vmlinux ELF symtab entry.
- The resolved address is not 4-byte aligned (the 4-byte watch
  length requires `addr & 0x3 == 0` on every supported
  architecture).
- All three user watchpoint slots are already allocated.
- `KVM_SET_GUEST_DEBUG` rejected the arm (host kernel limitation).

When registration fails, the executor bails the step immediately
with the symbol and the reason. Silent degradation is deliberately
avoided — a watch that never fires would look identical to a
healthy passing run, and the test author would never notice the
captures were missing.

## Periodic capture {#periodic-capture}

A single snapshot proves state was right once; scheduler bugs are
usually about how state *evolves* — a counter that stops advancing,
utilization that drifts after warmup. **Periodic capture** samples
guest BPF state on a cadence across the workload window, driven
entirely by the host: no scenario-code changes, no capture calls in
the test body. The result is a time-ordered series of samples that
feeds the [temporal assertion](temporal-assertions.md) patterns.

### Enabling it

Set `num_snapshots = N` on the test; `0` (the default) disables
periodic capture entirely.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(num_snapshots = 3, duration_s = 10)]
fn paced_capture(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

### When boundaries fire

The window is the **10%–90% slice** of the workload duration,
anchored at the moment the scenario actually starts — VM boot and
BPF verifier time do not eat the budget. The 10% buffers at each
end keep samples off ramp-up and ramp-down transients. The
remaining 80% divides into `N + 1` equal intervals, yielding `N`
interior boundaries at `0.1·d + (i+1)·0.8·d/(N+1)`. For a 10 s
workload, `num_snapshots = 3` captures at scenario start +
{3 s, 5 s, 7 s}.

The boundary clock is workload time, not wall-clock: a scenario
pause shifts every un-fired boundary by the pause duration.

Two validation rules, enforced when the entry is built:

- **Minimum spacing** — `0.8 · duration / (N + 1) >= 100 ms`.
  Boundaries closer than that would fire back-to-back with no
  workload progress between them. Reduce `num_snapshots` or extend
  `duration_s`.
- **Bridge cap** — `num_snapshots` cannot exceed 64
  (`MAX_STORED_SNAPSHOTS`). Validation rejects higher values rather
  than silently evicting the earliest samples.

### What a capture costs

Each boundary runs the same freeze pipeline as an on-demand capture.
On a healthy guest that is tens of milliseconds (10–100 ms steady
state; cold-cache or large guest-memory walks push higher). The host
watchdog deadline is extended by each freeze, so periodic captures do
not eat the workload's wall-clock budget — but they do briefly stop
the guest, which is why the spacing floor exists.

### Tags and best-effort delivery

Each capture lands on the host `SnapshotBridge` under
`periodic_NNN` (`periodic_000`, `periodic_001`, …), coexisting with
on-demand and watchpoint tags on the same bridge — filter with
`SampleSeries::periodic_only()` before asserting.

Delivery is best-effort: an early VM exit, rendezvous timeout, or
watchdog deadline can cut the sequence short, and the run loop
abandons the remainder after 2 consecutive rendezvous timeouts so a
sustained host overload does not pile up placeholder samples. Under
KASLR (the default), a boundary that would fire before the guest's
address slide is published is deferred, not dropped — it fires on
the next loop iteration. Assert a lower bound on coverage, not
equality:

```rust,ignore
fn check_coverage(result: &VmResult) -> Result<()> {
    anyhow::ensure!(result.periodic_target == 3);
    anyhow::ensure!(
        result.periodic_fired >= 2,
        "too few periodic samples ({}/{})",
        result.periodic_fired,
        result.periodic_target,
    );
    Ok(())
}
```

`periodic_target` mirrors the configured `num_snapshots`;
`periodic_fired` counts boundaries actually serviced (including
rendezvous-timeout placeholders). When `post_vm` is omitted on a
periodic-configured test, the macro installs a default callback
asserting at least one boundary fired with real BPF state.

### Draining the bridge

The assertion pipeline runs on the host after `vm.run()` returns —
inside a `post_vm` callback. The recommended path is
`drain_ordered_with_stats` fed into
`SampleSeries::from_drained_typed`, which preserves insertion order,
per-sample stats results, and timestamps:

```rust,ignore
use ktstr::prelude::*;

fn post_vm(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained_typed(
        result.snapshot_bridge.drain_ordered_with_stats(),
        result.monitor.clone(),
    )
    .periodic_only();

    anyhow::ensure!(
        !series.is_empty(),
        "no periodic samples — coordinator never fired",
    );

    // ... project a field and feed a temporal pattern ...
    Ok(())
}
```

Wire it in with `#[ktstr_test(num_snapshots = 3, post_vm = post_vm)]`.
Each drained entry carries the tag, the captured report, the typed
per-sample stats result (`Err(MissingStatsReason)` when the stats
request failed or no scheduler stats client was wired), a
pause-adjusted `elapsed_ms` timestamp, the scheduled
`boundary_offset_ms`, and the scenario phase stamp (`step_index`).
The other drain variants drop metadata the temporal pipeline needs —
see the
[`SnapshotBridge` rustdoc](https://ktstr.dev/rustdoc/ktstr/scenario/snapshot/struct.SnapshotBridge.html)
if you need them.

Then assert in two stages: build the series (drain, `periodic_only()`),
then project a column and pick a pattern — `nondecreasing` for
monotonic counters, `steady_within` for utilization-style metrics that
should hold once warmup ends, `converges_to` for "stabilizes near a
target by a deadline". [Projections and Temporal Assertions](temporal-assertions.md)
owns the sample anatomy, the full pattern surface, and the projection
helpers; [Errors carry the fix](#errors-carry-the-fix) above owns the
per-sample error routing (`PlaceholderSample`, `MissingStats`).

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

Watch ops need a second callback. A bridge built with only
`SnapshotBridge::new(cb)` rejects every `Op::watch_snapshot` with an
error naming the missing wiring; add the register hook:

```rust,ignore
let reg: WatchRegisterCallback = std::sync::Arc::new(|symbol: &str| {
    println!("would arm watchpoint on {symbol}");
    Ok(())
});
let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
let _guard = bridge.set_thread_local();
```
