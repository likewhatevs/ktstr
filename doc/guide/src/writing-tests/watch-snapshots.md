# Watch Snapshots

What did the scheduler look like at the exact instant the kernel
wrote a specific variable? `Op::watch_snapshot("symbol")` arms a
hardware data-write watchpoint on a named kernel symbol; every guest
write to it triggers a full snapshot capture, tagged with the symbol
name. Where [`Op::capture_snapshot`](snapshots.md) answers "what
does state look like at this point in *my* scenario",
`Op::watch_snapshot` answers "what was state when the *kernel* did
X".

Watch snapshots are supported on x86_64 and aarch64 KVM hosts; each
architecture's KVM plumbing maps the slots onto its native
hardware-watchpoint facility.

## Issuing a watch

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

In a VM-booting `#[ktstr_test]`, the wiring is automatic: the op
registers the symbol with the host coordinator, which resolves the
address from the vmlinux ELF, arms a free hardware watchpoint slot
via `KVM_SET_GUEST_DEBUG`, and stores one capture per fire on the
host-side bridge. Read the captures in `post_vm` through the same
[`Snapshot`](snapshots.md#the-accessor-surface) accessors every
capture kind shares. When a sidecar dump path is configured for the
run, each fire's report is also mirrored to a tagged JSON file for
post-hoc inspection.

## Choosing a symbol

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

## Three watches per scenario

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

## Failure modes

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

## Host-side unit tests

Outside a VM, a watch-capable fixture bridge needs both callbacks —
a bridge built with only `SnapshotBridge::new(cb)` rejects every
`Op::watch_snapshot` with an error naming the missing wiring:

```rust,ignore
let cb: CaptureCallback = std::sync::Arc::new(|_name| {
    Some(FailureDumpReport::default())
});
let reg: WatchRegisterCallback = std::sync::Arc::new(|symbol: &str| {
    println!("would arm watchpoint on {symbol}");
    Ok(())
});
let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
let _guard = bridge.set_thread_local();
```

Do not install a thread-local bridge in a VM-booting scenario — see
the warning in
[Snapshots](snapshots.md#harness-internals-manual-bridge-wiring).
