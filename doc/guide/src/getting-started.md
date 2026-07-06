# Getting Started

Every `#[ktstr_test]` boots a real Linux kernel in a KVM microVM with
the CPU topology the test declares, runs your workload inside it, and
checks the scheduler's behavior from the host. This page takes you
from nothing to a green run.

<div class="kt-steps">
<div class="kt-step" data-step="1"><strong>Install the tools</strong><p><code>cargo-nextest</code> runs the suite; <code>ktstr</code> installs both <code>cargo-ktstr</code> and the standalone host CLI.</p></div>
<div class="kt-step" data-step="2"><strong>Build or pick a kernel</strong><p>Use a released version, local source tree, or cached build. The guest kernel needs sched_ext.</p></div>
<div class="kt-step" data-step="3"><strong>Write one test</strong><p>Start under EEVDF, then add <code>scheduler = MY_SCHED</code> when your scheduler binary is ready.</p></div>
</div>

## Zero to green

```sh
cargo install --locked cargo-nextest
cargo install --locked ktstr
cargo ktstr kernel build --kernel 7.0
$EDITOR tests/sched_test.rs
cargo ktstr test --kernel 7.0
```

Both installs are required: `cargo ktstr test` delegates to nextest,
and the `ktstr` package installs `cargo-ktstr` (the cargo plugin
behind every command in this guide) plus the standalone `ktstr` host
CLI. The kernel build is a real `make -j$(nproc)` kernel build —
plan for that once; later runs reuse the cache. On a cached kernel,
the run shown [below](#run-it) took about 35 seconds end to end.

The shortest useful test file looks like this:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn smoke(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

That smoke test runs under the kernel default scheduler. It is still a
real VM run, so it proves KVM, kernel resolution, initramfs creation,
guest boot, workers, and result transport before you add a custom
scheduler.

## Prerequisites

**Linux only (x86_64, aarch64).** ktstr boots KVM virtual machines; it
does not build or run on other platforms.

You need:

- KVM access (`/dev/kvm`) — see [Troubleshooting](troubleshooting.md)
  if it's missing or unreadable
- Rust ≥ 1.94.1 (the crate's MSRV)
- clang, pkg-config, make, gcc, and autotools (autoconf, autopoint,
  flex, bison, gawk) — BPF skeletons and the vendored
  libbpf/libelf/zlib build
- BTF (`/sys/kernel/btf/vmlinux`) — present by default on most
  distros
- Internet access on first build (downloads busybox source; kernel
  builds download tarballs from kernel.org)

The host kernel only needs KVM. The *guest* kernel — the one your
tests boot — needs sched_ext, which landed in 6.12; the next section
builds one.

Install distro packages once:

```sh
# Ubuntu/Debian
sudo apt install clang pkg-config make gcc autoconf autopoint flex bison gawk
# Fedora
sudo dnf install clang pkgconf make gcc autoconf gettext-devel flex bison gawk
```

## Add the dependency

```toml
[dev-dependencies]
ktstr = "=0.24.0"
```

ktstr is pre-release: pin the exact patch version and keep the
installed `cargo-ktstr` on the same one — minor bumps may break the
test-facing API. To keep ktstr out of a scheduler crate's normal
builds, gate it behind a feature instead — see
[Test a New Scheduler](recipes/test-new-scheduler.md).

Install the matching CLI when pinning a specific crate version:

```sh
cargo install --locked --bin cargo-ktstr ktstr@0.24.0
```

## Build a kernel

`cargo ktstr kernel build` downloads a kernel tarball from
kernel.org, applies the embedded `ktstr.kconfig` fragment (sched_ext,
BPF, kprobes, minimal boot), builds it, and caches the result:

```sh
cargo ktstr kernel build                    # latest stable series with >= 8 point releases
cargo ktstr kernel build --kernel 7.0       # highest 7.0.x release
cargo ktstr kernel build --kernel 6.14.2    # exact version
cargo ktstr kernel build --kernel ../linux  # local source tree
```

Choose the kernel spec that matches your workflow:

| Spec | Use it when |
|---|---|
| `--kernel 7.0` | You want the newest cached or buildable 7.0.x release. |
| `--kernel 6.14.2` | You need an exact released version. |
| `--kernel ../linux` | You are testing a local kernel tree. |
| no `--kernel` | You want ktstr's discovery chain to choose. |

The bare `kernel build` form skips series with fewer than 8 maintenance
releases; brand-new majors tend to hit build issues on older toolchains.
Name a version explicitly to override. `cargo ktstr kernel list` shows
the cache and `cargo ktstr kernel clean --keep 3` prunes it. You can
also skip this step entirely: `cargo ktstr test --kernel 7.0` builds
and caches on first use.

## Write a test

One mental model before the first example: your test function runs
**inside the VM**, as the guest's init process. `execute_defs` and
friends create real cgroups and spawn real workers; `ctx` hands you
the guest topology (`ctx.topo`) and cgroup management
(`ctx.cgroups`).

Create a file in your crate's `tests/` directory (e.g.
`tests/sched_test.rs`). The simplest test runs a canned scenario:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]  // llcs = last-level caches
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    // Canned scenario: two cgroups of CPU spinners, default duration.
    scenarios::steady(ctx)
}
```

No `scheduler` attribute means the test runs under the kernel's
default EEVDF scheduler (see [Overview](overview.md)) — a useful
baseline before pointing at your own.

When the canned scenarios stop being enough, declare your own
cgroups, workloads, and cpusets with `CgroupDef` — the
[Tutorial](tutorial.md) builds that up one step at a time, and
[Writing Tests](writing-tests.md) is the reference.

## Point it at a sched_ext scheduler

Declare your scheduler once and reference it from any test:

```rust,ignore
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_mysched",   // your scheduler's binary name
});

#[ktstr_test(scheduler = MY_SCHED, llcs = 2, cores = 2, threads = 1)]
fn my_sched_steady(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

The binary is resolved on the host — `target/{debug,release}/`, the
test binary's directory, or a `KTSTR_SCHEDULER=/path` override — and
packed into the VM's initramfs. Full field reference:
[Scheduler Definitions](writing-tests/scheduler-definitions.md);
walkthrough: [Test a New Scheduler](recipes/test-new-scheduler.md).

## Run it

```sh
cargo ktstr test --kernel 7.0                          # everything
cargo ktstr test --kernel 7.0 -- -E 'test(my_test)'    # one test (nextest filter)
```

`cargo ktstr test` resolves the kernel — an explicit `--kernel`
version, path, or cache key, or, without the flag, a discovery chain
through environment variables, the kernel cache, and host kernels —
then wraps `cargo nextest run`. The full chain and flag grammar live
in [cargo ktstr](running-tests/cargo-ktstr.md).

Here is a real run against a local kernel tree (transcript captured
from ktstr's own suite — your run shows `ktstr/my_test` on the PASS
line instead):

<!-- captured: cargo ktstr test --kernel ../linux --no-perf-mode -- --features integration -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)' | ktstr 0.24.0-dev | kernel ../linux (b4dc42d2, sched_ext-for-7.2) | verified by independent rerun -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr test --kernel ../linux --no-perf-mode -- -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)'</span></div>

<pre><span class="t-dim">cargo ktstr: resolved kernel "path_linux_b5562c"
cargo ktstr: BTF type anchor at target/ktstr_btf_anchor.h</span>
...
────────────
 Nextest run ID d79da4e4-df49-4b40-a415-c1dff2739ce6 with nextest profile: default
    Starting 1 test across 120 binaries (13597 tests skipped)
        <span class="t-grn">PASS [   8.601s] (1/1) ktstr::scx_cleanup_test ktstr/scx_empty_run_exits_under_watchdog</span>
────────────
     <span class="t-b">Summary [   8.642s] 1 test run: 1 passed, 13597 skipped</span>

cargo ktstr: test outputs
...
    (1 stats sidecar(s), 0 wprof trace(s) written this run)</pre></div>

Reading it:

- The first two lines are kernel resolution and BTF setup:
  `--kernel ../linux` used a local source tree and found the matching
  image in ktstr's cache — no rebuild.
- Test names have the shape `crate::binary ktstr/test_name`; the
  `ktstr/` prefix marks the base variant, and the same test also
  generates `gauntlet/` topology variants, skipped by default (see
  [Running Tests](running-tests.md)). The 34 s covers everything:
  VM boot, scenario, teardown, evaluation.
- Every run writes a stats sidecar per test under
  `target/ktstr/{kernel}-{commit}/` — the raw material for
  regression gates ([Runs and Regression
  Gates](running-tests/runs.md)).

## What gets checked

A scheduler that dies fails the test by default: a crash, a load
failure, or a stall (the kernel watchdog kicks a stalled scheduler out
with an error) all exit non-zero and come back red, with the failure
trail from [Reading Failure Output](running-tests/failures.md).

> [!WARNING]
> Behavioral checks are a different story: nothing is asserted by
> default. A bare `#[ktstr_test]` reports pass even if the scheduler
> gave some workers no measured work or spread CPU time wildly — as
> long as it stayed alive.

Every behavioral check is an opt-in attribute: `not_stuck = true`
enables the zero-work-units, fairness-spread, and stuck-gap worker
checks, `max_spread_pct`, `min_iteration_rate`, and friends set
explicit thresholds. [Checking](concepts/checking.md) explains the model;
[Customize Checking](recipes/custom-checking.md) shows the override
flow.

## When a check fails

A failing check prints the violated threshold with the observed
value, then per-cgroup statistics. This excerpt is from a real run
that set an impossible `min_iteration_rate` floor:

<!-- captured: cargo ktstr test --kernel 7.0 -- --features integration -E 'test(=ktstr/throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">failed behavioral check</span></div>

<pre><span class="t-red">ktstr_test 'throughput_gate' [sched=scx-ktstr] [topo=1n1l2c1t] failed:</span>
  <span class="t-red">worker 71 iteration rate 41903.3/s below floor 50000000.0/s</span>
  <span class="t-red">worker 73 iteration rate 37834.5/s below floor 50000000.0/s</span>

--- stats ---
2 workers, 4 cpus, 2 migrations, worst_spread=0.0%, worst_gap=21ms
  cg0: workers=1 cpus=2 spread=0.0% gap=10ms migrations=1 iter=209600
  cg1: workers=1 cpus=2 spread=0.0% gap=21ms migrations=1 iter=189252
...
</pre></div>

The header names the test, scheduler, and topology variant; each
detail line names the check, observed value, and threshold. The full
output continues with timeline, scheduler-log, and monitor sections,
plus failure-dump artifacts and a ready-to-paste `cargo ktstr
replay` command — [Reading Failure
Output](running-tests/failures.md) walks the whole anatomy.

## Next steps

- [Tutorial: Zero to ktstr](tutorial.md) — build a complete test
  step by step, break it on purpose, and read the wreckage.
- [Test a New Scheduler](recipes/test-new-scheduler.md) — you have
  an `scx_*` binary and want it under test in five minutes.
- [Writing Tests](writing-tests.md) — the authoring reference:
  attributes, scenarios, snapshots, assertions.
