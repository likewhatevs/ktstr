<div align="center">

<h1><picture>
<source media="(prefers-color-scheme: dark)" srcset="doc/assets/wordmark-dark.png">
<img alt="ktstr" src="doc/assets/wordmark-light.png" width="266">
</picture></h1>

**Test Linux schedulers like code.**

Every test boots a real kernel in a KVM micro-VM with the topology it
declares — and ktstr watches what your scheduler does from the host,
without touching the guest.

[![CI](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg?token=E7GRAO2KZM)](https://codecov.io/gh/likewhatevs/ktstr)
[![crates.io](https://img.shields.io/crates/v/ktstr.svg)](https://crates.io/crates/ktstr)
[![docs.rs](https://img.shields.io/docsrs/ktstr)](https://docs.rs/ktstr)
[![docs](https://img.shields.io/badge/docs-ktstr.dev-blue)](https://ktstr.dev/guide/)
[![api](https://img.shields.io/badge/docs-api-blue)](https://ktstr.dev/rustdoc/ktstr/)

[Get started](https://ktstr.dev/guide/getting-started.html) ·
[See it in action](https://ktstr.dev/guide/features.html) ·
[Tutorial](https://ktstr.dev/guide/tutorial.html)

</div>

---

Scheduler bugs hide in topology: the fairness regression that only
shows up on an odd LLC count, the starvation that needs SMT siblings,
the crash that wants a NUMA crossing. Testing a
[sched_ext](https://github.com/sched-ext/scx) scheduler against those
shapes has meant scrounging hardware and hand-running repro scripts.
ktstr turns it into `cargo test`: declare the topology on the test,
and the VM actually has it.

## Quick taste

```rust
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
});

#[ktstr_test(scheduler = MY_SCHED, llcs = 2, cores = 4, threads = 1)]
fn steady_under_my_sched(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

Run it against any kernel — a released version, a local source tree,
or a git URL:

```console
$ cargo ktstr test --kernel 7.0
    PASS [  34.451s] (1/1) ktstr::failure_dump_e2e ktstr/failure_dump_renders_bss_fields
```

## When it breaks, you see why

A crash log tells you where the scheduler died. ktstr also tells you
what the state looked like on the way there: on a crash it boots a
second VM, attaches BPF probes along the crash path, and reruns the
scenario. `→` marks fields that changed between entry and exit:

```text
=== AUTO-PROBE: scx_exit fired ===

  ktstr_select_cpu                                    main.bpf.c:380
    task_struct *p
      pid        40
      dsq_id     SCX_DSQ_INVALID  →  SCX_DSQ_LOCAL
      slice      19982063         →  20000000
      scx_flags  RESET_RUNNABLE_AT|DEQD_FOR_SLEEP|ENABLED
  do_enqueue_task                                     kernel/sched/ext.c:1885
    rq *rq
      cpu        0
    task_struct *p
      pid        40
      dsq_id     SCX_DSQ_LOCAL
      scx_flags  QUEUED|DEQD_FOR_SLEEP|ENABLED
  ...
  bpf_prog_9a11f2edaac0b52f_ktstr_dispatch+0x57/0x1db
```

The chain shows what the scheduler did with its last tasks on the way
into the error — not just where it died. See
[Auto-Repro](https://ktstr.dev/guide/running-tests/auto-repro.html)
and
[Reading Failure Output](https://ktstr.dev/guide/running-tests/failures.html).

## What you get

- **Topology as code** — NUMA nodes, LLCs, cores, SMT declared on the
  test attribute, realized in the guest down to the ACPI tables.
- **Workloads and assertions as data** — cgroups, cpusets, workers,
  and checks (starvation, fairness, throughput floors) declared on
  the test; native [schbench and taobench
  ports](https://ktstr.dev/guide/concepts/work-types.html) with
  per-phase metrics.
- **[Gauntlet](https://ktstr.dev/guide/running-tests/gauntlet.html)** —
  one test fans out across a matrix of topology presets; multiple
  `--kernel` flags (or a `6.12..6.14` range) add the kernel axis. A
  `--kernel` value can be a version, a source tree, a `git+URL#tag=…`
  source, a local `.rpm`/`.deb` package, or a prebuilt distro kernel
  (`fedora`/`ubuntu`/`amazonlinux`/`steamos`, or the official GKE COS
  kernel via `gke`; downloaded from official artifacts and cached like
  a built kernel).
- **[Verifier sweep](https://ktstr.dev/guide/running-tests/verifier.html)** —
  every declared scheduler load-tested against the real kernel
  verifier across topologies, with per-program instruction counts and
  cycle-collapsed rejection logs.
- **[Regression gates](https://ktstr.dev/guide/running-tests/runs.html)** —
  per-run metric sidecars, `perf-delta --noise-adjust` gating between
  commits, thresholds decide instead of eyeballs.
- **[Interactive shell](https://ktstr.dev/guide/running-tests/cargo-ktstr.html#shell)** —
  `cargo ktstr shell --test my_failing_test` boots the exact VM a
  failing test saw.

## Install

```toml
[dev-dependencies]
ktstr = "0.26.0"
```

```sh
cargo install --locked cargo-nextest
cargo install --locked ktstr
```

Linux only (x86_64, aarch64), `/dev/kvm` required. ktstr is pre-1.0:
pin the exact patch version and install the matching CLI
(`cargo install --locked --bin cargo-ktstr ktstr@X.Y.Z`).
[Getting Started](https://ktstr.dev/guide/getting-started.html)
covers dependencies, kernel discovery, and `/dev/kvm` permissions.

## Documentation

Everything lives at **[ktstr.dev](https://ktstr.dev/guide/)** — the
[tutorial](https://ktstr.dev/guide/tutorial.html), the
[feature tour with real output](https://ktstr.dev/guide/features.html),
concepts, recipes, and the
[API reference](https://ktstr.dev/rustdoc/ktstr/).

## Contributing

Pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPL-2.0-only
