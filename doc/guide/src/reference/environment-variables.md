# Environment Variables

Every environment variable ktstr reads, grouped by task. Unless a row
says otherwise, an empty value is treated the same as unset — the
exceptions are called out per row.

Most of these have a CLI flag equivalent; prefer the flag in scripts
and the variable in CI job-level `env:` blocks.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Daily knobs</strong><p>Kernel selection, scheduler overrides, perf mode, skip mode, logging, and budgets.</p></div>
<div class="kt-doc-card"><strong>Caches</strong><p>Kernel cache roots, busybox/wprof build inputs, remote cache, and parallelism.</p></div>
<div class="kt-doc-card"><strong>Diagnostics</strong><p>Sidecar directories, CI stamps, guest console verbosity, and escape hatches.</p></div>
</div>

## Daily knobs

| Variable | Effect | Accepted values | Default |
|---|---|---|---|
| `KTSTR_KERNEL` | Selects the kernel for every entry point (build-time BTF resolution and runtime image discovery). Set automatically by `cargo ktstr test --kernel`. | Exact version (`6.14`), range (`6.14..7.0`), `git+URL#tag=…`/`#branch=…`/`#sha=…`, path, local package (`.rpm`/`.deb`/`.pkg.tar.zst`), distro spec (`fedora[-44]`/`f44`, `ubuntu[-24.04]`, `amazonlinux[-2023]`/`al2023`, `steamos[-3.8]`), or cache key | Auto-discovered |
| `KTSTR_TEST_KERNEL` | Points the test harness directly at a bootable image (`bzImage` on x86_64, `Image` on aarch64). Set-but-empty is a hard error, not a fallback. | Image path | Auto-discovered |
| `KTSTR_SCHEDULER` | Global binary override for every `SchedulerSpec::Discover` scheduler. See [Troubleshooting](../troubleshooting.md#scheduler-not-found) for the full resolution order. | Binary path | Build/discover |
| `KTSTR_SCHEDULER_PROFILE` | Cargo build profile for the scheduler-under-test (independent of the harness profile). Set by `cargo ktstr … --profile`. | Profile name | `release` |
| `KTSTR_NO_PERF_MODE` | Disable performance mode (pinning, RT scheduling, hugepages, KVM exit suppression). A budget-sized CPU reservation is still taken — see [Resource Budget](../concepts/resource-budget.md). Flag: `--no-perf-mode`. | Any non-empty value | Perf mode available |
| `KTSTR_NO_SKIP_MODE` | Turn resource-contention / insufficient-topology skips into hard failures. Flag: `--no-skip-mode`. Presence-only: even an empty value activates it. | Presence | Skip on contention |
| `KTSTR_CARGO_TEST_MODE` | Marks a direct `cargo test` / `cargo nextest` run without the `cargo ktstr` wrapper: no gauntlet expansion, no host CPU flocks, per-process initramfs builds, and `$PATH`-first scheduler discovery. | Any non-empty value | Full orchestration |
| `KTSTR_VERBOSE` | Verbose guest console output (`loglevel=7`, plus `earlyprintk=serial` on x86_64). | Exactly `"1"` | Quiet console |
| `KTSTR_LOG_PASSES` | Log every `Verdict` pass detail, not just failures — for "the test passed but what did the assertion see?". | Anything except empty or `"0"` | Failures only |
| `KTSTR_BUDGET_SECS` | Time budget for greedy coverage-maximizing test selection at list time. See [Running Tests](../running-tests.md). | Positive number (fractional ok); invalid values warn and are ignored | All tests listed |
| `RUST_BACKTRACE` | Verbose diagnostics on failure; `"1"` or `"full"` also enables the verbose guest console. Propagated to the guest. | `1`, `full` | Off |
| `RUST_LOG` | Tracing filter, host-side and guest-side (forwarded on the guest kernel command line). Example: `RUST_LOG=ktstr::flock=debug` surfaces flock-contention heartbeats. | tracing filter syntax | Off |

## Kernel builds and caches

| Variable | Effect | Accepted values | Default |
|---|---|---|---|
| `KTSTR_CACHE_DIR` | Override the cache root (kernel images, BTF anchors, blobs). The value is used verbatim — no per-type subdirectory is appended. | Absolute path | `$XDG_CACHE_HOME/ktstr/` or `~/.cache/ktstr/` (per-type subdir appended) |
| `KTSTR_GHA_CACHE` | Enable the GitHub Actions remote kernel cache. Needs `ACTIONS_CACHE_URL` (set by the runner) and a ktstr built with `--features remote-cache`. Local cache stays authoritative; remote failures are non-fatal. | Exactly `"1"` | Disabled |
| `KTSTR_KERNEL_PARALLELISM` | Width of the download/resolve fan-out for multi-kernel `--kernel` specs. Affects downloads only — builds serialize on the host CPU locks. | Positive integer; `0` / unparseable falls back to default | Host logical CPU count |
| `KTSTR_CACHE_STORE_LOCK_TIMEOUT` | Timeout for the exclusive lock taken while storing a built kernel into the cache. Raise on CI runners with slow shared disks. | humantime duration (`30s`, `2m`) | Compile-time default |
| `KTSTR_BUSYBOX_TARBALL` | Build-time (`build.rs`): read the busybox source tarball from a local path instead of downloading it. | Tarball path | Download |
| `KTSTR_SKIP_BUSYBOX_BUILD` | Build-time (`build.rs`): skip the busybox compile entirely; shell mode becomes unavailable. | Any non-empty value | Build busybox |
| `KTSTR_SKIP_WPROF_BUILD` | Build-time (`build.rs`, `wprof` feature only): skip fetching and compiling the bundled wprof tooling. | Any non-empty value | Build wprof |

## Sidecars and stats

| Variable | Effect | Accepted values | Default |
|---|---|---|---|
| `KTSTR_SIDECAR_DIR` | Override the per-test sidecar output directory. Skips both the pre-clear and the cross-process flock — the operator owns the directory, so concurrent runs pointing at the same path are unserialized. `stats` subcommands read the default pool; pass `--dir` to point them elsewhere. See [Runs and Regression Gates](../running-tests/runs.md). | Directory path | `{runs root}/{kernel}-{project_commit}/` |
| `KTSTR_CI` | Stamp every sidecar's `run_source` as `"ci"` instead of `"local"`, so CI-produced runs are filterable in the stats pool. | Any non-empty value | `"local"` |

## Resource coordination and escape hatches

See [Resource Budget](../concepts/resource-budget.md) for how these
interact; the first two are mutually exclusive at every entry point.

| Variable | Effect | Accepted values | Default |
|---|---|---|---|
| `KTSTR_CPU_CAP` | Cap the host CPUs reserved by a no-perf-mode VM or kernel build. Flag `--cpu-cap N` takes precedence. | Integer ≥ 1; `0` / non-numeric rejected | Kernel build: 30% of allowed CPUs (min 1). No-perf VM: the vCPU count, floored at 30%. |
| `KTSTR_BYPASS_LLC_LOCKS` | Skip host-side LLC flock acquisition entirely — no coordination against concurrent runs. | Any non-empty value | Coordinate |
| `KTSTR_LOCK_DIR` | Directory for the per-LLC / per-CPU flock files. Use when `/tmp` is constrained on a runner. | Directory path | `/tmp` |
| `KTSTR_CONTENTION_BYPASS` | Make transient KVM errnos immediate hard failures instead of retryable `ResourceContention` failures (only when the host is not near its limits) — stricter, for catching kernel-side regressions. | Exactly `"1"` | Retryable contention failure |
| `KTSTR_HOST_CGROUP_PARENT` | cgroup-v2 parent under which `host_only` tests create per-test cgroups. Must be a non-root subdirectory of `/sys/fs/cgroup`. | Path under `/sys/fs/cgroup` | `/sys/fs/cgroup/ktstr` |
| `KTSTR_CGROUP_WALK_ROOT` | Where the setup-time controller-enable walk starts, for delegated cgroup subtrees (systemd `Delegate=yes`, container `nsdelegate`). Must be a prefix of the configured parent. | Path prefix of the parent | `/sys/fs/cgroup` |
| `KTSTR_STUCK_POLL_MS` | Host-mode stuck-monitor poll cadence. | Milliseconds; empty / `0` / unparseable falls back | 500 ms |
| `KTSTR_WORKER_READY_MARKER_OVERRIDE` | Path where the jemalloc alloc worker writes its ready marker, for `noexec` or quota-constrained temp filesystems. | Absolute path | `/tmp/ktstr-worker-ready-<pid>` |

## Set by ktstr itself

`cargo ktstr` stamps these across the orchestrator → nextest →
test-binary boundary. Listed so you can recognize them in `ps` output
and CI logs — do not set them by hand.

| Variable | Carries |
|---|---|
| `KTSTR_KERNEL_LIST` | Multi-kernel fan-out list (`label=path;…`) when a run resolves 2+ kernels; each test expands to one variant per kernel. Takes precedence over `KTSTR_KERNEL` during variant expansion. |
| `KTSTR_KERNEL_COMMIT` | `dir=commit` map of each source kernel's HEAD, so per-test processes skip re-walking the kernel tree. |
| `KTSTR_PROJECT_COMMIT` | The project commit label perf-delta children must record in their sidecars. |
| `KTSTR_ORCHESTRATED` | Orchestration marker; VM-booting integration tests skip when it is absent (raw `cargo nextest run` would bypass their resource budgets). |
| `KTSTR_RUN_EPOCH` | Per-invocation session token that keeps parallel test processes from pre-clearing each other's freshly written sidecars. |
| `KTSTR_RUNS_ROOT` | Absolute runs root, stamped once so sidecar writers and post-run readers resolve the same directory regardless of CWD. |
| `KTSTR_PERF_ONLY` | Set by `perf-delta` runs: skip every test without `performance_mode`. Exporting it manually restricts any run the same way. |
| `KTSTR_VERIFIER_RAW` | Set by `cargo ktstr verifier --raw`: emit verifier logs verbatim, no cycle collapsing. |
| `KTSTR_VERIFIER_RESULT_DIR` | Directory where verifier cells write per-cell PASS/FAIL records for the summary grid. |
| `KTSTR_VERIFIER_SCHEDULER` | The `verifier --scheduler NAME` filter, forwarded to cell emission. |
| `KTSTR_BUSYBOX_PATH` | Path to the busybox blob `cargo ktstr` extracts at startup for shell-mode VMs and disk-template builds. |

## Probe wiring

Consulted by integration tests that boot a jemalloc-linked allocator
worker and attach the jemalloc probe to it. Both must be populated
before ktstr's early nextest dispatch runs, so tests set them from a
`#[ctor]` — see [Payloads and Included Files](../writing-tests/payloads.md)
for the wiring pattern. Leaving them unset is the normal case: no
probe is packed into the initramfs.

| Variable | Effect | Default |
|---|---|---|
| `KTSTR_JEMALLOC_PROBE_BINARY` | Absolute host path to `ktstr-jemalloc-probe`; packed into every VM's initramfs at `/bin/ktstr-jemalloc-probe` when set. | No probe packed |
| `KTSTR_JEMALLOC_ALLOC_WORKER_BINARY` | Absolute host path to the paired `ktstr-jemalloc-alloc-worker`, packed alongside the probe. | No worker packed |

## LLVM coverage

| Variable | Effect | Default |
|---|---|---|
| `LLVM_COV_TARGET_DIR` | Directory for extracted profraw files. | Parent of `LLVM_PROFILE_FILE`, or `<exe-dir>/llvm-cov-target/` |
| `LLVM_PROFILE_FILE` | Standard LLVM profiling output path; ktstr reads its parent as a fallback profraw directory. | None |

## Nextest protocol

| Variable | Effect | Default |
|---|---|---|
| `NEXTEST` | Set by nextest when it invokes the test binary. ktstr's early dispatch inspects it to decide whether to intercept `--list` / `--exact` for gauntlet expansion and budget selection. | None |

## VM-internal

Mostly set by the host on the guest kernel command line and read by
the guest init (via `/proc/cmdline`); a few (noted below) are
process-internal markers set inside the guest. Not intended for user
configuration; listed here for debugging.

| Variable | Description |
|---|---|
| `KTSTR_MODE` | Guest execution mode. `shell` requests the interactive shell; `disk_template` requests a one-shot mkfs template-build VM. Absent means the default test-dispatch path. |
| `KTSTR_TOPO` | Topology string (`numa_nodes,llcs,cores,threads`) for guest-side scenario resolution. |
| `KTSTR_TERM` | Terminal type forwarded from the host (sets guest `TERM`). |
| `KTSTR_COLORTERM` | Color capability forwarded from the host (sets guest `COLORTERM`). |
| `KTSTR_COLS` / `KTSTR_ROWS` | Host terminal size, used to size the guest pty when available. |
| `KTSTR_GUEST_INIT` | Process-internal marker set by the guest init — not a host-emitted cmdline token. Used to detect re-entrant worker spawns under PID-1 init. |
| `KTSTR_DISK0_FS` / `KTSTR_DISK0_MOUNT` / `KTSTR_DISK0_RO` | Disk-attach metadata (fs type, mount point, ro flag) for `#[ktstr_test(disk = ...)]`, consumed by the guest to mount the virtio-blk backing. |
