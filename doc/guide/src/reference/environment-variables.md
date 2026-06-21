# Environment Variables

Environment variables that control ktstr behavior.

## User-facing

| Variable | Description | Default |
|---|---|---|
| `KTSTR_KERNEL` | Kernel identifier for cargo-build-time BTF resolution (read by `build.rs`) and runtime image discovery. Accepts a path (`../linux`), version string (`6.14.2`), or cache key (use `cargo ktstr kernel list` for actual keys). During `cargo build`, only paths are used (`build.rs` extracts BTF from `vmlinux`). At runtime, version strings and cache keys resolve via the XDG cache; paths search only the specified directory (error if no image found). Set automatically by `cargo ktstr test --kernel`. **Overridden by `KTSTR_KERNEL_LIST` when present:** under multi-kernel runs the test binary's `--list` / `--exact` handlers consult `KTSTR_KERNEL_LIST` first and only fall back to `KTSTR_KERNEL` when the list env is unset; the producer-side `cargo ktstr` always sets `KTSTR_KERNEL` to the FIRST resolved entry alongside the full `KTSTR_KERNEL_LIST` so downstream code that inspects `KTSTR_KERNEL` directly still sees a valid path. | Auto-discovered |
| `KTSTR_KERNEL_LIST` | Multi-kernel wire format `label1=path1;label2=path2;…` consumed by the test binary's gauntlet expansion. Set by `cargo ktstr test` / `coverage` / `llvm-cov` when the resolved kernel set has 2 or more entries; the test binary's `--list` handler emits one variant per kernel (suffix `gauntlet/{name}/{preset}/{kernel_label}` or `ktstr/{name}/{kernel_label}`) and the `--exact` handler strips the suffix and re-exports `KTSTR_KERNEL` to the matching directory before booting the VM. Semicolon is the entry separator (paths can carry `:` on POSIX); `=` separates label from path. Empty value or unset means "single-kernel mode" — the test binary falls back to `KTSTR_KERNEL`. | None (single-kernel) |
| `KTSTR_CI` | Set to any non-empty value to flip every sidecar's `run_source` field from `"local"` (developer-machine default) to `"ci"`. Read at sidecar-write time by `detect_run_source`; surfaces through `cargo ktstr stats compare --run-source ci` so CI-produced runs can be partitioned from developer runs without per-run directory bookkeeping. Empty string counts as unset. The third value `"archive"` is applied at LOAD time (not write time) when `cargo ktstr stats compare --dir` / `list-values --dir` pulls sidecars from a non-default pool root — `KTSTR_CI` does not control that. | None (`run_source = "local"`) |
| `KTSTR_TEST_KERNEL` | Path to a bootable kernel image (`bzImage` on x86_64, `Image` on aarch64). See [Getting Started](../getting-started.md#kernel-discovery) and [Troubleshooting](../troubleshooting.md#no-kernel-found) for search order. | Auto-discovered |
| `KTSTR_CACHE_DIR` | Override the kernel image cache directory. When set, all cache operations use this path instead of the XDG default. | `$XDG_CACHE_HOME/ktstr/kernels/` or `$HOME/.cache/ktstr/kernels/` |
| `KTSTR_GHA_CACHE` | Set to `"1"` to enable remote kernel cache via GitHub Actions cache service. Requires `ACTIONS_CACHE_URL` (set by the GHA runner). Local cache is always authoritative; remote failures are non-fatal. | None (disabled) |
| `KTSTR_SCHEDULER` | Path to a scheduler binary for `SchedulerSpec::Discover`. See [Troubleshooting](../troubleshooting.md#scheduler-not-found) for search order. | Auto-discovered |
| `KTSTR_BUDGET_SECS` | Time budget in seconds for greedy test selection during `--list`. Must be positive. See [Running Tests](../running-tests.md). | None (all tests listed) |
| `KTSTR_SIDECAR_DIR` | Directory for per-test result sidecar JSON files. Used as-is when set, no key suffix. Consumed by the test harness (sidecar write path) and by bare `cargo ktstr stats` (sidecar read path). When this override is set, **pre-clear is skipped AND the per-run-key cross-process flock is skipped** — the operator chose the directory and owns its contents, so any pre-existing sidecars there are preserved, and ktstr does not coordinate concurrent writers against the override path. Two concurrent runs pointing the same `KTSTR_SIDECAR_DIR` at the same path therefore have no serialization between them; choose distinct override paths per process (or rely on the default-path branch, which acquires the flock automatically). `cargo ktstr stats list`, `cargo ktstr stats compare`, `cargo ktstr stats list-values`, and `cargo ktstr stats show-host` walk `{CARGO_TARGET_DIR or "target"}/ktstr/` by default and ignore `KTSTR_SIDECAR_DIR` — pass `--dir DIR` on `compare` / `list-values` / `show-host` to point them at an alternate run root. See [Runs](../running-tests/runs.md). | `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/` (where `{project_commit}` is the project HEAD short hex, suffixed `-dirty` when the worktree differs, or the literal `unknown` when not in a git repo — see [Runs](../running-tests/runs.md) for the unknown-commit collision semantics) |
| `KTSTR_NO_PERF_MODE` | Force `performance_mode=false`. Disables performance mode features (pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression) but still reserves a budget-sized LLC flock set (default 30% / vcpu-sized; see `KTSTR_CPU_CAP`) — only `KTSTR_BYPASS_LLC_LOCKS` skips flock acquisition. Presence is sufficient (any non-empty value). See [Performance Mode](../concepts/performance-mode.md#disabling-performance-mode). Also settable via `--no-perf-mode` CLI flag. | None (disabled) |
| `KTSTR_CARGO_TEST_MODE` | Marks a test invocation that runs without the cargo-ktstr wrapper (typically `KTSTR_KERNEL=… KTSTR_CARGO_TEST_MODE=1 cargo test -- some_test`). When active, the harness (1) skips the cross-process initramfs SHM cache and builds inline per VM run (process-local HashMap memoization still applies); (2) skips host-topology LLC / per-CPU flock acquisition — tests run on whatever CPUs the OS schedules them onto; (3) skips gauntlet variant expansion in nextest discovery — each `#[ktstr_test]` runs once with its declared topology, no `KTSTR_KERNEL_LIST` multi-kernel fan-out; (4) resolves `SchedulerSpec::Discover(name)` via `$PATH` first (before the sibling-dir / target-dir / cargo-build chain) so a user can install a scheduler on PATH and run a single test without driving the cargo-ktstr build pipeline. Empty string is treated as unset (rejection mirrors `KTSTR_NO_PERF_MODE`). Acceptable for development iteration; perf-mode tests still use their measurement contract internally but no peer-coordination flocks are taken. | None (full cargo-ktstr coordination) |
| `KTSTR_NO_SKIP_MODE` | Convert resource-contention and host-topology-insufficient skips into hard test failures (exit code `1`). Default behavior is to skip the test (exit code `0`) so a contended runner does not fail tests that simply could not start; setting this env var (or passing `--no-skip-mode` on `cargo ktstr test` / `coverage` / `llvm-cov`) opts into "if the test cannot run, the test fails". Use when a test environment is supposed to provision sufficient resources and a missing topology is a real configuration error. The CLI flag exports `KTSTR_NO_SKIP_MODE=1` for the test binary. Presence is sufficient (any value). | None (skip on contention / topology insufficiency) |
| `KTSTR_CPU_CAP` | Cap the number of host CPUs reserved by a no-perf-mode VM or kernel build to `N` (integer ≥ 1, a CPU count). The planner walks whole LLCs in consolidation- / NUMA-aware order, filtered to the calling process's sched_getaffinity cpuset, partial-taking the last LLC so `plan.cpus.len()` is EXACTLY `N`. CLI flag `--cpu-cap N` takes precedence; empty string is treated as unset; `0` or non-numeric values are rejected with a parse error. On `shell`, `--cpu-cap` is rejected at clap parse time unless `--no-perf-mode` is also passed (`requires = "no_perf_mode"`); on `kernel build`, no perf-mode concept applies. Library consumers that set `performance_mode=true` on `KtstrVmBuilder` directly see the env var silently ignored — the builder's perf-mode branch never consults `CpuCap::resolve`. Mutually exclusive with `KTSTR_BYPASS_LLC_LOCKS=1` at every entry point (rejection wording contains "resource contract"). See [Resource Budget](../concepts/resource-budget.md). | None (30% of allowed CPUs, minimum 1) |
| `KTSTR_BYPASS_LLC_LOCKS` | Skip host-side LLC flock acquisition entirely. No coordination against concurrent perf-mode runs. Presence is sufficient (any non-empty value). Mutually exclusive with `KTSTR_CPU_CAP` / `--cpu-cap` — the conflict is rejected at every entry point with an error containing "resource contract". See [Resource Budget](../concepts/resource-budget.md#ktstr_bypass_llc_locks--escape-hatch). | None (coordinate) |
| `KTSTR_KERNEL_PARALLELISM` | Override the rayon pool width `cargo ktstr` uses for `--kernel` per-spec fan-out in `resolve_kernel_set`. Parsed as `usize` after `.trim()`; whitespace around the value is tolerated. Values that fail to parse, are negative, or are `0` silently fall through to the default — a typoed export (`=abc`, `=0`) does NOT disable parallelism, it degrades to the host-CPU default. Useful when the default is wrong for the host: a fast NIC + slow CPU benefits from a higher value (more concurrent downloads); a contended CI runner benefits from a lower cap to leave bandwidth and CPU for sibling jobs. **Scope is narrow**: only the bounded `ThreadPool` `resolve_kernel_set` builds via `ThreadPoolBuilder::install` is affected — the global rayon pool that other code paths (nextest harness, polars groupby, etc.) consume is untouched. The build phase inside each per-spec resolve is already serialized at the LLC-flock layer, so raising this knob accelerates download fan-out only, not build time. | `std::thread::available_parallelism()` (host logical CPU count, falling back to `1` on a sandboxed host where `available_parallelism` errors) |
| `KTSTR_VERBOSE` | Set to `"1"` for verbose VM console output (`earlyprintk`, `loglevel=7`). | None |
| `KTSTR_LOCK_DIR` | Override the directory under which the per-LLC and per-CPU host flock files (`ktstr-llc-*.lock`, `ktstr-cpu-*.lock`) live. `cargo ktstr locks` and `ktstr locks` both consult this; the source-tree build lock at `{cache_root}/.locks/source-{path_hash}.lock` is independent. Use when `/tmp` is `noexec`-mounted or constrained on a CI runner. | `/tmp` |
| `KTSTR_CONTENTION_BYPASS` | Set to `1` (only the exact value `"1"` activates) to make a transient KVM errno surface as a HARD failure instead of being classified as `ResourceContention` (a SKIP) — but only when the post-failure resource snapshot reports `near_limit=false`. Used to catch kernel-side regressions that share a transient errno with genuine host pressure; it makes failures stricter, not more permissive. | None (transient errnos → ResourceContention skip) |
| `KTSTR_CACHE_STORE_LOCK_TIMEOUT` | humantime-parsed timeout for the cache-store exclusive lock acquired during kernel build cache writes. Tune up on CI runners where the cache-store flock occasionally exceeds the default. | (compile-time default in `src/cache/cache_dir.rs`) |
| `KTSTR_VERIFIER_RAW` | Set to `"1"` by `cargo ktstr verifier --raw` to make the test binary emit verifier output verbatim (no pattern-repetition collapsing). User-facing equivalent of the CLI flag. | None |
| `KTSTR_MODEL_OFFLINE` | Set to a non-empty value to disable the LLM model download path. When the model artifact is not already cached, the model load fails fast with an error (no network fetch is attempted); a pre-seeded cache still loads. Useful in air-gapped CI or for fast iteration when model downloads are not required. | None (network fetch enabled) |
| `KTSTR_LLM_DEBUG_RESPONSES` | Set to any non-empty value to log every LLM response payload at debug level. Use when triaging LLM-driven analysis output. | None |
| `KTSTR_LOG_PASSES` | Set to any value other than empty or `"0"` to log every `Verdict` pass-detail entry (not just failures). Useful for triaging "test passed but I want to see what the assertion saw" cases. | None (only failures rendered) |
| `KTSTR_WORKER_READY_MARKER_OVERRIDE` | Override the absolute file path where the `ktstr-jemalloc-alloc-worker` binary writes its ready marker. When set and non-empty the worker writes to this exact path (the host-side poll watches the same path) instead of the default. Set when the default location is on a `noexec` or quota-constrained filesystem. | None (`/tmp/ktstr-worker-ready-<pid>`) |
| `KTSTR_HOST_CGROUP_PARENT` | Override the cgroup-v2 parent directory under which `#[ktstr_test(host_only)]` tests create per-test cgroups. The default `/sys/fs/cgroup/ktstr` matches the always-root invariant. Validated upfront: empty string falls back to the default; a set value must be rooted under `/sys/fs/cgroup` and name a non-root subdirectory (e.g. `/sys/fs/cgroup/ktstr-foo`) — relative paths, empty strings via concat, or `/sys/fs/cgroup` itself are rejected with an actionable diagnostic. Non-root callers setting this env var are rejected: `CgroupManager::setup` walks every ancestor's `cgroup.subtree_control` from `/sys/fs/cgroup` down to the configured parent, which requires write access on each — cgroup-v2 user-delegation (`nsdelegate`-namespaced cgroup2 / `systemd-run --user --scope` with `Delegate=cpu cpuset memory io pids`) is supported by also setting `KTSTR_CGROUP_WALK_ROOT` to the delegated subtree boundary, so the `subtree_control` ancestor walk stops there instead of starting at `/sys/fs/cgroup`. host_only tests resolve real-host topology via `/sys/devices/system/cpu/online`; the parent override only affects WHERE per-test cgroups land, not WHICH CPUs the topology reports. | `/sys/fs/cgroup/ktstr` |
| `KTSTR_CGROUP_WALK_ROOT` | Override the cgroupfs root that `CgroupManager::setup` descends from when enabling controllers in each ancestor's `cgroup.subtree_control`. Empty / unset falls back to `/sys/fs/cgroup`; a non-empty value must be a prefix of the configured `KTSTR_HOST_CGROUP_PARENT` so the walk stays inside the operator-owned subtree. Exists for cgroup-v2 user delegation (systemd `Delegate=yes` / container `nsdelegate`). | `/sys/fs/cgroup` |
| `KTSTR_STALL_POLL_MS` | Host-mode stall-monitor poll cadence in milliseconds. Empty / unset / `0` / unparseable falls back to the default. | (compile-time default in `src/scenario/host_stall.rs`) |
| `RUST_BACKTRACE` | Gates verbose diagnostic output on failure. Also enables verbose VM console output (same as `KTSTR_VERBOSE=1`) when set to `"1"` or `"full"`. Propagated to the guest. | None |
| `RUST_LOG` | Controls every ktstr tracing filter — guest-side and host-side. **Guest-side**: propagated to the VM kernel command line and parsed by the guest tracing subscriber, so guest events are filtered by the same `RUST_LOG` value the host process saw at launch. **Host-side**: applied via the `EnvFilter` the inference engine installs on first call to `global_backend()` (`tracing_subscriber::fmt::try_init()` — a no-op when an outer subscriber was already installed). Two host-side targets are useful in practice: `"llama-cpp-2"` (literal hyphens — the `Metadata::target()` set by `llama_cpp_2::send_logs_to_tracing(LogOptions::default())`, carrying llama.cpp / GGML log lines: model-load progress, GGUF parse chatter, KV-cache reservation notes, error reasons) and `"ktstr::flock::acquire"` (the `module_path!()` default for `src/flock/acquire.rs`, where the shared flock-timeout primitive emits a `tracing::debug!("waiting on flock at …")` event on each poll iteration that loops back). Examples: `RUST_LOG=llama-cpp-2=info` widens model-load logging to INFO; `RUST_LOG=ktstr::flock=debug` surfaces flock-contention heartbeats; `RUST_LOG=llama-cpp-2=off` suppresses llama.cpp output entirely. `EnvFilter` does prefix-matching on `meta.target()` without underscore normalization (the hyphenated llama-cpp-2 target is a string literal, not a Rust path). The default `EnvFilter` derived from an unset `RUST_LOG` keeps only ERROR-level events, which is exactly the C-side rejection-reason text behind otherwise-opaque `InferenceError::ModelLoad` / `LlamaModelLoadError::NullResult` failures. Operators wanting a different sink (file, alternate format) can install their own subscriber FIRST — `try_init()` becomes a no-op and the operator's subscriber receives the events. | None (host-side: ERROR-level events on stderr) |

## jemalloc probe wiring

These variables are only consulted by integration tests that boot a
jemalloc-linked allocator worker inside the VM and attach the
`ktstr-jemalloc-probe` to it (see `tests/jemalloc_probe_tests.rs`).
Both are set from a `#[ctor]` in the test binary so they land before
the test harness dispatches.

### What `#[ctor]` is and why these variables need it

`#[ctor]` is a Rust attribute (provided by the
[`ctor` crate](https://crates.io/crates/ctor)) that marks a
function to run automatically at binary initialization — after the
dynamic linker sets up the process but before `main()` is called.
Linux implements this via the `.init_array` ELF section; the
attribute's generated code registers the function there. A function
under `#[ctor]` therefore runs exactly once per process, on the
main thread, before any code inside `main()` executes.

The two environment variables above are consulted by ktstr's
nextest pre-dispatch path (`ktstr_test_early_dispatch`), which
itself runs under a ktstr-owned `#[ctor]` that intercepts the
nextest protocol args (`--list`, `--exact`) before the standard
Rust test harness sees them. The probe-wiring variables must
already be populated when that early dispatch fires, so setting
them from plain test-body code is too late — the sidecar
enumeration and initramfs packing decisions have already run.
Tests needing probe integration install their own `#[ctor]` that
writes the two variables via `std::env::set_var`, ensuring both
ktstr's early dispatch and the VM launch path downstream see the
populated values.

The ctor hook runs under the `ctor` crate re-exported at
`ktstr::__private::ctor`, so a new test crate does not need to
add `ctor` to its own dependencies — it can use the re-export
via `ktstr::__private::ctor::ctor` and stay in sync with the
version ktstr itself depends on, avoiding the "two ctor
crates, two `.init_array` entries, ordering undefined" pitfall.
The attribute requires the `unsafe` marker that ctor 1.0
mandates for every constructor function; see
`src/test_support/runtime.rs` for the verbatim copy-paste form.

Leaving either variable unset is the normal case — the VM
launcher skips probe wiring entirely, and no initramfs entry is
added.

| Variable | Description | Default |
|---|---|---|
| `KTSTR_JEMALLOC_PROBE_BINARY` | Absolute host path to the `ktstr-jemalloc-probe` binary. When set, the probe is packed into every VM's base initramfs at `/bin/ktstr-jemalloc-probe`. Typically set by a `#[ctor]` in the integration test crate to `env!("CARGO_BIN_EXE_ktstr-jemalloc-probe")`. Empty string is treated the same as unset. | None (no probe packed) |
| `KTSTR_JEMALLOC_ALLOC_WORKER_BINARY` | Absolute host path to the paired `ktstr-jemalloc-alloc-worker` binary. Packed alongside the probe for the closed-loop tests that run the probe against a live allocator target. Same `#[ctor]` shape as above using `env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker")`. Empty string is treated the same as unset. | None (no worker packed) |

## LLVM coverage

| Variable | Description | Default |
|---|---|---|
| `LLVM_COV_TARGET_DIR` | Directory for extracted profraw files. | Parent of `LLVM_PROFILE_FILE`, or `<exe-dir>/llvm-cov-target/` |
| `LLVM_PROFILE_FILE` | Standard LLVM profiling output path. ktstr reads its parent as a fallback profraw directory. | None |

## Nextest protocol

| Variable | Description | Default |
|---|---|---|
| `NEXTEST` | Set by nextest when it invokes the test binary. ktstr's `#[ctor]` dispatch inspects this to decide whether to intercept the nextest protocol args (`--list`, `--exact`) for gauntlet expansion and budget-based selection before `main()` runs. Under plain `cargo test`, this is unset and the standard harness runs the `#[test]` wrappers directly. | None |

## VM-internal

Mostly set by the host on the guest kernel command line and read by
the guest init (via `/proc/cmdline`); a few (noted below) are
process-internal markers set inside the guest. Not intended for user
configuration; listed here for debugging.

| Variable | Description |
|---|---|
| `KTSTR_MODE` | Guest execution mode. `shell` requests the interactive shell; `disk_template` requests a one-shot mkfs template-build VM. Absent (no `KTSTR_MODE` token) means the default test-dispatch path. |
| `KTSTR_TOPO` | Topology string (`numa_nodes,llcs,cores,threads`) for guest-side scenario resolution. |
| `KTSTR_TERM` | Terminal type forwarded from the host (sets guest `TERM`). |
| `KTSTR_COLORTERM` | Color capability forwarded from the host (sets guest `COLORTERM`). |
| `KTSTR_COLS` | Host terminal column count, used to size the guest pty when available. |
| `KTSTR_ROWS` | Host terminal row count, used to size the guest pty when available. |
| `KTSTR_GUEST_INIT` | Process-internal marker set by the guest init (`ktstr-init`) via `std::env::set_var` — NOT a host-emitted cmdline token. Read via `std::env::var_os` by `src/workload/spawn` to detect re-entrant worker spawns under PID-1 init. |
| `KTSTR_DISK0_FS` / `KTSTR_DISK0_MOUNT` / `KTSTR_DISK0_RO` | Disk-attach metadata (fs type, mount point, ro flag) emitted by `src/vmm/setup.rs` when `#[ktstr_test(disk = ...)]` is set and consumed by `src/vmm/rust_init/mounts.rs` to mount the virtio-blk backing inside the guest. |

Guest↔host signaling uses bulk-port TLV frames on the virtio-console
port-1 channel — `MSG_TYPE_TEST_RESULT` (test verdict),
`MSG_TYPE_EXIT` (exit code), `MSG_TYPE_LIFECYCLE` (boot/payload phases
and scheduler-attach failures), and `MsgType::ExecExit` (shell-exec
exit code); see `src/vmm/wire.rs`. The earlier COM2 string sentinels
(`RESULT_START`/`RESULT_END`, `KTSTR_EXIT:`, `KTSTR_INIT_STARTED`,
`KTSTR_PAYLOAD_STARTING`, `KTSTR_EXEC_EXIT=N`) have been removed — the
guest no longer emits them and the host no longer scrapes them. None
of these are environment variables.
