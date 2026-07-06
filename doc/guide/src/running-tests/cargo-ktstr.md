# cargo ktstr

`cargo ktstr` is the host-side command for the whole workflow: it
resolves (and if needed builds and caches) the kernel, then drives
`cargo nextest run` so every `#[ktstr_test]` boots its VM against
exactly the kernel you asked for.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Run</strong><p><code>test</code>, <code>replay</code>, and common flags for ordinary ktstr suites.</p></div>
<div class="kt-doc-card"><strong>Prepare</strong><p><code>kernel</code>, <code>shell</code>, <code>export</code>, and cache-management flows.</p></div>
<div class="kt-doc-card"><strong>Analyze</strong><p><code>verifier</code>, <code>stats</code>, <code>perf-delta</code>, <code>ctprof</code>, and CI targeting.</p></div>
</div>

## Install

```sh
cargo install --locked ktstr   # installs both `ktstr` and `cargo-ktstr`
```

The test-fixture binaries (jemalloc probes, schbench/taobench
validators) are behind the non-default `integration` feature and are
not installed by default. To build from a workspace checkout
instead: `cargo build --bin cargo-ktstr`.

## Task map

| I want to... | Command | Depth |
|---|---|---|
| Run tests on a kernel | [`cargo ktstr test`](#test) | this page |
| Re-run last session's failures | [`cargo ktstr replay`](#replay) | this page |
| Manage cached kernels | [`cargo ktstr kernel`](#kernel) | this page |
| Sweep schedulers through the BPF verifier | [`cargo ktstr verifier`](#verifier) | [Verifier](verifier.md) |
| Analyze results, gate regressions | [`cargo ktstr stats` / `perf-delta`](#stats) | [Runs](runs.md) |
| Narrow CI to affected schedulers | [`cargo ktstr affected`](#affected) | [CI](../ci.md) |
| Reproduce a test on bare metal | [`cargo ktstr export`](#export) | this page |
| Debug interactively in a VM | [`cargo ktstr shell`](#shell) | [ktstr shell](ktstr.md#shell) |

## Common flags {#common-flags}

These flags mean the same thing on every subcommand that takes them.

<a id="multi-kernel-kernel-as-a-gauntlet-dimension"></a>

**`--kernel ID`** — one grammar everywhere:

```sh
cargo ktstr test --kernel ../linux                    # local source tree (builds + caches)
cargo ktstr test --kernel 6.14.2                      # version (auto-downloads on miss)
cargo ktstr test --kernel 6.14                        # major.minor prefix → latest patch release
cargo ktstr test --kernel 6.14.2-tarball-x86_64-kc... # cache key (from `kernel list`)
cargo ktstr test --kernel 6.12..6.14                  # range: every stable+longterm release inside
cargo ktstr test --kernel git+https://example.com/r.git#tag=v6.14   # git tag (#branch= / #sha= too)
cargo ktstr test --kernel 6.14.2 --kernel 7.0         # repeatable → multi-kernel matrix
```

Ranges expand against kernel.org's `releases.json`; both endpoints
are series-inclusive (`6.11..6.14` covers every `6.14.N`; spell
`6.14.2` for an exact bound), and EOL series silently drop out
unless you pass `--include-eol`. Git sources are fetched at the ref,
built, and cached; a moved branch tip rebuilds.

When `--kernel` resolves to two or more kernels, the kernel becomes
another gauntlet dimension: each (test × preset × kernel) tuple is a
distinct nextest case with a kernel-label suffix (see
[test name shapes](../running-tests.md#test-name-shapes)), and
result sidecars partition per kernel under
`target/ktstr/{kernel}-{project_commit}/`. Kernel resolution
finishes for every requested kernel before any test runs — a
missing kernel aborts up front rather than mid-matrix. `host_only`
tests run once regardless of kernel count.

**`--no-perf-mode`** — disable all
[performance mode](../concepts/performance-mode.md) features
(flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit
suppression). Also via `KTSTR_NO_PERF_MODE`.

**`--no-skip-mode`** — convert resource-contention and
host-topology-insufficient skips into hard failures (exit 1 instead
of 0). The default skips so a contended runner does not fail tests
that simply could not start.

**The three profiles** — independent of each other:

- `--release` — the harness build profile. Release mode applies
  stricter assertion thresholds (`gap_threshold_ms` 2000 vs debug's
  3000, `spread_threshold_pct` 15% vs 35%), so tests that barely
  pass in debug may fail under `--release`.
- `--profile NAME` — the cargo build profile for the
  scheduler-under-test (a discovered scheduler package). Defaults to
  `release`; pass `--profile dev` for a fast unoptimized scheduler
  build.
- `--nextest-profile NAME` — the nextest test profile from
  `.config/nextest.toml` (retries, timeouts, output settings).

**`--relevant` / `--base` / `--base-ref` / `--default-branch`** —
narrow the run to only the tests whose scheduler your working-tree
change touches, against a baseline commit (merge-base with `main` by
default). A broad or unattributable change fails safe and runs
everything; a docs-only change runs nothing. The CI-matrix
counterpart is [`affected`](#affected); both are documented in
[CI](../ci.md).

## test

Build the kernel (if needed) and run tests via `cargo nextest run`.
Also available as `cargo ktstr nextest` (a clap alias). Arguments
after `--` are passed through to nextest:

```sh
cargo ktstr test --kernel ../linux                         # everything
cargo ktstr test --kernel 7.0 -- -E 'test(my_test)'        # nextest filter
cargo ktstr test --kernel 7.0 -- --retries 2               # nextest retries
cargo ktstr test --kernel 7.0 -- --features integration    # cargo features
cargo ktstr test --relevant                                # only tests my edits affect
```

A real single-test run against a local kernel tree looks like this.
The source tree was already built, so ktstr resolved it to a cache key
and reused the image:

<!-- captured: cargo ktstr test --kernel ../linux --no-perf-mode -- --features integration -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)' | ktstr 0.24.0-dev | kernel ../linux (b4dc42d2, sched_ext-for-7.2) | verified by independent rerun -->
```text
$ cargo ktstr test --kernel ../linux --no-perf-mode -- --features integration -E 'test(=ktstr/scx_empty_run_exits_under_watchdog)'
cargo ktstr: resolved kernel "path_linux_b5562c"
cargo ktstr: BTF type anchor at ~/ktstr/target/ktstr_btf_anchor.h
...
    Finished `test` profile [unoptimized + debuginfo] target(s) in 27.24s
────────────
 Nextest run ID d79da4e4-df49-4b40-a415-c1dff2739ce6 with nextest profile: default
    Starting 1 test across 120 binaries (13597 tests skipped)
        PASS [   8.601s] (1/1) ktstr::scx_cleanup_test ktstr/scx_empty_run_exits_under_watchdog
────────────
     Summary [   8.642s] 1 test run: 1 passed, 13597 skipped

cargo ktstr: test outputs
  ~/ktstr/target/ktstr/7.1.0-486fe68-dirty
    (1 stats sidecar(s), 0 wprof trace(s) written this run)
```

The VM part of this run took about nine seconds; the longer compile
line was Rust rebuilding the test harness. For a `--kernel <path>`
source tree, a cache hit is announced on stderr as
`cargo ktstr: cache hit for {path} ({cache_key}, built {age} ago)` and
skips the kernel build entirely.

### Per-test exit codes

Each `#[ktstr_test]` process exits with one of three codes, so CI
gates and dashboards can triage runs:

| Code | Verdict | Meaning |
|------|---------|---------|
| `0` | Pass / Skip | Assertions passed, or the test never ran (host too small, resource contention, perf mode unavailable). Skips degrade to pass unless `--no-skip-mode`. |
| `1` | Fail | An assertion failed; an operator `--cpu-cap` the host cannot satisfy; a skip under `--no-skip-mode`; or `expect_err = true` and the test passed. |
| `2` | Inconclusive | A zero-denominator ratio gate could not evaluate — the workload produced no signal to ratio against. |

Exit code `2` is the silent-pass guard: a Pass at a `≤ threshold`
gate run against a `0 / 0` ratio that synthesized to `0.0` would
have shipped a false-green CI run. The harness records Inconclusive
instead (see [Checking](../concepts/checking.md)) and
projects it to a distinct exit code so external tooling can route
the run separately from real regressions. The values are exported as
`EXIT_PASS` / `EXIT_FAIL` / `EXIT_INCONCLUSIVE` in the prelude.

## replay

Re-run only the tests that failed in the last session, from the
result sidecars under `target/ktstr/`:

```sh
cargo ktstr replay              # print the nextest filter (dry-run)
cargo ktstr replay --exec       # actually run it
cargo ktstr replay -E crash     # narrow by test-name substring
cargo ktstr replay --dir PATH   # source sidecars from an archived tree
```

Dry-run is the default: the filter prints to stdout so you can
inspect it (or paste it into CI) before committing to the re-run.
`--profile` / `--nextest-profile` apply with `--exec`. Distinct from
[auto-repro](auto-repro.md), which fires inside the failing test
process; `replay` is post-hoc, across a whole session.

## coverage

Run tests with coverage via `cargo llvm-cov nextest`. Same kernel
resolution, multi-kernel semantics, and common flags as `test`;
arguments after `--` pass through to `cargo llvm-cov nextest`:

```sh
cargo ktstr coverage --kernel ../linux
cargo ktstr coverage -- --workspace --lcov --output-path lcov.info
```

Requires `cargo-llvm-cov` and the `llvm-tools-preview` rustup
component. Guest-side coverage is flushed over shared memory at VM
exit and merged into the report automatically; multi-kernel runs
merge every variant's profraw into a single report.

<a id="profraw-layout"></a>

Profraw files accumulate across runs — including host-side files
that plain `cargo ktstr test` writes next to the cargo-ktstr binary
(an `LLVM_PROFILE_FILE` injection that keeps `default.profraw` out
of your kernel source tree; export `LLVM_PROFILE_FILE` yourself to
opt out). To clean up:

```sh
cargo ktstr llvm-cov clean --profraw-only          # only *.profraw under target/llvm-cov-target/
rm -f target/debug/llvm-cov-target/default-*.profraw
rm -f target/release/llvm-cov-target/default-*.profraw
```

Avoid bare `cargo ktstr llvm-cov clean` (wipes reports too) and
`--workspace` (also runs `cargo clean`).

## llvm-cov

Raw passthrough to `cargo llvm-cov` for subcommands that don't fit
the `coverage` flow (`report`, `clean`, `show-env`), with the same
kernel-resolution plumbing:

```sh
cargo ktstr llvm-cov report --lcov --output-path lcov.info
```

Always pass a subcommand: a bare `cargo ktstr llvm-cov` falls
through to `cargo test`, which skips gauntlet variants and verifier
cells entirely (they exist only under the nextest harness).

## kernel

Manage cached kernel images: `list`, `build`, `clean`. The
standalone `ktstr kernel` subcommands are identical.

### How kernels are discovered {#what-it-does-path-mode-only}

There are two flows, and they intentionally differ:

**Explicit (`--kernel ...`)** — the full pipeline. For a source-tree
path: validate it is a kernel tree, look up the cache (clean git
trees only — a cache hit short-circuits straight to tests),
auto-configure (`make defconfig` if needed, append ktstr's kconfig
fragment, `make olddefconfig`), build with
`make -j$(nproc) KCFLAGS=-Wno-error`, validate that the critical
config options survived (`CONFIG_SCHED_CLASS_EXT`,
`CONFIG_DEBUG_INFO_BTF`, `CONFIG_BPF_SYSCALL`, tracing options —
the kernel build system silently drops options with unmet
dependencies, and each failure gets a remediation hint), generate
`compile_commands.json`, store the result in the cache, and run
nextest with `KTSTR_KERNEL` (and `KTSTR_KERNEL_LIST` for
multi-kernel) exported. Dirty or non-git trees build in place, get a
`_dirty` kernel label, and never cache. Version, range, and git
identifiers download/fetch + build + cache on miss.

**Implicit (no `--kernel`)** — discovery only, no builds. The test
framework checks `KTSTR_TEST_KERNEL`, then `KTSTR_KERNEL`, then the
newest valid cache entry, then local build trees (`./linux`,
`../linux`, `/lib/modules/$(uname -r)/build`), then host images
(`/boot/vmlinuz*`). Whatever pre-built image it finds is used as-is
— nothing is compiled and nothing new lands in the cache.

> [!TIP]
> If you iterate on a local kernel tree, pass `--kernel ../linux`
> (or run `cargo ktstr kernel build --kernel ../linux` once) so the
> build is cached and reused. Implicit discovery will find the tree
> but never populate the cache for it.

### kernel list {#kernel-list}

List cached kernels, newest first:

<!-- captured: cargo ktstr kernel list | ktstr 0.23.0 -->
```text
$ cargo ktstr kernel list
cache: ~/.cache/ktstr/kernels
  KEY                                              VERSION      SOURCE   ARCH    BUILT
  7.0.14-tarball-x86_64-kcabd40422                 7.0.14       tarball  x86_64  2026-07-04T23:06:12Z
  local-b4dc42d-x86_64-kcabd40422                  7.1.0        local    x86_64  2026-07-04T23:00:04Z
  local-5123e5a-x86_64-cfg1982ad42-kcabd40422      7.1.0        local    x86_64  2026-07-04T22:20:13Z
...
  local-c5d2724-x86_64-kcabd40422                  7.0.0        local    x86_64  2026-07-04T22:05:35Z
...
  7.0.13-tarball-x86_64-kc5f2631f0                 7.0.13       tarball  x86_64  2026-06-22T21:54:56Z (stale kconfig)
  7.1.1-tarball-x86_64-kc5f2631f0                  7.1.1        tarball  x86_64  2026-06-22T21:46:19Z (stale kconfig)
warning: entries marked (stale kconfig) were built against a different ktstr.kconfig. Rebuild with: kernel build --force --kernel <entry version> (add --extra-kconfig PATH if the entry also carries the (extra kconfig) tag).
```

`(stale kconfig)` marks entries built against a different ktstr
kconfig fragment (they are rebuilt automatically when requested);
`(EOL)` marks entries whose series has left kernel.org's active
releases list. `--json` emits the same data for scripting. With
`--kernel START..END`, `list` switches to preview mode: it prints
the versions the range expands to without downloading or building
anything — the cheap answer to "what does `6.12..6.16` actually
cover?".

### kernel build {#kernel-build}

Download (or use a source tree / git ref), build, and cache one
kernel or a range:

```sh
cargo ktstr kernel build                        # latest stable from kernel.org
cargo ktstr kernel build --kernel 6.14.2        # specific version
cargo ktstr kernel build --kernel 6.12          # latest 6.12.x patch release
cargo ktstr kernel build --kernel 6.11..6.14    # every release in the range
cargo ktstr kernel build --kernel ../linux      # local source tree
cargo ktstr kernel build --force --kernel 6.14.2
```

With no `--kernel`, it builds the latest stable series that has had
at least 8 maintenance releases — keeping CI off brand-new majors
whose early builds are more likely to break. Already-cached entries
are skipped unless `--force`.

| Flag | Description |
|------|-------------|
| `--force` | Rebuild even if cached. |
| `--clean` | `make mrproper` first (source-tree builds only). |
| `--cpu-cap N` | Reserve exactly N host CPUs for the build (build parallelism and a cgroup sandbox follow). See [Resource Budget](../concepts/resource-budget.md). |
| `--extra-kconfig PATH` | Extra kconfig fragment merged over ktstr's (user values win); lands in its own cache slot. |
| `--skip-sha256` | Skip tarball checksum verification (emits a warning). |
| `--include-eol` | With a range, also build EOL series from the linux-stable mirror. |

A bare relative name is read as a cache key — prefix a relative
source directory with `./`.

### kernel clean {#kernel-clean}

```sh
cargo ktstr kernel clean --keep 3                 # keep the 3 most recent valid entries
cargo ktstr kernel clean --corrupt-only --force   # drop only broken entries
```

Corrupt entries (missing metadata or image) never consume a `--keep`
slot. `--force` skips the confirmation prompt (required
non-interactively).

## verifier

Sweep every `declare_scheduler!`-registered scheduler through the
real kernel's BPF verifier across topologies, checking verify +
attach + dispatch per cell. Takes the same `--kernel` grammar as
`test` — repeatable, ranges (`6.12..6.14`, with `--include-eol` for
end-of-life series), paths, cache keys, git refs:

```sh
cargo ktstr verifier --kernel 7.0
cargo ktstr verifier --scheduler scx-ktstr     # one scheduler
cargo ktstr verifier --raw                     # no cycle collapse
```

See [BPF Verifier Sweep](verifier.md) for the cell model, real
output, and the kernels-filter contract.

## shell

Boot any kernel ktstr can resolve and land in a busybox shell inside
the VM — the fastest way to poke at a kernel you just built, check
what a guest actually exposes, or reproduce a failing test's
environment by hand:

```sh
cargo ktstr shell --test my_failing_test     # boot the exact VM a test gets
cargo ktstr shell --kernel ../linux          # local tree
cargo ktstr shell --kernel ./arch/x86/boot/bzImage
```

`--test NAME` derives topology, memory, and include files from a
registered `#[ktstr_test]` (mutually exclusive with `--topology` /
`--memory-mib`; `-i` is additive) — so "why does this only fail on
odd-3llc" becomes an interactive session in that exact machine.

For scripted checks, `--exec` runs a command and exits with its
status:

<!-- captured: cargo ktstr shell --kernel local-8cd2b47... --no-perf-mode --exec 'uname -r; cat /sys/kernel/sched_ext/state; nproc' | ktstr 0.23.0 | kernel v7.1 + sched_ext_exit patch -->
```text
$ cargo ktstr shell --kernel ../linux --no-perf-mode \
    --exec 'uname -r; cat /sys/kernel/sched_ext/state; nproc'
7.1.0-00001-g8cd2b47fb188
disabled
1
```

The standalone [`ktstr shell`](ktstr.md#shell) is the same boot flow
minus the cargo integration (`--test`, raw image paths).

## export

Export a registered test as a self-extracting `.run` file that
reproduces the scenario on bare metal, no VM: the ktstr binary, the
scheduler binary, and every declared include file, embedded in a
bash preamble that validates root, sched_ext support, cgroup2, the
no-other-scheduler-attached invariant, and topology compatibility
before launching.

```sh
cargo ktstr export my_test -o /tmp/my_test.run
<!-- captured: cargo ktstr export sched_basic_proportional --package ktstr -o /tmp/sched_basic_proportional.run | ktstr 0.23.0 -->
```

```text
wrote /tmp/sched_basic_proportional.run (90074903 bytes archive, 0 include files)
```

The generated script opens with the frozen test specification and
the preflight checks:

<!-- captured: head of the generated /tmp/sched_basic_proportional.run | ktstr 0.23.0 -->
```sh
#!/bin/bash
# Generated by `cargo ktstr export`. Do not edit; regenerate to update.
set -euo pipefail

# --- frozen test specification ---
KTSTR_TEST_NAME=sched_basic_proportional
KTSTR_SCHED_NAME=ktstr_sched
KTSTR_GIT_HASH=73730e0
NEED_LLCS=1
NEED_CORES_PER_LLC=2
NEED_THREADS_PER_CORE=1
NEED_NUMA_NODES=1
TEST_DURATION_SECS=12
TEST_WATCHDOG_SECS=15
```

Scheduler choice, scheduler args, and topology are frozen at export
time; `--duration`, `--watchdog-timeout`, and `--quiet` can be
overridden at `.run` invocation. `--package NAME` scopes the
workspace search (and disambiguates duplicate test names —
otherwise the first matching binary in path order wins);
`--release` builds the embedded binaries with the release profile to
match how you will run them.

Not exportable (rejected with actionable errors): `host_only` tests,
tests using `bpf_map_write` (they need the host-side probe surface),
and `KernelBuiltin` schedulers.

## completions

```sh
cargo ktstr completions bash >> ~/.local/share/bash-completion/completions/cargo
cargo ktstr completions zsh > ~/.zfunc/_cargo-ktstr
cargo ktstr completions fish > ~/.config/fish/completions/cargo-ktstr.fish
```

`SHELL` is one of `bash`, `zsh`, `fish`, `elvish`, `powershell`;
`--binary` overrides the completion target (default `cargo`).

## stats {#stats}

<a id="stats-show-host"></a>
<a id="list-values"></a>

Sidecar analysis for past runs: `stats` (analysis of the newest
run), `stats list` (run table), `stats list-metrics` (the regression
metric registry), `stats list-values` (distinct filter values in the
pool), `stats show-host --run ID` (archived host context), and
`stats explain-sidecar --run ID` (why optional fields are absent).
`cargo ktstr show-host` prints the same host context live, and
`cargo ktstr show-thresholds TEST` prints the merged assertion
thresholds a test will run with. See
[Runs and Regression Gates](runs.md).

## perf-delta {#perf-delta}

Compare `performance_mode` test metrics between HEAD and a baseline
commit, exiting non-zero when enough metrics regress to trip the
gate:

```sh
cargo ktstr perf-delta --noise-adjust 5 --kernel 6.14
```

Baseline resolution, the `--noise-adjust` statistics, and the full
flag set are documented in
[Runs and Regression Gates](runs.md#perf-delta); the worked
walkthrough is [A/B Compare Branches](../recipes/ab-compare.md).

## affected {#affected}

<a id="relevant"></a>

Emit the scheduler packages a `base..HEAD` diff affects, as a JSON
array for a GitHub Actions dynamic matrix
(`["scx_lavd","scx_rusty"]`). Attribution is fail-safe: any
uncertainty widens to the full set, and only a strictly docs-only
change emits `[]`. `--relevant` is the same engine applied locally
to narrow `test` / `coverage` / `perf-delta` runs. Both are
documented with the complete CI workflow in [CI](../ci.md).

## locks {#locks}

Enumerate every ktstr flock held on this host, read-only, naming
holder PIDs and cmdlines — the troubleshooting companion when a run
is stalled behind a peer's reservation. Identical to
[`ktstr locks`](ktstr.md#locks), where the lock roots and real
output are documented.
