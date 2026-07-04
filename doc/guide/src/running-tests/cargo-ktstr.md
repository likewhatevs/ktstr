# cargo-ktstr

`cargo ktstr` is a cargo plugin for kernel build, cache, and test
workflow. Subcommands in `--help` order: `test` (alias: `nextest`),
`coverage`, `llvm-cov`, `stats`, `replay`, `perf-delta`, `kernel`,
`verifier`, `completions`,
`show-host`, `show-thresholds`, `export`, `locks`, `shell`.

## test

Build the kernel (if needed) and run tests via `cargo nextest run`.
Also available as `cargo ktstr nextest` — a visible clap alias that
expands to the same subcommand, so the two forms are interchangeable.

```sh
cargo ktstr test                                               # auto-discover kernel
cargo ktstr test --kernel ../linux                             # local source tree
cargo ktstr test --kernel 6.14.2                               # version (auto-downloads on miss)
cargo ktstr test --kernel 6.14.2-tarball-x86_64-kc...          # cache key (from kernel list)
cargo ktstr test --kernel 6.12..6.14                           # range: every stable+longterm release in [6.12, 6.14]
cargo ktstr test --kernel git+https://example.com/r.git#tag=v6.14  # git URL + tag (or #branch=NAME)
cargo ktstr test --kernel git+https://example.com/r.git#sha=<full-40-hex-commit>  # a specific commit
cargo ktstr test --kernel 6.14.2 --kernel 6.15.0               # multi-kernel: repeatable
cargo ktstr test --release                                     # release profile (stricter assertions)
cargo ktstr test --relevant                                    # only tests the current diff affects
```

`--relevant` narrows the run to only the tests whose scheduler your
working-tree change touches (committed `base..HEAD` ∪ uncommitted +
untracked edits), intersected with any `-E` you pass. It is the local
counterpart to the `affected` CI matrix — see
[affected / --relevant](#relevant) for the attribution model, the
`--base` / `--base-ref` / `--default-branch` baseline knobs, and the
fail-safe behavior.

`--kernel` is **repeatable** and accepts a path, version string,
cache key, version range (`START..END`), or git source
(`git+URL#tag=NAME`, `#branch=NAME`, or `#sha=<40-hex>`). When absent, the test framework discovers a kernel
from `KTSTR_TEST_KERNEL`, then `KTSTR_KERNEL`, then falls back to
cache and filesystem lookup. When `--kernel` is a path,
cargo-ktstr configures and builds the kernel before running tests.
Version strings auto-download and build on cache miss (both
explicit patch versions like `6.14.2` and major.minor prefixes like
`6.14`). Cache keys resolve from the cache only — they error if not
cached (run `cargo ktstr kernel list` to see available keys).

Ranges (`START..END`) expand against kernel.org's `releases.json`
to every `stable` and `longterm` release whose version sits inside
`[START, END]` inclusive (mainline / linux-next rows are dropped).
The endpoints themselves do NOT need to appear in `releases.json` —
`6.10..6.16` brackets the surviving releases even if `6.10` and
`6.16` have aged out.

Both endpoints are series-inclusive: a 2-component `MAJOR.MINOR`
endpoint names the whole series, so `6.11..6.14` covers every `6.14.N`
point release (not just `6.14.0`). Spell an endpoint with an explicit
patch (`6.14.2`) to make it an exact bound — `6.11..6.14.2` stops at
`6.14.2`.

By default the expansion sees only series still listed in
`releases.json`, so a series that has reached end-of-life is
silently absent — `6.11..6.14` collapses to just the 6.11 series if
6.12 and 6.13 are EOL. Pass `--include-eol` to additionally enumerate
EOL series from the gregkh linux-stable mirror's tags: the range then
contributes the highest point release of every `stable` series inside
`[START, END]`, maintained or not. If the mirror tag list cannot be
fetched, expansion falls back to the active-release set with a
warning. `--include-eol` is accepted on every command that expands a
range (`test`, `coverage`, `llvm-cov`, `verifier`,
`kernel list`, and `kernel build`); it has no effect on a
single version, path, cache key, or git source.

Git sources (`git+URL#tag=NAME`, `#branch=NAME`, or `#sha=<40-hex>`)
are fetched at the given ref (GitHub via a codeload snapshot, other
hosts via a shallow clone), built, and cached. A repeat invocation against an
unchanged branch tip lands a cache hit; a moved tip rebuilds.

### Multi-kernel: kernel as a gauntlet dimension

When `--kernel` resolves to **two or more kernels** (multiple
`--kernel` flags, or a single `--kernel START..END` range that
expands to several releases), cargo-ktstr resolves all kernels
upfront and exports the resolved set to `cargo nextest` via the
`KTSTR_KERNEL_LIST` env var. The test binary's gauntlet expansion
adds the kernel as an additional dimension to the gauntlet
cross-product, so each `(test × scenario × topology × kernel)`
tuple becomes a distinct nextest test case. Two name shapes carry
the kernel suffix:

- **Base tests**: `ktstr/{name}/{kernel_label}` — one variant per
  registered `#[ktstr_test]` per kernel.
- **Gauntlet variants**: `gauntlet/{name}/{preset}/{kernel_label}` —
  one variant per (test × topology preset × kernel).

Single-kernel runs (zero or one resolved kernel) keep the
historical name shapes `ktstr/{name}` and
`gauntlet/{name}/{preset}` with no kernel suffix, so
existing CI baselines and per-test config overrides keep matching.

Kernel labels are semantic, operator-readable identifiers
sanitized to `kernel_[a-z0-9_]+`:

- Version / range expansion → `kernel_6_14_2`, `kernel_6_15_rc3`
- Cache key → version prefix only (`kernel_6_14_2` from
  `6.14.2-tarball-x86_64-kc<hash>`)
- Git source → `kernel_git_{owner}_{repo}_{kind}_{ref}` (e.g.
  `kernel_git_tj_sched_ext_branch_for_next` from
  `git+https://github.com/tj/sched_ext#branch=for-next`)
- Path → `kernel_path_{basename}_{hash6}` (e.g.
  `kernel_path_linux_a3f2b1`); the 6-char crc32 of the canonical
  path disambiguates two `linux` directories under different
  parents. Dirty-tree builds (uncommitted source changes, mid-build
  worktree mutations, or non-git trees) append `_dirty` to the
  label — e.g. `kernel_path_linux_a3f2b1_dirty` — so the test
  report distinguishes the non-reproducible run from a subsequent
  clean rebuild of the same path.
- Local cache entry → `kernel_local_{hash6}` (first 6 chars of
  the source tree's git short_hash, captured at cache-store
  time) or `kernel_local_unknown` for non-git trees. The
  hash6 keeps two distinct local trees from collapsing to the
  same label; the `unknown` literal is the shared bucket for
  every non-git tree (no discriminator exists at the cache
  layer to spread them apart).

Filter with nextest's `-E 'test(kernel_6_14)'` to pick a single
kernel from a multi-kernel matrix; nextest's parallelism, retries,
and `--ignored` flag all apply natively. Sidecars partition per
kernel: each kernel runs in its own
`target/ktstr/{kernel}-{project_commit}/` directory keyed on the
resolved kernel's identity and the project tree's HEAD short hex
(with `-dirty` suffix when the worktree differs). Coverage profraw does NOT partition
per kernel — `__llvm_profile_write_buffer` writes flat into
`target/llvm-cov-target/` with PID-keyed filenames
(`ktstr-test-{pid}-{counter}.profraw`), and cargo-llvm-cov merges
every variant's profraw automatically into the single output
report.

Build / download / clone failures abort BEFORE any test runs — a
missing kernel can't be tested, and continuing would mask which
kernel was requested-but-unavailable in the operator-visible error
stream. Test failures within a kernel are nextest-handled
normally.

**`host_only` tests under multi-kernel**: tests marked
`host_only` (those that run on the host without booting a VM)
skip the kernel suffix and list / run **once** regardless of
`KTSTR_KERNEL_LIST` cardinality. The dispatch sites
(`list_tests`, `list_tests_budget`, and `--exact`'s
`run_host_only_test` in `src/test_support/dispatch.rs`) all gate
on `entry.host_only` before consulting the resolved kernel set,
so a host-side test never observes the kernel directory and
multiplying it across kernels would just run N copies of
identical work for no signal.

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` (repeatable) | auto | Kernel identifier: path, version, cache key, range (`START..END`), or git source (`git+URL#tag=NAME`). Repeatable; a multi-kernel set fans the gauntlet across kernels. |
| `--include-eol` | off | When a `--kernel START..END` range is present, also enumerate EOL `stable` series from the gregkh linux-stable mirror's tags, not just the active series in `releases.json`. Each in-range EOL series contributes its highest point release. No effect on a single kernel, path, cache key, or git source. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |
| `--no-skip-mode` | off | Convert resource-contention and host-topology-insufficient skips into hard test failures (exit `1` instead of `0`). Default behavior skips so a contended runner does not fail tests that simply could not start; setting this flag opts into "if the test cannot run, the test fails". Exports `KTSTR_NO_SKIP_MODE=1` for the test binary. |
| `--release` | off | Build and run tests with the release profile (`--cargo-profile release` to nextest). Release mode applies **stricter assertion thresholds** (`gap_threshold_ms` 2000 vs debug's 3000, `spread_threshold_pct` 15% vs debug's 35%) — tests that barely pass in debug may fail under `--release`. `catch_unwind`-based tests and tests gated on `#[cfg(debug_assertions)]` are skipped. |
| `--profile NAME` | release | Cargo BUILD profile for the scheduler-under-test (a `SchedulerSpec::Discover` package): drives `cargo build -p <scheduler> --profile <NAME>` via the `KTSTR_SCHEDULER_PROFILE` env. Omitted, the scheduler builds `release` — an optimized scheduler is the only sensible default. INDEPENDENT of `--release` (the harness build profile): pass `--profile dev` for a fast unoptimized scheduler build, or any custom `[profile.<name>]`. Distinct from `--nextest-profile` (the nextest test profile). |
| `--nextest-profile NAME` | nextest default | Nextest TEST profile (`.config/nextest.toml`), forwarded to nextest as `--profile <NAME>` (retry / timeout / output settings). Distinct from `--profile` (the scheduler's cargo BUILD profile) and `--release` (the harness's cargo build profile). |
| `--relevant` | off | Narrow the run to only the tests whose scheduler the working-tree change touches (committed `base..HEAD` ∪ uncommitted + untracked), intersected with any `-E`. Broad / unattributable change → everything (fail-safe); docs-only → nothing. See [affected / --relevant](#relevant). |
| `--base COMMIT` | — | With `--relevant`: explicit attribution baseline (skips merge-base). Ignored without `--relevant`. |
| `--base-ref REF` | — | With `--relevant`: ref to merge-base against (defaults to `$GITHUB_BASE_REF` on a PR, else `--default-branch`). Ignored without `--relevant`. |
| `--default-branch BRANCH` | `main` | With `--relevant`: merge-base target when no `--base` / `--base-ref` / `$GITHUB_BASE_REF`. Ignored without `--relevant`. |

### What it does (path mode only)

These steps run only when `--kernel` is a source directory path.
Cached version and cache-key identifiers skip straight to test
execution (step 6); uncached version identifiers run through
download + configure + build + cache-store first. Ranges fan out
to per-version resolution (every release downloads + builds +
caches independently if not already present); git sources clone
shallow at the ref, build, and cache. Multi-kernel resolution
finishes for every requested kernel BEFORE step 6 — the
cargo-nextest invocation in step 6 sees the complete kernel set
as a single `KTSTR_KERNEL_LIST` export, so nextest fans the
gauntlet across kernels in a single run.

For path mode, the source tree is gix-discovered and classified
as either *clean* (HEAD reachable, index matches HEAD, worktree
matches index) or *dirty or non-git* (any tracked-file diff, or
the directory is not a git repo at all). The cache is keyed in
one of three shapes:

- `local-{hash7}-{arch}-kc{suffix}` — clean git tree, no user
  `.config` file in the source tree yet (build will run `make
  defconfig`). `{hash7}` is the source tree's HEAD short hash;
  `{suffix}` distinguishes ktstr framework kconfig fragments.
- `local-{hash7}-{arch}-cfg{user_config}-kc{suffix}` — clean git
  tree with a user `.config` whose CRC32 hash discriminates
  distinct configurations against the same commit, so iterative
  `.config` edits at a fixed commit populate distinct cache
  entries instead of colliding.
- `local-unknown-{path_hash}-{arch}-kc{suffix}` — dirty / non-git
  tree (HEAD does not describe the source). `{path_hash}` is the
  full 8-char (32-bit) CRC32 of the canonical source path so two
  parallel `cargo ktstr test --kernel ./linux-a` and
  `--kernel ./linux-b` runs do not collide on the same
  `local-unknown-...` slot.

Dirty / non-git trees never cache — the build pipeline runs in
the source directory, the kernel label gets a `_dirty` suffix,
and a subsequent run of the same path that goes clean produces a
distinct cache entry under the clean shape.

1. **Source-tree validation** — verifies `<kernel>/Makefile` and
   `<kernel>/Kconfig` both exist. If either is missing, bails
   with `not a kernel source tree`.
2. **Cache lookup** (clean trees only) — looks up the
   `local-{hash7}-{arch}[-cfg{user_config}]-kc{suffix}` key
   (the `cfg` segment present iff a user `.config` exists in the
   source tree). **Cache hit short-circuits to step 6**:
   cargo-ktstr exports the cache entry directory via
   `KTSTR_KERNEL` and emits a `cargo ktstr: cache hit for
   {input_path} ({cache_key}, built {age} ago)` line on stderr
   (the `, built {age} ago` suffix is omitted when the timestamp
   is unparseable or future-dated). Cache miss continues to
   step 3.
3. **Auto-configure** — if `<kernel>/.config` lacks the
   `CONFIG_SCHED_CLASS_EXT=y` sentinel, runs `make defconfig`
   (when no `.config` exists), appends `ktstr.kconfig` to
   `.config`, then runs `make olddefconfig`.
4. **Kernel build** — runs `make -j$(nproc) KCFLAGS=-Wno-error`,
   then runs `validate_kernel_config` to verify critical config
   options (`CONFIG_SCHED_CLASS_EXT`, `CONFIG_DEBUG_INFO_BTF`,
   `CONFIG_BPF_SYSCALL`, `CONFIG_FTRACE`, `CONFIG_KPROBE_EVENTS`,
   `CONFIG_BPF_EVENTS`) survived the build — the kernel build
   system silently disables options whose dependencies are not
   met, and the validator surfaces those failures with a per-
   option remediation hint. `make` handles the no-op case when
   the kernel is already built. For dirty / non-git trees this
   is the unconditional path; for clean trees, only reached on
   cache miss.
5. **compile_commands.json + cache store** — runs `make
   compile_commands.json` (skipped only for transient temp
   directories like extracted tarballs) so LSP / clangd work
   against the local tree. Then for clean trees, the kernel
   image + stripped vmlinux are persisted under the resolved
   `local-{hash7}-{arch}[-cfg{user_config}]-kc{suffix}` key with
   `metadata.json` recording the source tree path. A post-build
   re-check of the dirty state catches mid-build mutations
   (worktree edits or commits that happened during `make`) and
   skips the cache store on either signal so a racing-write build
   can not land under a stale identity. Dirty / non-git trees
   skip the cache store unconditionally (no stable HEAD identity
   for the cache key) but still get `compile_commands.json`.
6. **Test execution** — runs `cargo nextest run` once (spawned and
   waited on via `Command::status()`, not exec) with `KTSTR_KERNEL`
   set in the environment (single-kernel) or with
   both `KTSTR_KERNEL` and `KTSTR_KERNEL_LIST` (multi-kernel; the
   latter encodes the resolved kernel set as
   `label1=path1;label2=path2;…`). For clean Path-spec resolution
   `KTSTR_KERNEL` points at the cache entry directory; for dirty
   or non-git trees it points at the source tree directly. The
   test binary's gauntlet expansion adds the kernel as a fifth
   dimension when the list carries 2+ entries; nextest's
   parallelism, retries, and `-E` filtering apply natively to
   every (test × kernel) variant.

> **Implicit vs explicit kernel discovery diverge**: `cargo ktstr
> test --kernel ../linux` (explicit Path spec) routes through the
> cache pipeline above — the source tree is gix-classified, the
> `local-{hash7}-{arch}[-cfg{user_config}]-kc{suffix}` cache key is computed, the kernel is built
> (or short-circuited on cache hit), and the cache entry directory
> is exported via `KTSTR_KERNEL`. `cargo ktstr test` (no `--kernel`
> flag) does NOT run the build pipeline or produce a new cache
> entry. The test binary's `find_kernel` chain reads existing
> cache entries (most-recent-valid first; entries built with a
> different kconfig fragment are skipped) and falls back to local
> build trees (`./linux`, `../linux`) and host paths. Whatever
> pre-built image it finds is returned as-is — no cache key is
> computed for source trees discovered on the filesystem, no
> `make` is invoked, and the result does not land in the kernel
> cache for a future `cache_key`-keyed lookup. The `KTSTR_KERNEL`
> env var with a path value follows this same direct-image flow
> — the cache write path is reached only via the `cargo ktstr`
> `--kernel` argument (or via `cargo ktstr kernel build --kernel
> ../linux` as an explicit cache-populate step). Pass
> `--kernel ../linux` to opt into the cache pipeline so a clean
> tree's build is stored once and reused on subsequent runs.

### Passing nextest arguments

Arguments after `test` are passed through to `cargo nextest run`:

```sh
cargo ktstr test -- -E 'test(my_test)'        # nextest filter
cargo ktstr test -- --workspace               # all workspace tests
cargo ktstr test -- --retries 2               # nextest retries
```

### Per-test exit codes

Each `#[ktstr_test]` process exits with a code that projects the
`Fail > Inconclusive > Pass > Skip` verdict lattice to three
values. CI gates and dashboards triage runs by exit code:

| Code | Verdict      | Meaning                                                                                                                                                |
|------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `0`  | Pass / Skip  | All assertions passed, or the test never ran (host topology insufficient, resource contention, `performance_mode` unavailable on a host too small for the topology, or a per-test `cpu_budget` exceeding the allowed CPUs). These host-insufficiency skips degenerate to Pass at the process boundary — unless `--no-skip-mode` / `KTSTR_NO_SKIP_MODE` promotes them to exit `1`. |
| `1`  | Fail         | At least one assertion failed; OR an operator `--cpu-cap` / `KTSTR_CPU_CAP` the host cannot satisfy (`CpuBudgetUnsatisfiable`, an unconditional hard error — an author's per-test `cpu_budget` over the allowance skips in the `0` row instead); OR a host-insufficiency skip (`PerfModeUnavailable`, resource contention, topology insufficient) run under `--no-skip-mode` / `KTSTR_NO_SKIP_MODE`; OR `expect_err = true` and the test produced a Pass / Inconclusive (an `expect_err` test whose gate could not evaluate is unsatisfied just as it would be on a Pass). |
| `2`  | Inconclusive | A zero-denominator ratio gate could not evaluate — the workload produced no signal to ratio against, so neither pass nor fail is truthful.             |

Exit code `2` is the silent-pass guard: a Pass at a `≤ threshold`
gate run against a `0 / 0` ratio that synthesized to `0.0` would
have shipped a false-green CI run. The harness records Inconclusive
in that case (see [Verdict outcomes](../concepts/checking.md#verdict-outcomes))
and the dispatch layer projects it to a distinct exit code so
external tooling can route the run separately from real
regressions.

The integer values are also exposed as `pub const`s for tooling
that drives the harness programmatically:

```rust,ignore
use ktstr::prelude::{EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE};
```

## replay

Re-run only the tests that failed in the last session, by reading
sidecars under `target/ktstr/` (or `--dir`) and emitting a nextest
filter expression that targets exactly the failed set:

```sh
cargo ktstr replay              # print the nextest filter (dry-run)
cargo ktstr replay --exec       # invoke `cargo nextest run -E <filter>`
cargo ktstr replay -E starve    # narrow the failed-sidecar selection by substring
cargo ktstr replay --dir PATH   # source sidecars from an archived tree
```

Default behaviour is dry-run: the filter prints to stdout so it
can be piped into nextest by hand or pasted into a CI pipeline
before committing to the re-run. `--exec` skips the dry-run and
invokes nextest directly.

Distinct from the in-VM auto-repro (`auto_repro = true` on
`KtstrTestEntry`), which fires within the same test process when
the primary run fails; `replay` is post-hoc, after the test
process has exited, for the CI-friendly "re-run last session's
failures against the new code" workflow.

| Flag | Default | Description |
|------|---------|-------------|
| `--dir PATH` | `target/ktstr/` | Override the sidecar root. Same semantics as `cargo ktstr stats list-values --dir`. |
| `-E, --filter SUBSTR` | -- | Substring filter on `test_name` (case-sensitive). |
| `--exec` | dry-run | Invoke `cargo nextest run` with the computed filter instead of printing it. |
| `--profile NAME` | release | Cargo BUILD profile for the scheduler-under-test (see `cargo ktstr test --profile`). Only meaningful with `--exec` (the dry-run path runs nothing). |
| `--nextest-profile NAME` | nextest default | Nextest TEST profile forwarded to the re-run `cargo nextest run` as `--profile <NAME>`. Only meaningful with `--exec`. |

## coverage

Build the kernel (if needed) and run tests with coverage via
`cargo llvm-cov nextest`. Same kernel resolution and multi-kernel
semantics as `test`: `--kernel` is repeatable; multi-kernel runs
add the kernel suffix to every test name and partition the
sidecar tree per kernel via
`target/ktstr/{kernel}-{project_commit}/`, where `{project_commit}`
is the project HEAD short hex (with `-dirty` when the worktree
differs). Coverage profraw lands flat in
`target/llvm-cov-target/` with PID-keyed filenames — it does
NOT partition per kernel — and cargo-llvm-cov merges every
variant's profraw automatically into the single output report.

```sh
cargo ktstr coverage                                               # auto-discover kernel
cargo ktstr coverage --kernel ../linux                             # local source tree
cargo ktstr coverage --kernel 6.14.2                               # version (auto-downloads on miss)
cargo ktstr coverage --kernel 6.14.2 --kernel 6.15.0               # multi-kernel coverage matrix
cargo ktstr coverage --release                                     # release profile (stricter assertions)
cargo ktstr coverage -- --workspace --lcov --output-path lcov.info # lcov output
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` (repeatable) | auto | Same shapes and multi-kernel semantics as `cargo ktstr test --kernel`: each (test × kernel) variant runs as its own nextest subprocess so cargo-llvm-cov merges every variant's profraw automatically. |
| `--include-eol` | off | Same as `cargo ktstr test --include-eol`: when a `--kernel START..END` range is present, also expand EOL `stable` series from the gregkh linux-stable mirror. No effect on a single kernel. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |
| `--no-skip-mode` | off | Convert resource-contention and host-topology-insufficient skips into hard test failures. Same semantics as on `test`; exports `KTSTR_NO_SKIP_MODE=1` for the test binary. |
| `--release` | off | Collect coverage with the release profile (`--cargo-profile release` to llvm-cov nextest). Same stricter-threshold caveats as `test --release` — release mode applies `gap_threshold_ms=2000` / `spread_threshold_pct=15%`, and skips `catch_unwind`-based tests along with `#[cfg(debug_assertions)]`-gated tests. |
| `--profile NAME` | release | Cargo BUILD profile for the scheduler-under-test (see `cargo ktstr test --profile`). Omitted, the scheduler builds `release`; INDEPENDENT of `--release`. |
| `--nextest-profile NAME` | nextest default | Nextest TEST profile forwarded to `cargo llvm-cov nextest` as `--profile <NAME>` (see `cargo ktstr test --nextest-profile`). |
| `--relevant` | off | Narrow the run to only the tests whose scheduler the working-tree change touches, intersected with any `-E` (see `cargo ktstr test --relevant`). Broad change → everything (fail-safe); docs-only → nothing. See [affected / --relevant](#relevant). |
| `--base COMMIT` | — | With `--relevant`: explicit attribution baseline (skips merge-base). Ignored without `--relevant`. |
| `--base-ref REF` | — | With `--relevant`: ref to merge-base against (defaults to `$GITHUB_BASE_REF` on a PR, else `--default-branch`). Ignored without `--relevant`. |
| `--default-branch BRANCH` | `main` | With `--relevant`: merge-base target when no `--base` / `--base-ref` / `$GITHUB_BASE_REF`. Ignored without `--relevant`. |

Requires `cargo-llvm-cov` and the `llvm-tools-preview` rustup
component:

```sh
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
```

### Passing arguments

Arguments after `coverage` are passed through to
`cargo llvm-cov nextest`:

```sh
cargo ktstr coverage -- --workspace --profile ci --lcov --output-path lcov.info
cargo ktstr coverage -- --features integration
```

### profraw layout

Three populations of `*.profraw` files arise from `cargo ktstr`
runs. They land in different directories and are not all
collected by the same workflow:

| Filename shape | Directory | Producer | Collected by |
|---|---|---|---|
| `default-{pid}-{binary_hash}.profraw` | parent of `cargo-ktstr` binary, joined with `llvm-cov-target/` (e.g. `target/{profile}/llvm-cov-target/` for `cargo run --bin cargo-ktstr`, or `~/.cargo/bin/llvm-cov-target/` for an installed binary) | host-side `cargo ktstr test` (via `LLVM_PROFILE_FILE` injection) | not auto-collected; needs an explicit `cargo llvm-cov` report invocation |
| cargo-llvm-cov-managed (shape set by the outer harness) | `target/llvm-cov-target/` (workspace target dir, NOT under `{profile}`) | host-side `cargo ktstr coverage` (cargo-llvm-cov sets its own `LLVM_PROFILE_FILE`) | merged into the `cargo ktstr coverage` report automatically |
| `ktstr-test-{pid}-{counter}.profraw` | parent of the test binary's `LLVM_PROFILE_FILE` env var, falling back to `<test-binary parent>/llvm-cov-target/` (typically `target/{profile}/deps/llvm-cov-target/` when no env override is in play); under `cargo ktstr test`, inherits the host-side injected dir, so co-locates with `default-{pid}-{binary_hash}.profraw` | guest-side `__llvm_profile_write_buffer` flushed via the SHM ring at VM exit | merged into the `cargo ktstr coverage` report automatically |

`cargo ktstr test` injects `LLVM_PROFILE_FILE` (added to prevent
`default.profraw` leaking into a kernel source tree when the
shell cwd was the kernel dir; see
[Stale `vmlinux.btf` or `default.profraw`](../troubleshooting.md#stale-vmlinuxbtf-or-defaultprofraw-in-kernel-source-tree)).
The resulting host-side `default-{pid}-{binary_hash}.profraw`
files do NOT land in the `target/llvm-cov-target/` directory
that `cargo ktstr coverage` (cargo-llvm-cov) reads; they are NOT
picked up by a later `cargo ktstr coverage` run unless you
explicitly include them in a `cargo llvm-cov report`
invocation pointed at the cargo-ktstr binary's `llvm-cov-target/`
directory.

To clean accumulated profraw between runs:

```sh
# Remove ONLY *.profraw under target/llvm-cov-target/ (top-level glob, non-recursive):
cargo ktstr llvm-cov clean --profraw-only

# Drop host-side test-path profraw next to the cargo-ktstr binary.
# Run only the line(s) matching how cargo-ktstr was launched —
# the brace-list form is bash-only, so each path is its own command
# for portable POSIX shells (sh / dash):
rm -f target/debug/llvm-cov-target/default-*.profraw
rm -f target/release/llvm-cov-target/default-*.profraw

# If ktstr was installed via `cargo install`:
rm -f ~/.cargo/bin/llvm-cov-target/default-*.profraw
```

`--profraw-only` is the safe default: it removes only `*.profraw`
files at the top level of `target/llvm-cov-target/` (the cargo-
llvm-cov-managed dir) and leaves coverage reports, profdata, and
build artifacts intact. It does NOT touch the `default-*.profraw`
files next to the cargo-ktstr binary (under
`target/{profile}/llvm-cov-target/` for `cargo run` / `cargo build`,
or `~/.cargo/bin/llvm-cov-target/` for `cargo install`-deployed
binaries) produced by the host-side injection — remove those with
the explicit `rm -f` lines above for whichever launch mode you use.
Avoid `cargo ktstr llvm-cov clean` without arguments (recursively
wipes all of `target/llvm-cov-target/`, including reports) and
`--workspace` (additionally runs `cargo clean` on workspace
packages, removing build artifacts); both are destructive beyond
profraw.

To opt out of the host-side `LLVM_PROFILE_FILE` injection
entirely, export `LLVM_PROFILE_FILE` yourself before running
`cargo ktstr test` — the injector only fires when the env is
absent, so an explicit operator setting takes precedence.

## llvm-cov

Raw passthrough to `cargo llvm-cov` with arbitrary arguments. Use
this for `llvm-cov` subcommands that don't fit the `coverage`
flow — `report`, `clean`, `show-env`, etc. When you want
`cargo llvm-cov nextest`, prefer [`cargo ktstr coverage`](#coverage);
this subcommand carries the same kernel-resolution and
`--no-perf-mode` plumbing but hands every remaining argument to
`cargo llvm-cov` unchanged.

```sh
cargo ktstr llvm-cov report --lcov --output-path lcov.info    # generate report from prior run
cargo ktstr llvm-cov clean --workspace                         # wipe accumulated coverage data
cargo ktstr llvm-cov show-env                                  # print env cargo-llvm-cov would set
cargo ktstr llvm-cov --kernel ../linux report                  # pin kernel + passthrough
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` (repeatable) | auto | Kernel identifier: path, version, cache key, range (`START..END`), or git source (`git+URL#tag=NAME`). Same multi-kernel semantics as `cargo ktstr test --kernel`. |
| `--include-eol` | off | Same as `cargo ktstr test --include-eol`: when a `--kernel START..END` range is present, also expand EOL `stable` series from the gregkh linux-stable mirror. No effect on a single kernel. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |
| `--no-skip-mode` | off | Convert resource-contention and host-topology-insufficient skips into hard test failures. Same semantics as on `test`; exports `KTSTR_NO_SKIP_MODE=1` for the test binary. |

Note: a bare `cargo ktstr llvm-cov` (no trailing subcommand)
dispatches to `cargo llvm-cov`, which runs `cargo test` — ktstr
tests rely on the nextest harness for gauntlet expansion
(topology-preset variants), verifier cell emission, and VM
dispatch. Under bare `cargo test`, only the `#[test]` stubs run
and gauntlet variants + verifier cells are silently skipped.
Always pass a subcommand after `llvm-cov` (most often `nextest`,
for which `cargo ktstr coverage` is the shorter route).

## kernel

Manage cached kernel images. Three subcommands: `list`, `build`,
`clean`. The standalone `ktstr kernel` subcommands are identical.

### kernel list

List cached kernel images, sorted newest first. With a `--kernel`
range, switches to PREVIEW MODE: prints the versions a `START..END` range
expands to without performing any download or build.

```sh
cargo ktstr kernel list
cargo ktstr kernel list --json                    # JSON output for CI scripting
cargo ktstr kernel list --kernel 6.12..6.14        # preview range expansion
cargo ktstr kernel list --kernel 6.12..6.14 --json # preview as JSON
```

Default mode walks the local cache. Human-readable output shows
key, version, source type, arch, and build timestamp. Entries built
with a different `ktstr.kconfig` are marked `(stale kconfig)`.
Entries whose major.minor version is no longer in kernel.org's
active releases list are marked `(EOL)`; prefix lookups for EOL
series fall back to probing cdn.kernel.org for the latest patch
release.

`--kernel` range mode performs no cache reads: it fetches kernel.org's
`releases.json` once, expands the inclusive range against the
`stable` and `longterm` releases (mainline / linux-next dropped),
and prints one version per line on stdout. Use this to answer
"what does `--kernel 6.12..6.16` actually cover?" before paying
the build cost — no kernel is downloaded or compiled. With
`--json`, emits a JSON object carrying the literal range, the
parsed start / end, and the expanded `versions` array.

| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format. Each entry includes a boolean `eol` field (computed at list time by fetching kernel.org's `releases.json`) alongside the cached metadata. With a `--kernel` range, emits a single object `{range, start, end, versions}` instead. |
| `--kernel START..END` | Switch to range-preview mode. Format: `MAJOR.MINOR[.PATCH][-rcN]..MAJOR.MINOR[.PATCH][-rcN]`. Performs the single `releases.json` fetch a real range resolve does, expands inclusively, and prints the version list — no downloads, no builds, no cache lookups. A non-range `--kernel` is rejected (preview expands ranges only). |
| `--include-eol` | With a `--kernel` range, also enumerate EOL `stable` series from the gregkh linux-stable mirror's tags so the preview lists series that have aged out of `releases.json`. Ignored in the default cache-listing mode. |

### kernel build

Download, build, and cache a kernel image. Takes a single `--kernel`
(the unified grammar): a version / prefix (tarball download), a
`START..END` range (builds each release), a source-tree path (local
build), or a `git+URL#…` source (fetch + build).

```sh
cargo ktstr kernel build                                   # latest stable from kernel.org
cargo ktstr kernel build --kernel 6.14.2                   # specific version
cargo ktstr kernel build --kernel 6.15-rc3                 # RC release
cargo ktstr kernel build --kernel 6.12                     # latest 6.12.x patch release
cargo ktstr kernel build --kernel 6.11..6.14               # every release in the range
cargo ktstr kernel build --kernel ../linux                 # local source tree
cargo ktstr kernel build --kernel git+URL#tag=v6.14        # git source (tag / branch / sha)
cargo ktstr kernel build --force --kernel 6.14.2           # rebuild even if cached
```

When `--kernel` is omitted, fetches the latest stable series that
has had at least 8 maintenance releases — keeping CI off brand-new
majors whose early builds are more likely to break — from
kernel.org's `releases.json`. A major.minor prefix (e.g. `6.12`)
resolves to the highest patch release in that series. For EOL series
no longer in `releases.json`, probes cdn.kernel.org to find the
latest available tarball. A cache key (an already-built entry) is
rejected — there is nothing to build. Skips building when a cached
entry already exists (use `--force` to override). Stale entries
(built with a different `ktstr.kconfig`) are rebuilt automatically.
For a `--kernel <path>` source tree, generates `compile_commands.json`
for LSP support. Dirty local trees (uncommitted changes to tracked
files) are built but not cached.

| Flag | Description |
|------|-------------|
| `--kernel ID` | Kernel to build: a version (`6.14.2`), a `MAJOR.MINOR` prefix (`6.14`, latest patch), a `START..END` range (builds every release), a source-tree path (`./linux`, `~/linux`, or an absolute path; a bare relative name is read as a cache key, so prefix a relative source dir with `./`), or a git source (`git+URL#tag=NAME` / `#branch=NAME` / `#sha=<40-hex>`). Omitted, builds the latest stable. A cache key (already-built entry) is rejected. Not repeatable — one kernel or range per invocation. |
| `--force` | Rebuild even if a cached image exists. |
| `--clean` | Run `make mrproper` before configuring. Only meaningful for a `--kernel <path>` source tree. |
| `--cpu-cap N` | Reserve exactly N host CPUs for the build (integer ≥ 1; must be ≤ the calling process's `sched_getaffinity` cpuset size). When absent, 30% of the allowed CPUs are reserved (minimum 1). The planner walks whole LLCs in consolidation- and NUMA-aware order, partial-taking the last LLC so `plan.cpus.len() == N` exactly. Under `--cpu-cap`, `make -jN` parallelism matches the reserved CPU count and the build runs inside a cgroup v2 sandbox that pins gcc/ld to the reserved CPUs + NUMA nodes. Mutually exclusive with `KTSTR_BYPASS_LLC_LOCKS=1`. Also settable via `KTSTR_CPU_CAP` env var (CLI flag wins when both are present). |
| `--extra-kconfig PATH` | Additional kconfig fragment merged on top of the baked-in `ktstr.kconfig` (user values win on conflict). Lands in a distinct cache slot keyed by the extra fragment's hash, so it never collides with a baked-only build. |
| `--skip-sha256` | Skip SHA-256 verification of a downloaded stable tarball (emits a bypass warning). No effect on a `--kernel <path>` or git source, which download no tarball. |
| `--include-eol` | When `--kernel` is a `START..END` range, also enumerate EOL `stable` series from the gregkh linux-stable mirror so the range builds series that have aged out of `releases.json`. No effect on a single version, path, or git source. |

### kernel clean

Remove cached kernel images.

```sh
cargo ktstr kernel clean                          # remove all (with confirmation prompt)
cargo ktstr kernel clean --keep 3                 # keep 3 most recent
cargo ktstr kernel clean --force                  # skip confirmation prompt
cargo ktstr kernel clean --corrupt-only --force   # remove only corrupt entries
```

| Flag | Description |
|------|-------------|
| `--keep N` | Keep the N most recent VALID cached kernels. Corrupt entries (metadata missing or unparseable, image file absent) are always candidates for removal regardless of this value — a corrupt entry never consumes a keep slot. Mutually exclusive with `--corrupt-only`. |
| `--force` | Skip confirmation prompt. Required in non-interactive contexts. |
| `--corrupt-only` | Remove only corrupt cache entries (metadata missing or unparseable, image file absent). Valid entries are left untouched regardless of `--force`. Useful for clearing broken entries after an interrupted build without risking the curated set of good kernels. Mutually exclusive with `--keep`. |

## verifier

Collect BPF verifier statistics for every scheduler declared via
`declare_scheduler!` in the workspace's test binaries. Spawns
`cargo nextest run -E 'test(/^verifier/) & !test(/^verifier::/)'` (the
`verifier/...` cells only, not the verifier module's `verifier::tests::*`
unit tests) and lets nextest fan out per (scheduler × kernel-list entry ×
accepted topology preset) cell — the sweep runs each scheduler ACROSS
topologies, because whether it attaches and dispatches is
topology-dependent (a scheduler can attach on one topology and wedge on
another). Each cell boots its own VM on the topology named in the cell,
with performance mode disabled (its `verified_insns` count is
perf-mode-independent, so cells take only a shared `LOCK_SH` LLC
reservation and no longer starve each other on the LLC lock; a
`performance_mode` peer's `LOCK_EX` can still defer a cell, resolved by
nextest retry), loads the scheduler's BPF programs, and reports
per-program verified instruction counts from host-side memory
introspection. A cell PASSes only when the scheduler verifies (BPF
loads), attaches (the guest gate confirms sched_ext `enabled`), AND
dispatches an injected SpinWait workload (the guest confirms a worker
made forward progress after attach). After the run, one `verified_insns`
table per scheduler (rows = kernel, cols = BPF program, cell = the count
across topologies) and a topology × scheduler PASS/FAIL grid are
printed.

```sh
cargo ktstr verifier                              # auto-discover kernel
cargo ktstr verifier --kernel ../linux            # pin to one kernel
cargo ktstr verifier --kernel 6.14 --kernel 7.0   # multi-kernel sweep
cargo ktstr verifier --scheduler scx-ktstr        # one scheduler across topologies
cargo ktstr verifier --raw                        # raw verifier log
```

The sweep discovers schedulers from the `KTSTR_SCHEDULERS`
distributed slice populated by `declare_scheduler!`. `--scheduler
<NAME>` restricts the sweep to a single declared scheduler (matched
by its `declare_scheduler!` `name`) across topologies; omitted, every
declared scheduler is swept. To exclude a scheduler entirely, omit it
from the test binary (or declare it with `SchedulerSpec::Eevdf` /
`SchedulerSpec::KernelBuiltin` — both are skipped at cell-emission
time because neither has a userspace binary to verify, so naming one
with `--scheduler` matches no cell).

`--kernel` is repeatable; cargo-ktstr always exports
`KTSTR_KERNEL_LIST` to the nextest invocation (synthesizing a
single entry from auto-discovery when no `--kernel` is passed).
Each scheduler's `kernels = [...]` declaration acts as a
per-scheduler filter on the operator-supplied set; an empty (or
omitted) `kernels` field accepts every entry. See [BPF Verifier:
Matrix dimensions + filters](verifier.md#matrix-dimensions--filters)
for the full filter contract.

`--raw` exports `KTSTR_VERIFIER_RAW=1`; the cell handler reads
it via `env::var_os` and switches `format_verifier_output` from
the cycle-collapsed default to the raw scheduler-log dump. See
[BPF Verifier: Cycle collapse algorithm](verifier.md#cycle-collapse-algorithm)
for the rendering details.

| Flag | Description |
|------|-------------|
| `--kernel ID` (repeatable) | Kernel identifier: path, version, cache key, range (`START..END`), or git source (`git+URL#tag=NAME`). Raw image files (`bzImage`/`Image`) are NOT accepted — the verifier needs the cached `vmlinux` and kconfig fragment alongside the image. Source directories auto-build; version strings auto-download on cache miss. When absent, resolves via cache then filesystem, falling back to auto-download. Raw images are accepted only on `cargo ktstr shell`. |
| `--raw` | Print raw verifier output without cycle collapse. |
| `--include-eol` | When a `--kernel START..END` range is present, also expand EOL `stable` series from the gregkh linux-stable mirror. No effect on a single kernel. |
| `--scheduler NAME` | Restrict the sweep to a single declared scheduler (its `declare_scheduler!` `name`) across topologies. Omitted, every declared scheduler is swept. A name matching no declared BPF scheduler fails loud with an empty result set. Sets `KTSTR_VERIFIER_SCHEDULER` for the inner `cargo nextest run`. |
| `--profile NAME` | Cargo BUILD profile for the scheduler-under-test (see `cargo ktstr test --profile`). Omitted, the scheduler builds `release`. Sets `KTSTR_SCHEDULER_PROFILE` for the inner `cargo nextest run`. |
| `--nextest-profile NAME` | Nextest TEST profile forwarded to the inner `cargo nextest run` as `--profile <NAME>` (see `cargo ktstr test --nextest-profile`). |

See [BPF Verifier](verifier.md) for the cell-based dispatch
design and output format, and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md)
for the `declare_scheduler!` macro that registers a scheduler
in `KTSTR_SCHEDULERS`.

## shell

Shares the VM boot flow with `ktstr shell` and accepts the same
flags. See [ktstr shell](ktstr.md#shell) for the full flag
reference. The one behavior difference from `ktstr shell` is that
`cargo ktstr shell` accepts raw image file paths for `--kernel`.

```sh
cargo ktstr shell
cargo ktstr shell --kernel 6.14.2
cargo ktstr shell --topology 1,2,4,1
cargo ktstr shell -i ./my-binary -i strace
```

## completions

Generate shell completions for cargo-ktstr. See
[ktstr completions](ktstr.md#completions) for the base subcommand.

```sh
cargo ktstr completions bash >> ~/.local/share/bash-completion/completions/cargo
cargo ktstr completions zsh > ~/.zfunc/_cargo-ktstr
cargo ktstr completions fish > ~/.config/fish/completions/cargo-ktstr.fish
```

| Arg | Description |
|------|-------------|
| `SHELL` | Shell to generate completions for (`bash`, `zsh`, `fish`, `elvish`, `powershell`). |
| `--binary NAME` | Binary name for completions. Default: `cargo`. |

## stats

Sidecar analysis and per-record diagnostics. Cross-run regression
comparison is [`perf-delta`](#perf-delta).
See [Runs](runs.md) for the directory layout.

```sh
cargo ktstr stats                                             # print analysis of newest run
cargo ktstr stats list                                        # list runs
cargo ktstr stats list-metrics                                # list registered regression metrics
cargo ktstr stats list-values                                 # distinct filter values present in the pool
cargo ktstr stats show-host --run RUN_ID                       # archived host context for a run
cargo ktstr perf-delta --dual-run --kernel 6.14               # regression-gate HEAD vs a baseline commit
cargo ktstr stats explain-sidecar --run RUN_ID                                   # diagnose Option-field absences
```

When invoked without a subcommand, prints gauntlet analysis from
either the most recent run directory under
`{CARGO_TARGET_DIR or "target"}/ktstr/` (newest by mtime) or the
explicit directory in `KTSTR_SIDECAR_DIR` when that variable is
set. With `KTSTR_SIDECAR_DIR` set, that directory is the sidecar
source directly -- there is no newest-subdirectory walk under it:

- **Gauntlet analysis** -- outlier detection, per-scenario/topology
  dimension summaries, stimulus cross-tab.
- **BPF verifier stats** -- per-program verified instruction counts,
  warnings for programs near the 1M complexity limit.
- **BPF callback profile** -- per-program invocation counts, total
  CPU time, and average nanoseconds per call.
- **KVM stats** -- cross-VM averages for exits, halt polling, host
  preemptions.

### list

Print a table of run directories under
`{CARGO_TARGET_DIR or "target"}/ktstr/` with four columns:

- `RUN`: the run-directory leaf name, formatted as
  `{kernel}-{project_commit}` per [Runs](runs.md). `list` does NOT
  consult `KTSTR_SIDECAR_DIR` — that override only affects where
  the test harness writes sidecars; `list` always enumerates the
  default runs-root.
- `TESTS`: number of sidecars in the directory (and one level of
  subdirectories — `collect_sidecars` walks per-job gauntlet
  layouts).
- `DATE`: the earliest sidecar timestamp present in the directory
  — under last-writer-wins this equals the most recent run's
  first sidecar timestamp (the prior run's sidecars were
  pre-cleared at the new run's first write, so only the new
  run's timestamps remain). See [Runs](runs.md) for the full
  semantics.
- `ARCH`: the `host.arch` value from the run's first sidecar
  (e.g. `x86_64`, `aarch64`). Renders as `-` when no sidecar in
  the directory carries a populated host context — pre-host-
  context archives and host-only test stubs that never populate
  the field land in this bucket.

Rows are sorted by directory mtime, **most recent first**, so the
latest run lands at the top — the operator's usual interest.
Entries whose mtime cannot be read fall back to filename order as
a deterministic tiebreaker and sort to the end of the listing.

### list-metrics

List the registered regression metrics and their default
thresholds. Enumerates the `ktstr::stats::METRICS` registry: metric
name, polarity (higher/lower better), default absolute-delta gate,
default relative-delta gate, and display unit. Use this to see which metric names
`ComparisonPolicy.per_metric_percent` keys can reference, and what
each default absolute and relative gate starts at before an
override. Default output is a human-readable table; `--json` emits
a JSON array with the same fields.

```sh
cargo ktstr stats list-metrics              # table
cargo ktstr stats list-metrics --json       # JSON array
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit JSON instead of a table. |

### list-values

List the distinct values present per filterable dimension in the
sidecar pool. Walks every run directory under `target/ktstr/`
(or `--dir`), pools the sidecars, and reports per-dimension sets
for all nine dimensions: `kernel`, `commit`, `kernel_commit`,
`source`, `resolve_source`, `cpu_budget`, `scheduler`, `topology`, and
`work_type`.
The `commit` and `source` keys map to the internal
`SidecarResult::project_commit` / `run_source` fields; the JSON
wire keys keep the shorter spellings. `cpu_budget` is the only
numeric dimension — its JSON value is an array of integers (the
effective host-CPU budget each run's vCPU threads ran on), and
never-booted skip rows (budget 0) are omitted.

Use this to discover what filter values the pool actually carries
before narrowing a [`perf-delta`](#perf-delta) run (e.g. which
kernels or project commits are present); `list-values` is the
upstream answer to "what have I got?".

```sh
cargo ktstr stats list-values                       # text per-dim blocks
cargo ktstr stats list-values --json                # JSON object
cargo ktstr stats list-values --dir /tmp/archived   # archived sidecar tree
```

The text shape renders one block per dimension with values one
per line. The JSON shape emits a single object keyed by
dimension name with arrays of values:

```json
{
  "kernel": [null, "6.14.2", "6.15.0"],
  "commit": [null, "abcdef1", "abcdef1-dirty"],
  "kernel_commit": [null, "kabcde7", "kabcde7-dirty"],
  "source": [null, "ci", "local"],
  "resolve_source": [null, "auto_built", "target_debug"],
  "cpu_budget": [4, 16],
  "scheduler": ["eevdf", "scx_rusty"],
  "topology": ["1n2l4c1t", "1n4l2c1t"],
  "work_type": ["SpinWait", "PageFaultChurn"]
}
```

The JSON keys `commit` and `source` are the wire contract;
internally the corresponding fields are
`SidecarResult::project_commit` and `SidecarResult::run_source`.

`kernel`, `commit`, `kernel_commit`, and `source` are optional
on the source sidecar (`SidecarResult::kernel_version` /
`project_commit` / `kernel_commit` / `run_source` are
`Option<String>`); the textual sentinel `unknown` and JSON
`null` both denote a sidecar that did not record a value for
that dimension.

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit JSON instead of per-dimension text blocks. |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `stats show-host --dir`. |

### show-host (archived) {#stats-show-host}

Print the archived `HostContext` for a specific run: CPU identity,
memory/hugepage config, transparent-hugepage policy, NUMA node
count, kernel uname triple, kernel cmdline, and every
`/proc/sys/kernel/sched_*` tunable captured at archive time. Useful
for inspecting the same fingerprint `perf-delta`'s host-delta section
uses, available on a single run.

The command scans sidecars in the run directory in iteration order
and prints the FIRST sidecar that carries a populated host field —
older pre-enrichment sidecars may have `host: None`, and the
forward scan tolerates those. If no sidecar has a populated host
field the command fails with an actionable error rather than
returning empty output.

| Flag | Default | Description |
|------|---------|-------------|
| `--run ID` | required | Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`; from `cargo ktstr stats list`). |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `stats show-host --dir`: useful for archived sidecar trees copied off a CI host. |

### explain-sidecar

Diagnose `Option`-field absences across a run's sidecars. Loads
every `*.ktstr.json` under `--run ID` (or its subdirectories one
level deep, mirroring the gauntlet-job sidecar layout) and reports,
per sidecar, which `Option<T>` fields landed as `None` plus the
documented causes for each absence and a classification:

- `expected` — `None` is the steady-state shape; no operator
  action recovers it (e.g. `payload` for a scheduler-only test,
  `scheduler_commit` which no `SchedulerSpec` variant exposes
  today).
- `actionable` — `None` indicates a recoverable gap; re-running
  in a different environment (in-repo cwd, non-tarball kernel,
  non-host-only test) would populate the field.

Different gauntlet variants on the same run legitimately differ
on which fields populate (host-only vs VM-backed,
scheduler-only vs payload-bearing), so the report is per-sidecar
rather than aggregate.

Sidecars are loaded verbatim — this command does NOT rewrite
`run_source` to `"archive"` even when `--dir` is set. Diverges
intentionally from `list-values`; matches `show-host`.
The override would erase the only signal that surfaces the
pre-rename `source`-key drop case.

The output header reports `walked N sidecar file(s), parsed M valid`: `N` counts every
`.ktstr.json` file the walker visited, `M` counts how many
parsed against the current schema. `walked > parsed` signals a
corrupt or pre-1.0-schema sidecar — re-run the test to
regenerate under the current schema.

Per-`None` blocks in the text output also include a `fix:`
line for fields whose `None` is recoverable by an operator
action (e.g. `kernel_commit` recovers when `KTSTR_KERNEL`
points at a local kernel git tree). Fields whose `None` is
the steady-state shape (or a multi-cause set with no single
remediation) emit no `fix:` line.

When the walk encounters parse failures, the text output
appends a trailing `corrupt sidecars (N):` block listing
each corrupt path on its own line followed by the serde
error message indented as `error: ...`, optionally
followed by an `enriched: ...` line with operator-facing
remediation prose when the parse failure matches a known
schema-drift case (currently the `host` missing-field
case). When the walk encounters IO failures (file matched
the predicate but `read_to_string` failed before parsing
could begin — permission denied, mid-rotate truncation,
broken symlink, EISDIR), the text output appends a parallel
`io errors (N):` block, structured the same way (path on
its own line, `error: ...` line below) but carrying
`std::io::Error::Display` rather than serde-error text. IO
errors do NOT carry `enriched:` lines — there is no
schema-drift catalog for filesystem incidents; the raw
`std::io::Error` Display is the remediation surface.
Each block is suppressed independently when its source
vec is empty.

All-corrupt and all-IO-failure runs (every predicate-
matching file failed to parse, or every one failed to
read) are NOT a hard error — text output renders the
header (`walked N sidecar file(s), parsed 0 valid`)
followed directly by the `corrupt sidecars (N):` and/or
`io errors (N):` block(s), skipping the per-sidecar
breakdown that has nothing to render. JSON output mirrors
this with `valid: 0`, `_walk.errors` and/or
`_walk.io_errors` populated, and per-field counts at zero.
This preserves structured per-file visibility for
dashboard consumers facing total-failure runs of either
class.

All-corrupt and all-IO-failure runs exit 0 (not a hard
error); CI scripts must inspect the JSON channel for
failure detection rather than relying on exit code. Two
common gating policies, each appropriate for different
operational stances:

- **Lenient** (treat partial failures as warnings):
  `_walk.valid > 0`. Accepts any run with at least one
  successfully-parsed sidecar; per-file parse or IO
  failures surface in the JSON arrays for triage but do
  not fail the gate.
- **Strict** (fail on any sidecar failure):
  `_walk.errors.len() == 0 && _walk.io_errors.len() == 0`.
  Requires every predicate-matching file to parse cleanly.
  Both checks are required because the two arrays cover
  disjoint failure classes (parse vs read) — a run with
  zero parse errors but one IO error still has a missing
  sidecar.

The two policies are NOT equivalent: a run with one valid
and one corrupt sidecar passes lenient (`valid == 1 > 0`)
but fails strict (`errors.len() == 1 > 0`). Pick the
policy that matches the operational tolerance for partial
data.

`--json` emits a single object with three top-level keys:
`_schema_version` (a string version stamp — currently
`"1"` — that consumers can gate on for incompatible shape
changes), `_walk` (an envelope carrying `walked` / `valid`
counts — same numbers the text header reports under "walked
N sidecar file(s), parsed M valid" — plus an `errors` array
of `{path, error, enriched_message}` entries covering every
parse failure (`enriched_message` is a human-facing
remediation string when a known schema-drift case matches,
JSON null otherwise) AND an `io_errors` array of
`{path, error}` entries covering every IO failure (file
matched the predicate but `read_to_string` failed; `error`
carries the raw `std::io::Error` Display). Both arrays
emit on every render — empty array when no failures of
that class occurred — so dashboard consumers see a uniform
shape without `contains_key` branching. With both arrays,
`walked == valid + errors.len() + io_errors.len()` by
construction in the steady state — every predicate-matching
file lands in exactly one bucket. (Filesystem races between
the count and load passes can perturb this; see the rustdoc
on `WalkStats` for the full caveat.) Then `fields`. Each
entry under `fields` carries `none_count` and `some_count`
(counts across all valid sidecars in the run, summing to
`_walk.valid`), `classification`, `causes`, and `fix`
(string when a remediation applies, JSON null otherwise).

Output produced before the schema-version stamp landed has
no `_schema_version` key; consumers should treat the key's
absence as pre-stamp output (compatible with shape `"1"` in
practice but unstamped).

The version bumps on incompatible shape changes (key
rename, key removal, semantic shift in an existing key) but
NOT on additive changes (new optional top-level keys, new
entries in `fields`, new optional sub-keys under existing
entries). The stamp is emitted as a JSON string (e.g. `"1"`,
`"2"`); parse it by stripping the quotes and converting the
inner digits to an integer, then gate on `parsed >= 1`
(integer comparison) — never use raw string comparison, since
lexicographic order would put `"10"` ahead of `"2"`. Pin
loosely (e.g. accept any version `>= 1`) so dashboard code
keeps working when the catalog grows; tighten only on the
specific bumps a consumer cannot tolerate.

```sh
cargo ktstr stats explain-sidecar --run RUN_ID                       # text per-sidecar diagnostic
cargo ktstr stats explain-sidecar --run RUN_ID --json                 # aggregate JSON for dashboards
cargo ktstr stats explain-sidecar --run RUN_ID --dir /path/archive    # diagnose archived sidecars
```

| Flag | Default | Description |
|------|---------|-------------|
| `--run ID` | required | Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`; from `cargo ktstr stats list`). |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `stats show-host --dir`. |
| `--json` | off | Emit aggregate JSON instead of per-sidecar text. |

### Prerequisites

Run tests first to generate sidecar JSON files:

```sh
cargo ktstr test                     # generates target/ktstr/{kernel}-{project_commit}/*.json
cargo ktstr stats                    # reads the newest run
```

Set `KTSTR_SIDECAR_DIR` to override the sidecar directory; otherwise
the default is `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
where `{project_commit}` is the project HEAD short hex (with `-dirty`
when the worktree differs).

## perf-delta

Compare `performance_mode` test metrics between HEAD and a baseline
commit, exiting non-zero when a metric regresses past its threshold.
The verdict uses a polarity-aware, abs+rel dual-gate engine: the
baseline commit's sidecars are one side, HEAD's are the other, paired
per scenario.
It runs in any repo that consumes ktstr (it discovers the repo from the
cwd). The primary use is a scheduler author asserting that a change does
not regress a degenerate case: mark the degenerate-case scenarios
`#[ktstr_test(performance_mode)]`, then `perf-delta` runs them at HEAD
and at the baseline and fails if a metric got worse — so the commit
that adds a fix-flag is gated against the commit before it. The
baseline is the merge-base with `main` by default, or any `--base` /
`--base-ref` commit. The same shape is a CI perf-gate on a pull
request. See the [A/B Compare Branches](../recipes/ab-compare.md) recipe
for a worked walkthrough.

```sh
cargo ktstr perf-delta --dual-run --kernel 6.14            # HEAD vs merge-base(HEAD, main)
cargo ktstr perf-delta --dual-run --kernel 6.14 --base-ref release  # vs merge-base(HEAD, release)
cargo ktstr perf-delta --base abc1234                      # vs an explicit commit, cached sidecars
cargo ktstr perf-delta --dual-run --kernel 6.14 -E perf_throughput  # narrow within performance_mode
cargo ktstr perf-delta --dual-run --kernel 6.14 --threshold 5       # 5% uniform regression gate
```

**Baseline resolution** (highest precedence first):

1. `--base <commit>` — compare HEAD directly against this commit-ish,
   no merge-base. The testability / cached-baseline override.
2. `--base-ref <ref>` — compare against `merge-base(HEAD, <ref>)`.
3. `$GITHUB_BASE_REF` (set only on `pull_request` events) — compare
   against `merge-base(HEAD, origin/<ref>)`, the fetched
   remote-tracking ref. An empty value is treated as unset.
4. otherwise — `merge-base(HEAD, <--default-branch>)` (default `main`).

The resolved baseline is shortened to the 7-hex form the sidecar
`project_commit` records, so it lines up with pooled runs directly. The
command bails if the baseline resolves to HEAD (nothing to compare).

`perf-delta` compares on the **commit axis**: HEAD vs a baseline commit,
partitioned by `project_commit` (baseline resolution above; source
models below). A cross-config question — e.g. scheduler A vs scheduler B
at the same commit — is answered in-test via the Verdict DSL's
`better_across_phases`, not by this command.

**Two source models for the baseline run's sidecars:**

- **default (cached-baseline)** — compares sidecars ALREADY pooled
  under the runs-root from a prior run or a downloaded CI artifact. The
  caller supplies both runs; perf-delta only resolves the pair and
  compares.
- **`--dual-run`** — PRODUCES both runs first: it checks the baseline
  commit out in a detached `git worktree` and runs its
  `performance_mode` tests there (sidecars redirected into the main
  pool), runs HEAD's in the working tree, then compares. Both ends run
  `KTSTR_PERF_ONLY=1` so only `performance_mode` tests execute, narrowed
  by `-E`. The worktree is removed on return. `gix` has no
  worktree-creation API, so this shells `git worktree add/remove` —
  `git` must be on `PATH`. A non-zero child test exit is logged but does
  not abort; the sidecars that were written are still compared.

If no `performance_mode` sidecars are produced at the baseline (none
are defined yet, or `-E` matched none), the command prints a notice and
exits `0` — an empty perf set is "nothing to compare", not a failure.

| Flag | Default | Description |
|------|---------|-------------|
| `--dual-run` | off | Produce both runs via a baseline worktree before comparing (else compare already-pooled sidecars). |
| `--kernel SPEC` | — | Kernel both runs boot. Required with `--dual-run`. Same `--kernel` form as `cargo ktstr test`. |
| `--profile NAME` | release | Cargo BUILD profile for the scheduler-under-test on BOTH sides' `cargo ktstr test` (see `cargo ktstr test --profile`). Only meaningful on the run-producing path (`--dual-run` / `--noise-adjust`). |
| `--nextest-profile NAME` | nextest default | Nextest TEST profile forwarded to BOTH sides' `cargo ktstr test`. Only meaningful on the run-producing path (`--dual-run` / `--noise-adjust`). |
| `--base COMMIT` | — | Explicit baseline commit-ish (skips merge-base). |
| `--base-ref REF` | — | Ref to merge-base against. |
| `--default-branch BRANCH` | `main` | Merge-base target when no `--base`/`--base-ref`/`$GITHUB_BASE_REF`. |
| `-E, --filter EXPR` | all `performance_mode` | Nextest filter narrowing within the `performance_mode` set. |
| `--relevant` | off | Additionally narrow to the `performance_mode` tests the `base..HEAD` diff (∪ working tree) touches, from the same baseline; intersected with `--filter`. Broad change → compares everything; docs-only → nothing. See [affected / --relevant](#relevant). |
| `--threshold PCT` | registry defaults | Uniform relative regression gate (percent). Mutually exclusive with `--policy`. |
| `--policy PATH` | registry defaults | Per-metric threshold JSON. Mutually exclusive with `--threshold`. Schema: `{ "default_percent": N, "per_metric_percent": { "worst_spread": 5.0, ... } }` (priority: per-metric override → `default_percent` → each metric's registry `default_rel`). |
| `--noise-adjust N` | off | Self-tuning noise mode (requires `--kernel`): run each side N times and gate a confident regression on the two sides being SEPARATED (a two-sided Welch t-test at alpha=0.05, or fully disjoint `[min, max]` bands) AND MATERIAL (the registry `default_abs`/`default_rel` dual-gate), instead of a fixed `--threshold`. Implies dual-run production looped N times. Conflicts with `--threshold`, `--policy`, and `--dual-run`. |
| `--noise-spread-threshold PCT` | `5.0` | Per-side relative-spread limit (percent) above which `--noise-adjust` adds an ADVISORY "noisy spread" annotation to a metric's row. Advisory only — never suppresses a confident regression. Requires `--noise-adjust`. |
| `--no-phases` / `--phases-only` / `--steps-only` / `--phase N` / `--phase-threshold PCT` | full per-phase render | Per-phase output projection for the `--noise-adjust` spread block (render-only; does not change the verdict). Each **requires `--noise-adjust`** — per-phase output exists only on the noise-adjusted path. |

Runnable as `just perf-delta <kernel> [base]`.

## affected {#affected}

Emit the scheduler packages a `base..HEAD` diff affects, as a flat JSON
array for a GitHub Actions **dynamic matrix**. Run it inside a scheduler
repo that consumes ktstr (it discovers the repo + workspace from the
cwd); pipe its output into `strategy.matrix.scheduler: ${{ fromJSON(...)
}}` so CI spawns one job per affected scheduler instead of building and
testing the whole fleet on every push.

```sh
cargo ktstr affected                          # vs merge-base(HEAD, main)
cargo ktstr affected --base-ref release       # vs merge-base(HEAD, release)
cargo ktstr affected --base abc1234           # vs an explicit commit
# -> e.g. ["scx_lavd","scx_rusty"]
```

**Attribution** is the UNION of two layers — a scheduler is affected if a
changed path is reachable by either:

1. **cargo dependency closure** (from `cargo metadata`): a changed path
   is attributed to its owning workspace crate; the scheduler is affected
   if that crate is in the scheduler's transitive dependency closure.
   Catches shared *Rust* library changes.
2. **`.d` input set**: only when a native (`.c`/`.h`) source or an
   unattributable path changed, each scheduler is built once and its
   cargo `<artifact>.d` dep-info is parsed into the exact set of files
   that compiled into it — the Rust sources, the generated BPF skeletons,
   and (via clang's `-M`) every `.bpf.c` / header it text-includes,
   including cross-scheduler includes and shared headers. A pure-Rust
   change skips this build (the crate closure alone is sound).

**Fail-safe** — a false negative (silently skipping an affected
scheduler) is the worst outcome, so every uncertainty widens to the full
testable set, never to a skip: an unresolvable base, a diff failure, a
workspace-root / build-graph / `Cargo.lock` change, or any changed
non-docs path attributed to neither a scheduler `.d` nor a workspace
crate. A per-scheduler build/read failure marks that scheduler affected.
Only a strictly docs-only change (or `base == HEAD`) emits `[]`.

Only **Discover** (cargo-package) schedulers appear in the array —
package-less schedulers (EEVDF, kernel-builtin) have no package to key a
matrix cell on and must run in a separate unconditional CI leg.

**Baseline resolution** is identical to [`perf-delta`](#perf-delta):
`--base <commit>` (explicit, skips merge-base) → `--base-ref <ref>`
(merge-base against it) → `$GITHUB_BASE_REF` (on a PR, as
`origin/<ref>`) → `merge-base(HEAD, <--default-branch>)` (default
`main`).

| Flag | Default | Description |
|------|---------|-------------|
| `--base COMMIT` | — | Explicit baseline commit-ish (skips merge-base). |
| `--base-ref REF` | — | Ref to merge-base against. Defaults to `$GITHUB_BASE_REF` on a PR, else `--default-branch`. |
| `--default-branch BRANCH` | `main` | Merge-base target when no `--base`/`--base-ref`/`$GITHUB_BASE_REF`. |

### --relevant (local test narrowing) {#relevant}

`affected` produces a CI matrix; `--relevant` is its local inner-loop
counterpart. `cargo ktstr test --relevant` (also `coverage --relevant`
and `perf-delta --relevant`) runs the SAME attribution engine against the
working tree — the committed `base..HEAD` diff **UNIONed with uncommitted
and untracked edits** — and narrows the run to only the tests whose
scheduler the change touched.

```sh
cargo ktstr test --relevant                       # only tests my edits affect
cargo ktstr test --relevant --base-ref release    # vs merge-base(HEAD, release)
cargo ktstr test --relevant -E 'test(smoke)'      # relevant AND matching -E
```

The relevant set is folded into a single nextest filterset that
**intersects** (`&`) any `-E` you pass, so it always narrows — never
widens — the selection. Outcomes mirror `affected`:

- a broad / build-graph / unattributable change does **not** narrow — the
  full selection runs (the fail-safe);
- a strictly docs-only change (or a clean tree at `base`) narrows to
  nothing — the run executes zero tests;
- otherwise only the affected schedulers' tests run. Package-less
  schedulers (EEVDF, kernel-builtin) are conservatively included on any
  non-docs change.

`--base` / `--base-ref` / `--default-branch` select the attribution
baseline exactly as in `affected` (ignored without `--relevant`). On
`perf-delta`, `--relevant` reuses the SAME baseline for both the
attribution and the A/B comparison, and intersects with `--filter`.

## show-host (live) {#show-host-live}

Print the **live** host context used by the sidecar collector:
CPU identity, memory/hugepage config, transparent-hugepage
policy, NUMA node count, kernel uname triple
(sysname / release / machine), kernel cmdline, and every
`/proc/sys/kernel/sched_*` tunable. Useful for diagnosing
cross-run regressions that trace back to host-context drift
(sysctl change, THP policy flip, hugepage reservation) or for
confirming what a future run produced here would record.

```sh
cargo ktstr show-host
```

This is a **live** snapshot (reads `/proc`, `/sys`, and
`uname()` at invocation time). For the **archived** host
context captured at sidecar-write time for a past run, use
[`cargo ktstr stats show-host --run RUN_ID`](#stats-show-host)
instead — same `HostContext::format_human` formatter so the
two outputs are byte-for-byte comparable when the host is
unchanged.

For historical drift between archived runs (host-side diff
across two run partitions), use [`perf-delta`](#perf-delta) — its host-delta section reports
which host-context fields changed between the baseline and HEAD
sides using the same `HostContext::diff` logic.

## show-thresholds

Print the resolved assertion thresholds for the named test —
the same merged `Assert` value `run_ktstr_test_inner` evaluates
against worker reports, produced by the runtime merge chain
`Assert::default_checks().merge(&entry.scheduler.assert).merge(&entry.assert)`.
Surfaces every threshold field (or `none` when inherited or
unset) so an operator can see what the test will actually
check against without reading source or guessing which layer
contributed each bound.

```sh
cargo ktstr show-thresholds preempt_regression_fault_under_load
```

| Arg | Description |
|------|-------------|
| `TEST` | Function-name-only test identifier as registered in `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`). Use `cargo nextest list` to enumerate test names — then strip the `<binary>::` prefix that nextest prepends to each line before passing the name here. The `#[ktstr_test]` registry keys on the bare function name, so a name like `ktstr::my_test` (as printed by nextest) must be trimmed to `my_test` before it resolves. |

Fails with an actionable message when no registered test
matches the given name; the diagnostic includes a `Did you
mean ...?` Levenshtein suggestion when a near match exists.

## export

Export a registered test as a self-extracting `.run` file that
reproduces the scenario on bare metal without a VM. Bundles the
running ktstr binary, the scheduler binary, and every include file
the test declares into a gzipped tarball embedded in a bash
preamble. The preamble validates root access, sched_ext support,
cgroup2 mount, the no-other-scheduler-attached invariant, and
topology compatibility before extracting and launching. The output
is chmod-`+x`'d so the operator can execute the `.run` directly.

```sh
cargo ktstr export preempt_regression_fault_under_load             # writes ./preempt_regression_fault_under_load.run
cargo ktstr export my_test -o /tmp/my_test.run                     # custom output path
cargo ktstr export my_test --package my_workspace_member           # restrict workspace search
cargo ktstr export my_test --release                               # build embedded binaries with release profile
```

| Arg / Flag | Description |
|------|-------------|
| `TEST` | Function-name-only test identifier as registered in `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`). Strip the `<binary>::` prefix that `cargo nextest list` prepends — the registry keys on the bare function name. |
| `-o, --output <PATH>` | Output path for the `.run` file. Defaults to `<TEST>.run` in the current directory. |
| `-p, --package <NAME>` | Restrict the workspace search to a specific package. When omitted, every workspace member's tests is built and scanned for a matching `#[ktstr_test]` registration. Pass-through to `cargo build --tests --package <NAME>`. |
| `--release` | Build the test binaries with the release profile. Stricter assertion thresholds and `panic = "abort"` — match the profile the operator will run the `.run` file under, otherwise the embedded binary's behavior may drift from the dev-profile test runs the operator reproduced from. |

The frozen bits — scheduler choice, scheduler args, topology — match
the test as registered. Overridable on the target host at `.run`
invocation time: `--duration`, `--watchdog-timeout`, `--quiet`
(suppress banner). NOT overridable: `--cpus`, `--topology`,
`--affinity` — re-export to change those.

**Out of scope for v1** (rejected at export time with actionable
errors): `host_only` tests (they orchestrate cargo / nested VMs
from inside the test body), tests with `bpf_map_write` (need the
framework's host-side runtime probe surface), and `KernelBuiltin`
schedulers (need the `enable` / `disable` shell commands the
preamble doesn't emit yet).

**Name collisions:** if multiple workspace test binaries register a
`#[ktstr_test]` with the same name, the router visits candidates in
alphabetical order by absolute binary path and the FIRST binary that
admits the test wins. Use `--package` to scope the search to a
specific package and disambiguate deterministically.

## locks

Enumerate every ktstr flock held on this host — read-only,
does NOT attempt any flock acquire. Troubleshooting companion
for `--cpu-cap` contention: when a build or test is stalled
behind a peer's reservation, `cargo ktstr locks` names the
peer (PID + cmdline) without disturbing any of its flocks.

Scans four lock-file roots:

- `/tmp/ktstr-llc-*.lock` — per-LLC reservations held by
  perf-mode test runs and `--cpu-cap`-bounded builds.
- `/tmp/ktstr-cpu-*.lock` — per-CPU reservations from the
  same flow.
- `{cache_root}/.locks/*.lock` — cache-entry locks held
  during `kernel build` writes, and `source-{path_hash}.lock`
  files held for the duration of `kernel build --kernel <path>` and
  `cargo ktstr test --kernel <path>` against the same source tree.
- `{runs_root}/.locks/{kernel}-{project_commit}.lock` —
  per-run-key sidecar-write locks held for the duration of
  the (pre-clear + write) cycle to serialize concurrent
  ktstr processes targeting the same run directory.

Each lock is cross-referenced against `/proc/locks` to name
the holder PID and cmdline.

```sh
cargo ktstr locks                       # one-shot snapshot
cargo ktstr locks --json                # JSON snapshot
cargo ktstr locks --watch 1s            # redraw every second until SIGINT
cargo ktstr locks --watch 1s --json     # ndjson stream, one object per interval
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit the snapshot as JSON. Pretty-printed in one-shot mode; compact (one object per line, ndjson-style) under `--watch`. Stable field names — schema documented on `ktstr::cli::list_locks`. |
| `--watch DURATION` | unset | Redraw the snapshot at the given interval until SIGINT. Value is parsed by `humantime`: `100ms`, `1s`, `5m`, `1h`. Human output clears and redraws in place; `--json` emits one line-terminated object per interval. |

The same subcommand is available as
[`ktstr locks`](ktstr.md#locks) with identical flag
semantics.

## Install

```sh
cargo install --locked ktstr   # the two user-facing binaries
```

The four test-fixture binaries (`ktstr-jemalloc-probe`,
`ktstr-jemalloc-alloc-worker`, `ktstr-schbench-validate`,
`ktstr-taobench-validate`) require the non-default `integration`
feature, so a default `cargo install` builds only `ktstr` and
`cargo-ktstr` and never places the fixtures on `$PATH`.

Or build from the workspace:

```sh
cargo build --bin cargo-ktstr
```
