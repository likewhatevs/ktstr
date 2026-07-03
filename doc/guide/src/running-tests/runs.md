# Runs

Each `cargo ktstr test --kernel ../linux` invocation writes per-test result
sidecars into a *run directory* under
`{CARGO_TARGET_DIR or "target"}/ktstr/`. The directory is the
record of the latest test run for that (kernel, project commit)
pair -- there is no separate "baselines" cache.

> **Warning:** Re-running the suite at the same kernel and
> project commit reuses the same directory and **deletes prior
> sidecars** at the first sidecar write of the new run. To
> preserve a previous run's outputs, move the directory OUT of
> the runs root (e.g. `mv target/ktstr/6.14-abc1234
> ~/ktstr-archives/6.14-abc1234.archived-{date}`) — moving it
> into a sibling under `target/ktstr/` would still let
> `cargo ktstr stats list` walk into the archived copy because
> the listing filter only excludes dotfiles. Alternatively
> commit your changes (or amend to drop a `-dirty` suffix) so
> the next run lands in a separate snapshot directory.

## Layout

```text
target/
└── ktstr/
    ├── 6.14-abc1234/        # one run: kernel 6.14, project commit abc1234 (clean)
    │   ├── test_a.ktstr.json
    │   └── test_b.ktstr.json
    └── 7.0-def5678-dirty/   # another run: kernel 7.0, project commit def5678 with uncommitted changes
        ├── test_a.ktstr.json
        └── test_b.ktstr.json
```

Each subdirectory is keyed `{kernel}-{project_commit}`, where
`{kernel}` is the kernel version resolved from whichever kernel
selector applies — for a path-form `--kernel ../linux` it comes
from the directory's `metadata.json:version` field, falling back
to `include/config/kernel.release`; for a version selector
(`--kernel 6.14`) or cache-key form, the resolved version is
recorded directly; `unknown` only appears when no selector yields
a version. `{project_commit}` is the project tree's HEAD short
hex (7 chars), suffixed `-dirty` when the worktree differs from
HEAD, or the literal `unknown` when `detect_project_commit`
returns `None` — including the "test process not inside a git
repo" case, but also unreadable cwd, an unborn HEAD with no
commits, or a corrupt repo state.

The commit is discovered by walking parents of the test process's
working directory until a `.git` marker is found — for a scheduler
crate using ktstr as a dev-dependency, this is the **scheduler
crate's** commit, not ktstr's. The function
that performs the probe (`detect_project_commit`) is called from
the test process's cwd, so running tests from inside the scheduler
crate's clone yields that crate's HEAD. Run from inside ktstr's
clone if you want to record ktstr's HEAD instead.

Two runs sharing the same kernel and project commit (the typical
"re-run the suite without committing changes" loop) reuse the
same directory: the second run pre-clears any prior
`*.ktstr.json` files in the directory at first sidecar write so
the directory is a last-writer-wins snapshot of (kernel, project
commit), not an append-only archive of every invocation. Re-run
the suite to regenerate the sidecars; commit your changes (or
amend to drop the `-dirty` suffix) to land a separate snapshot
directory.

Pre-clear is **shallow** — only `*.ktstr.json` files in the
immediate run directory are removed. Subdirectories created by
external orchestrators (per-job gauntlet layouts, cluster shards)
are left untouched, but `cargo ktstr stats` walks one level of
subdirectories when collecting sidecars, so stale sidecar files
left in subdirectories from a prior run will still appear in
stats output. Operators driving subdirectory layouts must clean
those subdirectories themselves; pre-clear's contract covers the
top-level only.

### Filesystem requirement

The runs root must reside on a local filesystem (ext4, xfs,
btrfs, tmpfs). NFS and other remote filesystems are rejected by
the advisory lock used for cross-process sidecar-write
serialization.

### Unknown-commit collisions

When the test process is not inside a git repository (so
`detect_project_commit` returns `None`), the on-disk dirname uses
the literal sentinel `unknown` in the commit slot — every such run
lands in `{kernel}-unknown`. Concurrent or successive non-git runs
collide on this single directory, with the latest run pre-clearing
the previous one's sidecars. To disambiguate non-git runs, set
`KTSTR_SIDECAR_DIR` to a per-run path or place the project tree
under git so each run carries its own commit hash.

ktstr emits a one-shot stderr warning on first sidecar write
under this configuration; setting `KTSTR_SIDECAR_DIR` both
disambiguates the run and suppresses the warning (the override
branch returns from `sidecar_dir` before the warning site is
reached).

The `unknown` sentinel applies to the **dirname only**. The
in-memory `SidecarResult.project_commit` field stays `None`
(serialized as JSON `null`) for these runs — the dirname uses a
filesystem-safe sentinel, while the JSON field preserves the
original probe outcome. As a consequence, a `project_commit`
filter for the literal `unknown` will **not** match a sidecar
whose `project_commit` is `None` — the dirname sentinel and the
JSON field diverge here.

`KTSTR_SIDECAR_DIR` overrides the *sidecar* directory itself
(used as-is, no key suffix), not the parent. The override only
affects where new sidecars are written and what bare
`cargo ktstr stats` reads. When the override is set, **pre-clear
is skipped** — the operator chose that directory and owns its
contents, so any pre-existing sidecars there are preserved.
`cargo ktstr stats list`, `cargo ktstr stats list-values`, and
`cargo ktstr stats show-host` all walk
`{CARGO_TARGET_DIR or "target"}/ktstr/` by default — pass
`--dir DIR` on `list-values` / `show-host` to
point them at an alternate run root (e.g. an archived sidecar
tree copied off a CI host). They do NOT consult
`KTSTR_SIDECAR_DIR`.

## Workflow

1. **Run tests** for kernel A:

   ```sh
   cargo ktstr test --kernel 6.14
   ```

2. **Run again** for kernel B:

   ```sh
   cargo ktstr test --kernel 7.0
   ```

3. **List** runs:

   ```sh
   cargo ktstr stats list
   ```

   Each row carries `RUN`, `TESTS`, `DATE`, and `ARCH` columns.
   `DATE` is the earliest sidecar timestamp present in the
   directory — under the last-writer-wins semantics, this equals
   the **most recent run's first sidecar timestamp** (the prior
   run's sidecars were pre-cleared at the new run's first write,
   so only the new run's timestamps remain). `ARCH` is the
   `host.arch` value (`x86_64`, `aarch64`, …) from the first
   sidecar in the directory whose `host` field is populated
   (sidecars without host context are skipped), or `-` when no
   sidecar in the run carries one. Rows are ordered by directory
   mtime, most recent first.

4. **Compare** HEAD against a baseline commit (regression gate):

   ```sh
   cargo ktstr perf-delta --dual-run --kernel 6.14            # HEAD vs merge-base(HEAD, main)
   cargo ktstr perf-delta --base abc1234                      # vs an explicit baseline commit
   cargo ktstr perf-delta --dual-run --kernel 6.14 -E cgroup_steady  # narrow the perf set
   ```

   `perf-delta` pairs the baseline commit's `performance_mode`
   sidecars against HEAD's per scenario and reports per-metric
   regressions via the unified `MetricDef` registry (polarity,
   absolute and relative thresholds). Output is colored: red for
   regressions, green for improvements; it exits non-zero when a
   metric regresses past its gate. The canonical WIP-vs-baseline
   pattern is `perf-delta --base abc1234` from a `-dirty` working
   tree against the clean commit it was edited from. See
   [cargo-ktstr](cargo-ktstr.md#perf-delta) for the full flag set,
   and `cargo ktstr stats list-values` to discover the values a
   pool carries.

5. **Print analysis** for the most recent run (no subcommand):

   ```sh
   cargo ktstr stats
   ```

   Honors `KTSTR_SIDECAR_DIR` when set (non-empty); otherwise picks
   the newest subdirectory under `target/ktstr/` by mtime. Prints
   gauntlet analysis, BPF verifier stats, callback profile, and KVM
   stats.

6. **Inspect the archived host context** for a specific run:

   ```sh
   cargo ktstr stats show-host --run 6.14-abc1234
   cargo ktstr stats show-host --run archive-2024-01-15 --dir /tmp/archived-runs
   ```

   Resolves `--run` against `target/ktstr/` (or `--dir` when set),
   scans the run's sidecars in order, and renders the first populated
   host-context field via `HostContext::format_human`: CPU model,
   memory config, transparent-hugepage policy, NUMA node count, uname
   triple, kernel cmdline, and every `/proc/sys/kernel/sched_*`
   tunable. Same fingerprint the `perf-delta` host-delta section
   uses, but available on a single run. Fails with an actionable
   error when no sidecar carries a host field (pre-enrichment run).

## Metric registry discovery

Before configuring per-metric `ComparisonPolicy` overrides, enumerate
the available metric names:

```sh
cargo ktstr stats list-metrics
cargo ktstr stats list-metrics --json
```

Prints the `ktstr::stats::METRICS` registry: metric name, polarity
(higher / lower better), `default_abs` and `default_rel` gate
thresholds, and display unit. Use the metric names from this list as
keys in `ComparisonPolicy.per_metric_percent`; unknown names are
rejected at `--policy` load time so typos surface loudly. `--json`
emits the same data as a serde array — the row accessor function is
omitted (`#[serde(skip)]`) so the wire surface carries only
wire-stable fields.

## Sidecar format

Each test writes a `SidecarResult` JSON file containing the test name,
topology, scheduler, work type, pass/fail, per-cgroup stats, monitor
summary, stimulus events, verifier stats, KVM stats, effective sysctls,
kernel command-line args, kernel version, timestamp, and run ID. Files
are named with a `.ktstr.` infix for discovery. `cargo ktstr stats`
reads all sidecar files from a run directory (recursing one level for
gauntlet per-job subdirectories).

See also: [`KTSTR_SIDECAR_DIR`](../reference/environment-variables.md).

## Failure-dump artifact

Every failed test writes a JSON failure-dump file alongside the
sidecar:

```text
{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{commit}/{test_name}-{variant_hash:016x}.failure-dump.json
```

The full dump (BPF state, per-vCPU registers, scheduler exit reason,
map contents) is produced by the freeze coordinator when the
scheduler attached AND its exit path triggered — the
post-mortem snapshot path. Auto-repro runs additionally write a
sibling `{test_name}-{variant_hash:016x}.repro.failure-dump.json` containing the
auto-repro VM's own post-mortem snapshot (same schema as the
primary, with the additional `early` snapshot when the entry
enables dual-snapshot capture).

When the failure occurs BEFORE the BPF probe attaches —
`send_sys_rdy` timeout, VM boot failures, scheduler binary load
failures, anything that returns from `vm.run()` before the
scheduler's BPF code runs — the framework writes a **placeholder**
dump at the same path. The placeholder carries the lifecycle stage
the run reached (per `classify_init_stage`) plus a "no BPF state
captured" reason; every BPF-state field is set to `unavailable: Some(reason)`
so stats tooling distinguishes "capture happened, no data" from
"capture path failed for reason X". The `is_placeholder: true`
field on the JSON makes the stub vs. real distinction explicit for
machine consumers.

The placeholder write uses an atomic `.tmp` → `rename(2)` pattern
so a concurrent reader either sees no file or sees a complete
stub — never a truncated one.

The actionable diagnostic for pre-attach failures lives in the test
stderr's `BUG SUMMARY:` line (extracted from the kernel's
`triggered exit kind` emit or the scheduler log's `scx_bpf_error`
substring) and the `--- sched_ext dump ---` section. The on-disk
placeholder is for tooling that walks the sidecar dir; humans
should read the stderr.

The path is pre-cleared at the start of every primary VM dispatch
so a passing rerun after a prior failed invocation does not leave
stale dump content alongside the new sidecar.
