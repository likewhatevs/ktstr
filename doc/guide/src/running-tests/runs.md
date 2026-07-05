# Runs and Regression Gates

Every test run writes machine-readable results — one JSON sidecar
per test, grouped into a *run directory* keyed by (kernel, project
commit). That makes "did my change regress anything?" a one-command
question: `cargo ktstr perf-delta` pairs two commits' sidecars and
fails the build when metrics regress past their gates.

## The workflow

1. **Run tests** — each invocation writes sidecars into
   `target/ktstr/{kernel}-{project_commit}/`:

   ```sh
   cargo ktstr test --kernel 6.14
   ```

2. **List runs**:

   <!-- captured: cargo ktstr stats list | ktstr 0.23.0 -->
   ```text
   $ cargo ktstr stats list
    RUN                   TESTS  DATE                  ARCH
    7.0.14-73730e0-dirty  1      2026-07-04T23:28:34Z  x86_64
    7.1.0-73730e0-dirty   0      -                     -
    7.1.1-73730e0-dirty   0      -                     -
    7.0.0-73730e0-dirty   1      2026-07-04T22:16:24Z  x86_64
    7.1.0-73730e0         6      2026-07-04T21:41:43Z  x86_64
   ```

   Rows sort by directory mtime, most recent first. `DATE` is the
   run's first sidecar timestamp; `ARCH` comes from the first
   sidecar with host context (`-` when none has one).

3. **Gate a change** against a baseline commit:

   ```sh
   cargo ktstr perf-delta --noise-adjust 5 --kernel 6.14         # HEAD vs merge-base(HEAD, main)
   cargo ktstr perf-delta --base abc1234                         # vs an explicit commit, cached sidecars
   cargo ktstr perf-delta --noise-adjust 5 --kernel 6.14 -E cgroup_steady   # narrow the perf set
   ```

   The canonical WIP-vs-baseline pattern: run `perf-delta --base
   abc1234` from a `-dirty` working tree against the clean commit
   you edited from.

4. **Print analysis** of the most recent run (gauntlet outliers,
   BPF verifier stats, callback profile, KVM stats):

   ```sh
   cargo ktstr stats
   ```

5. **Inspect a run's archived host context** (the fingerprint the
   host-delta comparison uses — CPU identity, THP policy, sched
   sysctls):

   ```sh
   cargo ktstr stats show-host --run 6.14-abc1234
   ```

## perf-delta {#perf-delta}

`perf-delta` compares `performance_mode` test metrics between HEAD
and a baseline commit, per scenario, using the metric registry's
polarity and thresholds (enumerate it with
`cargo ktstr stats list-metrics` — see
[Assertable Metrics](../reference/assertable-metrics.md)). Output is
one row per compared metric with the baseline and HEAD values,
colored red for regressions and green for improvements; the command
exits non-zero once enough metrics regress to trip the failure gate
— by default 5 or more (`--fail-threshold`), so a lone noisy
regression does not flip CI red, or any metric named in
`--must-fail`. If the baseline produced no `performance_mode`
sidecars at all, it prints a notice and exits 0 — an empty perf set
is "nothing to compare", not a failure.

**Baseline resolution** (highest precedence first):

1. `--base <commit>` — compare HEAD directly against this
   commit-ish, no merge-base.
2. `--base-ref <ref>` — compare against `merge-base(HEAD, <ref>)`.
3. `$GITHUB_BASE_REF` (set on `pull_request` events) — compare
   against `merge-base(HEAD, origin/<ref>)`.
4. Otherwise `merge-base(HEAD, main)` (override the branch with
   `--default-branch`).

The resolved baseline is shortened to the 7-hex form sidecars
record, and the command bails if it resolves to HEAD (nothing to
compare).


<div class="kt-figure"><svg width="700" height="188" viewBox="0 0 700 188" role="img" aria-label="perf-delta noise-adjust: baseline and HEAD are checked out into scratch dirs, each runs the perf set N times, and the Welch t-test plus min-max band gate decides significance">
  <defs><marker id="kt-arr2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <g font-size="10.5" fill="var(--fg)">
    <rect x="10" y="16" width="188" height="60" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="26" y="38" font-weight="700" opacity=".85">merge-base(HEAD, main)</text>
    <text x="26" y="58" opacity=".7" font-family="var(--mono-font)">gix checkout → /tmp</text>
    <rect x="10" y="106" width="188" height="60" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="26" y="128" font-weight="700" opacity=".85">HEAD (dirty snapshotted)</text>
    <text x="26" y="148" opacity=".7" font-family="var(--mono-font)">gix checkout → /tmp</text>
    <path d="M200 46 L 262 46" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr2)"/>
    <path d="M200 136 L 262 136" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr2)"/>
    <rect x="266" y="16" width="150" height="60" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.2"/>
    <text x="282" y="42" font-weight="700" fill="var(--kt-accent)">N runs</text>
    <text x="282" y="60" opacity=".7">KTSTR_PERF_ONLY=1</text>
    <rect x="266" y="106" width="150" height="60" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.2"/>
    <text x="282" y="132" font-weight="700" fill="var(--kt-accent)">N runs</text>
    <text x="282" y="150" opacity=".7">KTSTR_PERF_ONLY=1</text>
    <path d="M418 46 L 470 82" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr2)"/>
    <path d="M418 136 L 470 100" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#kt-arr2)"/>
    <rect x="474" y="61" width="216" height="60" rx="9" fill="none" stroke="var(--kt-accent)" stroke-width="1.4"/>
    <text x="490" y="84" font-weight="700" fill="var(--kt-accent)">separated AND material?</text>
    <text x="490" y="102" opacity=".75">Welch t (α=.05) ∨ disjoint bands;</text>
    <text x="490" y="116" opacity=".75">abs ∧ rel registry gates</text>
  </g>
</svg></div>

A real comparison — one schbench perf test, two commits, on a
heavily loaded host (which is exactly the noise the gate exists to
absorb):

<!-- captured: cargo ktstr perf-delta --base 73730e0 | ktstr 0.23.0 | sidecars from kernel 7.0.14 runs -->
<div class="kt-term"><div class="kt-term-bar"><span class="kt-term-title">cargo ktstr perf-delta --base 73730e0</span></div>

<pre>averaged across 1 runs (73730e0) and 1 runs (6be77c7)
 TEST                                                  METRIC                                  73730e0        6be77c7         DELTA             VERDICT
 performance_mode_schbench_steady/7.0.14/perf_scx/...  max_imbalance_ratio                     3.00           4.00            +1.00x            <span class="t-red">REGRESSION</span>
 performance_mode_schbench_steady/7.0.14/perf_scx/...  system_time_ns                          0.00           6992960.00      +6992960.00ns     <span class="t-red">REGRESSION</span>
 performance_mode_schbench_steady/7.0.14/perf_scx/...  user_time_ns                            8498190459.00  17508213019.00  +9010022560.00ns  <span class="t-red">REGRESSION</span>
 performance_mode_schbench_steady/7.0.14/perf_scx/...  schbench_worker_run_delay_ns_per_sched  6450.24        7615.48         +1165.24ns        <span class="t-red">REGRESSION</span>
summary: 4 regressions, 0 improvements, 0 informational, 45 unchanged
host delta ('73730e0' → '6be77c7'):
  heap_state:
    allocated_bytes: 110558488 → 111278120
<span class="t-dim">    ...</span>
<span class="t-grn">exit 0</span>   # 4 &lt; --fail-threshold (5): noisy, not gated</pre></div>

The same data trips the gate when you tighten it — `--fail-threshold
1` or `--must-fail user_time_ns` both exit 1. That is the CI
contract: thresholds decide, not eyeballs.

**Two ways to get the baseline's numbers:**

- **Cached (default)** — both sides' sidecars must already be in
  the pool (a prior run, or a CI artifact you downloaded).
  `perf-delta` only resolves the pair and compares, applying
  `--threshold PCT` (uniform gate) or `--policy PATH` (per-metric
  JSON) over the registry defaults.
- **`--noise-adjust N`** (requires `--kernel`, N ≥ 2) — produces
  both sides fresh: it checks the baseline and HEAD out into
  scratch checkouts, runs each side's `performance_mode` tests N
  times, and gates on the observed spread. A regression counts only
  when the two sides are *separated* (a two-sided Welch t-test at
  α = 0.05, or fully disjoint min–max bands) **and** *material*
  (past the registry's absolute + relative dual gate). This is the
  mode to trust on a noisy machine; budget N × per-side wall time
  for it.

`perf-delta` compares on the commit axis. A cross-config question —
scheduler A vs scheduler B at the same commit — is answered in-test
(see [Compare a Scheduler vs EEVDF](../recipes/scheduler-vs-eevdf.md));
the worked A/B walkthrough with real gates is
[A/B Compare Branches](../recipes/ab-compare.md), and the CI
perf-gate job lives in [CI](../ci.md).

## Run directories

```text
target/
└── ktstr/
    ├── 6.14-abc1234/        # kernel 6.14, project commit abc1234 (clean)
    │   ├── test_a.ktstr.json
    │   └── test_b.ktstr.json
    └── 7.0-def5678-dirty/   # kernel 7.0, commit def5678 + uncommitted changes
        ├── test_a.ktstr.json
        └── test_b.ktstr.json
```

The key is `{kernel}-{project_commit}`: the resolved kernel version,
plus the project tree's HEAD short hex, suffixed `-dirty` when the
worktree differs from HEAD.

The commit is discovered from the test process's working directory —
for a scheduler crate using ktstr as a dev-dependency, that is the
**scheduler crate's** commit, not ktstr's. Run from whichever clone
you want the run keyed on.

> [!WARNING]
> A run directory is a last-writer-wins snapshot, not an archive.
> Re-running the suite at the same kernel and project commit
> pre-clears the prior sidecars at the new run's first write. To
> preserve a run, move the directory *out of* the runs root
> (`mv target/ktstr/6.14-abc1234 ~/ktstr-archives/...`) — a sibling
> inside `target/ktstr/` would still be walked by `stats list` — or
> commit your changes so the next run lands under a new key.

Pre-clear is shallow: only `*.ktstr.json` files at the top level are
removed. Subdirectories created by external orchestrators (per-job
gauntlet layouts) are left alone but still read by `stats`, so clean
those yourself when reusing them.

## Inspecting sidecars

Each sidecar records the test name, topology, scheduler, work type,
verdict, per-cgroup stats, monitor summary, verifier and KVM stats,
kernel version, host context, and timestamps. Discovery tooling:

- `cargo ktstr stats list-values` — the distinct values per
  filterable dimension (kernel, commit, scheduler, topology, work
  type, ...) across the pool: the upstream answer to "what have I
  got?" before narrowing a `perf-delta`.
- `cargo ktstr stats list-metrics` — the regression metric registry
  (names, polarity, default gates, units).
- `cargo ktstr stats explain-sidecar --run ID` — why optional
  fields are absent, per sidecar, with a fix when one exists:

  <!-- captured: cargo ktstr stats explain-sidecar --run 7.0.14-73730e0-dirty | ktstr 0.23.0 -->
  ```text
  walked 1 sidecar file(s), parsed 1 valid

  test: throughput_gate
    topology: 1n1l2c1t
    scheduler: my_sched
    ...
    populated optional fields (8): resolve_source, project_commit, monitor, kvm_stats, kernel_version, host, cleanup_duration_ms, run_source
    none fields (3):
      scheduler_commit [expected]
        - no SchedulerSpec variant currently exposes a reliable commit source — reserved on the schema for future enrichment (e.g. --version probe or ELF-note read on the resolved scheduler binary)
      payload [expected]
        - test declared no binary payload (scheduler-only test or pure-scenario test that never invokes ctx.payload(...))
      kernel_commit [actionable]
        - KTSTR_KERNEL is unset or empty
        ...
        fix: set KTSTR_KERNEL to a local kernel source tree that is a git repository (e.g. a git clone of the kernel)
  ```

  `expected` means `None` is the steady state; `actionable` means a
  different environment would populate the field. `--json` emits an
  aggregate object for dashboards.

`stats list-values`, `show-host`, and `explain-sidecar` all take
`--dir DIR` to point at an archived sidecar tree copied off a CI
host.

## Environment notes

- **Local filesystem required.** The runs root must live on ext4 /
  xfs / btrfs / tmpfs — the advisory lock that serializes concurrent
  sidecar writes rejects NFS and other remote filesystems.
- **Non-git runs collide.** When the test process is not in a git
  repository, the commit slot is the literal `unknown`, and every
  such run shares `{kernel}-unknown` (with pre-clear between them).
  Set `KTSTR_SIDECAR_DIR` or put the tree under git to disambiguate.
  The sidecar's own `project_commit` field stays `null` for these
  runs — the dirname sentinel and the JSON field intentionally
  diverge.
- **`KTSTR_SIDECAR_DIR`** overrides the sidecar directory itself
  (used as-is, no key suffix) for writes and for bare
  `cargo ktstr stats` reads. Pre-clear is skipped under the
  override — you chose the directory, you own its contents. The
  `stats list` / `list-values` / `show-host` subcommands do not
  consult it; use `--dir`.

## Failure artifacts

Failed tests additionally write a failure-dump JSON next to their
sidecar — see
[Reading Failure Output](failures.md#artifacts-on-disk) for the
path scheme, the placeholder-vs-full distinction, and the
investigation workflow.
