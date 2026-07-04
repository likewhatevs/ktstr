# A/B Compare Branches

> **Disambiguation**: this recipe covers **scheduler-behavior
> regression checks between a branch and its baseline** (per-metric
> scheduler-driven measurements via `cargo ktstr perf-delta`). For
> **host-context diffs** (kernel build, CPU model, sched_\* tunables,
> NUMA layout), see [Capture and Compare Host State](host-state.md) —
> `perf-delta` surfaces a host-delta section too, read from each
> sidecar's archived `host` field.

Compare scheduler behavior between HEAD and a baseline branch by
running the same `#[ktstr_test(performance_mode)]` scenarios against
each, then diffing per-metric results with dual-gate (absolute and
relative) significance and exiting non-zero when enough metrics regress
to trip the failure gate (by default 5 or more).
[`cargo ktstr perf-delta`](../running-tests/cargo-ktstr.md#perf-delta)
is the single command for this: the baseline commit's sidecars are
side A, HEAD's are side B, paired per scenario.

## Automated: `perf-delta --noise-adjust`

`perf-delta` resolves the baseline as `merge-base(HEAD, <ref>)` (or a
`$GITHUB_BASE_REF` PR target), then `--noise-adjust N` checks BOTH
commits out into their own plain `gix` checkouts, runs each side's
`performance_mode` tests N times, and compares from the observed spread
— no manual worktree bookkeeping.

```sh
cd ~/opensource/scx                # the scheduler crate under test

cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux                    # HEAD vs merge-base(HEAD, main)
cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux --base-ref release # vs merge-base(HEAD, release)
cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux -E cgroup_steady   # narrow the perf set
```

`--noise-adjust N` runs each side N times (N >= 2; >= 5 recommended) and
gates a confident regression on the two sides being SEPARATED (a Welch
t-test or disjoint `[min, max]` bands) AND MATERIAL (the registry
dual-gate) — the statistically robust verdict, since a single run per
side cannot tell a real regression from run-to-run noise. The command
exits non-zero once enough metrics regress to trip the failure gate — by
default 5 or more, so a lone noisy regression doesn't fail the run
(`--fail-threshold` tunes the count; `--must-fail M1,M2` fails on
specific metrics regardless of count). It drops straight into a CI
perf-gate on a pull request. See
[perf-delta](../running-tests/cargo-ktstr.md#perf-delta) for the full
flag set.

## Manual: compare already-pooled runs

When you want control over the worktrees or the test selection — or
you already have both runs' sidecars pooled from CI artifacts — run
the two branches yourself and point `perf-delta --base` at the
baseline commit; it compares the cached pool without producing new
runs (so it needs no `--kernel`).

```sh
cd ~/opensource/scx

# Baseline: check out and run the baseline branch's suite.
git worktree add ~/opensource/scx-main upstream/main
cd ~/opensource/scx-main
cargo ktstr test --kernel ../linux -- -E 'test(/performance_mode/)'

# Experimental: run HEAD's suite.
cd ~/opensource/scx
cargo ktstr test --kernel ../linux -- -E 'test(/performance_mode/)'

# Compare the pooled sidecars: HEAD vs the baseline commit.
cargo ktstr perf-delta --base <baseline-short-hex>
```

Each `cargo ktstr test` run writes its sidecars into
`target/ktstr/{kernel}-{project_commit}/`; the `{project_commit}` half
is the project tree's HEAD short hex captured at first sidecar write
(suffixed `-dirty` when the worktree differs from HEAD), so two
branches with distinct HEADs land in distinct directories and coexist
under one runs root. `perf-delta --base <hex>` partitions that pool by
`project_commit`: the baseline commit's sidecars are side A, HEAD's are
side B.

> **Warning:** the two runs MUST be at distinct commits. If both
> checkouts share the same HEAD they land in the same directory and the
> second run's pre-clear overwrites the first — the comparison
> degenerates to an identical pool. Confirm distinct commits with
> `git -C ~/opensource/scx rev-parse HEAD` before the second run.

The project commit is discovered by walking up from the test process's
**current working directory** to the enclosing `.git`, so the `cd`
steps are load-bearing: without them the probe records the wrong
commit. Use
[`cargo ktstr stats list-values`](../running-tests/cargo-ktstr.md#list-values)
to see the `project_commit` values a pool actually carries before
choosing `--base`.

## Comparing configurations (not commits)

`perf-delta` compares on the commit axis (HEAD vs a baseline). A
cross-config question — scheduler A vs scheduler B, or two tunings, at
the SAME commit — is answered in-test via the Verdict DSL's
`better_across_phases`: the test declares both configurations and
asserts the relationship directly, so the verdict travels with the
test rather than a separate compare invocation.

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
```
