# A/B Compare Branches

One command answers "did my branch make the scheduler slower?":
[`cargo ktstr perf-delta`](../running-tests/cargo-ktstr.md) runs
the same `performance_mode` scenarios against your branch and its
baseline commit, diffs every metric, and exits non-zero when enough
metrics regress to trip the failure gate. (For host-context diffs
or per-thread profiling instead, see the
[compare picker](../recipes.md#which-recipe-do-i-want).)

## Automated: `perf-delta --noise-adjust`

```sh
cd ~/src/my-sched                  # the scheduler crate under test

cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux                    # HEAD vs merge-base(HEAD, main)
cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux --base-ref release # vs merge-base(HEAD, release)
cargo ktstr perf-delta --noise-adjust 5 --kernel ../linux -E cgroup_steady   # narrow the perf set
```

`perf-delta` resolves the baseline as `merge-base(HEAD, <ref>)` (or
a `$GITHUB_BASE_REF` PR target), then `--noise-adjust N` checks
both commits out into their own plain checkouts, runs each
side's `performance_mode` tests N times, and compares from the
observed spread — no manual worktree bookkeeping.

A single run per side cannot tell a real regression from
run-to-run noise, so `--noise-adjust` gates a confident regression
on two conditions: the sides must be *separated* (a Welch
two-sample t-test, or fully disjoint `[min, max]` bands) and the
delta must be *material* (each metric's registry significance
gate). N must be at least 2 — variance needs two samples — and 5
or more is recommended for a well-powered test. Budget wall time
accordingly: the command produces 2×N full runs of your
`performance_mode` set, so at N=5 a one-minute suite costs about
ten minutes.

The command exits non-zero once enough metrics regress to trip the
failure gate — 5 or more by default, so a lone noisy regression
does not flip CI red. `--fail-threshold` tunes the count;
`--must-fail M1,M2` fails on the named metrics regardless of
count. This drops straight into a CI perf-gate on a pull request —
see [CI](../ci.md) for the workflow.

## Manual: compare already-pooled runs

Every `cargo ktstr test` run writes one stats sidecar per test into
`target/ktstr/{kernel}-{project_commit}/`; the accumulated sidecars
are the *pool* that `perf-delta` compares. When you want control
over the worktrees or test selection — or you already have both
runs' sidecars from CI artifacts — run the two branches yourself
and point `perf-delta --base` at the baseline commit. It compares
the cached pool without producing new runs (so it needs no
`--kernel`).

```sh
cd ~/src/my-sched

# Baseline: check out and run the baseline branch's suite.
git worktree add ~/src/my-sched-main upstream/main
cd ~/src/my-sched-main
KTSTR_PERF_ONLY=1 cargo ktstr test --kernel ../linux

# Experimental: run HEAD's suite.
cd ~/src/my-sched
KTSTR_PERF_ONLY=1 cargo ktstr test --kernel ../linux

# Compare the pooled sidecars: HEAD vs the baseline commit.
cargo ktstr perf-delta --base <baseline-short-hex>
```

The `{project_commit}` half of the sidecar directory is the
project tree's HEAD short hex captured at first sidecar write
(suffixed `-dirty` when the worktree differs from HEAD), so two
branches with distinct HEADs land in distinct directories and
coexist under one runs root. `perf-delta --base <hex>` partitions
that pool by `project_commit`: the baseline commit's sidecars are
side A, HEAD's are side B.

> [!WARNING]
> The two runs must be at distinct commits. If both checkouts share
> the same HEAD they land in the same directory and the second
> run's pre-clear overwrites the first — the comparison degenerates
> to an identical pool. Confirm distinct commits with
> `git -C ~/src/my-sched rev-parse HEAD` before the second run.

The project commit is discovered by walking up from the test
process's current working directory to the enclosing `.git`, so the
`cd` steps are load-bearing: without them the probe records the
wrong commit. Use `cargo ktstr stats list-values` to see the
`project_commit` values a pool actually carries before choosing
`--base`.

## Comparing configurations (not commits)

`perf-delta` compares on the commit axis (HEAD vs a baseline). A
cross-config question — scheduler A vs scheduler B, or two tunings,
at the same commit — is answered in-test: run both configurations
as phases of one scenario and assert the relationship directly
(e.g. `VmResult::better_across_phases`), so the verdict travels
with the test rather than a separate compare invocation.
[Compare a Scheduler vs EEVDF](scheduler-vs-eevdf.md) is the worked
example of that pattern.

## Cleanup

```sh
git worktree remove ~/src/my-sched-main
```
