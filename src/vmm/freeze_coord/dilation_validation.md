# verifier-hang-fixes performance-threshold validation

This file is committed empirical evidence that the `verifier-hang-fixes`
stack — the progress-watchdog rework (Tiers 1–2 progress ledger + flat
dead-man deadline), per-cell host-dilation collection, and the
unconditionally-RT (FIFO-2) sensing threads — does **not shift the
performance measurements** ktstr's threshold tests consume and does **not
blunt threshold/error detection**. The numbers below are a captured run.
There is no committed one-command harness; the deliverable is this captured
comparison, not a runner.

## Method

The same tests are run on two commits, from each commit's own build:

| side | commit | tree |
|---|---|---|
| baseline | `cc78f447` (main) | detached-HEAD git worktree |
| branch | `404adf72` (`verifier-hang-fixes` tip) | main working tree |

Both sides: dev (debug) test profile, kernel **7.1.3** from the local cache
(cache key `7.1.3-tarball-x86_64-kcbe1d947f`, pinned via `KTSTR_KERNEL` to
the same image for every run), one run at a time on an
otherwise-quiet 64-CPU host (load < 3 before each batch; no concurrent
builds, suites, or VMs). Each test ran **3 times per side** (N=3); each
metric is reported as **avg-of-3** with the run-to-run `[min–max]` spread.
Out-of-envelope cells were re-run (3 more reps) to separate tail noise from
systematic shifts, following the schbench-validation precedent
(`src/workload/schbench/validation.md`).

To reproduce: build each side (`cargo build --test <name> -p scx-ktstr
--bin scx-ktstr`), then dispatch each test directly through the ktstr
nextest protocol —

```
KTSTR_KERNEL=<7.1.3 cache dir> KTSTR_SIDECAR_DIR=<fresh dir> \
  NEXTEST=1 target/debug/deps/<test_bin>-<hash> --exact ktstr/<test_name>
```

— and read the measured values from the `*.ktstr.json` sidecar
(`stats.phases[].metrics.*`, `stats.ext_metrics.*`). The direct `--exact`
dispatch is required for the `assert_gate_matrix` entries (raw
`distributed_slice` registrations with no `#[test]` wrapper — invisible to a
plain `cargo nextest` filter). Host dilation is not in the sidecar; for the
branch runs it was surfaced with a temporary (not committed) `eprintln!` of
`VmResult::host_vcpu_schedstat.dilation()` in the two perf-mode tests'
`post_vm` callbacks.

### Chosen tests

Positives (2 perf-mode + 2 default-mode, all with sidecar-visible measured
values):

| test | mode | why |
|---|---|---|
| `performance_mode_schbench_steady` | perf | the low-variance steady schbench benchmark built for A/B comparison — every schbench metric surfaced |
| `performance_mode_perphase_metrics_across_detach` | perf | per-phase metrics across a real scheduler detach (scx phase vs EEVDF phase) |
| `schbench_pipe_runs_in_vm` | default | pipe (memory-transfer) workload; default mode is the path that newly gained RT sensing threads |
| `schbench_split_runs_in_vm` | default | shared-matrix cache-contention workload, also default mode |

Negatives (deliberately-violated runs that MUST be detected, from
`tests/assert_gate_matrix.rs` — `--degrade` scheduler + `expect_err`):

| test | mode | gate |
|---|---|---|
| `demo_gate_p99_wake_perf_on_negative` | perf | `max_p99_wake_latency_ns(1)` |
| `demo_gate_iter_rate_perf_off_negative` | default | `min_iteration_rate(1e9)` |

The `--degrade` scheduler (skip 1/2 dispatches) fails hard in the guest
(scx watchdog exit before the payload completes), so these negatives
exercise the **error-detection path** — the run must be seen as an error,
which `expect_err` then inverts to pass. Detection is binary (exit code);
the measured-value companion is each gate's healthy **positive** sibling,
which proves the metric pipeline still measures real numbers on both sides
(iteration-rate table below). The cgroup-level `p99_wake_latency_us`
reporting field reads 0/`wake_measured=false` for these SpinWait fixtures
on BOTH sides (the gate consumes raw ns samples internally), so no
percentile row is shown for the p99-wake pair.

### Why perf-mode measurements should be identical-mechanism

The perf-mode service threads (monitor/watchdog) were **already FIFO-2
before this branch** — only default/no-perf runs gained RT sensing threads
on the branch. So the two perf-mode tests compare byte-similar mechanisms,
and the default-mode tests (pipe, split, the `perf_off` gates) are the ones
that exercise the branch's new unconditional FIFO-2 sensing.

### Acceptance criterion

The branch's avg-of-3 falls inside (or within noise of) the baseline's
run-to-run `[min–max]` envelope for every metric, and every negative test
detects its violation **3/3 on both sides**. Single-digit-µs wakeup
percentiles get the precedent's `noise` treatment (integer µs buckets; 1 µs
of jitter reads as a large percentage but is not a divergence). Cells
outside the envelope are reported as-is with analysis — including the ones
where the branch is *better*.

## Results

Latencies are µs, lower=better; `loop_count`/`iter_rate` higher=better.
Verdict `yes` = branch avg inside the baseline `[min–max]` envelope.

### `performance_mode_schbench_steady` (perf mode, steady phase Step[1])

| metric | baseline (cc78f447) | branch (404adf72) | branch in baseline envelope |
|---|---|---|---|
| wakeup p99 (us) | 3.3 [3–4] | 3.0 [2–4] | yes |
| request p50 (us) | 45162.7 [45120–45248] | 45120.0 [44992–45248] | yes |
| request p99 (us) | 70272.0 [66944–73344] | 67520.0 [60992–72576] | yes |
| rps p50 | 43.3 [43–44] | 43.0 [43–43] | yes |
| schbench loop count | 657.7 [655–659] | 657.7 [657–659] | yes |

Branch-only observational: host dilation 1.0764 / 1.0777 / 1.0773
(avg 1.0771) — the new per-cell dilation sample, near-1.0 as expected for
1:1-pinned vCPUs on a quiet host.

### `performance_mode_perphase_metrics_across_detach` (perf mode)

| metric | baseline | branch | branch in baseline envelope |
|---|---|---|---|
| scx-phase wakeup p99 (us) | 4.0 [3–5] | 28.3 [2–79] | noise — see [1] |
| EEVDF-phase wakeup p99 (us) | 1097.0 [523–1402] | 664.0 [7–1402] | yes |

Branch-only observational: host dilation 1.2009 / 1.2050 / 1.2063
(avg 1.2041; re-runs 1.2031/1.2064/1.2056).

### `schbench_pipe_runs_in_vm` (default mode, pipe 64 KiB, 2 worker threads)

| metric | baseline | branch | branch in baseline envelope |
|---|---|---|---|
| wakeup p99 (us) | 4.0 [4–4] | 3.3 [3–4] | noise — see [2] |

### `schbench_split_runs_in_vm` (default mode, `--split 50`, 2 worker threads)

| metric | baseline | branch | branch in baseline envelope |
|---|---|---|---|
| wakeup p99 (us) | 9.0 [6–11] | 11.0 [11–11] | yes |
| request p99 (us) | 57664.0 [51648–64960] | 47210.7 [42560–51520] | NO (−8.6%, branch FASTER) — see [3] |

### Measured-value companion: gate positives (SpinWait fixtures)

`demo_gate_iter_rate_perf_off_positive` (default mode, gate
`min_iteration_rate(1.0)`):

| metric | baseline | branch | branch in baseline envelope |
|---|---|---|---|
| iteration rate (iters/s) | 39396.3 [37630–40352] | 41567.8 [37053–44736] | noise — see [4] |

`demo_gate_p99_wake_perf_on_positive` passed 3/3 on both sides (no
percentile row — see the chosen-tests note above).

### Negative detection

Each cell is detections/runs; a detection = the deliberately-degraded run
is recognized as an error (`expect_err` satisfied, exit 0).

| test | baseline | branch |
|---|---|---|
| `demo_gate_p99_wake_perf_on_negative` | 3/3 | 3/3 |
| `demo_gate_iter_rate_perf_off_negative` | 3/3 | 3/3 |

Full family sweep on the branch (all 20 `assert_gate_matrix` negatives —
every gated metric × perf on/off — 1 run each): **20/20 detected**. The
rest of the `expect_err` family (the `#[ktstr_test]`-registered negatives in
`ktstr_sched_tests.rs`, `silent_drop_e2e.rs`, `scenario_coverage.rs`,
`failure_dump_e2e.rs`, `cast_analysis_e2e.rs`, `vm_integration.rs`, etc.)
is nextest-visible and ran green in the branch's full host suite
(10508/10508 passed, 0 failed; the run this doc's branch tip shipped with).

## Reading the results

- **[1] scx-phase wakeup p99 (detach demo):** the branch's first-round avg
  is skewed by ONE 79 µs tail rep (other reps: 2, 4). A fresh 3-rep re-run
  landed 5/3/2 µs — inside the baseline envelope — so this is run-to-run
  tail noise, not a shift. This is exactly the variance the test's own doc
  warns about (mid-run scheduler swap amplifies p99 tails; the steady
  benchmark exists because of it), and the EEVDF phase of the same runs is
  inside the envelope.
- **[2] pipe wakeup p99:** single-digit integer-µs buckets; the baseline
  spread is degenerate ([4–4]) so any 1 µs of jitter falls "outside". The
  branch is 1 µs LOWER (better). Same `noise` treatment as the precedent's
  wakeup rows.
- **[3] split request p99 — the one systematic out-of-envelope cell,
  flagged for review.** Reproducible: across 6 runs/side (3 captured + 3
  re-run) baseline spans 51648–64960 µs (avg ~58.1 ms) while the branch
  spans 42560–54848 µs (avg ~47.6 ms) — the branch is ~10–18% FASTER on
  this default-mode contended workload, in the lower=better direction. This
  is an improvement, not a regression, and it cannot be FIFO-2 preemption
  cost (that would push latency UP). Best analysis: the branch's mode-aware
  halt-poll / per-cell boot-cost changes (`27f55bc5`) alter default-mode
  guest wakeup behavior, which the split workload's think-sleep-heavy
  request loop is sensitive to. Threshold semantics are unaffected (the
  test gates only on metric presence), but anyone using split request-p99
  as an absolute cross-commit baseline should re-baseline on this branch.
- **[4] gate-positive iteration rate:** branch avg is +3.0% above the
  baseline max edge, but the spreads overlap almost entirely
  (37053–44736 vs 37630–40352; the branch's min is BELOW the baseline's
  min), so this is overlapping run-to-run spread on a 2-vCPU SpinWait
  fixture, not a uniform shift — and it is in the higher=better direction.

## Summary

Across two perf-mode and two default-mode positives, every schbench
percentile, throughput, and loop-count metric on the branch lands inside
the baseline's run-to-run envelope or within integer-µs bucket noise, and
the two captured edge cells re-run inside it. The perf-mode measurements —
already-FIFO-2 mechanism on both sides — are indistinguishable
(steady-phase request p50 within 0.1%, loop count identical). The one
systematic departure is the default-mode split workload's request p99,
where the branch is reproducibly FASTER (~10–18%, [3]) — a directional
improvement flagged for review, not a blunted or shifted threshold. Error
detection is intact: the two chosen negatives detect 3/3 on both sides, the
full 20-test gate-matrix family detects 20/20 on the branch, and the
nextest-visible `expect_err` family is green in the branch's full suite.
The branch-only dilation samples read ~1.08 (steady) / ~1.20 (detach demo)
on this quiet host — plausible near-1.0 values confirming the new
collection works without perturbing the measurements it annotates.
