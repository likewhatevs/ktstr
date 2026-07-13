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
DELTA CONVENTION (this narrow section): out-of-envelope percentages
measure the branch avg's distance to the NEAREST baseline envelope
edge (the smallest defensible gap), not avg-to-avg; the wide section
below uses avg-to-avg and says so at its table.

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


## Wide-topology validation (post Tier-1 max-per-vCPU redesign)

Commit `643b99e3` reworked the progress watchdog's Tier-1 evidence after
CI's 192-CPU runners false-killed four healthy 256-vCPU cells six seconds
into Attach: the phase anchor was incoherent and the budget was denominated
in **summed** CPU, so a wide guest's diffuse idle-vCPU background burn
(ticks + IPIs across 256 vCPUs) crossed a per-vCPU-linear budget with no
wedge anywhere. The fix moves Tier-1 to a monitor-owned **max single-vCPU
in-phase** burn against a **flat, width-independent** budget; the summed CPU
still feeds Tier-2 trickle-stall and the Tier-3 deadman. This section is the
captured evidence that the reworked wide paths (a) still emit performance
measurements out of a wide cell, (b) still catch a real wide spinning wedge
fast, and (c) leave a legitimately-idle wide cell alive — plus the honest
limit of the summed-trickle detectors at width. (That §2 idle-wedge limit
was subsequently REMOVED by the Tier-2 CPU-term drop — see the addendum §4.)

### Why a wide shape needs near-1:1 — the CI-only regime

The false-fire only reproduces when a wide guest runs at **near 1:1**
vCPU:host-CPU — mostly-idle vCPUs each accruing *undiluted* background CPU.
CI's 256-vCPU shapes on 192-CPU runners were that regime; a 64-CPU host
cannot make 256 vCPUs run near-1:1 (they are 4x oversubscribed and heavily
diluted, so the per-vCPU background never reaches the budget scale that
tripped the old Tier-1). The reproduction here is therefore a **near-1:1
wide** guest: 56-64 vCPUs on the 64 host CPUs, undiluted — the same
per-vCPU-currency regime CI hit. Host dilation on the perf cell reads ~1.13
(below), confirming the vCPUs are near-1:1, not diluted.

### 1. Wide perf measurement, before/after

No committed test measures *performance* on a wide guest —
`snapshot_real_capture_wide_smp` captures BPF/vcpu-reg **state** at 256
vCPUs, not schbench metrics. So the measurement uses an **uncommitted scratch
copy** of `performance_mode_schbench_steady` (the low-variance steady A/B
shape) retopologized to a near-1:1 wide guest, run identically on both sides.

**Scratch shape** (both sides, `tests/performance_mode_e2e.rs`,
`wide_perf_schbench_steady_scratch`): `llcs = 8, cores = 7, threads = 1`
(**56 vCPUs**), `no_perf_mode`, same warmup(3s)+steady(15s) schbench backdrop
and `scheduler = scx-ktstr` as the committed steady test. **Mode choice:**
`no_perf_mode`, not `performance_mode` — perf-mode 1:1-pins each vCPU to a
host CPU, and 56-64 vCPUs on a 64-CPU host would leave no host CPU for the
monitor/schbench-worker service threads (64 vCPUs leaves literally none).
`no_perf_mode` floats the vCPU threads on the full host mask, which is also
the exact mode the false-firing wide-SMP cells use; 56 (not 64) keeps the
guest near-1:1 while reserving host headroom so the measurement is not
self-contended. The two sides differ only in the branch post_vm additionally
printing the branch-only per-cell dilation (baseline `cc78f447` has no
`host_vcpu_schedstat` field); the measured schbench values come from the
`*.ktstr.json` sidecar (`stats.phases[step_index=2].metrics.*`) identically
on both.

Steady phase (Step[1]) schbench metrics, **avg-of-6 [min-max]** per side (the
headline N=3 was re-run +3 to resolve the out-of-envelope cells, per the
narrow doc's precedent). Latencies µs, lower=better; rps/loop higher=better.
DELTA CONVENTION (this wide section): percentages are avg-to-avg —
unlike the narrow section's nearest-envelope-edge convention.

| metric | baseline (cc78f447) | branch | branch in baseline envelope |
|---|---|---|---|
| wakeup p99 (us) | 10214.7 [6776–13008] | 14266.7 [13008–15024] | **NO (+39.7%, branch slower) — see [W1]** |
| request p50 (us) | 92800.0 [90240–94848] | 89813.3 [89472–90752] | NO (−3.2%, branch FASTER) — see [W2] |
| request p99 (us) | 119125.3 [117888–119936] | 119722.7 [117376–123008] | yes (+0.5%) |
| rps p50 | 658.0 [655–661] | 656.3 [653–661] | yes |
| schbench loop count | 9847.2 [9816–9875] | 9821.3 [9762–9895] | yes |

Branch-only observational: host dilation avg **1.1323** [1.1198–1.1518] over
the 6 branch runs — near-1:1, undiluted, confirming the CI regime is
reproduced (not a diluted-oversubscribed shape). **No watchdog tier fired on
any of the 6 branch runs** — every run exit 0, no `cause=` line in any run's
stderr: the reworked width-independent Tier-1 does not false-kill a near-1:1
wide perf cell, and the wide cell still yields the full schbench metric set on
both sides.

- **[W1] wakeup p99 — the one systematic out-of-envelope cell, flagged for
  review.** Branch avg 14266.7 sits above the baseline envelope (the edges
  touch at 13008); across all 12 runs branch spans 13008–15024 and baseline
  6776–13008 — a clean ~40% separation. It is not a host-load artifact: the
  baseline re-runs at 1-min load 29–34 were still *faster* than branch runs at
  load 17–31, and the first-round branch runs at load 1–20 already showed the
  same high tail, so the shift is consistent across the load range both sides
  saw. Direction: branch is WORSE (higher wakeup tail). Mechanism: the branch's
  unconditionally-RT (FIFO-2) sensing threads — added to default/`no_perf`
  runs on this stack (see "Why perf-mode measurements should be
  identical-mechanism") — preempt the guest vCPU threads on the host; at width
  there are ~56 vCPU threads for the RT sensing to preempt, and wakeup p99 is
  the most preemption-sensitive schbench metric, so the cost that was
  invisible at the narrow 2-vCPU scale (pipe/split, notes [2]/[3]) surfaces as
  a measurable ~40% tail on the wide cell. Throughput is unaffected (loop
  count and rps within envelope) and request p50 is faster [W2], so this is a
  latency-tail redistribution under the new sensing, not a throughput
  regression or a blunted threshold — anyone using wide-cell wakeup-p99 as a
  cross-commit absolute baseline should re-baseline on this branch.
- **[W2] request p50:** branch 89472–90752 vs baseline 90240–94848 overlap
  only at the top edge — branch is ~3.2% FASTER on the median request
  (lower=better), the same directional default-path improvement the narrow
  split cell showed (note [3]).

### 2. Wide-wedge catch (branch)

Scratch copies of the Tier-1 spin and idle wedge fixtures at a wide topology
(`tests/progress_watchdog_e2e.rs`, `llcs = 8, cores = 8` = **64 vCPUs**,
`no_perf_mode`, reusing the committed `TEARDOWN_SPIN_SCHED` /
`TEARDOWN_IDLE_SCHED` injectors): the teardown fault injector spins/sleeps
**one** guest thread among 63 otherwise-idle vCPUs — the shape whose diffuse
background burn false-fired the old summed Tier-1.

**Spin wedge — Tier-1 fires, width-sound**
(`wide_teardown_spin_wedge_killed_by_tier1_scratch`):

```
watchdog: tier1-cpu-budget, kicking BSP
watchdog: deadline expired at 16.473224173s from VM start
  cause=tier1-cpu-budget, hard_timeout_fired=false, kill_set_by_AP=false
  phase=Teardown (Infra), monitor_live=true, evidence_channels_live=true
  max_vcpu_cpu_in_phase=12.031525413s vs budget=12s (currency=pthread), cpu_sum=18.557365599s, cpu_trickle_stalled=false
  effective_deadline=179.27319229s from VM start
WIDE_WEDGE_SPIN timed_out=true duration_s=18.8
```

Killed at **18.8 s** wall — « the 60 s fixture bound, and « the ~179 s
deadman the same dump reports for 64 vCPUs. The discriminator is exactly the
redesign: the lone spinner's `max_vcpu_cpu_in_phase = 12.03 s` crossed the
flat 12 s Teardown budget (8 s widened 3/2 for the pthread currency), while
`cpu_sum = 18.56 s` — the summed currency the OLD Tier-1 charged — is 54%
higher and folds in the 63 idle vCPUs' ~6.5 s of diffuse background. The max
evidence catches the real one-vCPU wedge without charging it the width. Same
bounded latency class as the 1-vCPU spin fixture (≈ 20-27 s expected there).

**Idle wedge — the honest conservative limit at width**
(`wide_teardown_idle_wedge_scratch`): the injector sleeps one thread among 63
idle vCPUs. The cell booted and reached the Teardown idle wedge (the wedge
family boots silently — the known-good spin cell above emitted nothing but
the vCPU-pin line before its fire), then **survived un-killed through both a
400 s and a 260 s run** — no `cause=` line ever emitted (the watchdog's kill
dump goes to unbuffered stderr; its absence is a definite no-fire), and the
179 s deadman deadline passed without a kill. Neither Tier-2 nor the Tier-3
deadman fired, and they cannot at this width: both gate on
`cpu_trickle_stalled`, and at 64 idle vCPUs the **summed** idle trickle (the
currency `643b99e3` deliberately kept for Tier-2/deadman) stays far above the
25 ms / 10 s pthread floor — the 63 idle vCPUs alone contributed ~6.5 s of
background CPU inside the spin cell's 12 s window — so the stall discriminator
never latches. Reported honestly: **the wide *idle* wedge is bounded only by
the outer harness `terminate-after` (or a dead monitor), not by any progress
tier** — it is NOT bounded « the deadman, because the deadman is itself
trickle-gated. This is the same summed-trickle property that (correctly) keeps
a *healthy* wide idle cell alive in §3: `643b99e3` narrowed only Tier-1's
width-unsoundness (the false-KILL of a healthy wide cell) and deliberately did
not tighten Tier-2/deadman's width-conservatism (the miss of a wide idle
wedge). The wide spin wedge — the CPU-burning shape the whole rework targets —
is caught fast; the wide idle wedge is left to the harness bound.
**Since resolved:** the Tier-2 CPU-term drop removed this limit — see §4.

### 3. Wide idle survival (branch, at 0c04d555)

The previously-false-killed CI shapes, re-run once each at this commit as the
recorded-in-doc confirmation they are green (256 vCPUs = 16 LLC × 16 core,
`no_perf_mode`, > 254-APIC-ID split-irqchip path):

| test | result |
|---|---|
| `wide_smp_guest_boots_all_cpus_online` | **PASS** (exit 0) — guest reports `total_cpus=256 online='0-255' n_online=256`: every vCPU online, no watchdog fire during the 4 s idle body |
| `snapshot_real_capture_wide_smp` | **PASS** (exit 0) — all 256 `vcpu_regs` slots captured through the freeze rendezvous; the "4.0x oversubscription" line is the expected informational warning (state capture, not timing) and non-fatal |

Both survive the long wide-idle phases the old Tier-1 killed at 6 s.

### 4. Addendum: the wide idle wedge is now caught — Tier-2's CPU term dropped

The §2 limit was a design defect, not physics: Tier-2's trickle conjunct was
belt-and-braces duplicating the protection its runnable conjunct already
carries. A starved cell WITH work always shows queued-or-running tasks in its
own rq memory (`nr_running` includes the on-CPU task, and guest memory is
readable regardless of host scheduling), so `!runnable_demand` alone exempts
every starved-alive shape — while no width-stable CPU floor can exist (the
guest housekeeping CPU's timekeeping/RCU duty measures 20-45 ms per 10 s
window at 64 vCPUs even on the busiest-single-vCPU currency, vs 1-10 ms at
1 vCPU). Tier-2 therefore dropped the CPU term entirely: `Infra &&
channels_live && !runnable_demand && wall_in_phase > backstop`. The trickle
discriminator survives only as the Tier-3 deadman's deferral gate,
re-denominated to the busiest single vCPU via per-vCPU window anchors (a
summed or per-tick-max currency degrades to ~the sum at width because idle
background burn rotates and serialises); a width misread there can only
DEFER, and the deferred runnable-piled shape stays bounded by the guest scx
watchdog.

The §2 scratch shapes are now COMMITTED fixtures
(`tests/progress_watchdog_e2e.rs`,
`teardown_{idle,spin}_wedge_wide_killed_by_tier{2,1}`, 64 vCPUs,
`no_perf_mode`), green alongside their narrow siblings (kernel 7.1.3,
sequential, quiet host):

| fixture | cause | watchdog kill | total | bound |
|---|---|---|---|---|
| narrow idle | tier2-idle-wedge | 18.7 s (wall_in_phase 15.05 s vs 15 s backstop) | 19.0 s | < 90 s |
| narrow spin | tier1-cpu-budget | 16.0 s (12.08 s vs 12 s budget) | 16.2 s | < 60 s |
| **wide idle (the §2 miss)** | **tier2-idle-wedge** | **18.8 s (wall_in_phase 15.04 s vs 15 s)** | **19.2 s** | < 90 s |
| wide spin | tier1-cpu-budget | 34.9 s (12.01 s vs 12 s; wedge-start variance — a prior run killed at 16.5 s) | 35.2 s | < 60 s |
| idle body survives | (no kill) | — | 12.0 s | no fire |

The wide-idle kill dump proves the mechanism — Tier-2 fires on the wall
backstop while the trickle verdict (deadman-only now) still reads "alive" on
exactly the housekeeping width residual that formerly made this wedge
immortal (`busiest_vcpu_window=38.6ms` > the 25 ms floor):

```
ktstr-watchdog: tier2-idle-wedge, kicking BSP
ktstr-watchdog: deadline expired at 18.791573169s from VM start
  cause=tier2-idle-wedge, hard_timeout_fired=false, kill_set_by_AP=false
  phase=Teardown (Infra), monitor_live=true, evidence_channels_live=true
  max_vcpu_cpu_in_phase=54.209745ms vs budget=12s (currency=pthread), cpu_sum=4.832328108s, cpu_trickle_stalled=false
  busiest_vcpu_window=38.603751ms vs trickle_floor=25ms
  wall_in_phase=15.036764228s vs backstop=15s
  progress_epoch=5 (milestones), wall_since_milestone=15.036764228s, runnable_demand=false, deadman_deferrals=0
```

(Watchdog output is branded `ktstr-watchdog` as of this change; the §2 dumps
above quote the older `watchdog:` prefix verbatim as captured.)

### Methodology / reproduction

- **Scratch copies, not committed.** Three functions were added to two test
  files on the branch and one on the baseline, exercised, then reverted;
  `git status` after this run shows only this doc modified.
  - `tests/performance_mode_e2e.rs` — `wide_perf_schbench_steady_scratch`
    (56 vCPUs, `no_perf_mode`, steady schbench A/B; branch prints dilation,
    baseline omits it since `cc78f447` lacks `host_vcpu_schedstat`).
  - `tests/progress_watchdog_e2e.rs` —
    `wide_teardown_spin_wedge_killed_by_tier1_scratch` (asserts `< 60 s`) and
    `wide_teardown_idle_wedge_scratch` (records the outcome; does not assert
    the 1-vCPU Tier-2 bound), both 64 vCPUs `no_perf_mode`.
- **Sides & build.** baseline `cc78f447` in its own detached worktree with its
  own build; branch `0c04d555` (`verifier-hang-fixes` tip) in the main tree.
- **Dispatch.** Same direct-nextest protocol as the narrow validation:
  `KTSTR_KERNEL=<7.1.3 cache dir> KTSTR_SIDECAR_DIR=<fresh> NEXTEST=1
  <test_bin> --exact ktstr/<name>`, one at a time on the quiet 64-CPU host,
  kernel **7.1.3** (`7.1.3-tarball-x86_64-kcbe1d947f`). The `ktstr/`-exact
  name is required so the scratch cell's own 56/64-vCPU topology runs (the
  `gauntlet/` variants would override it). Perf N=6/side; wedge and survival
  1 run each.
- **Acceptance criterion** is the narrow doc's honest-envelope test: branch
  avg-of-6 inside the baseline `[min-max]` for every metric, out-of-envelope
  cells reported as-is with mechanism analysis (including where the branch is
  faster), plus the wide spin wedge killed « its deadman. The wide-idle
  miss (§2) is recorded as the summed-trickle detectors' documented width
  limit, not a pass/fail of this validation.

## Final-product re-validation (B = `de820520`, the branch's final code state)

Everything above was captured at intermediate commits. This section
re-runs the COMPLETE comparison matrix against baseline A = `cc78f447`
at the final code state — after the Tier-2 CPU-term drop, the
contention witness, the CPU-second re-denomination, the CPU-time stuck
gate, the tri-state latency seam, and the build-reservation work all
landed — plus a three-point instrumentation-overhead ladder isolating
the two measurement-path additions. 132 runs total, **zero
contamination discards, zero failed runs**.

PROFILE NOTE: this campaign is RELEASE-profile both sides (the guest
schbench payload compiles into the test binary, so absolutes here are
~12x the dev-profile numbers above — e.g. steady loop count ~7.9k vs
658). No absolute below is comparable to the tables above; every ratio
is release-vs-release. DELTA CONVENTION (this section): avg-to-avg.

### Methodology deltas from the sections above

- Sides built once, in detached worktrees, before any measurement; one
  run at a time, rep-major interleaved across sides; 1-min loadavg <= 8
  enforced at every launch (the driver waits out spikes).
- The branch sides gate themselves with their own witness: every
  perf-mode branch run's sidecar `host_dilation` was checked against a
  D > 1.05 discard bar. Measured D = 1.0094-1.0270 across all 24 such
  runs — the machinery dogfooding the methodology it exists for. The
  wide no-perf cell read D = 1.126-1.154, confirming the near-1:1
  regime (§"Why a wide shape needs near-1:1").

### Instrumentation-overhead ladder (P0 `02603278` → P1 `213c355f` → P2 `94ef4735`)

P1/P0 isolates the witness's per-tick host schedstat reads; P2/P1 the
worker's per-checkpoint `CLOCK_THREAD_CPUTIME_ID` read; N=6/side.

On the real workload (perf steady schbench), both additions are
unmeasurable: every ratio 0.996-1.005, every metric inside P0's
envelope (the lone 0.875 on wakeup-p99 is integer-µs bucketing on a
[1-2] µs spread). On the amortization WORST CASE (the pure-SpinWait
iteration cell, one checkpoint per 1024 spins and nothing else), the
means dip monotonically ~0.6%/rung — the exact sign and scale the
added hot-path clock read predicts — cumulating to P2/P0 = 0.988,
with P2's avg still inside P0's own [min-max]. Rates were computed
from raw carrier components on every side (P0/P1 predate the
`cpu_sec` denomination marker), so the comparison is
denomination-proof.

### Final matrix — narrow cells (N=6/side, verdict = B avg inside A's [min-max])

`performance_mode_schbench_steady` (perf): wakeup p50/p99 identical
(1 µs both sides); request p50 0.996, request p99 0.993, rps 1.004,
loop count 1.005 — all in-envelope.
`performance_mode_perphase_metrics_across_detach` (perf): both phases
in-envelope (scx 0.875 bucket noise; EEVDF 0.068 inside A's known
[4-515] µs high-variance envelope).
`schbench_pipe_runs_in_vm` / `schbench_split_runs_in_vm` (default):
wakeups identical; split request p99 0.919 with B tighter — the same
faster-split direction as note [3]. `demo_gate_iter_rate_perf_off_positive`:
CPU-rate 0.994, in-envelope.

### Final matrix — wide 56-vCPU cell (the §1 shape re-run, N=6/side)

| metric | A avg [min-max] | B avg [min-max] | B/A | verdict |
|---|---|---|---|---|
| wakeup p50 (us) | 6.5 [5-10] | 15.83 [15-16] | 2.436 | OUT, B slower (+9.3 µs) — [F1] |
| wakeup p99 (us) | 3893 [3828-3996] | 3827 [3820-3836] | 0.983 | B marginally FASTER, far tighter — [F2]: the [W1] tail is GONE |
| request p50 (us) | 7229 [7160-7288] | 6613 [6536-6664] | 0.915 | OUT, B FASTER — [F3], the [W2] direction, larger |
| request p99 (us) | 11790 [10640-13810] | 11330 [11150-11570] | 0.961 | in-envelope |
| rps p50 | 7989 [7832-8168] | 8059 [8040-8072] | 1.009 | in-envelope |
| schbench loop count | 120300 [116000-122900] | 120800 [120400-121200] | 1.005 | in-envelope |

- **[F2] supersedes [W1].** The interim +40% wide wakeup-p99 tail does
  NOT reproduce at the final code state: B/A = 0.983 with B's spread
  at/below the bottom edge of A's, against a release-profile baseline
  envelope of ±2% (the dev-profile capture's was ±30%), so the null is
  sharp. The standing wide-tail concern recorded at [W1] is closed by
  this measurement.
- **[F1] is what remains of the sensing cost: a +9.3 µs wakeup-MEDIAN
  offset, width-only.** 6.5 → 15.8 µs with disjoint spreads; narrow
  wakeups are identical to the µs on both sides. The FIFO-2 sensing
  threads' per-wakeup preemption charge shows up as a small CONSTANT
  median shift instead of an erratic tail — a strictly better-behaved
  cost. Throughput unaffected. Anyone using wide-cell wakeup-p50 as an
  absolute cross-commit baseline should re-baseline.
- **[F3]**: wide request p50 reproducibly −8.5% (B faster, disjoint
  spreads) — the [W2]/note-[3] default-path improvement, larger at
  width. An improvement to re-baseline against, not a regression.

### Detection and the demotion path

12/12 negatives detected on both sides (p99-wake and iteration-rate
gates, 3 reps each per side). Zero `contention-indeterminate`
occurrences across all 48 B-side runs: the p99-wake negative fails via
the degraded scheduler's in-guest scx-watchdog exit (not the latency
gate), and the iteration-rate gate is CPU-denominated — outside the
demotion path by design. Detection is byte-for-byte unchanged at the
final code state.

### Verdict

Measurement-neutral everywhere the narrow shapes can see; at width, the
interim tail regression is gone, replaced by a small constant
wakeup-median offset ([F1]) in the mode whose contract already declines
timing fidelity; two reproducible improvements ([F3], split-tail); no
blunted thresholds; instrumentation overhead bounded at −1.2% on a
worst-case synthetic and unmeasurable on real workloads.
