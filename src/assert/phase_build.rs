use super::*;

/// Phase buckets attributed against the guest stimulus timeline, then
/// enriched with stimulus-event-derived per-phase `iteration_rate`.
///
/// Unlike the plain [`build_phase_buckets`] (which groups by the
/// bridge-stamped step_index), this re-groups each periodic capture by
/// the guest step whose stimulus window contains the capture's
/// workload-relative boundary offset (`Sample::boundary_offset_ms`).
/// That offset is derived from the boundary schedule rather than the
/// fire time, so it is immune to the deferred-fire burst that makes
/// every capture stamp the same late CURRENT_STEP (the
/// `phases.len() == 1` collapse). Captures with no offset (on-demand /
/// fixture) fall back to their stamped step_index. Because the bucket
/// windows are then workload-relative, the run-relative monitor samples
/// are shifted by the stimulus/monitor clock skew before windowing.
///
/// Additionally synthesizes a capture-free `PhaseBucket`
/// (`sample_count == 0`) for any stimulus `StepStart`-step that
/// captured no periodic samples — the uniform whole-workload boundary
/// placement (`compute_periodic_boundaries_ns`) is step-agnostic, so a
/// short interior step can land zero captures and otherwise leave no
/// bucket, silently dropping its capture-independent `iteration_rate`.
/// The synthesized bucket carries the step's full stimulus window so its
/// `iteration_rate` (from `StepStart`/`StepEnd` deltas) and
/// `avg_imbalance_ratio` (from in-window monitor samples) are still
/// recovered. The returned vec therefore holds one bucket per
/// (captured phase ∪ `StepStart`-step), sorted by `step_index` — NOT
/// one-per-captured-phase, so `len()` is no longer "number of captured
/// phases".
///
/// The `iteration_rate` enrichment lets
/// `crate::timeline::Timeline::from_phase_buckets` render the per-phase
/// throughput annotation without going through the legacy
/// `crate::timeline::Timeline::build` path.
///
/// For each `StepStart[k]` -> `StepEnd[k]` pair with
/// `total_iterations: Some(_)`, the per-phase rate is
/// `(later - earlier) / duration_s` where `duration_s` is the
/// elapsed-ms delta BETWEEN THE TWO STIMULUS EVENTS (guest clock),
/// not the PhaseBucket sample window. The rate is attributed to the
/// step the EARLIER event starts (`prev.step_index`); the attribution
/// loop skips any `is_step_end` (or `is_terminal`) `prev`, so only a
/// StepStart is ever the earlier member. Phases that don't overlap a
/// stimulus pair keep their PhaseBucket.metrics map unchanged (no
/// iteration_rate key).
///
/// SEMANTICS: `total_iterations` is the sum of the worker handles
/// alive at each event (see
/// [`crate::timeline::StimulusEvent::total_iterations`]). Each step's
/// rate is its STEP-LOCAL `StepStart[k]` -> `StepEnd[k]` delta — the
/// step's own workers measured over its own hold — so a bucket is
/// sourced ONLY by its own pair (the `is_step_end` guard drops the
/// inter-step `StepEnd[k]` -> `StepStart[k+1]` pair entirely). This
/// measures BOTH fresh-per-step workers (which read ~0 at each
/// StepStart, so the old cross-step delta produced no rate) and
/// persistent (Backdrop) workers (excluding the inter-step teardown
/// wall-time a cross-step window would span). On a clean run the
/// `(StepEnd[N], terminal)` pair is guard-skipped and the trailing
/// `is_terminal` event is not consumed; it supplies a step's right
/// boundary ONLY for legacy/synthetic data carrying a `ScenarioEnd`
/// frame but no `StepEnd` frames. A sched-died step has neither frame
/// (its early return skips both emissions), so the dead step's
/// `StepStart` is never a `prev` with a successor and it reports no
/// rate.
///
/// iteration_rate is registered as `MetricKind::Rate` with the Counter
/// components `total_phase_iterations` / `total_phase_duration_sec` and
/// `HigherBetter` polarity (more throughput is better). The per-step
/// producer below emits those two components (the iteration delta and the
/// window seconds — the ms→s `/1000` applied at the component, since
/// `derive_rate_metrics` does a bare num/den) rather than a ready ratio,
/// and the `derive_rate_metrics` post-pass re-derives `iteration_rate` =
/// Σiterations / Σseconds at every in-map aggregation level. Its per-run
/// run-scalar fold (one run's per-phase values → that run's `ext_metrics`)
/// runs through `populate_run_ext_metrics_from_phases`, which SUMS the
/// Counter components across phases (a synthesized zero-capture phase's
/// components are summed in, not zero-weighted out — the run-aggregate
/// completion of the per-step rate handling) and re-derives the rate. The
/// cross-sidecar-run rollup `group_and_average_by` likewise re-pools via
/// its `derive_rate_metrics` post-pass. `iteration_rate` has no cross-cgroup
/// axis to re-pool: it is derived from run-level phase buckets and
/// host-injected into the run `ext_metrics` by
/// `populate_run_ext_metrics_from_phases` (the eval layer) AFTER the
/// cross-cgroup `merge`, so `AssertResult::merge`'s worst-case
/// (min/max-by-polarity) `ext_metrics` fold never sees its components. The rate whose components ARE per-cgroup is the separate
/// pooled `iterations_per_cpu_sec`, re-pooled across a run's cgroups by
/// `populate_run_pooled_iterations_per_cpu_sec` (reading `stats.cgroups`
/// post-merge).
///
/// Live caller: `evaluate_vm_result` at `src/test_support/eval/mod.rs`
/// — has both the SampleSeries and the stimulus_events vec in scope.
pub fn build_phase_buckets_with_stimulus(
    samples: &crate::scenario::sample::SampleSeries,
    stimulus_events: &[crate::timeline::StimulusEvent],
) -> Vec<PhaseBucket> {
    let monitor_samples: &[crate::monitor::MonitorSample] =
        samples.monitor().map(|m| m.samples()).unwrap_or(&[]);
    // vCPU-preemption exemption window for the per-phase stall predicate,
    // threaded into fold_monitor_into_bucket -> compute_metrics so the
    // per-phase stuck_count uses the SAME predicate as the run-level
    // MonitorSummary::stuck_count. 0 (no monitor) derives from CONFIG_HZ
    // inside compute_metrics, matching from_samples_with_threshold.
    let preemption_threshold_ns: u64 = samples
        .monitor()
        .map(|m| m.preemption_threshold_ns())
        .unwrap_or(0);
    // Re-group periodic captures by the guest step whose stimulus
    // window contains each capture's workload-relative boundary offset,
    // NOT the step_index stamped at (deferred) fire time — see
    // [`crate::scenario::sample::SampleSeries::by_stimulus_phase`] for
    // why the scheduled offset is the timing-independent truth (it
    // survives the deferred-fire burst that collapses the stamped
    // step_index to one phase).
    let by_phase = samples.by_stimulus_phase(stimulus_events);
    // Bucket windows are workload-relative (boundary-offset) here, so the
    // monitor samples (run-relative) are shifted by the stimulus/monitor
    // clock skew before windowing.
    let monitor_to_window_offset_ms = monitor_clock_offset(stimulus_events, monitor_samples);
    let mut buckets = buckets_from_grouped(
        by_phase,
        monitor_samples,
        monitor_to_window_offset_ms,
        preemption_threshold_ns,
    );
    let step_starts = crate::scenario::sample::step_starts_from_stimulus(stimulus_events);
    synthesize_missing_step_buckets(
        &mut buckets,
        &step_starts,
        stimulus_events,
        monitor_samples,
        monitor_to_window_offset_ms,
        preemption_threshold_ns,
    );
    // Keep buckets in step_index order: buckets_from_grouped emits them
    // sorted (BTreeMap key order), but the synthesize loop appends out of
    // order. Downstream lookups resolve by step_index, but a sorted vec
    // matches the captured-bucket invariant and keeps rendered output
    // stable.
    buckets.sort_by_key(|b| b.step_index);
    fill_phase_iteration_rates(&mut buckets, stimulus_events);
    buckets
}

/// Synthesize a `PhaseBucket` for any scenario step that has a stimulus
/// StepStart but produced no capture bucket. Periodic boundaries are
/// placed uniformly over the whole workload (step-agnostic — see
/// `compute_periodic_boundaries_ns`), so a short interior step can
/// capture zero samples and leave no bucket. The iteration_rate
/// attribution in [`fill_phase_iteration_rates`] only mutates EXISTING
/// buckets, so without this seam that step's capture-independent rate
/// (derived purely from the StepStart/StepEnd total_iterations deltas,
/// needing no capture) would be silently dropped. The synthesized bucket
/// carries the step's true stimulus window so `fold_monitor_into_bucket`
/// still recovers its monitor-derived imbalance; `sample_count == 0` marks
/// it capture-free for downstream consumers. `Timeline::from_phase_buckets`
/// COMPARES this bucket's stimulus-derived throughput across the gap
/// (the iteration_rate is real, not a sampling artifact) but GATES the
/// monitor-derived metrics (imbalance/dsq/fallback/keep_last) behind
/// both sides having samples — see `detect_boundary_changes`. BASELINE
/// (step 0) is synthesized only if a StepStart carries
/// `step_index == 0` — it is not special-cased.
fn synthesize_missing_step_buckets(
    buckets: &mut Vec<PhaseBucket>,
    step_starts: &[(u64, u16)],
    stimulus_events: &[crate::timeline::StimulusEvent],
    monitor_samples: &[crate::monitor::MonitorSample],
    monitor_to_window_offset_ms: i64,
    preemption_threshold_ns: u64,
) {
    for &(start_ms, k) in step_starts {
        if buckets.iter().any(|b| b.step_index == k) {
            continue;
        }
        // Right edge: the step's own StepEnd, CLAMPED to the next
        // step-start so a non-monotonic (corrupt-wire) StepEnd[k] >=
        // StepStart[k+1] can never extend this synthesized window past the
        // next step's start (which would fold the same MonitorSample into
        // two adjacent synthesized buckets). Falls back to the next
        // step-start alone, else open-ended (the last step on data with no
        // StepEnd — e.g. sched-died — which the rate loop leaves rateless).
        // Earliest StepEnd elapsed for k (min, symmetric with start_ms's
        // earliest step-start), so a duplicate/corrupt StepEnd later in the
        // vec can't pick a wrong right edge.
        let step_end = stimulus_events
            .iter()
            .filter(|e| e.is_step_end && e.step_index == Some(k))
            .map(|e| e.elapsed_ms)
            .min();
        // `ms > start_ms` (strict): two DISTINCT step_index sharing the
        // same elapsed is corrupt-wire only (build_stimulus emits StepStarts
        // at distinct monotone guest-clock times), so the equal-elapsed
        // window-overlap edge is out of scope — its worst case is a
        // redundant double-fold of one monitor sample, never a drop or panic.
        let next_start = step_starts
            .iter()
            .filter(|&&(ms, _)| ms > start_ms)
            .map(|&(ms, _)| ms)
            .min();
        // Last-resort right edge for a last step with no StepEnd and no
        // successor: the scenario-end terminal event, bounding the window
        // at run-end instead of folding every trailing teardown monitor
        // sample into the bucket. NOTE the production sched-died path emits
        // NEITHER a StepEnd NOR a ScenarioEnd (its early return skips both),
        // so it carries no terminal here and falls to u64::MAX — matching
        // Timeline::build, which extends the last phase to end-of-monitor.
        // This terminal clamp therefore only binds for legacy/synthetic
        // data carrying a ScenarioEnd frame without StepEnd frames.
        let terminal = stimulus_events
            .iter()
            .find(|e| e.is_terminal)
            .map(|e| e.elapsed_ms);
        let end_ms = match (step_end, next_start) {
            (Some(se), Some(ns)) => se.min(ns),
            (Some(se), None) => se,
            (None, Some(ns)) => ns,
            (None, None) => terminal.unwrap_or(u64::MAX),
        };
        // BASELINE (step 0) is synthesized only if a StepStart carries
        // step_index == 0; build_stimulus 1-indexes scenario Steps (Step 0
        // -> step_index 1), so it never emits step_index 0 and BASELINE is
        // intentionally never synthesized (its pre-first-Step settle window
        // carries no meaningful per-step rate). The branch is kept for the
        // convention, not for a production path.
        let label = if k == 0 {
            "BASELINE".to_string()
        } else {
            format!("Step[{}]", k.saturating_sub(1))
        };
        let mut bucket = PhaseBucket {
            step_index: k,
            label,
            start_ms,
            end_ms,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: std::collections::BTreeMap::new(),
        };
        fold_monitor_into_bucket(
            &mut bucket,
            monitor_samples,
            monitor_to_window_offset_ms,
            preemption_threshold_ns,
        );
        buckets.push(bucket);
    }
}

/// Fill each phase bucket's `iteration_rate` Rate components from the
/// stimulus event `total_iterations` deltas. Walk events pairwise; for
/// each pair compute the rate. Sort events by elapsed_ms first so an
/// out-of-order arrival from the bulk-port drain doesn't silently lose the
/// delta to saturating_sub (the legacy Timeline::build path at
/// src/timeline.rs sorts the same way; without the sort, an inversion
/// produces duration_ms == 0 → skipped, a silent drop).
///
/// Must run AFTER the synthesize seam and the step_index sort so it sees
/// the full, ordered bucket set exactly as the returned vec will hold it.
fn fill_phase_iteration_rates(
    buckets: &mut [PhaseBucket],
    stimulus_events: &[crate::timeline::StimulusEvent],
) {
    let mut sorted_events: Vec<&crate::timeline::StimulusEvent> = stimulus_events.iter().collect();
    // Total-order on an elapsed_ms tie: StepEnd before StepStart
    // (`!is_step_end` is false=0 for StepEnd) so a zero-length inter-step
    // gap at the guest's coarse-ms clock attributes the step-local
    // StepStart[k]->StepEnd[k] rate to bucket k, never the cross-step
    // StepStart[k]->StepStart[k+1] delta (the is_step_end guard below only
    // stops StepEnd from being `prev`; it does not order a StepEnd before
    // the next StepStart on a tie). Mirrors Timeline::build's sort.
    sorted_events.sort_by_key(|e| (e.elapsed_ms, !e.is_step_end));
    for w in sorted_events.windows(2) {
        let prev = w[0];
        let curr = w[1];
        // The terminal scenario-end event and per-step StepEnd events are
        // right boundaries only — never the EARLIER member of a pair (the
        // step a rate is attributed to). Both carry an end-of-hold count,
        // not a step-start, so pairing them as `prev` would attribute the
        // WRONG window to a bucket:
        // - terminal sorts last so it is naturally only ever `curr`, but
        //   guard explicitly so a future caller producing a non-last or
        //   duplicate terminal can't fall into the `None` step_index
        //   timestamp-window branch below and attach a bogus rate.
        // - StepEnd[k] carries step_index k (same as StepStart[k]); if its
        //   step's own StepStart[k] -> StepEnd[k] pair returned None (a
        //   BACKWARD count, e < s — a counter reset; a stalled step e == s
        //   instead yields a measured-ZERO rate, see
        //   StimulusEvent::rate_components), the bucket k key is still empty,
        //   so without this guard the next pair StepEnd[k] -> StepStart[k+1]
        //   would or_insert an inter-step teardown-gap rate into bucket k,
        //   mislabeling gap throughput as step k's. The guard keeps bucket k
        //   sourced ONLY by its own StepStart[k] -> StepEnd[k] pair (whether
        //   a measured zero or a real rate), never a leaked gap rate.
        if prev.is_terminal || prev.is_step_end {
            continue;
        }
        // Emit the iteration_rate Rate's two components (the per-step
        // iteration delta + window seconds) rather than the ready ratio, so
        // derive_rate_metrics re-pools Σiters/Σseconds at every aggregation
        // level. rate_components is None only when the count went BACKWARD
        // (e < s, a counter reset), an event lacks total_iterations, or the
        // window is zero-length; a non-advancing count (e == s over a
        // positive window) yields Some((0.0, secs)) — a measured zero, not
        // None (the same shared formula Timeline::build's rate_to divides
        // for display).
        let Some((iters, secs)) = prev.rate_components(curr) else {
            continue;
        };
        // Attribute the rate to the step PREV starts: the guards above
        // leave `prev` as a StepStart (or a legacy/synthetic step event),
        // so the pair (prev, curr) measures iterations accumulated from
        // prev's step-start to curr's — the throughput DURING prev's step.
        // Match by prev.step_index, NOT a timestamp window: the bucket
        // windows here are workload-relative capture offsets (the step
        // INTERIOR, 10-90% of the workload per
        // compute_periodic_boundaries_ns), so prev.elapsed_ms — the
        // step-START — lands in the inter-step gap, inside no bucket's
        // [start_ms, end_ms) window; a window match would drop every
        // rate. This mirrors the legacy Timeline::build alignment:
        // phase[i] gets events[i]→events[i+1] where events[i] is
        // phase[i]'s left boundary = prev.
        //
        // Stimulus events without a step stamp (step_index == None:
        // legacy / synthetic callers) fall back to the timestamp
        // window — half-open [start_ms, end_ms) with a single-sample
        // (start == end) equality carve-out so a boundary event lands
        // in exactly one bucket.
        for bucket in buckets.iter_mut() {
            let in_bucket = match prev.step_index {
                Some(k) => bucket.step_index == k,
                None => {
                    if bucket.start_ms == bucket.end_ms {
                        prev.elapsed_ms == bucket.start_ms
                    } else {
                        prev.elapsed_ms >= bucket.start_ms && prev.elapsed_ms < bucket.end_ms
                    }
                }
            };
            if in_bucket {
                // Both components come from the SAME (prev, curr) pair and are
                // inserted together; or_insert keeps the first pair that
                // matched this bucket (mirrors the prior first-wins
                // iteration_rate attribution). The derive_rate_metrics
                // post-pass below produces iteration_rate from them.
                bucket
                    .metrics
                    .entry("total_phase_iterations".to_string())
                    .or_insert(iters);
                bucket
                    .metrics
                    .entry("total_phase_duration_sec".to_string())
                    .or_insert(secs);
                break;
            }
        }
    }
    // Stimulus/monitor components are injected ABOVE, AFTER
    // buckets_from_grouped's own derive_rate_metrics already ran; re-derive
    // here so a Rate over a post-injected component is produced for these
    // buckets too. Idempotent: re-deriving an unchanged map is a no-op.
    for bucket in buckets.iter_mut() {
        crate::stats::derive_rate_metrics(&mut bucket.metrics);
    }
}

/// Build per-phase metric buckets from a sample series.
///
/// Walks [`crate::scenario::sample::SampleSeries::by_stamped_phase`]
/// to group every stamped sample under its bridge-stamped
/// `step_index` (NOT re-derived from elapsed-ms windows; the
/// bridge stamp is authoritative because the capture path knows
/// the phase it fired from while the time window cannot recover
/// the phase when stimulus events arrive late or out of order).
///
/// For each phase observed (BASELINE under `step_index = 0`,
/// scenario Steps under `step_index = 1..=N` per the 1-indexed
/// phase convention) emits one [`PhaseBucket`] with `step_index`
/// as the key, `label` derived per the BASELINE/Step\[k\]
/// convention, `start_ms` / `end_ms` from the first / last
/// sample's `elapsed_ms`, `sample_count` from the bucketed
/// samples, and `metrics` from the per-kind reduction described
/// on [`PhaseBucket`]. Metrics whose per-sample reading returns
/// `None` for every sample in the bucket are omitted entirely
/// (absent → "no data") rather than collapsed to `Some(0.0)`
/// (real zero), preserving the sentinel-free contract.
///
/// Returns an empty `Vec` when the input series is empty (no
/// samples captured), distinct from returning a single empty
/// BASELINE bucket — the former means the periodic-capture path
/// never fired, the latter means it fired but no metric reading
/// came back.
///
/// Live production caller: `evaluate_vm_result` in
/// `src/test_support/eval/mod.rs` drains the snapshot bridge, builds
/// a `SampleSeries`, and routes it through this fn to populate
/// `AssertResult.stats.phases`. Exposed `pub` (not `pub(crate)`)
/// so out-of-tree consumers — payload authors writing custom
/// eval paths against the publicly-drainable
/// `result.snapshot_bridge` — can produce the same per-phase
/// aggregate shape without re-implementing the bucketing logic.
pub fn build_phase_buckets(samples: &crate::scenario::sample::SampleSeries) -> Vec<PhaseBucket> {
    // Borrowed per-tick monitor samples (None when no MonitorReport
    // was attached, e.g. host-only fixture tests).
    let monitor_samples: &[crate::monitor::MonitorSample] =
        samples.monitor().map(|m| m.samples()).unwrap_or(&[]);
    // vCPU-preemption exemption window for the per-phase stall predicate
    // (see build_phase_buckets_with_stimulus). 0 (no monitor) derives from
    // CONFIG_HZ inside compute_metrics.
    let preemption_threshold_ns: u64 = samples
        .monitor()
        .map(|m| m.preemption_threshold_ns())
        .unwrap_or(0);
    // Group by the bridge-stamped step_index. Without a stimulus
    // timeline to remap against, the stamped index is the only phase
    // signal available — see `build_phase_buckets_with_stimulus` (and
    // `SampleSeries::by_stimulus_phase`) for the offset-remap that
    // corrects a deferred-fire burst (every capture stamping the same
    // late CURRENT_STEP). The bucket window and the monitor folding
    // share the run-relative frame here, so the clock offset is `0`;
    // synthetic / legacy fixture samples carry `boundary_offset_ms =
    // None` and the window falls back to `elapsed_ms`.
    buckets_from_grouped(
        samples.by_stamped_phase(),
        monitor_samples,
        0,
        preemption_threshold_ns,
    )
}

/// Per-phase CPU-time delta (ns) for one field family
/// (`stime`/`signal_stime` or `utime`/`signal_utime`), folded host-side
/// from the frozen `task_struct` enrichments captured at the phase's
/// freeze boundaries. Backs the injected `system_time_ns` /
/// `user_time_ns` per-phase metrics.
///
/// The unit is the THREAD GROUP, not the individual task: the kernel's
/// `thread_group_cputime` is `signal_struct.{u,s}time` (the accumulator
/// a dying thread's time is folded into at exit) plus the live threads'
/// `task_struct.{u,s}time`. `signal_struct` is shared across a thread
/// group, so its value is counted once per `tgid`; the live counters are
/// summed across the group's threads. For each `tgid` the group total is
/// taken at the FIRST and LAST sample in which the group had a READABLE
/// total (ordered by capture time) and `last - first` is summed across
/// groups.
///
/// Per-group first-seen/last-seen, NOT a per-sample cross-task SUM then a
/// Counter `last - first`: the captured set changes between freezes
/// (system tasks churn in and out). A task carrying a large cumulative
/// counter that appears only in a LATER sample would dump its entire
/// pre-phase history into a summed delta, inflating the phase value
/// many-fold. Subtracting each group's OWN first-seen total cancels its
/// pre-phase history, so a late-joining group contributes only what it
/// accrued while observed — bounding the result by wall-clock × cores. A
/// group's thread exiting mid-phase does not dip the total: its time
/// moves from a `task_struct` counter into `signal_struct`, both of which
/// the group total includes.
///
/// A `None` `signal_field` is a `signal_struct` translate miss, NOT a
/// real zero (which reads `Some(0)`): such a sample is omitted for that
/// group so every endpoint is a full live+signal total — mixing a
/// live-only endpoint with a live+signal one would otherwise leak the
/// cumulative accumulator as a phantom positive. A numeric `tgid` reused
/// within the phase (process exit + PID realloc) can read lower at last
/// than first; `saturating_sub` clamps that to 0 rather than wrapping.
///
/// Returns `None` when no group was observed with a readable total across
/// at least two samples — no delta is measurable — keeping an absent
/// per-phase bucket key distinct from a real `0` (a qualifying group
/// whose counters did not advance yields `Some(0.0)`). Accumulates in
/// `u128` to stay exact before the final `f64` (a phase can total many
/// task-seconds of ns).
fn phase_group_cpu_delta(
    samples: &[crate::scenario::sample::Sample<'_>],
    task_field: impl Fn(&crate::monitor::task_enrichment::TaskEnrichment) -> u64,
    signal_field: impl Fn(&crate::monitor::task_enrichment::TaskEnrichment) -> Option<u64>,
) -> Option<f64> {
    // Per-tgid `thread_group_cputime` total at one sample: the shared
    // signal_struct accumulator (once per group) plus the sum of the
    // group's live threads' task_struct counter. A group is INCLUDED for
    // a sample only when its signal accumulator was readable there (some
    // thread of the tgid carried `Some`): a `None` is a signal_struct
    // translate miss (see `task_enrichment.rs`), NOT a real zero — a real
    // zero reads `Some(0)`. Omitting unreadable-signal groups keeps every
    // endpoint a FULL group total (live + signal), so the first/last delta
    // can never mix a live-only endpoint with a live+signal endpoint and
    // leak the cumulative accumulator as a phantom positive.
    let group_totals = |s: &crate::scenario::sample::Sample<'_>| {
        let mut live: std::collections::HashMap<i32, u128> = std::collections::HashMap::new();
        let mut signal: std::collections::HashMap<i32, Option<u128>> =
            std::collections::HashMap::new();
        for t in s.snapshot.task_enrichments() {
            *live.entry(t.tgid).or_insert(0) += u128::from(task_field(t));
            match signal_field(t) {
                // Shared across the group — any readable thread fixes it.
                Some(v) => {
                    signal.insert(t.tgid, Some(u128::from(v)));
                }
                None => {
                    signal.entry(t.tgid).or_insert(None);
                }
            }
        }
        live.into_iter()
            .filter_map(|(tgid, l)| {
                signal
                    .get(&tgid)
                    .copied()
                    .flatten()
                    .map(|sig| (tgid, l + sig))
            })
            .collect::<std::collections::HashMap<i32, u128>>()
    };

    // Order by capture time so first/last are the earliest/latest
    // boundary — the grouped vec is not guaranteed sorted (the
    // offset-remap in the stimulus path can reorder samples).
    let mut ordered: Vec<&crate::scenario::sample::Sample<'_>> = samples.iter().collect();
    // `elapsed_ms` is now Option: a sample with neither a
    // boundary offset nor a measured elapsed has no time anchor — sort
    // it LAST (u64::MAX), never first, so an untimestamped sample can't
    // become the spurious earliest first_seen endpoint.
    ordered.sort_by_key(|s| s.boundary_offset_ms.or(s.elapsed_ms).unwrap_or(u64::MAX));

    // Per tgid: its full group total at the first and last sample in which
    // it had a readable total, and how many such samples it had.
    let mut first_seen: std::collections::HashMap<i32, u128> = std::collections::HashMap::new();
    let mut last_seen: std::collections::HashMap<i32, u128> = std::collections::HashMap::new();
    let mut readable_count: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
    for s in ordered {
        for (tgid, total) in group_totals(s) {
            first_seen.entry(tgid).or_insert(total);
            last_seen.insert(tgid, total);
            *readable_count.entry(tgid).or_insert(0) += 1;
        }
    }

    let mut sum: u128 = 0;
    let mut measured = false;
    for (tgid, last) in &last_seen {
        // A delta needs the group observed with a readable total across
        // TWO boundaries; one readable sample gives first == last and no
        // measurable in-phase growth (and a tgid that appears in only one
        // sample, or whose signal_struct never translated, never reaches
        // 2). saturating_sub clamps a last < first read — a numeric tgid
        // reused within the phase (process exit + PID realloc to a fresh
        // group starting near zero) reads lower at last; clamp to 0 rather
        // than wrap.
        if readable_count.get(tgid).copied().unwrap_or(0) < 2 {
            continue;
        }
        measured = true;
        // first_seen always holds tgid (populated in the same pass).
        let first = first_seen.get(tgid).copied().unwrap_or(*last);
        sum += last.saturating_sub(first);
    }
    // None when no group was observed with a readable total across two
    // samples — unmeasurable, so the bucket key stays ABSENT (distinct
    // from a real 0: a qualifying group whose counters did not advance
    // yields Some(0.0)).
    measured.then_some(sum as f64)
}

/// Per-phase per-CPU spatial-max fold for a monotonic per-CPU counter: the
/// BUSIEST CPU's delta over the phase (`max_key`) + the concentration ratio
/// `max / mean` over the reporting CPUs (`concentration_key`). A custom
/// per-CPU-delta fold — NOT a read_sample arm (read_sample yields the cross-CPU
/// SUM as one f64, with no per-CPU vector). `field` selects the per-CPU counter
/// (`|c| c.irqs_sum` for hardirqs, `|c| c.softirqs[NET_RX]` for NET_RX softirqs).
///
/// Endpoints are the earliest + latest freeze BY `win` (the SAME key the bucket
/// window uses; `samples_in_phase` is NOT positionally ordered — the stimulus
/// offset-remap can reorder it). The per-CPU delta is correlated by the
/// per_cpu_time `cpu` FIELD (not vec position) over the CPUs present in BOTH
/// endpoints; a CPU absent from either is skipped (hotplug defense — a no-op on
/// ktstr's boot-fixed vCPU set, correct-by-construction if a hot-add Op lands).
/// `saturating_sub` clamps a per-CPU counter regress (the kernel counters are
/// monotonic; a regress would signal a reset). The per-CPU delta is taken
/// FIRST, the spatial max SECOND — so a busiest CPU that SHIFTS across the phase
/// is captured as max(per-CPU deltas), not a delta of spatial maxes.
///
/// Loud-absent: `max_key` needs at least 2 freezes spanning a positive `win`
/// window and at least 1 defined-delta CPU; `concentration_key` ADDITIONALLY
/// needs at least 2 defined-delta CPUs (a 1-CPU intersection makes max/mean == 1
/// a structural artifact) and a positive mean (a NaN would fail the sidecar's
/// is_finite insert-guard).
fn fold_per_cpu_spatial_max(
    metrics: &mut std::collections::BTreeMap<String, f64>,
    samples_in_phase: &[crate::scenario::sample::Sample<'_>],
    win: impl Fn(&crate::scenario::sample::Sample<'_>) -> Option<u64>,
    field: impl Fn(&crate::monitor::dump::PerCpuTimeStats) -> u64,
    max_key: &str,
    concentration_key: &str,
) {
    let placed: Vec<(&crate::scenario::sample::Sample<'_>, u64)> = samples_in_phase
        .iter()
        .filter(|s| !s.snapshot.per_cpu_time().is_empty())
        .filter_map(|s| win(s).map(|w| (s, w)))
        .collect();
    if let (Some(&(first_s, fw)), Some(&(last_s, lw))) = (
        placed.iter().min_by_key(|(_, w)| *w),
        placed.iter().max_by_key(|(_, w)| *w),
    ) && fw < lw
    {
        // Per-CPU delta over the cpu-field intersection of the two endpoint
        // freezes; clamp a regress to 0.
        let deltas: Vec<f64> = first_s
            .snapshot
            .per_cpu_time()
            .iter()
            .filter_map(|c0| {
                last_s
                    .snapshot
                    .per_cpu_time_at(c0.cpu)
                    .map(|c1| field(c1).saturating_sub(field(c0)) as f64)
            })
            .collect();
        if let Some(max) = deltas.iter().copied().reduce(f64::max) {
            metrics.entry(max_key.to_string()).or_insert(max);
            if deltas.len() >= 2 {
                let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
                if mean > 0.0 {
                    metrics
                        .entry(concentration_key.to_string())
                        .or_insert(max / mean);
                }
            }
        }
    }
}

/// Assemble [`PhaseBucket`]s from a pre-grouped phase map. Shared by
/// [`build_phase_buckets`] (grouping by the bridge-stamped step_index)
/// and [`build_phase_buckets_with_stimulus`] (grouping by the
/// offset-remapped step).
///
/// `monitor_to_window_offset_ms` is subtracted from each
/// [`crate::monitor::MonitorSample`] `elapsed_ms` before the window
/// test, bringing the monitor sample's run-relative timestamp into the
/// bucket-window frame: `0` when both share the run-relative frame, the
/// stimulus/monitor clock skew (see [`monitor_clock_offset`]) when the
/// window is workload-relative (boundary-offset) but the monitor samples
/// remain run-relative.
///
/// Each bucket folds the monitor samples whose (shifted) elapsed_ms
/// lands in the bucket window — supplying metrics like
/// `avg_imbalance_ratio` that need per-CPU full-class `rq.nr_running`,
/// which the bridge-captured Snapshot does not expose (Snapshot carries
/// scx_rq.nr_running only).
///
/// `preemption_threshold_ns` is forwarded to `fold_monitor_into_bucket`
/// (and thence `compute_metrics`) for the per-phase stall predicate; see
/// `fold_monitor_into_bucket`.
fn buckets_from_grouped(
    by_phase: std::collections::BTreeMap<u16, Vec<crate::scenario::sample::Sample<'_>>>,
    monitor_samples: &[crate::monitor::MonitorSample],
    monitor_to_window_offset_ms: i64,
    preemption_threshold_ns: u64,
) -> Vec<PhaseBucket> {
    let mut out: Vec<PhaseBucket> = Vec::with_capacity(by_phase.len());
    for (step_index, samples_in_phase) in by_phase {
        let label = if step_index == 0 {
            "BASELINE".to_string()
        } else {
            // Scenario-Step ordinal lives at `step_index - 1`
            // because phase 0 is BASELINE under the 1-indexed
            // encoding; saturate at 0 if the underflow guard
            // ever fires (unreachable for the current encoding
            // — step_index here came from the bucket key so the
            // `> 0` branch is satisfied — but keep the guard so
            // a future caller that hands in a synthetic
            // `step_index = 0` does not panic).
            format!("Step[{}]", step_index.saturating_sub(1))
        };
        let sample_count = samples_in_phase.len();
        // Bucket window: prefer each capture's workload-relative
        // scheduled boundary offset (stable across a deferred-fire
        // burst) over the run-relative fire time; fall back to
        // `elapsed_ms` per-sample when no offset was stamped (synthetic
        // / on-demand). min/max rather than first/last because the
        // offset-remap in the stimulus path can reorder samples
        // relative to drain order.
        let win = |s: &crate::scenario::sample::Sample<'_>| s.boundary_offset_ms.or(s.elapsed_ms);
        let (start_ms, end_ms) = if samples_in_phase.is_empty() {
            (0, u64::MAX)
        } else {
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for s in &samples_in_phase {
                // Skip a sample with no time anchor (no boundary offset
                // AND no measured elapsed): coercing it to 0
                // would pull start_ms to 0 and over-fold monitor samples.
                // If EVERY sample in the phase is unanchored, lo/hi stay
                // (u64::MAX, 0) — an inverted window that folds nothing,
                // the correct "no placeable samples" outcome.
                if let Some(w) = win(s) {
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
            }
            (lo, hi)
        };
        let mut metrics: std::collections::BTreeMap<String, f64> =
            std::collections::BTreeMap::new();
        for metric_def in crate::stats::METRICS {
            let per_sample_readings: Vec<f64> = samples_in_phase
                .iter()
                .filter_map(|s| metric_def.read_sample(s))
                .collect();
            if per_sample_readings.is_empty() {
                // No per-sample reading for any sample in this
                // bucket -- the metric is host-side-only
                // (cross-cgroup fold) or its dispatch arm has
                // not landed yet. Omit the key rather than
                // collapsing to `Some(0.0)` so the renderer
                // paints "absent" vs "real zero" distinctly.
                continue;
            }
            if let Some(reduced) =
                crate::stats::aggregate_samples_for_phase(metric_def, &per_sample_readings)
            {
                metrics.insert(metric_def.name.to_string(), reduced);
            }
        }
        // Per-phase system / user CPU time (ns), injected post-hoc as a
        // per-thread-GROUP delta over the phase's freeze samples (NOT a
        // read_sample metric — a per-sample cross-task sum then a Counter
        // delta inflates when the captured task set churns; see
        // `phase_group_cpu_delta`). Observer-free: reads the frozen
        // task_struct.{s,u}time + thread-group signal_struct accumulator
        // already captured in each sample's enrichments.
        if let Some(v) = phase_group_cpu_delta(&samples_in_phase, |t| t.stime, |t| t.signal_stime) {
            metrics.entry("system_time_ns".to_string()).or_insert(v);
        }
        if let Some(v) = phase_group_cpu_delta(&samples_in_phase, |t| t.utime, |t| t.signal_utime) {
            metrics.entry("user_time_ns".to_string()).or_insert(v);
        }
        // Per-CPU spatial axes (busiest-CPU counter delta + concentration) over
        // the per_cpu_time freezes: hardirqs (kstat.irqs_sum) and NET_RX softirqs
        // (kstat.softirqs[NET_RX]). Both are monotonic per-CPU counters; the fold
        // takes the per-CPU delta FIRST then the spatial max — see
        // `fold_per_cpu_spatial_max` for the shared endpoint / cpu-field
        // intersection / saturating / loud-absent rules.
        fold_per_cpu_spatial_max(
            &mut metrics,
            &samples_in_phase,
            win,
            |c| c.irqs_sum,
            "max_cpu_hardirqs",
            "max_cpu_hardirq_concentration",
        );
        fold_per_cpu_spatial_max(
            &mut metrics,
            &samples_in_phase,
            win,
            |c| c.softirqs[crate::monitor::btf_offsets::SOFTIRQ_NET_RX],
            "max_cpu_softirq_net_rx",
            "max_cpu_softirq_net_rx_concentration",
        );
        let mut bucket = PhaseBucket {
            step_index,
            label,
            start_ms,
            end_ms,
            sample_count,
            metrics,
            per_cgroup: std::collections::BTreeMap::new(),
        };
        // Per-phase MonitorSample windowing for monitor-derived metrics
        // (avg_imbalance_ratio). Factored into `fold_monitor_into_bucket`
        // so a synthesized zero-capture bucket (see
        // `build_phase_buckets_with_stimulus`) recovers its in-window
        // imbalance too — same formula and frame, but over the
        // synthesized bucket's FULL stimulus window vs this captured
        // bucket's narrower interior capture-offset span.
        fold_monitor_into_bucket(
            &mut bucket,
            monitor_samples,
            monitor_to_window_offset_ms,
            preemption_threshold_ns,
        );
        // Phase-wall denominators for the IRQ rates, co-inserted
        // both-or-neither with the read_sample-folded IRQ numerators above so
        // derive_rate_metrics pairs num/den from THIS bucket (never cross-paired
        // with a stimulus event's total_phase_duration_sec — a distinct
        // metric). The window is the CAPTURE span (start_ms/end_ms = first->last
        // freeze offset), the same span the Counter numerators accrue over:
        // self-consistent for the ns/ns fraction, and a capture-window rate for
        // the per-second rates (the d/span factor cancels in an A/B compare with
        // matched cadence). Gate on total_hardirqs (present iff per_cpu_time was
        // captured; irqtime-independent, unlike total_irq_time_ns) + a positive
        // window. ms->ns and ms->s scaling lives HERE (derive_rate_metrics does
        // bare num/den). One insertion covers both derive sites:
        // build_phase_buckets_with_stimulus folds over buckets_from_grouped's
        // output, so its later re-derive sees these denominators too.
        if bucket.metrics.contains_key("total_hardirqs") && bucket.end_ms > bucket.start_ms {
            let wall_ms = (bucket.end_ms - bucket.start_ms) as f64;
            bucket
                .metrics
                .entry("total_phase_wall_ns".to_string())
                .or_insert(wall_ms * 1_000_000.0);
            bucket
                .metrics
                .entry("total_phase_wall_sec".to_string())
                .or_insert(wall_ms / 1000.0);
        }
        // Derive Rate metrics AFTER every component source is folded in:
        // the METRICS reductions + system/user_time_ns above AND the
        // monitor-injected components (e.g. avg_imbalance_ratio, whose ONLY
        // source is fold_monitor_into_bucket). Deriving before the monitor
        // fold would silently drop a Rate over a monitor-only component. A
        // Rate has no samples of its own (see MetricKind::Rate); this is its
        // per-phase producer. Inert until a Rate registers. The stimulus
        // sibling build_phase_buckets_with_stimulus re-derives again after
        // its own later injections (idempotent).
        crate::stats::derive_rate_metrics(&mut bucket.metrics);
        out.push(bucket);
    }
    out
}

/// Fold the per-CPU full-class imbalance from the monitor samples whose
/// run-relative timestamp falls in `bucket`'s `[start_ms, end_ms)`
/// window into `bucket.metrics["avg_imbalance_ratio"]`.
///
/// The monitor sample's run-relative `elapsed_ms` is shifted into the
/// bucket-window frame (subtract `monitor_to_window_offset_ms`) before
/// the half-open `[start_ms, end_ms)` test so a MonitorSample whose
/// timestamp equals the boundary lands in exactly one bucket (not both
/// adjacent buckets — the closed-on-right form double-counted boundary
/// samples). Single-sample phases (`start_ms == end_ms`) use explicit
/// equality so the window is not empty.
///
/// Filters via [`crate::monitor::sample_looks_valid`] (implausible-DSQ
/// samples) before the fold; `compute_metrics` additionally drops
/// empty-cpus samples (which would default `imbalance_ratio` to 1.0 and
/// pull the mean toward "perfect balance", masking a real regression) —
/// matching the legacy `Timeline::build` path's filter discipline.
///
/// Folds the FULL monitor-derived metric set the legacy `Timeline::build`
/// reducer (`crate::timeline::compute_metrics`) produces —
/// `avg_imbalance_ratio`, `max_imbalance_ratio`, `avg_dsq_depth`,
/// `max_dsq_depth`, `stuck_count`, and the `total_fallback` /
/// `total_keep_last` counter deltas — over the bucket's window.
/// `avg_imbalance_ratio`, `max_imbalance_ratio`, and `stuck_count` are
/// folded for EVERY bucket with in-window monitor samples: none of the three
/// has a `read_sample` dispatch arm (`crate::stats` `read_sample` has arms
/// only for the dsq / fallback keys; these three fall to `_ => None`), so the
/// per-sample capture path never produces them and monitor is their only
/// per-bucket source on captured AND synthesized buckets alike — a captured
/// (common-case) phase must report its per-phase imbalance peak and stall
/// count, not drop them. (`avg_imbalance_ratio` is genuinely ext-metrics-only;
/// `max_imbalance_ratio` and `stuck_count` ALSO carry a typed `GauntletRow`
/// accessor sourced from the whole-run MonitorSummary, so their per-phase
/// fold here feeds per-phase RENDERING only — the run-level value stays the
/// typed accessor, and both `populate_run_ext_metrics*` skip them via
/// `TYPED_FIELD_NAMES` to avoid a double-source.) The dsq / fallback set
/// (`avg_dsq_depth`,
/// `max_dsq_depth`, `total_fallback`, `total_keep_last`) is folded ONLY for
/// a synthesized (`sample_count == 0`) bucket: a captured bucket sources
/// those from its read_sample captures and keeps its pre-synthesize
/// behavior, while a synthesized bucket has no captures, so monitor is its
/// only source — restoring the rendered timeline to PARITY with the old
/// `Timeline::build` fallback (the path a zero-capture-with-monitor run
/// took before the synthesize seam flipped it onto from_phase_buckets;
/// `format_phases` renders these folded metrics for a `sample_count == 0`
/// bucket via its `has_monitor_metrics` gate). Each key is `or_insert` so
/// it never overwrites a value already present. Parity with
/// `Timeline::build` is exact for the production case; for legacy
/// ScenarioEnd-but-no-StepEnd data the synthesized last-step window clamps
/// to the terminal rather than extending to end-of-monitor, and a
/// synthesized bucket's dsq metrics come from the monitor
/// `CpuSnapshot.local_dsq_depth` axis (vs a captured bucket's DSQ-walker
/// axis) — same metric, different sampling axis.
///
/// `bucket`'s `[start_ms, end_ms)` IS the window basis and differs by
/// bucket kind: a captured bucket's is the min/max of its samples'
/// interior capture offsets; a synthesized bucket's is its full
/// `[StepStart, StepEnd)` stimulus window. The monitor sample's
/// run-relative `elapsed_ms` is shifted into that frame (subtract
/// `monitor_to_window_offset_ms`) before the half-open test so a sample
/// on the boundary lands in exactly one bucket. `compute_metrics` returns
/// fallback / keep_last as RATES; this re-derives the bucket-native
/// counter DELTAS (so `phase_from_bucket` re-rates them over the bucket
/// window like the read_sample path) using the same `counter_delta` clamp.
fn fold_monitor_into_bucket(
    bucket: &mut PhaseBucket,
    monitor_samples: &[crate::monitor::MonitorSample],
    monitor_to_window_offset_ms: i64,
    preemption_threshold_ns: u64,
) {
    let start_ms = bucket.start_ms;
    let end_ms = bucket.end_ms;
    let in_window = |monitor_ms: u64| -> bool {
        let shifted = monitor_ms as i64 - monitor_to_window_offset_ms;
        if shifted < 0 {
            return false;
        }
        let m = shifted as u64;
        if start_ms == end_ms {
            m == start_ms
        } else {
            m >= start_ms && m < end_ms
        }
    };
    // Filter via sample_looks_valid (matches Timeline::build) so an invalid
    // sample (empty cpus -> imbalance_ratio 1.0 default) doesn't pull the
    // mean toward "perfect balance" and mask a real regression.
    let phase_monitor_samples: Vec<&crate::monitor::MonitorSample> = monitor_samples
        .iter()
        .filter(|s| in_window(s.elapsed_ms))
        .filter(|s| crate::monitor::sample_looks_valid(s))
        .collect();
    if phase_monitor_samples.is_empty() {
        return;
    }
    let pm = crate::timeline::compute_metrics(&phase_monitor_samples, preemption_threshold_ns);
    // sample_count == 0 IFF the bucket is synthesized: buckets_from_grouped
    // only emits >=1-sample buckets, the synthesize loop only emits 0-sample
    // ones. A CAPTURED bucket folds the source-less monitor signals (avg/max
    // imbalance + stuck_count — none have a read_sample arm) but NOT the
    // dsq / fallback set, which it sources from its read_sample captures.
    // Restricting only that dsq / fallback set to synthesized buckets
    // preserves captured-bucket behavior; a synthesized bucket has no
    // captures, so monitor is its only source and it takes the full set
    // (Timeline::build render parity).
    let synthesized = bucket.sample_count == 0;
    let mut put = |key: &str, v: f64| {
        if v.is_finite() {
            bucket.metrics.entry(key.to_string()).or_insert(v);
        }
    };
    if let Some(v) = pm.avg_imbalance {
        put("avg_imbalance_ratio", v);
    }
    // max_imbalance_ratio (Peak) and stuck_count (Counter) have NO read_sample
    // dispatch arm (both fall to `_ => None` in crate::stats read_sample), so
    // the per-sample capture path never produces them and a CAPTURED bucket
    // would otherwise never carry them — they would surface only on synthesized
    // (zero-capture) buckets. Fold them for EVERY monitor-bearing bucket — like
    // avg_imbalance_ratio above — so a captured (common-case) phase reports its
    // per-phase imbalance peak and stall count instead of dropping them. (Both
    // are monitor-axis signals: imbalance from full-class rq.nr_running, stalls
    // from non-advancing rq.clock across consecutive samples — neither is in
    // the guest Snapshot read_sample observes. Both ALSO carry a typed
    // run-level GauntletRow accessor, so this per-phase fold feeds per-phase
    // RENDERING only; TYPED_FIELD_NAMES keeps them out of the run-level
    // ext_metrics so the typed accessor stays the single run-level source.)
    // `or_insert` still guards against overwriting a value already present.
    if let Some(v) = pm.max_imbalance {
        put("max_imbalance_ratio", v);
    }
    if pm.stall_count > 0 {
        put("stuck_count", pm.stall_count as f64);
    }
    if synthesized {
        if let Some(v) = pm.avg_dsq_depth {
            put("avg_dsq_depth", v);
        }
        put("max_dsq_depth", pm.max_dsq_depth as f64);
        // Bucket-native counter totals for fallback / keep_last: the first
        // and last in-window samples carrying event counters, clamped with
        // the same counter_delta MonitorSummary uses (a mid-phase scheduler
        // restart can reset the counter, producing a negative raw delta).
        // phase_from_bucket re-rates these over the bucket window, matching
        // the read_sample representation captured buckets carry.
        let has_events =
            |s: &&crate::monitor::MonitorSample| s.cpus.iter().any(|c| c.event_counters.is_some());
        let first_ev = phase_monitor_samples.iter().copied().find(has_events);
        let last_ev = phase_monitor_samples.iter().copied().rev().find(has_events);
        if let (Some(first), Some(last)) = (first_ev, last_ev) {
            let fb = crate::monitor::counter_delta(
                last.sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0),
                first
                    .sum_event_field(|e| e.select_cpu_fallback)
                    .unwrap_or(0),
            );
            let kl = crate::monitor::counter_delta(
                last.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0),
                first.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0),
            );
            put("total_fallback", fb as f64);
            put("total_keep_last", kl as f64);
        }
    }
}

/// Clock skew (ms) between the host monitor's run-relative timeline and
/// the guest's scenario-relative stimulus timeline, computed the same
/// way as [`crate::timeline::Timeline::build`]: the first significant
/// monitor sample (elapsed > 500 ms, non-empty cpus) and the earliest
/// stimulus event roughly coincide at scenario start. Returns
/// `first_monitor_ms - first_stimulus_ms`; subtract from a monitor
/// sample's elapsed_ms to reach the scenario-relative (boundary-offset)
/// window frame. `0` when either timeline is empty (nothing to align,
/// so the run-relative frames are used as-is).
fn monitor_clock_offset(
    stimulus_events: &[crate::timeline::StimulusEvent],
    monitor_samples: &[crate::monitor::MonitorSample],
) -> i64 {
    if stimulus_events.is_empty() || monitor_samples.is_empty() {
        return 0;
    }
    let first_stimulus_ms = stimulus_events
        .iter()
        .map(|e| e.elapsed_ms)
        .min()
        .unwrap_or(0);
    let first_monitor_ms = monitor_samples
        .iter()
        .find(|s| s.elapsed_ms > 500 && !s.cpus.is_empty())
        .map(|s| s.elapsed_ms)
        .unwrap_or_else(|| monitor_samples.first().map(|s| s.elapsed_ms).unwrap_or(0));
    first_monitor_ms as i64 - first_stimulus_ms as i64
}
