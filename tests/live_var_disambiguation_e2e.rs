//! End-to-end coverage of the per-snapshot scheduler-identity stamp
//! built in [`FailureDumpReport::active_map_kvas`](ktstr::monitor::dump::FailureDumpReport)
//! paired with the KVA-aware filter in
//! [`Snapshot::active`](ktstr::scenario::snapshot::Snapshot::active).
//!
//! Boots `scx-ktstr` under one name, swaps to a SECOND `scx-ktstr`
//! instance (same binary, distinct name) via `Op::ReplaceScheduler`
//! mid-scenario, and verifies that the new live-disambiguation
//! plumbing resolves the scheduler's bss counters CORRECTLY across
//! the swap WITHOUT the test author having to pass a picker.
//!
//! ## What this proves
//!
//! - **Walker populates `active_map_kvas`.** Post-swap, both
//!   `bpf_bpf.bss` map instances coexist in the kernel briefly.
//!   The struct_ops walker (`identify_active_obj_from_struct_ops`)
//!   finds the LIVE instance via `*scx_root` and records the set of
//!   map KVAs in `report.active_map_kvas`.
//! - **`Snapshot::active()` filters by combined (obj-name + KVA)**.
//!   The same-binary case — both bss maps named `bpf_bpf.bss` —
//!   resolves to ONLY the live instance's KVA set.
//! - **`series.bpf_live_u64(name)` works without explicit picker.**
//!   No `pickers::max_by_sum_u64` argument; no `live_bpf_vars_via`
//!   closure. Just `series.bpf_live_u64("nr_dispatched")`.
//!
//! ## Counterfactual
//!
//! Before this work, the test author had to write:
//! ```ignore
//! let f = series.live_bpf_vars_via(
//!     ["nr_dispatched"],
//!     pickers::max_by_sum_u64,
//! );
//! ```
//! to avoid `AmbiguousVar` post-swap. With active-scheduler
//! disambiguation the picker dance disappears.
//!
//! ## When writing a NEW same-binary swap test
//!
//! To get the deterministic disambiguation guarantee this test
//! exercises, the scenario MUST pin the OUTGOING scheduler's bss
//! BEFORE `Op::ReplaceScheduler` runs:
//! ```ignore
//! let steps = vec![
//!     Step::with_op(
//!         Op::pin_bpf_map("<obj>.bss"),
//!         HoldSpec::fixed(Duration::from_secs(2)),
//!     ),
//!     Step::with_op(
//!         Op::replace_scheduler(&NEW_SCHED),
//!         HoldSpec::fixed(Duration::from_secs(12)),
//!     ),
//! ];
//! ```
//! Without the `Op::pin_bpf_map` step, the kernel frees the OUTGOING
//! bss as soon as libbpf drops its fds, so the multi-bss window is
//! sub-millisecond and the walker rarely sees the ambiguous case.
//! Tests that omit the pin will PASS most runs (single-bss path)
//! and surface `AmbiguousVar` intermittently in CI when timing
//! happens to land a periodic capture inside the multi-bss window.
//! Use `HoldSpec::fixed` (NOT `HoldSpec::frac`) for the post-swap
//! hold so dispatch latency doesn't eat the hold budget via the
//! `.min(remaining)` clamp at `ops/mod.rs:1305-1307`.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const PRIMARY_SCHED: Scheduler =
    Scheduler::named("live_var_primary").binary(SchedulerSpec::Discover("scx-ktstr"));

const STAGED_ALT_SCHED: Scheduler =
    Scheduler::named("live_var_alt").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Drains the bridge's periodic captures and asserts the
/// active-scheduler walker's KVA filter resolves the LIVE bss
/// uniquely across the swap. The key assertion: at least one
/// steady-state sample (in EACH phase) resolves
/// `bpf_live_u64("nr_dispatched")` to a non-error `u64` —
/// proving the disambiguation pipeline works on samples taken
/// while a scheduler is attached.
///
/// **Mid-swap tolerance.** A periodic boundary that fires during
/// the brief window between detach and re-attach captures a
/// scheduler-less snapshot (no bss/data/rodata maps) and
/// surfaces as `NoActiveScheduler` — expected. The assertion
/// counts steady-state Ok samples per phase instead of requiring
/// every sample to resolve.
///
/// **The counterfactual.** Without the KVA filter, post-swap
/// samples (when both bss copies coexist) would surface
/// `AmbiguousVar`. Reaching this assertion with steady-state Ok
/// samples in BOTH phases proves the filter narrows to the live
/// copy.
fn assert_live_var_resolves_across_swap(result: &VmResult) -> Result<()> {
    // Drain the bridge ONCE — the deterministic multi-bss gate at
    // the end needs to inspect raw `report.maps[]` +
    // `report.active_map_kvas` per sample (not just the projected
    // SeriesField). Pre-scan drained for those counts BEFORE
    // moving it into the series constructor (DrainedSnapshotEntry
    // is not Clone).
    use std::collections::BTreeSet;
    let post_swap_phase = ktstr::assert::Phase::step(1);
    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let mut multi_bss_phase1_count = 0usize;
    let mut walker_published_phase1_count = 0usize;
    for row in &drained {
        if row.step_index != Some(post_swap_phase.as_u16()) {
            continue;
        }
        let bss_copies = row
            .report
            .maps
            .iter()
            .filter(|m| m.name == "bpf_bpf.bss")
            .count();
        if bss_copies >= 2 {
            multi_bss_phase1_count += 1;
        }
        if !row.report.active_map_kvas.is_empty() {
            walker_published_phase1_count += 1;
        }
    }
    let series =
        ktstr::scenario::sample::SampleSeries::from_drained_typed(drained, result.monitor.clone())
            .periodic_only();
    anyhow::ensure!(
        !series.is_empty(),
        "no periodic samples captured — the test cannot exercise \
         the post-swap disambiguation path",
    );

    let nr_dispatched: ktstr::assert::SeriesField<u64> = series.bpf_live_u64("nr_dispatched");
    let total = nr_dispatched.values_iter().count();
    let ok_count = nr_dispatched.values_iter().filter(|s| s.is_ok()).count();
    let err_count = total - ok_count;
    let first_err = nr_dispatched
        .values_iter()
        .find_map(|s| s.as_ref().err().map(|e| format!("{e}")));
    let first_ambiguous = nr_dispatched.values_iter().find_map(|s| match s {
        Err(ktstr::scenario::snapshot::SnapshotError::AmbiguousVar { .. }) => {
            Some(format!("{:?}", s))
        }
        _ => None,
    });

    // Hard fail: AmbiguousVar means the live filter did NOT
    // narrow the post-swap snapshot — the core regression
    // signal this test exists to prevent.
    anyhow::ensure!(
        first_ambiguous.is_none(),
        "bpf_live_u64 surfaced AmbiguousVar — the active-scheduler \
         KVA filter did NOT narrow the same-binary post-swap snapshot. \
         If this fires in YOUR test: the scenario likely omits \
         `Op::pin_bpf_map(\"<obj>.bss\")` BEFORE `Op::ReplaceScheduler`. \
         Without the pin, the multi-bss window is timing-dependent and \
         AmbiguousVar surfaces intermittently. See the module-level \
         \"When writing a NEW same-binary swap test\" section. First \
         ambiguous: {}",
        first_ambiguous.unwrap_or_default(),
    );

    // Soft tolerance: mid-swap windows can produce
    // NoActiveScheduler samples (scheduler detached + new one
    // not yet attached). Require AT LEAST ONE sample to resolve
    // — proves the steady-state disambiguation works.
    anyhow::ensure!(
        ok_count >= 1,
        "no periodic sample resolved bpf_live_u64(\"nr_dispatched\") — \
         {err_count}/{total} samples errored. First error: {}",
        first_err.unwrap_or_else(|| "(unset)".to_string()),
    );

    // Reject ActiveFilterExcludedMaps too — the KVA filter
    // narrowing to zero would surface this and we want to know
    // if it ever happens under the e2e (would indicate stale
    // walker capture vs. live captured maps).
    let first_filter_excluded = nr_dispatched.values_iter().find_map(|s| match s {
        Err(ktstr::scenario::snapshot::SnapshotError::ActiveFilterExcludedMaps { .. }) => {
            Some(format!("{:?}", s))
        }
        _ => None,
    });
    anyhow::ensure!(
        first_filter_excluded.is_none(),
        "bpf_live_u64 surfaced ActiveFilterExcludedMaps — the KVA whitelist \
         excluded every captured `<active_obj>.*` map (stale walker capture, \
         KVA aliasing, or walker mispointing). First: {}",
        first_filter_excluded.unwrap_or_default(),
    );

    // Hard-pin per-phase resolution. A regression that breaks
    // the NEW infrastructure (KVA filter narrows wrong) could
    // silently let phase 0 (only primary bss exists) resolve
    // while phase 1 (both bss copies coexist) fails — `ok_count
    // >= 1` would pass without exercising the new code path.
    // Require BOTH the post-swap phase to have at least one
    // sample AND at least one Ok in it; the `if-then` form would
    // give a free pass when phase 1 happened to have zero
    // samples (timing).
    let mut phases_with_samples: BTreeSet<u16> = BTreeSet::new();
    let mut phases_with_ok: BTreeSet<u16> = BTreeSet::new();
    for (phase_opt, slot) in nr_dispatched.phases_iter().zip(nr_dispatched.values_iter()) {
        if let Some(phase) = phase_opt {
            phases_with_samples.insert(phase.as_u16());
            if slot.is_ok() {
                phases_with_ok.insert(phase.as_u16());
            }
        }
    }
    anyhow::ensure!(
        phases_with_samples.contains(&post_swap_phase.as_u16()),
        "no post-swap (Phase::step(1)) samples captured — phase-1 \
         hold may be too short or num_snapshots too low to land a \
         sample after the swap. phases_with_samples={phases_with_samples:?}",
    );
    // Diagnostic: collect the first error per post-swap sample so the
    // assertion message names the actual failure shape (AmbiguousVar,
    // NoActiveScheduler with multi-copy reason, WalkerDriftedWithinPhase,
    // ...). Without this the test reports only "0 ok in post-swap"
    // and the operator can't tell which layer of the disambiguation
    // chain broke.
    let mut post_swap_errors: Vec<String> = Vec::new();
    for (phase_opt, slot) in nr_dispatched.phases_iter().zip(nr_dispatched.values_iter()) {
        if phase_opt == Some(post_swap_phase)
            && let Err(e) = slot
        {
            post_swap_errors.push(format!("{e:?}"));
        }
    }
    anyhow::ensure!(
        phases_with_ok.contains(&post_swap_phase.as_u16()),
        "no post-swap (Phase::step(1)) sample resolved bpf_live_u64 — \
         the same-binary disambiguation path is the load-bearing \
         new code. phases_with_samples={phases_with_samples:?} \
         phases_with_ok={phases_with_ok:?} ok_count={ok_count}/{total}. \
         post-swap errors observed: {post_swap_errors:?}",
    );

    // Hard deterministic gate. The `Op::pin_bpf_map("bpf_bpf.bss")`
    // step in Phase 1 of the scenario below holds the OUTGOING
    // bss alive across the swap, so post-swap captures MUST observe
    // ≥2 `bpf_bpf.bss` copies (multi-bss window). The target-free
    // struct_ops walker
    // (`monitor::bpf_prog::find_active_struct_ops_obj_no_target`)
    // MUST publish `active_map_kvas` for every multi-bss sample so
    // the consumer's downstream KVA filter narrows to the live
    // copy. A miss on either gate is a real regression in the
    // pin primitive or the target-free walker, respectively.
    anyhow::ensure!(
        multi_bss_phase1_count >= 1,
        "post-swap snapshots NEVER captured the multi-bss window — \
         the `Op::pin_bpf_map(\"bpf_bpf.bss\")` step in Phase 1 should \
         have held the OUTGOING bss alive across the swap. Check \
         the apply-time PinBpfMap log for a bail, OR the post-swap \
         hold (HoldSpec::fixed) for periodic-boundary distribution.",
    );
    anyhow::ensure!(
        walker_published_phase1_count >= 1,
        "post-swap snapshots captured the multi-bss window ({multi_bss_phase1_count} \
         samples with ≥2 bpf_bpf.bss copies) but the active-scheduler walker \
         NEVER published active_map_kvas. The target-free struct_ops walker is \
         expected to find the unique alive STRUCT_OPS prog with global-section \
         maps in used_maps and publish its kva set — see \
         `monitor::dump::identify_active_obj_from_struct_ops`'s primary path.",
    );
    Ok(())
}

/// The full live-disambiguation e2e: two phases, same binary,
/// swap mid-scenario. Periodic captures land in BOTH phases.
#[ktstr_test(
    scheduler = PRIMARY_SCHED,
    staged_schedulers = [STAGED_ALT_SCHED],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 15,
    watchdog_timeout_s = 30,
    num_snapshots = 8,
    cleanup_budget_ms = 5000,
    post_vm = assert_live_var_resolves_across_swap,
)]
fn live_var_resolves_across_same_binary_swap(ctx: &Ctx) -> Result<AssertResult> {
    use std::time::Duration;
    let steps = vec![
        // Phase 1 (Phase::step(0) → tag u16=1): primary scheduler
        // runs alone and we pin its bss via Op::pin_bpf_map. The pin
        // holds an OwnedFd against `bpf_bpf.bss`, bumping the kernel
        // refcount via __bpf_map_inc_not_zero so the OUTGOING bss
        // survives across the swap (kernel only frees the map when
        // every fd holder releases — see kernel/bpf/syscall.c:955).
        // 2s settle covers the BPF object load before the pin walks
        // the map ID space.
        Step::with_op(
            Op::pin_bpf_map("bpf_bpf.bss"),
            HoldSpec::fixed(Duration::from_secs(2)),
        ),
        // Phase 2 (Phase::step(1) → tag u16=2): swap to the staged
        // alt scheduler (same scx-ktstr binary, distinct name →
        // distinct `bpf_object` instance → distinct bss map_kva,
        // even though map name is `bpf_bpf.bss` in BOTH). With the
        // Phase 1 pin, both bss copies coexist throughout this step
        // — the multi-bss disambiguation path
        // (identify_active_obj_from_struct_ops's target-free walker)
        // is exercised on every post-swap capture. HoldSpec::fixed
        // avoids the frac-based clamp at ops/mod.rs:1305-1307 that
        // would collapse the post-swap hold to ~0 if dispatch
        // overran.
        Step::with_op(
            Op::replace_scheduler(&STAGED_ALT_SCHED),
            HoldSpec::fixed(Duration::from_secs(12)),
        ),
    ];
    execute_steps(ctx, steps)
}
