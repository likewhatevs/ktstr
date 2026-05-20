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
//!   `scx_ktstr.bss` map instances coexist in the kernel briefly.
//!   The struct_ops walker (`identify_active_obj_from_struct_ops`)
//!   finds the LIVE instance via `*scx_root` and records the set of
//!   map KVAs in `report.active_map_kvas`.
//! - **`Snapshot::active()` filters by combined (obj-name + KVA)**.
//!   The same-binary case — both bss maps named `scx_ktstr.bss` —
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
    let series = result.periodic_series();
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
         active-scheduler disambiguation is broken. First ambiguous: {}",
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

    // Hard-pin per-phase resolution: a regression that breaks
    // the NEW infrastructure (e.g., the KVA filter narrows wrong)
    // could silently let phase 0 (only primary bss exists, no
    // ambiguity to resolve) resolve while phase 1 (both bss
    // copies coexist, requires the KVA filter) consistently
    // fails — the `ok_count >= 1` check above would pass without
    // exercising the new code path. Require a successful
    // resolution in the POST-SWAP phase specifically. Use
    // SeriesField::phases_iter() to bind each value to its phase.
    use std::collections::BTreeSet;
    let post_swap = ktstr::assert::Phase::step(1);
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
    // The KVA filter's load-bearing case is phase 1 (post-swap,
    // both bss copies coexist). Require it specifically.
    if phases_with_samples.contains(&post_swap.as_u16()) {
        anyhow::ensure!(
            phases_with_ok.contains(&post_swap.as_u16()),
            "no post-swap (Phase::step(1)) sample resolved bpf_live_u64 — \
             the same-binary disambiguation path is the load-bearing \
             new code. phases_with_samples={phases_with_samples:?} \
             phases_with_ok={phases_with_ok:?} ok_count={ok_count}/{total}",
        );
    }
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
    duration_s = 8,
    watchdog_timeout_s = 15,
    num_snapshots = 4,
    cleanup_budget_ms = 5000,
    post_vm = assert_live_var_resolves_across_swap,
)]
fn live_var_resolves_across_same_binary_swap(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        // Phase 0: primary scheduler runs alone.
        Step::hold(HoldSpec::frac(0.3)),
        // Swap to the staged alt scheduler (same scx-ktstr binary,
        // distinct name → distinct `bpf_object` instance in the
        // kernel → distinct bss map KVA, even though the map name
        // is `scx_ktstr.bss` in BOTH).
        Step::with_op(
            Op::replace_scheduler(&STAGED_ALT_SCHED),
            HoldSpec::frac(0.7),
        ),
    ];
    execute_steps(ctx, steps)
}
