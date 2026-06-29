//! End-to-end proof that `#[ktstr_test(watch_bpf_maps = ...)]` reads a NAMED
//! scheduler BPF-map field observer-effect-free into an assertable run-level
//! metric.
//!
//! Boots scx-ktstr with a `watch_bpf_maps` target on the scheduler's `.bss`
//! global `ktstr_alloc_count` (a non-static `__u64` monotonic counter bumped
//! via `__sync_fetch_and_add` -> the `.bss` DATASEC), declared as a
//! `BpfMapAgg::ScalarCounter` so it folds to its final accumulated total, and
//! asserts on the HOST (`post_vm`) that the Dynamic run-level metric
//! `bpf_bpf_alloc_count` is PRESENT and a finite, positive count. The key is
//! the active scheduler's obj prefix plus the declared label (`alloc_count`):
//! scx-ktstr's kernel-visible global-section map is `bpf_bpf.bss`, so the obj
//! prefix the watcher resolves (via the same `find_active_struct_ops_obj` +
//! `extract_global_section_obj_prefix` path) is `bpf_bpf`. The sibling live
//! e2es `pin_bpf_map_e2e` / `live_var_disambiguation_e2e` match live maps by
//! that exact `bpf_bpf.bss` name; the booted-VM run here is the arbiter.
//!
//! Present (Some) is the load-bearing assertion: it proves the full
//! observer-effect-free host-read path ran against a real attached scheduler —
//! the free-running monitor lazily built a guest-memory map accessor AFTER the
//! scheduler attached, resolved the active-scheduler obj prefix, found the
//! `.bss` map, resolved the BTF dot-path + leaf width, read the value, and
//! folded it run-level. ABSENT FAILS loudly (`ok_or_else` -> Err), never a
//! vacuous SKIP — a `None` would mean the `.bss` global never resolved.
//!
//! The resolver dot-path/width logic and the scalar/per-CPU fold math are
//! pinned by the CI-runnable host unit tests
//! (`resolve_map_field_offset_width_*` in `btf_offsets`, `bpf_map_fields_*` in
//! the monitor summary tests); this booted-VM test is the integration half.

use anyhow::{Result, anyhow, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{BpfMapAgg, VmResult, WatchBpfMap};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Watch scx-ktstr's `.bss` allocation counter — a bare `__u64` monotonic
/// global, so the dot-path is the variable name itself (no struct descent) and
/// `BpfMapAgg::ScalarCounter` folds it to its final accumulated total (not the
/// mean of the rising series). The resulting metric key is
/// `<scheduler-obj>_<label>` = `bpf_bpf_alloc_count` (scx-ktstr's obj prefix is
/// `bpf_bpf`, from its kernel-visible `bpf_bpf.bss` global-section map).
const WATCH: &[&WatchBpfMap] =
    &[&WatchBpfMap::new(".bss", "ktstr_alloc_count", BpfMapAgg::ScalarCounter, "alloc_count")];

/// Host-side check that the watched `.bss` field surfaced as a run-level metric.
fn assert_watch(result: &VmResult) -> Result<()> {
    let v = result.run_metric("bpf_bpf_alloc_count").ok_or_else(|| {
        // Diagnostic: if the obj prefix is ever not `bpf_bpf`, surface what the
        // alternative `bpf` prefix would yield so a single boot is conclusive.
        let alt = result.run_metric("bpf_alloc_count");
        anyhow!(
            "bpf_bpf_alloc_count absent — the watch_bpf_maps observer-effect-free \
             host-read path did not resolve/read the scx-ktstr .bss global \
             (accessor build / obj-prefix / find_map / BTF dot-path / read). \
             alt prefix bpf_alloc_count = {alt:?}"
        )
    })?;
    // > 0, not just present: scx-ktstr's alloc path runs under scheduling, so a
    // resolved-and-read counter's final total is non-zero (a present-but-0 read
    // would signal a wrong offset/width). The sibling failure_dump_e2e relies
    // on the same non-zero-under-load behavior of this counter.
    ensure!(
        v.is_finite() && v > 0.0,
        "bpf_bpf_alloc_count must be a finite, positive count, got {v}"
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    watch_bpf_maps = WATCH,
    duration_s = 10,
    watchdog_timeout_s = 40,
    auto_repro = false,
    post_vm = assert_watch,
)]
fn watch_bpf_map_field_surfaces(ctx: &Ctx) -> Result<AssertResult> {
    // Two spin-loaded cgroups keep scx-ktstr scheduling across the hold so the
    // free-running monitor has post-attach samples to resolve + read the `.bss`
    // field from.
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0"), ctx.cgroup_def("cg_1")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
