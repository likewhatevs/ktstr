//! Reproducer: scx_mitosis aborts when a freed mid-range cell leaves a
//! sparse cell-id range its per-LLC drain cannot relocate.
//!
//! ## Mechanism
//!
//! In LLC-aware mode each cell has one DSQ per LLC. When a cell loses all
//! CPUs in an LLC, `recalc_cell_llc_counts` drains that orphaned DSQ by
//! moving its tasks to another LLC the cell still owns. The destination
//! search picks the first LLC with a surviving CPU; if the cell owns NO
//! LLC with CPUs, the search yields none. The buggy version then calls
//! `scx_bpf_error("cell N has no CPUs in any LLC; cannot drain")`, which
//! aborts the whole scheduler. The fixed version returns from the
//! empty-cell drain cleanly.
//!
//! This is reached deterministically via a sparse cell-id range. With
//! `--cell-parent-cgroup /mitosis`, mitosis allocates a cell id to every
//! cgroup created under `/mitosis` (in inotify event order). When a
//! mid-range cell is freed, mitosis zeroes that config slot (empty
//! cpumask) but the slot's previous `cpu_cnt` was `> 0`, so the next
//! `apply_cell_config` processes a `>0 -> 0` transition for a cell that
//! now owns no LLC — the drain's destination search returns invalid and
//! the abort fires on that apply (within about one monitor interval of the
//! middle cgroup being freed).
//!
//! ## What this test does
//!
//! A churner thread repeatedly creates a trio of ephemeral cgroups under
//! the cell-parent, lets mitosis allocate them cell ids, then removes the
//! MIDDLE one. The freed cell sits between the trio's first and last cells
//! in cell-id space, so its zeroed config slot is a sparse middle still
//! covered by a higher surviving cell (the trio's last) — exactly the
//! configuration that drives the drain into the empty-cell abort. The gap
//! is intrinsic to freeing the middle of the trio and does not depend on
//! any other cell's id. `cg_a` carries a steady workload so the scheduler
//! is active. The ephemerals are kept task-free: a task in the freed cell
//! would make even the fixed scheduler abort (its next enqueue hits
//! `pick_llc_for_task` on a cell with no LLC), muddying the discriminator.
//!
//! ## Verdict
//!
//! A `post_vm_unconditional` callback (runs even on the guest-fail/crash path)
//! matches the scheduler's `scx_bpf_error` reason in `result.stderr`. With
//! `loglevel=7` (scheduler kargs) the guest kernel routes that printk to COM1,
//! which the harness captures as `result.stderr`. The buggy scheduler aborts
//! with "no CPUs in any LLC; cannot drain" → the callback fails the test with
//! that attribution (RED). The fixed scheduler runs the whole duration without
//! it → GREEN. A crash with a DIFFERENT reason is still a guest-fail (RED) but
//! is not mis-attributed to this bug.
//!
//! ## Running
//!
//! Gated behind the `integration` feature (needs an external scx_mitosis
//! binary + a sched_ext kernel). `--features integration` is a TRAILING
//! arg to `ktstr test` (forwarded to the inner `cargo nextest run`); `-E`
//! is a nextest top-level flag so it goes before any `--`:
//!
//! ```text
//! KTSTR_SCHEDULER=/path/to/scx_mitosis \
//!   cargo run --bin cargo-ktstr -- \
//!   ktstr test --kernel ../linux --features integration --profile ci \
//!   -E 'test(mitosis_cell_id_gap_drain_abort)'
//! ```

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::backdrop::Backdrop;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_scenario};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::{WorkSpec, WorkType};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// scx_mitosis in LLC-aware mode. The drain abort fires whenever
/// `--enable-llc-awareness` is set (work-stealing / borrowing / rebalancing
/// are not required, so the arg set is minimal). `--cpu-controller-disabled`
/// keeps cell sizing cpuset-driven. `--cell-parent-cgroup /mitosis` is passed
/// EXPLICITLY: the harness does NOT inject it from the workload cgroup path, and
/// without it mitosis watches its default cell-parent and never sees the
/// churner's cgroups (no cells, no gap, no abort). Binary via `KTSTR_SCHEDULER`.
const MITOSIS: Scheduler = Scheduler::named("scx_mitosis")
    .binary(SchedulerSpec::Discover("scx_mitosis"))
    .kargs(&["loglevel=7"])
    .sched_args(&[
        "--cpu-controller-disabled",
        "--enable-llc-awareness",
        "--cell-parent-cgroup",
        "/mitosis",
        "--monitor-interval-s",
        "1",
        "--exit-dump-len",
        "4194304",
    ]);

/// Cell-parent cgroup path in the guest — matches the `--cell-parent-cgroup`
/// sched_arg above (where mitosis watches for cells) and `workload_root_cgroup`.
const CELL_PARENT: &str = "/sys/fs/cgroup/mitosis";

#[ktstr_test(
    scheduler = MITOSIS,
    llcs = 4,
    cores = 2,
    threads = 1,
    no_perf_mode,
    workload_root_cgroup = "/mitosis",
    duration_s = 10,
    auto_repro = false,
    post_vm_unconditional = assert_cannot_drain_abort,
)]
fn mitosis_cell_id_gap_drain_abort(ctx: &Ctx) -> Result<AssertResult> {
    // A Backdrop cell carrying a steady workload so the scheduler is active
    // during the churn. The sparse cell-id gap does NOT depend on cg_a: it is
    // intrinsic to freeing the MIDDLE of the churner's trio (the trio's last
    // cell survives with a higher id, so num_cells still covers the freed
    // middle slot).
    let backdrop = Backdrop::new().push_cgroup(
        CgroupDef::named("cg_a").work(WorkSpec::default().workers(2).work_type(WorkType::SpinWait)),
    );

    // One whole-duration hold; all cell churn is driven by the churner below
    // (execute_scenario requires at least one Step).
    let steps = vec![Step::new(vec![], HoldSpec::Frac(1.0))];

    // Churner: each cycle creates a trio of bare ephemeral cells (mitosis
    // floors every cell to >=1 CPU, so each gets cpu_cnt>0 without an explicit
    // cpuset — which can't be written here, the controller is not delegated to
    // raw-mkdir cgroups), holds longer than the monitor interval so a config
    // with the trio PRESENT reaches BPF (the middle cell's cpu_cnt is pushed
    // >0), then frees ONLY the MIDDLE and holds again so a config where the
    // middle is a zeroed gap reaches BPF. On that gap config the buggy recalc
    // drain finds the middle owns no surviving LLC and aborts ("no CPUs in any
    // LLC; cannot drain"); the fixed scheduler drains it cleanly. The outer two
    // are removed only after the gap config is applied, and no cgroup is created
    // during the gap window, so the freed middle id is not reused (reuse would
    // repopulate the slot and erase the gap).
    let stop = Arc::new(AtomicBool::new(false));
    let stop_c = stop.clone();
    let churner = std::thread::spawn(move || {
        // Hold > --monitor-interval-s (1s) so each config push reaches BPF.
        let settle = Duration::from_millis(1600);
        let mut i: u64 = 0;
        while !stop_c.load(Ordering::Relaxed) {
            for j in 0..3 {
                let _ = std::fs::create_dir(format!("{CELL_PARENT}/ephemeral_{i}_{j}"));
            }
            std::thread::sleep(settle); // trio present, middle at cpu_cnt>0
            let _ = std::fs::remove_dir(format!("{CELL_PARENT}/ephemeral_{i}_1"));
            std::thread::sleep(settle); // middle is a zeroed gap -> 1.1.0 aborts
            let _ = std::fs::remove_dir(format!("{CELL_PARENT}/ephemeral_{i}_0"));
            let _ = std::fs::remove_dir(format!("{CELL_PARENT}/ephemeral_{i}_2"));
            std::thread::sleep(settle); // settle before next trio (no id reuse)
            i = i.wrapping_add(1);
        }
    });

    // The verdict is the post_vm_unconditional callback
    // (assert_cannot_drain_abort): it matches result.stderr for the cannot-drain
    // reason even on the crash path (the body does not run past execute_scenario
    // once the scheduler aborts). For the fixed scheduler the body returns Ok
    // (no crash, runs to completion) -> GREEN.
    let result = execute_scenario(ctx, backdrop, steps);

    stop.store(true, Ordering::Relaxed);
    let _ = churner.join();
    result
}

/// Host-side verdict, run even on the guest-fail/crash path (unlike `post_vm`):
/// attribute the failure to the cannot-drain abort specifically. With
/// `loglevel=7` the guest kernel routes the `scx_bpf_error` printk to COM1,
/// which the harness captures as `result.stderr`. The buggy scheduler aborts
/// with the cannot-drain reason → FAIL (RED) with attribution; a crash with a
/// DIFFERENT reason stays a generic guest-fail (not mis-attributed); a clean
/// run is GREEN.
fn assert_cannot_drain_abort(result: &VmResult) -> Result<()> {
    // With `loglevel=7` (scheduler kargs) the guest kernel routes the
    // `scx_bpf_error` printk to COM1, captured as `result.stderr`. Match the
    // stable prefix of the reason (the "(cell, llc=N)" suffix varies).
    if result
        .stderr
        .contains("no CPUs in any LLC; cannot drain")
    {
        anyhow::bail!(
            "scx_mitosis drain-abort bug present: recalc_cell_llc_counts aborted \
             (cell has no CPUs in any LLC; cannot drain)"
        );
    }
    Ok(())
}
