//! End-to-end coverage for per-CPU BPF map projection:
//! [`BpfMapProjector::field_cpu_sum_u64`] (cross-CPU aggregate) and
//! [`BpfMapProjector::cpu`] → [`BpfMapCpuProjector::field_u64`]
//! (per-CPU select), exercised against a REAL `BPF_MAP_TYPE_PERCPU_ARRAY`
//! (scx-ktstr's `ktstr_pcpu_fix`) captured from a booted scheduler.
//!
//! No prior test captured a real per-CPU map (scx-ktstr had none — see
//! its `ktstr_dump_cpu` "no per-cpu state" line), so this validates the
//! whole per-CPU capture → render → projector chain end-to-end, not
//! just the host-side projector logic the bpf.rs chain tests cover with
//! synthetic data.
//!
//! scx-ktstr stamps each dispatching CPU's own slot with
//! `KTSTR_PERCPU_MAGIC + cpu_id` from `ktstr_dispatch` (via
//! `bpf_map_lookup_elem`, which returns the running CPU's slot). After a
//! workload across a 2-core VM:
//!   * the per-CPU SUM of `magic` peaks at >= MAGIC in some capture
//!     (at least one CPU dispatched and stamped its slot), and
//!   * per-CPU SELECT reads back `MAGIC + n` for a CPU `n` that ran.
//!
//! Vacuity: it does NOT pin WHICH CPUs ran or the exact sum (the
//! scheduler picks placement); it pins that a real per-CPU map's
//! captured slots flow through both projector paths with the
//! BPF-stamped values.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Mirrors `KTSTR_PERCPU_MAGIC` in `scx-ktstr/src/bpf/main.bpf.c`.
const KTSTR_PERCPU_MAGIC: u64 = 0x5A5A_0000_0000;

/// VM core count (matches the `cores` attr below). CPUs `0..CORES` may
/// have dispatched and stamped `ktstr_pcpu_fix`.
const CORES: usize = 2;

fn assert_percpu_projection(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "periodic_fired = {} of {} — no capture fired; per-CPU projection \
         cannot be exercised without at least one periodic sample",
        result.periodic_fired,
        result.periodic_target,
    );
    let series = result.periodic_series();

    // Cross-CPU SUM: at least one CPU dispatched and stamped its slot
    // with MAGIC + its id, so the per-CPU sum of `magic` peaks at
    // >= MAGIC in at least one captured sample. Reading a value at all
    // proves the PERCPU_ARRAY was captured + rendered + projected
    // end-to-end (the bpf.rs chain tests cover the reduction logic with
    // synthetic data; this proves the real capture path feeds it).
    let sum_samples: Vec<u64> = series
        .bpf_map("ktstr_pcpu_fix")
        .at(0)
        .field_cpu_sum_u64("magic")
        .values_iter()
        .filter_map(|r| r.as_ref().ok().copied())
        .collect();
    anyhow::ensure!(
        !sum_samples.is_empty(),
        "field_cpu_sum_u64 produced no readable sample for ktstr_pcpu_fix — \
         the per-CPU map was not captured/projected end-to-end (map missing \
         from the snapshot, or every per-CPU slot unreadable)",
    );
    let peak_sum = sum_samples.iter().copied().max().unwrap_or(0);
    anyhow::ensure!(
        peak_sum >= KTSTR_PERCPU_MAGIC,
        "per-CPU sum of `magic` peaked at {peak_sum}, expected >= {KTSTR_PERCPU_MAGIC} \
         — at least one CPU should have dispatched and stamped MAGIC + its id; \
         a peak below MAGIC means no CPU's stamp was captured",
    );

    // Per-CPU SELECT: a CPU that dispatched reads back MAGIC + its id
    // via .cpu(n).field_u64. At least one CPU in 0..CORES must have run
    // the workload, so at least one select resolves the stamped value.
    let mut selected = None;
    for n in 0..CORES {
        let vals: Vec<u64> = series
            .bpf_map("ktstr_pcpu_fix")
            .cpu(n)
            .field_u64("magic")
            .values_iter()
            .filter_map(|r| r.as_ref().ok().copied())
            .collect();
        if vals.iter().any(|&v| v == KTSTR_PERCPU_MAGIC + n as u64) {
            selected = Some(n);
            break;
        }
    }
    anyhow::ensure!(
        selected.is_some(),
        "no CPU in 0..{CORES} read back MAGIC + its id via \
         .cpu(n).field_u64(\"magic\") — per-CPU select did not resolve a \
         populated slot (the select narrow is not reading the stamped \
         per-CPU value)",
    );

    Ok(())
}

/// Boots scx-ktstr on a 2-core VM with a multi-worker workload so both
/// CPUs dispatch (stamping `ktstr_pcpu_fix`), with periodic captures so
/// the host snapshots the populated per-CPU map.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 10,
    watchdog_timeout_s = 20,
    num_snapshots = 4,
    auto_repro = false,
    post_vm = assert_percpu_projection,
)]
fn percpu_projection_reads_real_percpu_array_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_pcpu").workers(4)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
