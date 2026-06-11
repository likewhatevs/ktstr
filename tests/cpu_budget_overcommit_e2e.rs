//! Observe that `cpu_budget` overcommit produces guest-visible steal time.
//!
//! A `no_perf_mode` VM with `cpu_budget` < vCPUs masks every vCPU thread to
//! a smaller host-CPU pool (`set_thread_cpumask`). With each vCPU kept
//! runnable by a SpinWait worker, the host scheduler's time-sharing of the
//! oversubscribed vCPUs onto the budget's CPUs surfaces as steal time in the
//! guest's `/proc/stat` — the direct, guest-visible symptom of the overcommit.
//!
//! The topology uses 8 vCPUs, which is <= the host CPU count on the CI
//! runners, so the accrued steal is attributable to the `cpu_budget` mask
//! (2 host CPUs, 4x overcommit) rather than host-CPU-count oversubscription.
//! This makes the assertion a transitive check that the mask is actually
//! enforced: if `cpu_budget` did NOT limit the vCPU threads to the budget's
//! CPUs, 8 vCPUs <= host CPUs would run without contention and accrue no
//! steal. (On a host with fewer than 8 CPUs the steal is also host-count
//! driven, so the assertion still holds but no longer isolates the mask.)

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::WorkType;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};

/// Steal time (USER_HZ ticks) from the aggregate `cpu` line of
/// `/proc/stat`. Field layout after the `cpu` label: user nice system idle
/// iowait irq softirq STEAL guest guest_nice — so steal is the 8th
/// whitespace token counting the label as token 0.
fn read_steal_ticks() -> Result<u64> {
    let stat = std::fs::read_to_string("/proc/stat")?;
    let cpu = stat
        .lines()
        .next()
        .filter(|l| l.starts_with("cpu "))
        .ok_or_else(|| anyhow::anyhow!("/proc/stat missing aggregate `cpu ` line"))?;
    cpu.split_whitespace()
        .nth(8)
        .ok_or_else(|| anyhow::anyhow!("/proc/stat `cpu` line has no steal field: {cpu:?}"))?
        .parse::<u64>()
        .map_err(|e| anyhow::anyhow!("parse steal field from {cpu:?}: {e}"))
}

/// 8 vCPUs (2 LLCs x 4 cores x 1 thread) masked to `cpu_budget = 2` host
/// CPUs (4x overcommit). SpinWait on all 8 keeps them runnable; the steal
/// the guest accrues while the host time-shares them onto 2 CPUs is the
/// observable overcommit symptom — and, since 8 vCPUs <= host CPUs on CI,
/// evidence the `cpu_budget` mask is enforced.
#[ktstr_test(llcs = 2, cores = 4, threads = 1, no_perf_mode, cpu_budget = 2, duration_s = 5)]
fn cpu_budget_overcommit_accrues_guest_steal(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    ensure!(total == 8, "test assumes 8 vCPUs (2 x 4 x 1); got {total}");

    let before = read_steal_ticks()?;
    let steps = vec![Step {
        setup: vec![
            ctx.cgroup_def("load")
                .workers(8)
                .work_type(WorkType::SpinWait),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let result = execute_steps(ctx, steps)?;
    let after = read_steal_ticks()?;

    ensure!(
        after > before,
        "expected guest steal to accrue under 4x cpu_budget overcommit \
         (8 vCPUs masked to 2 host CPUs) but it did not advance: \
         before={before} after={after} ticks — either the cpu_budget mask \
         is not limiting the vCPU threads to the budget's host CPUs, or \
         guest KVM steal-time accounting (CONFIG_PARAVIRT_TIME_ACCOUNTING, \
         a required ktstr guest config) is unavailable"
    );
    Ok(result)
}
