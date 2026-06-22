//! Clone-mode-specific WorkType coverage gauntlet.
//!
//! Boots one small VM with no sched_ext scheduler attached (EEVDF is
//! the in-kernel default) and drives the two [`WorkType`] variants whose
//! worker bodies the Fork gauntlet
//! (`tests/worktype_coverage_fork_gauntlet_e2e.rs`) cannot reach,
//! because each is admissible under exactly one [`CloneMode`]:
//!
//!   - [`WorkType::EpollStorm`] — Thread-only. `WorkloadHandle::spawn`
//!     rejects `CloneMode::Fork` for it: the variant publishes the
//!     eventfd/epoll fd numbers through a shared mmap for siblings to
//!     consume, but forked children hold independent fd tables that
//!     never contain those post-fork descriptors. Run under
//!     `CloneMode::Thread` (shared fd table).
//!   - [`WorkType::CgroupChurn`] — Fork-only. `WorkloadHandle::spawn`
//!     rejects `CloneMode::Thread` for it: the worker writes its tid to
//!     a sibling `cgroup.procs`, which the kernel resolves to the whole
//!     tgid and would migrate every sibling thread (including the
//!     harness). Run under `CloneMode::Fork` (default) so each worker
//!     is its own tgid.
//!
//! Both arms assert liveness: spawn succeeds, at least one report comes
//! back, and the workers recorded non-zero `work_units + iterations`.
//! The CgroupChurn arm pre-creates the sibling cgroup directories its
//! worker rotates through (`/sys/fs/cgroup/wt-cgroup-churn-<i>` for
//! `i in 0..groups`) so the `cgroup.procs` write succeeds and the
//! success branch of the dispatch arm is covered. The guest runs as
//! root, so the directory creation has the required privilege.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::workload::{CloneMode, WorkType, WorkloadConfig, WorkloadHandle};
use std::time::Duration;

/// Spawn `cfg`, run it briefly, collect reports, and record liveness
/// failures into `result`: spawn succeeds, at least one report comes
/// back, and the workers recorded non-zero `work_units + iterations`.
fn run_arm(label: &str, cfg: WorkloadConfig, result: &mut AssertResult) {
    let mut handle = match WorkloadHandle::spawn(&cfg) {
        Ok(h) => h,
        Err(e) => {
            result.record_fail(AssertDetail::new(
                DetailKind::Other,
                format!("{label}: spawn failed: {e:#}"),
            ));
            return;
        }
    };
    handle.start();
    std::thread::sleep(Duration::from_millis(500));
    let reports = handle.stop_and_collect();

    if reports.is_empty() {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!("{label}: zero reports — spawn or collection broken"),
        ));
        return;
    }

    let units: u64 = reports.iter().map(|r| r.work_units + r.iterations).sum();
    if units == 0 {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "{label}: no work_units/iterations across {} workers — \
                 the dispatch arm produced no measurable work",
                reports.len(),
            ),
        ));
    }
}

#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 1024,
    max_spread_pct = 80.0,
    duration_s = 15,
    watchdog_timeout_s = 60
)]
fn worktype_clone_mode_gauntlet_covers_epoll_and_cgroup(_ctx: &Ctx) -> Result<AssertResult> {
    let mut result = AssertResult::pass();

    // EpollStorm is Thread-only: Fork-mode spawn is rejected because
    // the eventfd/epoll fds are created post-fork on worker 0 and
    // published through shared mmap, which only resolves under a shared
    // fd table (Thread mode).
    run_arm(
        "EpollStorm",
        WorkloadConfig {
            num_workers: 4,
            work_type: WorkType::EpollStorm {
                producers: 2,
                consumers: 2,
                events_per_burst: 8,
            },
            clone_mode: CloneMode::Thread,
            ..Default::default()
        },
        &mut result,
    );

    // CgroupChurn is Fork-only: Thread-mode spawn is rejected because a
    // tid write to cgroup.procs migrates the whole tgid. Pre-create the
    // sibling cgroups the worker rotates through (groups = 2, so
    // wt-cgroup-churn-0 and wt-cgroup-churn-1) so the cgroup.procs write
    // hits its success path. The guest runs as root.
    let groups = 2usize;
    for i in 0..groups {
        let dir = format!("/sys/fs/cgroup/wt-cgroup-churn-{i}");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            result.note(format!(
                "CgroupChurn: create_dir_all({dir}) failed: {e}; the worker \
                 falls back to the open-failure path (still bumps \
                 iterations/work_units), so liveness is asserted regardless"
            ));
        }
    }

    // Fork is the default clone mode; omit clone_mode so CgroupChurn
    // runs as Fork (each worker its own tgid).
    run_arm(
        "CgroupChurn",
        WorkloadConfig {
            num_workers: 2,
            work_type: WorkType::CgroupChurn {
                groups,
                cycle_ms: 10,
            },
            ..Default::default()
        },
        &mut result,
    );

    Ok(result)
}
