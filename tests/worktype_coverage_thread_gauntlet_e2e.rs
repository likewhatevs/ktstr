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
//! The CgroupChurn arm relies on the worker auto-creating its rotation
//! cgroups (`<workload_root>/wt-cgroup-churn-<i>` for `i in 0..groups`,
//! default root `/sys/fs/cgroup/ktstr`) at entry, then asserts those
//! cgroups exist post-run so the `cgroup.procs` write success path is
//! actually covered (not the open-failure fallback that liveness alone
//! would mask). The guest runs as root, so the cgroup creation has the
//! required privilege.

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
    // tid write to cgroup.procs migrates the whole tgid. The worker now
    // auto-creates the rotation cgroups under the workload root at entry,
    // so no hand-setup is needed. The guest runs as root.
    let groups = 2usize;

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

    // Pin the auto-provisioning: the worker must have created each
    // rotation cgroup under the resolved workload root (default
    // /sys/fs/cgroup/ktstr), so the cgroup.procs writes hit the success
    // path rather than the open-failure fallback that the liveness check
    // above would otherwise mask. The cgroups persist post-run (workers
    // exit; no remove), and this test body runs in the same guest as the
    // forked workers, so it observes the dirs they created.
    // This literal mirrors `churn_cgroup_name(i)` and the default
    // `resolve_cgroup_root` in worker/mod.rs (a `tests/` crate cannot
    // import those `pub(crate)`/private helpers). A name-format change
    // there makes this path miss, so the assertion fails loudly.
    for i in 0..groups {
        let dir = format!("/sys/fs/cgroup/ktstr/wt-cgroup-churn-{i}");
        if !std::path::Path::new(&dir).is_dir() {
            result.record_fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "CgroupChurn: auto-created cgroup {dir} is missing — the \
                     worker did not provision it under the workload root"
                ),
            ));
        }
    }

    Ok(result)
}
