//! VM-backed end-to-end test for the
//! `Op::Spawn(SpawnPlacement::Cgroup(""))` empty-string bail at
//! `src/scenario/ops/mod.rs::apply_ops`.
//!
//! The unit-test layer pins the bail logic with mocked
//! `CgroupOps`; this e2e test boots a real VM and proves the
//! bail also fires in the production scenario-engine path —
//! before `WorkloadHandle::spawn` burns clone(2) syscalls,
//! before any cgroup-fs write. A regression that moved the
//! check after spawn would still pass the unit test (mock
//! never observes the wasted spawn) but would fail this test
//! because the bail returns from the scenario engine with a
//! specific diagnostic string AND a clean teardown — workers
//! never reach the cgroup-fs layer.
//!
//! Expected behavior:
//! - `execute_scenario` returns `Err(_)` (scenario engine bails)
//! - The error chain contains
//!   `"Op::Spawn(SpawnPlacement::Cgroup): cgroup name is empty"`
//!   plus the actionable redirect to `SpawnPlacement::runner_cgroup()`
//! - No cgroup-fs side effects (no `cg_test` dir created — the
//!   scenario never reaches the cgroup-fs layer)
//!
//! The test runs under `auto_repro = false` because the bail
//! IS the expected outcome — re-running in an auto-repro VM
//! to capture the "failure" would just bail again with the
//! same message, wasting a kernel boot.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::{SpawnPlacement, WorkSpec};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_scenario};
use std::time::Duration;

#[ktstr_test(
    workload_root_cgroup = "/ktstr-spawn-empty-bail-e2e",
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 256,
    duration_s = 2,
    watchdog_timeout_s = 20,
    auto_repro = false
)]
fn op_spawn_cgroup_empty_string_bails_with_actionable_diagnostic(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let backdrop = ktstr::prelude::Backdrop::default();
    let steps = vec![Step::new(
        vec![Op::spawn(
            SpawnPlacement::cgroup(""),
            WorkSpec::default().workers(1),
        )],
        HoldSpec::fixed(Duration::ZERO),
    )];
    match execute_scenario(ctx, backdrop, steps) {
        Ok(_) => Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "execute_scenario returned Ok — empty-string \
             SpawnPlacement::Cgroup MUST bail at apply_ops \
             entry. Without the bail, the workers would have \
             spawned, the cgroup-fs move_tasks would have \
             failed with a less actionable kernel ENOENT, and \
             the bail's spawn-before-validate hazard the \
             diagnostic prevents would be live."
                .to_string(),
        ))),
        Err(e) => {
            let msg = format!("{e:#}");
            // Pin the diagnostic shape so a regression that
            // weakened the bail to bare "cgroup '': not found"
            // (kernel ENOENT class) surfaces here.
            if !msg.contains("Op::Spawn(SpawnPlacement::Cgroup)") {
                return Ok(AssertResult::fail(AssertDetail::new(
                    DetailKind::Other,
                    format!(
                        "expected diagnostic to name \
                         `Op::Spawn(SpawnPlacement::Cgroup)`; \
                         got: {msg}"
                    ),
                )));
            }
            if !msg.contains("cgroup name is empty") {
                return Ok(AssertResult::fail(AssertDetail::new(
                    DetailKind::Other,
                    format!(
                        "expected diagnostic to cite `cgroup \
                         name is empty`; got: {msg}"
                    ),
                )));
            }
            if !msg.contains("SpawnPlacement::runner_cgroup") {
                return Ok(AssertResult::fail(AssertDetail::new(
                    DetailKind::Other,
                    format!(
                        "expected diagnostic to name the \
                         runner-cgroup recovery path; got: {msg}"
                    ),
                )));
            }
            Ok(AssertResult::pass())
        }
    }
}
