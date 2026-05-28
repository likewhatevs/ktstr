#![cfg(feature = "wprof")]
//! End-to-end: `#[ktstr_test(scheduler, wprof, expect_auto_repro)]`
//! produces EXIT_PASS via the verdict-inversion path when the primary
//! VM's test body fails AND the auto-repro VM lands a shape-valid
//! `.repro.wprof.pb` artifact.
//!
//! Unlike the `neg_wprof_auto_repro_fires_on_forced_fail` test (in
//! `tests/wprof_auto_repro_e2e.rs`) which IS expected to fail from
//! nextest's perspective (the `neg_` prefix advertises that), this
//! test is POSITIVE: it MUST pass from nextest's perspective.
//! Failure here means the inversion chain regressed at one of the
//! load-bearing sites:
//!
//! 1. `apply_expect_auto_repro_inversion` did NOT set
//!    `VmResult.expect_auto_repro_satisfied = true` (eval.rs).
//!    Likely cause: artifact missing (auto-repro VM crashed),
//!    artifact truncated (mid-write), or path-resolve bail
//!    (entry_name=None on the VmResult).
//! 2. The eval layer did NOT wrap the failure `Err` with the
//!    `ExpectAutoReproSatisfied` marker (eval.rs:
//!    `run_ktstr_test_inner_impl`'s context-wrap after the helper).
//! 3. The `result_to_exit_code` dispatch arm did NOT downcast the
//!    marker correctly (dispatch.rs — note: `anyhow::Error::downcast_ref::<C>()`
//!    is the context-aware idiom; `chain().any(|c| c.is::<C>())`
//!    DOES NOT match context-attached markers and was the bug
//!    fixed in an earlier commit).
//!
//! ## Why this works at all
//!
//! The framework treats `record_fail` as a failed test outcome:
//! `evaluate_vm_result` sees `!check_result.is_pass()` and returns
//! `Err`. The guest's `exit_code_for_result` maps the non-pass
//! result to exit code 1, so the host's `VmResult.success` is
//! false. The auto-repro path fires (per the effective_auto_repro
//! gate: `auto_repro && scheduler.is_some() && !expect_err`) and
//! the auto-repro VM produces `.repro.wprof.pb` via the wprof
//! binary. The host's `MsgType::WprofTrace` arm writes the
//! artifact to the sidecar dir. `apply_expect_auto_repro_inversion`
//! then probes the artifact via `assert_wprof_pb_shape`, sets the
//! satisfaction field, and the eval layer wraps the failure `Err`
//! with the `ExpectAutoReproSatisfied` marker. Dispatch sees the
//! marker, prints the inversion notice, and routes the verdict to
//! `EXIT_PASS`. nextest sees exit 0 → PASS.
//!
//! ## Why the body cannot self-verify
//!
//! The test body runs INSIDE the primary VM. The auto-repro VM
//! boots AFTER the primary VM exits; the inversion check happens
//! on the HOST after the auto-repro VM produces (or fails to
//! produce) the artifact. The body has no visibility into either
//! the auto-repro VM's lifecycle or the host's inversion logic.
//! The dispatch flip is the verification mechanism: a regression
//! that breaks the chain at ANY of the 3 sites above causes
//! nextest to see EXIT_FAIL → the test FAILS, surfacing the
//! regression at the operator's CI gate without requiring
//! manual artifact inspection.
//!
//! Runs on the self-hosted CI runners (`[ktstr-x64]` /
//! `[ktstr-arm64]`). ktstr supplies the guest kernel itself via
//! its kernel-build cache. The dispatch + helper logic is also
//! exercised
//! at unit-test granularity in `src/test_support/eval.rs::tests`
//! (apply_expect_auto_repro_inversion) and
//! `src/test_support/dispatch.rs::tests` (verdict matrix); this
//! e2e is the integration-layer pin that catches a regression at
//! a hop unit tests cannot reach (the actual VM-to-VM artifact
//! transport).

mod common;

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec, WorkType};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 3,
    watchdog_timeout_s = 60,
    wprof,
    auto_repro = true,
    expect_auto_repro,
)]
fn pos_expect_auto_repro_satisfied_on_forced_fail(ctx: &Ctx) -> Result<AssertResult> {
    // Minimal SpinWait workload — gives wprof's sched tracer
    // something to capture on both the primary VM and the
    // auto-repro VM, so the `.repro.wprof.pb` artifact is
    // non-degenerate and clears `assert_wprof_pb_shape`'s size +
    // tag gates.
    let steps = vec![Step {
        setup: vec![
            ctx.cgroup_def("wl")
                .workers(1)
                .work_type(WorkType::SpinWait),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    // Force a failure to drive the auto-repro chain. Unlike the
    // `neg_` sibling, this test EXPECTS the resulting verdict to
    // flip to PASS via the expect_auto_repro inversion path. If
    // nextest reports this test FAILED, the inversion broke at one
    // of the three sites named in the file header.
    let expected_repro_pb = ctx.repro_wprof_pb_path()?;
    result.record_fail(AssertDetail::new(
        DetailKind::Other,
        format!(
            "forced fail to drive auto-repro chain — verdict MUST \
             invert to PASS via expect_auto_repro. The body-recorded \
             fail is intentional, and a passing verdict from nextest \
             is the contract being verified. \
             A FAIL from nextest means one of: \
             (a) `.repro.wprof.pb` did not land at {repro} \
             (auto-repro VM crashed or wprof's WprofTrace transport \
             broke), \
             (b) `apply_expect_auto_repro_inversion` failed to set \
             `VmResult.expect_auto_repro_satisfied = true` despite \
             a valid artifact (eval.rs gate regression), or \
             (c) the `result_to_exit_code` dispatch arm did not \
             downcast the `ExpectAutoReproSatisfied` marker \
             (dispatch.rs — switched from `chain().any(c.is)` to \
             `anyhow::Error::downcast_ref` to handle anyhow's \
             ContextError boxing).",
            repro = expected_repro_pb.display(),
        ),
    ));

    Ok(result)
}
