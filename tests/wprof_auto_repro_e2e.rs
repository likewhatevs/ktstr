//! End-to-end: `#[ktstr_test(wprof, auto_repro = true)]` fires
//! the auto-repro path when the primary VM's scenario reports a
//! failure.
//!
//! Distinct from `tests/wprof_artifact_e2e.rs` (which pins the
//! happy-path primary-VM wprof chain on a PASSING test) — this
//! test pins the FAILURE-PATH wprof chain:
//!
//!   1. Primary VM runs the scenario; wprof attaches and produces
//!      `{sidecar_dir}/{test_name}.wprof.pb`.
//!   2. The scenario records a deliberate fail via
//!      `AssertResult::record_fail` (no scheduler death, no
//!      stall, just an explicit assertion-mismatch marker).
//!   3. `evaluate_vm_result` (src/test_support/eval.rs) sees the
//!      failed AssertResult, runs the auto-repro branch IF
//!      `entry.auto_repro && scheduler.is_some() && !entry.expect_err`
//!      (eval.rs L1787 `effective_auto_repro` gate).
//!   4. Auto-repro boots a second VM under the same scheduler,
//!      captures its wprof trace, and writes
//!      `{sidecar_dir}/{test_name}.repro.wprof.pb` per
//!      `src/test_support/probe.rs::write_auto_repro_sidecar_artifacts`
//!      (L498-516).
//!   5. Both `.wprof.pb` artifacts persist in the sidecar dir
//!      after the test reports its (intentional) failure.
//!
//! ## Why this test is expected to FAIL when run
//!
//! Auto-repro only fires when the primary VM's scenario fails.
//! This test forces a synthetic failure to drive the auto-repro
//! chain end-to-end without requiring a real scheduler crash.
//! It does NOT use the `expect_auto_repro` macro attribute
//! (which converts (fail + artifacts) → PASS via dispatch-layer
//! verdict inversion — see `tests/expect_auto_repro_e2e.rs` for
//! the canonical positive test of that path). Instead, this test
//! runs in "fault-injection / operator-verifies-artifacts" mode:
//!
//!   * `#[ignore]` keeps the test out of every default test
//!     run. CI MUST NEVER enable an `--run-ignored only` pass
//!     against this file — the fail is the contract, not a
//!     regression.
//!   * An operator who wants to validate the auto-repro path
//!     runs the test via the project's canonical
//!     `cargo ktstr test ... -- --run-ignored only -E
//!     'test(/neg_wprof_auto_repro/)'` invocation (or the
//!     equivalent nextest selector for the operator's host),
//!     observes the intentional fail, then verifies
//!     `{sidecar_dir}/{test_name}.repro.wprof.pb` exists on the
//!     HOST filesystem (the test body itself cannot read the
//!     repro artifact — see below).
//!
//! ## Audience
//!
//! This is the framework's own operator-verifiable probe of the
//! auto-repro chain — exercised when an operator needs to
//! inspect the raw `.repro.wprof.pb` artifact directly.
//! Scheduler-author end users validating their own scheduler's
//! auto-repro behavior should use `expect_auto_repro = true`
//! (see `tests/expect_auto_repro_e2e.rs`) which gives them a
//! clean PASSING test without the manual artifact-inspection
//! step.
//!
//! ## Why the body cannot self-verify the repro artifact
//!
//! Two independent reasons:
//!
//!   1. **Timing**: the `.repro.wprof.pb` is written by
//!      `attempt_auto_repro` AFTER the test body returns and
//!      the primary VM exits. No amount of waiting inside the
//!      body would observe it.
//!   2. **Isolation**: the `#[ktstr_test]` scenario function
//!      runs INSIDE the guest VM (see
//!      `src/test_support/eval.rs` L1500-1507). Per
//!      `src/vmm/rust_init.rs` L2358-2371 the guest mount
//!      table includes only procfs/sysfs/devtmpfs/cgroup2/
//!      tmpfs — there is no virtio-fs mount of the host
//!      sidecar directory. `std::fs::read(sidecar_dir().join(
//!      "..."))` from the body would resolve to a host path
//!      the guest cannot open, returning ENOENT regardless of
//!      whether the host-side write succeeded.
//!
//! The `expect_auto_repro` primitive solves the (fail +
//! artifacts) → PASS reporting problem but does NOT solve
//! self-verification — it inverts the verdict at the dispatch
//! layer after the primary VM exits, not from within the test
//! body. Adding a polling loop OR a wait inside the body would
//! solve neither timing nor isolation.
//!
//! ## Shell payload provisioning + gating
//!
//! `#[ignore]` per the project's "gating-prereqs-only"
//! convention for `#[ignore]` (`#[ignore]` reserves the marker
//! for tests gated on VM/external prereqs; matches
//! `tests/wprof_artifact_e2e.rs`,
//! `tests/wprof_args_override_e2e.rs`, and
//! `tests/failure_dump_e2e.rs`). A real KVM VM boot is
//! required.
//!
//! Hang protection: `watchdog_timeout_s = 60` caps the
//! per-test runtime so a regression that wedges auto-repro
//! (e.g. attempt_auto_repro deadlock OR the second-VM boot
//! loop) surfaces as a watchdog timeout rather than an
//! indefinite hang. An operator seeing a watchdog timeout
//! (instead of the documented fault-injection fail) should
//! treat that as an auto-repro chain regression.

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
    ignore,
)]
fn neg_wprof_auto_repro_fires_on_forced_fail(ctx: &Ctx) -> Result<AssertResult> {
    // Minimal SpinWait workload — gives wprof's sched tracer
    // something to capture on the primary VM, so a future
    // host-side post_vm hook (when added per the project-wide
    // shape-check fix) has a non-degenerate trace to assert on.
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

    // Force a failure to trigger the auto-repro path.
    // `AssertResult::record_fail` appends to the result without
    // bailing — control returns to the framework, which sees
    // the failed result and (per `evaluate_vm_result`'s
    // effective_auto_repro gate: `auto_repro &&
    // scheduler.is_some() && !expect_err`) invokes
    // `attempt_auto_repro`. The auto-repro VM boot is what
    // produces the `.repro.wprof.pb` sidecar artifact.
    //
    // The expected repro path is computed for the operator
    // diagnostic only — the test body cannot read it (see
    // "Why the body cannot self-verify" in the file header).
    // The path string is rendered in the fail message below
    // so the operator finds it without grepping sources.
    let expected_repro_pb = ctx.repro_wprof_pb_path()?;
    result.record_fail(AssertDetail::new(
        DetailKind::Other,
        format!(
            "forced fail to trigger auto-repro chain. This test is \
             a fault-injection probe gated `#[ignore]`: the fail \
             itself is the contract, NOT a regression. \n\n\
             Operator verification (HOST-side, after this test \
             reports failed):\n  \
             0. Before running this test, delete any stale repro \
             artifact: `rm -f {repro}` — the sidecar dir is \
             `(kernel, project_commit)`-keyed and project_commit \
             falls back to `unknown` in dirty worktrees, so a \
             prior run's `.repro.wprof.pb` from the same \
             dirty-worktree session would otherwise pass a \
             post-run `ls` even if the current run's auto-repro \
             chain regressed.\n  \
             1. Confirm the file exists with a fresh mtime: \
             `stat -c '%Y %n' {repro}` — the mtime must be ≥ \
             the wall-clock time the current test run started \
             (an mtime older than that means the file is stale \
             from a prior run).\n  \
             2. Confirm shape via the project's \
             `assert_wprof_pb_shape` semantics in \
             `ktstr::test_support::wprof::assert_wprof_pb_shape` \
             (named consts WPROF_PB_MIN_BYTES + \
             PERFETTO_TRACE_PACKETS_TAG carry the floor + Perfetto \
             tag rationale — defer to the helper rather than baking \
             the values into this message so a floor bump doesn't \
             silently drift).\n  \
             3. If you ALSO see a watchdog-timeout failure instead \
             of this documented forced-fail, the auto-repro chain \
             hung — likely deadlock in attempt_auto_repro or a \
             second-VM boot loop.\n\n\
             A missing or invalid repro `.wprof.pb` means the \
             auto-repro chain broke at one of: \
             `evaluate_vm_result`'s effective_auto_repro gate \
             (src/test_support/eval.rs L1787), \
             `attempt_auto_repro`'s second-VM boot \
             (src/test_support/probe.rs), or \
             `write_auto_repro_sidecar_artifacts`'s \
             MsgType::WprofTrace arm (probe.rs L498-516).",
            repro = expected_repro_pb.display(),
        ),
    ));

    Ok(result)
}
