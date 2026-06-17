#![cfg(feature = "wprof")]
//! End-to-end: `#[ktstr_test(wprof)]` produces a `.wprof.pb`
//! Perfetto trace in the sidecar dir on a passing scx-ktstr run.
//!
//! Distinct from `tests/wprof_e2e.rs` which only verifies the
//! embed → `/bin/wprof` file landing in shell-mode VMs. This
//! test pins the full chain:
//!   1. macro `#[ktstr_test(wprof)]` sets `KtstrTestEntry::wprof = true`
//!   2. primary VM builder wires `WprofConfig` from
//!      `KTSTR_WPROF_PATH` env (set by cargo-ktstr install_env)
//!   3. initramfs packs `/bin/wprof` + DT_NEEDED libs
//!   4. kernel cmdline carries `KTSTR_WPROF_ARGS=...`
//!   5. guest init's `spawn_wprof_if_configured` spawns wprof
//!   6. on workload exit, guest sends `MsgType::WprofTrace`
//!      with the Perfetto `.pb` bytes
//!   7. host dispatch arm writes `{sidecar_dir}/{test_name}.wprof.pb`
//!
//! Any break in the chain surfaces as the `post_vm` assertion
//! failing.
//!
//! ## Host-side verification via post_vm
//!
//! The `.wprof.pb` shape check runs in the HOST-side
//! [`assert_wprof_pb_landed`] callback rather than the
//! guest test body. The body runs INSIDE the guest VM and
//! cannot read the host sidecar directory (no virtio-fs mount
//! for it per `src/vmm/rust_init/mounts.rs`'s mount table); the
//! `post_vm` callback runs HOST-side after `vm.run()` returns
//! with `.wprof.pb` already on disk, so it CAN read and
//! validate the file.
//!
//! Runs on the self-hosted CI runners (`[ktstr-x64]` /
//! `[ktstr-arm64]`). ktstr supplies the guest kernel itself via
//! its kernel-build cache.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec, VmResult, WorkType};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Boots scx-ktstr with wprof attached, runs a minimal SpinWait
/// workload, then asserts the Perfetto `.pb` lands in the
/// sidecar dir with non-trivial size and a plausible protobuf
/// leading byte via the host-side `post_vm` callback (the
/// [`VmResult::assert_wprof_pb_landed`](ktstr::vmm::VmResult::assert_wprof_pb_landed)
/// method ref — the macro's `post_vm = PATH` parser accepts
/// UFCS-style method paths directly; the function-item coerces
/// to the same `fn(&VmResult) -> anyhow::Result<()>` shape
/// [`PostVmCallback`](ktstr::test_support::PostVmCallback)
/// requires, eliminating the prior one-line wrapper fn that
/// just delegated to the method).
///
/// `auto_repro = false` — the test passes, so auto-repro never
/// fires; the assertion exercises the PRIMARY-VM wprof wire-up
/// (not the auto-repro path covered by `crate::test_support::probe`).
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 3,
    watchdog_timeout_s = 15,
    wprof,
    auto_repro = false,
    post_vm = VmResult::assert_wprof_pb_landed,
)]
fn wprof_artifact_lands_in_sidecar_on_scx_ktstr_primary(ctx: &Ctx) -> Result<AssertResult> {
    // Minimal SpinWait workload — enough scheduler activity that
    // wprof's sched-event tracer populates the ringbuf, but
    // bounded by the 3 s test duration.
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
    let result = execute_steps(ctx, steps)?;
    // Shape assertion runs in `assert_wprof_pb_landed`
    // above — host-side, with access to the sidecar file.
    Ok(result)
}
