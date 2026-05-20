//! End-to-end coverage of the BPF-map-fd pin primitive
//! ([`Op::PinBpfMap`](ktstr::scenario::ops::Op::PinBpfMap)
//! and the underlying
//! [`bpf_pin::open_bpf_map_fd_by_name`](ktstr::scenario::bpf_pin::open_bpf_map_fd_by_name)).
//!
//! Boots scx-ktstr, settles long enough for the BPF object to load,
//! then applies `Op::pin_bpf_map("bpf_bpf.bss")` followed by a second
//! `Op::pin_bpf_map("bpf_bpf.bss")` step.
//!
//! What this proves:
//!
//! - **Apply path succeeds against live kernel state.** A failing
//!   pin would surface as an `Err` from `execute_steps`; the test
//!   never reaches `post_vm` in that case. Reaching `post_vm` =
//!   both pin Steps applied without bailing.
//! - **Idempotent same-name re-pin.** The second pin Step's no-op
//!   semantic at `src/scenario/ops/mod.rs`'s `Op::PinBpfMap` arm
//!   (`Entry::Vacant` skip) is exercised: a literal second open
//!   would still succeed against the live map but the no-op path
//!   is what production callers hit and must work.
//! - **RAII teardown.** The BackdropState's `pinned_bpf_maps`
//!   HashMap drops at scenario end, closing the OwnedFd, releasing
//!   the kernel refcount. No explicit assertion — the test ending
//!   without leak (cleanup_budget_ms honored) is the signal.
//!
//! The map name `bpf_bpf.bss` is the kernel-visible name of scx-ktstr's
//! default libbpf object's `.bss` section (libbpf names the object
//! `bpf_bpf` for the test fixture; the `<obj>.bss` convention
//! produces `bpf_bpf.bss`).

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const PIN_TEST_SCHED: Scheduler =
    Scheduler::named("pin_bpf_map_e2e").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Boots scx-ktstr, holds for the BPF object to load, pins
/// `bpf_bpf.bss`, holds, pins again (idempotent no-op), holds.
/// Test passes if `post_vm` is reached with a healthy scheduler.
#[ktstr_test(
    scheduler = PIN_TEST_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 4,
    watchdog_timeout_s = 15,
    cleanup_budget_ms = 5000,
)]
fn pin_bpf_map_apply_path_succeeds(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        // Settle: scx-ktstr boots + attaches its BPF object.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // First pin: open and hold the bss fd. Failure bails
        // execute_steps; post_vm is not invoked.
        Step::with_op(
            Op::pin_bpf_map("bpf_bpf.bss"),
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // Second pin same name: idempotent no-op per the
        // BackdropState `Entry::Vacant` guard. Exercises the
        // skip path; must not error.
        Step::with_op(
            Op::pin_bpf_map("bpf_bpf.bss"),
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
    ];
    execute_steps(ctx, steps)
}
