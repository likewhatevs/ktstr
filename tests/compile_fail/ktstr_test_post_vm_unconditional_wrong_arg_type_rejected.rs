use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// `post_vm_unconditional` carries the same signature contract as
// `post_vm` — `fn(&VmResult) -> anyhow::Result<()>`. Slot-parity
// pin: a regression that loosened one slot's type-check would
// silently loosen the other, so both slots get explicit
// wrong-arg-type coverage.
fn bad_post_vm_unconditional(_s: &str) -> anyhow::Result<()> {
    Ok(())
}

#[ktstr_test(scheduler = SCHED, post_vm_unconditional = bad_post_vm_unconditional)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
