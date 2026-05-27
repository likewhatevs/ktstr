use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// host_only skips the VM boot that owns the auto-repro machinery,
// so the auto-repro path cannot fire and the assertion is
// structurally unsatisfiable. The macro rejects the combination
// at parse-time.
#[ktstr_test(scheduler = SCHED, wprof, expect_auto_repro, host_only = true)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
