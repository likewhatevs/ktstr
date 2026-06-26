use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// survives_storm asserts the run PASSES with the scheduler alive;
// expect_err asserts the run FAILS. The two are contradictory, so the
// macro rejects the combination at parse time.
#[ktstr_test(scheduler = SCHED, survives_storm, expect_err = true)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
