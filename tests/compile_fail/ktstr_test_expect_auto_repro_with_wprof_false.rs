use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// The .repro.wprof.pb artifact whose presence satisfies
// expect_auto_repro is written by the wprof binary attached in
// the auto-repro VM. wprof = false suppresses the binary, so no
// artifact lands and the assertion is structurally unsatisfiable.
#[ktstr_test(scheduler = SCHED, expect_auto_repro, wprof = false)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
