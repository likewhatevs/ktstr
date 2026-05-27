use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// The effective auto-repro gate disables the path when
// expect_err = true (see entry.auto_repro && scheduler.is_some()
// && !entry.expect_err in eval.rs). With auto-repro disabled the
// expect_auto_repro assertion can never be satisfied. The user
// must pick one path or the other.
#[ktstr_test(scheduler = SCHED, wprof, expect_auto_repro, expect_err = true)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
