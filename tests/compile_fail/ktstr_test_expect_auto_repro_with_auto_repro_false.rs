use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// expect_auto_repro asserts the auto-repro path FIRED. With
// auto_repro = false the path cannot fire, so the assertion is
// structurally unsatisfiable — the macro rejects the combination
// at parse-time rather than letting it silently no-op at runtime.
#[ktstr_test(scheduler = SCHED, wprof, expect_auto_repro, auto_repro = false)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
