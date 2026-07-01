use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// host_only and a `scheduler = ...` attribute are mutually exclusive:
// host_only skips the VM boot that owns the scheduler lifecycle, so the
// declared scheduler would never attach. validate_host_only_mutex runs
// before validate_expect_auto_repro_mutex, so with scheduler = SCHED
// present the scheduler mutex is the diagnostic that fires here (see the
// pinned .stderr), not the expect_auto_repro + host_only mutex. The
// macro rejects the combination at parse-time.
#[ktstr_test(scheduler = SCHED, wprof, expect_auto_repro, host_only = true)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
