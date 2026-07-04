// Pins the `max_cpus` cross-field check arm against the default
// topology. Default total_cpus = 2; `max_cpus = Some(1)` excludes
// any preset topology whose total_cpus is 2 or more — including the
// default topology (total_cpus=2), so every gauntlet preset would
// reject this test at runtime and the test would never execute.
// The macro catches the inconsistency at expand time.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        max_cpus: Some(1),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
