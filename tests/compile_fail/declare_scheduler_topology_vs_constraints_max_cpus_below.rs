// Pins the `max_cpus` cross-field check arm. `topology` declares
// total_cpus = 2*4*1 = 8 but `constraints.max_cpus = Some(4)`
// caps the accepted topology at 4 CPUs, so the declared 8-CPU
// topology can never satisfy it — every gauntlet preset would
// reject the test at runtime, which the macro rejects at compile
// time.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    constraints = TopologyConstraints {
        max_cpus: Some(4),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
