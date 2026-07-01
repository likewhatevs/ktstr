// Pins the `max_llcs` cross-field check arm. `topology` declares
// 4 LLCs but `constraints.max_llcs = Some(2)` is below that, so
// every gauntlet preset carrying that topology fails the max_llcs
// filter (topo.num_llcs() <= max_llcs) and the test would match
// zero presets — the macro rejects it at compile time.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 4, 2, 1),
    constraints = TopologyConstraints {
        max_llcs: Some(2),
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
