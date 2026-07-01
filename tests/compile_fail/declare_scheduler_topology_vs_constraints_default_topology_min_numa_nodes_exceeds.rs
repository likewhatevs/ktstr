// Pins the `min_numa_nodes` cross-field check arm against the
// default topology (when `topology` field is omitted). Default
// numa_nodes = 1; `constraints.min_numa_nodes = 2` exceeds the
// default effective topology's numa_nodes (1), so every gauntlet
// preset would reject the test at runtime — the macro rejects it
// at compile time.
use ktstr::declare_scheduler;
#[allow(unused_imports)]
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    constraints = TopologyConstraints {
        min_numa_nodes: 2,
        ..TopologyConstraints::DEFAULT
    },
});

fn main() {}
