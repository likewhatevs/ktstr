// Pins macro-time rejection of empty `kernels = [..]` entries. An
// empty string parses as `CacheKey("")`; as a per-scheduler `kernels`
// filter (sched_kernel_filter_accepts / entry_matches_spec) it
// sanitizes to `kernel_` and can never match a real kernel label, so
// the scheduler cell would silently never emit and the test would
// never run. The macro rejects it at expand time so the diagnostic
// lands on the literal.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    kernels = [""],
});

fn main() {}
