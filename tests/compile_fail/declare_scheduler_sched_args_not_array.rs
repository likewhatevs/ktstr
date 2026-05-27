// `declare_scheduler!`'s `sched_args` field requires an array literal
// `[..]`. A scalar literal (string, int, etc.) is rejected at
// macro-expansion time so the operator gets a typed error instead of
// a confusing downstream "expected &[&str], got &str" deep in the
// scheduler invocation path.
use ktstr::declare_scheduler;

declare_scheduler!(SCHED_ARGS_NOT_ARRAY, {
    name = "sched_args_not_array",
    binary = "scx_test",
    sched_args = "not_an_array",
});

fn main() {}
