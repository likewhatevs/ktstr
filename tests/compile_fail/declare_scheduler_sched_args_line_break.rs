// A sched_args element containing a line break is rejected at expansion:
// `/sched_args` frames one argument per line, so the element would be
// re-split into separate scheduler argv entries in the guest.
use ktstr::declare_scheduler;

declare_scheduler!(LINE_BREAK_ARGS, {
    name = "line_break_args",
    binary = "scx_foo",
    sched_args = ["--flag\nvalue"],
});

fn main() {}
