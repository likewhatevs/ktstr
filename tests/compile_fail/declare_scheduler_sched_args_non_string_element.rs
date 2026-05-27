// `declare_scheduler!`'s `sched_args` field elements must each be
// string literals (the runtime feeds them verbatim as CLI args to the
// scheduler binary). A non-string element (int, path, expression) is
// rejected at macro-expansion time so the operator gets a typed error
// instead of an "expected &str" deeper in the codegen.
use ktstr::declare_scheduler;

declare_scheduler!(SCHED_ARGS_NON_STRING_ELEMENT, {
    name = "sched_args_non_string_element",
    binary = "scx_test",
    sched_args = ["--exit-dump-len", 1048576],
});

fn main() {}
