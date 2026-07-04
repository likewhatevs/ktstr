// `extern` is a Rust keyword, so it is neither a valid visibility
// prefix nor a valid const name. syn::Visibility::parse treats it as
// Inherited (no prefix, no error); the const-name syn::Ident parse
// then rejects it at macro-expand time with `expected identifier,
// found keyword `extern`` (see the .stderr sibling).
use ktstr::declare_scheduler;

declare_scheduler!(extern MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
});

fn main() {}
