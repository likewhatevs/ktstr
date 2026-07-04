// Visibility prefix must precede the const name. A trailing
// visibility token after the name is rejected at parse time
// (syn::Visibility::parse consumed Inherited for the leading
// position; const_name then consumes `MY_SCHED`, and the parser
// peeks the next token, sees the misplaced `pub`, and returns an
// explicit diagnostic before the `,` is parsed).
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED pub, {
    name = "my_sched",
    binary = "scx_my_sched",
});

fn main() {}
