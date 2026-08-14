// An extra_sched_args element containing a line break is rejected at
// expansion: `/sched_args` frames one argument per line, so the element
// would be re-split into separate scheduler argv entries in the guest.
use ktstr::ktstr_test;

#[ktstr_test(extra_sched_args = ["--flag\nvalue"])]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
