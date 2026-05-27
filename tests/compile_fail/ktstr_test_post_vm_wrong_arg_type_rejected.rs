use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// `post_vm` requires a callback with signature
// `fn(&VmResult) -> anyhow::Result<()>`. A free-fn taking `&str`
// (the wrong arg type) must be rejected by rustc at the
// macro-emit site so a test author cannot accidentally wire an
// unrelated callback at the slot. The macro itself accepts any
// path expression (`syn::Expr::Path`); the type-system gate is
// what catches the wrong-sig case at compile time.
fn bad_post_vm(_s: &str) -> anyhow::Result<()> {
    Ok(())
}

#[ktstr_test(scheduler = SCHED, post_vm = bad_post_vm)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
