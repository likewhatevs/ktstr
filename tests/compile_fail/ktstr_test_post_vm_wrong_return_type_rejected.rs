use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};
use ktstr::vmm::VmResult;

const SCHED: Scheduler = Scheduler::named("test_sched").binary(SchedulerSpec::Discover("dummy"));

// `post_vm` requires a callback with signature
// `fn(&VmResult) -> anyhow::Result<()>`. A free-fn returning `()`
// (the wrong return type) must be rejected by rustc at the
// macro-emit site so a test author cannot wire a non-failing
// callback at the slot. The macro itself accepts any path
// expression (`syn::Expr::Path`); the type-system gate is what
// catches the wrong-sig case at compile time.
fn bad_post_vm(_result: &VmResult) {}

#[ktstr_test(scheduler = SCHED, post_vm = bad_post_vm)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
