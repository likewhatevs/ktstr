use ktstr::ktstr_test;

ktstr::declare_scheduler!(HOST_ONLY_CONFLICT_SCHED, {
    name = "host_only_conflict_sched",
    binary = "host_only_conflict_sched_bin",
});

#[ktstr_test(host_only = true, scheduler = HOST_ONLY_CONFLICT_SCHED)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
