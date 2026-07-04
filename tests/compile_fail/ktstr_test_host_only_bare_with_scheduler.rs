use ktstr::ktstr_test;

ktstr::declare_scheduler!(BARE_HOST_ONLY_CONFLICT_SCHED, {
    name = "bare_host_only_conflict_sched",
    binary = "bare_host_only_conflict_sched_bin",
});

// Pin that the BARE-form `host_only` (no `= true`) routes through
// AttrValues::assign_bool identically to the explicit form, and therefore
// trips the host_only-vs-scheduler mutex check at expansion time.
// A regression that special-cased the bare-form path through a
// different code arm (skipping the _set flag write or the value
// store) would silently accept this combination — this fixture
// catches that class of bug.
#[ktstr_test(host_only, scheduler = BARE_HOST_ONLY_CONFLICT_SCHED)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
