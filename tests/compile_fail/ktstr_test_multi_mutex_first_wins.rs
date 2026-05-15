use ktstr::ktstr_test;

ktstr::declare_scheduler!(MULTI_MUTEX_CONFLICT_SCHED, {
    name = "multi_mutex_conflict_sched",
    binary = "multi_mutex_conflict_sched_bin",
});

// Three concurrent mutex conflicts: host_only collides with
// scheduler, num_snapshots, AND auto_repro all at once. Pin that
// the FIRST check in ktstr-macros (host_only + scheduler) wins,
// so a refactor that reorders or merges the four mutex blocks
// (following the "Ordering note" comment in ktstr-macros/src/
// lib.rs — grep by that exact phrase) doesn't silently change
// which diagnostic the operator sees first. Pinning precedence
// guards against disorienting diagnostic churn under
// multi-conflict edits.
#[ktstr_test(
    host_only = true,
    scheduler = MULTI_MUTEX_CONFLICT_SCHED,
    num_snapshots = 3,
    auto_repro = true,
)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
