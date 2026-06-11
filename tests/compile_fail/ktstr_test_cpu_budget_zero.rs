use ktstr::ktstr_test;

// no_perf_mode is present so the cpu_budget-requires-no_perf_mode gate
// passes — this fixture isolates the cpu_budget = 0 zero-reject.
#[ktstr_test(cpu_budget = 0, no_perf_mode)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
