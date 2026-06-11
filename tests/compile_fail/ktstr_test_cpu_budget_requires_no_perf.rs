use ktstr::ktstr_test;

// cpu_budget is non-zero (passes the zero-reject) but no_perf_mode is
// absent — this fixture isolates the cpu_budget-requires-no_perf_mode gate.
#[ktstr_test(cpu_budget = 4)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
