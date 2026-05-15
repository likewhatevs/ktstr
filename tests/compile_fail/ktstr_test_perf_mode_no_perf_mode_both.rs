use ktstr::ktstr_test;

#[ktstr_test(performance_mode = true, no_perf_mode = true)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
