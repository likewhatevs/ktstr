use ktstr::ktstr_test;

#[ktstr_test(host_only = true, config = "first", config = "second")]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
