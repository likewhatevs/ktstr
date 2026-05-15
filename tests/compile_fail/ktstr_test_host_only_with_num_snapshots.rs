use ktstr::ktstr_test;

#[ktstr_test(host_only = true, num_snapshots = 3)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
