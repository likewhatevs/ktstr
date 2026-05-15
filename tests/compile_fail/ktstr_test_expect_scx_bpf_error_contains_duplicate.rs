use ktstr::ktstr_test;

#[ktstr_test(host_only = true, expect_err = true, expect_scx_bpf_error_contains = "a", expect_scx_bpf_error_contains = "b")]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
