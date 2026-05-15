use ktstr::ktstr_test;

#[ktstr_test(host_only = true, expect_err = true, expect_scx_bpf_error_matches = "a", expect_scx_bpf_error_matches = "b")]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
