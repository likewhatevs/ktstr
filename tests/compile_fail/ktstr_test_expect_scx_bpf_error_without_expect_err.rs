use ktstr::ktstr_test;

// Omit `expect_err` entirely. It defaults to `false`, so pairing
// the matcher with the default falls into the same conflict as
// pairing it with `expect_err = false` explicitly. The macro check
// fires on `(matcher_set) && !expect_err` regardless of whether
// expect_err was explicit or defaulted.
#[ktstr_test(expect_scx_bpf_error_contains = "out of range")]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
