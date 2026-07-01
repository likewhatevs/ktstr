use ktstr::ktstr_test;

// survives_storm requires a scheduler — the kernel default (EEVDF) has no
// scx scheduler to die or be ejected, so survival is vacuous. The macro
// rejects survives_storm without a scheduler attribute at parse time.
#[ktstr_test(survives_storm)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
