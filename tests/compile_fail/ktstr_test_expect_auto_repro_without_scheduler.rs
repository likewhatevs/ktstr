use ktstr::ktstr_test;

// The auto-repro path re-launches the failing scheduler in a
// second VM. Without a scheduler attribute there is nothing to
// relaunch, so the assertion is structurally unsatisfiable. The
// macro rejects the combination at parse-time.
#[ktstr_test(wprof, expect_auto_repro)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
