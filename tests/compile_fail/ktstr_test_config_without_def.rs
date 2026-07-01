// `#[ktstr_test(config = "...")]` paired with the default scheduler
// (`Scheduler::EEVDF` — `config_file_def: None`) must fail at
// compile time. The macro emits a `const __KTSTR_CONFIG_PAIRING_<NAME>: () = { ... };`
// block that const-evaluates `(scheduler).config_file_def.is_some()`
// against the macro-known `config_set` flag and panics on mismatch.
use ktstr::ktstr_test;

#[ktstr_test(config = "{}")]
fn config_without_def(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
