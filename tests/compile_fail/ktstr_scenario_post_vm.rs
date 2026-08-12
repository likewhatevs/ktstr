use ktstr::ktstr_scenario;

fn check(_result: &ktstr::prelude::VmResult) -> anyhow::Result<()> {
    Ok(())
}

#[ktstr_scenario(llcs = 1, cores = 2, threads = 1, post_vm = check)]
fn bad() -> ktstr::scenario::ScenarioDef {
    ktstr::scenario::ScenarioDef::with_defs(vec![ktstr::scenario::ops::CgroupDef::named("cg_0")])
}

fn main() {}
