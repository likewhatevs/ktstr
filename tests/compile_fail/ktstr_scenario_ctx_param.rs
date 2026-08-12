use ktstr::ktstr_scenario;

#[ktstr_scenario(llcs = 1, cores = 2, threads = 1)]
fn bad(ctx: &ktstr::scenario::Ctx) -> ktstr::scenario::ScenarioDef {
    ktstr::scenario::ScenarioDef::with_defs(vec![ctx.cgroup_def("cg_0")])
}

fn main() {}
