//! Host-side unit tests for [`ScenarioDef`]. No VM, no `&Ctx` — which
//! is the point: everything a `ScenarioDef` knows is knowable here.

use std::time::Duration;

use super::ScenarioDef;
use crate::assert::Assert;
use crate::scenario::Ctx;
use crate::scenario::ops::{CgroupDef, HoldSpec, Setup, Step};

/// `with_defs` produces exactly the shape `execute_defs` runs: one
/// step, the given defs, `HoldSpec::FULL`, no ops. Pinned because
/// `with_defs` is the constructor most ported tests use, so a drift
/// here silently changes the meaning of every one of them.
#[test]
fn with_defs_is_one_full_hold_step() {
    let def = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")]);
    assert_eq!(def.steps().len(), 1, "with_defs builds a single step");
    let step = &def.steps()[0];
    assert!(step.ops.is_empty(), "with_defs adds no ops: {:?}", step.ops);
    assert!(
        matches!(step.hold, HoldSpec::Frac(f) if (f - 1.0).abs() < f64::EPSILON),
        "with_defs holds for the full duration (HoldSpec::FULL == Frac(1.0)), got {:?}",
        step.hold,
    );
    let Setup::Defs(defs) = &step.setup else {
        panic!(
            "with_defs must build a static Setup::Defs, got {:?}",
            step.setup
        );
    };
    let names: Vec<&str> = defs.iter().map(|d| d.name.as_ref()).collect();
    assert_eq!(
        names,
        vec!["cg_0", "cg_1"],
        "defs kept in declaration order"
    );
}

/// A scenario built from steps alone inherits `ctx.assert`; one built
/// with `with_checks` / `set_checks` carries the override. `run`
/// forwards `checks()` to `execute_steps_with`, so this is the
/// property that decides whether a ported test's gates still apply.
#[test]
fn checks_are_none_unless_declared() {
    let bare = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0")]);
    assert!(
        bare.checks().is_none(),
        "a scenario with no declared checks inherits ctx.assert",
    );

    let gate = Assert::default_checks().min_iteration_rate(5000.0);
    let with = ScenarioDef::with_checks(
        vec![Step::with_defs(
            vec![CgroupDef::named("cg_0")],
            HoldSpec::FULL,
        )],
        gate,
    );
    assert_eq!(
        with.checks().map(|c| c.min_iteration_rate),
        Some(Some(5000.0)),
        "with_checks carries the override",
    );

    let set = bare.set_checks(gate);
    assert_eq!(
        set.checks().map(|c| c.min_iteration_rate),
        Some(Some(5000.0)),
        "set_checks REPLACES the (absent) override",
    );
}

/// `From<Step>` and `From<Vec<Step>>` exist so a scenario body can end
/// in a bare step or step list. Both must land on the same value as
/// the explicit constructor.
#[test]
fn from_step_and_from_vec_agree_with_new() {
    let mk = || Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::frac(0.5));

    let from_one: ScenarioDef = mk().into();
    let from_vec: ScenarioDef = vec![mk()].into();
    let explicit = ScenarioDef::new(vec![mk()]);

    assert_eq!(from_one.steps().len(), 1);
    assert_eq!(
        format!("{:?}", from_one.steps()),
        format!("{:?}", explicit.steps()),
        "From<Step> == new(vec![step])",
    );
    assert_eq!(
        format!("{:?}", from_vec.steps()),
        format!("{:?}", explicit.steps()),
        "From<Vec<Step>> == new(steps)",
    );
}

/// Multi-step scenarios keep their steps in order with their
/// individual holds — the `sched_dynamic_add` shape (add a cgroup
/// half-way through the run).
#[test]
fn multi_step_preserves_order_and_holds() {
    let def = ScenarioDef::new(vec![
        Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::frac(0.5)),
        Step::with_defs(vec![CgroupDef::named("cg_1")], HoldSpec::frac(0.5)),
    ]);
    let names: Vec<&str> = def
        .steps()
        .iter()
        .map(|s| match &s.setup {
            Setup::Defs(defs) => defs[0].name.as_ref(),
            Setup::Factory(_) => panic!("static defs expected"),
        })
        .collect();
    assert_eq!(names, vec!["cg_0", "cg_1"]);
    for step in def.steps() {
        assert!(
            matches!(step.hold, HoldSpec::Frac(f) if (f - 0.5).abs() < f64::EPSILON),
            "each step keeps its own half-duration hold, got {:?}",
            step.hold,
        );
    }
}

/// `is_declarative` is the boundary a data consumer must respect: true
/// for static def lists, false as soon as any step defers its cgroup
/// list to an `fn(&Ctx)`. Getting this backwards would let a consumer
/// read an empty cgroup list off a factory step and believe it.
#[test]
fn is_declarative_rejects_factory_setup() {
    let static_def = ScenarioDef::new(vec![
        Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::FULL),
        Step::hold(HoldSpec::fixed(Duration::from_secs(1))),
    ]);
    assert!(
        static_def.is_declarative(),
        "static def lists (and op-only steps) are fully inspectable",
    );

    fn per_ctx(ctx: &Ctx) -> Vec<CgroupDef> {
        vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)]
    }
    let mut factory_step = Step::hold(HoldSpec::FULL);
    factory_step.setup = Setup::with_factory(per_ctx);
    let factory_def = ScenarioDef::new(vec![
        Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::FULL),
        factory_step,
    ]);
    assert!(
        !factory_def.is_declarative(),
        "a single Setup::Factory step makes the whole scenario \
         non-inspectable — the predicate must not report per-step",
    );
}

/// An empty scenario is representable (`Default`) and trivially
/// declarative. Cheap, but it pins that `is_declarative`'s `all()` is
/// vacuously true rather than special-cased to false.
#[test]
fn default_is_empty_and_declarative() {
    let def = ScenarioDef::default();
    assert!(def.steps().is_empty());
    assert!(def.checks().is_none());
    assert!(def.is_declarative());
}

/// `into_steps` yields exactly what `steps()` borrowed — the handoff a
/// consumer uses to feed the existing `execute_steps` surface.
#[test]
fn into_steps_round_trips() {
    let def = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0")]);
    let borrowed = format!("{:?}", def.steps());
    let owned = def.into_steps();
    assert_eq!(format!("{owned:?}"), borrowed);
}
