//! Order-independence and override-precedence coverage for the
//! cgroup-level defaults that flow through
//! [`CgroupDef::merged_works`] (`nice` / `comm` / `uid` / `gid` /
//! `numa_node`) plus the per-WorkSpec [`CgroupDef::pcomm`] writer.
//!
//! The merged-works contract: each cgroup-level default is applied
//! to every [`WorkSpec`] in `works` whose own field is unset
//! (`None` across the board — `nice` is now `Option<i32>` and
//! shares the same `None`-as-unset polarity as the rest).
//! Builder ordering must not change the outcome —
//! `def.work(spec).nice(n)` and `def.nice(n).work(spec)` both
//! produce a merged WorkSpec carrying `nice = Some(n)`.
//! WorkSpecs that explicitly set `Some(_)` (including `Some(0)`)
//! must keep their own. `pcomm` is the lone exception: it lives
//! only on `WorkSpec`, so `CgroupDef::pcomm` writes the value
//! into every WorkSpec at builder time and overwrites any prior
//! per-WorkSpec value, with `def.pcomm(..).work(spec)` emitting
//! a default leader and pushing the explicit `spec` after (so
//! the `spec`'s own pcomm survives).

use super::*;
use crate::scenario::Ctx;
use crate::workload::WorkSpec;

// Future contributor: if you hit an AmbiguousIfImpl compile error
// here after adding `derive(Default)` (or a manual Default impl)
// on CgroupDef, the rationale is in src/scenario/ops/types/cgroup_def.rs
// at the foot-of-file no-Default comment block (TL;DR: the
// derived/zeroed default produced `name = "cg_0"`, which collides
// with the conventional first cgroup name in nearly every scenario
// — `..Default::default()` callers would silently share state with
// the scenario's first named entry). Use `CgroupDef::named(...)` at
// every construction site.
assert_not_impl_default!(CgroupDef);

#[test]
fn merged_works_nice_order_independent() {
    let pre = CgroupDef::named("cg").nice(7).work(WorkSpec::default());
    let post = CgroupDef::named("cg").work(WorkSpec::default()).nice(7);
    assert_eq!(pre.merged_works()[0].nice, Some(7));
    assert_eq!(post.merged_works()[0].nice, Some(7));
}

#[test]
fn merged_works_comm_order_independent() {
    let pre = CgroupDef::named("cg").comm("hot").work(WorkSpec::default());
    let post = CgroupDef::named("cg").work(WorkSpec::default()).comm("hot");
    assert_eq!(pre.merged_works()[0].comm.as_deref(), Some("hot"));
    assert_eq!(post.merged_works()[0].comm.as_deref(), Some("hot"));
}

#[test]
fn merged_works_uid_order_independent() {
    let pre = CgroupDef::named("cg").uid(1234).work(WorkSpec::default());
    let post = CgroupDef::named("cg").work(WorkSpec::default()).uid(1234);
    assert_eq!(pre.merged_works()[0].uid, Some(1234));
    assert_eq!(post.merged_works()[0].uid, Some(1234));
}

#[test]
fn merged_works_gid_order_independent() {
    let pre = CgroupDef::named("cg").gid(4321).work(WorkSpec::default());
    let post = CgroupDef::named("cg").work(WorkSpec::default()).gid(4321);
    assert_eq!(pre.merged_works()[0].gid, Some(4321));
    assert_eq!(post.merged_works()[0].gid, Some(4321));
}

#[test]
fn merged_works_numa_node_order_independent() {
    let pre = CgroupDef::named("cg")
        .numa_node(2)
        .work(WorkSpec::default());
    let post = CgroupDef::named("cg")
        .work(WorkSpec::default())
        .numa_node(2);
    assert_eq!(pre.merged_works()[0].numa_node, Some(2));
    assert_eq!(post.merged_works()[0].numa_node, Some(2));
}

/// Per-WorkSpec values must beat cgroup-level defaults. The merge
/// rule is "fill if unset", not "overwrite", so a WorkSpec that
/// explicitly carries `nice` / `comm` / `uid` / `gid` /
/// `numa_node` keeps its own value regardless of the
/// cgroup-level default. Pins the polarity of each merge gate.
#[test]
fn merged_works_workspec_overrides_default() {
    let spec = WorkSpec::default()
        .nice(3)
        .comm("override")
        .uid(11)
        .gid(22)
        .numa_node(5);
    let def = CgroupDef::named("cg")
        .nice(7)
        .comm("default")
        .uid(99)
        .gid(88)
        .numa_node(0)
        .work(spec);
    let merged = def.merged_works();
    assert_eq!(merged.len(), 1);
    let w = &merged[0];
    assert_eq!(w.nice, Some(3), "WorkSpec nice must beat default_nice");
    assert_eq!(
        w.comm.as_deref(),
        Some("override"),
        "WorkSpec comm must beat default_comm",
    );
    assert_eq!(w.uid, Some(11), "WorkSpec uid must beat default_uid");
    assert_eq!(w.gid, Some(22), "WorkSpec gid must beat default_gid");
    assert_eq!(
        w.numa_node,
        Some(5),
        "WorkSpec numa_node must beat default_numa_node",
    );
}

/// `Some(0)` on a WorkSpec is an explicit "use nice 0" — it
/// must opt out of the cgroup-level `default_nice` merge.
/// `merged_works` only fills `default_nice` when `w.nice.is_none()`,
/// so `Some(0)` keeps its own value and does not pick up the
/// cgroup default. Pins the core semantic differentiator that
/// `Option<i32>` provides over a plain `i32` (where 0 would be
/// ambiguous with "unset").
#[test]
fn merged_works_workspec_nice_some_zero_opts_out_of_default() {
    let spec = WorkSpec::default().nice(0);
    let def = CgroupDef::named("cg").nice(7).work(spec);
    let merged = def.merged_works();
    assert_eq!(
        merged[0].nice,
        Some(0),
        "Some(0) must opt out of cgroup default nice(7)"
    );
}

/// `CgroupDef::pcomm` writes `pcomm` into every WorkSpec in
/// `works`. When the builder runs `pcomm(..)` AFTER a `work(spec)`
/// where the spec has its own `pcomm`, the pcomm setter
/// overwrites the per-WorkSpec value — pcomm is a fan-out
/// applied to every existing WorkSpec, not a "merge if unset"
/// default. Pins the documented overwrite semantics.
#[test]
fn pcomm_after_work_overrides() {
    let spec = WorkSpec::default().pcomm("explicit");
    let def = CgroupDef::named("cg").work(spec).pcomm("forced");
    let works = def.merged_works();
    assert_eq!(works.len(), 1);
    assert_eq!(
        works[0].pcomm.as_deref(),
        Some("forced"),
        "pcomm() after work() must overwrite the WorkSpec's own pcomm \
         (the convenience method is a fan-out, not a merge-if-unset \
         default)",
    );
}

/// `CgroupDef::pcomm` called BEFORE any `work(spec)` pushes a
/// default WorkSpec carrying the pcomm; subsequent `work(spec)`
/// calls APPEND new entries to `works`, and those new entries
/// keep their own `pcomm` (or `None`) — pcomm only writes to
/// WorkSpecs that exist when the builder runs. Pins the
/// behaviour so a future change that fan-out-rewrote `pcomm` over
/// later-appended WorkSpecs surfaces here as a regression.
#[test]
fn pcomm_then_work_appends() {
    let extra = WorkSpec::default().pcomm("appended");
    let def = CgroupDef::named("cg").pcomm("initial").work(extra);
    let works = def.merged_works();
    assert_eq!(works.len(), 2, "pcomm() pushes one default + work() one");
    // First WorkSpec is the default that pcomm() created — its
    // pcomm is the one pcomm() set.
    assert_eq!(
        works[0].pcomm.as_deref(),
        Some("initial"),
        "pcomm() before any work() pushes a default WorkSpec carrying \
         the pcomm value",
    );
    // Second WorkSpec was appended AFTER pcomm() ran, so it keeps
    // its own pcomm — pcomm() does NOT retroactively touch
    // later-pushed entries.
    assert_eq!(
        works[1].pcomm.as_deref(),
        Some("appended"),
        "WorkSpec appended after pcomm() keeps its own pcomm — \
         pcomm() only writes to WorkSpecs that exist at call time",
    );
}

/// An empty `works` list yields a single synthesised
/// `WorkSpec::default()` with the cgroup-level defaults applied.
/// Pins the documented "empty → one default" substitution in
/// [`CgroupDef::merged_works`], plus the fact that the synthesised
/// WorkSpec sees the same merge rules as a hand-pushed default
/// — no shortcut path that bypasses the cgroup defaults.
#[test]
fn merged_works_empty_works_substitutes_default() {
    let def = CgroupDef::named("cg")
        .nice(11)
        .comm("default")
        .uid(7)
        .gid(8)
        .numa_node(1);
    let works = def.merged_works();
    assert_eq!(
        works.len(),
        1,
        "empty works must yield exactly one default WorkSpec"
    );
    let w = &works[0];
    assert_eq!(w.nice, Some(11));
    assert_eq!(w.comm.as_deref(), Some("default"));
    assert_eq!(w.uid, Some(7));
    assert_eq!(w.gid, Some(8));
    assert_eq!(w.numa_node, Some(1));
}

/// `merged_works` must be pure — calling it twice on the same
/// CgroupDef produces identical output and does not mutate the
/// cgroup's `works` vec or the cgroup-level defaults. Pins the
/// `&self` signature's contract against a future refactor that
/// drained `self.works` for performance.
#[test]
fn merged_works_does_not_mutate_self() {
    let def = CgroupDef::named("cg")
        .nice(5)
        .comm("named")
        .uid(101)
        .gid(202)
        .numa_node(3)
        .work(WorkSpec::default());
    let first = def.merged_works();
    let second = def.merged_works();
    assert_eq!(first.len(), second.len());
    assert_eq!(first.len(), 1);
    let a = &first[0];
    let b = &second[0];
    assert_eq!(a.nice, b.nice);
    assert_eq!(a.comm, b.comm);
    assert_eq!(a.uid, b.uid);
    assert_eq!(a.gid, b.gid);
    assert_eq!(a.numa_node, b.numa_node);
    // Underlying state untouched.
    assert_eq!(def.works.len(), 1);
    assert_eq!(def.default_nice, Some(5));
    assert_eq!(def.default_comm.as_deref(), Some("named"));
    assert_eq!(def.default_uid, Some(101));
    assert_eq!(def.default_gid, Some(202));
    assert_eq!(def.default_numa_node, Some(3));
}

/// [`CpuLimits::default()`] MUST return `max_period_us = 100_000`
/// (the kernel-rejected `0` would cause `EINVAL` on
/// `cpu.max` write, which is why the hand-rolled `Default` impl
/// exists). The other fields (`max_quota_us`, `weight`) MUST default to
/// `None`. A regression that switched back to `derive(Default)`
/// would silently produce `max_period_us = 0` and tests would
/// fail at apply-setup time with a kernel ENOENT/EINVAL rather
/// than at construction.
#[test]
fn cpu_limits_default_period_is_100_000_microseconds() {
    let limits = CpuLimits::default();
    assert_eq!(
        limits.max_period_us, 100_000,
        "CpuLimits::default().max_period_us MUST be 100_000 \
         (the kernel-default microsecond period). A regression \
         back to derive(Default) would produce 0, causing \
         EINVAL on cpu.max write.",
    );
    assert_eq!(limits.max_quota_us, None);
    assert_eq!(limits.weight, None);
}

/// [`Step::default()`] MUST produce `setup = Setup::Defs(vec![])`,
/// `ops = []`, `hold = HoldSpec::FULL`. This is the documented
/// identity-step: zero defs, zero ops, full-duration hold. A
/// regression that switched any field's default would change
/// the test-author-visible identity of `Step::default()` and
/// silently shift spread-default behavior across every call
/// site. Spread composability: a struct-update of the form
/// `Step { ops: my_ops, ..Default::default() }` must produce a
/// step with `setup = empty`, `hold = FULL`, and the explicit
/// `ops` — verified by constructing one and asserting the
/// non-spread fields match the Default contract.
#[test]
fn step_default_field_state_and_spread_composability() {
    let s = Step::default();
    assert!(
        matches!(s.setup, Setup::Defs(ref v) if v.is_empty()),
        "Step::default().setup MUST be Setup::Defs(vec![]); got {:?}",
        s.setup,
    );
    assert!(s.ops.is_empty(), "Step::default().ops MUST be []");
    assert!(
        matches!(s.hold, HoldSpec::Frac(f) if f == 1.0),
        "Step::default().hold MUST be HoldSpec::FULL \
         (HoldSpec::Frac(1.0)); got {:?}",
        s.hold,
    );

    // Spread composability: explicit `ops` lands, other fields
    // come from Default.
    let composed = Step {
        ops: vec![Op::add_cgroup("cg_spread")],
        ..Default::default()
    };
    assert_eq!(composed.ops.len(), 1);
    assert!(matches!(
        composed.setup,
        Setup::Defs(ref v) if v.is_empty()
    ));
    assert!(matches!(composed.hold, HoldSpec::Frac(f) if f == 1.0));
}

/// Pin that `Setup::with_factory(f)` and `Setup::default()` resolve to
/// the same shape a direct `Setup::Defs(v)` / `Setup::Factory(f)`
/// construction yields. `Setup` has no `PartialEq` (it holds a
/// fn pointer in `Factory`), so we discriminate on variant
/// shape + payload.
#[test]
fn setup_constructors_match_direct_variants() {
    // Default::default() -> Defs(vec![])
    let from_default: Setup = Default::default();
    assert!(
        matches!(from_default, Setup::Defs(ref v) if v.is_empty()),
        "Setup::default() must produce Setup::Defs(Vec::new())"
    );
    // with_factory(fn) -> Factory(fn)
    fn make_zero_cgroups(_: &Ctx) -> Vec<CgroupDef> {
        Vec::new()
    }
    let from_factory = Setup::with_factory(make_zero_cgroups);
    assert!(matches!(from_factory, Setup::Factory(_)));
}

/// GAP 4: pin that `Step::set_hold` replaces only the `hold`
/// field, leaving `setup` and `ops` untouched. A regression
/// where the builder accidentally reset `ops` or `setup` (e.g.
/// after a refactor that switched to a `Step::with(..)` style)
/// would silently drop test scenarios — set_ops calls would
/// look applied yet vanish after the subsequent set_hold.
///
/// Builds the Step with a NON-EMPTY setup via `Step::with_defs`
/// so any reset of `setup` is observable. Using
/// `Step::default().set_ops(...)` would yield an empty setup
/// before AND after the set_hold call — a setup-reset bug
/// would silently pass the "still empty" check.
#[test]
fn step_set_hold_replaces_hold_preserves_setup_and_ops() {
    let original = Step::with_defs(vec![CgroupDef::named("cg_setup")], HoldSpec::FULL)
        .set_ops(vec![Op::add_cgroup("cg_x")]);
    // Pin pre-state — non-empty setup, 1 op, full hold.
    assert!(
        matches!(
            &original.setup,
            Setup::Defs(v) if v.len() == 1 && v[0].name.as_ref() == "cg_setup"
        ),
        "fixture must have non-empty setup"
    );
    assert_eq!(original.ops.len(), 1);
    assert!(matches!(&original.hold, HoldSpec::Frac(f) if (*f - 1.0).abs() < f64::EPSILON));
    // Apply set_hold with a non-default value.
    let after = original.set_hold(HoldSpec::Frac(0.25));
    // hold replaced
    assert!(matches!(after.hold, HoldSpec::Frac(f) if (f - 0.25).abs() < f64::EPSILON));
    // ops preserved
    assert_eq!(after.ops.len(), 1);
    assert!(matches!(&after.ops[0], Op::AddCgroup { name } if name.as_ref() == "cg_x"));
    // setup preserved — non-empty cg_setup def survives
    assert!(
        matches!(
            &after.setup,
            Setup::Defs(v) if v.len() == 1 && v[0].name.as_ref() == "cg_setup"
        ),
        "Step::set_hold must not mutate setup; got {:?}",
        after.setup
    );
}
