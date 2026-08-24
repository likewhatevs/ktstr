//! [`ScenarioDef`] — a scenario as a VALUE rather than as a function
//! body.
//!
//! Today a ktstr test's workload is expressed as arbitrary Rust that
//! happens to end in a call to [`execute_steps`](ops::execute_steps):
//!
//! ```ignore
//! #[ktstr_test(scheduler = SCHED, llcs = 1, cores = 2, threads = 1)]
//! fn my_test(ctx: &Ctx) -> Result<AssertResult> {
//!     let steps = vec![Step::with_defs(
//!         vec![CgroupDef::named("cg_0")],
//!         HoldSpec::FULL,
//!     )];
//!     execute_steps(ctx, steps)
//! }
//! ```
//!
//! The *shape* of that body is declarative — a list of steps — but
//! nothing in the type system says so, and nothing can recover the
//! step list without running the test inside a booted guest. A
//! `ScenarioDef` is that step list lifted out of the body into a
//! value that can be built, inspected, compared, and printed on the
//! host with no VM and no `&Ctx`:
//!
//! ```ignore
//! #[ktstr_scenario(scheduler = SCHED, llcs = 1, cores = 2, threads = 1)]
//! fn my_test() -> ScenarioDef {
//!     ScenarioDef::with_defs(vec![CgroupDef::named("cg_0")])
//! }
//! ```
//!
//! See [`crate::ktstr_scenario`] for the entrypoint that consumes
//! these.
//!
//! # Why no `&Ctx`
//!
//! A scenario builder deliberately takes no [`Ctx`]. The context is a
//! *runtime* value (it exists only inside the guest, once the topology
//! is materialised), so any body that reads it is not extractable.
//!
//! The coupling is also mostly illusory. The overwhelmingly common
//! `&Ctx` use in existing scenario bodies is
//! [`Ctx::cgroup_def`](crate::scenario::Ctx::cgroup_def), which is
//! defined as `CgroupDef::named(n).workers(ctx.workers_per_cgroup)`.
//! But a [`ops::CgroupDef`] with no explicit work spec
//! resolves its worker count through the same default: an empty
//! `works` list yields a `WorkSpec::default()` whose `num_workers` is
//! `None`, and the step runner's `resolve_num_workers` then applies
//! `unwrap_or(ctx.workers_per_cgroup)`. So `ctx.cgroup_def("cg_0")`
//! and `CgroupDef::named("cg_0")` resolve to the *same* worker count —
//! the former binds the default eagerly, the latter leaves it symbolic
//! for the runner to bind at apply-setup time.
//!
//! Leaving it symbolic is the strictly more useful form: the value
//! stays a description of the workload rather than a description
//! pre-specialised to one host's topology.
//!
//! # What is *not* declarative
//!
//! [`ops::Setup::Factory`] holds an
//! `fn(&Ctx) -> Vec<CgroupDef>` — an escape hatch for cgroup lists
//! that genuinely depend on the runtime topology. A `ScenarioDef` may
//! legally contain one (nothing in the type forbids it), but its
//! cgroup list cannot be recovered without a `Ctx`.
//! [`ScenarioDef::is_declarative`] reports whether a given value is
//! free of them, and is the honest boundary of what a consumer can
//! read out of a scenario statically.

use anyhow::Result;

use crate::assert::{Assert, AssertResult};
use crate::scenario::Ctx;
use crate::scenario::ops::{self, CgroupDef, HoldSpec, Setup, Step};

/// A scenario expressed as data: an ordered list of
/// [`ops::Step`]s plus an optional [`Assert`] override.
///
/// This is exactly the pair of arguments
/// [`execute_steps_with`](ops::execute_steps_with) takes, minus the
/// `&Ctx` — which is to say, it is everything about a scenario that
/// is knowable before the guest boots.
///
/// # Construction
///
/// Per the [builder conventions](crate::scenario#builder-method-conventions):
///
/// - [`ScenarioDef::new`] — base constructor from a step list.
/// - [`ScenarioDef::with_defs`] — alternate constructor for the
///   single-step "declare these cgroups and hold for the whole run"
///   shape (the analogue of [`ops::execute_defs`]).
/// - [`ScenarioDef::with_checks`] — alternate constructor from steps
///   plus an [`Assert`] override (the analogue of
///   [`ops::execute_steps_with`]).
/// - [`ScenarioDef::set_checks`] — REPLACE the checks on an existing
///   value.
/// - `From<Step>` / `From<Vec<Step>>` — so a scenario body can end in
///   a bare step or step list and let the caller coerce.
///
/// # Execution
///
/// [`ScenarioDef::run`] dispatches to
/// [`execute_steps_with`](ops::execute_steps_with), so a scenario
/// executes through exactly the same runtime path as a hand-written
/// body. There is no second execution engine.
#[derive(Clone, Debug, Default)]
#[must_use = "a ScenarioDef describes a workload; dropping it runs nothing"]
pub struct ScenarioDef {
    /// The step sequence, in order.
    steps: Vec<Step>,
    /// Optional [`Assert`] override. `None` inherits `ctx.assert` (the
    /// merged three-layer config) exactly as
    /// [`execute_steps`](ops::execute_steps) does.
    checks: Option<Assert>,
}

impl ScenarioDef {
    /// Build a scenario from an ordered step list, inheriting
    /// `ctx.assert` for its checks.
    pub fn new(steps: Vec<Step>) -> Self {
        Self {
            steps,
            checks: None,
        }
    }

    /// Build a single-step scenario that declares `defs` and holds for
    /// the full scenario duration.
    ///
    /// The value-level analogue of [`ops::execute_defs`], and the
    /// single most common scenario shape in the test suite.
    pub fn with_defs(defs: Vec<CgroupDef>) -> Self {
        Self::new(vec![Step::with_defs(defs, HoldSpec::FULL)])
    }

    /// Build a scenario from a step list with an explicit [`Assert`]
    /// override, which replaces `ctx.assert` at execution time.
    ///
    /// The value-level analogue of [`ops::execute_steps_with`].
    pub fn with_checks(steps: Vec<Step>, checks: Assert) -> Self {
        Self {
            steps,
            checks: Some(checks),
        }
    }

    /// REPLACE the [`Assert`] override on an existing scenario.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn set_checks(mut self, checks: Assert) -> Self {
        self.checks = Some(checks);
        self
    }

    /// The scenario's steps, in order.
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The scenario's [`Assert`] override, if it declared one.
    #[must_use]
    pub fn checks(&self) -> Option<&Assert> {
        self.checks.as_ref()
    }

    /// Consume the scenario and yield its steps.
    #[must_use]
    pub fn into_steps(self) -> Vec<Step> {
        self.steps
    }

    /// Whether every step's cgroup setup is a static
    /// [`ops::Setup::Defs`] list — i.e. whether the whole
    /// scenario can be read without a [`Ctx`].
    ///
    /// `false` means at least one step uses
    /// [`ops::Setup::Factory`], whose
    /// `fn(&Ctx) -> Vec<CgroupDef>` cannot be evaluated on the host.
    /// Such a scenario still runs normally; it just is not statically
    /// inspectable, so a consumer that reads scenarios as data must
    /// check this and skip (or report) the ones that fail it rather
    /// than silently emitting an empty cgroup list.
    #[must_use]
    pub fn is_declarative(&self) -> bool {
        self.steps
            .iter()
            .all(|step| matches!(step.setup, Setup::Defs(_)))
    }

    /// Execute the scenario against a runtime context.
    ///
    /// Dispatches to [`ops::execute_steps_with`] with this scenario's
    /// checks, so the execution path is identical to that of a
    /// hand-written scenario body.
    pub fn run(&self, ctx: &Ctx) -> Result<AssertResult> {
        ops::execute_steps_with(ctx, self.steps.clone(), self.checks.as_ref())
    }
}

impl From<Step> for ScenarioDef {
    fn from(step: Step) -> Self {
        Self::new(vec![step])
    }
}

impl From<Vec<Step>> for ScenarioDef {
    fn from(steps: Vec<Step>) -> Self {
        Self::new(steps)
    }
}

#[cfg(test)]
mod tests;
