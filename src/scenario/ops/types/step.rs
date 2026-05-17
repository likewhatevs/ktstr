//! Scenario-step composition: [`Setup`] (defs / factory variants
//! with manual `Clone`+`Debug`+`Default`+`From` impls), [`Step`]
//! (ops + setup + hold), [`HoldSpec`] (frac / fixed / loop hold
//! shapes), plus the [`Op`] / [`OpKind`] helper impls that operate
//! around step-execution (`Op::discriminant`, `OpKind::bit_index`,
//! the `Op::*` per-variant constructor surface).
//!
//! The `impl Op` blocks here are siblings to the `Op` enum
//! definition in [`super::op`]; Rust permits multiple impl blocks
//! across files in the same crate. The split tracks responsibility:
//! op.rs owns the variant taxonomy + the `CpusetSpec` constructor
//! surface; this file owns the step-side helpers (bit-index map for
//! the `op_kinds` bitmask, per-variant constructor sugar).

use std::borrow::Cow;
use std::time::Duration;

use crate::scenario::Ctx;
use crate::workload::{AffinityIntent, WorkSpec, WorkType};

use super::{CgroupDef, CpusetSpec, KernelTarget, KernelValue, Op, OpKind};

// ---------------------------------------------------------------------------
// Step / HoldSpec
// ---------------------------------------------------------------------------

/// How to produce the CgroupDefs for a step's setup phase.
///
/// Construct via `Setup::Defs(vec)` (variant constructor for a static
/// list), [`Setup::with_factory`] (runtime-generated from `&Ctx`),
/// `Setup::default()` (no cgroups — `Setup::Defs(Vec::new())`), or via
/// [`From<Vec<CgroupDef>>`](Self::from) (`vec![def1, def2].into()`).
pub enum Setup {
    /// Static list of cgroup definitions.
    Defs(Vec<CgroupDef>),
    /// Factory that generates definitions from the runtime context.
    Factory(fn(&Ctx) -> Vec<CgroupDef>),
}

impl Clone for Setup {
    fn clone(&self) -> Self {
        match self {
            Setup::Defs(defs) => Setup::Defs(defs.clone()),
            Setup::Factory(f) => Setup::Factory(*f),
        }
    }
}

impl std::fmt::Debug for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Setup::Defs(defs) => f.debug_tuple("Defs").field(defs).finish(),
            Setup::Factory(_) => f
                .debug_tuple("Factory")
                .field(&"fn(&Ctx) -> Vec<CgroupDef>")
                .finish(),
        }
    }
}

impl Setup {
    /// Construct a [`Setup::Factory`] from a function pointer.
    /// `with_factory` per the scenario-module builder convention
    /// (`with_X` = constructor variant; see the module-level docs
    /// on [`crate::scenario`]). The `Defs` variant is constructed
    /// directly via `Setup::Defs(vec)` (variant constructor) or
    /// `Default::default()` for the empty case; `Factory` needs a
    /// named constructor because variant construction with a `fn`
    /// pointer literal is awkward to read.
    pub const fn with_factory(f: fn(&Ctx) -> Vec<CgroupDef>) -> Self {
        Setup::Factory(f)
    }

    pub(in crate::scenario::ops) fn resolve(&self, ctx: &Ctx) -> Vec<CgroupDef> {
        match self {
            Setup::Defs(defs) => defs.clone(),
            Setup::Factory(f) => f(ctx),
        }
    }

    pub(in crate::scenario::ops) fn is_empty(&self) -> bool {
        match self {
            Setup::Defs(defs) => defs.is_empty(),
            Setup::Factory(_) => false,
        }
    }
}

impl Default for Setup {
    /// Empty `Setup::Defs` — no cgroups created. The `Factory` variant
    /// cannot serve as a Default because it holds a fn pointer with no
    /// semantic no-op.
    fn default() -> Self {
        Setup::Defs(Vec::new())
    }
}

impl From<Vec<CgroupDef>> for Setup {
    fn from(defs: Vec<CgroupDef>) -> Self {
        Setup::Defs(defs)
    }
}

/// A sequence of ops followed by a hold period.
///
/// For non-`Loop` steps, `ops` are applied first, then `setup` cgroups
/// are created, configured, and populated. For `Loop` steps, `setup`
/// runs once before the ops loop.
///
/// Construct via [`Step::new`] (ops-only, no setup), [`Step::with_defs`]
/// (cgroup setup + hold), or [`Step::with_payload`] (payload-driven step).
/// For chained mutation of the ops list, [`Step::set_ops`] REPLACES
/// the existing vec — Backdrop's `extend_ops` semantics (APPEND) are not
/// mirrored here because Step is single-phase.
#[derive(Clone, Debug)]
pub struct Step {
    /// Cgroup setup applied before (non-`Loop`) or once above (`Loop`)
    /// the ops list. Runtime cgroups are spawned from this spec.
    pub setup: Setup,
    /// Ordered operations applied each time the step body runs:
    /// cpuset edits, task moves, spawn/despawn, etc.
    pub ops: Vec<Op>,
    /// How long, and whether to loop, after the ops finish one pass.
    pub hold: HoldSpec,
}

impl Step {
    /// Create a step with ops only (no CgroupDef setup).
    #[must_use = "dropping a Step discards its ops and hold for that scenario phase"]
    pub fn new(ops: Vec<Op>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(Vec::new()),
            ops,
            hold,
        }
    }

    /// Create a step with CgroupDef setup and a hold period.
    ///
    /// Most steps only need cgroup definitions and a hold duration.
    /// Use [`set_ops`](Step::set_ops) to chain ops onto the step.
    #[must_use = "dropping a Step discards its CgroupDef setup and hold for that scenario phase"]
    pub fn with_defs(defs: Vec<CgroupDef>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(defs),
            ops: vec![],
            hold,
        }
    }

    /// Replace the ops for a step, consuming and returning it.
    ///
    /// Named `set_ops` rather than `extend_ops` because the semantics
    /// are REPLACE, not EXTEND — contrast
    /// [`Backdrop::extend_ops`](crate::scenario::backdrop::Backdrop::extend_ops),
    /// which appends. A chained `Step::new(ops).set_ops(more)`
    /// drops `ops` and keeps only `more`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn set_ops(mut self, ops: Vec<Op>) -> Self {
        self.ops = ops;
        self
    }

    /// Replace the hold spec for a step, consuming and returning it.
    /// Sibling of [`set_ops`](Self::set_ops) — both REPLACE a single
    /// field. Bare-verb `set_` prefix matches `set_ops` for
    /// prefix-consistency within Step; the convention reserves
    /// `with_X` for alternative constructors (see [`with_defs`](Self::with_defs),
    /// [`with_payload`](Self::with_payload)).
    #[must_use = "builder methods consume self; bind the result"]
    pub const fn set_hold(mut self, hold: HoldSpec) -> Self {
        self.hold = hold;
        self
    }

    /// Create a step that spawns a single userspace
    /// [`Payload`](crate::test_support::Payload) binary in the
    /// background and holds for the given duration before teardown.
    ///
    /// Shorthand for `Step::new(vec![Op::run_payload(payload,
    /// vec![])], hold)`. The returned step is chainable — add
    /// `.set_ops(...)` to replace the ops vec (note the
    /// REPLACE-not-EXTEND semantics), or use
    /// `Op::wait_payload(name)` / `Op::kill_payload(name)` on later
    /// steps to control the spawned child.
    ///
    /// Test authors who want the payload placed in a named cgroup
    /// should use `Op::run_payload_in_cgroup` directly; this
    /// convenience targets the common "one payload, whole step"
    /// shape.
    #[must_use = "dropping a Step discards its payload and hold for that scenario phase"]
    pub fn with_payload(payload: &'static crate::test_support::Payload, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(Vec::new()),
            ops: vec![Op::run_payload(payload, vec![])],
            hold,
        }
    }
}

impl Default for Step {
    /// Empty setup, no ops, hold for the full scenario duration
    /// ([`HoldSpec::FULL`]). Useful as a sentinel in test fixtures
    /// that compose Steps via `..Default::default()` field overrides.
    fn default() -> Self {
        Self {
            setup: Setup::Defs(Vec::new()),
            ops: Vec::new(),
            hold: HoldSpec::FULL,
        }
    }
}

/// How a step advances after its ops are applied. `Frac` and `Fixed`
/// hold for a duration; `Loop` repeatedly re-applies `Step::ops` at a
/// fixed interval instead of holding.
///
/// Construct via the constructor methods ([`Self::fixed`],
/// [`Self::frac`], [`Self::loop_at`], or the [`Self::FULL`] const) —
/// variant syntax is reserved for pattern-matching in `match` arms.
///
/// `Copy` because every variant carries only `Copy` types (`f64`,
/// [`Duration`]); reuse the same `HoldSpec` value across multiple
/// [`Step::new`] / [`Step::with_defs`] / [`Step::with_payload`]
/// calls in a construction loop without an explicit `.clone()`.
/// `PartialEq` is derived so tests can `assert_eq!(step.hold, ...)`
/// and user code can pattern-compare values directly — float
/// equality on `Frac(f64)` follows IEEE 754 semantics (so
/// `HoldSpec::Frac(0.1 + 0.2) != HoldSpec::Frac(0.3)`, and
/// `HoldSpec::Frac(f64::NAN) != HoldSpec::Frac(f64::NAN)` —
/// [`Self::validate`] rejects NaN at intake so the non-reflexive
/// case is unreachable through validated construction, but the
/// derive inherits the IEEE 754 contract at the type level). `Eq` /
/// `Hash` remain impossible because `Frac` carries a float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HoldSpec {
    /// Fraction of the total scenario duration.
    Frac(f64),
    /// Fixed duration.
    Fixed(Duration),
    /// Repeat the step's ops in a loop at the given interval until the
    /// remaining scenario time is exhausted.
    Loop { interval: Duration },
}

impl HoldSpec {
    /// Hold for the full scenario duration. Equivalent to
    /// [`HoldSpec::frac(1.0)`](Self::frac) and resolves to
    /// `ctx.duration` at scenario-run time.
    pub const FULL: HoldSpec = HoldSpec::Frac(1.0);

    /// Hold for a fixed wall-clock duration. Sugar for
    /// `HoldSpec::Fixed(d)` that reads naturally in chain position
    /// (`Step::new(ops, HoldSpec::fixed(Duration::from_secs(5)))`)
    /// and surfaces in IDE autocomplete next to [`Self::frac`] and
    /// [`Self::loop_at`].
    ///
    /// For the common `settle + duration * fraction` pattern, prefer
    /// [`Ctx::settled_hold`](crate::scenario::Ctx::settled_hold) over
    /// `HoldSpec::fixed(ctx.settle + ctx.duration.mul_f64(frac))`.
    pub const fn fixed(d: Duration) -> HoldSpec {
        HoldSpec::Fixed(d)
    }

    /// Hold for a fraction of `ctx.duration` (the scenario duration
    /// configured on [`Ctx`](crate::scenario::Ctx)). Sugar for
    /// `HoldSpec::Frac(f)`; the resolved wall-clock hold is
    /// `ctx.duration * f` (e.g. `0.5` = half the scenario
    /// duration). `f` must be finite and `> 0.0` — see
    /// [`Self::validate`] for the rejection rules.
    pub const fn frac(f: f64) -> HoldSpec {
        HoldSpec::Frac(f)
    }

    /// Repeat the step's ops at the given interval until the
    /// remaining scenario time is exhausted. Sugar for
    /// `HoldSpec::Loop { interval }`; the value is `interval`. Must
    /// be non-zero — see [`Self::validate`] for the rejection rule.
    ///
    /// Named `loop_at` (verb-preposition) rather than `r#loop`
    /// because the variant name `Loop` collides with the Rust
    /// keyword `loop` — `loop_at` reads as "loop AT this interval"
    /// and avoids the raw-identifier escape. Sibling constructors
    /// [`Self::fixed`] / [`Self::frac`] match their variant names
    /// directly (no keyword conflict).
    pub const fn loop_at(interval: Duration) -> HoldSpec {
        HoldSpec::Loop { interval }
    }

    /// Reject hold values that are vacuous (no-op step) or would
    /// panic downstream.
    ///
    /// Rules:
    /// - `Fixed(Duration::ZERO)` — valid for settle steps and
    ///   op-only steps that apply changes without holding.
    /// - `Frac(f)` with `!f.is_finite()` (NaN/Inf) — propagates into
    ///   `Duration::from_secs_f64(f)` which panics.
    /// - `Frac(f)` with `f <= 0.0` — zero is vacuous, negative
    ///   panics in `Duration::from_secs_f64`.
    /// - `Loop { interval: Duration::ZERO }` — busy-polls the
    ///   deadline loop without yielding; almost always a typo.
    pub fn validate(&self) -> std::result::Result<(), String> {
        match self {
            HoldSpec::Fixed(_) => Ok(()),
            HoldSpec::Frac(f) if !f.is_finite() => Err(format!(
                "HoldSpec::Frac({f}) is not finite (NaN/Inf) — would \
                     panic in Duration::from_secs_f64"
            )),
            HoldSpec::Frac(f) if *f <= 0.0 => Err(format!(
                "HoldSpec::Frac({f}) must be > 0.0; negative values \
                     panic in Duration::from_secs_f64 and zero is vacuous"
            )),
            HoldSpec::Loop { interval } if interval.is_zero() => {
                Err("HoldSpec::Loop { interval: Duration::ZERO } would \
                     busy-spin the deadline check without yielding; use a \
                     non-zero interval"
                    .into())
            }
            _ => Ok(()),
        }
    }
}

impl Op {
    /// Return a unique bit index for each Op variant (for op_kinds bitmask).
    ///
    /// Dispatched via [`OpKind`] — the auto-generated fieldless shadow
    /// enum from `#[derive(strum::EnumDiscriminants)]` on [`Op`]. The
    /// indirection is load-bearing: `OpKind` also derives `EnumIter`,
    /// so `op_kind_bit_indices_are_unique_and_contiguous` can
    /// exhaustively verify every `OpKind` maps to a distinct,
    /// contiguous bit index — guarding against a new variant slipping
    /// in with a duplicated or gap-leaving index.
    pub(in crate::scenario::ops) fn discriminant(&self) -> u32 {
        OpKind::from(self).bit_index()
    }
}

impl OpKind {
    /// Unique bit index per variant, used by [`Op::discriminant`] for
    /// the `op_kinds` bitmask. Contiguous from 0 — the
    /// `op_kind_bit_indices_are_unique_and_contiguous` test iterates
    /// every variant via `EnumIter` and pins this.
    pub(in crate::scenario::ops) fn bit_index(self) -> u32 {
        match self {
            OpKind::AddCgroup => 0,
            OpKind::AddCgroupDef => 1,
            OpKind::RemoveCgroup => 2,
            OpKind::SetCpuset => 3,
            OpKind::ClearCpuset => 4,
            OpKind::SwapCpusets => 5,
            OpKind::SpawnWorkers => 6,
            OpKind::StopCgroup => 7,
            OpKind::SetAffinity => 8,
            OpKind::SpawnHost => 9,
            OpKind::MoveAllTasks => 10,
            OpKind::RunPayload => 11,
            OpKind::WaitPayload => 12,
            OpKind::KillPayload => 13,
            OpKind::FreezeCgroup => 14,
            OpKind::UnfreezeCgroup => 15,
            OpKind::CaptureSnapshot => 16,
            OpKind::WatchSnapshot => 17,
            OpKind::WriteKernelHot => 18,
            OpKind::WriteKernelCold => 19,
            OpKind::ReadKernelHot => 20,
            OpKind::ReadKernelCold => 21,
        }
    }
}

impl Op {
    /// Create a new cgroup.
    pub fn add_cgroup(name: impl Into<Cow<'static, str>>) -> Self {
        Op::AddCgroup { name: name.into() }
    }

    /// Create a cgroup mid-step from a full [`CgroupDef`].
    ///
    /// Mirrors `Step::with_defs(vec![def], ...)` semantics — applies
    /// the def's cpuset / cpu / memory / io / pids knobs and spawns
    /// its workers in one op — but at apply-ops time rather than
    /// during the step's setup pass. Use when the cgroup's
    /// declaration needs to depend on state observed by an earlier
    /// op in the same step (e.g. spawn a per-LLC cgroup after an
    /// `Op::SetCpuset` narrows the parent's available CPUs). The
    /// dedup check mirrors `apply_setup`'s rejection of name
    /// collisions with prior Backdrop or step-local CgroupDef
    /// declarations.
    pub fn add_cgroup_def(def: CgroupDef) -> Self {
        Op::AddCgroupDef { def }
    }

    /// Remove a cgroup (stops its workers first).
    pub fn remove_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::RemoveCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set a cgroup's cpuset.
    pub fn set_cpuset(cgroup: impl Into<Cow<'static, str>>, cpus: CpusetSpec) -> Self {
        Op::SetCpuset {
            cgroup: cgroup.into(),
            cpus,
        }
    }

    /// Clear a cgroup's cpuset (allow all CPUs).
    pub fn clear_cpuset(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::ClearCpuset {
            cgroup: cgroup.into(),
        }
    }

    /// Swap cpusets between two cgroups.
    pub fn swap_cpusets(a: impl Into<Cow<'static, str>>, b: impl Into<Cow<'static, str>>) -> Self {
        Op::SwapCpusets {
            a: a.into(),
            b: b.into(),
        }
    }

    /// Spawn workers in a cgroup.
    pub fn spawn_workers(cgroup: impl Into<Cow<'static, str>>, work: WorkSpec) -> Self {
        Op::SpawnWorkers {
            cgroup: cgroup.into(),
            work,
        }
    }

    /// Spawn workers in a cgroup with the given [`WorkType`] and every
    /// other [`WorkSpec`] knob defaulted. Sugar for the common
    /// single-knob spawn case where the test only cares about
    /// `work_type` and is happy with `Default::default()` for
    /// scheduling policy, affinity, mempolicy, etc. Mirrors the
    /// [`CgroupDef::named(...).work_type(...)`](super::CgroupDef::work_type)
    /// shape at the Op layer so test authors composing mid-step
    /// spawns get the same one-liner ergonomics as authors composing
    /// CgroupDefs upfront.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// Op::spawn_workers(
    ///     cgroup,
    ///     WorkSpec { work_type, ..WorkSpec::default() },
    /// )
    /// ```
    ///
    /// For non-default knobs (worker count, affinity, …) construct
    /// a [`WorkSpec`] explicitly and route through
    /// [`Self::spawn`] — the sugar is intentionally minimal so a
    /// non-default knob forces the explicit-WorkSpec call site.
    pub fn spawn_in_cgroup(
        cgroup: impl Into<Cow<'static, str>>,
        work_type: WorkType,
    ) -> Self {
        Op::SpawnWorkers {
            cgroup: cgroup.into(),
            work: WorkSpec {
                work_type,
                ..WorkSpec::default()
            },
        }
    }

    /// Stop all workers in a cgroup.
    pub fn stop_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::StopCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set worker affinity in a cgroup.
    pub fn set_affinity(cgroup: impl Into<Cow<'static, str>>, affinity: AffinityIntent) -> Self {
        Op::SetAffinity {
            cgroup: cgroup.into(),
            affinity,
        }
    }

    /// Spawn workers in the parent cgroup.
    pub const fn spawn_host(work: WorkSpec) -> Self {
        Op::SpawnHost { work }
    }

    /// Move all tasks from one cgroup to another.
    pub fn move_all_tasks(
        from: impl Into<Cow<'static, str>>,
        to: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::MoveAllTasks {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Spawn a [`Payload`](crate::test_support::Payload) binary in the
    /// background. `args` is appended to `payload.default_args`.
    /// Placement is inherited from the caller; use
    /// [`run_payload_in_cgroup`](Self::run_payload_in_cgroup) to put
    /// the child into a named cgroup.
    pub fn run_payload(payload: &'static crate::test_support::Payload, args: Vec<String>) -> Self {
        Op::RunPayload {
            payload,
            args,
            cgroup: None,
        }
    }

    /// Spawn a [`Payload`](crate::test_support::Payload) in the
    /// background and place the child in a cgroup (relative to the
    /// scenario's parent cgroup).
    pub fn run_payload_in_cgroup(
        payload: &'static crate::test_support::Payload,
        args: Vec<String>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::RunPayload {
            payload,
            args,
            cgroup: Some(cgroup.into()),
        }
    }

    /// Block until the payload named `name` exits, evaluate checks,
    /// and record metrics. Matches whichever cgroup the payload is
    /// in when exactly one copy of the name is live; bails when two
    /// or more copies are live (use
    /// [`wait_payload_in_cgroup`](Self::wait_payload_in_cgroup) to
    /// disambiguate).
    pub fn wait_payload(name: impl Into<Cow<'static, str>>) -> Self {
        Op::WaitPayload {
            name: name.into(),
            cgroup: None,
        }
    }

    /// Block until the payload named `name` that's running inside
    /// the given `cgroup` exits. Use this form when two or more
    /// copies of the same payload are live in different cgroups
    /// and a cgroup-less `wait_payload` would be ambiguous. An
    /// empty-string `cgroup` matches payloads that inherited their
    /// parent's placement (spawned via `Op::run_payload(..., cgroup:
    /// None)`); explicit names match payloads placed via
    /// [`Op::run_payload_in_cgroup`] or
    /// [`CgroupDef::workload`](crate::scenario::ops::CgroupDef::workload).
    pub fn wait_payload_in_cgroup(
        name: impl Into<Cow<'static, str>>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::WaitPayload {
            name: name.into(),
            cgroup: Some(cgroup.into()),
        }
    }

    /// SIGKILL the payload named `name`, evaluate checks, and record
    /// metrics. Matches the unique live copy by name; bails on
    /// ambiguity. See [`wait_payload`](Self::wait_payload) for the
    /// full ambiguity rules and
    /// [`kill_payload_in_cgroup`](Self::kill_payload_in_cgroup)
    /// for the disambiguating form.
    pub fn kill_payload(name: impl Into<Cow<'static, str>>) -> Self {
        Op::KillPayload {
            name: name.into(),
            cgroup: None,
        }
    }

    /// SIGKILL the payload named `name` that's running inside the
    /// given `cgroup`. See
    /// [`wait_payload_in_cgroup`](Self::wait_payload_in_cgroup) for
    /// the placement-matching contract.
    pub fn kill_payload_in_cgroup(
        name: impl Into<Cow<'static, str>>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::KillPayload {
            name: name.into(),
            cgroup: Some(cgroup.into()),
        }
    }

    /// Freeze every task in a cgroup via `cgroup.freeze`.
    pub fn freeze_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::FreezeCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Unfreeze every task in a cgroup via `cgroup.freeze`.
    pub fn unfreeze_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::UnfreezeCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Capture a host-side diagnostic snapshot under `name`. See
    /// [`Op::CaptureSnapshot`] for the full request/reply protocol and
    /// no-bridge fallback semantics.
    pub fn capture_snapshot(name: impl Into<Cow<'static, str>>) -> Self {
        Op::CaptureSnapshot { name: name.into() }
    }

    /// Register a write-driven snapshot watch on `symbol`. See
    /// [`Op::WatchSnapshot`] for the symbol-resolution rules and
    /// guard rails (max 3 watches per scenario, verbatim vmlinux
    /// ELF symtab match, 4-byte alignment requirement).
    pub fn watch_snapshot(symbol: impl Into<Cow<'static, str>>) -> Self {
        Op::WatchSnapshot {
            symbol: symbol.into(),
        }
    }

    /// Live-vCPU write of a single (target, value) pair. Singleton
    /// convenience that wraps the pair into the
    /// [`Op::WriteKernelHot`] batch shape. See the variant doc for
    /// the live-vCPU orchestration contract and the
    /// caller-side-synchronisation requirement.
    ///
    /// **Two consecutive singleton calls produce two separate Ops**
    /// — not auto-merged at construction time. For dispatching
    /// multiple hot writes as a single op, use
    /// [`Self::write_kernel_hot_batch`]. The executor's
    /// adjacent-op auto-merge (which would collapse N adjacent
    /// singleton hot writes into one dispatch) is queued as a
    /// dedicated follow-up task; until it lands, each
    /// `write_kernel_hot` call is its own dispatch.
    pub fn write_kernel_hot(target: KernelTarget, value: KernelValue) -> Self {
        Op::WriteKernelHot {
            writes: vec![(target, value)],
        }
    }

    /// Live-vCPU batched write. See [`Op::WriteKernelHot`] for the
    /// live-vCPU orchestration contract. The batch is issued in
    /// iteration order.
    pub fn write_kernel_hot_batch(
        writes: impl IntoIterator<Item = (KernelTarget, KernelValue)>,
    ) -> Self {
        Op::WriteKernelHot {
            writes: writes.into_iter().collect(),
        }
    }

    /// Auto-freezing write of a single (target, value) pair.
    /// Singleton convenience that wraps the pair into the
    /// [`Op::WriteKernelCold`] batch shape. See the variant doc for
    /// the rendezvous-and-batched-writes contract and the
    /// no-inter-CPU-skew guarantee.
    ///
    /// **Two consecutive singleton calls produce two separate Ops**
    /// — two freeze rendezvous cycles + observable inter-CPU skew
    /// between them. For correct multi-CPU seeding always use
    /// [`Self::write_kernel_cold_batch`]. The executor's
    /// adjacent-op auto-merge (which would collapse N adjacent
    /// singleton cold writes into one rendezvous) is queued as a
    /// dedicated follow-up task; until it lands, callers needing
    /// inter-CPU-coherent writes MUST batch explicitly.
    pub fn write_kernel_cold(target: KernelTarget, value: KernelValue) -> Self {
        Op::WriteKernelCold {
            writes: vec![(target, value)],
        }
    }

    /// Auto-freezing batched write. See [`Op::WriteKernelCold`] for
    /// the freeze-rendezvous-and-batched-writes contract. All
    /// writes in the batch land within a single freeze rendezvous —
    /// no inter-CPU skew from N separate freeze cycles.
    pub fn write_kernel_cold_batch(
        writes: impl IntoIterator<Item = (KernelTarget, KernelValue)>,
    ) -> Self {
        Op::WriteKernelCold {
            writes: writes.into_iter().collect(),
        }
    }

    /// Live-vCPU read of `target` into the snapshot bridge keyed by
    /// `tag`. See [`Op::ReadKernelHot`] for the live-vCPU
    /// orchestration contract and the read-vs-guest-write race
    /// caveat.
    ///
    /// Reads are singleton-only: each read produces one bridge
    /// entry keyed by `tag`. A batched read (multi-target single op)
    /// is a future surface — it would need either Vec<(tag, target)>
    /// with parallel result slots or a HashMap result, both
    /// distinct contracts from the write batch's "do N writes".
    /// For now, dispatch N reads as N separate ops.
    pub fn read_kernel_hot(tag: impl Into<Cow<'static, str>>, target: KernelTarget) -> Self {
        Op::ReadKernelHot {
            tag: tag.into(),
            target,
        }
    }

    /// Auto-freezing read of `target` into the snapshot bridge keyed
    /// by `tag`. See [`Op::ReadKernelCold`] for the
    /// rendezvous-coherent-read contract and
    /// [`Self::read_kernel_hot`] for the singleton-only rationale.
    ///
    /// Until the adjacent-cold-op auto-merge (queued as a dedicated
    /// follow-up task) lands, each `read_kernel_cold` triggers its
    /// own freeze rendezvous. Where multiple cold reads are needed
    /// within the same coherent snapshot, prefer
    /// [`Op::CaptureSnapshot`] (which already orchestrates a single
    /// rendezvous for all snapshot reads).
    pub fn read_kernel_cold(tag: impl Into<Cow<'static, str>>, target: KernelTarget) -> Self {
        Op::ReadKernelCold {
            tag: tag.into(),
            target,
        }
    }
}
