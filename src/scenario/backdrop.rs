//! Persistent scenario state that lives across every Step in a
//! `#[ktstr_test]` run.
//!
//! Tests usually express "a scheduler is under load for N seconds"
//! as a Step sequence. Some tests also want entities that persist
//! for the WHOLE run — a long-running binary payload, a synthetic
//! workload that spans the whole scenario, a cgroup whose identity
//! is referenced by multiple Steps. Those go in a [`Backdrop`].
//!
//! # Step vs Backdrop
//!
//! - A [`Step`](super::ops::Step) is bounded: everything it creates
//!   (cgroups, workload handles, payload handles) is torn down when
//!   the step finishes. The runtime enforces this automatically —
//!   no explicit teardown op is required.
//! - A [`Backdrop`] is persistent: what it sets up lives for the
//!   entire Step sequence. Its cgroups are created once before the
//!   first Step and RAII-removed at scenario end; its payloads
//!   spawn once and are killed (with metric emission) after the
//!   last Step tears down.
//!
//! In the "bursty load + scheduler stress test" pattern:
//!
//! - The bursty payload (a persistent fio, a running stress-ng) is
//!   a `Backdrop::push_payload(...)` entry — it runs THROUGHOUT the
//!   test, irrespective of which Step is currently applying ops.
//! - Each Step handles a discrete phase ("settle", "inject
//!   contention", "measure") with its own CgroupDefs that come and
//!   go.
//!
//! Steps may reference Backdrop-owned cgroups by name through
//! cgroup-addressing ops (`Op::SetCpuset`, `Op::MoveAllTasks`, etc.)
//! — name lookups resolve step-local first, then fall through to
//! the Backdrop. Step-local cgroups must not shadow a Backdrop
//! cgroup name. Step-local `Op::RemoveCgroup` / `Op::StopCgroup`
//! targeting a Backdrop cgroup is permitted; later Steps that
//! reference the removed cgroup by name surface a kernel-layer
//! `cgroup missing` error rather than getting a Backdrop typo
//! caught early.

use super::ops::{CgroupDef, Op};
use crate::test_support::Payload;

/// Persistent state for a Step sequence.
///
/// Hold long-running entities here instead of re-declaring them in
/// every Step. [`execute_scenario`](super::ops::execute_scenario)
/// owns the Backdrop for the duration of the run, sets up every
/// declared entity once before the first Step, and tears them down
/// at the end (success or Err).
///
/// # Empty default
///
/// Scenarios with no persistent state pass [`Backdrop::new()`],
/// which is also what the shorthand
/// [`execute_steps`](super::ops::execute_steps) /
/// [`execute_defs`](super::ops::execute_defs) wrappers forward to
/// internally. There is no cost to using the empty default — the
/// runtime skips the Backdrop setup phase entirely when every vec
/// is empty.
///
/// # `Clone` and cgroup-name collisions
///
/// `Backdrop` derives [`Clone`], so a test can copy a base Backdrop
/// and attach the copies to different scenarios. **Do not pass a
/// cloned Backdrop into a sibling scenario in the same process /
/// VM without rewriting the cgroup names first.** Every cgroup in
/// `cgroups` is created at the same path
/// (`/sys/fs/cgroup/<parent>/<name>`); two scenarios both calling
/// `setup` on a Backdrop with the same names silently share the
/// cgroup's tasks and counters — the second `setup` finds the path
/// already exists, skips `mkdir`, and attaches its workers alongside
/// the first scenario's. No `EEXIST` surfaces (the kernel-level
/// `mkdir(2)` race is absorbed by `std::fs::create_dir_all`), so
/// diagnose by unexpected `cgroup.procs` task counts or doubled
/// metric counters rather than a returned error. A typical safe
/// shape is
/// `base.clone().rename_cgroups(|n| format!("{n}_{idx}"))` (caller-
/// provided helper) before attaching to scenario `idx`. The clone
/// derive is provided for builder-style composition (forking a
/// base, then conditionally appending entries) where the resulting
/// Backdrop is attached to ONE scenario — sibling-scenario use
/// requires the rename pass.
///
/// # Example
///
/// ```no_run
/// use ktstr::prelude::*;
///
/// #[derive(Payload)]
/// #[payload(binary = "stress-ng")]
/// #[default_args("--cpu", "2")]
/// struct BgLoadPayload;
///
/// // Worker-bearing cgroup + empty move target + long-running payload,
/// // all persistent for the scenario.
/// let backdrop = Backdrop::new()
///     .push_cgroup(CgroupDef::named("bg_cell").cpuset(CpusetSpec::disjoint(0, 2)))
///     .push_op(Op::add_cgroup("bg_overflow"))
///     .push_payload(&BG_LOAD);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Backdrop {
    /// Long-lived cgroups created once and removed at scenario end.
    /// Any Step can reference them by name via `Op::MoveAllTasks`,
    /// `Op::SetCpuset`, etc. Every [`CgroupDef`] here spawns at
    /// least one worker (declared [`WorkSpec`](crate::workload::WorkSpec)
    /// entries, or a single default WorkSpec when `works` is empty).
    /// Declare empty move-target cgroups via [`Self::ops`] /
    /// [`Self::push_op`] using [`Op::AddCgroup`] instead.
    ///
    /// # Ordering guarantee
    ///
    /// Cgroups are created in DECLARATION ORDER — the order they
    /// appear in this `Vec`. The Backdrop setup phase iterates
    /// `cgroups` front-to-back and runs each [`CgroupDef`](crate::scenario::ops::CgroupDef)'s setup
    /// (`mkdir`, cpuset/sysfs writes, worker spawn) one at a time.
    /// `push_cgroup(a).push_cgroup(b)` creates `a` first, then `b`.
    ///
    /// This matters for any scheduler whose internal IDs are
    /// assigned in cgroup-creation order — `scx_mitosis`, for
    /// example, allocates `cell_id`s monotonically on cgroup
    /// creation (observed via cgroup-fs inotify, not on first
    /// task attach) and reuses freed IDs LIFO from a free-list
    /// when prior cells were destroyed. The cgroup declared first
    /// gets `cell_id = 1`, the second gets `cell_id = 2`, and so
    /// on for the initial allocation sequence. A test that wants a
    /// sparse `cell_id` range (e.g. remove the middle cell to leave
    /// a gap) can rely on the framework-side declaration order:
    /// declare `cg_a`, `cg_b`, `cg_c` to get `cell_id = 1, 2, 3`,
    /// then `Op::RemoveCgroup("cg_b")` at a Step boundary leaves
    /// cells 1 and 3 live with a `cell_id = 2` hole. The next
    /// single-cgroup allocation after that hole reuses the freed
    /// `cell_id = 2` before bumping `next_cell_id` further — LIFO
    /// from the free-list, not lowest-free.
    ///
    /// **Multi-delete caveat.** When several cgroups are removed
    /// in one inotify-batched event (or two `RemoveCgroup` ops fire
    /// before scx_mitosis services any of them), scx_mitosis
    /// inserts the freed IDs into the free-list via
    /// `HashSet::iter()` — hash-bucket order, NOT removal order.
    /// Subsequent allocations still pop LIFO, but the LIFO is
    /// against an arbitrarily-permuted insertion order, so
    /// "remove a, b, c in this order → reuse c, b, a" does NOT
    /// hold for multi-cell batches. Single-delete patterns
    /// (one `RemoveCgroup` per Step) reuse the freed ID
    /// deterministically.
    ///
    /// The cell_id assignment itself is the scheduler's
    /// responsibility, not the framework's. The Backdrop only
    /// guarantees the cgroup-creation order; the scheduler binary
    /// observes the resulting creation order and assigns whatever
    /// internal IDs its policy dictates.
    pub cgroups: Vec<CgroupDef>,
    /// Long-lived binary payloads spawned once before the first
    /// Step. The runtime holds the live handles for the duration of
    /// the Step sequence and drains them via `.kill()` (preserving
    /// metric emission) at scenario teardown.
    pub payloads: Vec<&'static Payload>,
    /// Raw [`Op`]s applied during Backdrop setup, before any Step
    /// runs. Run AFTER [`Self::cgroups`] apply_setup and BEFORE
    /// [`Self::payloads`] spawn, in declaration order. Backdrop
    /// ops run with full authority — they can target Backdrop
    /// cgroups with [`Op::RemoveCgroup`] / [`Op::StopCgroup`] /
    /// [`Op::MoveAllTasks`] where step-local ops would be
    /// rejected, since the Backdrop owns the cgroups it's setting
    /// up. Any cgroup / handle / payload these ops create is
    /// tracked by the Backdrop slot and tears down at scenario
    /// end. The typical use is [`Op::AddCgroup`] for empty
    /// move-target cgroups (a [`CgroupDef`] can't express the
    /// zero-worker case because apply_setup forces a worker
    /// spawn).
    pub ops: Vec<Op>,
}

impl Backdrop {
    /// Fresh empty Backdrop — no persistent state. Builder entry
    /// point: chain [`Self::push_cgroup`] / [`Self::push_payload`] /
    /// [`Self::push_op`] to populate.
    ///
    /// Equivalent to [`Default::default`](Self::default), but
    /// `const fn` so the value is usable in `static`/`const`
    /// contexts (`Default::default` is not yet const-stable). Prefer
    /// `Backdrop::new()` at construction sites; `..Default::default()`
    /// remains available inside non-const struct-update expressions.
    #[must_use = "dropping a Backdrop discards the scenario layout"]
    pub const fn new() -> Self {
        Backdrop {
            cgroups: Vec::new(),
            payloads: Vec::new(),
            ops: Vec::new(),
        }
    }

    /// See [`Self::cgroups`] for the ordering guarantee and the
    /// `cell_id` allocation example.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn push_cgroup(mut self, def: CgroupDef) -> Self {
        self.cgroups.push(def);
        self
    }

    /// See [`Self::cgroups`] for the ordering guarantee.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn extend_cgroups<I: IntoIterator<Item = CgroupDef>>(mut self, defs: I) -> Self {
        self.cgroups.extend(defs);
        self
    }

    /// Binary-kind payload with no extra args. See [`Self::payloads`]
    /// for lifecycle. For custom args or cgroup placement use
    /// [`Self::push_op`] with [`Op::run_payload`] / [`Op::run_payload_in_cgroup`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn push_payload(mut self, payload: &'static Payload) -> Self {
        self.payloads.push(payload);
        self
    }

    /// See [`Self::push_payload`]; use [`Self::extend_ops`] with
    /// [`Op::run_payload`] entries for per-payload args.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn extend_payloads<I: IntoIterator<Item = &'static Payload>>(
        mut self,
        payloads: I,
    ) -> Self {
        self.payloads.extend(payloads);
        self
    }

    /// See [`Self::ops`] for run order. Typical use:
    /// [`Op::AddCgroup`] for empty move-target cgroups (a
    /// [`CgroupDef`] always spawns at least one worker).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn push_op(mut self, op: Op) -> Self {
        self.ops.push(op);
        self
    }

    /// See [`Self::ops`] for run order.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn extend_ops<I: IntoIterator<Item = Op>>(mut self, ops: I) -> Self {
        self.ops.extend(ops);
        self
    }

    /// True when the Backdrop has no persistent entities declared.
    /// `execute_scenario` checks this to skip the Backdrop setup
    /// phase entirely — zero overhead for scenarios that do not
    /// use persistent state.
    pub fn is_empty(&self) -> bool {
        self.cgroups.is_empty() && self.payloads.is_empty() && self.ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PAYLOAD: Payload = Payload::binary("test_bin", "/bin/true");

    /// Distinct-named sibling of `TEST_PAYLOAD` so payload-order
    /// tests can discriminate position via the name field. Without
    /// a second const, two-element tests reusing `TEST_PAYLOAD`
    /// produce `[test_bin, test_bin]` regardless of order — a
    /// `reverse()` regression would silently pass any name-based
    /// assertion.
    const TEST_PAYLOAD_2: Payload = Payload::binary("test_bin_2", "/bin/false");

    /// Additional distinct-named payloads so
    /// `extend_payloads_preserves_declaration_order_for_many_entries`
    /// can exercise a 7-distinct-payload pattern matching the
    /// cgroups sibling test — any pairwise swap of any two indices
    /// surfaces as a name mismatch at the swapped index.
    const TEST_PAYLOAD_3: Payload = Payload::binary("test_bin_3", "/bin/false");
    const TEST_PAYLOAD_4: Payload = Payload::binary("test_bin_4", "/bin/false");
    const TEST_PAYLOAD_5: Payload = Payload::binary("test_bin_5", "/bin/false");
    const TEST_PAYLOAD_6: Payload = Payload::binary("test_bin_6", "/bin/false");
    const TEST_PAYLOAD_7: Payload = Payload::binary("test_bin_7", "/bin/false");

    #[test]
    fn empty_backdrop_has_no_entities() {
        let b = Backdrop::new();
        assert!(b.cgroups.is_empty());
        assert!(b.payloads.is_empty());
        assert!(b.ops.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn new_returns_empty() {
        let b = Backdrop::new();
        assert!(b.is_empty());
    }

    #[test]
    fn push_cgroup_populates_cgroups() {
        let b = Backdrop::new().push_cgroup(CgroupDef::named("cg0"));
        assert_eq!(b.cgroups.len(), 1);
        assert_eq!(b.cgroups[0].name.as_ref(), "cg0");
        assert!(!b.is_empty());
    }

    #[test]
    fn extend_cgroups_preserves_input_order() {
        let b = Backdrop::new().extend_cgroups([
            CgroupDef::named("cg0"),
            CgroupDef::named("cg1"),
            CgroupDef::named("cg2"),
        ]);
        assert_eq!(b.cgroups.len(), 3);
        assert_eq!(b.cgroups[0].name.as_ref(), "cg0");
        // Middle index pinned so a regression that swaps middle
        // elements via a sort or partition while leaving first/last
        // in place gets caught.
        assert_eq!(b.cgroups[1].name.as_ref(), "cg1");
        assert_eq!(b.cgroups[2].name.as_ref(), "cg2");
    }

    /// Declaration order is preserved across a many-entry batch.
    /// Catches sort/partition/reverse-style regressions that the
    /// 3-element sibling above wouldn't surface — a swap of indices
    /// 1 and 2 there is invisible to a {0, len-1} assertion, but
    /// here every index is checked.
    ///
    /// Names are deliberately non-monotonic in ASCII order so a
    /// regression that stably-sorts the input would produce a
    /// different sequence and trip the per-index assertion. If
    /// names were `["cg0", "cg1", ..., "cg6"]` (already sorted),
    /// a sort_by_name regression would be invisible.
    #[test]
    fn extend_cgroups_preserves_declaration_order_for_many_entries() {
        const NAMES: [&str; 7] = ["cg5", "cg0", "cg3", "cg6", "cg1", "cg4", "cg2"];
        let b = Backdrop::new().extend_cgroups(NAMES.map(CgroupDef::named));
        assert_eq!(b.cgroups.len(), NAMES.len());
        for (i, expected) in NAMES.iter().enumerate() {
            assert_eq!(
                b.cgroups[i].name.as_ref(),
                *expected,
                "index {i} should be {expected}"
            );
        }
    }

    #[test]
    fn push_payload_populates_payloads() {
        let b = Backdrop::new().push_payload(&TEST_PAYLOAD);
        assert_eq!(b.payloads.len(), 1);
        assert_eq!(b.payloads[0].name, "test_bin");
        assert!(!b.is_empty());
    }

    #[test]
    fn extend_payloads_preserves_order() {
        // Use TEST_PAYLOAD_2 in the second slot so the assertion
        // actually discriminates position — `[&TEST_PAYLOAD,
        // &TEST_PAYLOAD]` would assert `[test_bin, test_bin]`
        // regardless of order and let a `reverse()` regression pass.
        let b = Backdrop::new().extend_payloads([&TEST_PAYLOAD, &TEST_PAYLOAD_2]);
        assert_eq!(b.payloads.len(), 2);
        assert_eq!(b.payloads[0].name, "test_bin");
        assert_eq!(b.payloads[1].name, "test_bin_2");
    }

    /// Declaration order is preserved across a many-entry payload
    /// batch. Sibling of `extend_cgroups_preserves_declaration_order_for_many_entries`
    /// — uses 7 fully-distinct payloads so any pairwise swap (including
    /// non-adjacent same-name swaps that an interleaved 2-distinct-payload
    /// pattern would miss) surfaces as a name mismatch at the swapped
    /// index. Catches arbitrary shuffle/sort/partition/reverse
    /// regressions across a larger collection than the 2-entry
    /// `extend_payloads_preserves_order` can surface.
    ///
    /// Inputs are deliberately non-monotonic in ASCII order of their
    /// `name` field so a regression that stably-sorts the input would
    /// produce a different sequence and trip the per-index assertion.
    #[test]
    fn extend_payloads_preserves_declaration_order_for_many_entries() {
        let inputs = [
            &TEST_PAYLOAD_5,
            &TEST_PAYLOAD,
            &TEST_PAYLOAD_3,
            &TEST_PAYLOAD_7,
            &TEST_PAYLOAD_2,
            &TEST_PAYLOAD_6,
            &TEST_PAYLOAD_4,
        ];
        let expected = [
            "test_bin_5",
            "test_bin",
            "test_bin_3",
            "test_bin_7",
            "test_bin_2",
            "test_bin_6",
            "test_bin_4",
        ];
        let b = Backdrop::new().extend_payloads(inputs);
        assert_eq!(b.payloads.len(), expected.len());
        for (i, name) in expected.iter().enumerate() {
            assert_eq!(b.payloads[i].name, *name, "index {i} should be {name}");
        }
    }

    #[test]
    fn push_then_extend_preserves_order() {
        // Use distinct payload consts so the assertion can verify
        // that the `push_payload` entry comes BEFORE the
        // `extend_payloads` entries — a regression that prepends
        // (instead of appends) would pass a count-only check.
        let b = Backdrop::new()
            .push_payload(&TEST_PAYLOAD)
            .extend_payloads([&TEST_PAYLOAD_2]);
        assert_eq!(b.payloads.len(), 2);
        assert_eq!(b.payloads[0].name, "test_bin");
        assert_eq!(b.payloads[1].name, "test_bin_2");
    }

    #[test]
    fn chain_builds_in_order() {
        let b = Backdrop::new()
            .push_cgroup(CgroupDef::named("cg_a"))
            .push_payload(&TEST_PAYLOAD)
            .push_cgroup(CgroupDef::named("cg_b"));
        assert_eq!(b.cgroups.len(), 2);
        assert_eq!(b.cgroups[0].name.as_ref(), "cg_a");
        assert_eq!(b.cgroups[1].name.as_ref(), "cg_b");
        assert_eq!(b.payloads.len(), 1);
        assert!(!b.is_empty());
    }

    #[test]
    fn default_impl_matches_new() {
        let d: Backdrop = Default::default();
        assert!(d.is_empty());
        assert_eq!(d.cgroups.len(), Backdrop::new().cgroups.len());
        assert_eq!(d.payloads.len(), Backdrop::new().payloads.len());
        assert_eq!(d.ops.len(), Backdrop::new().ops.len());
    }

    /// Compile-time pin: [`Backdrop::new`] must remain `const fn` so
    /// downstream consumers can use it in `static`/`const` item
    /// initializers. A regression that drops the `const` keyword (or
    /// adds a non-const operation to the body) breaks this `const _`
    /// item at build time rather than silently slipping past
    /// [`default_impl_matches_new`].
    const _BACKDROP_NEW_IS_CONST_EVALUABLE: Backdrop = Backdrop::new();

    #[test]
    fn push_op_populates_ops() {
        let b = Backdrop::new().push_op(Op::add_cgroup("empty_target"));
        assert_eq!(b.ops.len(), 1);
        assert!(matches!(&b.ops[0], Op::AddCgroup { name } if name.as_ref() == "empty_target"));
        assert!(!b.is_empty());
    }

    #[test]
    fn extend_ops_preserves_order() {
        let b =
            Backdrop::new().extend_ops(vec![Op::add_cgroup("cg_1"), Op::add_cgroup("cg_1/sub")]);
        assert_eq!(b.ops.len(), 2);
        assert!(matches!(&b.ops[0], Op::AddCgroup { name } if name.as_ref() == "cg_1"));
        assert!(matches!(&b.ops[1], Op::AddCgroup { name } if name.as_ref() == "cg_1/sub"));
    }

    #[test]
    fn chain_push_op_interleaves_with_other_builders() {
        let b = Backdrop::new()
            .push_cgroup(CgroupDef::named("cg_workers"))
            .push_op(Op::add_cgroup("cg_empty"))
            .push_payload(&TEST_PAYLOAD);
        assert_eq!(b.cgroups.len(), 1);
        assert_eq!(b.ops.len(), 1);
        assert_eq!(b.payloads.len(), 1);
        assert!(!b.is_empty());
    }

    /// `Backdrop::clone()` must produce an independent value: mutating
    /// the clone leaves the original untouched. The default derived
    /// `Clone` on a struct of `Vec<T>` fields produces deep copies,
    /// but a future refactor that swaps any field to an `Rc`-shared
    /// or `Cow`-shared container would silently turn the clone into
    /// an alias — and the cgroup-name-collision footgun the type
    /// docs warn about would expand into a "mutate one Backdrop,
    /// corrupt the other" surprise. Pin independence per field
    /// (cgroups, payloads, ops) against [`Backdrop::new`],
    /// AND verify the cloned vec received the pushed value (not just
    /// "1 element"). A regression that cloned `original.cgroups` as
    /// `vec![CgroupDef::named("wrong")]` and then pushed the
    /// expected value would yield len=2, which a length-only
    /// assertion misses.
    #[test]
    fn clone_is_independent_per_field() {
        let original = Backdrop::new();
        let mut cloned = original.clone();
        cloned.cgroups.push(CgroupDef::named("cg_added_to_clone"));
        cloned.payloads.push(&TEST_PAYLOAD);
        cloned.ops.push(Op::add_cgroup("cg_op_on_clone"));
        assert!(
            original.cgroups.is_empty(),
            "original.cgroups must stay empty after clone mutation"
        );
        assert!(
            original.payloads.is_empty(),
            "original.payloads must stay empty after clone mutation"
        );
        assert!(
            original.ops.is_empty(),
            "original.ops must stay empty after clone mutation"
        );
        // Verify clone has exactly the pushed values — catches a
        // "started with a non-empty clone" bug that length=1 alone
        // would miss.
        assert_eq!(cloned.cgroups.len(), 1, "clone cgroups: unexpected count");
        assert_eq!(
            cloned.cgroups[0].name.as_ref(),
            "cg_added_to_clone",
            "clone cgroups: pushed value not present at index 0"
        );
        assert_eq!(cloned.payloads.len(), 1, "clone payloads: unexpected count");
        assert!(
            std::ptr::eq(cloned.payloads[0], &TEST_PAYLOAD),
            "clone payloads: pushed pointer not present at index 0"
        );
        assert_eq!(cloned.ops.len(), 1, "clone ops: unexpected count");
        assert!(
            matches!(&cloned.ops[0], Op::AddCgroup { name } if name.as_ref() == "cg_op_on_clone"),
            "clone ops: pushed Op not present at index 0"
        );
    }

    /// Lock-step pin: [`Backdrop::default`] must agree with
    /// [`Backdrop::new`] field-by-field. Both produce a no-state
    /// builder seed; a regression where Default seeds a non-empty
    /// field would silently introduce a phantom cgroup/payload/op
    /// into every spread-default callsite.
    #[test]
    fn default_matches_new() {
        let from_new = Backdrop::new();
        let from_trait: Backdrop = Default::default();
        assert_eq!(
            from_trait.cgroups.len(),
            from_new.cgroups.len(),
            "cgroups Vec drift"
        );
        assert_eq!(
            from_trait.payloads.len(),
            from_new.payloads.len(),
            "payloads Vec drift"
        );
        assert_eq!(from_trait.ops.len(), from_new.ops.len(), "ops Vec drift");
        assert!(from_new.cgroups.is_empty());
        assert!(from_new.payloads.is_empty());
        assert!(from_new.ops.is_empty());
    }
}
