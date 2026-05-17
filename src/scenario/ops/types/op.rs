//! `Op` operation taxonomy + `CpusetSpec` enum with its constructor
//! impl block. Owns the variant-set (and `OpKind` discriminator
//! enum) plus the cpuset-spec construction surface. Resolution-time
//! `CpusetSpec` logic lives in a sibling impl block in
//! [`super::resolve`] — Rust permits multiple impl blocks across
//! files in the same crate; the split tracks the construction /
//! resolution responsibility boundary.
//!
//! See the parent module ([`super`]) for the file-layout overview
//! and the cross-impl-block convention.

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::workload::{AffinityIntent, WorkSpec};

use super::CgroupDef;

/// Atomic operation on the cgroup topology.
///
/// Names use `Cow<'static, str>` so ops can reference compile-time
/// literals (zero-cost) or runtime-generated strings (owned).
///
/// # `#[non_exhaustive]`
///
/// `Op` is `#[non_exhaustive]` — see [`crate::non_exhaustive`] for
/// the cross-crate pattern-match rule. `Op`-specific construction
/// convention: prefer the per-op constructors (e.g. `Op::add_cgroup`,
/// `Op::run_payload`) over naming variants directly; new
/// constructors are added alongside new variants and are the stable
/// surface.
#[derive(Clone, Debug, strum::EnumDiscriminants)]
#[strum_discriminants(name(OpKind))]
#[strum_discriminants(derive(strum::EnumIter))]
#[strum_discriminants(vis(pub))]
#[non_exhaustive]
pub enum Op {
    /// Create a new cgroup under the managed cgroup parent, with no
    /// cpuset, no controller knobs, and no workers — the
    /// operator-friendly way to declare an empty move-target cgroup
    /// that later receives tasks via [`Op::MoveAllTasks`] or
    /// similar. For mid-step cgroups that need cpuset / cpu /
    /// memory / io / pids / workers, use [`Op::add_cgroup_def`]
    /// instead; for setup-time cgroups with the same knobs, declare
    /// via [`Step::with_defs`].
    AddCgroup { name: Cow<'static, str> },
    /// Create a cgroup mid-step from a full [`CgroupDef`] — cpuset,
    /// cpu/memory/io/pids knobs, and worker spawns all apply in one
    /// op, mirroring the way `Step::with_defs` materializes a
    /// step-local CgroupDef at setup time. Use this when the
    /// add-cgroup-with-cpuset-and-workers sequence needs to happen
    /// after the step's setup pass (e.g. driven by an earlier op's
    /// observed state) instead of as part of the step's setup. The
    /// embedded `def` is dedup-checked the same way `apply_setup`
    /// rejects collisions with prior Backdrop or step-local
    /// CgroupDef declarations.
    AddCgroupDef { def: CgroupDef },
    /// Remove a cgroup (stops its workers first). Permitted against
    /// both step-local and Backdrop-owned cgroups; removing a
    /// Backdrop cgroup mid-scenario drops it from the Backdrop
    /// tracking list so a later `Op::AddCgroup` with the same name
    /// can re-create the cgroup. A typo'd cgroup name surfaces
    /// later as a kernel-layer "cgroup missing" error on the next
    /// op that references the name, not at the RemoveCgroup site.
    RemoveCgroup { cgroup: Cow<'static, str> },
    /// Set a cgroup's cpuset to the resolved CPU set.
    SetCpuset {
        cgroup: Cow<'static, str>,
        cpus: CpusetSpec,
    },
    /// Clear a cgroup's cpuset (allow all CPUs).
    ClearCpuset { cgroup: Cow<'static, str> },
    /// Read both cgroups' cpusets and swap them.
    SwapCpusets {
        a: Cow<'static, str>,
        b: Cow<'static, str>,
    },
    /// Spawn workers and move them into the target cgroup.
    ///
    /// The work type is used as-is; gauntlet `work_type_override` does
    /// not apply. Use [`CgroupDef`] with `swappable(true)` when the
    /// work type should be overridable.
    Spawn {
        cgroup: Cow<'static, str>,
        work: WorkSpec,
    },
    /// Stop all workers in a cgroup (does not remove the cgroup).
    /// Permitted against both step-local and Backdrop-owned cgroups;
    /// stopping a Backdrop cgroup's workers mid-scenario leaves the
    /// cgroup hierarchy intact but makes subsequent ops that expect
    /// those workers (e.g. wait/kill payload) fail to find them.
    StopCgroup { cgroup: Cow<'static, str> },
    /// Set worker affinity in a cgroup. Resolved at apply time via
    /// `resolve_affinity_for_cgroup()`.
    SetAffinity {
        cgroup: Cow<'static, str>,
        affinity: AffinityIntent,
    },
    /// Spawn workers in the parent cgroup (not in a managed cgroup).
    ///
    /// `WorkSpec` is resolved to a `WorkloadConfig` at apply time, matching
    /// the resolution pattern used by `Op::Spawn`.
    SpawnHost { work: WorkSpec },
    /// Move all tasks from one cgroup to another.
    ///
    /// Each task is moved via `cgroup.procs`. If any move fails, the
    /// error propagates and handle name keys are left unchanged (workers
    /// remain addressed under `from`). On success, handle name keys are
    /// updated to `to` so subsequent ops address the moved workers.
    ///
    /// # Lifetime / ownership-direction asymmetry
    ///
    /// `MoveAllTasks` is asymmetric with respect to cgroup ownership:
    /// the legality of a move depends on the relative lifetimes of
    /// the `from` and `to` cgroups, not just on which one is the
    /// source.
    ///
    /// | `from` ownership      | `to` ownership        | Outcome |
    /// |-----------------------|-----------------------|---------|
    /// | step-local            | step-local            | Allowed; both die at step teardown together. |
    /// | step-local            | Backdrop (persistent) | Allowed; handle ownership transfers from step-local set to Backdrop set so the worker survives step teardown. |
    /// | Backdrop              | Backdrop              | Allowed; both persist for the scenario. |
    /// | Backdrop              | step-local            | **Rejected at apply time.** A persistent worker would be stranded inside a cgroup that gets `rmdir`'d at step boundary; the kernel migrates the orphaned task to the cgroup root with a frozen-task warning in dmesg. The `bail!` diagnostic names the offending pair and tells the operator to either declare the destination in the Backdrop too, or move the worker back into a Backdrop-owned cgroup. |
    ///
    /// The Backdrop→Backdrop and step→step cases are unconditionally
    /// allowed because both endpoints share a lifetime; the
    /// step→Backdrop case is allowed because the kernel moves
    /// reference-count once and the framework's
    /// `ScenarioState::rename_handles`
    /// transfers the handle into the persistent slot in the same
    /// step. The Backdrop→step case is the only one that produces
    /// a guaranteed orphan, hence the asymmetric reject.
    ///
    /// # Backdrop-setup exemption
    ///
    /// `MoveAllTasks` ops running INSIDE a Backdrop's `setup_ops`
    /// pass (`state.target_backdrop=true`) are exempt from the
    /// Backdrop→step-local check: at that point, "step-local"
    /// cgroups don't exist yet (the Backdrop is the only cgroup
    /// scope), and the rule reduces to a pure source-ownership
    /// check that the apply path handles already.
    MoveAllTasks {
        from: Cow<'static, str>,
        to: Cow<'static, str>,
    },
    /// Spawn a userspace [`Payload`](crate::test_support::Payload)
    /// binary in the background and track its
    /// [`PayloadHandle`](crate::scenario::payload_run::PayloadHandle)
    /// under the step's payload-handle set.
    ///
    /// Subsequent [`Op::WaitPayload`] / [`Op::KillPayload`] address
    /// the running child by the composite
    /// (`Payload::name`, `cgroup`) key — the same payload can run
    /// concurrently in two different cgroups without a dedup
    /// collision, but the lookup from the waiting op must match
    /// the pair the run op recorded. See [`Op::WaitPayload`] /
    /// [`Op::KillPayload`] for the ambiguity rules when the
    /// waiting op supplies only the name.
    ///
    /// Only [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are spawnable; scheduler-kind payloads are rejected at
    /// apply time with an actionable error.
    ///
    /// `args` is appended to `payload.default_args`. `cgroup`, when
    /// set, places the child in the named cgroup (resolved relative
    /// to the scenario's parent cgroup) via
    /// [`PayloadRun::in_cgroup`](crate::scenario::payload_run::PayloadRun::in_cgroup);
    /// unset inherits the spawning process's cgroup.
    ///
    /// Handles not explicitly consumed by `WaitPayload` / `KillPayload`
    /// are drained at step-teardown by `collect_step` (step-local) or
    /// at scenario end by `collect_backdrop` (when the handle lives on
    /// the Backdrop), matching the [`CgroupDef::workload`] semantics.
    ///
    /// # Scheduler-kind rejection across surfaces
    ///
    /// Three surfaces accept a `&Payload` and each rejects a
    /// scheduler-kind Payload differently — deliberately, to match
    /// the lifecycle of the caller:
    ///
    /// | Surface                                                                                   | Rejection             | When          |
    /// |-------------------------------------------------------------------------------------------|-----------------------|---------------|
    /// | [`PayloadRun::run`](crate::scenario::payload_run::PayloadRun::run) (`ctx.payload(&X)...`) | `Err(anyhow::Error)`  | scenario-time |
    /// | [`CgroupDef::workload`]                                                                   | `panic!`              | declaration-time |
    /// | `Op::RunPayload` (this variant)                                                           | `Err(anyhow::Error)`  | apply-ops-time |
    ///
    /// Rationale: `CgroupDef::workload` is a builder invoked during
    /// test construction (nextest `--list` phase) — a panic there
    /// surfaces the misuse before any VM boot, with a full
    /// backtrace pointing at the offending call. `ctx.payload()`
    /// and `Op::RunPayload` both run inside an executing scenario
    /// where one bad misuse should not crash the whole test run;
    /// they `bail!` with an actionable message and let the
    /// surrounding step-sequence skip to teardown. The three
    /// paths are symmetric in *what* they reject (scheduler-kind
    /// Payloads in non-scheduler slots); they differ only in
    /// *how* the misuse is surfaced, matched to caller context.
    RunPayload {
        payload: &'static crate::test_support::Payload,
        args: Vec<String>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// Block until the payload named `name` exits naturally, then
    /// evaluate its checks and record metrics to the per-test sidecar.
    ///
    /// The target is looked up by composite key (`name`, `cgroup`).
    /// `cgroup: None` matches the unique live copy (whatever its
    /// placement); if two or more copies of the same payload are
    /// live in different cgroups, the lookup bails with an
    /// "ambiguous — specify cgroup" error so the test doesn't
    /// silently wait on the wrong one. Use
    /// [`Op::wait_payload_in_cgroup`] to disambiguate.
    ///
    /// A consumed or unknown `(name, cgroup)` pair returns `Err`
    /// with an actionable message — test authors must not silently
    /// wait for payloads that were never started or have already
    /// been consumed by a prior `WaitPayload`/`KillPayload`.
    ///
    /// **No timeout.** `WaitPayload` waits indefinitely for the
    /// child to exit. A binary that never terminates (e.g. a
    /// benchmark configured without `--runtime=N`, or a stress-ng
    /// run without `--timeout`) will hang the step until the
    /// outer test watchdog fires. For time-boxed long-running
    /// payloads, prefer [`KillPayload`](Self::KillPayload) paired
    /// with a [`HoldSpec::fixed`] / [`HoldSpec::frac`] step
    /// boundary that guarantees forward progress; the payload's
    /// own CLI (`--runtime`, `--timeout`) is the reliable way to
    /// cap a single invocation's runtime.
    ///
    /// Check failures from the payload are recorded to the sidecar
    /// for regression analysis but do NOT fail the step or the test
    /// in-process. Use
    /// [`ctx.payload(&X).run()`](crate::scenario::payload_run::PayloadRun::run)
    /// directly if the test body needs to gate on check results.
    WaitPayload {
        name: Cow<'static, str>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// SIGKILL the payload named `name`, reap the child, evaluate
    /// checks, and record metrics. Mirrors the behavior of
    /// step-teardown drain for an explicitly-targeted payload.
    ///
    /// The target is looked up by composite key (`name`, `cgroup`)
    /// — see [`Op::WaitPayload`] for the ambiguity rules.
    ///
    /// A consumed or unknown `(name, cgroup)` pair returns `Err`
    /// with an actionable message, identical to [`Op::WaitPayload`]'s
    /// lookup semantics.
    ///
    /// Check failures from the payload are recorded to the sidecar
    /// for regression analysis but do NOT fail the step or the test
    /// in-process. Use
    /// [`ctx.payload(&X).run()`](crate::scenario::payload_run::PayloadRun::run)
    /// directly if the test body needs to gate on check results.
    KillPayload {
        name: Cow<'static, str>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// Freeze every task in the named cgroup via `cgroup.freeze`.
    ///
    /// Writes `"1"` to the cgroup's `cgroup.freeze` file. The kernel's
    /// `cgroup_freeze_write` dispatches the asynchronous freeze path;
    /// tasks transition to the frozen state without external SIGSTOP,
    /// and `cgroup.events` reaches `frozen 1` once every task has
    /// parked. Idempotent — freezing an already-frozen cgroup is a
    /// no-op.
    ///
    /// # Auto-unfreeze at teardown
    ///
    /// `Op::FreezeCgroup` is paired with [`Op::UnfreezeCgroup`] to
    /// release. A test that omits the unfreeze still tears down
    /// cleanly: [`crate::cgroup::CgroupManager::remove_cgroup`]
    /// auto-unfreezes the cgroup before draining tasks (see the
    /// kernel's `cgroup_freezer_migrate_task`, which clears the
    /// task's freeze state when it migrates to an unfrozen
    /// destination), so step teardown is robust to a stuck-frozen
    /// cgroup. Pair the ops explicitly when the scenario needs
    /// observable unfreeze timing inside the step body.
    ///
    /// # Worked example
    ///
    /// Three-Step suspend/resume sequence: a `Backdrop`-resident
    /// long-running workload is paused mid-scenario and resumed
    /// later, exercising how the scheduler responds to a sudden
    /// idle window.
    ///
    /// ```text
    /// Step 1 (run): apply cgroup; workload spins for 2s.
    /// Step 2 (suspend): Op::freeze_cgroup("workers"); hold 1s.
    ///                   The cgroup's tasks park via cgroup.freeze,
    ///                   schedstat gauges drop to zero, and the
    ///                   scheduler observes a sudden idle subtree.
    /// Step 3 (resume): Op::unfreeze_cgroup("workers"); hold 2s.
    ///                  Tasks return to runnable state, the
    ///                  scheduler must re-pick them onto the
    ///                  cgroup's CPUs without spuriously preempting
    ///                  unrelated workloads.
    /// ```
    ///
    /// # Observer-cgroup deadlock warning
    ///
    /// Do NOT freeze a cgroup that hosts the test's own observation
    /// machinery. The freeze path stops every task in the cgroup —
    /// including any thread that:
    /// - opens `/proc/<pid>/sched` or other procfs entries owned by
    ///   tasks inside the frozen cgroup, then waits on the read,
    /// - holds a futex shared with frozen tasks (the unfreeze must
    ///   land before the wait can complete),
    /// - synchronously waits on a stalled-task pipe whose
    ///   producer is in the frozen cgroup.
    ///
    /// The framework's stimulus-event SHM ring and the `BlkWorker`
    /// epoll loop both run outside the test cgroup tree, so they
    /// are unaffected — but a test author who explicitly places an
    /// observer thread inside the same cgroup as its observation
    /// targets will deadlock the scenario when the freeze fires.
    /// Place observers in a sibling cgroup (or in the parent) so
    /// `cgroup.freeze` is scoped to the workload subtree alone.
    ///
    /// Pair with [`Op::UnfreezeCgroup`] to release. Useful for
    /// scheduler suspend/resume tests where the test body wants to
    /// observe how the scheduler handles a suddenly-frozen workload
    /// and the resumption sequence afterwards.
    ///
    /// Treats a missing cgroup as a step failure: the
    /// `cgroup.freeze` write fails with `ENOENT` and the error
    /// propagates via the `apply_ops` `with_context` chain.
    /// Freezing a non-existent cgroup is NOT a no-op; only
    /// freezing an already-frozen cgroup is.
    FreezeCgroup { cgroup: Cow<'static, str> },
    /// Unfreeze every task in the named cgroup via `cgroup.freeze`.
    ///
    /// Writes `"0"` to the cgroup's `cgroup.freeze` file. Inverse of
    /// [`Op::FreezeCgroup`]. Idempotent.
    UnfreezeCgroup { cgroup: Cow<'static, str> },
    /// Capture a host-side diagnostic snapshot under `name`. The
    /// freeze coordinator pauses every vCPU long enough to read
    /// the BPF map state, vCPU registers, and per-CPU
    /// counters into a
    /// [`FailureDumpReport`](crate::monitor::dump::FailureDumpReport),
    /// then resumes the guest. The report is keyed by `name` on
    /// the active
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge);
    /// downstream test code reads it via
    /// [`Snapshot`](crate::scenario::snapshot::Snapshot).
    ///
    /// On-demand snapshots are orthogonal to the error-class
    /// freeze trigger — the request flows through a separate
    /// channel, does not transition the coordinator's
    /// `freeze_state`, and is serviced even after `Done`. The only
    /// scheduling rule: at most one capture in flight at a time
    /// (each request waits for the previous freeze's vCPUs to
    /// fully resume before issuing).
    ///
    /// **Guest → host wire.** Locked at an in-kernel ioeventfd
    /// doorbell at a dedicated MMIO GPA inside the MMIO gap
    /// (e.g. `MMIO_GAP_START + 0x3000`). The guest writes the tag
    /// into a small SHM-resident slot and then writes the doorbell
    /// GPA via the existing `/dev/mem` mmap pattern that the SHM
    /// ring already uses. KVM dispatches in-kernel
    /// (`KVM_IOEVENTFD`) without a vCPU userspace exit, the
    /// freeze coordinator wakes on `eventfd_signal`, and the
    /// installed `CaptureCallback` returns the resulting report
    /// through a paired reply completion. See
    /// [`CaptureCallback`](crate::scenario::snapshot::CaptureCallback)
    /// for the full protocol.
    ///
    /// **No active bridge ⇒ no-op.** When the executor runs in a
    /// context with no installed
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// (e.g. unit tests that exercise the executor without
    /// spinning up a VM), this op emits a `tracing::warn!` and
    /// continues. Existing scenarios that never declare snapshot
    /// ops keep their behavior unchanged.
    ///
    /// # Example
    ///
    /// Declare a snapshot mid-step, fetch the captured report
    /// after the scenario completes, and assert against a
    /// BTF-rendered field:
    ///
    /// ```ignore
    /// use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
    /// use ktstr::scenario::snapshot::{Snapshot, SnapshotBridge};
    ///
    /// // Wire up the bridge before execute_steps runs (host-side
    /// // VM setup typically performs this step automatically).
    /// let bridge = SnapshotBridge::new(/* capture callback */);
    /// let _guard = bridge.clone().set_thread_local();
    ///
    /// let steps = vec![Step {
    ///     setup: vec![CgroupDef::named("workers").workers(2)].into(),
    ///     ops: vec![Op::snapshot("after_spawn")],
    ///     hold: HoldSpec::FULL,
    /// }];
    /// execute_steps(ctx, steps)?;
    ///
    /// // Inspection.
    /// let captured = bridge.drain();
    /// let report = captured.get("after_spawn").expect("snapshot recorded");
    /// let snap = Snapshot::new(report);
    /// let nr_cpus = snap.var("nr_cpus_onln").as_u64()?;
    /// assert!(nr_cpus > 0, "snapshot captured live nr_cpus_onln");
    /// ```
    Snapshot { name: Cow<'static, str> },
    /// Capture a snapshot whenever the guest writes to the named
    /// kernel symbol. The snapshot is tagged with the symbol
    /// itself; one fire = one capture.
    ///
    /// Symbol resolution at op execution time is a verbatim match
    /// against the vmlinux ELF symbol table: the freeze coordinator
    /// walks `Elf::syms` and accepts the symbol whose strtab entry
    /// equals the requested string byte-for-byte. There is no
    /// prefix stripping, BTF lookup, kallsyms walk, or per-CPU
    /// offset arithmetic — the string must match an entry that
    /// `nm vmlinux` would print (e.g. `"jiffies_64"`,
    /// `"scx_watchdog_timestamp"`).
    ///
    /// The `register_watch` callback on a host-side
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// is for **host-side unit testing only** — it lets in-process
    /// executor tests record the symbol and return without arming
    /// any hardware. Production in-VM scenarios run via the
    /// virtio-console port 1 `MSG_TYPE_SNAPSHOT_REQUEST` TLV frame
    /// and the host coordinator's `arm_user_watchpoint` path
    /// (`src/vmm/freeze_coord.rs`); the thread-local bridge is
    /// never installed inside the guest.
    ///
    /// # Guard rails
    ///
    /// - **Maximum of 3 watch ops per scenario.** The KVM
    ///   hardware-watchpoint plumbing reserves slot 0 for the
    ///   existing `*scx_root->exit_kind` trigger (used by the
    ///   error-trigger path); only the remaining three user
    ///   watchpoint slots are available for on-demand watches. The
    ///   bridge's `register_watch` rejects a 4th
    ///   `Op::WatchSnapshot` and fails the step when the cap is
    ///   exceeded.
    /// - **Symbol resolution failures bail immediately.** A
    ///   missing symbol or unaligned address surfaces as an `Err`
    ///   from `execute_steps` so the test author notices the
    ///   watch did not attach. Silent degradation would leave the
    ///   scenario running with no captures and look identical to
    ///   a healthy passing run.
    /// - **4-byte alignment.** The resolved KVA must be 4-byte
    ///   aligned: the framework arms 4-byte data-write watches,
    ///   which require `addr & 0x3 == 0` on every supported
    ///   architecture. Mis-aligned addresses bail at setup with
    ///   the resolved KVA in the error.
    ///
    /// **Guest → host wire.** The registration request rides the
    /// same ioeventfd doorbell as [`Op::Snapshot`] (separate tag
    /// namespace), so symbol resolution + user watchpoint slot
    /// allocation + `KVM_SET_GUEST_DEBUG` arming happen on the host
    /// without a vCPU userspace exit. Once armed, the
    /// `KVM_EXIT_DEBUG` dispatch path drives the resulting
    /// captures directly into the freeze coordinator (no
    /// per-fire doorbell write needed). See
    /// [`WatchRegisterCallback`](crate::scenario::snapshot::WatchRegisterCallback)
    /// for the full protocol.
    ///
    /// Note: high-frequency variables (rq counters, jiffies)
    /// will fire watches every few microseconds and fire
    /// thousands of times (each overwriting the prior capture
    /// under the same tag); the framework does not rate-limit
    /// captures, so the test author owns the frequency choice.
    /// Use [`Op::Snapshot`] for time-driven captures when
    /// frequency is the concern.
    WatchSnapshot { symbol: Cow<'static, str> },
}

/// How to compute a cpuset from topology.
///
/// # `#[non_exhaustive]`
///
/// `CpusetSpec` is `#[non_exhaustive]` — see
/// [`crate::non_exhaustive`] for the cross-crate pattern-match and
/// construction rules shared by every such type.
///
/// Variant-specific guidance for `CpusetSpec`: prefer the
/// associated constructor functions — [`Self::llc`], [`Self::numa`],
/// [`Self::range`], [`Self::disjoint`], [`Self::overlap`], and
/// [`Self::exact`] — over naming variant literals like
/// `CpusetSpec::Llc(0)` or `CpusetSpec::Range { start_frac,
/// end_frac }`. Two reasons:
///
/// 1. **Stability across variant reshaping.** A future commit that
///    adds a field to `Range` (e.g. a stride parameter) breaks every
///    caller that spelled out `CpusetSpec::Range { start_frac,
///    end_frac }`; the `Self::range(..)` constructor absorbs the
///    new field behind a defaulted parameter. The `#[non_exhaustive]`
///    attribute is what reserves that freedom for the enum; the
///    constructor convention is how callers opt into benefiting from
///    it.
/// 2. **Semantic consistency with [`Self::exact`].** The `exact`
///    constructor accepts any `IntoIterator<Item = usize>` (arrays,
///    ranges, `Vec`, `BTreeSet`) and converts to `BTreeSet<usize>`
///    internally; callers that bypass it and write
///    `CpusetSpec::Exact(set)` directly must hand-build the
///    `BTreeSet` — duplicate bookkeeping a future-proofed constructor
///    erases.
///
/// Test code that needs to *inspect* a variant via pattern match
/// necessarily references the variant literal (the name is load-
/// bearing for the match), so the construction-side rule is a
/// convention for *production* call sites, not a hard constraint.
/// Inside this crate, matchers obey the pattern-side rule above;
/// constructors obey this rule.
///
/// `Clone + Debug + PartialEq`. `Eq` / `Hash` are impossible
/// because [`Range`](Self::Range) and [`Overlap`](Self::Overlap)
/// carry `f64` fractions; `Default` has no honest value (`Llc(0)`
/// vs. `Range(0..1)` vs. `Exact(empty)` are all different
/// "no-op" semantics).
///
/// Note: `f64::NAN != f64::NAN` per IEEE 754, so a `CpusetSpec`
/// containing NaN fractions will not equal a clone of itself;
/// `validate()` rejects NaN inputs.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CpusetSpec {
    /// All CPUs in a given LLC index.
    Llc(usize),
    /// All CPUs in a given NUMA node index.
    Numa(usize),
    /// Fractional range of usable CPUs [start_frac..end_frac).
    Range { start_frac: f64, end_frac: f64 },
    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    Disjoint { index: usize, of: usize },
    /// Like Disjoint but each set overlaps neighbors by `frac` of its size.
    Overlap { index: usize, of: usize, frac: f64 },
    /// Exact CPU set (no topology resolution).
    Exact(BTreeSet<usize>),
}

impl CpusetSpec {
    /// Construct an `Exact` cpuset from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        CpusetSpec::Exact(cpus.into_iter().collect())
    }

    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    pub const fn disjoint(index: usize, of: usize) -> Self {
        CpusetSpec::Disjoint { index, of }
    }

    /// Like [`disjoint`](Self::disjoint) but each set overlaps neighbors by `frac` of its size.
    pub const fn overlap(index: usize, of: usize, frac: f64) -> Self {
        CpusetSpec::Overlap { index, of, frac }
    }

    /// Fractional range of usable CPUs `[start_frac..end_frac)`.
    pub const fn range(start_frac: f64, end_frac: f64) -> Self {
        CpusetSpec::Range {
            start_frac,
            end_frac,
        }
    }

    /// All CPUs in a given LLC index.
    pub const fn llc(index: usize) -> Self {
        CpusetSpec::Llc(index)
    }

    /// All CPUs in a given NUMA node index.
    pub const fn numa(index: usize) -> Self {
        CpusetSpec::Numa(index)
    }
}
