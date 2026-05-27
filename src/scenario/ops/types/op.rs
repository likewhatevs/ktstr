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
    /// via [`super::super::Step::with_defs`].
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
    /// Spawn workers and place them according to `placement`.
    ///
    /// The work type is used as-is; gauntlet `work_type_override` does
    /// not apply. Use [`CgroupDef`] with `swappable(true)` when the
    /// work type should be overridable.
    ///
    /// Placement contract (bullets follow [`SpawnPlacement`] variant
    /// declaration order):
    ///   * [`SpawnPlacement::RunnerCgroup`] — spawn workers in the
    ///     spawner's own cgroup; the handler issues ZERO cgroup ops
    ///     and the workers inherit whatever cgroup the test runner
    ///     sits in. `WorkSpec::workers_pct` is rejected for this
    ///     placement because there's no managed cgroup whose cpuset
    ///     would supply the percentage denominator.
    ///   * [`SpawnPlacement::Cgroup`] — spawn workers and move them
    ///     into the named cgroup; the cgroup must already exist
    ///     (declared via [`CgroupDef`] in `Step.setup`, via
    ///     [`Op::AddCgroup`] / [`Op::AddCgroupDef`] earlier in the
    ///     same step, or on the persistent
    ///     [`Backdrop`](crate::scenario::Backdrop)).
    Spawn {
        placement: SpawnPlacement,
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
    /// Move all tasks from one cgroup to another.
    ///
    /// Each task is moved via `cgroup.procs`. If any move fails, the
    /// error propagates and handle name keys are left unchanged (workers
    /// remain addressed under `from`). On success, handle name keys are
    /// updated to `to` so subsequent ops address the moved workers.
    ///
    /// # Self-move rejection
    ///
    /// A self-move (`from == to`) is rejected at handler entry — the
    /// kernel cgroup.procs write is idempotent on same-cgroup targets
    /// so the op would silently no-op, masking either a stale op the
    /// test author forgot to remove or a typo. The bail names both
    /// sides so the operator can pick the right fix. The check also
    /// catches the symmetric empty-string pair (`("", "")`), which
    /// would otherwise no-op a RunnerCgroup-to-RunnerCgroup transfer.
    ///
    /// # Empty-string source
    ///
    /// Passing `from = ""` matches workers spawned by
    /// [`Op::Spawn`] with [`SpawnPlacement::RunnerCgroup`] —
    /// RunnerCgroup-placement handles are tracked under the
    /// empty-string key (workers stay in the spawner's own cgroup,
    /// outside any managed hierarchy). `Op::move_all_tasks("",
    /// "named")` is the canonical way to materialize
    /// RunnerCgroup-placement workers into a managed cgroup
    /// mid-scenario; after the move the captured handles re-key
    /// to `"named"` and lose their empty-string identity,
    /// behaving like any other managed worker (lifetime tied to
    /// `"named"`'s ownership slot per the table below).
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
    /// with a [`super::super::HoldSpec::fixed`] / [`super::super::HoldSpec::frac`] step
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
    ///     ops: vec![Op::capture_snapshot("after_spawn")],
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
    CaptureSnapshot { name: Cow<'static, str> },
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
    /// - **Silent-misfire detection (KASLR-on guests).** When the
    ///   host coordinator's `kaslr_offset` is zero AND the
    ///   resolved kernel symbol lives in the x86_64 high-half
    ///   address range, `arm_user_watchpoint` emits a
    ///   `tracing::warn!` (once per unique `(symbol, link_kva)`
    ///   per process) noting the arm targets the link-time KVA
    ///   while the runtime symbol lives at `link_kva +
    ///   runtime_kaslr_slide`. The arm STILL completes (rejecting
    ///   it would regress every caller running before the host
    ///   coordinator's runtime-KASLR-slide derivation lands);
    ///   operators who hit the warn can boot the guest with the
    ///   `nokaslr` cmdline to use `Op::WatchSnapshot`, or omit
    ///   the op from KASLR-on test runs entirely.
    ///
    /// **Guest → host wire.** The registration request rides the
    /// same ioeventfd doorbell as [`Op::CaptureSnapshot`] (separate tag
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
    /// Use [`Op::CaptureSnapshot`] for time-driven captures when
    /// frequency is the concern.
    WatchSnapshot { symbol: Cow<'static, str> },
    /// Live-vCPU write of one or more [`KernelTarget`] / [`KernelValue`]
    /// pairs into running guest memory. The host coordinator routes
    /// each pair to the appropriate `GuestKernel::write_*` helper
    /// (no freeze rendezvous, vCPUs keep executing). A Release fence
    /// is issued after the last write so a weakly-ordered guest's
    /// `smp_load_acquire` observes the bytes in write order — but
    /// concurrent guest readers can still race against in-flight
    /// stores, and the caller owns any guest-side synchronisation
    /// the test requires (`READ_ONCE` / `smp_load_acquire` on the
    /// target field).
    ///
    /// Same orchestration pattern as the existing
    /// `BpfMapAccessor::write_value` path: synchronous host-side
    /// memory mutation on a worker thread, no vCPU pause. Use this
    /// for scratch fields, debug flags, scx-ktstr-private state,
    /// and anything the guest reads with proper barriers.
    ///
    /// **Batch shape.** `writes` carries 1+ pairs; the executor
    /// issues them in order. For a single write the
    /// [`Op::write_kernel_hot`](#method.write_kernel_hot) singleton
    /// constructor wraps a 1-element vec.
    ///
    /// **Dispatch.** The executor's arm calls
    /// `dispatch_kernel_op_request` (`src/scenario/ops/mod.rs:3717`), which
    /// uses the in-process `SnapshotBridge` callback when one is
    /// installed (the test-fixture seam) and falls back to the
    /// virtio-console port-1 wire path (`MsgType::KernelOpRequest`)
    /// in-guest. The wire request is consumed by
    /// `dispatch_kernel_op_batch` (`src/vmm/freeze_coord/kernel_op_dispatch.rs`),
    /// invoked from the freeze coordinator's apply path.
    ///
    /// **See also.** [`KernelTarget`] — scroll to the
    /// "Semantic risk" section for the single source of truth
    /// on which scheduler-bookkeeping targets are safe vs
    /// silently load-bearing.
    WriteKernelHot {
        /// Ordered list of `(target, value)` pairs to write.
        writes: Vec<(KernelTarget, KernelValue)>,
    },
    /// Auto-freezing batched write of one or more
    /// [`KernelTarget`] / [`KernelValue`] pairs while every vCPU is
    /// parked at the freeze rendezvous. Reuses the same coordinator
    /// path that [`Op::CaptureSnapshot`] triggers: one rendezvous,
    /// every write in the batch lands while paused, then resume.
    ///
    /// **Batching is a hard correctness requirement.** Multi-CPU
    /// seeds (e.g. a planned `with_uptime` helper writing per-CPU
    /// `rq.clock` on every CPU at the same instant) must land in
    /// ONE freeze window —
    /// N separate cold-write ops would mean N rendezvous cycles
    /// and observable inter-CPU skew. The variant payload is a
    /// `Vec` precisely to make batched writes the natural shape.
    /// The executor's `apply_ops` pre-pass auto-merges adjacent
    /// singleton `Op::WriteKernelCold` ops into one merged op as
    /// a safety net — N adjacent `write_kernel_cold(...)` calls
    /// collapse into one rendezvous regardless of whether the
    /// caller used [`crate::scenario::ops::Op::write_kernel_cold_batch`]
    /// or chained singletons.
    ///
    /// **Dispatch.** The executor's arm calls
    /// `dispatch_kernel_op_request` (`src/scenario/ops/mod.rs:3717`), which
    /// uses the in-process `SnapshotBridge` callback when one is
    /// installed (the test-fixture seam) and falls back to the
    /// virtio-console port-1 wire path (`MsgType::KernelOpRequest`)
    /// in-guest. The wire request lands at the freeze coordinator's
    /// rendezvous boundary via
    /// `dispatch_kernel_op_batch` (`src/vmm/freeze_coord/kernel_op_dispatch.rs`).
    ///
    /// Use this for: multi-field atomic writes, all-CPUs-at-once
    /// seeding, one-shot setup that must complete before the guest
    /// observes any partial state. Use [`Op::WriteKernelHot`] when
    /// the guest is OK with live-write semantics + caller-side
    /// synchronisation.
    ///
    /// **See also.** [`KernelTarget`] — scroll to the
    /// "Semantic risk" section for the single source of truth
    /// on which scheduler-bookkeeping targets are safe vs
    /// silently load-bearing.
    WriteKernelCold {
        /// Ordered list of `(target, value)` pairs to write inside
        /// a single freeze rendezvous.
        writes: Vec<(KernelTarget, KernelValue)>,
    },
    /// Live-vCPU read of a [`KernelTarget`] into the
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// drain log keyed by `tag`. Mirrors [`Op::WriteKernelHot`]:
    /// no freeze rendezvous, host-side worker thread issues the
    /// read while the guest keeps executing. The caller assumes
    /// the read may race against guest writes; for read-write
    /// coherency pair the op with a guest-side `smp_store_release`
    /// on the target.
    ///
    /// Use this for: read-back of values previously written via
    /// [`Op::WriteKernelHot`], lightweight polling of single fields
    /// the test wants to observe without pausing the guest.
    ///
    /// **Width.** The `width` field picks which
    /// `crate::monitor::guest::GuestKernel` `read_*` family the
    /// host dispatcher invokes — `u32` / `u64` / `Bytes(len)`.
    /// The reply lands as a [`crate::vmm::wire::KernelOpValue`] of
    /// the matching shape in the bridge's drain log; a u32 field
    /// must be read with `KernelValueWidth::u32()` (a u64 read of
    /// a u32 field returns the field's bytes plus 4 adjacent
    /// bytes).
    ///
    /// **Dispatch.** Same bridge-first / wire-fallback model as
    /// [`Op::WriteKernelHot`]; the wire request is consumed by
    /// `dispatch_kernel_op_batch` (`src/vmm/freeze_coord/kernel_op_dispatch.rs`).
    ReadKernelHot {
        /// Bridge-keyed tag under which the read result lands.
        tag: Cow<'static, str>,
        /// Address to read.
        target: KernelTarget,
        /// Width specifier: picks the read family + the reply
        /// value shape.
        width: KernelValueWidth,
    },
    /// Auto-freezing read of a [`KernelTarget`] into the
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// drain log keyed by `tag`, taken while every vCPU is parked
    /// at the freeze rendezvous. Reuses the same coordinator path
    /// that [`Op::CaptureSnapshot`] triggers. Coherent with
    /// respect to guest state — no concurrent guest write can race
    /// against the read.
    ///
    /// Use this for: ground-truth reads that must reflect a stable
    /// guest state, snapshot-style point-in-time reads. Note: each
    /// `Op::ReadKernelCold` triggers its OWN freeze rendezvous —
    /// `apply_ops`'s pre-pass folds adjacent
    /// `Op::WriteKernelCold` ops into one rendezvous but does NOT
    /// fold reads (per-entry wire tags are needed for the
    /// multi-read reply-routing contract; queued as a wire-format
    /// follow-up). For multi-read coherent snapshots, prefer
    /// [`Op::CaptureSnapshot`] (which already orchestrates a single
    /// rendezvous for all snapshot reads).
    ///
    /// **Width.** Same `width` semantics as [`Op::ReadKernelHot`]:
    /// pick the read family explicitly so the dispatcher invokes
    /// the matching `GuestKernel::read_*` helper.
    ///
    /// **Dispatch.** Bridge-first / wire-fallback like the other
    /// `*Kernel*` variants; the wire request lands at the freeze
    /// coordinator's rendezvous boundary via
    /// `dispatch_kernel_op_batch` (`src/vmm/freeze_coord/kernel_op_dispatch.rs`).
    ReadKernelCold {
        /// Bridge-keyed tag under which the read result lands.
        tag: Cow<'static, str>,
        /// Address to read.
        target: KernelTarget,
        /// Width specifier: picks the read family + the reply
        /// value shape.
        width: KernelValueWidth,
    },
    /// Attach a scheduler mid-scenario: spawn the named staged
    /// scheduler from `/staging/schedulers/<name>/` inside the guest
    /// and wait for it to publish its first BPF object accessors.
    ///
    /// **Dispatch** (`dispatch_attach_scheduler` at
    /// `src/scenario/ops/mod.rs:3510`): waits up to 60s for the
    /// accessor-init worker to quiesce (handles the case where the
    /// boot scheduler's first publish is still in flight), captures
    /// the pre-spawn publish seqno, spawns the staged scheduler
    /// binary, re-installs the sched_exit_monitor against the new
    /// SCHED_PID, then waits up to 30s for a fresh accessor publish.
    ///
    /// **Already-attached behavior.** No framework-level idempotency
    /// guard: if a scheduler is already running, the kernel rejects
    /// the new attach at the `scx_enable_state() != SCX_DISABLED`
    /// gate (`kernel/sched/ext.c:6621`, returns `-EBUSY`); the
    /// spawned binary exits, no fresh publish lands, and the dispatch
    /// bails on the 30s publish-wait timeout. Use
    /// [`Op::DetachScheduler`] (then `AttachScheduler`) or
    /// [`Op::ReplaceScheduler`] to swap schedulers.
    ///
    /// The `scheduler` reference holds a `'static` lifetime: the
    /// test author declares each [`crate::test_support::Scheduler`]
    /// at static scope (via `declare_scheduler!` or a
    /// `static MY_SCHED: Scheduler = ...` item) and passes the
    /// borrow into the constructor. The staging slot that ships the
    /// binary into the initramfs is `KtstrTestEntry::staged_schedulers`;
    /// the dispatch arm reads its path via
    /// `test_support::staged::staged_scheduler_binary_path`.
    AttachScheduler {
        scheduler: &'static crate::test_support::Scheduler,
    },
    /// Detach the currently-running scheduler.
    ///
    /// **Dispatch** (`dispatch_detach_scheduler` →
    /// `kill_current_scheduler` at `src/scenario/ops/mod.rs:3386`):
    /// stops the host's sched_exit_monitor so the intentional kill
    /// isn't promoted into a test-fatal scheduler-died signal,
    /// writes `'S'` to `/proc/sysrq-trigger` to start the kernel-
    /// side `scx_disable` cascade asynchronously (avoiding the
    /// D-state stall inside `bpf_scx_unreg`'s
    /// `kthread_flush_work(&sch->disable_work)` at
    /// `kernel/sched/ext.c:7372-7381`), sends `SIGTERM` to the
    /// scheduler pid, waits up to `SCHED_LIFECYCLE_KILL_GRACE` (10s)
    /// for the kernel BPF state to reach `SCX_DISABLED`, then
    /// clears the `SCHED_PID` atomic (defined at
    /// `src/vmm/rust_init.rs:90`) so subsequent
    /// `crate::vmm::rust_init::sched_pid()` reads return `None`.
    ///
    /// Bails when no scheduler is currently attached (SCHED_PID is
    /// 0), when the SIGTERM syscall fails, or when the
    /// `SCX_DISABLED` wait times out. NOT idempotent: a second
    /// detach with no scheduler attached bails rather than no-oping.
    /// For defensive "ensure clean slate" scaffolds, gate on
    /// `crate::vmm::rust_init::sched_pid()` returning `Some` before
    /// emitting the Detach step rather than relying on no-op
    /// tolerance.
    DetachScheduler,
    /// Kill the currently-running scheduler and respawn the BOOT
    /// scheduler. Useful for hot-restart validation of the boot
    /// scheduler. Bails if no scheduler is currently attached.
    ///
    /// **v0 limitation.** Always respawns the boot scheduler at
    /// `/scheduler` + `/sched_args` regardless of which scheduler
    /// was most-recently attached — after an `Op::AttachScheduler`
    /// or `Op::ReplaceScheduler` to a staged scheduler, this op
    /// restarts the BOOT scheduler, not the most-recently-attached
    /// one. For restarting a staged scheduler, use
    /// [`Op::ReplaceScheduler`] with the same staged spec.
    ///
    /// **Dispatch** (`dispatch_restart_scheduler` at
    /// `src/scenario/ops/mod.rs:3589`): kills the current scheduler
    /// via the shared `kill_current_scheduler` helper, spawns the
    /// boot scheduler from the hardcoded `/scheduler` + `/sched_args`
    /// paths with log at `/tmp/sched.log`, then re-installs the
    /// sched_exit_monitor against the re-spawned boot pid.
    RestartScheduler,
    /// Detach the currently-running scheduler and attach a different
    /// one. Equivalent to `[DetachScheduler, AttachScheduler {
    /// scheduler: new }]` but expressed as a single op so the
    /// no-scheduler window is bounded and the per-phase scheduler
    /// tagging on the sidecar can record the transition atomically.
    ///
    /// The mid-experiment swap case the operator typically wants:
    /// run scheduler A for the first phase of a multi-step test, swap
    /// to scheduler B (or A-with-different-CLI-args, modeled as a
    /// distinct `Scheduler` declaration) for the second phase, and
    /// assert a per-phase metric delta across the boundary.
    ///
    /// Bails if no scheduler is currently attached — there is no
    /// scheduler to detach from, so the "replace" semantic has no
    /// meaning. Use [`Op::AttachScheduler`] for the first attach.
    ///
    /// **Dispatch** (`dispatch_replace_scheduler` at
    /// `src/scenario/ops/mod.rs:3616`): kills the current scheduler
    /// via the shared `kill_current_scheduler` helper, spawns the
    /// named staged scheduler binary from
    /// `/staging/schedulers/<name>/`, re-installs the
    /// sched_exit_monitor against the new SCHED_PID, waits up to
    /// `REPLACE_NOT_TRYING_DEADLINE_S` (5s) for the accessor-init
    /// worker to quiesce, captures the pre-publish seqno, then
    /// waits up to 10s for fresh accessors to publish against the
    /// new BPF object. The 10s budget aligns with
    /// `SCHED_LIFECYCLE_KILL_GRACE` and covers a cold-cache vmlinux
    /// re-parse during the worker reinit.
    ReplaceScheduler {
        scheduler: &'static crate::test_support::Scheduler,
    },
    /// Open a BPF map fd by name and hold it for the scenario lifetime.
    ///
    /// **Why this exists.** `Op::ReplaceScheduler` kills the outgoing
    /// scheduler process; libbpf's drop path then releases the map
    /// fds the loader was holding. Once the last refcount on a map
    /// drops, the kernel frees it — typically before any post-swap
    /// freeze captures, so the multi-bss "same-binary swap window"
    /// case (two `<obj>.bss` copies coexisting briefly) closes too
    /// fast to be reliably observed in a test. `PinBpfMap` holds an
    /// extra refcount on the named map so the kernel keeps it alive
    /// until the scenario ends.
    ///
    /// **Semantics.** Walks the kernel's map ID space (via
    /// [`libbpf_rs::query::MapInfoIter`], which wraps
    /// `BPF_MAP_GET_NEXT_ID` + `BPF_MAP_GET_FD_BY_ID` +
    /// `BPF_OBJ_GET_INFO_BY_FD`) and keeps the fd whose name matches.
    /// The held fd lives in the scenario's Backdrop state and drops
    /// (via std `OwnedFd` `Drop`) at scenario teardown. Multiple
    /// `PinBpfMap` ops with **distinct** names accumulate; pinning the
    /// **same** name twice is a no-op (the second call returns without
    /// re-opening the fd, so the originally-pinned map instance is the
    /// one held — not the second-call-time instance).
    ///
    /// **Name truncation.** BPF map names are capped at
    /// `BPF_OBJ_NAME_LEN = 16` bytes including the trailing NUL, so
    /// 15 usable chars max per `kernel/bpf/syscall.c`'s
    /// `bpf_obj_name_cpy`. Pass the kernel-visible name (typically
    /// `<obj>.bss` / `<obj>.data` / `<obj>.rodata`). When a libbpf
    /// object name + section suffix exceeds the 15-char cap, libbpf
    /// truncates the object prefix at load time and the kernel-side
    /// name is the truncated form; the framework does not auto-
    /// truncate the user-supplied string, so pass the post-truncation
    /// form. Reading the map names from a prior
    /// [`crate::monitor::dump::FailureDumpReport`]'s `maps[].name`
    /// or via `bpftool map list` is the safe way to discover the
    /// exact string the kernel sees.
    ///
    /// **Order.** Place this op AFTER the scheduler that owns the
    /// target map has attached (typically a small fixed hold suffices
    /// — ~100ms for the small scx-ktstr fixture, longer for
    /// heavyweight schedulers). For the same-binary swap-window
    /// scenario specifically: pin the **outgoing** scheduler's bss
    /// **before** `Op::ReplaceScheduler` runs — pinning after the
    /// swap is too late because the outgoing scheduler's bss has
    /// already been freed by libbpf's drop path. The pin walker
    /// picks the lowest-id matching map, so the outgoing copy (the
    /// older id) is the one held; the incoming scheduler's load
    /// then creates a second copy that's also kept alive because
    /// the outgoing refcount blocks the kernel from freeing the id.
    ///
    /// **Failure surface.** The pin runs at Step apply time inside
    /// `execute_steps` / `execute_scenario`. A failure (no matching
    /// map found in the walk) bails out of the apply path as an
    /// `Err` from `execute_steps`; the scenario stops before the
    /// next Step runs and the `post_vm` callback is not invoked.
    /// The underlying [`libbpf_rs::query::MapInfoIter`] silently
    /// terminates iteration on any non-`ENOENT` errno from the BPF
    /// ID walk (including `EPERM` from missing `CAP_SYS_ADMIN`), so
    /// such errors surface as the no-matching-map case rather than
    /// a distinct EPERM error — acceptable because ktstr always runs
    /// as root inside the guest, so the CAP_SYS_ADMIN gate at
    /// `kernel/bpf/syscall.c:4741` is always satisfied and the EPERM
    /// path is unreachable in practice.
    ///
    /// **Example.**
    /// ```ignore
    /// let steps = vec![
    ///     // Phase 0: primary scheduler runs alone; pin BEFORE the swap.
    ///     Step::with_op(
    ///         Op::pin_bpf_map("<obj>.bss"),
    ///         HoldSpec::frac(0.3),
    ///     ),
    ///     // Phase 1: swap to a same-binary alt — the pinned map
    ///     // keeps the OUTGOING bss alive across the teardown.
    ///     Step::with_op(
    ///         Op::replace_scheduler(&STAGED_ALT_SCHED),
    ///         HoldSpec::frac(0.7),
    ///     ),
    /// ];
    /// ```
    ///
    /// **See also.** [`crate::scenario::bpf_pin::open_bpf_map_fd_by_name`]
    /// for the underlying helper and `tests/live_var_disambiguation_e2e.rs`
    /// for the swap-window conditional walker-fired gate this pin is
    /// designed to make deterministic.
    PinBpfMap { name: Cow<'static, str> },
    /// Capture the current `cgroup.procs` of `cgroup` and store the
    /// PID list on the active [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// under `tag`.
    ///
    /// Synchronous read of the cgroup-v2 `cgroup.procs` pseudofile in
    /// the dispatching thread (in-scenario — runs wherever
    /// `execute_scenario` runs; inside the guest VM for `#[ktstr_test]`
    /// e2e tests, on the host for host-only scenarios). Returns the
    /// thread-group leaders (PIDs / TGIDs) the kernel reports at apply
    /// time. The snapshot is appended to the bridge's per-tag drain
    /// log; test bodies drain via
    /// [`SnapshotBridge::drain_cgroup_procs`](crate::scenario::snapshot::SnapshotBridge::drain_cgroup_procs)
    /// (or the by-tag lookup
    /// [`SnapshotBridge::cgroup_procs_by_tag`](crate::scenario::snapshot::SnapshotBridge::cgroup_procs_by_tag))
    /// after the scenario completes to read the captured pids back.
    ///
    /// Distinct from [`Op::CaptureSnapshot`]: that op routes through
    /// the host-side freeze coordinator (TLV transport in production,
    /// thread-local bridge in test fixtures); this op runs entirely
    /// in-process against the local cgroupfs.
    ///
    /// # Use cases
    ///
    /// Pin "did my workers land in cgroup X" assertions without the
    /// shell-probe + tmpfs-roundtrip pattern. Typical shape:
    ///
    /// ```ignore
    /// use ktstr::prelude::SnapshotBridge;
    /// use std::sync::Arc;
    ///
    /// // Install a bridge (dummy capture cb — only cgroup-procs drain
    /// // is used). MUST clone before set_thread_local, which consumes
    /// // self — the clone shares the Arc-internal state and is what
    /// // we drain on after the scenario completes.
    /// let bridge = SnapshotBridge::new(Arc::new(|_| None));
    /// let bridge_for_drain = bridge.clone();
    /// let _guard = bridge.set_thread_local();
    ///
    /// let backdrop = Backdrop::new().push_op(Op::add_cgroup("workers"));
    /// let steps = vec![
    ///     Step::new(
    ///         vec![
    ///             Op::spawn(SpawnPlacement::cgroup("workers"),
    ///                       WorkSpec::default().workers(4)),
    ///             Op::capture_cgroup_procs("after_spawn", "workers"),
    ///         ],
    ///         HoldSpec::fixed(Duration::ZERO),
    ///     ),
    /// ];
    /// let _ = execute_scenario(&ctx, backdrop, steps)?;
    ///
    /// // Either drain the whole log or look up by tag.
    /// let after = bridge_for_drain.cgroup_procs_by_tag("after_spawn")
    ///     .expect("Op::CaptureCgroupProcs(\"after_spawn\", ...) snapshot");
    /// assert_eq!(after.pids.len(), 4);
    /// ```
    ///
    /// # Within-Step ordering
    ///
    /// Ops in a single Step apply sequentially in vec order, so a
    /// `Op::CaptureCgroupProcs` placed AFTER `Op::Spawn` /
    /// `Op::MoveAllTasks` observes the post-spawn / post-migrate
    /// kernel state. The producing ops complete synchronously (their
    /// `cgroup.procs` writes block on kernel commit), so the capture
    /// sees every PID those ops placed.
    ///
    /// # PID vs TID grain
    ///
    /// Reads `cgroup.procs` (thread-group leaders), NOT `cgroup.threads`
    /// (per-thread TIDs). Grain implications by spawn op:
    ///
    /// - `Op::Spawn` → ktstr workers are 1-thread-per-worker, so
    ///   `workers(N)` produces `N` pids in `cgroup.procs`.
    /// - `Op::RunPayload` → an `execve`'d binary is ONE process; even
    ///   if the binary spawns 100 threads, `cgroup.procs` reports the
    ///   single thread-group leader. Tests asserting per-thread
    ///   placement would need a sibling `cgroup.threads` accessor
    ///   (future Op variant if a use case arises).
    ///
    /// # Tag uniqueness
    ///
    /// `tag` is the snapshot key the test body uses to find the
    /// capture in the drain log. The apply-ops dispatch rejects an
    /// empty `tag` with an actionable bail. Multiple captures of
    /// the same `cgroup` under DIFFERENT tags surface as separate
    /// entries (lets a scenario capture pre/post snapshots of the
    /// same cgroup); multiple captures with the same `(tag, cgroup)`
    /// also append rather than overwrite — tag uniqueness is a caller
    /// convention, not a framework-enforced contract. The by-tag
    /// lookup [`SnapshotBridge::cgroup_procs_by_tag`](crate::scenario::snapshot::SnapshotBridge::cgroup_procs_by_tag)
    /// returns the FIRST match; callers who care about multiplicity
    /// must use [`SnapshotBridge::drain_cgroup_procs`](crate::scenario::snapshot::SnapshotBridge::drain_cgroup_procs)
    /// and filter the Vec manually.
    ///
    /// # Empty / unknown cgroup
    ///
    /// - Empty cgroup (exists but holds no tasks): captured snapshot
    ///   has `pids = vec![]`. Lets callers assert "no tasks landed
    ///   here" without conflating with "no such cgroup."
    /// - Unknown cgroup (directory missing): apply bails with a
    ///   layered anyhow chain — the outer wrap names the op + tag +
    ///   cgroup; the inner [`crate::cgroup::CgroupOps::read_procs`]
    ///   context surfaces the resolved path + the actionable hint
    ///   about `Op::AddCgroup` / `workload_root_cgroup`. Use
    ///   `format!("{err:#}")` (alternate display) to flatten both
    ///   layers in test assertions.
    ///
    /// # See also
    ///
    /// - [`Op::CaptureSnapshot`] — diagnostic-snapshot capture (full
    ///   scheduler state dump via FailureDumpReport). Distinct from
    ///   this op's cgroup-procs read AND drains via a separate
    ///   `SnapshotBridge::drain` / `drain_ordered` channel, not
    ///   `drain_cgroup_procs`.
    /// - [`crate::cgroup::CgroupOps::read_procs`] — the underlying
    ///   trait method this op dispatches through.
    CaptureCgroupProcs {
        /// Snapshot key. Must be non-empty. Used by
        /// [`SnapshotBridge::drain_cgroup_procs`](crate::scenario::snapshot::SnapshotBridge::drain_cgroup_procs)
        /// consumers to find this capture in the drain log.
        tag: Cow<'static, str>,
        /// Cgroup to read `cgroup.procs` from. Must be a name
        /// already tracked by the scenario (created via
        /// `Op::AddCgroup`, a `CgroupDef` in setup, or pushed on
        /// the Backdrop). Must be non-empty.
        cgroup: Cow<'static, str>,
    },
}

/// Placement target for [`Op::Spawn`].
///
/// The previous taxonomy had two ops (`SpawnWorkers` and `SpawnHost`)
/// representing the two placement choices; the unified `Op::Spawn`
/// variant parameterises the placement so the framework has ONE
/// spawn op with the placement as data. `SpawnPlacement` is
/// `#[non_exhaustive]`; further placements are added here rather
/// than as new `Op` variants.
///
/// # `#[non_exhaustive]`
///
/// `SpawnPlacement` is `#[non_exhaustive]` — see
/// [`crate::non_exhaustive`] for the cross-crate pattern-match and
/// construction rules shared by every such type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SpawnPlacement {
    /// Spawn workers in the spawner's own cgroup — the test
    /// runner's cgroup, NOT any managed workload cgroup declared
    /// via [`CgroupDef`] or [`Op::AddCgroup`]. The handler issues
    /// ZERO cgroup ops; the workers inherit whatever cgroup the
    /// test runner sits in.
    ///
    /// Inside a guest VM the runner's cgroup is typically the
    /// root (cgid=1), so RunnerCgroup workers appear in snapshots
    /// under the root cgroup rather than under your workload's
    /// named hierarchy.
    ///
    /// `WorkSpec::workers_pct` is rejected for this placement —
    /// there's no managed cgroup whose cpuset would supply the
    /// percentage denominator. Use an explicit `.workers(N)`
    /// count, or switch to `Cgroup(name)` against a cgroup whose
    /// cpuset gives `workers_pct` a denominator.
    ///
    /// # Why "RunnerCgroup"?
    ///
    /// The previous shape used `SpawnHost` — "host" referred to
    /// the spawner's own cgroup (analogous to the
    /// scheduler-observability "host tasks vs workload tasks"
    /// distinction in sched_ext schedulers, e.g. mitosis's cell
    /// 0). `RunnerCgroup` names the placement target precisely
    /// (the test-runner process's cgroup) without the
    /// host-vs-guest-machine ambiguity that "host" carried.
    RunnerCgroup,
    /// Spawn workers and move them into the named managed
    /// cgroup. The cgroup must already exist when the spawn op
    /// applies — declared via [`CgroupDef`] in `Step.setup`,
    /// via [`Op::AddCgroup`] / [`Op::AddCgroupDef`] earlier in
    /// the same step, or on the persistent
    /// [`Backdrop`](crate::scenario::Backdrop).
    Cgroup(Cow<'static, str>),
}

impl SpawnPlacement {
    /// Construct [`SpawnPlacement::Cgroup`] from any string-like
    /// input (`&'static str`, `String`, `Cow<'static, str>`).
    /// Mirrors the [`impl Into<Cow<'static, str>>`] convention
    /// used by every other cgroup-name constructor on [`Op`]
    /// (`Op::add_cgroup`, `Op::spawn_workers`, `Op::move_all_tasks`,
    /// ...) so callers pass `"name"` not `"name".into()`.
    pub fn cgroup(name: impl Into<Cow<'static, str>>) -> Self {
        SpawnPlacement::Cgroup(name.into())
    }

    /// Construct [`SpawnPlacement::RunnerCgroup`]. Const so it
    /// composes inside `const` scenarios + builds.
    pub const fn runner_cgroup() -> Self {
        SpawnPlacement::RunnerCgroup
    }
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

/// Host-side write/read target for the kernel-memory ops
/// ([`Op::WriteKernelHot`] / [`Op::WriteKernelCold`] /
/// [`Op::ReadKernelHot`] / [`Op::ReadKernelCold`]).
///
/// Each variant names a kernel address by the resolution path the
/// host coordinator will take when the op fires; the actual
/// `GuestKernel` write helpers consume the resolved KVA. The variant
/// chosen here picks WHICH translation path (KASLR-aware kernel-image
/// base for [`Self::Symbol`], `PAGE_OFFSET` for [`Self::Direct`],
/// page-table walk for [`Self::Kva`], or per-CPU dereference for
/// [`Self::PerCpuField`]).
///
/// # Semantic risk — writing to load-bearing scheduler state
///
/// ktstr does not gate or filter target addresses. The framework
/// trusts the test author to know what they are pointing at. That
/// trust includes a class of fields where a raw write silently
/// breaks downstream kernel invariants the test author did not
/// intend to perturb. By design, mitigation is documentation-only:
/// the framework will not refuse a write nor emit a runtime warn —
/// the test author owns the choice. The cases to know about:
///
/// **Per-runqueue counters maintained by the scheduler classes.**
/// Raw writes skip the side-effects the kernel encodes in the
/// maintainer functions, leaving cross-class accounting in an
/// inconsistent state.
///
/// * **`struct rq.nr_running`** — the per-CPU runqueue task count.
///   `add_nr_running` / `sub_nr_running` (`kernel/sched/sched.h`)
///   also (a) fire the `sched_update_nr_running_tp` tracepoint and
///   (b) call `sched_update_tick_dependency(rq)` (the
///   `NOHZ_FULL` per-CPU tick gating logic); `add_nr_running`
///   additionally sets the root-domain `overloaded` bit
///   (`rq->rd->overloaded`) on the `prev_nr < 2 && new_nr >= 2`
///   transition. A bare 8-byte store skips all of those; the
///   counter and the root-domain overload signal diverge, the
///   NOHZ_FULL CPU may stop or start receiving ticks against the
///   test author's intent, and downstream load-balance decisions
///   read a count that no longer matches reality.
/// * **`struct cfs_rq.h_nr_runnable` / `h_nr_queued` /
///   `h_nr_idle`** (`kernel/sched/sched.h` `struct cfs_rq`) —
///   hierarchical CFS task counts maintained by
///   `account_entity_enqueue` / `dequeue` with cascade up the task
///   group tree. Raw write skips parent-cfs_rq propagation and
///   breaks group scheduling accounting.
/// * **`struct rt_rq.rt_nr_running`** (`kernel/sched/sched.h`
///   `struct rt_rq`) — RT class runqueue task count; updated by
///   `inc_rt_tasks` / `dec_rt_tasks` which also maintain the
///   per-rt_rq `overloaded` bit and the `highest_prio.curr/next`
///   priority-pushable tracking.
/// * **`struct dl_rq.dl_nr_running` / `running_bw` / `this_bw`**
///   (`kernel/sched/sched.h` `struct dl_rq`) — DEADLINE class
///   counters and bandwidth tracking; `add_running_bw` /
///   `sub_running_bw` (in `kernel/sched/deadline.c`) implement the
///   admission-control accounting that SUGOV's `cpu_bw_dl()`
///   consumes for frequency selection. A raw write to any of
///   these breaks admission control + DVFS.
///
/// **PELT (Per-Entity Load Tracking) averages.** These are
/// exponential moving averages whose internal `_sum` accumulators
/// are advanced against `cfs_rq_clock_pelt(cfs_rq)` (see
/// `kernel/sched/fair.c update_load_avg`, which calls into
/// `kernel/sched/pelt.c __update_load_avg_se` /
/// `__update_load_avg_cfs_rq`). Writing only the visible
/// `_avg` value desynchronises it from the `_sum` it was
/// computed from; the next `update_load_avg` decays both and
/// corrupts the next several passes.
///
/// * **`struct sched_avg`** fields on `task_struct.se.avg` and
///   `cfs_rq.avg`: `load_avg`, `runnable_avg`, `util_avg`,
///   `util_est`, plus `load_sum` / `runnable_sum` / `util_sum`
///   / `last_update_time` / `period_contrib` (see
///   `include/linux/sched.h struct sched_avg`).
/// * **`cfs_rq.removed.{load_avg,util_avg,runnable_avg}`** —
///   pending-decay buffer for departing entities; flushed at the
///   next `update_load_avg`.
/// * **`rq.cpu_capacity`** — set by `update_cpu_capacity`
///   (`kernel/sched/fair.c`, called from the load-balance path
///   `update_group_capacity`) from per-CPU RT capacity scaling;
///   initialized at boot in `kernel/sched/core.c sched_init`.
///   Raw writes are overwritten on the next load-balance tick
///   that triggers a capacity recomputation.
///
/// **Cgroup / task-group accounting.** Updating the task-group
/// hierarchy bypasses the cascade that the kernel performs over
/// every group entity.
///
/// * **`task_group.shares`** — cgroup CPU shares, normally set
///   via `sched_group_set_shares` (`kernel/sched/fair.c`) which
///   cascades into `update_load_set` + walks every task in the
///   group. Raw write skips the cascade and produces
///   inconsistent per-entity load weights.
/// * **`task_group.cfs_bandwidth.{quota, period, runtime}`** —
///   CFS bandwidth control. `tg_set_cfs_bandwidth`
///   (`kernel/sched/core.c`) is the cgroup-fs writer; the
///   per-cfs_rq runtime distribution is performed by
///   `__refill_cfs_bandwidth_runtime` (`kernel/sched/fair.c`)
///   gated by the `cfs_bandwidth_used()` static-key
///   (`kernel/sched/fair.c`) registered via
///   `start_cfs_bandwidth` (`kernel/sched/fair.c`). Raw writes
///   skip all of those.
///
/// **The right shape for influencing these fields is to drive the
/// kernel into the desired state through real activity** —
/// [`Op::Spawn`] with [`SpawnPlacement::RunnerCgroup`] (inherits the spawner's cgroup, typically
/// cgid=1 inside guest VMs) or
/// [`Op::Spawn`] with [`SpawnPlacement::Cgroup`] (runs inside a named cgroup) of a
/// synthetic [`WorkloadConfig`](crate::workload::WorkloadConfig)
/// for fake-load, real preemption pressure for sched_avg.
///
/// ## Fields that ARE safe to write raw (with caveats)
///
/// * **`jiffies_64`** (`include/linux/jiffies.h`) — the global
///   timekeeping tick counter. Safe to advance FORWARD only;
///   backward jumps trigger soft-lockup watchdog warnings and
///   can stall `time_after_eq` waiters whose expiry now appears
///   to be in the past in a way the timer wheel cannot
///   reconcile.
/// * **Per-CPU `rq.clock`** (`struct rq.clock`,
///   `kernel/sched/sched.h`) — the scheduler's per-CPU
///   wall-time clock. Not generically safe: `update_rq_clock`
///   (`kernel/sched/core.c`) overwrites it at every
///   scheduling tick + every enqueue/dequeue from
///   `sched_clock_cpu(cpu)`, so a raw write lasts at most until
///   the next tick (~1 ms with `HZ=1000`). The
///   `rq_clock_skip_update()` helper sets `RQCF_REQ_SKIP` in
///   `rq->clock_update_flags`, which suppresses one
///   `update_rq_clock` call, but its semantics are tightly
///   coupled to the RQCF_ACT_SKIP / RQCF_REQ_SKIP state
///   machine in `__schedule` — a self-contained "freeze
///   rq.clock at value X across step Y" pattern is the
///   framework's responsibility (planned), not a one-shot
///   raw-write primitive. Bumping `rq.clock_task` directly
///   is also NOT safe — that field is computed by
///   `update_rq_clock_task` from `rq->clock` minus IRQ and
///   steal-time deltas (`prev_irq_time` and
///   `prev_steal_time_rq`) and a raw write desynchronises it
///   from the inputs.
/// * **Per-CPU `rq.scx.clock`** (sched_ext per-CPU clock) — safe
///   ONLY when paired with setting `SCX_RQ_CLK_VALID` in
///   `rq.scx.flags`. The flag gates `scx_bpf_now()` reads;
///   writing the clock without the flag leaves `scx_bpf_now()`
///   returning stale data, and clearing the flag without
///   resetting the clock makes downstream BPF readers fall
///   back to the host TSC unexpectedly. Atomic bit-set without
///   read-back is provided by [`KernelValue::OrU32`] — the RMW
///   variant whose width matches `struct scx_rq.flags` (`u32`
///   at `kernel/sched/sched.h:802`). Note there is no
///   `OrU64` sibling: a 64-bit RMW at this field address would
///   corrupt the adjacent `u32 nr_immed` field at
///   `kernel/sched/sched.h:803`. Width is the variant tag, so
///   wrong-width writes are a compile-time error rather than a
///   silent field-overflow bug at runtime. Pair `OrU32(SCX_RQ_CLK_VALID)`
///   with the prior `U64(clock_val)` write in a single
///   `Op::WriteKernelCold` batch so both land under one freeze
///   rendezvous and the kernel's documented
///   write-clock-BEFORE-OR-flag ordering (per
///   `kernel/sched/sched.h:1843-1848` `scx_rq_clock_update`)
///   holds.
/// * **`scx-ktstr` private bss / per-CPU scratch** — the
///   fixture scheduler exposes a dedicated write surface for
///   test use; raw writes there don't propagate into core
///   sched code by construction.
///
/// # `#[non_exhaustive]`
///
/// `KernelTarget` is `#[non_exhaustive]` — see
/// [`crate::non_exhaustive`] for the cross-crate pattern-match rule.
/// Prefer the per-variant constructors ([`Self::symbol`],
/// [`Self::direct`], [`Self::kva`], [`Self::per_cpu_field`]) over
/// naming variant literals.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelTarget {
    /// Kernel text/data/bss symbol. The host resolves
    /// `name → KVA → PA` via the runtime kernel image base + KASLR
    /// `phys_base`, exactly as
    /// `crate::monitor::guest::GuestKernel::write_symbol_u64`
    /// already does for the existing write-symbol helper.
    Symbol(Cow<'static, str>),
    /// Direct-mapped kernel virtual address — translated via
    /// `kva - PAGE_OFFSET`. Use this when the caller has already
    /// resolved a SLAB / per-CPU / physmem KVA and just wants the
    /// host to write at that address.
    Direct(u64),
    /// Vmalloc'd / vmap'd kernel virtual address — translated via
    /// page-table walk through the guest's `CR3`. Use this for BPF
    /// maps, vmalloc'd memory, and any other address that does NOT
    /// live in the direct map.
    Kva(u64),
    /// Per-CPU field of a kernel struct, resolved at op dispatch
    /// time. The variant carries the symbolic intent only (`symbol`,
    /// `field`, `cpu`); the dispatcher looks up `symbol` in the
    /// vmlinux symbol table, adds `__per_cpu_offset[cpu]`, and adds
    /// the BTF-resolved byte offset of `field` within `symbol`'s
    /// struct type to yield the per-CPU field's runtime KVA.
    ///
    /// `symbol` must be in the v1 supported set: `runqueues` →
    /// `struct rq`, `kernel_cpustat` → `struct kernel_cpustat`,
    /// `kstat` → `struct kernel_stat`, `tick_cpu_sched` →
    /// `struct tick_sched`. Unknown symbols fail with a typed error
    /// (the wire variant doesn't carry struct type, so the
    /// dispatcher maps via a hardcoded table — extend it AND
    /// `KernelSymbols::from_elf` to add). KASLR-on round-trip
    /// coverage is an outstanding follow-up; ktstr defaults to
    /// `nokaslr` so the kaslr_offset slide is 0 on the standard
    /// test path.
    ///
    /// Lazy resolution keeps the construction surface pure-data
    /// (the test author needs no `GuestKernel`/BTF/symbol-table
    /// handle to construct the variant); resolution failures
    /// surface as op-execution errors at the same layer as
    /// missing-symbol failures in other snapshot ops.
    PerCpuField {
        /// Kernel symbol naming the per-CPU template
        /// (e.g. `"runqueues"`).
        symbol: Cow<'static, str>,
        /// Field name within the symbol's struct
        /// (e.g. `"clock"` for `struct rq.clock`).
        field: Cow<'static, str>,
        /// CPU index whose per-CPU instance to address.
        cpu: u32,
    },
    /// Per-task field of `struct task_struct` — SCX-managed tasks
    /// only (the dispatcher's L6+L7 validation gates reject non-SCX
    /// tasks). Resolved at dispatch by walking `init_task.tasks`
    /// plus each leader's `signal->thread_head` to locate the task
    /// with matching `pid` AND matching `expected_start_time_ns`
    /// (anti-PID-reuse identity), then adding the BTF-resolved
    /// nested-path byte offset of `field` within `task_struct`.
    /// See `crate::vmm::wire::KernelOpTarget::TaskField` for the
    /// 8-layer validation chain the dispatcher applies.
    ///
    /// `expected_start_time_ns` is `task->start_time` captured at
    /// WorkSpec spawn time. Get it via
    /// [`crate::workload::WorkloadHandle::worker_pids`] for
    /// the PID list, then read `/proc/<pid>/stat` field 22 +
    /// convert from jiffies to ns via
    /// `* 1_000_000_000 / sysconf(_SC_CLK_TCK)`.
    TaskField {
        /// Guest-side `pid_t` of the target task. Both leaders and
        /// non-leader threads are addressable.
        pid: u32,
        /// `task->start_time` (ns) recorded at spawn time. The
        /// dispatcher's L2 check rejects writes when the observed
        /// `task->start_time` differs (PID-reuse identity guard).
        expected_start_time_ns: u64,
        /// Dot-separated nested-member path within `task_struct`.
        /// SCX-only fields recommended (e.g. `"scx.dsq_vtime"`,
        /// `"start_boottime"`). `"se.vruntime"` writes are
        /// silently discarded by EEVDF's `place_entity` on enqueue
        /// (`kernel/sched/fair.c:5329-5414` since 6.6) AND rejected
        /// by the SCX-only class gate; do not use.
        field: Cow<'static, str>,
    },
}

impl KernelTarget {
    /// Kernel text/data/bss symbol target. Resolves at op-dispatch
    /// time via the runtime kernel image base + KASLR `phys_base`.
    ///
    /// **Heads up.** See the `# Semantic risk` section on the
    /// enclosing [`KernelTarget`] type doc before pointing this
    /// at a scheduler-bookkeeping symbol.
    pub fn symbol(name: impl Into<Cow<'static, str>>) -> Self {
        KernelTarget::Symbol(name.into())
    }

    /// Direct-mapped KVA target. Translates via `kva - PAGE_OFFSET`.
    /// For per-CPU bases the caller must add
    /// `__per_cpu_offset[cpu]` to the base symbol KVA before
    /// constructing the variant; use [`Self::per_cpu_field`]
    /// instead for the framework-resolved per-CPU shape.
    ///
    /// **Heads up.** See the `# Semantic risk` section on the
    /// enclosing [`KernelTarget`] type doc before pointing this
    /// at a scheduler-bookkeeping address.
    pub const fn direct(kva: u64) -> Self {
        KernelTarget::Direct(kva)
    }

    /// Vmalloc'd / vmap'd KVA target. Translates via page-table
    /// walk through the guest's `CR3`.
    ///
    /// **Heads up.** See the `# Semantic risk` section on the
    /// enclosing [`KernelTarget`] type doc before pointing this
    /// at a scheduler-bookkeeping address.
    pub const fn kva(kva: u64) -> Self {
        KernelTarget::Kva(kva)
    }

    /// Per-CPU field of a kernel struct. Resolves at op-dispatch
    /// time via `symbol_kva + __per_cpu_offset[cpu] + BTF byte
    /// offset of field`.
    ///
    /// **Heads up.** See the `# Semantic risk` section on the
    /// enclosing [`KernelTarget`] type doc before pointing this
    /// at a per-CPU scheduler-bookkeeping field.
    pub fn per_cpu_field(
        symbol: impl Into<Cow<'static, str>>,
        field: impl Into<Cow<'static, str>>,
        cpu: u32,
    ) -> Self {
        KernelTarget::PerCpuField {
            symbol: symbol.into(),
            field: field.into(),
            cpu,
        }
    }

    /// Per-task `struct task_struct` field target — SCX-managed
    /// tasks only. Resolves at dispatch via `init_task.tasks` +
    /// per-leader `signal->thread_head` walks to find the task
    /// with matching `pid` AND matching `expected_start_time_ns`
    /// (anti-PID-reuse), then BTF nested-path offset of `field`.
    ///
    /// `expected_start_time_ns` is `task->start_time` (set once by
    /// `kernel/fork.c::copy_process` via `ktime_get_ns()`).
    /// Get worker PIDs via
    /// [`crate::workload::WorkloadHandle::worker_pids`] then
    /// read `/proc/<pid>/stat` field 22 at spawn time and convert
    /// to ns: `field_22_jiffies * 1_000_000_000 /
    /// sysconf(_SC_CLK_TCK)`.
    ///
    /// `field` is dot-separated nested-member path. The dispatcher
    /// applies an 8-layer validation chain (pid match, start_time
    /// identity, lifetime, on_rq=0, scx queued-empty, ext
    /// sched_class, SCHED_EXT policy, start_boottime != 0) before
    /// the write/read lands — see
    /// `crate::vmm::wire::KernelOpTarget::TaskField` for the full
    /// contract.
    ///
    /// **SCX-only.** The dispatcher rejects non-SCX tasks via the
    /// class+policy gates. Recommended fields: `"scx.dsq_vtime"`
    /// (DSQ priority key, preserved across dequeue/enqueue),
    /// `"start_boottime"` (task fork timestamp).
    ///
    /// **Do NOT write `"se.vruntime"`.** EEVDF's `place_entity`
    /// (`kernel/sched/fair.c:5329-5414`, since 6.6) overwrites
    /// `se->vruntime` on every enqueue; direct vruntime writes are
    /// silently discarded for sleeping tasks (our validation gate).
    /// CFS-class tasks are rejected before reaching the write
    /// regardless, but the field-level warning is the actionable
    /// guidance for "why won't my vruntime write stick" debugging.
    ///
    /// **Heads up.** The dispatcher's L4 (`on_rq == 0`) + L5
    /// (`scx.dsq == NULL` AND `scx.runnable_node` empty) gates
    /// reject writes on queued/running tasks per CFS rb-tree + SCX
    /// DSQ ordering safety. Test authors must use blocking workload
    /// patterns (e.g. [`crate::workload::WorkType::FutexPingPong`],
    /// `WorkType::WaitOnFutex`, `WorkType::Sleep`) so workers are
    /// sleeping when the cold-path Op fires.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Escape-hatch primitive: seed a specific worker's
    /// // scx.dsq_vtime to ~30 days. WorkSpec.uptime (separate API)
    /// // wraps this; use the escape hatch when the scenario knows
    /// // the exact PID + start_time tuple.
    /// use ktstr::prelude::*;
    /// use std::time::Duration;
    ///
    /// let workers = handle.worker_pids();         // Vec<libc::pid_t>
    /// let worker_pid = workers[0] as u32;
    /// // Read `/proc/<pid>/stat` field 22, convert from jiffies to
    /// // nanoseconds via `* 1_000_000_000 / sysconf(_SC_CLK_TCK)`.
    /// // (Helper expected to land alongside WorkSpec.uptime.)
    /// let start_time_ns: u64 = read_start_time_ns(worker_pid)?;
    ///
    /// let seed_vtime_ns = (30 * 86_400_u64) * 1_000_000_000; // 30 days
    /// let writes = vec![(
    ///     KernelTarget::task_field(worker_pid, start_time_ns, "scx.dsq_vtime"),
    ///     KernelValue::u64(seed_vtime_ns),
    /// )];
    /// // Worker MUST be in a blocking pattern (FutexPingPong, etc.)
    /// // at op-fire time; the dispatcher's 8-layer validation
    /// // rejects writes against runnable/queued tasks.
    /// ```
    pub fn task_field(
        pid: u32,
        expected_start_time_ns: u64,
        field: impl Into<Cow<'static, str>>,
    ) -> Self {
        KernelTarget::TaskField {
            pid,
            expected_start_time_ns,
            field: field.into(),
        }
    }
}

/// Value payload for the kernel-memory write ops, and the result
/// shape for the read ops.
///
/// The variant tag picks both the width (`u32` vs `u64` vs a byte
/// slice) and the underlying `crate::monitor::guest::GuestKernel`
/// write helper the host coordinator will invoke (`write_*_u32`,
/// `write_*_u64`, `write_*_bytes` per the [`KernelTarget`] class).
///
/// # `#[non_exhaustive]`
///
/// `KernelValue` is `#[non_exhaustive]` so new value widths can be
/// added without breaking external pattern-matchers. Prefer the
/// per-variant constructors over naming variant literals.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelValue {
    /// 32-bit unsigned little-endian write. Atomic when the
    /// resolved host PA is 4-byte aligned. Misaligned PAs fall
    /// through to a per-byte volatile loop in
    /// `crate::monitor::reader::GuestMem`
    /// `write_volatile_bytes` (the 4-byte fast path branches on
    /// `ptr.align_offset(align_of::<u32>()) == 0` and only emits
    /// a single `write_volatile` when alignment holds); torn
    /// intermediate state is observable to concurrent guest readers
    /// in the fallback case.
    ///
    /// **For setting individual bits without disturbing the
    /// surrounding value**, use [`Self::OrU32`] instead — that
    /// variant performs read-modify-write OR semantics under the
    /// freeze rendezvous (e.g. setting `SCX_RQ_CLK_VALID` in
    /// `rq.scx.flags` without clobbering the other 31 flag bits).
    /// A plain `U32(value)` write replaces every bit; OrU32 sets
    /// only the bits in the mask.
    U32(u32),
    /// 64-bit unsigned little-endian write. Atomic when the
    /// resolved host PA is 8-byte aligned. See the alignment note
    /// on [`Self::U32`] for the misaligned fall-through behaviour.
    ///
    /// **No `OrU64` sibling exists by design.** The canonical
    /// scheduler-flags use case ([`KernelValue::OrU32`] →
    /// `struct scx_rq.flags`) is on a `u32` field per
    /// `kernel/sched/sched.h:802`; a 64-bit RMW at that address
    /// would corrupt the adjacent `u32 nr_immed` field at
    /// `kernel/sched/sched.h:803`. If a future u64 RMW use case
    /// emerges with a verified width, add the variant then.
    U64(u64),
    /// Variable-length byte payload. Written non-atomically; the
    /// `GuestKernel::write_*_bytes` helpers emit a Release fence
    /// after the copy so a weakly-ordered guest's
    /// `smp_load_acquire` observes the bytes in write order — the
    /// fence orders the stores but does NOT atomicize the
    /// multi-byte write versus a concurrent guest reader.
    Bytes(Vec<u8>),
    /// 32-bit unsigned read-modify-write OR. The dispatcher reads
    /// the live u32 at the resolved host PA, ORs the carried mask
    /// into it, and writes the new value back. Width is u32 — the
    /// canonical use case is OR-ing a single-bit kernel flag (e.g.
    /// `SCX_RQ_CLK_VALID = 1 << 5`) into `struct scx_rq.flags`,
    /// declared `u32` at `kernel/sched/sched.h:802` inside the
    /// struct opened at L793. A 64-bit RMW at a u32 field address
    /// would either silently truncate the upper 32 bits or
    /// corrupt the adjacent `u32 nr_immed` field at
    /// `kernel/sched/sched.h:803`, so the variant tag itself
    /// picks the width and rules out width mismatch at the call
    /// site.
    ///
    /// **Atomicity** (cold-path dispatcher): the host coordinator
    /// holds the freeze rendezvous for the duration of the RMW —
    /// every guest vCPU is parked on a futex inside `handle_freeze`
    /// (no kernel-side writer is scheduled), and the host
    /// coordinator is the only writer of guest memory in scope.
    /// `read_u32 → OR mask → write_u32` therefore runs atomic
    /// **by quiesce**: no concurrent kernel writer can interleave
    /// between the load and the store. No `compare_exchange` loop
    /// is required for cold-path dispatch.
    ///
    /// At the host CPU level the read and write are separate
    /// (non-instruction-atomic) operations: a hypothetical
    /// concurrent host writer of guest memory would be a race.
    /// The freeze coordinator is the sole such writer by design
    /// (per the cold-path threat model documented at
    /// [`super::Op::WriteKernelCold`]), so the parked-vCPU
    /// contract is sufficient.
    ///
    /// **Alignment**: the dispatcher delegates u32 reads/writes
    /// to `crate::monitor::guest::GuestKernel`'s
    /// `read_*_u32` / `write_*_u32` helpers, which use a
    /// single-instruction `write_volatile` at 4-byte-aligned host
    /// PAs and fall through to a per-byte volatile loop on
    /// misalignment. Under the freeze rendezvous the per-byte
    /// fallback is safe (no concurrent kernel writer), so
    /// misaligned PAs do not produce a torn-RMW race —
    /// but kernel ABI alignment for `u32` fields is enforced by
    /// the compiler at the kernel side regardless, so misaligned
    /// PAs for legitimate symbol/field writes do not arise in
    /// practice.
    ///
    /// **Hot-path future** (when [`super::Op::WriteKernelHot`]
    /// gains `OrU32` support — currently rejected per the
    /// [`super::Op::WriteKernelHot`] doc): the live-guest race
    /// model requires a `compare_exchange` loop over
    /// `core::sync::atomic::AtomicU32::from_ptr` (Rust 1.75+) at
    /// 4-byte alignment, with explicit rejection of misaligned
    /// PAs (per-byte fallback cannot be made atomic vs. a live
    /// kernel writer).
    ///
    /// **Ordering**: cold-path dispatch happens while every vCPU
    /// is parked at the freeze rendezvous, so no concurrent
    /// guest write races our RMW for single-op use cases. The
    /// `SCX_RQ_CLK_VALID` case specifically requires
    /// **write-clock-BEFORE-OR-flag** ordering per the kernel's
    /// own `scx_rq_clock_update` at `kernel/sched/sched.h:1843-1848`
    /// (which does `WRITE_ONCE(rq->scx.clock, val)` then
    /// `smp_store_release(&rq->scx.flags, flags |
    /// SCX_RQ_CLK_VALID)`); a host-side caller that wants the
    /// same observable invariant must batch the clock write +
    /// the OR-flag in the same `Op::WriteKernelCold` batch and
    /// rely on the freeze rendezvous's vCPU-pause to serialise
    /// against guest readers.
    OrU32(u32),
}

impl KernelValue {
    /// 32-bit unsigned value.
    pub const fn u32(val: u32) -> Self {
        KernelValue::U32(val)
    }

    /// 64-bit unsigned value.
    pub const fn u64(val: u64) -> Self {
        KernelValue::U64(val)
    }

    /// Variable-length byte payload.
    pub fn bytes(data: impl Into<Vec<u8>>) -> Self {
        KernelValue::Bytes(data.into())
    }

    /// 32-bit unsigned read-modify-write OR mask. See
    /// [`Self::OrU32`] for the width-, atomicity-, and ordering-
    /// contract. The canonical use case is OR-ing a single-bit
    /// kernel flag like `SCX_RQ_CLK_VALID` into `struct scx_rq.flags`.
    pub const fn or_u32(mask: u32) -> Self {
        KernelValue::OrU32(mask)
    }
}

impl From<&KernelTarget> for crate::vmm::wire::KernelOpTarget {
    /// 1:1 mapping of every Op-side [`KernelTarget`] variant to its
    /// wire-side peer. `Cow → String` coercion for the symbolic
    /// forms; copy for the integer/`u32` forms. Used by the
    /// executor's `Op::WriteKernel*` / `Op::ReadKernel*` dispatch
    /// arms when building [`crate::vmm::wire::KernelOpRequestPayload`].
    fn from(target: &KernelTarget) -> Self {
        match target {
            KernelTarget::Symbol(name) => Self::Symbol(name.to_string()),
            KernelTarget::Direct(kva) => Self::Direct(*kva),
            KernelTarget::Kva(kva) => Self::Kva(*kva),
            KernelTarget::PerCpuField { symbol, field, cpu } => Self::PerCpuField {
                symbol: symbol.to_string(),
                field: field.to_string(),
                cpu: *cpu,
            },
            KernelTarget::TaskField {
                pid,
                expected_start_time_ns,
                field,
            } => Self::TaskField {
                pid: *pid,
                expected_start_time_ns: *expected_start_time_ns,
                field: field.to_string(),
            },
        }
    }
}

impl From<&KernelValue> for crate::vmm::wire::KernelOpValue {
    /// 1:1 mapping of every Op-side [`KernelValue`] variant to its
    /// wire-side peer. The `Bytes` arm clones the inner `Vec<u8>`
    /// so the source variant remains usable after dispatch (large
    /// payloads pay the clone cost — see
    /// [`crate::vmm::wire::KernelOpValue::Bytes`] for the wire
    /// representation).
    fn from(value: &KernelValue) -> Self {
        match value {
            KernelValue::U32(v) => Self::U32(*v),
            KernelValue::U64(v) => Self::U64(*v),
            KernelValue::Bytes(b) => Self::Bytes(b.clone()),
            KernelValue::OrU32(mask) => Self::OrU32(*mask),
        }
    }
}

/// Width specifier for the [`Op::ReadKernelHot`] /
/// [`Op::ReadKernelCold`] ops — picks which
/// `crate::monitor::guest::GuestKernel`
/// `read_*_u32` / `read_*_u64` / `read_*_bytes` family the host
/// dispatcher invokes for the read. Mirrors [`KernelValue`]'s
/// variant tags but without payload data (reads do not carry an
/// outgoing value — only a width hint that the dispatcher uses to
/// size the resulting [`crate::vmm::wire::KernelOpValue`] in the
/// reply).
///
/// # `#[non_exhaustive]`
///
/// `KernelValueWidth` is `#[non_exhaustive]` so new widths can be
/// added without breaking external pattern-matchers. Prefer the
/// per-variant constructors ([`Self::u32`], [`Self::u64`],
/// [`Self::bytes`]) over naming variant literals.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelValueWidth {
    /// Read a `u32` little-endian. Atomic when the resolved host
    /// PA is 4-byte aligned (see [`KernelValue::U32`]'s alignment
    /// note for the misaligned fall-through behaviour).
    U32,
    /// Read a `u64` little-endian. Atomic at 8-byte alignment;
    /// otherwise a per-byte loop is used (same fall-through as
    /// [`KernelValue::U64`]).
    U64,
    /// Read exactly `len` raw bytes. Non-atomic; reads through the
    /// `crate::monitor::guest::GuestKernel` `read_*_bytes`
    /// helpers' chunked-page primitive.
    Bytes(usize),
}

impl KernelValueWidth {
    /// `u32` read width.
    pub const fn u32() -> Self {
        KernelValueWidth::U32
    }

    /// `u64` read width.
    pub const fn u64() -> Self {
        KernelValueWidth::U64
    }

    /// `len`-byte read width. Produces a
    /// [`crate::vmm::wire::KernelOpValue::Bytes`] of exactly `len`
    /// bytes in the reply.
    pub const fn bytes(len: usize) -> Self {
        KernelValueWidth::Bytes(len)
    }
}

impl From<&KernelValueWidth> for crate::vmm::wire::KernelOpValue {
    /// Map a [`KernelValueWidth`] to a zero-filled
    /// [`crate::vmm::wire::KernelOpValue`] of the requested width
    /// for the read-entry's value-hint slot. The wire payload's
    /// `value` discriminant tells the host dispatcher which read
    /// family to invoke; the byte contents are written by the
    /// host before replying.
    fn from(width: &KernelValueWidth) -> Self {
        match width {
            KernelValueWidth::U32 => Self::U32(0),
            KernelValueWidth::U64 => Self::U64(0),
            KernelValueWidth::Bytes(len) => Self::Bytes(vec![0u8; *len]),
        }
    }
}
