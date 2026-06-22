//! `KtstrTestEntry::validate` and its phase helpers, split out of
//! `entry.rs` to keep that file under its grandfathered file-size
//! ceiling while the validator stays decomposed into sub-200-line
//! phases. These are inherent methods on
//! [`crate::test_support::KtstrTestEntry`]; its fields are `pub`, so a
//! sibling-module impl has full access.

impl crate::test_support::KtstrTestEntry {
    /// Reject values that would boot a broken VM or leave assertions
    /// vacuously passing. The `#[ktstr_test]` proc macro enforces the
    /// same constraints at compile time for attribute-built entries;
    /// this method covers directly-constructed entries (library
    /// callers building `KtstrTestEntry` values to push into
    /// [`KTSTR_TESTS`] programmatically).
    ///
    /// Rules:
    /// - `name` must be non-empty (empty names collapse into each
    ///   other in nextest output and in sidecar lookups).
    /// - `name` must not contain `/` or `\` (path separators embed in
    ///   sidecar filenames and nextest test IDs; a separator would
    ///   create a synthetic subdirectory in sidecar output and
    ///   mangle `cargo nextest run -E 'test(name)'` filtering).
    /// - `memory_mib` must be `> 0` (a VM with zero memory cannot boot).
    /// - `duration` must be `> 0` (a zero-duration run never exercises
    ///   the scheduler and produces no telemetry).
    pub fn validate(&self) -> anyhow::Result<()> {
        self.validate_basics_and_staging()?;
        self.validate_mode_flags()?;
        self.validate_snapshots()?;
        self.validate_config_and_topology()?;
        Ok(())
    }

    /// Validate name shape, memory/duration sizing, payload/host_only
    /// device conflicts, and staged_schedulers name uniqueness.
    fn validate_basics_and_staging(&self) -> anyhow::Result<()> {
        if self.name.is_empty() {
            anyhow::bail!(
                "KtstrTestEntry.name must be non-empty (empty names \
                 collide in nextest output and sidecar lookups)"
            );
        }
        if self.name.contains('/') || self.name.contains('\\') {
            anyhow::bail!(
                "KtstrTestEntry '{}' name must not contain path \
                 separators ('/' or '\\') — they embed in sidecar \
                 filenames and nextest test IDs, creating synthetic \
                 subdirectories in sidecar output and mangling \
                 nextest -E 'test(name)' filtering",
                self.name,
            );
        }
        if self.memory_mib == 0 {
            anyhow::bail!(
                "KtstrTestEntry '{}'.memory_mib must be > 0 (a VM with \
                 zero memory cannot boot)",
                self.name,
            );
        }
        if self.duration.is_zero() {
            anyhow::bail!(
                "KtstrTestEntry '{}'.duration must be > 0 (a zero-duration \
                 run never exercises the scheduler and produces no data \
                 for assertions)",
                self.name,
            );
        }
        if let Some(p) = self.payload
            && p.is_scheduler()
        {
            anyhow::bail!(
                "KtstrTestEntry '{}'.payload must be PayloadKind::Binary, \
                 not Scheduler-kind (schedulers belong in the `scheduler` \
                 slot; the `payload` slot is for userspace binaries \
                 composed under the scheduler)",
                self.name,
            );
        }
        if self.host_only && self.disk.is_some() {
            anyhow::bail!(
                "KtstrTestEntry '{}'.host_only=true with disk=Some(..) — \
                 host_only skips the VM boot that owns the virtio-blk \
                 device lifecycle, so the disk would never be attached. \
                 Drop one of host_only or disk.",
                self.name,
            );
        }
        if self.host_only && self.network.is_some() {
            anyhow::bail!(
                "KtstrTestEntry '{}'.host_only=true with network=Some(..) — \
                 host_only skips the VM boot that owns the virtio-net \
                 device lifecycle, so the NIC would never be attached. \
                 Drop one of host_only or network.",
                self.name,
            );
        }
        // staged_schedulers names must (a) pass the per-name shape
        // checks (non-empty, no path separators, no NUL bytes, no
        // leading dot, not a reserved framework slot — see
        // [`crate::test_support::staged::validate_staged_scheduler_name`])
        // and (b) be unique within the set AND disjoint from the
        // boot scheduler's `name`. A collision on either axis would
        // land two distinct schedulers at the same guest path —
        // silent overwrite, the second-staged binary clobbering the
        // first OR shadowing a boot-time framework slot. The
        // boot-name seed catches the "stage all the schedulers I
        // might use" misuse (author includes the boot scheduler in
        // the staged set thinking it's required there too). Bails
        // here at validate time so the error surfaces ahead of any
        // VM boot or initramfs construction.
        let mut seen_names: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        seen_names.insert(self.scheduler.name);
        let staged_who = format!("KtstrTestEntry '{}'.staged_schedulers", self.name);
        for staged in self.staged_schedulers {
            crate::test_support::staged::validate_staged_scheduler_name(&staged_who, staged.name)?;
            if !seen_names.insert(staged.name) {
                if staged.name == self.scheduler.name {
                    anyhow::bail!(
                        "KtstrTestEntry '{}'.staged_schedulers cannot include \
                         the boot scheduler '{}' — the boot slot already \
                         stages it. Staged entries are the ADDITIONAL \
                         candidates the test will swap TO via \
                         Op::AttachScheduler / Op::ReplaceScheduler.",
                        self.name,
                        staged.name,
                    );
                }
                anyhow::bail!(
                    "KtstrTestEntry '{}'.staged_schedulers has duplicate \
                     Scheduler.name '{}'; each staged scheduler must have \
                     a unique name (the name maps 1:1 to the guest-side \
                     staging path)",
                    self.name,
                    staged.name,
                );
            }
        }
        Ok(())
    }

    /// Validate host_only/scheduler, performance/no-perf, and
    /// cpu_budget mode-flag combinations and the scx_bpf_error matcher
    /// gate.
    fn validate_mode_flags(&self) -> anyhow::Result<()> {
        // Defense-in-depth for the programmatic-construction path
        // (struct-literal `KtstrTestEntry { .. }` in integration tests,
        // gauntlet-rewritten entries). The macro at
        // ktstr-macros/src/lib.rs rejects `host_only = true` paired with
        // any `scheduler = ...` attribute at compile time, but
        // programmatic construction bypasses that gate. Match against
        // `SchedulerSpec::Eevdf` (the value-level marker for the
        // no-scx-scheduler placeholder) so a struct literal that sets
        // `scheduler: &SOME_REAL_SCHED` under host_only is caught while
        // the default `scheduler: &Scheduler::EEVDF` (whose binary is
        // `SchedulerSpec::Eevdf`) is accepted. The variant-based check
        // is spec-safe — unlike a pointer-identity check against
        // `&Scheduler::EEVDF`, which depends on rustc/LLVM's const-
        // deduplication of `&CONST_EXPR` materializations.
        if self.host_only
            && !matches!(
                self.scheduler.binary,
                crate::test_support::SchedulerSpec::Eevdf
            )
        {
            anyhow::bail!(
                "KtstrTestEntry '{}'.host_only=true with scheduler=&{:?} — \
                 host_only skips the VM boot that owns the scheduler \
                 lifecycle, so the declared scheduler would never attach. \
                 Drop one of host_only or scheduler; the host's \
                 currently-active scheduler (default EEVDF when none is \
                 loaded) runs the test under host_only.",
                self.name,
                self.scheduler.name,
            );
        }
        if self.performance_mode && self.no_perf_mode {
            anyhow::bail!(
                "KtstrTestEntry '{}'.performance_mode=true with \
                 no_perf_mode=true — the two flags are contradictory \
                 (\"I want pinning\" vs. \"I explicitly don't want \
                 pinning\"). Drop one of them.",
                self.name,
            );
        }
        // `cpu_budget` of zero cannot run a VM. The builder would
        // otherwise clamp it to 1 (builder.rs effective_cap), silently
        // running with a budget the author never asked for. Reject
        // explicitly — mirrors the macro's compile-time reject and the
        // memory_mib / cleanup_budget_ms zero-rejects in
        // validate_cross_attr.
        if self.cpu_budget == Some(0) {
            anyhow::bail!(
                "KtstrTestEntry '{}'.cpu_budget=Some(0) — a zero host-CPU \
                 budget cannot run a VM. Use a positive budget, or drop \
                 cpu_budget to auto-size the no-perf mask to the vCPU count.",
                self.name,
            );
        }
        // `cpu_budget` is consulted only on the no_perf_mode path
        // (builder.rs sizes the shared vCPU-thread mask from it). A
        // budget set without no_perf_mode is a silent no-op — the VM
        // runs with the default mask and the requested overcommit never
        // happens, so a contention test would quietly run un-contended.
        // Reject at validate time (nextest discovery) for the
        // programmatic-construction path; ktstr-macros enforces the same
        // gate at compile time for the `#[ktstr_test]` path.
        if self.cpu_budget.is_some() && !self.no_perf_mode {
            anyhow::bail!(
                "KtstrTestEntry '{}'.cpu_budget={:?} with no_perf_mode=false \
                 — cpu_budget sizes the no-perf vCPU-thread mask and is \
                 ignored unless no_perf_mode is set (under performance_mode \
                 vCPUs are pinned 1:1). Set no_perf_mode=true or drop \
                 cpu_budget.",
                self.name,
                self.cpu_budget,
            );
        }
        if (self.assert.expect_scx_bpf_error_contains.is_some()
            || self.assert.expect_scx_bpf_error_matches.is_some())
            && !self.expect_err
        {
            anyhow::bail!(
                "KtstrTestEntry '{}' sets an scx_bpf_error matcher \
                 (expect_scx_bpf_error_contains or expect_scx_bpf_error_matches) \
                 without expect_err = true — a reproducer matcher narrows \
                 which failure counts as the expected bug and only \
                 applies to expected-error tests. Set expect_err = true \
                 or drop the matcher.",
                self.name,
            );
        }
        Ok(())
    }

    /// Validate num_snapshots against the storage cap, host_only, and
    /// the minimum periodic-capture interval.
    fn validate_snapshots(&self) -> anyhow::Result<()> {
        // Periodic snapshots route through SnapshotBridge::store, which
        // FIFO-evicts at MAX_STORED_SNAPSHOTS. Allowing num_snapshots
        // past the cap would silently lose the earliest samples — a
        // periodic run with N=128 today would only retain
        // periodic_064..periodic_127 in the bridge.
        let max = crate::scenario::snapshot::MAX_STORED_SNAPSHOTS as u32;
        if self.num_snapshots > max {
            anyhow::bail!(
                "KtstrTestEntry '{}'.num_snapshots={} exceeds \
                 MAX_STORED_SNAPSHOTS={} — the bridge would FIFO-evict \
                 the earliest periodic samples. Lower the count or split \
                 into multiple test entries.",
                self.name,
                self.num_snapshots,
                max,
            );
        }
        if self.num_snapshots > 0 {
            // host_only skips the VM boot that owns the freeze
            // coordinator's run-loop. Without that loop there is no
            // thread to stamp `scenario_start_ns`, no thread to fire
            // `freeze_and_capture(false)` at each boundary, and no
            // `SnapshotBridge` plumbed onto a `VmResult` for the
            // test author to drain post-run. The combination is
            // unsatisfiable; reject at validate time so a
            // misconfigured entry surfaces during nextest discovery
            // rather than as silently-empty bridge results.
            if self.host_only {
                anyhow::bail!(
                    "KtstrTestEntry '{}'.host_only=true with \
                     num_snapshots={} > 0 — host_only skips the VM \
                     boot that owns the freeze coordinator's \
                     periodic-capture loop, so no snapshot would \
                     ever fire. Drop one of host_only or \
                     num_snapshots.",
                    self.name,
                    self.num_snapshots,
                );
            }
            // Refuse interval shorter than the minimum useful capture
            // cadence. Each boundary fire freezes every vCPU, walks
            // BPF maps, serialises the dump, and writes to the
            // bridge — under the FREEZE_RENDEZVOUS_TIMEOUT (30 s)
            // hard ceiling but commonly tens of milliseconds on a
            // healthy guest. An interval shorter than ~100 ms would
            // back-to-back the captures with no actual workload
            // progress between them, defeating the periodic-sampling
            // purpose. Compute the interval in nanoseconds in u128
            // to avoid overflow on long durations: the formula
            // mirrors the run-loop's
            // `compute_periodic_boundaries_ns` (10 % pre-buffer,
            // 80 % usable span, divided into N+1 equal intervals).
            let usable_span_ns = self
                .duration
                .as_nanos()
                .saturating_sub(2u128.saturating_mul(self.duration.as_nanos() / 10));
            let interval_ns = usable_span_ns / (self.num_snapshots as u128 + 1);
            const MIN_INTERVAL_NS: u128 = 100 * 1_000_000; // 100 ms
            if interval_ns < MIN_INTERVAL_NS {
                anyhow::bail!(
                    "KtstrTestEntry '{}'.num_snapshots={} with \
                     duration={:?} produces a periodic interval of \
                     {} ns ({} ms) — below the 100 ms minimum the \
                     freeze-and-capture path can sustain without \
                     back-to-back firing. Either reduce num_snapshots \
                     or extend duration so 0.8·duration / (N+1) >= 100 ms.",
                    self.name,
                    self.num_snapshots,
                    self.duration,
                    interval_ns,
                    interval_ns / 1_000_000,
                );
            }
        }
        Ok(())
    }

    /// Validate scheduler config_file_def/config_content pairing,
    /// workload-slot payload kinds, and entry/scheduler topology
    /// constraints.
    fn validate_config_and_topology(&self) -> anyhow::Result<()> {
        // Pair `scheduler.config_file_def` with `config_content`. The
        // `#[ktstr_test]` macro emits a `const _: () = assert!(...)`
        // block that catches the same mismatch at compile time for
        // attribute-built entries; this branch covers programmatic
        // construction (callers building `KtstrTestEntry` values
        // directly) and surfaces the misconfiguration before VM boot
        // rather than as a silent missing-`--config` flag.
        let scheduler_has_def = self.scheduler.config_file_def.is_some();
        let entry_has_content = self.config_content.is_some();
        if scheduler_has_def && !entry_has_content {
            anyhow::bail!(
                "KtstrTestEntry '{}'.scheduler '{}' declares \
                 `config_file_def` but the entry does not supply \
                 `config_content`; the scheduler binary expects an \
                 inline config and would launch without `--config`. \
                 Set `config = ...` on `#[ktstr_test]` or assign \
                 `config_content` directly.",
                self.name,
                self.scheduler.name,
            );
        }
        if !scheduler_has_def && entry_has_content {
            anyhow::bail!(
                "KtstrTestEntry '{}'.config_content is set but the \
                 scheduler '{}' does not declare `config_file_def`; \
                 the content would be silently dropped at dispatch. \
                 Remove `config = ...` or add \
                 `config_file_def(arg_template, guest_path)` to the \
                 scheduler.",
                self.name,
                self.scheduler.name,
            );
        }
        // Mirror the payload-slot gate for every workload entry. The
        // `workloads` slot is for userspace binaries composed with
        // the primary payload under the scheduler; a scheduler-kind
        // Payload here would be silently ignored at spawn time. The
        // narrow typo path post-`declare_scheduler!` rollout is
        // pasting [`Payload::KERNEL_DEFAULT`] (the only Scheduler-kind
        // Payload still in the prelude) into a `workloads = [...]`
        // attribute instead of the `scheduler = ...` slot.
        for (idx, w) in self.workloads.iter().enumerate() {
            if w.is_scheduler() {
                anyhow::bail!(
                    "KtstrTestEntry '{}'.workloads[{idx}] (name='{}') must be \
                     PayloadKind::Binary, not Scheduler-kind (schedulers belong \
                     in the `scheduler` slot; the `workloads` slot is for \
                     userspace binaries composed under the scheduler)",
                    self.name,
                    w.name,
                );
            }
        }
        // Reject inverted topology ranges before they silently filter
        // every gauntlet preset to zero matches. The per-entry
        // constraints gate which gauntlet presets the test author wants
        // to exercise; an inverted bound (e.g. min_numa_nodes=5 with
        // max_numa_nodes=Some(2)) would yield false on every preset.
        self.constraints
            .validate()
            .map_err(|e| anyhow::anyhow!("KtstrTestEntry '{}'.constraints: {e}", self.name))?;
        // Same for the scheduler-level constraints, which apply on top
        // of the per-entry ones. A scheduler whose declared topology
        // requirements are themselves inverted has the same silent-
        // filter pathology regardless of what test entries declare.
        self.scheduler.constraints.validate().map_err(|e| {
            anyhow::anyhow!(
                "KtstrTestEntry '{}'.scheduler '{}'.constraints: {e}",
                self.name,
                self.scheduler.name
            )
        })?;
        Ok(())
    }
}
