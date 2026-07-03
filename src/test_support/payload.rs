//! Generalized test payload — scheduler or binary workload.
//!
//! [`Payload`] is the primitive that `#[ktstr_test]` consumes for both
//! the scheduler slot and the optional binary/workload slots. A
//! payload's [`PayloadKind`] determines how it's launched: a
//! [`Scheduler`] reference invokes the
//! existing scheduler-spawn path; a bare binary name spawns the binary
//! via the runtime [`PayloadRun`](crate::scenario::payload_run::PayloadRun)
//! builder.
//!
//! The constants this module exposes — particularly
//! [`Payload::KERNEL_DEFAULT`] — are used as the default scheduler
//! slot when no `scheduler = ...` attribute is supplied on a
//! `#[ktstr_test]`. `KERNEL_DEFAULT` wraps whatever scheduler the
//! running kernel selects when no sched_ext scheduler is attached
//! (EEVDF on Linux 6.6+) and surfaces on the wire as
//! `"kernel_default"`.
//!
//! [`KtstrTestEntry`](crate::test_support::KtstrTestEntry) carries
//! `payload` and `workloads` fields populated by the `#[ktstr_test]`
//! macro's `payload = ...` and `workloads = [...]` attributes.

use crate::test_support::Scheduler;

// ---------------------------------------------------------------------------
// Payload + PayloadKind
// ---------------------------------------------------------------------------

/// A test payload — either a scheduler or a userspace binary to run
/// inside the guest VM.
///
/// `Payload` unifies the two launch modes under one `#[ktstr_test]`
/// attribute surface: tests declare `scheduler = SOME_SCHED` for
/// scheduler-centric runs, `payload = SOME_BIN` for binary runs, or
/// both with `workloads = [...]` to compose binaries under a
/// scheduler. See [`PayloadKind`] for the two variants.
///
/// Use [`Payload::KERNEL_DEFAULT`] as the default scheduler
/// placeholder when a test doesn't attach a sched_ext scheduler —
/// it wraps the kernel's default scheduler (EEVDF on Linux 6.6+)
/// via [`Scheduler::EEVDF`].
///
/// `Payload` intentionally does NOT implement [`serde::Serialize`] /
/// [`serde::Deserialize`]. It is a compile-time-static definition that
/// references `&'static Scheduler` and `&'static [&'static str]`
/// slices — lifetimes that serialization cannot round-trip. Runtime
/// telemetry (per-payload metrics, exit codes, names) is serialized
/// via [`PayloadMetrics`] and [`Metric`] instead; those own their
/// data.
///
/// `#[non_exhaustive]` reserves the right to add fields without
/// breaking downstream code. Out-of-crate callers cannot construct
/// `Payload` via struct literal — use the const-fn constructors
/// ([`Payload::new`], [`Payload::binary`]) or the
/// `#[derive(Payload)]` macro, which routes through [`Payload::new`]
/// under the hood.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct Payload {
    /// Short, stable name used in logs and sidecar records.
    pub name: &'static str,
    /// Launch kind — scheduler reference or binary name.
    pub kind: PayloadKind,
    /// How the framework extracts metrics from the payload's
    /// stdout, with stderr fallback when stdout yields no metrics.
    /// See [`OutputFormat`] for the per-variant contract and
    /// `scenario::payload_run` for the fallback mechanics.
    pub output: OutputFormat,
    /// Default CLI args appended when this payload runs. Test bodies
    /// can extend via `.arg(...)` or replace via `.clear_args()` +
    /// `.arg(...)` on the runtime builder.
    pub default_args: &'static [&'static str],
    /// Author-declared default checks evaluated against extracted
    /// [`PayloadMetrics`]. Payloads that need exit-code gating
    /// should include [`MetricCheck::ExitCodeEq(0)`](MetricCheck::ExitCodeEq)
    /// here; the runtime evaluates `ExitCodeEq` as a pre-pass
    /// before metric checks.
    pub default_checks: &'static [MetricCheck],
    /// Declared metric hints — polarity, unit. Unhinted metrics
    /// extracted from output land as [`Polarity::Unknown`].
    pub metrics: &'static [MetricHint],
    /// Host-side file specs resolved at runtime. Each entry is
    /// resolved through the framework's include-file pipeline — the
    /// same resolver used by CLI `-i` / `--include-files` arguments:
    /// bare names are searched in the host's `PATH`, explicit paths
    /// (absolute, relative, or containing `/`) must exist on the
    /// host, and directories are walked recursively. The entry's
    /// scheduler / payload / workloads / extra_include_files are
    /// aggregated at test time via
    /// [`KtstrTestEntry::all_include_files`](crate::test_support::KtstrTestEntry::all_include_files)
    /// and resolved through the same pipeline the `ktstr shell -i`
    /// path uses. Populate via the
    /// `#[include_files("helper", ...)]` attribute on
    /// `#[derive(Payload)]` or by spelling the array in the struct
    /// literal.
    pub include_files: &'static [&'static str],
    /// When `true`, the payload's spawn path does NOT place the
    /// child into its own process group via
    /// `CommandExt::process_group(0)`. The child inherits the
    /// parent ktstr process's pgid. Default (`false`) keeps the
    /// existing "fresh pgrp → killpg-reaches-descendants" model
    /// — see `src/scenario/payload_run.rs::build_command`.
    ///
    /// Opt-in for tty-dependent binaries: a shell-like tool that
    /// uses the controlling terminal's foreground process group
    /// for signal delivery (job-control signals, SIGHUP on tty
    /// close) reads a fresh pgrp as "no job control", which
    /// breaks interactive shells and `less`-style readers.
    /// Payloads that need tty job-control semantics set this
    /// true so they stay in the parent's pgrp and keep the
    /// inherited controlling-terminal association.
    ///
    /// Trade-off on the `true` branch: multi-process payloads
    /// can no longer be killed via `killpg(child_pid, SIGKILL)`
    /// because the child is not a pgrp leader; the kill path
    /// falls back to single-pid `kill(pid, SIGKILL)` and any
    /// descendants that the payload forks must either react to
    /// SIGHUP / pipe close or run the risk of orphaning. Most
    /// payloads should leave this `false`.
    pub uses_parent_pgrp: bool,
    /// When `Some`, the listed flag names form an allowlist that
    /// `Op::RunPayload` validation checks against at scenario-
    /// execution time (inside `apply_ops`, before the payload
    /// spawn) — any user-supplied `--flag` whose name is not in
    /// the allowlist produces an error surfaced through the step
    /// executor, surfacing typos as loud errors instead of silent
    /// no-ops that only manifest as "feature didn't activate" in
    /// the test output.
    ///
    /// `None` (default) disables validation — the payload accepts
    /// arbitrary flag sets. Use `None` for payloads that wrap
    /// binaries with open-ended flag surfaces (stress-ng, fio,
    /// schbench) where enumerating every accepted flag is either
    /// impossible or high-churn.
    ///
    /// `Some(&[])` is legal but rarely intended: it rejects EVERY
    /// long flag, including ones the wrapped binary legitimately
    /// accepts. Use `None` for "no validation" and a non-empty
    /// slice for "validate against this allowlist" — an empty
    /// slice means "only positional args and short flags are
    /// acceptable", which is almost never what a Payload author
    /// wants.
    ///
    /// Flag names in the slice are bare (no leading `--`) and
    /// match the syntax of `Op::RunPayload`'s per-flag slot.
    pub known_flags: Option<&'static [&'static str]>,
}

impl std::fmt::Debug for Payload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Manual impl renders the payload's identity + slice counts
        // rather than the full struct (which would expand every
        // nested Scheduler/Sysctl/CgroupPath into a multi-screen
        // dump on every trace line). Scheduler IS Debug — the
        // derived impl would just be too verbose for the routine
        // tracing call sites this type appears in.
        f.debug_struct("Payload")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("output", &self.output)
            .field("default_args_len", &self.default_args.len())
            .field("default_checks_len", &self.default_checks.len())
            .field("metrics_len", &self.metrics.len())
            .finish()
    }
}

/// How a payload is launched inside the guest.
///
/// Two variants — scheduler and binary — map to the two launch paths
/// in the runtime. "Kernel default" (EEVDF) is represented as
/// `Scheduler(&Scheduler::EEVDF)` rather than a dedicated variant
/// because [`Scheduler`] already carries the no-userspace-binary
/// taxonomy via its own `binary: SchedulerSpec` field.
#[derive(Clone, Copy)]
pub enum PayloadKind {
    /// Wraps an existing [`Scheduler`] definition. The scheduler's
    /// own `binary: SchedulerSpec` carries the Eevdf/Discover/Path/
    /// KernelBuiltin taxonomy — no duplication at the Payload level.
    Scheduler(&'static Scheduler),
    /// Bare userspace binary looked up by name in the guest. Not a
    /// scheduler — runs as a workload under whatever scheduler the
    /// test declares.
    ///
    /// # How the binary reaches the guest
    ///
    /// The stored `&'static str` is the executable name passed to
    /// `std::process::Command::new` inside the guest (see
    /// [`PayloadRun::run`](crate::scenario::payload_run::PayloadRun::run)),
    /// which resolves it against the guest's `PATH`. The framework
    /// resolves binaries through the include-file pipeline — for
    /// `#[ktstr_test]` entries via declarative `include_files` /
    /// `extra_include_files`, or via `-i` on `ktstr shell`.
    ///
    /// Supply a binary through the framework's include-file
    /// pipeline. The pipeline is wired up to the `shell` subcommand
    /// of both `ktstr` and `cargo ktstr` through the repeatable
    /// `-i` / `--include-files` flag. Each `-i` argument accepts:
    ///
    /// - an explicit path (absolute, relative, or containing `/`) —
    ///   must exist on the host;
    /// - a bare name — searched in `PATH` on the host;
    /// - a directory — walked recursively, preserving structure under
    ///   `/include-files/<dirname>/...` in the guest.
    ///
    /// Every regular file ends up at `/include-files/<name>` (or
    /// deeper for directory walks). Dynamically-linked ELFs pull in
    /// their `DT_NEEDED` shared libraries automatically; the guest
    /// init prepends every `/include-files/*` subdirectory containing
    /// an executable to `PATH`, so a binary packaged with `-i` is
    /// runnable by bare name from a test body.
    ///
    /// Example — launch a shell VM with `fio` available by bare name:
    ///
    /// ```sh
    /// cargo ktstr shell -i fio --exec "fio --version"
    /// ```
    ///
    /// The `fio` binary is resolved against the host's `PATH`, copied
    /// to `/include-files/fio` in the guest, exposed on the guest
    /// `PATH`, and spawnable as `fio` from any guest-side process.
    ///
    /// # `#[ktstr_test]` entries
    ///
    /// Declarative `include_files` on `#[derive(Payload)]` and
    /// `extra_include_files` on `#[ktstr_test]` handle binary
    /// packaging automatically — no CLI `-i` and no bespoke harness
    /// needed.
    ///
    /// # Scheduler config files
    ///
    /// Scheduler-kind payloads that set
    /// [`Scheduler`]'s `config_file`
    /// field get automatic packaging: the config file is placed at
    /// `/include-files/{filename}` without a `-i` flag — the field
    /// is the source the harness reads.
    ///
    /// # Binary-kind packaging
    ///
    /// Payloads built via `#[derive(Payload)]` get automatic binary
    /// packaging: the derive macro prepends the `binary = "..."`
    /// spec to the emitted `include_files` slice, so the spawn
    /// target is packaged into the guest without requiring a
    /// separate `#[include_files("...")]` entry. Auxiliary files
    /// the payload needs (helpers, config files, fixtures) still
    /// go on `#[include_files(...)]` — the derive only injects the
    /// primary binary.
    ///
    /// Payloads constructed manually via struct literal (rather
    /// than the derive) do not get this auto-injection: the
    /// harness does not derive `include_files` from the
    /// `PayloadKind::Binary(name)` at aggregation time. Manual
    /// constructions must list the binary in
    /// [`Payload::include_files`](Payload::include_files)
    /// themselves, or declare it on
    /// [`extra_include_files`](crate::test_support::KtstrTestEntry::extra_include_files)
    /// at the `#[ktstr_test]` level. A binary referenced at spawn
    /// time but neither auto-injected nor listed as an include is
    /// expected to already be present in the guest filesystem
    /// (e.g. a standard `busybox` applet on the base image);
    /// otherwise the omission surfaces as `ENOENT` at `exec` time
    /// inside the guest.
    ///
    /// # Fork / kill semantics
    ///
    /// A binary-kind payload is spawned in its own process group via
    /// `CommandExt::process_group(0)` in
    /// [`build_command`](crate::scenario::payload_run) so the
    /// framework can reach every descendant the binary forks. Direct consequences for test
    /// authors:
    ///
    /// - `std::process::Child::kill()` only targets the direct child
    ///   — a `fork()`ed descendant (stress-ng worker, fio `--numjobs`,
    ///   schbench worker mode, pipeline subshells under `sh -c`)
    ///   survives. Never call `child.kill()` directly on a payload
    ///   `Child`; the handle's `kill()` wrapper fans out SIGKILL to
    ///   the whole process group via `killpg`.
    /// - [`PayloadHandle::kill`](crate::scenario::payload_run::PayloadHandle::kill),
    ///   [`PayloadHandle::wait`](crate::scenario::payload_run::PayloadHandle::wait)
    ///   cleanup, and the panic-safety Drop arm all route through
    ///   `kill_payload_process_group`, which issues `killpg(pgid,
    ///   SIGKILL)` followed by a single-pid SIGKILL fallback so
    ///   descendants and the leader both exit. This is the only kill
    ///   path test authors need.
    /// - Pipe drainers (stdout / stderr reader threads) block on EOF,
    ///   which only arrives after every descendant holding the
    ///   write ends closes them. A bare `child.kill()` leaves the
    ///   descendants holding the pipes open and
    ///   `wait_and_capture` hangs
    ///   forever — motivating the `killpg` requirement.
    Binary(&'static str),
}

impl std::fmt::Debug for PayloadKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Manual impl renders variant + identity summary (just the
        // scheduler/binary name) rather than the full Scheduler
        // struct that the derived Debug would expand. Trace lines
        // that print a PayloadKind want one-line attribution, not a
        // multi-screen scheduler config dump.
        match self {
            PayloadKind::Scheduler(s) => f.debug_tuple("Scheduler").field(&s.name).finish(),
            PayloadKind::Binary(name) => f.debug_tuple("Binary").field(name).finish(),
        }
    }
}

impl Payload {
    /// Placeholder payload that wraps the current kernel-default
    /// scheduler — [`Scheduler::EEVDF`] on Linux 6.6+ (the "no scx
    /// scheduler attached" case). Used as the default value of the
    /// `scheduler` slot on
    /// [`KtstrTestEntry`](crate::test_support::KtstrTestEntry) so
    /// tests without an explicit `scheduler = ...` attribute still
    /// get a valid, non-optional reference. Wire name is
    /// `"kernel_default"` — the Rust const and the serialized form
    /// agree, so the const describes what it selects for (the
    /// kernel's default) rather than naming a specific scheduler that
    /// a future kernel release could replace.
    ///
    /// ## `kernel_default` vs `eevdf` in sidecars
    ///
    /// `KERNEL_DEFAULT.name` is `"kernel_default"` (the intent-level
    /// label), while `KERNEL_DEFAULT.scheduler_name()` returns
    /// `"eevdf"` (the inner [`Scheduler::EEVDF`]'s `.name`). The two
    /// names answer different questions:
    ///
    /// - `"kernel_default"` answers "what did the test author select?"
    ///   — a future kernel release replacing EEVDF keeps this label
    ///   stable, so an in-memory match on author intent survives
    ///   kernel upgrades.
    /// - `"eevdf"` answers "what scheduler actually ran?" — the
    ///   concrete scheduling class in effect.
    ///
    /// **Sidecar serialization reads the scheduler-slot `Scheduler`
    /// directly.** The `SidecarResult.scheduler` field
    /// (src/test_support/sidecar/mod.rs) is populated by reading
    /// `entry.scheduler.name` — a field access on the
    /// `&'static Scheduler` held in the entry's scheduler slot, with
    /// no Payload indirection. When the scheduler slot holds
    /// `&Scheduler::EEVDF` (the framework default), the sidecar
    /// records `"eevdf"`. The outer `KERNEL_DEFAULT.name`
    /// (`"kernel_default"`) is NOT written to the sidecar — it stays
    /// in-memory only, used by logs, `#[ktstr_test]`-declaration
    /// lookups, and `Payload::display_name()` on the payload-slot
    /// surface. Cross-kernel-version comparisons via sidecar
    /// `scheduler` therefore see `"eevdf"` today and whatever future
    /// scheduling class replaces EEVDF tomorrow; author-intent
    /// filtering on `"kernel_default"` requires consulting the
    /// in-memory `Payload::name` directly, not the sidecar.
    pub const KERNEL_DEFAULT: Payload = Payload::new(
        "kernel_default",
        PayloadKind::Scheduler(&Scheduler::EEVDF),
        OutputFormat::ExitCode,
        &[],
        &[],
        &[],
        &[],
        false,
        None,
    );

    /// Short, human-readable name for logging and sidecar output.
    pub const fn display_name(&self) -> &'static str {
        self.name
    }

    /// Return the inner [`Scheduler`] reference when this payload
    /// wraps one. Returns `None` for [`PayloadKind::Binary`].
    pub const fn as_scheduler(&self) -> Option<&'static Scheduler> {
        match self.kind {
            PayloadKind::Scheduler(s) => Some(s),
            PayloadKind::Binary(_) => None,
        }
    }

    /// True when this payload wraps a [`Scheduler`] (scheduler
    /// slot). False for binary payloads.
    pub const fn is_scheduler(&self) -> bool {
        matches!(self.kind, PayloadKind::Scheduler(_))
    }

    /// Primary const constructor for a [`Payload`].
    ///
    /// Takes every field by position so the `#[derive(Payload)]`
    /// macro can emit a single call instead of a struct-literal.
    /// `#[non_exhaustive]` on the struct prevents out-of-crate
    /// struct-literal construction; this constructor — defined in
    /// the same crate as `Payload` — is not subject to that
    /// restriction, so the macro-expanded tokens that reach
    /// downstream crates compile cleanly.
    ///
    /// For one-field constructions prefer [`Payload::binary`] — it
    /// calls into this helper and pins the non-identity fields to
    /// the exit-code-only defaults.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        name: &'static str,
        kind: PayloadKind,
        output: OutputFormat,
        default_args: &'static [&'static str],
        default_checks: &'static [MetricCheck],
        metrics: &'static [MetricHint],
        include_files: &'static [&'static str],
        uses_parent_pgrp: bool,
        known_flags: Option<&'static [&'static str]>,
    ) -> Payload {
        Payload {
            name,
            kind,
            output,
            default_args,
            default_checks,
            metrics,
            include_files,
            uses_parent_pgrp,
            known_flags,
        }
    }

    /// Minimal const constructor for a binary-kind [`Payload`]. Fills
    /// the non-identity fields with the exit-code-only defaults — no
    /// CLI args, no author-declared checks, no metric hints, and
    /// [`OutputFormat::ExitCode`] — so a `#[ktstr_test]` entry or a
    /// direct unit test can declare a runnable binary with one line
    /// instead of spelling out the full struct literal.
    ///
    /// The `binary` string is the executable name passed to
    /// `std::process::Command::new` inside the guest. Supply it to
    /// the guest via `-i` / `--include-files` for CLI invocations or
    /// pre-install it in the initramfs for `#[ktstr_test]` entries —
    /// see [`PayloadKind::Binary`] for the full packaging contract.
    pub const fn binary(name: &'static str, binary: &'static str) -> Payload {
        Payload::new(
            name,
            PayloadKind::Binary(binary),
            OutputFormat::ExitCode,
            &[],
            &[],
            &[],
            &[],
            false,
            None,
        )
    }

    // -----------------------------------------------------------------
    // Scheduler-slot forwarding accessors
    //
    // These methods let every site that consumed `entry.scheduler:
    // &Scheduler` read the equivalent field off `entry.scheduler:
    // &Payload` without the caller having to unwrap
    // `as_scheduler()`. For a scheduler-kind payload the accessor
    // forwards to the inner `Scheduler`. For a binary-kind payload
    // the accessor returns a sensible default — usually the empty
    // slice or the no-op value — matching the semantics a binary
    // payload in the scheduler slot should carry (no sysctls, no
    // kargs, no scheduler-specific CLI flags).
    //
    // The binary-kind branch is not "best effort": a binary payload
    // in the scheduler slot is a valid configuration (pure userspace
    // test under the kernel default scheduler), and every accessor
    // below returns exactly what that scenario should see.
    // -----------------------------------------------------------------

    /// The scheduler's display name.
    ///
    /// Returns a compile-time-fixed LABEL, not a runtime reflection
    /// of the scheduling class the live kernel is actually running.
    /// A sidecar written on a kernel whose default is a successor
    /// scheduling class still records whatever string this method
    /// returns — the label comes from the `Payload` / inner
    /// `Scheduler` definition, nothing queries `/proc` or the live
    /// policy. Consumers that need to know the running kernel's
    /// scheduling class must cross-reference the sidecar's
    /// `host.kernel_release` with kernel-version-to-scheduler
    /// knowledge maintained outside the sidecar.
    ///
    /// Branch behavior:
    /// - `PayloadKind::Scheduler(s)` → `s.name` — the label attached
    ///   to that specific scheduler, e.g. `"eevdf"` for
    ///   [`Scheduler::EEVDF`] or `"scx_rusty"` for a scx_*
    ///   scheduler. This is what scheduler-kind payloads (including
    ///   `Payload::KERNEL_DEFAULT`, which wraps [`Scheduler::EEVDF`])
    ///   surface.
    /// - `PayloadKind::Binary(_)` → `"kernel_default"` — a binary
    ///   payload runs under whatever scheduler the test declares
    ///   elsewhere (or the kernel default if it declares none), so
    ///   the binary-kind payload carries no scheduler identity of
    ///   its own. The returned string is a LABEL ("test author did
    ///   not pin a scheduler here"), NOT a statement about which
    ///   scheduling class the VM actually ran under — the live
    ///   kernel may be running EEVDF, a successor class, or an scx
    ///   scheduler the binary's test harness attached separately;
    ///   `scheduler_name()` does not observe any of that. Only a
    ///   scheduler-kind payload explicitly wrapping
    ///   [`Scheduler::EEVDF`] returns the `"eevdf"` label; every
    ///   binary-kind payload returns `"kernel_default"` regardless
    ///   of what class is running.
    pub const fn scheduler_name(&self) -> &'static str {
        match self.kind {
            PayloadKind::Scheduler(s) => s.name,
            PayloadKind::Binary(_) => "kernel_default",
        }
    }

    /// The scheduler's binary spec when scheduler-kind; `None` for
    /// binary-kind payloads. Consumers that dispatch on the
    /// `SchedulerSpec` variant (e.g. `KernelBuiltin { enable, disable }`
    /// hook invocation) use this rather than the `scheduler_name`
    /// shortcut.
    pub const fn scheduler_binary(&self) -> Option<&'static crate::test_support::SchedulerSpec> {
        match self.kind {
            PayloadKind::Scheduler(s) => Some(&s.binary),
            PayloadKind::Binary(_) => None,
        }
    }

    /// True when this payload drives an active scheduling policy
    /// (anything other than the kernel default EEVDF). Forwards to
    /// `SchedulerSpec::has_active_scheduling` for scheduler-kind
    /// payloads; binary-kind payloads always return `false` — a
    /// binary runs under whatever scheduler the test declares, and
    /// does not itself impose one.
    ///
    /// Returns `true` for `KernelBuiltin` scheduler-kind payloads.
    /// See [`Self::has_bpf_scheduler`] for the narrower gate that
    /// excludes them.
    pub const fn has_active_scheduling(&self) -> bool {
        match self.kind {
            PayloadKind::Scheduler(s) => s.binary.has_active_scheduling(),
            PayloadKind::Binary(_) => false,
        }
    }

    /// True when this payload drives a userspace BPF scheduler
    /// binary. Forwards to `SchedulerSpec::has_bpf_scheduler` for
    /// scheduler-kind payloads; binary-kind payloads always return
    /// `false`.
    ///
    /// Distinct from [`Self::has_active_scheduling`]: this excludes
    /// `KernelBuiltin` scheduler-kind payloads (in-kernel policy,
    /// no userspace BPF binary). Use whenever the caller assumes a
    /// BPF binary is attached — verifier_stats wiring, BPF-attach
    /// monitor thresholds, or auto-repro probe gating.
    pub const fn has_bpf_scheduler(&self) -> bool {
        match self.kind {
            PayloadKind::Scheduler(s) => s.binary.has_bpf_scheduler(),
            PayloadKind::Binary(_) => false,
        }
    }

    /// Guest sysctls applied before the scheduler starts. Empty slice
    /// for binary-kind payloads.
    pub const fn sysctls(&self) -> &'static [crate::test_support::Sysctl] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.sysctls,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Extra guest kernel command-line arguments appended when
    /// booting the VM. Empty slice for binary-kind payloads.
    pub const fn kargs(&self) -> &'static [&'static str] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.kargs,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Scheduler CLI args prepended before per-test `extra_sched_args`.
    /// Empty slice for binary-kind payloads.
    pub const fn sched_args(&self) -> &'static [&'static str] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.sched_args,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Cgroup parent path. `None` for binary-kind payloads and for
    /// scheduler-kind payloads that did not set one.
    pub const fn cgroup_parent(&self) -> Option<crate::test_support::CgroupPath> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.cgroup_parent,
            PayloadKind::Binary(_) => None,
        }
    }

    /// Host-side path to the scheduler config file. `None` for
    /// binary-kind payloads and for scheduler-kind payloads that
    /// did not set one.
    pub const fn config_file(&self) -> Option<&'static str> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.config_file,
            PayloadKind::Binary(_) => None,
        }
    }

    /// Inline config file definition. `None` for binary-kind payloads
    /// and for scheduler-kind payloads that did not set one.
    pub const fn config_file_def(&self) -> Option<(&'static str, &'static str)> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.config_file_def,
            PayloadKind::Binary(_) => None,
        }
    }

    /// Scheduler-wide assertion overrides. For binary-kind payloads
    /// returns `Assert::NO_OVERRIDES` — the default identity value
    /// merge that leaves per-entry assertions untouched.
    pub const fn assert(&self) -> &'static crate::assert::Assert {
        match self.kind {
            PayloadKind::Scheduler(s) => &s.assert,
            PayloadKind::Binary(_) => &crate::assert::Assert::NO_OVERRIDES,
        }
    }

    /// Default VM topology for this payload. Scheduler-kind payloads
    /// expose the topology declared on the inner `Scheduler` so tests
    /// that inherit from the scheduler slot stay consistent with the
    /// rest of the scheduler's test surface; binary-kind payloads
    /// return a minimal placeholder
    /// ([`Topology::DEFAULT_FOR_PAYLOAD`](crate::test_support::Topology::DEFAULT_FOR_PAYLOAD))
    /// — a pure binary workload has no scheduler-level topology
    /// opinion, so per-entry `#[ktstr_test(...)]` overrides are what
    /// actually drive the VM shape.
    pub const fn topology(&self) -> crate::test_support::Topology {
        match self.kind {
            PayloadKind::Scheduler(s) => s.topology,
            PayloadKind::Binary(_) => crate::test_support::Topology::DEFAULT_FOR_PAYLOAD,
        }
    }

    /// Gauntlet topology constraints. Scheduler-kind payloads forward
    /// to the inner `Scheduler::constraints`; binary-kind payloads
    /// return `TopologyConstraints::DEFAULT`.
    pub const fn constraints(&self) -> crate::test_support::TopologyConstraints {
        match self.kind {
            PayloadKind::Scheduler(s) => s.constraints,
            PayloadKind::Binary(_) => crate::test_support::TopologyConstraints::DEFAULT,
        }
    }
}

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// How the framework extracts metrics from a payload's output.
///
/// `ExitCode` records only the exit code; no text parsing. `Json`
/// finds a JSON document region and walks numeric leaves into
/// [`Metric`] values.
///
/// For `Json`, extraction is stdout-primary with a stderr fallback:
/// the extractor runs first against stdout, and only when that
/// yields an empty metric set AND stderr is non-empty does it retry
/// against stderr. Well-behaved binaries keep stdout canonical;
/// payloads that emit structured output only on stderr (schbench's
/// `show_latencies` → `fprintf(stderr, ...)`) still parse. The
/// streams are never merged. `ExitCode` produces no metrics from
/// either stream — `extract_metrics` is invoked (the control flow
/// is variant-agnostic for simplicity) but the `ExitCode` arm
/// returns `Ok(vec![])` immediately, so the stderr fallback runs
/// and also returns empty. Observable behavior: exit code only, no
/// metrics.
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Pass/fail from exit code alone. Stdout is archived for
    /// debugging but not parsed. `extract_metrics` is still invoked
    /// in the evaluate pipeline (variant-agnostic control flow) but
    /// returns `Ok(vec![])` immediately for this variant; the
    /// stderr fallback runs too and also returns empty. Observable
    /// behavior: no metrics extracted regardless of stream content.
    ExitCode,
    /// Parse the primary stream (stdout, or stderr on fallback) as
    /// JSON: find the JSON region within mixed output, extract
    /// numeric leaves as metrics keyed by dotted path (e.g.
    /// `jobs.0.read.iops`).
    Json,
}

// ---------------------------------------------------------------------------
// Polarity, MetricCheck, Metric
// ---------------------------------------------------------------------------

/// Regression direction for a metric.
///
/// Used by `cargo ktstr test-stats` to classify deltas between runs.
/// Declared explicitly on [`MetricHint`]; unhinted metrics default to
/// [`Polarity::Unknown`] and are recorded without regression
/// classification.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Polarity {
    /// Bigger is better (throughput, IOPS, bogo_ops/sec). Regression
    /// = decrease from baseline.
    HigherBetter,
    /// Smaller is better (latency percentiles, error rates).
    /// Regression = increase from baseline.
    LowerBetter,
    /// A target value that the metric should hover near. Regression
    /// = absolute distance exceeds a threshold, symmetric in either
    /// direction. The inner `f64` MUST be finite (not NaN/inf);
    /// construct via [`Polarity::target`], which enforces this at
    /// runtime in both debug and release.
    TargetValue(f64),
    /// Direction not declared; the metric is recorded but not
    /// classified as regression-relevant. This is the CONSERVATIVE
    /// default for an UNCLASSIFIED metric: the comparison path treats
    /// it as higher-is-worse (see `MetricDef::higher_is_worse` /
    /// `classify_direction`) so a real regression in a metric someone
    /// forgot to classify is still caught rather than silently ignored.
    Unknown,
    /// Deliberately directionless: the metric is recorded and DISPLAYED
    /// in comparisons but is NEVER classified as a regression or
    /// improvement and NEVER affects the exit code. Distinct from
    /// [`Polarity::Unknown`] (the conservative higher-is-worse default
    /// for *unclassified* metrics): `Informational` is the explicit
    /// "this counter has no good/bad direction" choice — e.g. wakeup /
    /// context-switch / yield counts, where more is neither inherently
    /// better nor worse. `classify_direction` returns `None` for it.
    Informational,
}

impl Polarity {
    /// Map the legacy `higher_is_worse: bool` used by
    /// `MetricDef` to a `Polarity`.
    ///
    /// The sense is INVERSE: `true` (bigger values are regressions)
    /// maps to [`Polarity::LowerBetter`] (we want the metric to go
    /// down); `false` maps to [`Polarity::HigherBetter`].
    pub const fn from_higher_is_worse(higher_is_worse: bool) -> Polarity {
        if higher_is_worse {
            Polarity::LowerBetter
        } else {
            Polarity::HigherBetter
        }
    }

    /// The regression direction of this polarity: `Some(true)` = an INCREASE is
    /// a regression (`LowerBetter` / `TargetValue` / the conservative `Unknown`),
    /// `Some(false)` = a DECREASE is a regression (`HigherBetter`), `None` =
    /// directionless (`Informational`, never gates). The single source of truth
    /// that `MetricDef::classify_direction` delegates to.
    pub const fn classify_direction(&self) -> Option<bool> {
        match self {
            Polarity::Informational => None,
            Polarity::HigherBetter => Some(false),
            Polarity::LowerBetter | Polarity::TargetValue(_) | Polarity::Unknown => Some(true),
        }
    }

    /// Construct a [`Polarity::TargetValue`] from a finite `target`.
    ///
    /// # Panics
    ///
    /// Panics in both debug and release builds when `target` is not
    /// finite (`NaN`, `+inf`, `-inf`). Non-finite values produce
    /// incorrect regression verdicts in the comparison pipeline, so
    /// the check runs unconditionally rather than via
    /// `debug_assert!`.
    pub fn target(target: f64) -> Polarity {
        assert!(
            target.is_finite(),
            "Polarity::TargetValue target must be finite, got {target}"
        );
        Polarity::TargetValue(target)
    }
}

/// Payload-author metric declaration: polarity + display unit.
///
/// Attached to a [`Payload`] via the `metrics` field. Metrics
/// extracted from output are looked up against this table by name to
/// set their [`Polarity`] and [`Metric::unit`]. Unmatched metrics
/// land with `Polarity::Unknown` and an empty unit string.
#[derive(Debug, Clone, Copy)]
pub struct MetricHint {
    /// Dotted-path metric name (e.g. `jobs.0.read.iops`).
    pub name: &'static str,
    /// Regression direction for this metric.
    pub polarity: Polarity,
    /// Human-readable unit for display (e.g. `iops`, `ns`). Empty
    /// string means "no unit"; matches the sentinel used by
    /// `MetricDef`.
    pub unit: &'static str,
}

/// Assertion check evaluated against an extracted
/// [`PayloadMetrics`] (or the exit code for
/// [`MetricCheck::ExitCodeEq`]).
#[derive(Debug, Clone, Copy)]
pub enum MetricCheck {
    /// Fail when the named metric is below `value`.
    Min { metric: &'static str, value: f64 },
    /// Fail when the named metric exceeds `value`.
    Max { metric: &'static str, value: f64 },
    /// Fail when the named metric is outside `[lo, hi]`.
    Range {
        metric: &'static str,
        lo: f64,
        hi: f64,
    },
    /// Fail when the named metric is missing from the extracted set.
    Exists(&'static str),
    /// Fail when the payload's exit code is not equal to `expected`.
    ExitCodeEq(i32),
}

impl MetricCheck {
    /// Fail when the named metric is below `value`. Missing metric
    /// fails loudly per the evaluation pipeline's missing-metric
    /// contract.
    pub const fn min(metric: &'static str, value: f64) -> MetricCheck {
        MetricCheck::Min { metric, value }
    }

    /// Fail when the named metric exceeds `value`. Missing metric
    /// fails loudly.
    pub const fn max(metric: &'static str, value: f64) -> MetricCheck {
        MetricCheck::Max { metric, value }
    }

    /// Fail when the named metric falls outside `[lo, hi]` (inclusive
    /// on both ends). Missing metric fails loudly.
    ///
    /// # Panics
    ///
    /// Panics at construction when `lo > hi` — a reversed-bounds
    /// range describes an empty interval that no finite metric can
    /// satisfy, almost certainly a user error rather than an
    /// intentional always-fails check. Failing at the constructor
    /// surfaces the typo at the call site instead of letting the
    /// evaluator run an unsatisfiable check against every probe
    /// value. NaN bounds also trip this gate because `lo <= hi`
    /// is false for any NaN argument.
    pub const fn range(metric: &'static str, lo: f64, hi: f64) -> MetricCheck {
        assert!(
            lo <= hi,
            "MetricCheck::range: lo must be <= hi (reversed bounds are an empty interval)"
        );
        MetricCheck::Range { metric, lo, hi }
    }

    /// Fail when the named metric is absent from the extracted set.
    /// Presence-only — the metric value can be any finite number,
    /// including zero or negative.
    pub const fn exists(metric: &'static str) -> MetricCheck {
        MetricCheck::Exists(metric)
    }

    /// Fail when the payload's exit code differs from `expected`.
    /// Evaluated before metric-path checks so a mis-exited binary
    /// reports the exit-code mismatch rather than chained
    /// missing-metric failures.
    pub const fn exit_code_eq(expected: i32) -> MetricCheck {
        MetricCheck::ExitCodeEq(expected)
    }

    /// Typed sibling of [`Self::min`] — a typo-proof `BuiltinMetric` instead of a
    /// registry-name string. `const` (via `BuiltinMetric::wire_name`) so it
    /// composes in `const` payload-check tables exactly like [`Self::min`]. Use
    /// this for a registered built-in metric; keep [`Self::min`] for the dynamic
    /// keyspace (dotted JSON-leaf paths from [`OutputFormat::Json`], scheduler-runtime keys).
    ///
    /// ```
    /// use ktstr::prelude::*;
    /// // The typed checks compose into a `const` table — exactly the
    /// // `&'static [MetricCheck]` a `Payload` carries in `default_checks`:
    /// const CHECKS: &[MetricCheck] = &[
    ///     MetricCheck::min_builtin(BuiltinMetric::TaobenchTotalQps, 1000.0),
    ///     MetricCheck::exists_builtin(BuiltinMetric::SchbenchLoopCount),
    /// ];
    /// assert_eq!(CHECKS.len(), 2);
    /// ```
    pub const fn min_builtin(metric: crate::stats::BuiltinMetric, value: f64) -> MetricCheck {
        MetricCheck::Min {
            metric: metric.wire_name(),
            value,
        }
    }

    /// Typed sibling of [`Self::max`] — see [`Self::min_builtin`].
    pub const fn max_builtin(metric: crate::stats::BuiltinMetric, value: f64) -> MetricCheck {
        MetricCheck::Max {
            metric: metric.wire_name(),
            value,
        }
    }

    /// Typed sibling of [`Self::range`] — see [`Self::min_builtin`]. Same
    /// reversed-bounds construction panic as [`Self::range`].
    pub const fn range_builtin(
        metric: crate::stats::BuiltinMetric,
        lo: f64,
        hi: f64,
    ) -> MetricCheck {
        assert!(
            lo <= hi,
            "MetricCheck::range_builtin: lo must be <= hi (reversed bounds are an empty interval)"
        );
        MetricCheck::Range {
            metric: metric.wire_name(),
            lo,
            hi,
        }
    }

    /// Typed sibling of [`Self::exists`] — see [`Self::min_builtin`].
    pub const fn exists_builtin(metric: crate::stats::BuiltinMetric) -> MetricCheck {
        MetricCheck::Exists(metric.wire_name())
    }
}

/// Which of the payload's output streams a [`Metric`] was extracted
/// from.
///
/// Captures WHERE the bytes came from (payload stdout vs stderr).
/// This matters for diagnosing "surprise metrics" in post-run
/// analysis: a metric tagged [`Self::Stderr`] signals a payload
/// whose structured output landed on the diagnostic stream —
/// well-behaved payloads keep stdout canonical per the
/// [`OutputFormat`] doc contract, so a stderr tag is a review hint
/// ("is this payload misconfigured, or did the fallback
/// intentionally pick it up?").
///
/// Populated by the extraction pipeline in
/// [`crate::scenario::payload_run`]: the stdout-primary branch
/// stamps [`Stdout`](Self::Stdout), the stderr-fallback branch
/// stamps [`Stderr`](Self::Stderr). The streams are never merged;
/// one or the other produces the metric set, and that identity
/// propagates through [`Metric::stream`].
///
/// Status: persisted on the sidecar for future review-tooling
/// (CI dashboards, `cargo ktstr stats`-style filters); not yet
/// consumed by `stats compare` or any automated pipeline. The
/// field is wired end-to-end from the payload-pipeline to the
/// sidecar JSON today so that downstream review tools can start
/// filtering on it without a schema change — but no production
/// consumer reads it yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
// `#[non_exhaustive]`: kept so a future probe-authored stream source
// (e.g. a BPF-map reader) can land without a sidecar wire-format
// migration, and so downstream matches must retain a `_ =>` arm. The
// forward-compat guarantee is independent of the current variant count
// — do not strip this on the grounds that the enum has only two
// variants today.
#[non_exhaustive]
pub enum MetricStream {
    /// Extracted from the payload's stdout (the happy path for
    /// fio / stress-ng / most benchmark tools).
    Stdout,
    /// Extracted from the payload's stderr via the stderr-fallback
    /// contract (for payloads that emit structured summaries to
    /// stderr — e.g. schbench's `show_latencies` →
    /// `fprintf(stderr, ...)`).
    Stderr,
}

/// A single extracted metric from a payload's output.
///
/// Populated by the extraction pipeline after the payload exits.
/// Sidecar serialization carries these alongside the pass/fail
/// verdict so test-stats can classify regressions across runs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Metric {
    /// Dotted-path name matching the JSON leaf.
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// Regression direction, copied from the matching
    /// [`MetricHint`] or left as [`Polarity::Unknown`] when no hint
    /// matches.
    pub polarity: Polarity,
    /// Display unit string; empty when no unit was declared.
    pub unit: String,
    /// Which of the payload's output streams the metric was read
    /// from — stdout on the happy path, stderr under the
    /// stderr-fallback contract. See [`MetricStream`] for the
    /// "well-behaved payloads keep stdout canonical" review hint.
    pub stream: MetricStream,
}

/// All metrics extracted from a single payload run plus the process
/// exit code.
///
/// Each concurrent payload (primary or workload, foreground or
/// background) produces one `PayloadMetrics` value. Sidecar stores
/// these as a `Vec<PayloadMetrics>` so per-payload provenance is
/// preserved across composed tests. Payload identity (name and
/// cgroup placement) is carried by the enclosing sidecar record —
/// not by `PayloadMetrics` itself, which holds only the extracted
/// metrics and exit code.
///
/// The `payload_index` field stamps every per-invocation emission
/// (one `PayloadMetrics` value per `.run()` / `.wait()` / `.kill()` /
/// `.try_wait()` call) with a monotonically increasing per-process
/// counter — assigned at emit time inside the guest VM.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PayloadMetrics {
    /// Per-invocation index assigned at emit time. Monotonically
    /// increasing within a single guest VM process, starting at 0.
    pub payload_index: usize,
    /// Extracted metrics. Empty when [`OutputFormat::ExitCode`] is
    /// used or when JSON parsing found no numeric leaves.
    pub metrics: Vec<Metric>,
    /// Process exit code (0 = success). Used by
    /// [`MetricCheck::ExitCodeEq`] in the check
    /// evaluation pre-pass.
    pub exit_code: i32,
}

impl PayloadMetrics {
    /// Look up a metric by exact name. Returns `None` when the
    /// metric is not in the set.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_check_builtin_constructors_carry_the_wire_name() {
        use crate::stats::BuiltinMetric;
        // The typed const constructors store the same wire name as the &str
        // form, so a built-in payload check is typo-proof without changing the
        // stored key. (MetricCheck is Copy, not PartialEq, so match on the arm.)
        match MetricCheck::min_builtin(BuiltinMetric::TaobenchTotalQps, 1000.0) {
            MetricCheck::Min { metric, value } => {
                assert_eq!(metric, "taobench_total_qps");
                assert_eq!(value, 1000.0);
            }
            other => panic!("min_builtin must produce Min, got {other:?}"),
        }
        match MetricCheck::max_builtin(BuiltinMetric::TaobenchSlowQps, 5.0) {
            MetricCheck::Max { metric, value } => {
                assert_eq!(metric, "taobench_slow_qps");
                assert_eq!(value, 5.0);
            }
            other => panic!("max_builtin must produce Max, got {other:?}"),
        }
        match MetricCheck::range_builtin(BuiltinMetric::WakeupP99LatencyUs, 1.0, 2.0) {
            MetricCheck::Range { metric, lo, hi } => {
                assert_eq!(metric, "wakeup_p99_latency_us");
                assert_eq!((lo, hi), (1.0, 2.0));
            }
            other => panic!("range_builtin must produce Range, got {other:?}"),
        }
        match MetricCheck::exists_builtin(BuiltinMetric::SchbenchLoopCount) {
            MetricCheck::Exists(metric) => assert_eq!(metric, "schbench_loop_count"),
            other => panic!("exists_builtin must produce Exists, got {other:?}"),
        }
    }

    #[test]
    fn metric_check_builtin_constructors_compose_into_a_const_default_checks_table() {
        use crate::stats::BuiltinMetric;
        // The *_builtin constructors are `const fn`, so a typed check table is a
        // plain `const` `&'static [MetricCheck]` — exactly the type of
        // `Payload::default_checks`. This pins const-composition (the property the
        // `const fn` buys) and documents the intended usage: a typo-proof
        // default_checks table built from BuiltinMetric variants, not bare strings.
        const CHECKS: &[MetricCheck] = &[
            MetricCheck::min_builtin(BuiltinMetric::TaobenchTotalQps, 1000.0),
            MetricCheck::max_builtin(BuiltinMetric::TaobenchSlowQps, 5.0),
            MetricCheck::range_builtin(BuiltinMetric::WakeupP99LatencyUs, 1.0, 2.0),
            MetricCheck::exists_builtin(BuiltinMetric::SchbenchLoopCount),
        ];
        // Same wire names the string constructors would store, in declaration order.
        assert_eq!(CHECKS.len(), 4);
        assert!(matches!(
            CHECKS[0],
            MetricCheck::Min {
                metric: "taobench_total_qps",
                ..
            }
        ));
        assert!(matches!(
            CHECKS[1],
            MetricCheck::Max {
                metric: "taobench_slow_qps",
                ..
            }
        ));
        assert!(matches!(
            CHECKS[2],
            MetricCheck::Range {
                metric: "wakeup_p99_latency_us",
                ..
            }
        ));
        assert!(matches!(
            CHECKS[3],
            MetricCheck::Exists("schbench_loop_count")
        ));

        // The typed table is a `&'static [MetricCheck]` (the const's lifetime is
        // elided) — it drops straight into a Payload's default_checks slot (the
        // composition the const fn enables).
        const _: &[MetricCheck] = CHECKS;
    }

    #[test]
    fn payload_kernel_default_const_is_scheduler_kind() {
        assert!(matches!(
            Payload::KERNEL_DEFAULT.kind,
            PayloadKind::Scheduler(_)
        ));
        assert_eq!(Payload::KERNEL_DEFAULT.display_name(), "kernel_default");
        assert!(matches!(
            Payload::KERNEL_DEFAULT.output,
            OutputFormat::ExitCode
        ));
        assert!(Payload::KERNEL_DEFAULT.default_args.is_empty());
        assert!(Payload::KERNEL_DEFAULT.default_checks.is_empty());
        assert!(Payload::KERNEL_DEFAULT.metrics.is_empty());
    }

    #[test]
    fn payload_kernel_default_wraps_scheduler_eevdf() {
        match Payload::KERNEL_DEFAULT.kind {
            PayloadKind::Scheduler(s) => {
                assert_eq!(s.name, Scheduler::EEVDF.name);
            }
            PayloadKind::Binary(_) => panic!("EEVDF should be Scheduler-kind, got Binary"),
        }
    }

    /// [`Payload::binary`] fills a binary-kind [`Payload`] with the
    /// exit-code-only defaults — empty `default_args`,
    /// `default_checks`, `metrics`, and `OutputFormat::ExitCode`.
    /// Evaluated in a `const` block so any future drift that makes
    /// the constructor non-const surfaces here at compile time; the
    /// runtime assertions pin the field-level defaults so a
    /// drive-by change (e.g. flipping `output` to `Json`) reshapes
    /// every `Payload::binary(…)` call site visibly.
    #[test]
    fn payload_binary_const_constructor_shape() {
        const P: Payload = Payload::binary("fio_payload", "fio");
        assert_eq!(P.name, "fio_payload");
        assert!(matches!(P.kind, PayloadKind::Binary("fio")));
        assert!(matches!(P.output, OutputFormat::ExitCode));
        assert!(P.default_args.is_empty());
        assert!(P.default_checks.is_empty());
        assert!(P.metrics.is_empty());
        assert!(!P.is_scheduler());
        assert!(P.as_scheduler().is_none());
    }

    #[test]
    fn check_constructors() {
        assert!(matches!(
            MetricCheck::min("x", 1.0),
            MetricCheck::Min { .. }
        ));
        assert!(matches!(
            MetricCheck::max("x", 1.0),
            MetricCheck::Max { .. }
        ));
        assert!(matches!(
            MetricCheck::range("x", 1.0, 2.0),
            MetricCheck::Range { .. }
        ));
        assert!(matches!(MetricCheck::exists("x"), MetricCheck::Exists("x")));
        assert!(matches!(
            MetricCheck::exit_code_eq(0),
            MetricCheck::ExitCodeEq(0)
        ));
    }

    #[test]
    fn metric_set_get_returns_value() {
        let pm = PayloadMetrics {
            payload_index: 0,
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 1000.0,
                polarity: Polarity::HigherBetter,
                unit: "iops".to_string(),
                stream: MetricStream::Stdout,
            }],
            exit_code: 0,
        };
        assert_eq!(pm.get("iops"), Some(1000.0));
        assert_eq!(pm.get("missing"), None);
    }

    #[test]
    fn polarity_target_value_carries_data() {
        let p = Polarity::TargetValue(42.0);
        match p {
            Polarity::TargetValue(v) => assert_eq!(v, 42.0),
            _ => panic!("expected TargetValue variant"),
        }
    }

    /// Wire-format round-trip for every [`MetricStream`] variant.
    /// Pins the serde representation so a sidecar written by one
    /// version of ktstr deserializes under another — a silent wire
    /// change (rename, internal tag, numeric encoding) would
    /// surface here, not as a missing-field error on every
    /// existing sidecar.
    #[test]
    fn metric_stream_serde_round_trip() {
        for s in [MetricStream::Stdout, MetricStream::Stderr] {
            let js = serde_json::to_string(&s).expect("serialize");
            let de: MetricStream = serde_json::from_str(&js).expect("deserialize");
            assert_eq!(
                de, s,
                "MetricStream::{s:?} wire format must round-trip \
                 identically; serialized as {js}, deserialized to \
                 {de:?}",
            );
        }
    }

    #[test]
    fn polarity_serde_round_trip() {
        for p in [
            Polarity::HigherBetter,
            Polarity::LowerBetter,
            Polarity::TargetValue(2.78),
            Polarity::Unknown,
        ] {
            let js = serde_json::to_string(&p).unwrap();
            let de: Polarity = serde_json::from_str(&js).unwrap();
            assert_eq!(de, p);
        }
    }

    // PayloadKind::Binary construction + pattern match.
    #[test]
    fn payload_kind_binary_construction_and_match() {
        const FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        match FIO.kind {
            PayloadKind::Binary(name) => assert_eq!(name, "fio"),
            PayloadKind::Scheduler(_) => panic!("expected Binary, got Scheduler"),
        }
        assert!(!FIO.is_scheduler());
        assert!(FIO.as_scheduler().is_none());
    }

    // Const bindings verify const-fn actually works in const context.
    const _MIN: MetricCheck = MetricCheck::min("x", 1.0);
    const _MAX: MetricCheck = MetricCheck::max("x", 2.0);
    const _RANGE: MetricCheck = MetricCheck::range("x", 1.0, 2.0);
    const _EXISTS: MetricCheck = MetricCheck::exists("x");
    const _EXIT: MetricCheck = MetricCheck::exit_code_eq(0);
    const _KERNEL_DEFAULT_REF: &Payload = &Payload::KERNEL_DEFAULT;
    const _KERNEL_DEFAULT_IS_SCHED: bool = Payload::KERNEL_DEFAULT.is_scheduler();
    const _KERNEL_DEFAULT_DISPLAY: &str = Payload::KERNEL_DEFAULT.display_name();

    // Proves an arbitrary `Payload` (not just `Payload::KERNEL_DEFAULT`) is
    // const-constructible via struct literal — the #[derive(Payload)]
    // proc-macro emits exactly this shape.
    const _PAYLOAD_CONST_BUILD: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &["--output-format=json"],
        default_checks: &[MetricCheck::exit_code_eq(0)],
        metrics: &[MetricHint {
            name: "jobs.0.read.iops",
            polarity: Polarity::HigherBetter,
            unit: "iops",
        }],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
    };

    #[test]
    fn const_bindings_are_usable() {
        assert!(matches!(_MIN, MetricCheck::Min { .. }));
        assert!(matches!(_MAX, MetricCheck::Max { .. }));
        assert!(matches!(_RANGE, MetricCheck::Range { .. }));
        assert!(matches!(_EXISTS, MetricCheck::Exists("x")));
        assert!(matches!(_EXIT, MetricCheck::ExitCodeEq(0)));
        assert_eq!(_KERNEL_DEFAULT_REF.name, "kernel_default");
        const { assert!(_KERNEL_DEFAULT_IS_SCHED) };
        assert_eq!(_KERNEL_DEFAULT_DISPLAY, "kernel_default");
    }

    // from_higher_is_worse helper.
    #[test]
    fn polarity_from_higher_is_worse_flips_sense() {
        assert_eq!(Polarity::from_higher_is_worse(true), Polarity::LowerBetter);
        assert_eq!(
            Polarity::from_higher_is_worse(false),
            Polarity::HigherBetter
        );
    }

    /// Round-trip bool → Polarity → bool for HigherBetter /
    /// LowerBetter yields the identity. Pins the "inverse sense"
    /// contract documented on `MetricDef::higher_is_worse` and
    /// `Polarity::from_higher_is_worse` so a future polarity
    /// refactor can't accidentally flip one direction without the
    /// other and silently break delta-classification downstream.
    ///
    /// The test synthesizes a throw-away `MetricDef` for each
    /// polarity because the production `METRICS` table's entries
    /// live in `stats.rs` and are test-only not importable from
    /// here — constructing the struct literal directly keeps the
    /// round-trip self-contained.
    #[test]
    fn higher_is_worse_polarity_round_trip() {
        use crate::stats::{MetricDef, MetricKind};

        // true (higher-is-worse) → LowerBetter → true.
        let m = MetricDef {
            name: "t",
            polarity: Polarity::from_higher_is_worse(true),
            kind: MetricKind::Counter,
            default_abs: 0.0,
            default_rel: 0.0,
            display_unit: "",
            accessor: |_| None,
        };
        assert_eq!(m.polarity, Polarity::LowerBetter);
        assert!(m.higher_is_worse(), "LowerBetter → higher_is_worse = true");

        // false (higher-is-better) → HigherBetter → false.
        let m = MetricDef {
            name: "f",
            polarity: Polarity::from_higher_is_worse(false),
            kind: MetricKind::Counter,
            default_abs: 0.0,
            default_rel: 0.0,
            display_unit: "",
            accessor: |_| None,
        };
        assert_eq!(m.polarity, Polarity::HigherBetter);
        assert!(
            !m.higher_is_worse(),
            "HigherBetter → higher_is_worse = false"
        );
    }

    /// `MetricDef::higher_is_worse` is total over every `Polarity`
    /// variant — the current implementation lumps `LowerBetter`,
    /// `TargetValue`, and `Unknown` all into `true`. Pinned so a
    /// subtle change (e.g. TargetValue → its own category) doesn't
    /// silently flip regression direction for every test using
    /// target metrics.
    #[test]
    fn higher_is_worse_covers_all_polarity_variants() {
        use crate::stats::{MetricDef, MetricKind};
        fn make(p: Polarity) -> MetricDef {
            MetricDef {
                name: "x",
                polarity: p,
                kind: MetricKind::Counter,
                default_abs: 0.0,
                default_rel: 0.0,
                display_unit: "",
                accessor: |_| None,
            }
        }
        assert!(!make(Polarity::HigherBetter).higher_is_worse());
        assert!(make(Polarity::LowerBetter).higher_is_worse());
        assert!(make(Polarity::TargetValue(42.0)).higher_is_worse());
        assert!(make(Polarity::Unknown).higher_is_worse());
    }

    #[test]
    fn polarity_target_accepts_finite() {
        let p = Polarity::target(0.5);
        assert_eq!(p, Polarity::TargetValue(0.5));
    }

    /// `Polarity::target(NaN)` must panic in release too — non-finite
    /// target values produce silent incorrect regression verdicts in
    /// `compare_rows`, so the gate is a runtime `assert!` (not
    /// `debug_assert!`). Pins that a release build won't silently
    /// let NaN slip through.
    #[test]
    #[should_panic(expected = "Polarity::TargetValue target must be finite")]
    fn polarity_target_rejects_nan_panics() {
        let _ = Polarity::target(f64::NAN);
    }

    /// `Polarity::target(+inf)` panics symmetrically with NaN.
    /// `compare_rows` would otherwise produce inf-vs-finite verdicts
    /// that depend on IEEE-754 infinity arithmetic rather than
    /// meaningful regression direction.
    #[test]
    #[should_panic(expected = "Polarity::TargetValue target must be finite")]
    fn polarity_target_rejects_positive_infinity_panics() {
        let _ = Polarity::target(f64::INFINITY);
    }

    /// `Polarity::target(-inf)` ditto.
    #[test]
    #[should_panic(expected = "Polarity::TargetValue target must be finite")]
    fn polarity_target_rejects_negative_infinity_panics() {
        let _ = Polarity::target(f64::NEG_INFINITY);
    }

    /// `Polarity::TargetValue(NaN)` — which bypasses the
    /// `Polarity::target` constructor's runtime assert when a hand-
    /// built struct literal is used — serializes to
    /// `{"TargetValue":null}` via serde_json because
    /// `serde_json::Number::from_f64` returns `None` on non-finite
    /// values and the default serializer falls back to `null`.
    /// The resulting document does NOT round-trip: deserialization
    /// fails because `null` can't satisfy the inner `f64` slot.
    /// So NaN cannot survive a sidecar write + read pair, even
    /// though the write step silently coerces it. Pins both halves
    /// of this asymmetric guard so a future serde-attribute change
    /// (e.g. `serialize_with = "serialize_nan_as_zero"`) or a
    /// custom deserializer gets surfaced here.
    #[test]
    fn polarity_target_nan_serializes_as_null_and_fails_to_round_trip() {
        let p = Polarity::TargetValue(f64::NAN);
        let s = serde_json::to_string(&p).expect("NaN→null serialization is the current behavior");
        assert_eq!(s, "{\"TargetValue\":null}");
        assert!(
            serde_json::from_str::<Polarity>(&s).is_err(),
            "the null-coerced round-trip must fail to deserialize so a NaN written \
             by an un-guarded producer cannot silently re-enter a run",
        );
    }

    /// Raw `NaN` / `Infinity` tokens are not valid JSON, so a
    /// sidecar file hand-edited (or emitted by a non-serde writer)
    /// to contain them will be rejected at parse time. Pairs with
    /// the null-round-trip test above.
    #[test]
    fn polarity_target_nan_cannot_deserialize_from_non_json_literals() {
        assert!(serde_json::from_str::<Polarity>("{\"TargetValue\":NaN}").is_err());
        assert!(serde_json::from_str::<Polarity>("{\"TargetValue\":Infinity}").is_err());
        assert!(serde_json::from_str::<Polarity>("{\"TargetValue\":-Infinity}").is_err());
    }

    /// `MetricCheck::range(metric, lo, hi)` panics when `lo > hi`.
    /// A reversed-bounds range describes an empty interval that no
    /// finite metric can satisfy — almost always a user error.
    /// Failing loudly at the constructor surfaces the typo at the
    /// call site rather than letting the evaluator run an
    /// unsatisfiable check against every probe value.
    #[test]
    #[should_panic(expected = "lo must be <= hi")]
    fn check_range_reversed_bounds_panics_at_construction() {
        let _ = MetricCheck::range("iops", 100.0, 50.0);
    }

    /// `MetricCheck::range_builtin` carries the same reversed-bounds construction
    /// panic as [`MetricCheck::range`] (the typed sibling shares the `lo <= hi`
    /// assert), so the typed payload-check path fails just as loudly on a
    /// reversed range as the string form.
    #[test]
    #[should_panic(expected = "lo must be <= hi")]
    fn check_range_builtin_reversed_bounds_panics_at_construction() {
        let _ = MetricCheck::range_builtin(
            crate::stats::BuiltinMetric::WakeupP99LatencyUs,
            100.0,
            50.0,
        );
    }

    /// Equal bounds (`lo == hi`) describe a single-point interval —
    /// allowed; the metric must equal that exact value. Pins the
    /// `<=` (not `<`) gate in the constructor.
    #[test]
    fn check_range_equal_bounds_construct() {
        let r = MetricCheck::range("iops", 50.0, 50.0);
        match r {
            MetricCheck::Range { metric, lo, hi } => {
                assert_eq!(metric, "iops");
                assert_eq!(lo, 50.0);
                assert_eq!(hi, 50.0);
            }
            _ => panic!("expected Range variant"),
        }
    }

    /// NaN as either bound trips the `lo <= hi` gate (NaN comparisons
    /// always return false), so the constructor panics. Prevents an
    /// always-fails check from slipping into the evaluator pipeline.
    #[test]
    #[should_panic(expected = "lo must be <= hi")]
    fn check_range_nan_lo_panics() {
        let _ = MetricCheck::range("iops", f64::NAN, 50.0);
    }

    #[test]
    #[should_panic(expected = "lo must be <= hi")]
    fn check_range_nan_hi_panics() {
        let _ = MetricCheck::range("iops", 50.0, f64::NAN);
    }

    // Debug + helper method surface.
    #[test]
    fn payload_debug_renders_identity_fields() {
        let s = format!("{:?}", Payload::KERNEL_DEFAULT);
        assert!(s.contains("Payload"), "debug output: {s}");
        assert!(s.contains("eevdf"), "debug output: {s}");
        assert!(
            s.contains("kind: Scheduler(\"eevdf\")"),
            "debug output: {s}"
        );
    }

    #[test]
    fn payload_kind_debug_renders_variant_and_identity() {
        let binary = PayloadKind::Binary("fio");
        let s = format!("{binary:?}");
        assert!(s.contains("Binary"), "debug output: {s}");
        assert!(s.contains("fio"), "debug output: {s}");

        let sched = Payload::KERNEL_DEFAULT.kind;
        let s = format!("{sched:?}");
        assert!(s.contains("Scheduler"), "debug output: {s}");
        assert!(s.contains("eevdf"), "debug output: {s}");
    }

    #[test]
    fn output_format_derive_debug_clone_copy() {
        // Debug must name the variant (mirrors
        // `payload_kind_debug_renders_variant_and_identity`).
        let s = format!("{:?}", OutputFormat::Json);
        assert!(s.contains("Json"), "Debug must name the variant: {s}");

        // Copy semantics observably: after `let b = a;` the original
        // `a` is still usable (a move would make this fail to compile),
        // and both render identically.
        let a = OutputFormat::Json;
        let b = a; // Copy, not move.
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "copied value must Debug-render identically to the original",
        );
    }

    #[test]
    fn as_scheduler_extracts_ref_for_scheduler_kind() {
        let s = Payload::KERNEL_DEFAULT
            .as_scheduler()
            .expect("Scheduler kind");
        assert_eq!(s.name, "eevdf");
    }

    #[test]
    fn payload_clone_preserves_identity() {
        let a = Payload::KERNEL_DEFAULT;
        assert_eq!(a.name, Payload::KERNEL_DEFAULT.name);
        assert_eq!(a.is_scheduler(), Payload::KERNEL_DEFAULT.is_scheduler());
        assert_eq!(a.as_scheduler().map(|s| s.name), Some("eevdf"));
    }

    /// `PayloadMetrics` postcard wire-format roundtrip. The type ships
    /// guest→host inside `MSG_TYPE_PAYLOAD_METRICS` and the
    /// postcard codec rejects any nested type carrying
    /// `#[serde(untagged)]` or `#[serde(tag, content)]` at encode
    /// time (`WontImplement`) — a regression introduced by a
    /// contributor adding adjacent tagging to `Metric` or any
    /// downstream field would surface as
    /// `ERR_NO_TEST_FUNCTION_OUTPUT` on the host without this pin.
    /// Sister inventory: see comment block at src/vmm/wire.rs.
    #[test]
    fn payload_metrics_postcard_roundtrip() {
        let original = PayloadMetrics {
            payload_index: 7,
            metrics: vec![
                Metric {
                    name: "tput".to_string(),
                    value: 12345.6,
                    polarity: Polarity::HigherBetter,
                    unit: "ops/s".to_string(),
                    stream: MetricStream::Stdout,
                },
                Metric {
                    name: "lat_p99".to_string(),
                    value: 0.025,
                    polarity: Polarity::LowerBetter,
                    unit: "s".to_string(),
                    stream: MetricStream::Stderr,
                },
            ],
            exit_code: 0,
        };
        let bytes = postcard::to_allocvec(&original).expect("encode");
        let back: PayloadMetrics = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back.payload_index, original.payload_index);
        assert_eq!(back.exit_code, original.exit_code);
        assert_eq!(back.metrics.len(), original.metrics.len());
        for (a, b) in original.metrics.iter().zip(back.metrics.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.value, b.value);
        }
    }
}
