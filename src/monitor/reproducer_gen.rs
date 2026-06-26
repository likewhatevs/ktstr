//! Translate [`super::debug_capture::WorkloadFingerprint`] hints into
//! ktstr test specs (`WorkloadConfig` values) and generated source
//! code.
//!
//! Pipeline position: this module sits at the END of the live-host
//! pipeline.
//!
//! ```text
//! BpfSyscallAccessor    ──┐
//! LiveHostKernelEnv      ─┤   ┌───────────────────┐    ┌──────────────────┐
//! KallsymsTable          ─┼─→ │ DebugCapture      │ →  │ ReproducerSpec   │
//! dmesg_scx parser       ─┤   │ + Fingerprint     │    │ (this module)    │
//! ctprof::CtprofSnapshot ──┘   └───────────────────┘    └──────────────────┘
//!                                (capture pipeline)        (this module)
//! ```
//!
//! # The translation contract
//!
//! The generator's job is to take fingerprint HINTS (best-effort
//! projections from capture data) and emit a test spec the framework
//! will execute. The generator is NOT a classifier — it does not
//! decide "this workload is locking-bound" or "this is a cache
//! pressure pathology"; it just maps observed shapes to primitive
//! types. Pathology classification is a separate, downstream
//! concern (the reader / LLM consuming `ktstr show / compare`).
//!
//! # Mapping table
//!
//! | fingerprint hint              | ktstr type                              |
//! |-------------------------------|-----------------------------------------|
//! | WorkloadGroupHint.thread_count| WorkloadConfig::num_workers             |
//! | AffinityHint::SingleCpu{cpus} | AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::SingleCpu when empty |
//! | AffinityHint::Exact{cpus}     | AffinityIntent::Exact(set) [‡]           |
//! | AffinityHint::Inherit         | AffinityIntent::Inherit                  |
//! | AffinityHint::LlcAligned{cpus}| AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::LlcAligned when empty |
//! | AffinityHint::CrossCgroup{cpus}| AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::CrossCgroup when empty |
//! | AffinityHint::SmtSiblingPair{cpus}| AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::SmtSiblingPair when empty |
//! | AffinityHint::RandomSubset { from, count: popcount-per-thread } | AffinityIntent::RandomSubset { from, count } when both non-empty; placeholder otherwise [†] |
//! | WorkTypeHint::SpinWait         | WorkType::SpinWait                       |
//! | WorkTypeHint::YieldHeavy      | WorkType::YieldHeavy                    |
//! | WorkTypeHint::Mixed           | WorkType::Mixed                         |
//! | WorkTypeHint::Bursty{b,s}     | WorkType::Bursty { burst_duration: b, sleep_duration: s } |
//! | WorkTypeHint::PipeIo          | WorkType::PipeIo { burst_iters: 1024 }  |
//! | WorkTypeHint::FutexPingPong   | WorkType::FutexPingPong { spin_iters: 1024 } |
//! | WorkTypeHint::CachePressure   | WorkType::CachePressure { size_kib, stride } |
//! | WorkTypeHint::IoSyncWrite     | WorkType::IoSyncWrite                   |
//! | WorkTypeHint::IoRandRead      | WorkType::IoRandRead                    |
//! | WorkTypeHint::IoConvoy        | WorkType::IoConvoy                      |
//! | SchedPolicyHint::Other{nice}  | SchedPolicy::Normal + nice              |
//! | SchedPolicyHint::Fifo{prio}   | SchedPolicy::Fifo(prio)                 |
//! | SchedPolicyHint::RoundRobin   | SchedPolicy::RoundRobin(prio)           |
//! | SchedPolicyHint::Deadline     | SchedPolicy::Deadline(...)              |
//! | SchedPolicyHint::Batch        | SchedPolicy::Batch                      |
//! | SchedPolicyHint::Idle         | SchedPolicy::Idle                       |
//! | SchedPolicyHint::Ext          | (no explicit policy — scx default)      |
//!
//! [†] The `RandomSubset` row emits an empty-pool / zero-count
//! placeholder that the spawn-time affinity gate REJECTS when the
//! producer did not record a resolved pool. The resulting spec is
//! not runnable as-is — hand-edit `from` to the actual CPU pool and
//! `count` to the desired sample size before running, or change to
//! `AffinityIntent::Inherit`. When the producer DID record a pool,
//! the generator emits a fully-populated `AffinityIntent::RandomSubset`
//! that the spawn-time gate accepts directly.
//!
//! [‡] Empty `cpus` emits a hand-edit-required note alongside the
//! placeholder `AffinityIntent::Exact(empty)` — the spawn-time
//! affinity gate rejects an empty Exact set, so the rendered spec
//! is NOT runnable as-is until the user pastes in the observed CPUs
//! (or switches to `AffinityIntent::Inherit`). Non-empty `cpus`
//! emits a resolved-collapse note and the runnable
//! `AffinityIntent::Exact` directly.
//!
//! `IoRandRead` and `IoConvoy` hints are accepted by the projection
//! layer but not yet emitted by the capture pipeline; all real-disk-
//! IO captures currently project to `IoSyncWrite`. The mapping is
//! ready for the day the pipeline learns to discriminate IO modes
//! from the captured open-flag + IO-shape signals documented on
//! [`super::debug_capture::WorkTypeHint`].
//!
//! Hints that don't fire produce framework defaults. Hints that
//! fire ambiguously (multiple variants in one fingerprint) pick
//! the first observed-frequency-ranked entry; the rest are emitted
//! as `notes` on [`ReproducerSpec`] so the human / generator
//! consumer can choose to override.
//!
//! # Output formats
//!
//! Two surfaces:
//!
//! - [`ReproducerSpec`] — a programmatic value that the framework
//!   can execute directly. Used by tests and tooling that
//!   construct workloads in-process.
//! - [`render_run_file_source`] / [`render_ktstr_test_source`] —
//!   generated Rust source string that recreates the spec via
//!   library APIs. Used by the `cargo ktstr export` /
//!   `cargo ktstr capture-reproduce` flow that emits a self-
//!   contained test file.

use std::collections::BTreeSet;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::debug_capture::{
    AffinityHint, CgroupHint, DebugCapture, SchedPolicyHint, WorkTypeHint, WorkloadFingerprint,
    WorkloadGroupHint,
};
use crate::workload::{AffinityIntent, SchedPolicy, WorkType, WorkloadConfig};

/// One reproducer spec — a `WorkloadConfig` value plus diagnostic
/// notes about confidence / ambiguity.
///
/// The framework can execute the `config` directly. The `notes` and
/// `cgroup_hints` fields surface low-confidence projections so the
/// caller (or generated source) can flag them for the user.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(dead_code)] // wired by the (separate) cargo ktstr capture-
// reproduce subcommand; library lands the type.
pub struct ReproducerSpec {
    /// The mappable WorkloadConfig — what the framework executes.
    /// Skipped from serde because WorkloadConfig isn't (and doesn't
    /// need to be) Serialize. The serialized form is a pair of
    /// the source-rendered spec text + the `notes`.
    #[serde(skip)]
    pub config: WorkloadConfig,
    /// Cgroup-shaped hints that don't fit on `WorkloadConfig`
    /// directly — the framework's [`crate::workload`] surface
    /// doesn't yet have a `cgroup` field on `WorkloadConfig`, so
    /// the generator emits cgroup hints alongside for the test
    /// harness or `.run` shar to apply at setup time. Maps to
    /// `cgroup_def!` macro in generated source.
    pub cgroup_hints: Vec<CgroupHint>,
    /// Notes about projection quality — ambiguous mappings,
    /// unmapped variants, low-confidence hints (sample size = 1,
    /// fingerprint gaps cited in input).
    ///
    /// Each entry is a typed [`ReproducerNote`] whose kind
    /// (`Informational` / `Resolved` / `UnresolvedAffinity` /
    /// `UnmappedWorkType`) drives [`Self::is_runnable`] and
    /// [`Self::unresolved_count`] without substring matching.
    pub notes: Vec<ReproducerNote>,
    /// The scheduler name the capture was running, when known.
    /// Lets the generated test pick the right scheduler binary
    /// to attach.
    pub scheduler_name: String,
}

/// Diagnostic note attached to a [`ReproducerSpec`].
///
/// The variant classifies the note's effect on runnability:
/// `UnresolvedAffinity` and `UnmappedWorkType` are the only kinds
/// that block [`ReproducerSpec::is_runnable`]; `Informational` and
/// `Resolved` are zero-cost surface for the generated source's
/// comment block. Replaced an earlier substring-marker scheme that
/// probed the rendered text — see `Self::is_unresolved` for the
/// runnability filter.
///
/// Each variant carries the rendered text the generator emits — the
/// text format is unchanged from the prior `Vec<String>` shape, so
/// tests that match on note wording continue to work via
/// [`Self::message`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReproducerNote {
    /// Default-projection or fingerprint-gap context. Does NOT
    /// affect runnability. Examples: "no workload groups in
    /// fingerprint — defaulting num_workers=1", "additional
    /// affinity hints not modeled: ...", "fingerprint gap: ...",
    /// "SchedPolicyHint::Ext observed; framework defaults to scx
    /// routing — no policy override emitted".
    Informational(String),
    /// A topology-aware [`AffinityHint`] carried resolved CPUs and
    /// the generator collapsed it to [`AffinityIntent::Exact`], OR
    /// a populated `RandomSubset` whose pool the spawn-time gate
    /// accepts directly. Does NOT block runnability — the rendered
    /// spec runs without scenario-engine resolution.
    Resolved(String),
    /// An affinity hint whose projection produces a placeholder the
    /// spawn-time affinity gate REJECTS. Spec is NOT runnable until
    /// the user hand-edits the rendered source (or switches to
    /// `AffinityIntent::Inherit`). Counted by
    /// [`ReproducerSpec::unresolved_count`].
    UnresolvedAffinity(String),
    /// The projected [`WorkType`] is one `render_work_type`
    /// dispatches to `render_work_type_todo` (renders as
    /// `WorkType::SpinWait /* TODO: ... */`). Spec is NOT runnable
    /// until the user replaces the placeholder with a real builder
    /// call. Counted by [`ReproducerSpec::unresolved_count`].
    UnmappedWorkType(String),
}

impl ReproducerNote {
    /// The rendered note text — the same string the prior
    /// `Vec<String>` field carried directly. Drives the comment
    /// block in [`render_run_file_source`] and lets test assertions
    /// match wording through `note.message().contains("...")`.
    #[allow(dead_code)]
    pub fn message(&self) -> &str {
        match self {
            ReproducerNote::Informational(s)
            | ReproducerNote::Resolved(s)
            | ReproducerNote::UnresolvedAffinity(s)
            | ReproducerNote::UnmappedWorkType(s) => s,
        }
    }

    /// `true` for the kinds [`ReproducerSpec::unresolved_count`]
    /// counts — `UnresolvedAffinity` and `UnmappedWorkType`. Both
    /// signal "the rendered spec is NOT runnable until the user
    /// hand-edits".
    fn is_unresolved(&self) -> bool {
        matches!(
            self,
            ReproducerNote::UnresolvedAffinity(_) | ReproducerNote::UnmappedWorkType(_)
        )
    }
}

impl std::fmt::Display for ReproducerNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl ReproducerSpec {
    /// Returns `true` when the spec is runnable as-is (the spawn-time
    /// affinity gate accepts it without hand-editing AND the
    /// rendered source carries no `WorkType::SpinWait /* TODO: ... */`
    /// placeholder). `false` means at least one hand-edit-required
    /// note is present, or [`Self::config`]'s `work_type` is one the
    /// generator does not know how to render as a runnable builder
    /// call yet.
    ///
    /// The check is the union of two signals:
    ///
    /// 1. Hand-edit-required notes — counted via
    ///    [`Self::unresolved_count`], which sums every
    ///    [`ReproducerNote::UnresolvedAffinity`] and
    ///    [`ReproducerNote::UnmappedWorkType`] entry by enum kind
    ///    (no substring matching).
    /// 2. Direct check on [`Self::config`]'s `work_type` —
    ///    `is_unmapped_work_type` returns `true` for variants that
    ///    `render_work_type` dispatches to `render_work_type_todo`.
    ///    This catches specs constructed without going through
    ///    [`generate_spec`] / `map_work_type` (e.g. callers that set
    ///    `config.work_type` directly), so a manually-built spec with
    ///    `WorkType::CacheYield { .. }` is correctly classified as
    ///    NOT runnable even though no unresolved note was pushed.
    ///
    /// `Informational` and `Resolved` notes do not affect the
    /// outcome — they are surfaced in the rendered comment block but
    /// carry no runnability-blocking semantic.
    #[allow(dead_code)]
    pub fn is_runnable(&self) -> bool {
        self.unresolved_count() == 0 && !is_unmapped_work_type(&self.config.work_type)
    }

    /// Number of hand-edit-required notes in [`Self::notes`]. Counts
    /// every [`ReproducerNote::UnresolvedAffinity`] (affinity
    /// hand-edit prompts) and [`ReproducerNote::UnmappedWorkType`]
    /// (work-type TODO prompts); useful for surfacing "this
    /// reproducer needs N edits" messaging in `cargo ktstr` tooling.
    ///
    /// Does NOT include the `is_unmapped_work_type` direct-config
    /// signal that [`Self::is_runnable`] folds in — that path
    /// catches manually-constructed specs without notes and is
    /// outside the "count of edit prompts" semantic.
    #[allow(dead_code)]
    pub fn unresolved_count(&self) -> usize {
        self.notes.iter().filter(|n| n.is_unresolved()).count()
    }
}

/// Produce a [`ReproducerSpec`] from a [`DebugCapture`].
///
/// Pure function: same capture always produces the same spec. The
/// projection is deterministic and dependency-free — only the
/// fingerprint hints matter. Fingerprint gaps propagate into
/// `spec.notes` so the caller can see why a particular field
/// fell back to default.
///
/// Picks the FIRST hint of each kind when multiple are present
/// (fingerprint atoms are documented to be sorted by
/// frequency-descending). Subsequent hints are recorded in `notes`
/// as alternative observations the human / LLM can choose to
/// override.
#[allow(dead_code)]
pub fn generate_spec(capture: &DebugCapture) -> ReproducerSpec {
    let mut spec = ReproducerSpec {
        scheduler_name: failure_scheduler_name(capture),
        ..Default::default()
    };

    map_workload_groups(&capture.fingerprint, &mut spec);
    map_affinity(&capture.fingerprint, &mut spec);
    map_work_type(&capture.fingerprint, &mut spec);
    map_sched_policy(&capture.fingerprint, &mut spec);
    spec.cgroup_hints = capture.fingerprint.cgroup_hints.clone();

    // Carry fingerprint gaps forward — the user sees them so they
    // know where to refine the capture or hand-edit the spec.
    for gap in &capture.fingerprint.gaps {
        spec.notes.push(ReproducerNote::Informational(format!(
            "fingerprint gap: {gap}"
        )));
    }

    spec
}

fn failure_scheduler_name(capture: &DebugCapture) -> String {
    // Always returns the empty string. The current
    // [`FailureDumpReport`] shape does not carry a scheduler-name
    // field, and [`DebugCapture`] itself does not duplicate it. The
    // returned empty string signals to [`render_run_file_source`]
    // that the rendered source should not emit a `// Scheduler:`
    // comment line — the consumer fills in the scheduler name when
    // pasting the generated reproducer into a test file.
    let _ = capture;
    String::new()
}

fn map_workload_groups(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.workload_groups.first() else {
        spec.notes.push(ReproducerNote::Informational(
            "no workload groups in fingerprint — defaulting num_workers=1".into(),
        ));
        return;
    };
    spec.config.num_workers = primary.thread_count.max(1) as usize;
    push_extras_note(
        &mut spec.notes,
        "additional workload groups not modeled in primary spec",
        fp.workload_groups
            .iter()
            .skip(1)
            .map(|g: &WorkloadGroupHint| format!("{} ({} threads)", g.cgroup_path, g.thread_count)),
    );
}

/// Emit an "additional X observed: ..." note when the fingerprint
/// carries more than one hint of a kind. Centralises the pattern
/// shared by [`map_workload_groups`], [`map_affinity`],
/// [`map_work_type`], and [`map_sched_policy`]: skip-first-then-render
/// the secondary entries, comma-join them, and only push when the
/// iterator yields anything.
///
/// `header` is the lead phrase before the colon (e.g.
/// `"additional affinity hints not modeled"`). `entries` is an
/// iterator of pre-rendered `String` descriptions for each
/// secondary entry. The pushed note is classified
/// [`ReproducerNote::Informational`] — secondary-hint enumeration
/// is descriptive context, never a runnability blocker.
fn push_extras_note(
    notes: &mut Vec<ReproducerNote>,
    header: &str,
    entries: impl Iterator<Item = String>,
) {
    let alts: Vec<String> = entries.collect();
    if !alts.is_empty() {
        notes.push(ReproducerNote::Informational(format!(
            "{header}: {}",
            alts.join(", ")
        )));
    }
}

/// Build the user-facing note attached when a topology-aware
/// [`AffinityHint`] is projected without a resolved CPU set. The 4
/// topology-aware variants (`SingleCpu`, `LlcAligned`, `CrossCgroup`,
/// `SmtSiblingPair`) share the same structure: name the variant,
/// describe what the scenario engine resolves at apply time, then
/// point the user at the concrete `AffinityIntent::Exact(...)` they
/// should hand-edit to if they want to spawn directly via
/// [`crate::workload::WorkloadHandle::spawn`] (which rejects the
/// topology-aware variants — they require scenario context the
/// spawn-time gate doesn't have).
///
/// `hand_edit_target` carries paste-ready Rust (an
/// `AffinityIntent::exact(...)` call with angle-bracket placeholders)
/// so the generated note can be copied into a test file with minimal
/// editing.
fn topology_aware_note(variant: &str, engine_action: &str, hand_edit_target: &str) -> String {
    // The pushed note is classified
    // [`ReproducerNote::UnresolvedAffinity`] at the call site
    // ([`map_topology_aware_affinity`]); the typed variant — not
    // the wording — is what
    // [`ReproducerSpec::unresolved_count`] uses to count
    // hand-edit-required notes. The "spawn-time affinity gate
    // rejects" wording survives so existing reproducer-output
    // consumers see the same human-facing diagnostic.
    format!(
        "AffinityHint::{variant} observed without resolved CPUs; \
         emitting AffinityIntent::{variant} — the scenario engine \
         {engine_action} at apply time. The spawn-time affinity gate \
         rejects this variant (no topology context); use the \
         scenario engine or hand-edit to {hand_edit_target}"
    )
}

/// Build the note attached when a topology-aware [`AffinityHint`]
/// carried resolved CPUs and the generator collapsed it to
/// [`AffinityIntent::Exact`]. The note preserves the original
/// pattern classification (SingleCpu / LlcAligned / CrossCgroup /
/// SmtSiblingPair) so the consumer can see what the producer
/// observed before resolution, even though the emitted spec is a
/// flat `Exact`.
fn topology_resolved_note(variant: &str, cpus: &[u32]) -> String {
    format!(
        "AffinityHint::{variant} observed with resolved CPUs {cpus:?}; \
         emitting AffinityIntent::Exact directly so the spec runs \
         without scenario-engine resolution",
    )
}

/// Build a `BTreeSet<usize>` from a slice of `u32` CPU IDs. Centralises
/// the `u32 → usize` widening the `AffinityHint` payload requires
/// before it can populate an [`AffinityIntent::Exact`] / `RandomSubset`
/// pool.
fn cpus_to_set(cpus: &[u32]) -> BTreeSet<usize> {
    cpus.iter().map(|&c| c as usize).collect()
}

/// Resolve a topology-aware [`AffinityHint`] payload to an
/// [`AffinityIntent`] and append the matching note to `spec.notes`.
///
/// The 4 topology-aware variants (`SingleCpu`, `LlcAligned`,
/// `CrossCgroup`, `SmtSiblingPair`) share the same shape: an empty
/// `cpus` payload falls back to the matching topology-aware
/// `AffinityIntent` variant (resolved by the scenario engine at apply
/// time) plus a hand-edit note, while a non-empty payload collapses
/// to [`AffinityIntent::Exact`] containing the observed CPUs.
///
/// `variant` names the [`AffinityHint`] variant for the note text,
/// `topology_intent` is the matching topology-aware intent emitted on
/// the empty path, `engine_action` and `hand_edit_target` describe
/// the scenario engine's resolution and the user's hand-edit target
/// respectively (the strings appear in [`topology_aware_note`]).
fn map_topology_aware_affinity(
    cpus: &[u32],
    variant: &str,
    topology_intent: AffinityIntent,
    engine_action: &str,
    hand_edit_target: &str,
    spec: &mut ReproducerSpec,
) -> AffinityIntent {
    if cpus.is_empty() {
        spec.notes
            .push(ReproducerNote::UnresolvedAffinity(topology_aware_note(
                variant,
                engine_action,
                hand_edit_target,
            )));
        topology_intent
    } else {
        spec.notes
            .push(ReproducerNote::Resolved(topology_resolved_note(
                variant, cpus,
            )));
        AffinityIntent::Exact(cpus_to_set(cpus))
    }
}

fn map_affinity(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.affinity_hints.first() else {
        return;
    };
    spec.config.affinity = match primary {
        AffinityHint::Inherit => AffinityIntent::Inherit,
        AffinityHint::SingleCpu { cpus } => map_topology_aware_affinity(
            cpus,
            "SingleCpu",
            AffinityIntent::SingleCpu,
            "picks the concrete CPU from the cgroup's cpuset",
            "AffinityIntent::exact([<cpu>])",
            spec,
        ),
        AffinityHint::LlcAligned { cpus } => map_topology_aware_affinity(
            cpus,
            "LlcAligned",
            AffinityIntent::LlcAligned,
            "resolves the LLC mask from the cgroup's cpuset",
            "AffinityIntent::exact([<llc_cpu_0>, <llc_cpu_1>, ...])",
            spec,
        ),
        AffinityHint::CrossCgroup { cpus } => map_topology_aware_affinity(
            cpus,
            "CrossCgroup",
            AffinityIntent::CrossCgroup,
            "expands to the full topology",
            "AffinityIntent::exact([<cpu_0>, <cpu_1>, ...])",
            spec,
        ),
        AffinityHint::SmtSiblingPair { cpus } => map_topology_aware_affinity(
            cpus,
            "SmtSiblingPair",
            AffinityIntent::SmtSiblingPair,
            "picks an SMT-sibling pair from the cgroup's effective cpuset, \
             or the full topology when no cpuset is active",
            "AffinityIntent::exact([<sibling_a>, <sibling_b>])",
            spec,
        ),
        AffinityHint::Exact { cpus } => {
            // Empty `cpus` produces an empty Exact set that the
            // spawn-time affinity gate rejects. Emit a note so the
            // reproducer surface is consistent — every other arm
            // pushes a note when it lands a placeholder, and the
            // resolved arms push a `topology_resolved_note`. An
            // empty Exact is the malformed shape; a populated Exact
            // is the runnable shape.
            if cpus.is_empty() {
                spec.notes.push(ReproducerNote::UnresolvedAffinity(
                    "AffinityHint::Exact observed with no CPUs; emitting \
                     AffinityIntent::Exact(empty) — the spawn-time \
                     affinity gate rejects an empty Exact set, so this \
                     spec is NOT runnable as-is. Hand-edit to \
                     AffinityIntent::exact([<cpu_0>, <cpu_1>, ...]) \
                     with the observed CPUs, or change to \
                     AffinityIntent::Inherit."
                        .into(),
                ));
            } else {
                spec.notes
                    .push(ReproducerNote::Resolved(topology_resolved_note(
                        "Exact", cpus,
                    )));
            }
            AffinityIntent::Exact(cpus_to_set(cpus))
        }
        AffinityHint::RandomSubset { from, count } => {
            if from.is_empty() || *count == 0 {
                spec.notes.push(ReproducerNote::UnresolvedAffinity(
                    "AffinityHint::RandomSubset observed without a \
                     resolved pool / count; emitting \
                     AffinityIntent::RandomSubset { from: empty, count: 0 } \
                     as a placeholder — the spawn-time affinity gate \
                     rejects empty-pool / zero-count RandomSubset, so \
                     this spec is NOT runnable as-is. Hand-edit `from` \
                     to the actual CPU pool and `count` to the desired \
                     sample size before running, or change to \
                     AffinityIntent::Inherit."
                        .into(),
                ));
                AffinityIntent::RandomSubset {
                    from: BTreeSet::new(),
                    count: 0,
                }
            } else {
                spec.notes.push(ReproducerNote::Resolved(format!(
                    "AffinityHint::RandomSubset observed with resolved \
                     pool {from:?} count={count}; emitting \
                     AffinityIntent::RandomSubset directly so the \
                     spawn-time affinity gate accepts it without \
                     hand-editing",
                )));
                AffinityIntent::RandomSubset {
                    from: cpus_to_set(from),
                    count: *count as usize,
                }
            }
        }
    };

    push_extras_note(
        &mut spec.notes,
        "additional affinity hints not modeled",
        fp.affinity_hints.iter().skip(1).map(|a| format!("{a:?}")),
    );
}

fn map_work_type(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.work_type_hints.first() else {
        spec.notes.push(ReproducerNote::Informational(
            "no work-type hint in fingerprint — defaulting to \
             WorkType::SpinWait"
                .into(),
        ));
        return;
    };
    let work_type = match primary {
        WorkTypeHint::SpinWait => WorkType::SpinWait,
        WorkTypeHint::YieldHeavy => WorkType::YieldHeavy,
        WorkTypeHint::Mixed => WorkType::Mixed,
        WorkTypeHint::Bursty {
            burst_duration,
            sleep_duration,
        } => WorkType::Bursty {
            burst_duration: *burst_duration,
            sleep_duration: *sleep_duration,
        },
        WorkTypeHint::PipeIo => WorkType::PipeIo { burst_iters: 1024 },
        WorkTypeHint::FutexPingPong => WorkType::FutexPingPong { spin_iters: 1024 },
        WorkTypeHint::CachePressure { size_kib, stride } => WorkType::CachePressure {
            size_kib: *size_kib as usize,
            stride: *stride as usize,
        },
        WorkTypeHint::IoSyncWrite => WorkType::IoSyncWrite,
        WorkTypeHint::IoRandRead => WorkType::IoRandRead,
        WorkTypeHint::IoConvoy => WorkType::IoConvoy,
    };
    record_work_type(work_type, spec);

    push_extras_note(
        &mut spec.notes,
        "additional work-type hints observed",
        fp.work_type_hints.iter().skip(1).map(|w| format!("{w:?}")),
    );
}

/// Assign `work_type` to `spec.config.work_type` and push a
/// [`ReproducerNote::UnmappedWorkType`] note when the assigned variant
/// is one [`render_work_type`] dispatches to [`render_work_type_todo`].
///
/// Extracted from [`map_work_type`] so the unmapped-projection branch
/// has a production code path that tests can drive directly. Calling
/// this with `WorkType::ForkExit` (or any other variant
/// [`is_unmapped_work_type`] returns `true` for) exercises the
/// safety-net branch end-to-end — the test is no longer a manual
/// re-construction of the same logic.
///
/// No current [`WorkTypeHint`] variant projects to an unmapped
/// [`WorkType`], so the unmapped branch is reachable today only via
/// this helper. When a future hint variant lands on a TODO arm the
/// existing call site in [`map_work_type`] starts firing the branch
/// automatically — no duplicated logic to keep in sync.
fn record_work_type(work_type: WorkType, spec: &mut ReproducerSpec) {
    spec.config.work_type = work_type;
    if is_unmapped_work_type(&spec.config.work_type) {
        spec.notes.push(ReproducerNote::UnmappedWorkType(format!(
            "no fingerprint mapping for WorkType::{:?} — \
             render_run_file_source emits a TODO-decorated \
             SpinWait placeholder; hand-edit the rendered source to \
             a real builder call before running",
            spec.config.work_type,
        )));
    }
}

/// Return `true` when `w` is a [`WorkType`] variant that
/// [`render_work_type`] dispatches to [`render_work_type_todo`] (i.e.
/// renders as `WorkType::SpinWait /* TODO: ... */` because no
/// fingerprint mapping exists yet). Mirrors the runnable / TODO split
/// in [`render_work_type`] one-for-one — a new variant added there
/// must be classified here in the same pass.
///
/// Used by [`ReproducerSpec::is_runnable`] and [`map_work_type`] to
/// flag specs whose `work_type` cannot be rendered as a real builder
/// call. The runnable arm returns `false`; every TODO arm returns
/// `true`.
fn is_unmapped_work_type(w: &WorkType) -> bool {
    match w {
        // Variants the projection layer maps from a fingerprint hint
        // — render as runnable builder calls in [`render_work_type`].
        WorkType::SpinWait
        | WorkType::YieldHeavy
        | WorkType::Mixed
        | WorkType::IoSyncWrite
        | WorkType::IoRandRead
        | WorkType::IoConvoy
        | WorkType::Bursty { .. }
        | WorkType::PipeIo { .. }
        | WorkType::FutexPingPong { .. }
        | WorkType::CachePressure { .. } => false,
        // Variants no fingerprint hint currently projects to —
        // [`render_work_type`] dispatches each of these to
        // [`render_work_type_todo`]. Adding one to the runnable arm
        // above MUST also flip this match in lock-step.
        WorkType::CacheYield { .. }
        | WorkType::CachePipe { .. }
        | WorkType::FutexFanOut { .. }
        | WorkType::Sequence { .. }
        | WorkType::ForkExit
        | WorkType::NiceSweep
        | WorkType::AffinityChurn { .. }
        | WorkType::CrossAffinityChurn { .. }
        | WorkType::PolicyChurn { .. }
        | WorkType::FanOutCompute { .. }
        | WorkType::Schbench { .. }
        | WorkType::PageFaultChurn { .. }
        | WorkType::MutexContention { .. }
        | WorkType::Custom { .. }
        | WorkType::ThunderingHerd { .. }
        | WorkType::PriorityInversion { .. }
        | WorkType::ProducerConsumerImbalance { .. }
        | WorkType::RtStarvation { .. }
        | WorkType::AsymmetricWaker { .. }
        | WorkType::WakeChain { .. }
        | WorkType::NumaWorkingSetSweep { .. }
        | WorkType::CgroupChurn { .. }
        | WorkType::CgroupAttachStorm { .. }
        | WorkType::SignalStorm { .. }
        | WorkType::PreemptStorm { .. }
        | WorkType::EpollStorm { .. }
        | WorkType::NumaMigrationChurn { .. }
        | WorkType::IdleChurn { .. }
        | WorkType::AluHot { .. }
        | WorkType::SmtSiblingSpin
        | WorkType::IpcVariance { .. } => true,
    }
}

fn map_sched_policy(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.sched_policy_hints.first() else {
        return;
    };
    match primary {
        SchedPolicyHint::Other { nice } => {
            spec.config.sched_policy = SchedPolicy::Normal;
            spec.config.nice = Some(*nice);
        }
        SchedPolicyHint::Fifo { priority } => {
            spec.config.sched_policy = SchedPolicy::Fifo(*priority);
        }
        SchedPolicyHint::RoundRobin { priority } => {
            spec.config.sched_policy = SchedPolicy::RoundRobin(*priority);
        }
        SchedPolicyHint::Deadline {
            runtime_ns,
            deadline_ns,
            period_ns,
        } => {
            spec.config.sched_policy = SchedPolicy::deadline(
                Duration::from_nanos(*runtime_ns),
                Duration::from_nanos(*deadline_ns),
                Duration::from_nanos(*period_ns),
            );
        }
        SchedPolicyHint::Batch => spec.config.sched_policy = SchedPolicy::Batch,
        SchedPolicyHint::Idle => spec.config.sched_policy = SchedPolicy::Idle,
        SchedPolicyHint::Ext => {
            // No SchedPolicy mapping for SCHED_EXT — the harness
            // routes tasks through scx by default. Note for the
            // generated source.
            spec.notes.push(ReproducerNote::Informational(
                "SchedPolicyHint::Ext observed; framework defaults to \
                 scx routing — no policy override emitted"
                    .into(),
            ));
        }
    }

    push_extras_note(
        &mut spec.notes,
        "additional sched-policy hints observed",
        fp.sched_policy_hints
            .iter()
            .skip(1)
            .map(|s| format!("{s:?}")),
    );
}

/// Render a `.run` shar-style entry-point source string from a
/// [`ReproducerSpec`].
///
/// The output is a small Rust program that constructs the
/// `WorkloadConfig` via builder calls and runs it through the
/// framework's standalone harness. Designed to drop into the
/// `.run` archive shape that `cargo ktstr export` already
/// produces — a self-contained reproducer the user can run
/// independently of their original test environment.
///
/// `template_name` becomes the generated function name. Use a
/// stable, descriptive name that survives a re-run with the same
/// capture (e.g. derived from `capture.scheduler_name` +
/// `capture.started_ns`).
#[allow(dead_code)]
pub fn render_run_file_source(spec: &ReproducerSpec, template_name: &str) -> String {
    let mut s = String::new();
    s.push_str("// Auto-generated reproducer from a debug capture.\n");
    s.push_str("// Edit the WorkloadConfig builder calls to refine\n");
    s.push_str("// the projection.\n\n");

    if !spec.scheduler_name.is_empty() {
        s.push_str(&format!("// Scheduler: {}\n", spec.scheduler_name));
    }
    if !spec.notes.is_empty() {
        s.push_str("//\n// Generator notes:\n");
        for note in &spec.notes {
            s.push_str(&format!("// - {note}\n"));
        }
        s.push('\n');
    }

    s.push_str("use ktstr::workload::*;\n");
    s.push_str("use std::collections::BTreeSet;\n");
    s.push_str("use std::time::Duration;\n\n");

    s.push_str(&format!("pub fn {template_name}() -> WorkloadConfig {{\n"));
    // Build from `WorkloadConfig::default()` and overlay every
    // captured value explicitly. Each `.workers/.affinity/...` call
    // pins the captured value directly, so the rendered spec runs
    // identically even if the upstream default changes — the
    // reproducer is independent of the host's `WorkloadConfig`
    // defaults at render time.
    s.push_str("    WorkloadConfig::default()\n");
    s.push_str(&format!("        .workers({})\n", spec.config.num_workers));
    s.push_str(&format!(
        "        .affinity({})\n",
        render_affinity(&spec.config.affinity)
    ));
    s.push_str(&format!(
        "        .work_type({})\n",
        render_work_type(&spec.config.work_type)
    ));
    s.push_str(&format!(
        "        .sched_policy({})\n",
        render_sched_policy(&spec.config.sched_policy)
    ));
    // Emit `.nice(n)` only when the spec carries `Some(n)`; `None`
    // is the framework's "skip the `setpriority(2)` call, inherit
    // the parent's nice" state, which is the type-level default
    // — emitting nothing here lands the same `nice = None` on the
    // generated `WorkloadConfig` and avoids a spurious `.nice(0)`
    // override of an unset field.
    if let Some(n) = spec.config.nice {
        s.push_str(&format!("        .nice({n})\n"));
    }
    s.push_str("}\n");

    if !spec.cgroup_hints.is_empty() {
        s.push_str("\n// Cgroup hints — apply at harness setup:\n");
        for h in &spec.cgroup_hints {
            s.push_str(&format!(
                "// {} (weight={:?}, mem_max={:?}, cpuset={:?}, cpu_max_quota_us={:?})\n",
                h.path, h.cpu_weight, h.memory_max_bytes, h.cpuset_cpus, h.cpu_max_quota_us,
            ));
        }
    }

    s
}

/// Render a `#[ktstr_test]`-decorated function from a
/// [`ReproducerSpec`].
///
/// Wraps [`render_run_file_source`]'s body with the proc-macro
/// attribute and a `#[allow(unused)]` import block. Output is
/// drop-in to a Rust file under `tests/`.
#[allow(dead_code)]
pub fn render_ktstr_test_source(spec: &ReproducerSpec, template_name: &str) -> String {
    let body = render_run_file_source(spec, template_name);
    // Prefix the generated function with the ktstr_test attribute.
    // The attribute applies to functions returning `WorkloadConfig`;
    // body's `pub fn` already matches that shape.
    body.replace(
        &format!("pub fn {template_name}"),
        &format!("#[ktstr::ktstr_test]\npub fn {template_name}"),
    )
}

fn render_affinity(a: &AffinityIntent) -> String {
    match a {
        AffinityIntent::Inherit => "AffinityIntent::Inherit".into(),
        AffinityIntent::SingleCpu => "AffinityIntent::SingleCpu".into(),
        AffinityIntent::LlcAligned => "AffinityIntent::LlcAligned".into(),
        AffinityIntent::CrossCgroup => "AffinityIntent::CrossCgroup".into(),
        AffinityIntent::SmtSiblingPair => "AffinityIntent::SmtSiblingPair".into(),
        AffinityIntent::RandomSubset { from, count } => {
            let cpus: Vec<String> = from.iter().map(|c| c.to_string()).collect();
            format!(
                "AffinityIntent::RandomSubset {{ from: BTreeSet::from([{}]), count: {} }}",
                cpus.join(", "),
                count
            )
        }
        AffinityIntent::Exact(set) => {
            let cpus: Vec<String> = set.iter().map(|c| c.to_string()).collect();
            format!(
                "AffinityIntent::Exact(BTreeSet::from([{}]))",
                cpus.join(", ")
            )
        }
    }
}

/// Render a [`WorkType`] back to Rust source. Exhaustive match: a
/// new variant added to [`WorkType`] in `crate::workload` is a
/// compile error here, forcing the reproducer generator to make a
/// deliberate decision about how to render it (either a real
/// builder call or a `TODO` placeholder via
/// [`render_work_type_todo`]). The exhaustive form prevents the
/// "silent collapse to SpinWait" failure mode of the previous
/// wildcard arm.
fn render_work_type(w: &WorkType) -> String {
    match w {
        // Variants the projection layer maps from a fingerprint hint
        // — render them as runnable builder calls.
        WorkType::SpinWait => "WorkType::SpinWait".into(),
        WorkType::YieldHeavy => "WorkType::YieldHeavy".into(),
        WorkType::Mixed => "WorkType::Mixed".into(),
        WorkType::IoSyncWrite => "WorkType::IoSyncWrite".into(),
        WorkType::IoRandRead => "WorkType::IoRandRead".into(),
        WorkType::IoConvoy => "WorkType::IoConvoy".into(),
        WorkType::Bursty {
            burst_duration,
            sleep_duration,
        } => format!(
            "WorkType::Bursty {{ \
             burst_duration: Duration::from_millis({}), \
             sleep_duration: Duration::from_millis({}) \
             }}",
            burst_duration.as_millis(),
            sleep_duration.as_millis(),
        ),
        WorkType::PipeIo { burst_iters } => {
            format!("WorkType::PipeIo {{ burst_iters: {burst_iters} }}")
        }
        WorkType::FutexPingPong { spin_iters } => {
            format!("WorkType::FutexPingPong {{ spin_iters: {spin_iters} }}")
        }
        WorkType::CachePressure { size_kib, stride } => {
            format!("WorkType::CachePressure {{ size_kib: {size_kib}, stride: {stride} }}")
        }
        // Variants that no fingerprint hint currently projects to.
        // Each produces an explicit TODO placeholder so the rendered
        // source compiles, surfaces the hand-edit requirement, and
        // names the unmapped variant. When a future
        // `WorkTypeHint::*` variant projects to one of these, move
        // the corresponding arm above this comment with a real
        // builder call.
        WorkType::CacheYield { .. } => render_work_type_todo("CacheYield"),
        WorkType::CachePipe { .. } => render_work_type_todo("CachePipe"),
        WorkType::FutexFanOut { .. } => render_work_type_todo("FutexFanOut"),
        WorkType::Sequence { .. } => render_work_type_todo("Sequence"),
        WorkType::ForkExit => render_work_type_todo("ForkExit"),
        WorkType::NiceSweep => render_work_type_todo("NiceSweep"),
        WorkType::AffinityChurn { .. } => render_work_type_todo("AffinityChurn"),
        WorkType::CrossAffinityChurn { .. } => render_work_type_todo("CrossAffinityChurn"),
        WorkType::PolicyChurn { .. } => render_work_type_todo("PolicyChurn"),
        WorkType::FanOutCompute { .. } => render_work_type_todo("FanOutCompute"),
        WorkType::Schbench { .. } => render_work_type_todo("Schbench"),
        WorkType::PageFaultChurn { .. } => render_work_type_todo("PageFaultChurn"),
        WorkType::MutexContention { .. } => render_work_type_todo("MutexContention"),
        WorkType::Custom { .. } => render_work_type_todo("Custom"),
        WorkType::ThunderingHerd { .. } => render_work_type_todo("ThunderingHerd"),
        WorkType::PriorityInversion { .. } => render_work_type_todo("PriorityInversion"),
        WorkType::ProducerConsumerImbalance { .. } => {
            render_work_type_todo("ProducerConsumerImbalance")
        }
        WorkType::RtStarvation { .. } => render_work_type_todo("RtStarvation"),
        WorkType::AsymmetricWaker { .. } => render_work_type_todo("AsymmetricWaker"),
        WorkType::WakeChain { .. } => render_work_type_todo("WakeChain"),
        WorkType::NumaWorkingSetSweep { .. } => render_work_type_todo("NumaWorkingSetSweep"),
        WorkType::CgroupChurn { .. } => render_work_type_todo("CgroupChurn"),
        WorkType::CgroupAttachStorm { .. } => render_work_type_todo("CgroupAttachStorm"),
        WorkType::SignalStorm { .. } => render_work_type_todo("SignalStorm"),
        WorkType::PreemptStorm { .. } => render_work_type_todo("PreemptStorm"),
        WorkType::EpollStorm { .. } => render_work_type_todo("EpollStorm"),
        WorkType::NumaMigrationChurn { .. } => render_work_type_todo("NumaMigrationChurn"),
        WorkType::IdleChurn { .. } => render_work_type_todo("IdleChurn"),
        WorkType::AluHot { .. } => render_work_type_todo("AluHot"),
        WorkType::SmtSiblingSpin => render_work_type_todo("SmtSiblingSpin"),
        WorkType::IpcVariance { .. } => render_work_type_todo("IpcVariance"),
    }
}

/// Render a placeholder `WorkType` builder call for a variant the
/// projection layer doesn't yet know how to translate from a
/// fingerprint hint. Output names the variant explicitly so the
/// hand-edit prompt is unambiguous, avoiding the silent
/// collapse-to-SpinWait failure mode the previous-generation
/// wildcard arm produced — every unmapped variant now resolves
/// through this fn with its own variant-named TODO placeholder.
fn render_work_type_todo(variant: &str) -> String {
    format!(
        "WorkType::SpinWait /* TODO: no fingerprint mapping for \
         WorkType::{variant} — refine from capture */"
    )
}

fn render_sched_policy(p: &SchedPolicy) -> String {
    match p {
        SchedPolicy::Normal => "SchedPolicy::Normal".into(),
        SchedPolicy::Batch => "SchedPolicy::Batch".into(),
        SchedPolicy::Idle => "SchedPolicy::Idle".into(),
        SchedPolicy::Fifo(prio) => format!("SchedPolicy::Fifo({prio})"),
        SchedPolicy::RoundRobin(prio) => format!("SchedPolicy::RoundRobin({prio})"),
        SchedPolicy::Deadline {
            runtime,
            deadline,
            period,
        } => format!(
            "SchedPolicy::Deadline {{ runtime: Duration::from_nanos({}), \
             deadline: Duration::from_nanos({}), period: Duration::from_nanos({}) }}",
            runtime.as_nanos(),
            deadline.as_nanos(),
            period.as_nanos(),
        ),
    }
}

#[cfg(test)]
#[path = "reproducer_gen_tests.rs"]
mod tests;
