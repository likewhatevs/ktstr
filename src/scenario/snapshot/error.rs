//! [`SnapshotError`] (every fallible accessor's structured error) plus
//! its [`std::fmt::Display`] impl and [`SnapshotResult`] alias. Lives in its own
//! file so the variant catalogue is easy to scan when adding a new
//! accessor — `cargo doc` surfaces the same single-page view as the
//! source.

use super::HEX_KEY_PREFIX;

// ---------------------------------------------------------------------------
// Missing-stats reason
// ---------------------------------------------------------------------------

/// Why a sample's `stats` slot is unavailable — carried on
/// [`SnapshotError::MissingStats`] so operator diagnostics name
/// the specific failure mode rather than the generic "stats
/// absent". Built by [`From<&crate::vmm::sched_stats::SchedStatsError>`]
/// for the relay-failure path, plus dedicated variants for the
/// pre-client gates that the `crate::vmm::SchedStatsError` enum doesn't
/// cover (no scheduler binary configured).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum MissingStatsReason {
    /// No `scheduler_binary` was configured on the run, so the
    /// freeze coordinator never wired a `crate::vmm::SchedStatsClient`.
    /// Every periodic sample bypasses the stats request entirely
    /// and lands here.
    NoSchedulerBinary,
    /// The guest relay never connected to the scheduler's Unix
    /// socket (no scheduler running, or the scheduler refused the
    /// connection).
    NoScheduler { reason: String },
    /// The host-side coordinator marked the run as freezing while
    /// this stats request was in flight (or about to start);
    /// scx_stats responses are undefined while the scheduler's
    /// userspace thread is paused.
    DuringFreeze,
    /// The run-wide cancel flag was set (watchdog fired or the
    /// run is shutting down) while this stats request was in
    /// flight or about to start.
    Cancelled,
    /// The scheduler returned a non-zero `errno` in the typed
    /// `crate::vmm::StatsResponse` envelope. The `args` payload is preserved
    /// so operators can render the scheduler-side message.
    SchedulerError { errno: i32, args: serde_json::Value },
    /// The typed envelope was decoded but the inner `args` map
    /// did not contain the expected `"resp"` key — protocol
    /// mismatch with the scheduler.
    MissingResp { args: serde_json::Value },
    /// The caller passed a stats request larger than the client's
    /// `crate::vmm::sched_stats::MAX_REQUEST_BYTES` cap.
    RequestTooLarge { size: usize, max: usize },
    /// The scheduler's response grew past
    /// `crate::vmm::sched_stats::MAX_RESPONSE_BYTES` without ever emitting a newline.
    ResponseTooLarge { size: usize, max: usize },
    /// The shared response mutex was poisoned by a previous
    /// panic; the stats client cannot recover for this sample.
    MutexPoisoned,
}

impl std::fmt::Display for MissingStatsReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSchedulerBinary => {
                write!(f, "no scheduler_binary configured for this run")
            }
            Self::NoScheduler { reason } => {
                write!(f, "guest relay reports no scheduler: {reason}")
            }
            Self::DuringFreeze => {
                write!(
                    f,
                    "stats request cancelled — freeze coordinator paused the scheduler"
                )
            }
            Self::Cancelled => {
                write!(
                    f,
                    "stats request cancelled — run-wide cancel flag was set (watchdog or shutdown)"
                )
            }
            Self::SchedulerError { errno, args } => {
                write!(f, "scheduler returned errno={errno} (args={args})")
            }
            Self::MissingResp { args } => {
                write!(f, "scheduler envelope missing 'resp' key (args={args})")
            }
            Self::RequestTooLarge { size, max } => {
                write!(f, "stats request {size} bytes exceeds {max}-byte cap")
            }
            Self::ResponseTooLarge { size, max } => {
                write!(f, "stats response {size} bytes exceeds {max}-byte cap")
            }
            Self::MutexPoisoned => {
                write!(f, "stats client response mutex was poisoned")
            }
        }
    }
}

impl From<&anyhow::Error> for MissingStatsReason {
    /// Downcast the anyhow chain to a typed
    /// `crate::vmm::SchedStatsError`
    /// when one is present (every `SchedStatsClient` failure path
    /// boxes a typed variant via `anyhow::anyhow!(SchedStatsError::…)`,
    /// so the downcast succeeds on every well-formed sched_stats
    /// error). Falls back to [`MissingStatsReason::NoScheduler`]
    /// carrying the rendered display when the downcast fails — that
    /// covers serde / IO / other errors that didn't originate inside
    /// `crate::vmm::SchedStatsClient` but still surface through the same
    /// `Result<_, anyhow::Error>` return.
    fn from(e: &anyhow::Error) -> Self {
        if let Some(typed) = e.downcast_ref::<crate::vmm::sched_stats::SchedStatsError>() {
            return Self::from(typed);
        }
        Self::NoScheduler {
            reason: e.to_string(),
        }
    }
}

impl From<&crate::vmm::sched_stats::SchedStatsError> for MissingStatsReason {
    fn from(e: &crate::vmm::sched_stats::SchedStatsError) -> Self {
        use crate::vmm::sched_stats::SchedStatsError as S;
        match e {
            S::Poisoned => Self::MutexPoisoned,
            S::RequestTooLarge { size, max } => Self::RequestTooLarge {
                size: *size,
                max: *max,
            },
            S::ResponseTooLarge { size, max } => Self::ResponseTooLarge {
                size: *size,
                max: *max,
            },
            S::DuringFreeze => Self::DuringFreeze,
            S::Cancelled => Self::Cancelled,
            S::NoScheduler { reason } => Self::NoScheduler {
                reason: reason.clone(),
            },
            S::SchedulerError { errno, args } => Self::SchedulerError {
                errno: *errno,
                args: args.clone(),
            },
            S::MissingResp { args } => Self::MissingResp { args: args.clone() },
        }
    }
}

// ---------------------------------------------------------------------------
// Excluded map payload
// ---------------------------------------------------------------------------

/// One captured map that the KVA-whitelist filter rejected.
/// Payload for [`SnapshotError::ActiveFilterExcludedMaps::excluded_maps`].
/// The `map_kva` field name matches
/// [`crate::monitor::dump::FailureDumpMap::map_kva`] (the
/// source-of-truth field), and a `map_kva == 0` here flags a
/// capture where the per-map KVA was not recorded (synthetic
/// fixture or capture-path bug — production captures filter zero
/// KVAs out at the walker level).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ExcludedMap {
    pub name: String,
    pub map_kva: u64,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Reason a snapshot accessor or terminal read could not resolve.
///
/// Returned by every fallible accessor (`Snapshot::map`,
/// `SnapshotEntry::get`, `SnapshotField::as_u64`, …) so a missing
/// field, type mismatch, or absent map surfaces as a structured
/// error the test author can `?`-propagate. Each variant carries
/// the path / alternatives needed to fix the call site without
/// re-running the test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum SnapshotError {
    /// No map matched the requested name. `available` enumerates
    /// the captured map names so a typo surfaces in test output.
    MapNotFound {
        requested: String,
        available: Vec<String>,
    },
    /// No top-level global variable matched the requested name in
    /// any `*.bss` / `*.data` / `*.rodata` global-section map.
    /// `available` lists the union of every section's top-level
    /// member names.
    VarNotFound {
        requested: String,
        available: Vec<String>,
    },
    /// More than one global-section map exposes a top-level member
    /// with the requested name, so [`super::Snapshot::var`] cannot pick a
    /// deterministic answer. `found_in` lists every map (in capture
    /// order) where the name was seen — the caller should disambiguate
    /// via [`super::Snapshot::map`] and walk into the named map directly
    /// (e.g. `snap.map("scx_obj.bss")?.at(0).get("nr_cpus")`).
    AmbiguousVar {
        requested: String,
        found_in: Vec<String>,
    },
    /// A path component did not match any
    /// `crate::monitor::btf_render::RenderedValue::Struct` member at that depth. `requested`
    /// is the user-supplied lookup string; `walked` is the prefix
    /// that resolved successfully; `component` is the failing
    /// segment; `available` lists the struct's actual member names.
    FieldNotFound {
        requested: String,
        walked: String,
        component: String,
        available: Vec<String>,
    },
    /// A path component reached a non-Struct value where a struct
    /// was expected (e.g. descending into a `Uint` leaf).
    /// `requested` is the user-supplied lookup string; `kind` names
    /// the actual variant for diagnostics.
    NotAStruct {
        requested: String,
        walked: String,
        component: String,
        kind: String,
    },
    /// A typed accessor (`as_u64` etc.) was called on a rendered
    /// shape it cannot decode (e.g. `as_str` on a `Struct`).
    /// `expected` names the scalar type the accessor requires;
    /// `actual` names the rendered variant; `requested` is the
    /// user-supplied lookup string (empty when the accessor was
    /// invoked on a leaf without a path walk).
    TypeMismatch {
        expected: String,
        actual: String,
        requested: String,
    },
    /// A map index was out of range for the underlying entry list.
    IndexOutOfRange {
        map: String,
        index: usize,
        len: usize,
    },
    /// A per-CPU slot was out of range or unmapped.
    PerCpuSlot {
        map: String,
        cpu: u32,
        len: usize,
        unmapped: bool,
    },
    /// A predicate-based lookup (`find`, `max_by`) found no match.
    /// `len` is the number of entries the lookup traversed before
    /// giving up; `available_keys` is a small sample (up to
    /// `NO_MATCH_KEY_SAMPLE` entries) of rendered keys seen during
    /// the traversal so an operator can distinguish "empty map"
    /// (`len == 0`) from "populated map with no predicate hit"
    /// (`len > 0`) and inspect the sample to debug the predicate.
    /// Keys are rendered via `crate::monitor::btf_render::RenderedValue`'s `Display` impl and
    /// each is capped at `NO_MATCH_KEY_CHAR_CAP` chars with an
    /// ellipsis to keep the failure message readable for wide struct
    /// keys.
    ///
    /// Aggregation methods (`max_by`, `cpu_max_u64` / `cpu_min_u64`
    /// / `cpu_max_f64` / `cpu_min_f64`) produce this variant for
    /// empty / all-None inputs; their NoMatch always carries
    /// `len == 0` and empty `available_keys`. Only `find` can
    /// produce `len > 0` here.
    NoMatch {
        map: String,
        op: String,
        len: usize,
        available_keys: Vec<String>,
    },
    /// A path string contained an empty component (e.g. `"a..b"`).
    /// `requested` is the user-supplied lookup string.
    EmptyPathComponent { requested: String },
    /// `EntryAccessor::get` was called on a per-CPU entry without
    /// narrowing to a CPU first via [`super::SnapshotMap::cpu`].
    PerCpuNotNarrowed { map: String },
    /// Hash entry has no rendered key/value side (BTF type id was
    /// missing at capture time, leaving the hex bytes only).
    NoRendered { map: String, side: String },
    /// The sample's underlying `crate::monitor::dump::FailureDumpReport`
    /// is a placeholder produced by
    /// `crate::monitor::dump::FailureDumpReport::placeholder` —
    /// the freeze-rendezvous path could not collect real data
    /// (typical cause: vCPU rendezvous timed out). Temporal
    /// patterns in [`crate::assert::temporal`] route this variant
    /// through their per-sample skip handling so a placeholder
    /// sample never falsely registers as zero progress against a
    /// monotonicity / rate / steady / ratio band. The `reason`
    /// string mirrors `FailureDumpReport::scx_walker_unavailable`
    /// when present (set by `placeholder()` to the constructor
    /// argument), giving the operator the cause without re-walking
    /// the report.
    PlaceholderSample { tag: String, reason: String },
    /// A [`SampleSeries::stats`](crate::scenario::sample::SampleSeries::stats)
    /// projection ran on a sample whose `stats` field carries an
    /// `Err` — the stats client was not wired (no
    /// `scheduler_binary`) or the per-sample stats request failed.
    /// The carried [`MissingStatsReason`] identifies the *why* so
    /// operator diagnostics distinguish "no scheduler configured"
    /// from "scheduler refused the request" from "watchdog
    /// cancelled the request" without re-walking the source error.
    /// Distinguishes a per-sample stats coverage gap from an
    /// in-stats-JSON path miss (`TypeMismatch` /
    /// `FieldNotFound`) so the temporal-assertion site can
    /// branch on the cause without re-walking the source.
    MissingStats {
        tag: String,
        reason: MissingStatsReason,
    },
    /// A [`SampleSeries::host`](crate::scenario::sample::SampleSeries::host)
    /// projection ran on a sample whose `per_cpu_time` slice did
    /// not include `cpu` — placeholder report (freeze rendezvous
    /// timed out), or a kernel that didn't surface per-CPU
    /// `kernel_stat`/`tick_cpu_sched`/`kernel_cpustat` resolution
    /// for the requested CPU. Distinguishes a per-sample host-data
    /// coverage gap from a kernel-walker failure (`Unavailable` on
    /// the broader Snapshot accessor) so the temporal-assertion
    /// site can decide whether to fail strict or skip with a
    /// rendered Note.
    HostFieldUnavailable { tag: String, cpu: u32 },
    /// [`super::Snapshot::var`] / [`super::Snapshot::live_var`] /
    /// [`super::Snapshot::map`] was called on a snapshot whose
    /// underlying `crate::monitor::dump::FailureDumpReport` is a
    /// placeholder (the freeze-rendezvous path could not collect
    /// real data — typical cause: vCPU rendezvous timed out). The
    /// captured `report.maps` is empty by construction so the
    /// var/map lookup has nothing to walk. Distinct from
    /// [`Self::VarNotFound`] (which means "the captured report did
    /// not contain a global by this name") so the assertion site
    /// can distinguish "freeze failed" from "typo in field name".
    /// `tag` carries the capture tag (if any).
    PlaceholderSnapshot { tag: Option<String> },
    /// [`super::Snapshot::active`] / [`super::Snapshot::live_var`]
    /// could not identify a currently-active scheduler from the
    /// snapshot's `*scx_root` + `prog_runtime_stats`. Typical
    /// causes: snapshot taken in the dead window between
    /// [`crate::scenario::ops::Op::DetachScheduler`] +
    /// [`crate::scenario::ops::Op::AttachScheduler`]; snapshot
    /// taken in the post-swap settle window before the new
    /// scheduler's progs have advanced their run counter; snapshot
    /// captured before any scheduler attached. Distinct from
    /// [`Self::AmbiguousVar`] (which means "the snapshot has
    /// multiple scheduler bss copies and the call did not opt
    /// into active-only filtering") so the assertion site can
    /// distinguish "no scheduler is running right now" from
    /// "multiple are running, pick one".
    NoActiveScheduler { reason: String },
    /// [`super::Snapshot::var`] / [`super::Snapshot::map`] (or one
    /// of the `live_*` shortcuts) ran against an active-filtered
    /// view where the KVA whitelist excluded EVERY captured map
    /// that shared the active obj prefix (i.e. the admitted set
    /// for this obj was empty). Distinct from [`Self::VarNotFound`]
    /// — `VarNotFound` means "the active filter admitted maps but
    /// none carry the requested name"; this variant means "the
    /// active filter admitted zero maps for this obj, so the
    /// lookup never got the chance to walk anything."
    ///
    /// The variant never fires when at least one captured
    /// `<active_obj>.*` map passes the KVA whitelist — in that
    /// case the lookup miss is a real typo or absent symbol and
    /// the standard `VarNotFound` / `MapNotFound` carries the
    /// admitted list. This narrow firing scope prevents
    /// false-positives that would otherwise mask genuine typos
    /// in same-binary post-swap captures.
    ///
    /// Typical causes when this DOES fire: stale walker capture
    /// (captured KVAs predate the most recent struct_ops swap),
    /// same-binary post-swap window where the report still
    /// carries the old instance's maps, or a walker bug that
    /// resolved `*scx_root` against a different binary's map set.
    ActiveFilterExcludedMaps {
        /// User-supplied lookup string (the `var` / `map`
        /// argument). For [`super::Snapshot::live_vars_via`] this
        /// carries the joined name list `"[a, b, c]"`.
        requested: String,
        /// Obj name the active filter pinned to
        /// (`*scx_root → struct_ops map → obj prefix` resolution).
        active_obj: String,
        /// Maps captured under the active obj prefix that the KVA
        /// whitelist rejected.
        excluded_maps: Vec<ExcludedMap>,
        /// KVA whitelist the walker populated for the active obj.
        /// A non-empty set whose every entry mismatched the
        /// captured `map_kva` values points at stale capture or
        /// KVA aliasing; an empty set is unreachable through this
        /// variant (no filter means no exclusion).
        whitelist_kvas: Vec<u64>,
    },
    /// A walker-resolved [`crate::scenario::sample::SampleSeries::bpf_live_u64`]
    /// / `bpf_live_i64` / `bpf_live_f64` projection detected that
    /// the snapshot's per-snapshot walker output
    /// ([`crate::monitor::dump::FailureDumpReport::active_map_kvas`])
    /// disagrees with an earlier same-phase snapshot's walker
    /// output for the same lookup. The framework pins the first
    /// non-empty walker output it sees per phase and surfaces this
    /// variant for every later same-phase snapshot whose walker
    /// resolved to a different KVA set — without this gate the
    /// projected series would silently switch between bss copies
    /// mid-phase (typical cause: post-`Op::ReplaceScheduler` swap
    /// window where the walker re-publishes mid-phase) and
    /// downstream reducers like
    /// [`crate::assert::temporal::SeriesField::counter_delta_per_phase`]
    /// would see non-monotonic counter values. The drifted
    /// samples become per-sample `Err` slots; the temporal
    /// patterns' standard error-skip semantics apply.
    WalkerDriftedWithinPhase {
        phase: crate::assert::Phase,
        pinned_kvas: Vec<u64>,
        sample_kvas: Vec<u64>,
        requested: String,
    },
    /// A user-supplied projection closure (the kind passed to
    /// [`crate::scenario::sample::SampleSeries::bpf`]) signalled
    /// failure for reasons that don't fit the structured variants
    /// above. `reason` is the closure's free-form explanation —
    /// "lookup returned None for sched_id A, B, C" — so the failure
    /// message stays diagnostic without forcing the closure to
    /// synthesize an `available: Vec<String>` it cannot populate.
    ///
    /// Closures should reach for the structured variants
    /// ([`Self::VarNotFound`], [`Self::MapNotFound`], etc.) when
    /// they can; this variant is the escape hatch for higher-level
    /// disambiguation logic (e.g. "I walked vars(name) and none of
    /// the candidates matched my active-instance fingerprint").
    /// Surfaces in temporal-assertion failure messages as
    /// `projection failed: <reason>`.
    ProjectionFailed { reason: String },
    /// A captured map's contents could not be rendered at dump time:
    /// `crate::monitor::dump::FailureDumpMap::error` is set and the
    /// map carries no entries / value to walk. Surfaced by
    /// [`super::Snapshot::var`] / [`super::SnapshotMap::at`] /
    /// [`super::SnapshotMap::find`] / [`super::SnapshotMap::max_by`]
    /// instead of [`Self::VarNotFound`] /
    /// [`Self::IndexOutOfRange`] `{ len: 0 }` /
    /// [`Self::NoMatch`] `{ len: 0 }` so a guest-memory render
    /// failure is distinguishable from a genuinely-absent variable
    /// or a legitimately-empty map. Without this distinction a map
    /// whose contents failed to read reads identically to "the
    /// symbol does not exist", masking the capture failure.
    /// `map` names the owning map; `error` mirrors
    /// `FailureDumpMap.error`.
    MapRenderIncomplete { map: String, error: String },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::MapNotFound {
                requested,
                available,
            } => {
                write!(
                    f,
                    "snapshot has no map '{requested}' (captured maps: {available:?})"
                )
            }
            SnapshotError::VarNotFound {
                requested,
                available,
            } => {
                write!(
                    f,
                    "snapshot has no global variable '{requested}' in any \
                     *.bss/*.data/*.rodata map (available globals: {available:?})"
                )
            }
            SnapshotError::AmbiguousVar {
                requested,
                found_in,
            } => {
                write!(
                    f,
                    "snapshot global '{requested}' is ambiguous (found in \
                     {found_in:?}); use Snapshot::active().var(name) (or the \
                     shorthand Snapshot::live_var(name)) to pick the active \
                     scheduler's copy automatically, or Snapshot::map(name) \
                     to address a specific scheduler's bss explicitly"
                )
            }
            SnapshotError::FieldNotFound {
                requested,
                walked,
                component,
                available,
            } => {
                write!(
                    f,
                    "path '{requested}': component '{component}' (after walking '{walked}') \
                     not found (members at this depth: {available:?})"
                )
            }
            SnapshotError::NotAStruct {
                requested,
                walked,
                component,
                kind,
            } => {
                write!(
                    f,
                    "path '{requested}': component '{component}' (after walking '{walked}') \
                     expected a Struct, got {kind}"
                )
            }
            SnapshotError::TypeMismatch {
                expected,
                actual,
                requested,
            } => {
                write!(
                    f,
                    "path '{requested}': cannot read as {expected} — actual rendered \
                     variant is {actual}"
                )
            }
            SnapshotError::IndexOutOfRange { map, index, len } => {
                write!(f, "map '{map}': index {index} out of range (length {len})")
            }
            SnapshotError::PerCpuSlot {
                map,
                cpu,
                len,
                unmapped,
            } => {
                if *unmapped {
                    write!(f, "map '{map}': cpu {cpu} per-CPU slot is unmapped (None)")
                } else {
                    write!(
                        f,
                        "map '{map}': cpu {cpu} out of range (have {len} per-CPU slots)"
                    )
                }
            }
            SnapshotError::NoMatch {
                map,
                op,
                len,
                available_keys,
            } => {
                if *len == 0 {
                    write!(f, "map '{map}': {op} matched no entries (map is empty)")
                } else if available_keys.is_empty() {
                    write!(
                        f,
                        "map '{map}': {op} matched none of {len} entries (sample keys unavailable)"
                    )
                } else {
                    write!(
                        f,
                        "map '{map}': {op} matched none of {len} entries (first {sampled}: {available_keys:?})",
                        sampled = available_keys.len(),
                    )?;
                    // The `hex:` prefix is only ever produced by
                    // `render_entry_key`'s fallback path when the
                    // entry's `key` field was `None` at capture time.
                    // Typed `RenderedValue::Display` does not emit
                    // this prefix for any scalar variant; `Struct`
                    // emits `TypeName{...}` inline or `TypeName:`
                    // breadcrumb, where a `hex:` collision would
                    // require a BTF struct literally named `hex` —
                    // no real kernel scheduler does that. The hint
                    // therefore fires only when BTF was uniformly
                    // absent for this map's key type at capture time,
                    // and names the kernel-side fix so the operator
                    // does not have to reverse-engineer the `hex:`
                    // discriminator.
                    if available_keys.iter().all(|k| k.starts_with(HEX_KEY_PREFIX)) {
                        write!(
                            f,
                            " (BTF missing at capture — keys shown as hex bytes; \
                             rebuild guest kernel with CONFIG_DEBUG_INFO_BTF=y for \
                             typed keys)"
                        )?;
                    }
                    Ok(())
                }
            }
            SnapshotError::EmptyPathComponent { requested } => {
                write!(
                    f,
                    "path '{requested}' has an empty component (consecutive '.')"
                )
            }
            SnapshotError::PerCpuNotNarrowed { map } => {
                write!(
                    f,
                    "map '{map}': per-CPU entry without a CPU narrow — call .cpu(N) first"
                )
            }
            SnapshotError::NoRendered { map, side } => {
                write!(
                    f,
                    "map '{map}': {side} has no rendered structure (no BTF type at capture time)"
                )
            }
            SnapshotError::PlaceholderSample { tag, reason } => {
                write!(
                    f,
                    "sample '{tag}' is a placeholder report (capture pipeline did not land): \
                     {reason}"
                )
            }
            SnapshotError::MissingStats { tag, reason } => {
                write!(f, "sample '{tag}': stats absent ({reason})")
            }
            SnapshotError::HostFieldUnavailable { tag, cpu } => {
                write!(
                    f,
                    "sample '{tag}': per_cpu_time has no entry for cpu {cpu} \
                     (placeholder report or kernel-walker resolution failure)"
                )
            }
            SnapshotError::PlaceholderSnapshot { tag } => match tag {
                Some(t) => write!(
                    f,
                    "snapshot '{t}' is a placeholder — the freeze-rendezvous \
                     path could not capture real data; no maps to walk"
                ),
                None => f.write_str(
                    "snapshot is a placeholder — the freeze-rendezvous path \
                     could not capture real data; no maps to walk",
                ),
            },
            SnapshotError::NoActiveScheduler { reason } => {
                write!(
                    f,
                    "snapshot has no currently-active scheduler ({reason}); \
                     use Snapshot::vars(name) to enumerate every observed \
                     copy explicitly, Snapshot::live_var(name) to keep the \
                     typed error path while opting into the active filter, \
                     or Snapshot::map(\"<obj>.<section>\") to address a \
                     specific scheduler's bss directly"
                )
            }
            SnapshotError::ActiveFilterExcludedMaps {
                requested,
                active_obj,
                excluded_maps,
                whitelist_kvas,
            } => {
                let excluded_rendered = excluded_maps
                    .iter()
                    .map(|m| format!("{}@{:#x}", m.name, m.map_kva))
                    .collect::<Vec<_>>()
                    .join(", ");
                let some_zero = excluded_maps.iter().any(|m| m.map_kva == 0);
                let some_alias = excluded_maps
                    .iter()
                    .any(|m| m.map_kva != 0 && !whitelist_kvas.contains(&m.map_kva));
                let cause = match (some_zero, some_alias) {
                    (false, true) => {
                        "this snapshot pre-dates your most recent \
                         Op::ReplaceScheduler / Op::AttachScheduler — \
                         wait for the next periodic boundary (or re-run \
                         the test) so the walker re-publishes the live \
                         scheduler's KVAs"
                    }
                    (true, false) => {
                        "the captured maps have no recorded KVAs — \
                         the snapshot pre-dates the walker plumbing, \
                         or the capture path failed to record per-map KVAs"
                    }
                    (true, true) => {
                        "some captured maps lack KVAs and some disagree \
                         with the walker's whitelist — both \
                         pre-walker-capture state and a post-swap window \
                         can produce this; re-run the test to regenerate \
                         the snapshot"
                    }
                    (false, false) => "captured KVAs were neither absent nor in disagreement",
                };
                write!(
                    f,
                    "snapshot lookup '{requested}' returned no hits under the \
                     active filter (obj='{active_obj}'): the walker's KVA \
                     whitelist {whitelist_kvas:#x?} excluded {n} captured map(s) \
                     sharing the obj prefix: {excluded_rendered} — {cause}. \
                     Reach for Snapshot::vars('{requested}') to enumerate every \
                     copy across all obj prefixes, or Snapshot::map(\"<name>\") \
                     to address one of the excluded maps directly.",
                    n = excluded_maps.len(),
                )
            }
            SnapshotError::WalkerDriftedWithinPhase {
                phase,
                pinned_kvas,
                sample_kvas,
                requested,
            } => {
                write!(
                    f,
                    "walker drift within {phase:?}: lookup '{requested}' resolved against \
                     KVA set {sample_kvas:#x?}, but an earlier same-phase snapshot pinned \
                     {pinned_kvas:#x?}. The walker re-published mid-phase (typical cause: \
                     a post-Op::ReplaceScheduler swap window). The drifted sample is \
                     surfaced as Err so per-phase reducers (counter_delta_per_phase, \
                     ratio_across_phases) see monotonic Ok-sequences from one walker \
                     decision; address by stepping the phase past the swap settle window \
                     or by reading via the explicit picker form."
                )
            }
            SnapshotError::ProjectionFailed { reason } => {
                write!(f, "projection failed: {reason}")
            }
            SnapshotError::MapRenderIncomplete { map, error } => {
                write!(
                    f,
                    "map '{map}' contents unavailable (render failed at capture): {error}"
                )
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Result alias for snapshot accessors.
pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

/// Typed shape of one entry drained from the snapshot bridge's
/// ordered per-tag store. Fields:
/// * `tag`: snapshot name the report was stored under.
/// * `report`: `crate::monitor::dump::FailureDumpReport` of the
///   captured guest state.
/// * `stats`: scheduler-side stats JSON or a typed
///   [`MissingStatsReason`] when capture happened without a
///   wired stats client.
/// * `elapsed_ms`: optional wall-clock anchor (ms since run-start).
/// * `step_index`: scenario phase index stamped at capture time.
///   `Some(idx)` for captures stored via the step-aware entry
///   points ([`crate::scenario::snapshot::SnapshotBridge::capture_with_step`]
///   or [`crate::scenario::snapshot::SnapshotBridge::store_with_stats_and_step`]);
///   `None` for fixture-injected captures via the unstamped legacy
///   paths ([`crate::scenario::snapshot::SnapshotBridge::capture`]
///   / [`crate::scenario::snapshot::SnapshotBridge::store`]
///   / [`crate::scenario::snapshot::SnapshotBridge::store_with_stats`]).
///
/// Used by [`crate::scenario::snapshot::SnapshotBridge::drain_ordered_with_stats`]
/// and [`crate::scenario::sample::SampleSeries::from_drained_typed`].
/// `#[non_exhaustive]` so future additive fields stay
/// pattern-match-compatible via rest-pattern destructure
/// (`DrainedSnapshotEntry { tag, report, .. }`).
#[derive(Debug)]
#[non_exhaustive]
pub struct DrainedSnapshotEntry {
    pub tag: String,
    pub report: crate::monitor::dump::FailureDumpReport,
    pub stats: std::result::Result<serde_json::Value, MissingStatsReason>,
    pub elapsed_ms: Option<u64>,
    /// Workload-relative boundary offset (ms) a periodic capture was
    /// scheduled for (`boundary_ns - scenario_anchor_ns`); `None` for
    /// non-periodic / on-demand captures. Distinct from `elapsed_ms`
    /// (run_start-relative fire time). See
    /// [`crate::scenario::snapshot::SnapshotBridge`]'s store doc.
    pub boundary_offset_ms: Option<u64>,
    pub step_index: Option<u16>,
}

#[cfg(test)]
mod tests_api_gaps {
    use super::*;

    /// Pin: `SnapshotError::ProjectionFailed { reason }` renders as
    /// `projection failed: <reason>` so the temporal-assertion
    /// failure path surfaces the closure's diagnostic without
    /// re-wrapping. Closure call-sites synthesize this variant
    /// when the structured variants (`VarNotFound`, `MapNotFound`,
    /// `AmbiguousVar`) require an `available: Vec<String>` they
    /// cannot populate.
    #[test]
    fn projection_failed_display_carries_reason() {
        let e = SnapshotError::ProjectionFailed {
            reason: "live_var_via picker rejected all 2 candidates".to_string(),
        };
        let rendered = format!("{e}");
        assert_eq!(
            rendered,
            "projection failed: live_var_via picker rejected all 2 candidates"
        );
    }

    /// Pin: `ProjectionFailed` participates in the same
    /// `PartialEq` / `Hash` derive set as every other variant —
    /// pattern-match callers can assert "yes, my projection
    /// closure failed" without falling through to a `_` arm.
    #[test]
    fn projection_failed_eq_and_hash_round_trip() {
        let a = SnapshotError::ProjectionFailed {
            reason: "x".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);
        let mut seen = std::collections::HashSet::new();
        seen.insert(a);
        assert!(seen.contains(&b));
    }

    /// `From<&SchedStatsError>` must map EVERY scheduler-stats failure
    /// to its corresponding MissingStatsReason. A swapped arm would
    /// mislabel a stats failure (e.g. a size-cap breach reported as a
    /// poisoned mutex), and the operator's diagnostic would point at the
    /// wrong cause. Pin each variant's mapping.
    #[test]
    fn missing_stats_reason_from_sched_stats_error_maps_each_variant() {
        use crate::vmm::sched_stats::SchedStatsError as S;
        use serde_json::json;
        assert_eq!(
            MissingStatsReason::from(&S::Poisoned),
            MissingStatsReason::MutexPoisoned,
        );
        assert_eq!(
            MissingStatsReason::from(&S::DuringFreeze),
            MissingStatsReason::DuringFreeze,
        );
        assert_eq!(
            MissingStatsReason::from(&S::Cancelled),
            MissingStatsReason::Cancelled,
        );
        assert_eq!(
            MissingStatsReason::from(&S::RequestTooLarge { size: 10, max: 5 }),
            MissingStatsReason::RequestTooLarge { size: 10, max: 5 },
        );
        assert_eq!(
            MissingStatsReason::from(&S::ResponseTooLarge { size: 20, max: 8 }),
            MissingStatsReason::ResponseTooLarge { size: 20, max: 8 },
        );
        assert_eq!(
            MissingStatsReason::from(&S::NoScheduler {
                reason: "no sock".into()
            }),
            MissingStatsReason::NoScheduler {
                reason: "no sock".into()
            },
        );
        assert_eq!(
            MissingStatsReason::from(&S::SchedulerError {
                errno: 13,
                args: json!({"k": 1}),
            }),
            MissingStatsReason::SchedulerError {
                errno: 13,
                args: json!({"k": 1}),
            },
        );
        assert_eq!(
            MissingStatsReason::from(&S::MissingResp {
                args: json!({"a": 2}),
            }),
            MissingStatsReason::MissingResp {
                args: json!({"a": 2}),
            },
        );
    }

    /// `From<&anyhow::Error>` downcasts a typed SchedStatsError in the
    /// chain to its mapped variant; a non-typed error falls back to
    /// NoScheduler carrying the rendered display (so serde/IO errors
    /// surfacing through the same Result still classify).
    #[test]
    fn missing_stats_reason_from_anyhow_downcasts_typed_else_no_scheduler() {
        use crate::vmm::sched_stats::SchedStatsError as S;
        let typed: anyhow::Error = anyhow::Error::new(S::DuringFreeze);
        assert_eq!(
            MissingStatsReason::from(&typed),
            MissingStatsReason::DuringFreeze,
            "a typed SchedStatsError in the anyhow chain must downcast",
        );
        let other = anyhow::anyhow!("plain io failure");
        match MissingStatsReason::from(&other) {
            MissingStatsReason::NoScheduler { reason } => {
                assert!(reason.contains("plain io failure"));
            }
            x => panic!("expected NoScheduler fallback, got {x:?}"),
        }
    }

    /// Every MissingStatsReason renders a non-empty operator message,
    /// and the value-bearing variants surface their payload.
    #[test]
    fn missing_stats_reason_display_covers_each_variant() {
        use serde_json::json;
        let cases = [
            MissingStatsReason::NoSchedulerBinary,
            MissingStatsReason::NoScheduler { reason: "r".into() },
            MissingStatsReason::DuringFreeze,
            MissingStatsReason::Cancelled,
            MissingStatsReason::SchedulerError {
                errno: 1,
                args: json!({}),
            },
            MissingStatsReason::MissingResp { args: json!({}) },
            MissingStatsReason::RequestTooLarge { size: 1, max: 2 },
            MissingStatsReason::ResponseTooLarge { size: 3, max: 4 },
            MissingStatsReason::MutexPoisoned,
        ];
        for c in &cases {
            assert!(
                !format!("{c}").is_empty(),
                "Display must be non-empty for {c:?}"
            );
        }
        assert!(
            format!(
                "{}",
                MissingStatsReason::RequestTooLarge { size: 10, max: 5 }
            )
            .contains("10"),
        );
        assert!(
            format!(
                "{}",
                MissingStatsReason::SchedulerError {
                    errno: 13,
                    args: json!({})
                }
            )
            .contains("13"),
        );
    }
}
