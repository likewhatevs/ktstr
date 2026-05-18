//! [`SnapshotError`] (every fallible accessor's structured error) plus
//! its [`Display`] impl and [`SnapshotResult`] alias. Lives in its own
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
/// pre-client gates that the [`SchedStatsError`] enum doesn't
/// cover (no scheduler binary configured).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum MissingStatsReason {
    /// No `scheduler_binary` was configured on the run, so the
    /// freeze coordinator never wired a [`SchedStatsClient`].
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
    /// [`StatsResponse`] envelope. The `args` payload is preserved
    /// so operators can render the scheduler-side message.
    SchedulerError { errno: i32, args: serde_json::Value },
    /// The typed envelope was decoded but the inner `args` map
    /// did not contain the expected `"resp"` key — protocol
    /// mismatch with the scheduler.
    MissingResp { args: serde_json::Value },
    /// The caller passed a stats request larger than the client's
    /// [`MAX_REQUEST_BYTES`] cap.
    RequestTooLarge { size: usize, max: usize },
    /// The scheduler's response grew past
    /// [`MAX_RESPONSE_BYTES`] without ever emitting a newline.
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
    /// [`SchedStatsError`](crate::vmm::sched_stats::SchedStatsError)
    /// when one is present (every `SchedStatsClient` failure path
    /// boxes a typed variant via `anyhow::anyhow!(SchedStatsError::…)`,
    /// so the downcast succeeds on every well-formed sched_stats
    /// error). Falls back to [`MissingStatsReason::NoScheduler`]
    /// carrying the rendered display when the downcast fails — that
    /// covers serde / IO / other errors that didn't originate inside
    /// [`SchedStatsClient`] but still surface through the same
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
    /// with the requested name, so [`Snapshot::var`] cannot pick a
    /// deterministic answer. `found_in` lists every map (in capture
    /// order) where the name was seen — the caller should disambiguate
    /// via [`Snapshot::map`] and walk into the named map directly
    /// (e.g. `snap.map("scx_obj.bss")?.at(0).get("nr_cpus")`).
    AmbiguousVar {
        requested: String,
        found_in: Vec<String>,
    },
    /// A path component did not match any
    /// [`RenderedValue::Struct`] member at that depth. `requested`
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
    /// Keys are rendered via [`RenderedValue`]'s `Display` impl and
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
    /// narrowing to a CPU first via [`SnapshotMap::cpu`].
    PerCpuNotNarrowed { map: String },
    /// Hash entry has no rendered key/value side (BTF type id was
    /// missing at capture time, leaving the hex bytes only).
    NoRendered { map: String, side: String },
    /// The sample's underlying [`crate::monitor::dump::FailureDumpReport`]
    /// is a placeholder produced by
    /// [`crate::monitor::dump::FailureDumpReport::placeholder`] —
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
                     {found_in:?}); use Snapshot::map(name) to disambiguate"
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
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Result alias for snapshot accessors.
pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

/// Typed shape of one entry drained from the snapshot bridge's
/// ordered per-tag store. Fields:
/// * `tag`: snapshot name the report was stored under.
/// * `report`: [`crate::monitor::dump::FailureDumpReport`] of the
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
    pub step_index: Option<u16>,
}
