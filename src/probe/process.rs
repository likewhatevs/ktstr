//! Probe skeleton lifecycle: load, attach, run, collect.
//!
//! When the pipeline is split into two phases (see [`PhaseBInput`]):
//! - **Phase A** attaches kprobes + the trigger + kernel fexit before
//!   the scheduler starts; it runs under the initial skeleton load.
//! - **Phase B** attaches fentry/fexit to the scheduler's BPF
//!   struct_ops callbacks after the scheduler is up and BPF programs
//!   are discoverable.
//!
//! In the single-phase path, all probes attach after the scheduler is
//! up.
//!
//! ## Two-phase sync mechanism
//!
//! Phase A runs on a probe worker thread. Caller and worker
//! synchronize via two `Latch`es, one status slot, and one mpsc channel:
//!
//! 1. Caller spawns the probe worker, which loads the skeleton and
//!    attaches kprobes + trigger + kernel fexit, publishes Phase A's
//!    success/failure status, then signals the `probes_ready` latch
//!    (see `publish_phase_a` below). On success the worker enters the
//!    ringbuf poll loop.
//! 2. Caller waits on `probes_ready` and reads the published status.
//!    Only after success does it start the scheduler — the scheduler
//!    launches with Phase A probes already attached, so the trigger
//!    and any kprobes that fire during scheduler init are observed.
//! 3. After the scheduler is up, the caller discovers BPF programs
//!    by scheduler pid (see `discover_bpf_symbols` /
//!    `expand_bpf_to_kernel_callers`) and sends a [`PhaseBInput`]
//!    over the channel. The Phase B input includes BPF program FDs
//!    held open while the scheduler is alive.
//! 4. The probe worker's poll loop calls `try_recv` on the channel
//!    every 100 ms; on receipt it attaches BPF fentry/fexit + extra
//!    kprobes for kernel callers, then signals the `done` latch on
//!    the [`PhaseBInput`].
//! 5. Caller waits on `done` and proceeds to the test scenario with
//!    full instrumentation in place.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use super::btf::{BtfFunc, RenderHint, STRUCT_FIELDS};
use super::stack::StackFunction;

use crate::bpf_skel::types;
use crate::sync::{Latch, RwLockExt};

/// Cross-thread mirror of the BPF probe's `ktstr_err_exit_detected`
/// BSS latch. The probe-poll thread reads the BSS slot via volatile
/// load each iteration and `store`s the observed value here; scenario
/// code calls [`sched_exit_kind`] from any thread to read the current
/// classification synchronously without owning the skeleton.
///
/// Encoded as `u32` for `AtomicU32`-friendliness:
/// - `0` (initial / probe-not-yet-armed / scheduler still alive):
///   [`SchedExitKind::Unknown`].
/// - `1` ([`SchedExitKind::Clean`]) — probe armed, BSS latch read as
///   `0` after at least one poll iteration. Means the scheduler is
///   either still alive (latch never set) or exited via the clean
///   `SCX_EXIT_NONE` path.
/// - `2` ([`SchedExitKind::Crashed`]) — probe armed, BSS latch
///   observed at `!= 0`. Set by the selected BPF exit handler
///   after any successfully claimed non-clean kernel exit
///   (`SCX_EXIT_ERROR`, `SCX_EXIT_ERROR_STALL`, watchdog kick,
///   BPF-side error).
///
/// 0 → 1/2 transitions are monotonic for a single test run; once the
/// accepted exit path returns, the latch never clears. Cross-thread
/// observation uses Release / Acquire so the scenario-side load sees
/// every prior probe-side store.
pub(crate) static PROBE_SCHED_EXIT_STATE: AtomicU32 = AtomicU32::new(0);

const PROBE_EXIT_STATE_CLEAN: u32 = 1;
const PROBE_EXIT_STATE_CRASHED: u32 = 2;

/// Classification of the scheduler's exit observed by the probe
/// poll thread. Mirrors the BSS latch in
/// [`PROBE_SCHED_EXIT_STATE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedExitKind {
    /// Probe has not observed the scheduler's exit yet — either the
    /// scheduler is still alive, the probe never armed for this run,
    /// or the poll thread has not completed a single iteration since
    /// the prior reset. Callers gating clean-vs-crash classification
    /// on a known-dead scheduler treat `Unknown` as a probe-coverage
    /// gap, not as evidence of either outcome.
    Unknown,
    /// Probe armed AND completed at least one observation with the
    /// BSS latch unset. Scheduler is either still alive or exited
    /// cleanly via the `SCX_EXIT_NONE` path that does not latch the
    /// error flag.
    Clean,
    /// Probe armed AND observed the BSS latch set. The probe observed a
    /// successful non-clean scheduler exit (error, stall,
    /// watchdog kick, BPF-side error) before this read.
    Crashed,
}

/// Read the current probe-observed scheduler exit classification.
/// Synchronous, cross-thread-safe (Acquire load mirrors the probe's
/// Release stores). Returns [`SchedExitKind::Unknown`] when the
/// probe has not yet completed its first poll iteration.
pub fn sched_exit_kind() -> SchedExitKind {
    match PROBE_SCHED_EXIT_STATE.load(Ordering::Acquire) {
        PROBE_EXIT_STATE_CLEAN => SchedExitKind::Clean,
        PROBE_EXIT_STATE_CRASHED => SchedExitKind::Crashed,
        _ => SchedExitKind::Unknown,
    }
}

/// Render the useful tail of libbpf's verifier/load log for an early Phase-A
/// failure.
///
/// The complete bounded log remains in `ProbeDiagnostics`; guest init needs a
/// concise reason before it reboots, otherwise a missing typed-BTF target is
/// reduced to the opaque outer `ESRCH`. Preserve the final six non-empty
/// records (where libbpf places the program and target diagnostics), collapse
/// whitespace, and cap the result so this control error cannot flood stdout.
fn bounded_libbpf_tail(log: &str) -> String {
    const MAX_LINES: usize = 6;
    const MAX_CHARS: usize = 1_000;

    let mut lines: Vec<String> = log
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .rev()
        .take(MAX_LINES)
        .collect();
    lines.reverse();
    let condensed = lines.join(" | ");
    if condensed.chars().count() > MAX_CHARS {
        format!(
            "…{}",
            condensed
                .chars()
                .rev()
                .take(MAX_CHARS - 1)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        )
    } else {
        condensed
    }
}

/// Kernel-generation-specific hook used to observe sched_ext exits.
///
/// All programs live in the same BPF object. Userspace selects exactly one
/// before load so a missing target cannot reject the whole object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerExitTrigger {
    /// Per-scheduler typed tracepoint emitted after a successful claim.
    SchedExtExitTracepoint,
    /// Raw entry/return pair around the preceding generation's exit path.
    ScxVexitKprobePair,
    /// Global-era error dump path used by 6.14.
    ScxDumpState,
}

impl SchedulerExitTrigger {
    fn diagnostic_type(self) -> &'static str {
        match self {
            Self::SchedExtExitTracepoint => "tp_btf",
            Self::ScxVexitKprobePair => "kprobe+kretprobe",
            Self::ScxDumpState => "fentry-global",
        }
    }

    fn section(self) -> &'static str {
        match self {
            Self::SchedExtExitTracepoint => "tp_btf/sched_ext_exit",
            Self::ScxVexitKprobePair => "kprobe+kretprobe/scx_vexit",
            Self::ScxDumpState => "fentry/scx_dump_state",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SchedulerExitProgramSelection {
    tracepoint: bool,
    vexit_enter: bool,
    vexit_return: bool,
    dump_fentry: bool,
}

/// Return the complete pre-load autoload selection for a scheduler-exit
/// trigger. Keeping the raw scx_vexit entry and return halves in one
/// projection prevents a future selector change from loading only half of the
/// correlation pair.
fn scheduler_exit_program_selection(
    trigger: SchedulerExitTrigger,
) -> SchedulerExitProgramSelection {
    SchedulerExitProgramSelection {
        tracepoint: trigger == SchedulerExitTrigger::SchedExtExitTracepoint,
        vexit_enter: trigger == SchedulerExitTrigger::ScxVexitKprobePair,
        vexit_return: trigger == SchedulerExitTrigger::ScxVexitKprobePair,
        dump_fentry: trigger == SchedulerExitTrigger::ScxDumpState,
    }
}

/// Prefer the exact post-claim tracepoint whenever it exists. It has a stable
/// two-argument ABI while `scx_vexit`'s trailing by-value `va_list` has an
/// architecture- and compiler-dependent BTF representation. The raw
/// `scx_vexit` kprobe/kretprobe pair is therefore selected by function
/// presence when the tracepoint is absent, and the two-argument global-era
/// dump hook is the final fallback.
fn select_scheduler_exit_trigger(
    has_sched_ext_exit_tracepoint: bool,
    has_scx_vexit: bool,
    has_scx_dump_state_2: bool,
) -> Option<SchedulerExitTrigger> {
    if has_sched_ext_exit_tracepoint {
        Some(SchedulerExitTrigger::SchedExtExitTracepoint)
    } else if has_scx_vexit {
        Some(SchedulerExitTrigger::ScxVexitKprobePair)
    } else if has_scx_dump_state_2 {
        Some(SchedulerExitTrigger::ScxDumpState)
    } else {
        None
    }
}

fn btf_has_function(btf: &btf_rs::Btf, name: &str) -> bool {
    btf.resolve_types_by_name(name)
        .is_ok_and(|types| types.iter().any(|ty| matches!(ty, btf_rs::Type::Func(_))))
}

fn btf_has_function_arity(btf: &btf_rs::Btf, name: &str, arity: usize) -> bool {
    btf.resolve_types_by_name(name).is_ok_and(|types| {
        types.iter().any(|ty| {
            let btf_rs::Type::Func(func) = ty else {
                return false;
            };
            matches!(
                btf.resolve_chained_type(func),
                Ok(btf_rs::Type::FuncProto(proto)) if proto.parameters.len() == arity
            )
        })
    })
}

/// `tp_btf/<name>` attachment is keyed by the kernel-generated
/// `btf_trace_<name>` typedef, not the ordinary tracefs event record.
///
/// A typed tracepoint typedef chains to a function pointer. Its prototype has
/// one hidden tracepoint context parameter followed by the explicit event
/// arguments; `sched_ext_exit(sch, kind)` therefore has arity three in BTF.
/// Validate the whole chain so a same-named, non-attachable typedef cannot
/// select a program that libbpf will reject at load time.
fn btf_has_typed_tracepoint_arity(btf: &btf_rs::Btf, name: &str, arity: usize) -> bool {
    let target = format!("btf_trace_{name}");
    btf.resolve_types_by_name(&target).is_ok_and(|types| {
        types.iter().any(|ty| {
            let btf_rs::Type::Typedef(typedef) = ty else {
                return false;
            };
            let Ok(btf_rs::Type::Ptr(ptr)) = btf.resolve_chained_type(typedef) else {
                return false;
            };
            matches!(
                btf.resolve_chained_type(&ptr),
                Ok(btf_rs::Type::FuncProto(proto)) if proto.parameters.len() == arity
            )
        })
    })
}

/// Override the cross-thread exit-state mirror. The probe poll thread
/// is the production writer (it mirrors the BSS `ktstr_err_exit_detected`
/// latch into [`PROBE_SCHED_EXIT_STATE`]); this lets host-side tests,
/// where no BPF probe runs, simulate an observed crash latch. Pair a
/// `Crashed` set with a reset to [`SchedExitKind::Unknown`] so neighbor
/// tests start from a clean mirror.
#[cfg(test)]
pub(crate) fn set_probe_sched_exit_state(kind: SchedExitKind) {
    let encoded = match kind {
        SchedExitKind::Unknown => 0,
        SchedExitKind::Clean => PROBE_EXIT_STATE_CLEAN,
        SchedExitKind::Crashed => PROBE_EXIT_STATE_CRASHED,
    };
    PROBE_SCHED_EXIT_STATE.store(encoded, Ordering::Release);
}

/// Input for Phase B probe attachment (BPF fentry/fexit).
///
/// Sent via mpsc channel after the scheduler starts and BPF programs
/// are discoverable. Phase A (kprobes + trigger + kernel fexit) runs
/// before the scheduler; Phase B attaches fentry/fexit to the
/// scheduler's BPF struct_ops callbacks.
pub struct PhaseBInput {
    /// BPF functions and kernel callers discovered from the running
    /// scheduler. Includes both BPF callbacks (for fentry) and their
    /// kernel-side callers from `expand_bpf_to_kernel_callers` (for
    /// additional kprobes).
    pub functions: Vec<StackFunction>,
    /// Pre-opened BPF program FDs keyed by prog_id.
    pub bpf_prog_fds: std::collections::HashMap<u32, i32>,
    /// BTF-resolved function signatures for BPF callbacks and kernel callers.
    pub btf_funcs: Vec<BtfFunc>,
    /// Signaled by the probe worker thread once Phase B attachment
    /// completes so the dispatch path can proceed past its wait.
    pub done: Arc<Latch>,
    /// Starting func_idx for Phase B functions. Must equal the number
    /// of functions in Phase A to avoid index collisions in the shared
    /// `func_meta_map` and `probe_data` maps.
    pub func_idx_offset: u32,
}

/// Ring buffer event type for the trigger (matches `EVENT_TRIGGER`
/// in `intf.h`). Currently the only record type emitted on the
/// `ktstr_events` ringbuf — `EVENT_SCX_EVENT` was removed alongside
/// the `tp_btf/sched_ext_event` BPF handler.
const EVENT_TRIGGER: u32 = 2;

/// Maximum string length carried in a probe_event entry (matches
/// `MAX_STR_LEN` in `intf.h`). Used to size the `RbEvent.str_val`
/// field for byte-level wire compatibility with `struct probe_event`
/// in `intf.h`; the Rust dispatch path leaves it zeroed because
/// the only producer that populated it (the removed
/// `tp_btf/sched_ext_event` handler) is gone.
const MAX_STR_LEN: usize = 64;

/// Pipeline diagnostics from a probe run.
///
/// Tracks how many functions/events survived each stage so users can
/// see WHERE data is being lost (filter, attach, capture, stitch).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProbeDiagnostics {
    /// Kernel functions resolved to IPs.
    pub kprobe_resolved: u32,
    /// Kernel functions that failed IP resolution.
    pub kprobe_resolve_failed: Vec<String>,
    /// Kprobes successfully attached.
    pub kprobe_attached: u32,
    /// Kprobes that failed to attach (name, error).
    pub kprobe_attach_failed: Vec<(String, String)>,
    /// BPF functions with valid prog IDs for fentry.
    pub fentry_candidates: u32,
    /// Fentry probes successfully attached.
    pub fentry_attached: u32,
    /// Fentry probes that failed (name, error).
    pub fentry_attach_failed: Vec<(String, String)>,
    /// Total keys in probe_data map at readout.
    pub probe_data_keys: u32,
    /// Keys with unmatched IPs (no func_meta entry).
    pub probe_data_unmatched_ips: u32,
    /// Events read from probe_data before stitching.
    pub events_before_stitch: u32,
    /// Events surviving tptr+time stitching.
    pub events_after_stitch: u32,
    /// Whether the trigger fired.
    pub trigger_fired: bool,
    /// Which trigger mechanism attached ("tp_btf", "fexit", or
    /// "fentry-global").
    pub trigger_type: String,
    /// Error from the selected scheduler-exit hook.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_attach_error: Option<String>,
    /// Panic payload from the guest-side probe-collection thread
    /// when its `JoinHandle::join()` returned `Err`. `None` on a
    /// clean run (thread exited normally — events may still be
    /// empty if the trigger never fired). `Some(payload)`
    /// distinguishes "the probe thread crashed before producing
    /// events" from "the probe thread ran cleanly and observed no
    /// trigger" — the COM2 payload's `events: []` is otherwise
    /// indistinguishable between those two cases. Any consumer of
    /// the payload (host harness, render layer, downstream test
    /// verdict) MUST treat `Some(_)` as a failure even when
    /// `trigger.fired == false` and `events` is empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_thread_panic: Option<String>,
    /// BPF-side kprobe fire count (cross-CPU sum of the
    /// `KTSTR_PCPU_PROBE_COUNT` slot in `ktstr_pcpu_counters`).
    pub bpf_kprobe_fires: u64,
    /// BPF-side kprobe commit count (cross-CPU sum of the
    /// `KTSTR_PCPU_KPROBE_RETURNS` slot).
    /// `bpf_kprobe_fires - bpf_kprobe_returns` is the number of
    /// kprobe fires that bailed before pushing into `probe_data`
    /// (meta-map miss or scratch-slot miss).
    pub bpf_kprobe_returns: u64,
    /// BPF-side trigger fire count (cross-CPU sum of the
    /// `KTSTR_PCPU_TRIGGER_COUNT` slot).
    pub bpf_trigger_fires: u64,
    /// BPF-side func_meta_map misses (cross-CPU sum of the
    /// `KTSTR_PCPU_META_MISS` slot — IP not found in map).
    pub bpf_meta_misses: u64,
    /// IPs that missed func_meta_map lookup (from BSS ktstr_miss_log).
    pub bpf_miss_ips: Vec<u64>,
    /// BPF-side `bpf_ringbuf_reserve` failures inside the trigger
    /// handler (cross-CPU sum of `KTSTR_PCPU_RINGBUF_DROPS`).
    /// Non-zero means the userspace consumer fell behind on the
    /// events ringbuf, so auto-repro will see a missing trigger
    /// event even though the scheduler did fire.
    pub bpf_ringbuf_drops: u64,
    /// Nanosecond timestamp captured by the BPF trigger handler on
    /// the first accepted error-class scheduler exit (from BSS
    /// ktstr_last_trigger_ts). 0 when no error-class exit fired.
    pub bpf_first_trigger_ns: u64,
    /// `kind` argument captured by the BPF trigger handler on the
    /// first accepted error-class scheduler exit (from BSS
    /// `ktstr_exit_kind_snap`). 0 when no error-class exit fired,
    /// otherwise one of the [`SCX_EXIT_*`](super::scx_defs) values.
    /// Used by the host renderer to disambiguate "trigger fired with
    /// kind=STALL/ERROR (no causal task; events suppressed)" from
    /// "trigger never fired" when the post-stitch event count is 0.
    pub bpf_exit_kind_snap: u32,
    /// `true` when the readout phase reached the no-causal-tptr
    /// branch and emitted events grouped by frequency rather than
    /// stitched against a real trigger task pointer. Set in
    /// [`run_probe_skeleton`] when the trigger fired with
    /// `args[0] == 0` (kind=STALL or generic ERROR) or never fired
    /// at all but at least one captured kprobe event had a non-zero
    /// task pointer. Surfaced by the host renderer as
    /// `events: ... — trigger absent, grouped by frequency` so the
    /// operator does not misread the candidate chain as a verified
    /// stitch.
    pub stitch_fallback_used: bool,
    /// Cumulative count of `tp_btf/sched_switch +
    /// sched_migrate_task + sched_wakeup` records committed into
    /// the dedicated `timeline_events` ringbuf by the timeline
    /// handlers (cross-CPU sum of the
    /// `KTSTR_PCPU_TIMELINE_COUNT` slot). Zero before any of
    /// those tracepoints fire; otherwise grows continuously
    /// while the probe runs. Combined with `bpf_timeline_drops`
    /// it lets an operator tell whether a failure-time drain
    /// saw the full window or only the tail.
    pub bpf_timeline_count: u64,
    /// `bpf_ringbuf_reserve` failures across the three timeline
    /// tracepoint handlers (sched_switch / sched_migrate_task /
    /// sched_wakeup), aggregated as cross-CPU sum of the
    /// `KTSTR_PCPU_TIMELINE_DROPS` slot. Each drop is one new
    /// event lost — the ring's existing contents are NOT evicted
    /// on overflow, so the drain on failure recovers the OLDEST
    /// captured events first.
    pub bpf_timeline_drops: u64,
}

/// Structured probe event captured by the BPF skeleton.
///
/// One per (function, task_ptr) combination. `fields` contains BTF-resolved
/// struct field values keyed as `"param:struct.field"` (from
/// [`build_field_keys`]). Events are sorted by `ts` and stitched by
/// `task_struct` pointer before output.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProbeEvent {
    /// Index into the run's function list (matches the slot used in
    /// the shared `func_meta_map`).
    pub func_idx: u32,
    /// `task_struct` pointer for the thread that triggered the probe;
    /// used to stitch entry/exit pairs.
    pub task_ptr: u64,
    /// Nanosecond timestamp captured at entry.
    pub ts: u64,
    /// First six callee arguments captured verbatim from the call
    /// site (after convention lowering).
    pub args: [u64; 6],
    /// BTF-resolved struct field values decoded post-hoc by the
    /// caller; each entry is `(field_key, raw_value)`.
    pub fields: Vec<(String, u64)>,
    /// Kernel stack trace captured at entry, most recent frame first.
    pub kstack: Vec<u64>,
    /// Optional UTF-8 string associated with the event (e.g. a
    /// scheduler-provided exit reason).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub str_val: Option<String>,
    /// Post-mutation field values captured by fexit.
    /// Same field keys as `fields`, paired by index.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exit_fields: Vec<(String, u64)>,
    /// Timestamp when fexit fired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_ts: Option<u64>,
}

/// Parse `/proc/kallsyms` into a `name -> address` map. Returns `None`
/// when the file is unreadable (expected outside a privileged context)
/// OR when every parsed entry has a zero address (see
/// [`accept_kallsyms_map`] for the kptr_restrict rationale).
fn load_kallsyms() -> Option<std::collections::HashMap<String, u64>> {
    let raw = std::fs::read_to_string("/proc/kallsyms").ok()?;
    accept_kallsyms_map(parse_kallsyms(&raw))
}

/// Return `Some(map)` when at least one entry has a non-zero address,
/// otherwise `None`. The all-zero case is what the kernel emits under
/// `kernel.kptr_restrict=2` for non-CAP_SYSLOG callers — the file is
/// readable, all symbol names are present, but every line carries
/// `0000000000000000` for its address. Caching such a map would
/// poison every later [`resolve_func_ip`] lookup with `Some(0)`,
/// masking the unprivileged state from the retry-after-sudo path;
/// treating it as a load failure instead lets the next caller (after
/// [`RETRY_MIN_INTERVAL`]) try again under the new privilege level.
fn accept_kallsyms_map(
    map: std::collections::HashMap<String, u64>,
) -> Option<std::collections::HashMap<String, u64>> {
    if !map.values().any(|&a| a != 0) {
        tracing::warn!(
            entries = map.len(),
            "/proc/kallsyms parsed with zero addresses only — kptr_restrict \
             likely active; declining to cache",
        );
        return None;
    }
    Some(map)
}

/// Parse kallsyms-format text (one `HEX TYPE NAME ...` line per
/// symbol) into a `name -> address` map. Extracted from
/// [`load_kallsyms`] so unit tests can exercise the parser without
/// touching `/proc/kallsyms`, which is usually unreadable in the
/// unprivileged contexts the crate runs under.
///
/// Skipped lines (silently, without affecting other symbols):
/// - lines with fewer than 3 whitespace-separated tokens (addr,
///   type, name — all three are required; the type column is
///   accepted but ignored)
/// - lines whose first token is not a hex-parseable `u64`
///
/// A permanently-empty map is a valid return value — callers treat
/// it as "no symbols found" rather than an error.
fn parse_kallsyms(raw: &str) -> std::collections::HashMap<String, u64> {
    // No pre-scan — HashMap grows from empty in a single pass over
    // `raw`.
    let mut map = std::collections::HashMap::new();
    for line in raw.lines() {
        let mut parts = line.split_whitespace();
        let Some(addr) = parts.next() else { continue };
        let _ty = parts.next();
        let Some(sym) = parts.next() else { continue };
        let Ok(addr) = u64::from_str_radix(addr, 16) else {
            continue;
        };
        map.insert(sym.to_string(), addr);
    }
    map
}

/// Resolve a kernel function name to its address via /proc/kallsyms.
///
/// The parsed `name -> address` map is cached on first successful load
/// so later lookups avoid re-reading and re-splitting ~200k lines.
/// Callers that resolve many functions in a batch (auto-probe attach,
/// probe-stack load) drop from O(N\*M) line scans to O(N) hash lookups.
///
/// A failed load (unreadable `/proc/kallsyms` — typical for
/// unprivileged processes where the file is either missing or
/// returns zeroed addresses) is rate-limited to one retry per
/// `RETRY_MIN_INTERVAL` (1 s); calls within that window return
/// `None` immediately. This matters both for performance — a caller
/// resolving N symbols under a permanently unreadable
/// `/proc/kallsyms` pays one load attempt, not N — and for
/// privilege-escalation correctness, where a test harness that
/// re-execs under `sudo` after the first miss still sees a retry
/// within seconds.
pub fn resolve_func_ip(name: &str) -> Option<u64> {
    use std::sync::{OnceLock, RwLock};
    use std::time::Instant;
    static CACHE: OnceLock<RwLock<Option<std::collections::HashMap<String, u64>>>> =
        OnceLock::new();
    // Rate-limit the load retry on chronic failure. Without a
    // floor, a caller that resolves N symbols under a permanently
    // unreadable /proc/kallsyms triggers N * (read + parse) of a
    // ~10 MB file. The floor turns that into one retry per window
    // and returns `None` fast in between.
    static LAST_LOAD_ATTEMPT: OnceLock<RwLock<Option<Instant>>> = OnceLock::new();

    let slot = CACHE.get_or_init(|| RwLock::new(None));
    // Fast path: take the read lock when the cache is populated.
    // Post-load the lookup is read-only and batches resolving many
    // symbols contend only on the shared read lock.
    {
        let read = slot.read_unpoisoned();
        if let Some(map) = read.as_ref() {
            return map.get(name).copied();
        }
    }
    // Optional fast-decline: when the retry clock rules the caller
    // out, avoid the write-lock acquire entirely. This is a
    // performance hint only — correctness is enforced by the
    // re-check below under the write lock, so a concurrent racer
    // that slips past this gate still gets serialized.
    let last_slot = LAST_LOAD_ATTEMPT.get_or_init(|| RwLock::new(None));
    {
        let last = last_slot.read_unpoisoned();
        if let Some(t) = *last
            && t.elapsed() < RETRY_MIN_INTERVAL
        {
            return None;
        }
    }
    // Slow path: escalate to write lock to populate. Re-check both
    // the cache and the retry clock under the write lock so N
    // concurrent first-callers don't stampede into N serialized
    // loads: only the winner gets past the timestamp gate, everyone
    // else observes `*last = Some(now)` and bails.
    let mut write = slot.write_unpoisoned();
    if write.is_none() {
        let mut last = last_slot.write_unpoisoned();
        if last.is_none_or(|t| t.elapsed() >= RETRY_MIN_INTERVAL) {
            *write = load_kallsyms();
            *last = Some(Instant::now());
        }
    }
    write.as_ref()?.get(name).copied()
}

/// Minimum interval between retry attempts when `/proc/kallsyms` is
/// unreadable; see [`resolve_func_ip`] for the rationale.
const RETRY_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Build a `func_idx -> task_struct_param_idx` map for stitching by
/// task pointer. Resolves each function's task-struct argument
/// position from [`super::stack::BPF_OP_CALLERS`] first, falling back
/// to the BTF param list (Phase A `btf_funcs` chained with Phase B
/// `phase_b_btf`).
///
/// Entries with `pidx >= 6` are dropped with a warn rather than
/// stored: the stitch code reads `ProbeEvent::args[pidx]` against a
/// fixed-size `[u64; 6]` (matching the BPF-side capture limit), so
/// any larger index would panic. A function with its task_struct
/// past arg-6 simply cannot be stitched here — the BPF probe never
/// captured that arg.
fn build_task_param_idx(
    func_ips: &[(u32, u64, String)],
    btf_funcs: &[BtfFunc],
    phase_b_btf: &[BtfFunc],
) -> std::collections::HashMap<u32, usize> {
    func_ips
        .iter()
        .filter_map(|(idx, _, name)| {
            // BPF_OP_CALLERS: (op_fragment, kernel_caller, task_arg_idx)
            let pidx = if let Some((_, _, tidx)) = super::stack::BPF_OP_CALLERS
                .iter()
                .find(|(_, caller, _)| *caller == name.as_str())
            {
                *tidx as usize
            } else {
                // Fallback: BTF params with task_struct
                let btf = btf_funcs
                    .iter()
                    .chain(phase_b_btf.iter())
                    .find(|f| f.name == *name)?;
                btf.params
                    .iter()
                    .position(|p| p.struct_name.as_deref() == Some("task_struct"))?
            };
            if pidx >= 6 {
                tracing::warn!(
                    func = %name,
                    pidx,
                    "task_struct param index out of args[6] bounds — \
                     skipping stitch entry",
                );
                return None;
            }
            Some((*idx, pidx))
        })
        .collect()
}

/// Populate a `func_meta` with field specs from BTF-resolved offsets.
/// Shared between kprobe and fentry paths.
///
/// Invariant: `meta.nr_field_specs = max(field_idx) + 1`, NOT the count
/// of specs. The BPF side writes `entry.fields[field_idx]` positionally,
/// and the Rust side reads `entry.fields[..nr_field_specs]` positionally
/// against [`build_field_keys`] (which includes skipped fields). A
/// reorder of either loop that turns this into a count would silently
/// mismatch keys to values.
fn populate_field_specs(meta: &mut types::func_meta, field_specs: &[super::btf::FieldSpec]) {
    let n = field_specs.len().min(16);
    let max_fidx = field_specs
        .iter()
        .take(n)
        .map(|fs| fs.field_idx)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    meta.nr_field_specs = max_fidx.min(16);
    for fs in field_specs.iter().take(n) {
        let slot = fs.field_idx as usize;
        if slot < 16 {
            meta.specs[slot] = types::field_spec {
                param_idx: fs.param_idx,
                offset: fs.offset,
                size: fs.size,
                field_idx: fs.field_idx,
                ptr_offset: fs.ptr_offset,
            };
        }
    }
}

/// Build field key names for a function based on its BTF info.
///
/// Returns a vec mapping `field_idx` to an output key name. Format:
/// - Known struct param: `"p:task_struct.pid"`
/// - Auto-discovered BPF struct: `"ctx:task_ctx.field_a"`
/// - Scalar param: `"flags:val.flags"`
///
/// Processes at most 6 params (fentry/kprobe register limit) and
/// at most 16 fields total (matching `MAX_FIELDS` in intf.h).
///
/// Invariant: keys are emitted in the same order [`populate_field_specs`]
/// consumes `field_specs`, so `entry.fields[i]` maps to the i-th key.
fn build_field_keys(btf_func: &BtfFunc) -> Vec<(String, RenderHint)> {
    let mut keys = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for param in &btf_func.params[..max_params] {
        if let Some(ref sname) = param.struct_name {
            if let Some((_, fields)) = STRUCT_FIELDS.iter().find(|(s, _)| *s == sname) {
                for (_, key) in *fields {
                    // Known struct fields use dedicated decoders in
                    // decode.rs — hint is irrelevant (Default/Hex).
                    keys.push((format!("{}:{}.{}", param.name, sname, key), RenderHint::Hex));
                    field_idx += 1;
                    if field_idx >= 16 {
                        break;
                    }
                }
            }
        } else if !param.auto_fields.is_empty() {
            let tname = param.type_name.as_deref().unwrap_or("void");
            for (fname, _, hint) in &param.auto_fields {
                keys.push((format!("{}:{}.{}", param.name, tname, fname), *hint));
                field_idx += 1;
                if field_idx >= 16 {
                    break;
                }
            }
        } else if !param.is_ptr {
            keys.push((
                format!("{}:val.{}", param.name, param.name),
                RenderHint::Hex,
            ));
            field_idx += 1;
        }
    }

    keys
}

/// Detect which param (if any) is a char * string.
/// Uses BTF type detection first, then name heuristic as fallback.
/// Returns 0xff if none found.
fn detect_str_param(btf_func: &BtfFunc) -> u8 {
    let max = btf_func.params.len().min(6);
    // BTF-based: check is_string_ptr flag set by parse_btf_functions.
    for (i, p) in btf_func.params[..max].iter().enumerate() {
        if p.is_string_ptr {
            return i as u8;
        }
    }
    // Name heuristic fallback.
    for (i, p) in btf_func.params[..max].iter().enumerate() {
        if !p.is_ptr || p.struct_name.is_some() {
            continue;
        }
        let n = p.name.as_str();
        if matches!(n, "fmt" | "msg" | "str" | "reason" | "buf" | "s")
            || n.contains("str")
            || n.contains("msg")
            || n.contains("fmt")
        {
            return i as u8;
        }
    }
    0xff
}

/// Pre-open BPF program FDs while the scheduler is alive.
///
/// Returns a map from `bpf_prog_id` to owned fd. Holding these FDs
/// keeps the BPF programs alive via kernel refcounting even after the
/// scheduler exits. Must be called before the test function runs
/// (which may crash the scheduler).
pub fn open_bpf_prog_fds(functions: &[StackFunction]) -> std::collections::HashMap<u32, i32> {
    let mut fds = std::collections::HashMap::new();
    for f in functions {
        if let Some(prog_id) = f.bpf_prog_id {
            let fd = unsafe { libbpf_rs::libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
            if fd >= 0 {
                fds.insert(prog_id, fd);
            }
        }
    }
    fds
}

/// `&ProgramMut<'_>` newtype that asserts thread-shared access is
/// safe for [`libbpf_rs::ProgramMut::attach_kprobe`].
///
/// `ProgramMut` holds a `NonNull<bpf_program>` which keeps it
/// `!Sync` at the Rust type-system level. Concurrent attach is
/// nonetheless sound:
/// - `bpf_program__attach_kprobe_opts` (the libbpf C path
///   `attach_kprobe` calls) takes `const struct bpf_program *prog` —
///   it does not mutate the program through that pointer.
/// - Each attach call creates an independent `perf_event_open` fd
///   plus an independent `BPF_LINK_CREATE` syscall; those resources
///   are not shared with the program object.
/// - The only program-adjacent state the path touches is the global
///   feature-detection cache via `kernel_supports` →
///   `feat_supported`, which uses `READ_ONCE` / `WRITE_ONCE` and is
///   idempotent across racing readers (see
///   `libbpf-sys/libbpf/src/features.c::feat_supported`).
///   All threads call only `attach_kprobe` on the inner reference; no
///   mutating method is invoked here.
///
/// `Send` is paired with `Sync` because `std::thread::scope` requires
/// the captured reference's referent to be `Sync` for `&_` to be
/// `Send`.
struct AttachableProgRef<'a, 'obj> {
    inner: &'a libbpf_rs::ProgramMut<'obj>,
}
// SAFETY: see the type-level doc.
unsafe impl Send for AttachableProgRef<'_, '_> {}
// SAFETY: see the type-level doc.
unsafe impl Sync for AttachableProgRef<'_, '_> {}

/// Attach `prog` as a kprobe to every name in `func_names`,
/// parallelising across worker threads.
///
/// Each attach is an independent `perf_event_open` + `BPF_LINK_CREATE`
/// syscall pair. Sequentialised, the per-attach syscall round-trip
/// dominates the auto-repro setup phase when the crash backtrace
/// names many kernel functions. Spawning a small worker pool lets
/// the kernel run the attaches concurrently and shrinks the total
/// wall-clock cost of the loop to roughly `total / parallelism`
/// (bounded by kernel-side serialisation inside `perf_event_open` for
/// kprobe registration).
///
/// Worker count is `min(func_names.len(), 8)` — a fixed cap that
/// matches the typical small ktstr backtrace width while avoiding
/// thread-spawn overhead when there are only a handful of probes.
/// Returns one `(name, Result<Link, libbpf_rs::Error>)` entry per
/// input in input order, so the caller can populate
/// [`ProbeDiagnostics::kprobe_attach_failed`] / `links` with the
/// same shape as the prior sequential loop.
fn parallel_attach_kprobes<'a, 'obj>(
    prog: &'a libbpf_rs::ProgramMut<'obj>,
    func_names: &[String],
) -> Vec<(String, libbpf_rs::Result<libbpf_rs::Link>)> {
    if func_names.is_empty() {
        return Vec::new();
    }

    let prog_ref = AttachableProgRef { inner: prog };
    // `min(N, 8)` balances syscall parallelism against thread-spawn
    // overhead. Empirically the kernel kprobe registration path
    // serialises on `kprobe_mutex` so going wider yields diminishing
    // returns; the cap keeps us comfortably under that wall.
    const MAX_WORKERS: usize = 8;
    let workers = func_names.len().min(MAX_WORKERS);

    // Slot the function names into round-robin per-worker buckets
    // so each thread owns a disjoint subset. Index-tagged so the
    // output preserves input order regardless of which worker
    // finishes first.
    let mut buckets: Vec<Vec<(usize, String)>> = (0..workers).map(|_| Vec::new()).collect();
    for (i, name) in func_names.iter().enumerate() {
        buckets[i % workers].push((i, name.clone()));
    }

    let mut results: Vec<Option<(String, libbpf_rs::Result<libbpf_rs::Link>)>> =
        (0..func_names.len()).map(|_| None).collect();

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for bucket in buckets {
            if bucket.is_empty() {
                continue;
            }
            let prog_ref = &prog_ref;
            handles.push(s.spawn(move || {
                let mut out: Vec<(usize, String, libbpf_rs::Result<libbpf_rs::Link>)> =
                    Vec::with_capacity(bucket.len());
                for (i, name) in bucket {
                    let r = prog_ref.inner.attach_kprobe(false, &name);
                    out.push((i, name, r));
                }
                out
            }));
        }
        for h in handles {
            // Worker panics propagate as the join Err — re-panic on
            // the main thread to surface bugs rather than silently
            // dropping attach results.
            let out = h.join().expect("kprobe attach worker panicked");
            for (i, name, r) in out {
                results[i] = Some((name, r));
            }
        }
    });

    results
        .into_iter()
        .map(|o| o.expect("every input slot must be filled by exactly one worker"))
        .collect()
}

/// Attach the fentry program in slot 0..=3 on the fentry skeleton.
///
/// The fentry skeleton exposes four indexed programs
/// (`ktstr_fentry_0`..`ktstr_fentry_3`) matching the 4-slot batch
/// model in `src/bpf/fentry_probe.bpf.c`. Call sites previously spelled the
/// full 4-arm `match slot { ... }` inline; routing through this
/// family of helpers keeps the dispatch in one place so a future
/// slot addition is a one-line change per helper instead of
/// scattered across every batch.
///
/// Returns `None` for slot indices outside 0..=3, matching the
/// existing `continue;` behaviour at call sites.
fn attach_fentry_by_slot(
    skel: &crate::bpf_skel::fentry::FentryProbeSkel<'_>,
    slot: usize,
) -> Option<libbpf_rs::Result<libbpf_rs::Link>> {
    Some(match slot {
        0 => skel.progs.ktstr_fentry_0.attach_trace(),
        1 => skel.progs.ktstr_fentry_1.attach_trace(),
        2 => skel.progs.ktstr_fentry_2.attach_trace(),
        3 => skel.progs.ktstr_fentry_3.attach_trace(),
        _ => return None,
    })
}

/// Attach the fexit program in slot 0..=3 on the fentry skeleton.
/// Sibling of [`attach_fentry_by_slot`]; see its doc for the routing
/// rationale.
fn attach_fexit_by_slot(
    skel: &crate::bpf_skel::fentry::FentryProbeSkel<'_>,
    slot: usize,
) -> Option<libbpf_rs::Result<libbpf_rs::Link>> {
    Some(match slot {
        0 => skel.progs.ktstr_fexit_0.attach_trace(),
        1 => skel.progs.ktstr_fexit_1.attach_trace(),
        2 => skel.progs.ktstr_fexit_2.attach_trace(),
        3 => skel.progs.ktstr_fexit_3.attach_trace(),
        _ => return None,
    })
}

/// Borrow the open fentry program in slot 0..=3 for pre-load
/// configuration (`set_attach_target`, `set_autoload`).
///
/// Pre-load sibling of [`attach_fentry_by_slot`]: operates on
/// [`OpenFentryProbeSkel`] before [`OpenSkel::load`] consumes it.
/// Returns `None` for slot indices outside 0..=3, matching the
/// existing `continue;` behaviour at call sites.
///
/// [`OpenFentryProbeSkel`]: crate::bpf_skel::fentry::OpenFentryProbeSkel
/// [`OpenSkel::load`]: libbpf_rs::skel::OpenSkel::load
fn fentry_prog_mut_by_slot<'a, 'obj>(
    open_skel: &'a mut crate::bpf_skel::fentry::OpenFentryProbeSkel<'obj>,
    slot: usize,
) -> Option<&'a mut libbpf_rs::OpenProgramMut<'obj>> {
    Some(match slot {
        0 => &mut open_skel.progs.ktstr_fentry_0,
        1 => &mut open_skel.progs.ktstr_fentry_1,
        2 => &mut open_skel.progs.ktstr_fentry_2,
        3 => &mut open_skel.progs.ktstr_fentry_3,
        _ => return None,
    })
}

/// Borrow the open fexit program in slot 0..=3 for pre-load
/// configuration. Pre-load sibling of [`attach_fexit_by_slot`];
/// see [`fentry_prog_mut_by_slot`] for the routing rationale.
fn fexit_prog_mut_by_slot<'a, 'obj>(
    open_skel: &'a mut crate::bpf_skel::fentry::OpenFentryProbeSkel<'obj>,
    slot: usize,
) -> Option<&'a mut libbpf_rs::OpenProgramMut<'obj>> {
    Some(match slot {
        0 => &mut open_skel.progs.ktstr_fexit_0,
        1 => &mut open_skel.progs.ktstr_fexit_1,
        2 => &mut open_skel.progs.ktstr_fexit_2,
        3 => &mut open_skel.progs.ktstr_fexit_3,
        _ => return None,
    })
}

/// Disable autoload on both the fentry and fexit programs for
/// slot 0..=3 so the verifier skips them at
/// [`OpenSkel::load`][libbpf_rs::skel::OpenSkel::load]. Used for
/// unused batch slots and slots whose
/// [`set_attach_target`][libbpf_rs::OpenProgramMut::set_attach_target]
/// call failed — either leaves the skeleton with placeholder
/// targets the verifier would reject.
///
/// No-op for slot indices outside 0..=3.
fn disable_slot_programs(
    open_skel: &mut crate::bpf_skel::fentry::OpenFentryProbeSkel<'_>,
    slot: usize,
) {
    if let Some(p) = fentry_prog_mut_by_slot(open_skel, slot) {
        p.set_autoload(false);
    }
    if let Some(p) = fexit_prog_mut_by_slot(open_skel, slot) {
        p.set_autoload(false);
    }
}

/// Write the per-slot rodata fields (`ktstr_fentry_func_idx_N`,
/// `ktstr_fentry_is_kernel_N`) for slot 0..=3. Mirrors the BPF
/// side's positional `rodata` layout in `src/bpf/fentry_probe.bpf.c`.
///
/// No-op for slot indices outside 0..=3.
fn set_rodata_slot(
    rodata: &mut crate::bpf_skel::fentry::types::rodata,
    slot: usize,
    idx: u32,
    is_kernel: bool,
) {
    let k = is_kernel as u8;
    match slot {
        0 => {
            rodata.ktstr_fentry_func_idx_0 = idx;
            rodata.ktstr_fentry_is_kernel_0 = k;
        }
        1 => {
            rodata.ktstr_fentry_func_idx_1 = idx;
            rodata.ktstr_fentry_is_kernel_1 = k;
        }
        2 => {
            rodata.ktstr_fentry_func_idx_2 = idx;
            rodata.ktstr_fentry_is_kernel_2 = k;
        }
        3 => {
            rodata.ktstr_fentry_func_idx_3 = idx;
            rodata.ktstr_fentry_is_kernel_3 = k;
        }
        _ => {}
    }
}

/// Bounded retry count for the kernel-function fexit batch load.
///
/// On a heavily contended host the fexit `BPF_PROG_LOAD` can be
/// rejected with a transient error even for a privileged loader whose
/// same targets load cleanly on a quiet host — observed as an
/// intermittent "Permission denied (os error 13)" on the busiest
/// arm64 CI lanes, which dropped the exit-side counters and left the
/// failure dump a placeholder. os error 13 is EACCES (not EPERM):
/// the errno `BPF_PROG_LOAD` uses for a verifier rejection, so the
/// failure can also be deterministic for a given kernel/arch — on the
/// arm64 7.1 CI kernel the `enqueue_task_scx` fexit load fails all
/// three attempts in every observed run. Retrying the open+load a few
/// times with a short backoff — a cost bounded well inside the
/// probe's boot budget — recovers the transient case and costs ~200ms
/// in the deterministic one. `set_attach_target` is deterministic, so
/// a chunk with no valid target is never retried.
const KERNEL_FEXIT_LOAD_ATTEMPTS: u32 = 3;

// More than one attempt is the whole point: it turns a transient load
// EACCES into a recovered load rather than an immediate placeholder dump.
const _: () = assert!(
    KERNEL_FEXIT_LOAD_ATTEMPTS >= 2,
    "the kernel-fexit batch load must retry at least once"
);

/// Base backoff between kernel-fexit load attempts, scaled by the
/// zero-based attempt number (see [`kernel_fexit_load_backoff`]).
const KERNEL_FEXIT_LOAD_BACKOFF: std::time::Duration = std::time::Duration::from_millis(100);

/// Backoff before load `attempt` (zero-based): no wait before the
/// first try, then a linear ramp (0, base, 2·base, …). Factored out so
/// the schedule is unit-testable without loading BPF.
fn kernel_fexit_load_backoff(attempt: u32) -> std::time::Duration {
    KERNEL_FEXIT_LOAD_BACKOFF * attempt
}

/// Open, configure, load, and attach one batch (`chunk`) of
/// kernel-function fexit targets against vmlinux BTF (fd 0), retrying
/// the load [`KERNEL_FEXIT_LOAD_ATTEMPTS`] times on a transient
/// failure.
///
/// `chunk` is `(func_idx, func_name)` pairs, at most `batch` long;
/// `batch` is the skeleton's fexit slot count. `probe_data_fd` and
/// `func_meta_fd` are the main skeleton's maps, reused so the fexit
/// programs write into the shared maps. Attached links are pushed onto
/// `fexit_links`; `label` tags the log lines with the caller's phase.
///
/// Returns `Ok(())` once a load succeeds (individual attach failures
/// are logged, not fatal) or when the chunk had no valid attach target
/// (a deterministic outcome that is not retried). Returns
/// `Err((error, targeted_names))` when every attempt failed so the
/// caller can surface the error and record the affected functions.
fn load_kernel_fexit_batch(
    chunk: &[(u32, &str)],
    batch: usize,
    probe_data_fd: std::os::unix::io::BorrowedFd<'_>,
    func_meta_fd: std::os::unix::io::BorrowedFd<'_>,
    fexit_links: &mut Vec<libbpf_rs::Link>,
    label: &str,
) -> Result<(), (libbpf_rs::Error, Vec<String>)> {
    use crate::bpf_skel::fentry::*;
    use libbpf_rs::skel::{OpenSkel, SkelBuilder};

    let mut load_err: Option<libbpf_rs::Error> = None;

    for attempt in 0..KERNEL_FEXIT_LOAD_ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(kernel_fexit_load_backoff(attempt));
        }

        // `.load()` consumes the open skeleton, so the whole open must
        // be rebuilt each attempt; the obj storage lives for this
        // iteration only.
        let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
        let mut fentry_open = match FentryProbeSkelBuilder::default().open(&mut fentry_open_obj) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, label, "kernel fexit skeleton open failed");
                load_err = Some(e);
                continue;
            }
        };

        if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
            rodata.ktstr_enabled = true;
            for (slot, &(idx, _)) in chunk.iter().enumerate() {
                set_rodata_slot(rodata, slot, idx, true);
            }
        }

        // Only fexit programs are needed here (entry capture is the
        // kprobe skeleton's job): disable the fentry half and point
        // each fexit program at its kernel function.
        let mut ok = vec![false; chunk.len()];
        for (slot, &(_, name)) in chunk.iter().enumerate() {
            let Some(fentry_prog) = fentry_prog_mut_by_slot(&mut fentry_open, slot) else {
                continue;
            };
            fentry_prog.set_autoload(false);

            let Some(fexit_prog) = fexit_prog_mut_by_slot(&mut fentry_open, slot) else {
                continue;
            };
            match fexit_prog.set_attach_target(0, Some(name.to_string())) {
                Ok(()) => {
                    ok[slot] = true;
                    tracing::debug!(
                        slot,
                        func = name,
                        label,
                        "kernel fexit: set_attach_target ok"
                    );
                }
                Err(e) => {
                    tracing::debug!(slot, func = name, %e, label, "kernel fexit: set_attach_target failed");
                    fexit_prog.set_autoload(false);
                }
            }
        }

        if !ok.iter().any(|&v| v) {
            // No valid attach target in this chunk; retrying cannot
            // change a deterministic set_attach_target outcome.
            return Ok(());
        }

        // Disable both halves of every unused slot. A generic
        // `SEC("fentry")` program without an attach target is not a
        // valid load target on all kernels; leaving an unused entry
        // half enabled can reject the whole short final batch with
        // EINVAL even though every used fexit target is valid.
        for slot in 0..batch {
            if !ok.get(slot).copied().unwrap_or(false) {
                disable_slot_programs(&mut fentry_open, slot);
            }
        }

        if let Err(e) = fentry_open.maps.probe_data.reuse_fd(probe_data_fd) {
            tracing::warn!(%e, label, "kernel fexit: probe_data reuse_fd failed");
        }
        if let Err(e) = fentry_open.maps.func_meta_map.reuse_fd(func_meta_fd) {
            tracing::warn!(%e, label, "kernel fexit: func_meta_map reuse_fd failed");
        }

        let fentry_skel = match fentry_open.load() {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!(%e, attempt, label, "kernel fexit: batch load attempt failed");
                load_err = Some(e);
                continue;
            }
        };

        for (slot, &(_, name)) in chunk.iter().enumerate() {
            if !ok[slot] {
                continue;
            }
            let Some(result) = attach_fexit_by_slot(&fentry_skel, slot) else {
                continue;
            };
            match result {
                Ok(link) => {
                    tracing::debug!(func = name, label, "kernel fexit attached");
                    fexit_links.push(link);
                }
                Err(e) => {
                    tracing::debug!(%e, func = name, label, "kernel fexit attach failed");
                }
            }
        }

        drop(fentry_skel);
        return Ok(());
    }

    let targeted: Vec<String> = chunk.iter().map(|&(_, name)| name.to_string()).collect();
    Err((
        load_err.expect("load_err is Some whenever every load attempt failed"),
        targeted,
    ))
}

/// Causal-task filter for the trigger event's `task_ptr` (sourced
/// from `args[0]`).
///
/// The selected BPF scheduler-exit handler sets `args[0]` to
/// `bpf_get_current_task()` only for `SCX_EXIT_ERROR_BPF` (where
/// `current` IS the causal task); for `SCX_EXIT_ERROR` (kworker /
/// sysrq context) it emits `args[0] == 0`. A zero `task_ptr` means
/// "no causal task" and must suppress probe stitching, so this
/// returns `None` for 0 and `Some(p)` otherwise. `run_probe_skeleton`
/// calls this on the last trigger event; the `args0_*` tests pin both
/// sides of the contract against this exact predicate.
fn causal_tptr(task_ptr: u64) -> Option<u64> {
    Some(task_ptr).filter(|&p| p != 0)
}

/// Run the BPF probe skeleton for auto-repro.
///
/// Operates in two modes depending on `phase_b_rx`:
///
/// **Single-phase (`phase_b_rx = None`):** loads the kprobe skeleton
/// and fentry/fexit skeleton together, attaches all probes, then
/// polls until the trigger fires.
///
/// **Two-phase (`phase_b_rx = Some`):** Phase A attaches kprobes +
/// kernel fexit + the kernel-selected exit trigger before the
/// scheduler starts, signals `ready`, then polls the ring buffer
/// while waiting for Phase B input via the channel. When Phase B
/// input arrives, attaches fentry/fexit to BPF struct_ops callbacks
/// and additional kprobes for kernel callers. Signals `done` on the
/// `PhaseBInput` when Phase B attachment completes. If the trigger
/// fires before Phase B input arrives, fentry is skipped — the
/// crash happened before BPF programs could be probed.
///
/// Newest kernels trigger directly after a successful `scx_claim_exit()` via
/// `tp_btf/sched_ext_exit`; the preceding generation triggers from a raw
/// entry/return pair around `scx_vexit()`; global-era kernels trigger on their
/// filtered two-argument `scx_dump_state()` error entry. If no compatible
/// target is available, auto-repro fails closed before object load.
///
/// Returns accumulated func_names from both phases as the third
/// tuple element.
pub fn run_probe_skeleton(
    functions: &[StackFunction],
    btf_funcs: &[BtfFunc],
    stop: &AtomicBool,
    bpf_prog_fds: &std::collections::HashMap<u32, i32>,
    ready: &Latch,
    phase_a_status: &std::sync::Mutex<Option<Result<(), String>>>,
    phase_b_rx: Option<std::sync::mpsc::Receiver<PhaseBInput>>,
) -> (
    Option<Vec<ProbeEvent>>,
    ProbeDiagnostics,
    Vec<(u32, String)>,
) {
    use crate::bpf_skel::*;
    use libbpf_rs::skel::{OpenSkel, SkelBuilder};
    use libbpf_rs::{Link, MapCore, MapFlags, RingBufferBuilder};

    tracing::debug!(n = functions.len(), "run_probe_skeleton");

    let publish_phase_a = |status: Result<(), String>| {
        let mut slot = phase_a_status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(status);
        }
        drop(slot);
        ready.set();
    };

    let mut diag = ProbeDiagnostics::default();

    // Capture libbpf's own log (verifier rejection reasons, "load
    // failed: -E..." lines) into a buffer so a skeleton-load failure
    // surfaces its ROOT CAUSE through `diag.trigger_attach_error`
    // rather than the bare errno. The guest's stderr is not forwarded
    // to the host, so without this the pipeline reports a one-line
    // "Permission denied" for a verifier rejection whose actual reason
    // (e.g. a bad program prototype) lives only in the swallowed log.
    // `set_print` is process-global; the buffer is cleared at entry so
    // each load attempt starts fresh, and only its tail is read back
    // on failure below.
    static LIBBPF_LOG: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
    fn libbpf_log_capture(_lvl: libbpf_rs::PrintLevel, msg: String) {
        if let Ok(mut b) = LIBBPF_LOG.lock() {
            b.push_str(&msg);
        }
    }
    libbpf_rs::set_print(Some((libbpf_rs::PrintLevel::Debug, libbpf_log_capture)));
    if let Ok(mut b) = LIBBPF_LOG.lock() {
        b.clear();
    }

    // Load vmlinux BTF once and use it for both trigger selection and
    // kprobe metadata below. All trigger programs live in probe.o; every
    // unselected program must have autoload disabled before libbpf resolves
    // its attach target.
    let cached_btf = crate::monitor::btf_offsets::cached_vmlinux_btf();
    let exit_trigger = cached_btf
        .as_deref()
        .and_then(|btf| {
            select_scheduler_exit_trigger(
                btf_has_typed_tracepoint_arity(btf, "sched_ext_exit", 3),
                btf_has_function(btf, "scx_vexit"),
                btf_has_function_arity(btf, "scx_dump_state", 2),
            )
        })
        .ok_or_else(|| {
            "kernel BTF exposes no compatible sched_ext exit target \
             (`sched_ext_exit`, `scx_vexit`, or two-argument \
             `scx_dump_state`)"
                .to_string()
        });
    let exit_trigger = match exit_trigger {
        Ok(trigger) => trigger,
        Err(error) => {
            diag.trigger_attach_error = Some(error.clone());
            publish_phase_a(Err(error));
            return (None, diag, Vec::new());
        }
    };

    // Open skeleton. Two MaybeUninit slots: the first backs the
    // initial load attempt; the second backs the fallback retry when
    // optional programs cause ESRCH. Both must outlive `skel`.
    let mut open_object = std::mem::MaybeUninit::uninit();
    let mut open_object_fallback = std::mem::MaybeUninit::uninit();
    let builder = ProbeSkelBuilder::default();
    let mut open_skel = match builder.open(&mut open_object) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(%e, "probe skeleton open failed");
            diag.trigger_attach_error = Some(format!("skeleton open: {e}"));
            publish_phase_a(Err(format!("probe skeleton open failed: {e}")));
            return (None, diag, Vec::new());
        }
    };

    // Enable probes (must set before load — rodata is immutable after)
    if let Some(rodata) = open_skel.maps.rodata_data.as_mut() {
        rodata.ktstr_enabled = true;
    }
    let trigger_programs = scheduler_exit_program_selection(exit_trigger);
    open_skel
        .progs
        .ktstr_trigger_tp
        .set_autoload(trigger_programs.tracepoint);
    open_skel
        .progs
        .ktstr_trigger_vexit_enter
        .set_autoload(trigger_programs.vexit_enter);
    open_skel
        .progs
        .ktstr_trigger_vexit_return
        .set_autoload(trigger_programs.vexit_return);
    open_skel
        .progs
        .ktstr_trigger_dump_fentry
        .set_autoload(trigger_programs.dump_fentry);

    // Load skeleton. Try with all programs first; if a missing optional
    // typed-BTF target causes ESRCH, re-open with optional programs disabled.
    // The fallback unconditionally fires on any load error, not
    // strictly ESRCH — libbpf doesn't surface a stable errno through
    // its Error type, so an exact ESRCH match is brittle. The retry
    // is cheap (re-open + load with autoload disabled on the optional
    // set), so the broader gate is acceptable. When the retry ALSO
    // fails, we surface BOTH errors so the operator sees the original
    // root cause alongside the retry failure — a verifier rejection
    // on a non-optional program would otherwise be masked by the
    // retry's unrelated error.
    let (skel, optional_programs_loaded) = match open_skel.load() {
        Ok(s) => (s, true),
        Err(first_err) => {
            tracing::debug!(
                %first_err,
                "probe skeleton load failed with all programs; \
                 retrying with optional programs disabled"
            );
            let builder2 = ProbeSkelBuilder::default();
            let mut open_skel2 = match builder2.open(&mut open_object_fallback) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(%e, "probe skeleton re-open failed");
                    diag.trigger_attach_error = Some(format!(
                        "skeleton open (retry): {e}; original load error: {first_err}"
                    ));
                    publish_phase_a(Err(format!(
                        "probe skeleton re-open failed: {e}; original load error: {first_err}"
                    )));
                    return (None, diag, Vec::new());
                }
            };
            if let Some(rodata) = open_skel2.maps.rodata_data.as_mut() {
                rodata.ktstr_enabled = true;
            }
            let trigger_programs = scheduler_exit_program_selection(exit_trigger);
            open_skel2
                .progs
                .ktstr_trigger_tp
                .set_autoload(trigger_programs.tracepoint);
            open_skel2
                .progs
                .ktstr_trigger_vexit_enter
                .set_autoload(trigger_programs.vexit_enter);
            open_skel2
                .progs
                .ktstr_trigger_vexit_return
                .set_autoload(trigger_programs.vexit_return);
            open_skel2
                .progs
                .ktstr_trigger_dump_fentry
                .set_autoload(trigger_programs.dump_fentry);
            open_skel2.progs.ktstr_pi_fentry.set_autoload(false);
            open_skel2.progs.ktstr_pi_fexit.set_autoload(false);
            open_skel2.progs.ktstr_lock_contend.set_autoload(false);
            open_skel2.progs.ktstr_tl_switch.set_autoload(false);
            open_skel2.progs.ktstr_tl_migrate.set_autoload(false);
            open_skel2.progs.ktstr_tl_wakeup.set_autoload(false);
            open_skel2
                .progs
                .ktstr_preempt_disable_tp
                .set_autoload(false);
            open_skel2.progs.ktstr_preempt_enable_tp.set_autoload(false);
            match open_skel2.load() {
                Ok(s) => (s, false),
                Err(e) => {
                    tracing::error!(
                        %e, %first_err,
                        "probe skeleton load failed (retry); \
                         surfacing both original and retry errors"
                    );
                    let libbpf_log = LIBBPF_LOG
                        .lock()
                        .map(|b| {
                            // Keep the tail: the verifier's rejection
                            // reason and the "load failed: -E..." line
                            // land at the end of the log. Slice on bytes
                            // and lossily reconstitute so a cut through a
                            // multi-byte char cannot panic.
                            let bytes = b.as_bytes();
                            let start = bytes.len().saturating_sub(8000);
                            String::from_utf8_lossy(&bytes[start..]).into_owned()
                        })
                        .unwrap_or_default();
                    diag.trigger_attach_error = Some(format!(
                        "skeleton load (retry): {e}; original error before retry: {first_err}; \
                         LIBBPF-LOG>>>\n{libbpf_log}\n<<<LIBBPF-LOG"
                    ));
                    let libbpf_tail = bounded_libbpf_tail(&libbpf_log);
                    publish_phase_a(Err(format!(
                        "probe skeleton load failed after minimal retry: {e}; \
                         original load error: {first_err}; \
                         libbpf tail: {libbpf_tail}"
                    )));
                    return (None, diag, Vec::new());
                }
            }
        }
    };

    // Populate func_meta_map with function IPs and metadata
    let mut func_ips: Vec<(u32, u64, String)> = Vec::new(); // (idx, ip, display_name)
    let mut bpf_funcs: Vec<(u32, &StackFunction)> = Vec::new(); // BPF functions for fentry

    // Load vmlinux BTF once and reuse across every kprobe meta
    // population in the loop below. The previous code called
    // `resolve_field_specs(_, None)` per function, which re-parsed
    // the multi-MB vmlinux BTF on every iteration (>1 s per kernel
    // with thousands of kprobed functions). Loading once turns the
    // hot path into pure type lookups against a borrowed handle.
    // `None` (load failure) leaves `cached_btf` empty and downstream
    // call sites fall back to the no-BTF path — same behaviour as
    // the previous per-call `Err(...) -> Vec::new()` branch.
    //
    // `cached_vmlinux_btf` memoises the first successful parse
    // process-wide so repeated auto-repro cycles in the same nextest
    // process share one `Arc<Btf>` instead of re-reading and
    // re-parsing the multi-MB blob each time.
    for (idx, func) in functions.iter().enumerate() {
        if func.is_bpf {
            bpf_funcs.push((idx as u32, func));
            continue;
        }
        let ip = match resolve_func_ip(&func.raw_name) {
            Some(ip) => ip,
            None => {
                tracing::warn!(func = %func.raw_name, "could not resolve function IP");
                diag.kprobe_resolve_failed.push(func.raw_name.clone());
                continue;
            }
        };

        let mut meta = types::func_meta {
            func_idx: idx as u32,
            ..Default::default()
        };

        // Populate field specs from BTF-resolved offsets.
        if let Some(btf_func) = btf_funcs.iter().find(|f| f.name == func.raw_name) {
            let field_specs = match cached_btf.as_ref() {
                Some(btf) => super::btf::resolve_field_specs_with_btf(btf_func, btf),
                None => Vec::new(),
            };
            populate_field_specs(&mut meta, &field_specs);
            // Detect char * params for string capture.
            meta.str_param_idx = detect_str_param(btf_func);
        }

        let key_bytes = ip.to_ne_bytes();
        let meta_bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const _ as *const u8,
                std::mem::size_of::<types::func_meta>(),
            )
        };

        if let Err(e) = skel
            .maps
            .func_meta_map
            .update(&key_bytes, meta_bytes, MapFlags::ANY)
        {
            tracing::warn!(%e, func = %func.raw_name, "failed to update func_meta_map");
            continue;
        }

        tracing::debug!(func = %func.raw_name, ip, nr = meta.nr_field_specs, "kprobe meta");
        diag.kprobe_resolved += 1;
        func_ips.push((idx as u32, ip, func.display_name.clone()));
    }

    if func_ips.is_empty() && bpf_funcs.is_empty() && phase_b_rx.is_none() {
        tracing::warn!("no kprobe IPs resolved and no BPF functions for fentry");
        diag.trigger_attach_error =
            Some("no functions resolved — kprobes and trigger skipped".to_string());
        publish_phase_a(Err("no probe functions resolved during Phase A".to_string()));
        return (None, diag, Vec::new());
    }
    if func_ips.is_empty() && (phase_b_rx.is_some() || !bpf_funcs.is_empty()) {
        tracing::debug!("no kernel functions resolved to IPs, proceeding with fentry only");
    }

    // Attach kprobes to each function for entry capture. Exit capture
    // for kernel functions uses fexit via the fentry skeleton (batched
    // separately below with fd=0 for vmlinux BTF).
    //
    // Parallelised via [`parallel_attach_kprobes`]: each attach is an
    // independent `perf_event_open` + `BPF_LINK_CREATE` syscall pair,
    // and the sequential loop's round-trip cost dominated the auto-
    // repro Phase A setup time when the crash backtrace named many
    // functions. Worker pool runs them concurrently; results land
    // back in the original input order so the post-attach
    // `kprobe_attach_failed` / `links` shape matches the prior loop.
    //
    // At most one kprobe per address, tracked in `probed_ips` and carried
    // into Phase B. A second kprobe on an address is pure redundancy: the
    // kernel folds both into an aggrprobe whose members run the same
    // `ktstr_probe` program and resolve the same IP-keyed `func_meta`, so
    // the duplicate costs an extra trap and yields no extra data. Phase B
    // derives kernel bridge callers from the scheduler's BPF ops and
    // readily re-names a function Phase A already probed.
    let mut links: Vec<(Link, String)> = Vec::new();
    let mut probed_ips: HashSet<u64> = HashSet::new();
    let mut attach_input: Vec<String> = Vec::new();
    let mut attach_ips: Vec<u64> = Vec::new();
    for (idx, ip, _) in &func_ips {
        let raw = &functions[*idx as usize].raw_name;
        if !probed_ips.insert(*ip) {
            tracing::debug!(func = %raw, ip, "kprobe already planned for ip, skipping");
            continue;
        }
        attach_input.push(raw.clone());
        attach_ips.push(*ip);
    }
    for ((raw, result), ip) in parallel_attach_kprobes(&skel.progs.ktstr_probe, &attach_input)
        .into_iter()
        .zip(attach_ips)
    {
        match result {
            Ok(link) => {
                links.push((link, raw));
            }
            Err(e) => {
                // Nothing is planted at this address, so leave it open
                // for Phase B to try again.
                probed_ips.remove(&ip);
                tracing::warn!(%e, func = %raw, "kprobe attach failed");
                diag.kprobe_attach_failed.push((raw, e.to_string()));
            }
        }
    }
    diag.kprobe_attached = links.len() as u32;
    tracing::debug!(attached = links.len(), total = func_ips.len(), "kprobes");

    // Attach fentry+fexit for BPF callbacks and kernel functions.
    // Batched in groups of FENTRY_BATCH per skeleton load to reduce
    // verifier passes. BPF callbacks use prog FD + sentinel IP.
    // Kernel functions use fd=0 (vmlinux BTF) + real IP.
    const FENTRY_BATCH: usize = 4;
    let mut fentry_links: Vec<Link> = Vec::new();
    let mut fexit_links: Vec<Link> = Vec::new();

    struct FentryTarget<'a> {
        slot: usize,
        fd: i32,
        idx: u32,
        name: &'a str,
        ok: bool,
        is_kernel: bool,
    }

    // Build combined list of targets: BPF callbacks + kernel functions.
    let valid_bpf: Vec<_> = bpf_funcs
        .iter()
        .filter(|(_, f)| f.bpf_prog_id.is_some())
        .collect();
    diag.fentry_candidates = valid_bpf.len() as u32;

    // Kernel functions that were attached via kprobe also get fentry+fexit
    // for exit capture. fd=0 targets vmlinux BTF.
    struct KernelFentryTarget {
        idx: u32,
        name: String,
    }
    let kernel_fexit_targets: Vec<KernelFentryTarget> = func_ips
        .iter()
        .map(|(idx, _, name)| KernelFentryTarget {
            idx: *idx,
            name: name.clone(),
        })
        .collect();

    // Interleave: process BPF targets first, then kernel targets.
    // Each gets batched into the fentry skeleton in groups of 4.
    //
    // When phase_b_rx is Some (Phase A/B split), BPF callback fentry
    // attachment is deferred to Phase B after the scheduler starts.
    // Only kernel fexit (fd=0) and kprobes run in Phase A.

    // --- BPF callback batches (skip in Phase A when split is active) ---
    if phase_b_rx.is_none() {
        for chunk in valid_bpf.chunks(FENTRY_BATCH) {
            let mut targets: Vec<FentryTarget<'_>> = Vec::new();
            for (slot, (idx, func)) in chunk.iter().enumerate() {
                let prog_id = func.bpf_prog_id.unwrap();
                let fd = if let Some(&pre_fd) = bpf_prog_fds.get(&prog_id) {
                    let dup_fd = unsafe { libc::dup(pre_fd) };
                    if dup_fd < 0 {
                        tracing::warn!(prog_id, func = %func.display_name, "fentry: dup failed");
                        diag.fentry_attach_failed.push((
                            func.display_name.clone(),
                            format!("dup(pre_fd={pre_fd}) failed"),
                        ));
                        continue;
                    }
                    dup_fd
                } else {
                    let fd = unsafe { libbpf_rs::libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
                    if fd < 0 {
                        tracing::warn!(prog_id, func = %func.display_name, "fentry: failed to get fd");
                        diag.fentry_attach_failed.push((
                            func.display_name.clone(),
                            format!("bpf_prog_get_fd_by_id({prog_id}) returned {fd}"),
                        ));
                        continue;
                    }
                    fd
                };
                targets.push(FentryTarget {
                    slot,
                    fd,
                    idx: *idx,
                    name: &func.display_name,
                    ok: false,
                    is_kernel: false,
                });
            }
            if targets.is_empty() {
                continue;
            }

            use crate::bpf_skel::fentry::*;
            let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
            let fentry_builder = FentryProbeSkelBuilder::default();
            let mut fentry_open = match fentry_builder.open(&mut fentry_open_obj) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(%e, "fentry skeleton open failed");
                    for t in &targets {
                        unsafe { libc::close(t.fd) };
                    }
                    continue;
                }
            };

            // Set rodata: func_idx and is_kernel per slot.
            if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
                rodata.ktstr_enabled = true;
                for t in &targets {
                    set_rodata_slot(rodata, t.slot, t.idx, t.is_kernel);
                }
            }

            for t in targets.iter_mut() {
                // Set fentry attach target.
                let Some(fentry_prog) = fentry_prog_mut_by_slot(&mut fentry_open, t.slot) else {
                    continue;
                };
                match fentry_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                    Ok(()) => {
                        t.ok = true;
                        tracing::debug!(
                            slot = t.slot,
                            func = t.name,
                            "fentry: set_attach_target ok"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(slot = t.slot, func = t.name, %e, "fentry: set_attach_target failed");
                        diag.fentry_attach_failed
                            .push((t.name.to_string(), format!("set_attach_target: {e}")));
                        continue;
                    }
                }
                // Set fexit attach target on the same function.
                let Some(fexit_prog) = fexit_prog_mut_by_slot(&mut fentry_open, t.slot) else {
                    continue;
                };
                if let Err(e) = fexit_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                    tracing::debug!(slot = t.slot, func = t.name, %e, "fexit: set_attach_target failed (entry-only)");
                    // Disable autoload so the verifier doesn't reject the
                    // skeleton due to a stale placeholder target.
                    fexit_prog.set_autoload(false);
                }
            }

            if !targets.iter().any(|t| t.ok) {
                for t in &targets {
                    unsafe { libc::close(t.fd) };
                }
                continue;
            }

            // Disable autoload on unused or failed fentry/fexit slots so the
            // verifier doesn't reject the placeholder target.
            let used_slots: std::collections::HashSet<usize> =
                targets.iter().filter(|t| t.ok).map(|t| t.slot).collect();
            for slot in 0..FENTRY_BATCH {
                if !used_slots.contains(&slot) {
                    disable_slot_programs(&mut fentry_open, slot);
                }
            }
            tracing::debug!(
                active = used_slots.len(),
                disabled = FENTRY_BATCH - used_slots.len(),
                "fentry: loading batch",
            );
            // Reuse the main skeleton's maps so fentry events land in the
            // same probe_data map that the Rust side reads.
            use std::os::unix::io::AsFd;
            if let Err(e) = fentry_open
                .maps
                .probe_data
                .reuse_fd(skel.maps.probe_data.as_fd())
            {
                tracing::warn!(%e, "fentry: probe_data reuse_fd failed");
            }
            if let Err(e) = fentry_open
                .maps
                .func_meta_map
                .reuse_fd(skel.maps.func_meta_map.as_fd())
            {
                tracing::warn!(%e, "fentry: func_meta_map reuse_fd failed");
            }

            let fentry_skel = match fentry_open.load() {
                Ok(s) => {
                    tracing::debug!("fentry: batch load success");
                    for t in &targets {
                        unsafe { libc::close(t.fd) };
                    }
                    s
                }
                Err(e) => {
                    tracing::warn!(%e, "fentry: batch load failed");
                    for t in &targets {
                        if t.ok {
                            diag.fentry_attach_failed
                                .push((t.name.to_string(), format!("batch load: {e}")));
                        }
                        unsafe { libc::close(t.fd) };
                    }
                    continue;
                }
            };

            // Populate func_meta and attach each slot.
            for t in &targets {
                if !t.ok {
                    continue;
                }

                let sentinel_ip = (t.idx as u64) | (1u64 << 63);
                let mut meta = crate::bpf_skel::types::func_meta {
                    func_idx: t.idx,
                    ..Default::default()
                };

                if let Some(btf_func) = btf_funcs.iter().find(|f| f.name == t.name) {
                    // Try vmlinux BTF first (for known struct params like
                    // task_struct and auto-discovered vmlinux fields),
                    // then BPF program BTF (for BPF-local types like task_ctx).
                    let mut field_specs = match cached_btf.as_ref() {
                        Some(btf) => super::btf::resolve_field_specs_with_btf(btf_func, btf),
                        None => Vec::new(),
                    };
                    if field_specs.is_empty()
                        && let Some(prog_id) = functions
                            .iter()
                            .find(|f| f.display_name == t.name)
                            .and_then(|f| f.bpf_prog_id)
                    {
                        field_specs = super::btf::resolve_bpf_field_specs(btf_func, prog_id);
                    }
                    populate_field_specs(&mut meta, &field_specs);
                    meta.str_param_idx = detect_str_param(btf_func);
                }

                let Some(result) = attach_fentry_by_slot(&fentry_skel, t.slot) else {
                    continue;
                };
                let link = match result {
                    Ok(link) => {
                        tracing::debug!(func = t.name, "fentry attached");
                        link
                    }
                    Err(e) => {
                        tracing::warn!(%e, func = t.name, "fentry attach failed");
                        diag.fentry_attach_failed
                            .push((t.name.to_string(), e.to_string()));
                        continue;
                    }
                };

                // func_meta_map.update + func_ips.push run AFTER the
                // fentry attach succeeds. Reversing the order would
                // orphan map entries and func_ip tuples on attach
                // failure — downstream reporting ("successfully
                // probed N funcs") would then show false-positive
                // successes for probes that never fired. If the
                // map update fails after the attach succeeded, drop
                // the Link (which detaches the program) so the
                // post-attach state matches what func_ips reports.
                let key_bytes = sentinel_ip.to_ne_bytes();
                let meta_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &meta as *const _ as *const u8,
                        std::mem::size_of::<crate::bpf_skel::types::func_meta>(),
                    )
                };
                if let Err(e) =
                    skel.maps
                        .func_meta_map
                        .update(&key_bytes, meta_bytes, MapFlags::ANY)
                {
                    tracing::warn!(%e, func = t.name, "fentry: failed to update func_meta_map; dropping attached link");
                    drop(link);
                    continue;
                }
                fentry_links.push(link);
                func_ips.push((t.idx, sentinel_ip, t.name.to_string()));
                // Attach fexit for exit-side capture.
                let Some(fexit_result) = attach_fexit_by_slot(&fentry_skel, t.slot) else {
                    continue;
                };
                match fexit_result {
                    Ok(link) => {
                        tracing::debug!(func = t.name, "fexit attached");
                        fexit_links.push(link);
                    }
                    Err(e) => {
                        tracing::debug!(%e, func = t.name, "fexit attach failed (entry-only)");
                    }
                }
            }

            drop(fentry_skel);
        }
        diag.fentry_attached = fentry_links.len() as u32;
        if !valid_bpf.is_empty() {
            tracing::debug!(
                fentry = fentry_links.len(),
                fexit = fexit_links.len(),
                total = valid_bpf.len(),
                "BPF probes",
            );
        }
    } // end if phase_b_rx.is_none() — BPF callback batches

    // --- Kernel function fexit batches (fd=0 = vmlinux BTF) ---
    use std::os::unix::io::AsFd;
    for chunk in kernel_fexit_targets.chunks(FENTRY_BATCH) {
        let names: Vec<(u32, &str)> = chunk.iter().map(|kt| (kt.idx, kt.name.as_str())).collect();
        if let Err((e, targeted)) = load_kernel_fexit_batch(
            &names,
            FENTRY_BATCH,
            skel.maps.probe_data.as_fd(),
            skel.maps.func_meta_map.as_fd(),
            &mut fexit_links,
            "kernel fexit",
        ) {
            tracing::warn!(
                %e,
                kind = ?e.kind(),
                attempts = KERNEL_FEXIT_LOAD_ATTEMPTS,
                targets = ?targeted,
                "kernel fexit: batch load failed",
            );
            for name in targeted {
                diag.fentry_attach_failed
                    .push((name, format!("batch load: {e}")));
            }
        }
    }
    if !kernel_fexit_targets.is_empty() {
        tracing::debug!(
            fexit = fexit_links.len(),
            total = kernel_fexit_targets.len(),
            "kernel fexit probes",
        );
    }

    // Attach the selected trigger program(s). The newest tracepoint is emitted
    // only for a successful claim, the preceding generation's raw return probe
    // rejects losing concurrent claims, and the global-era entry hook rechecks
    // scx_exit_info.kind to reject the SysRq-D caller.
    let trigger_attach: Result<Vec<(Link, String)>, String> = match exit_trigger {
        SchedulerExitTrigger::SchedExtExitTracepoint => skel
            .progs
            .ktstr_trigger_tp
            .attach_trace()
            .map(|link| vec![(link, "tp_btf/sched_ext_exit".to_string())])
            .map_err(|e| e.to_string()),
        SchedulerExitTrigger::ScxVexitKprobePair => (|| {
            let enter = skel
                .progs
                .ktstr_trigger_vexit_enter
                .attach_kprobe(false, "scx_vexit")
                .map_err(|e| format!("entry kprobe: {e}"))?;
            let ret = skel
                .progs
                .ktstr_trigger_vexit_return
                .attach_kprobe(true, "scx_vexit")
                .map_err(|e| format!("return kprobe: {e}"))?;
            Ok(vec![
                (enter, "kprobe/scx_vexit".to_string()),
                (ret, "kretprobe/scx_vexit".to_string()),
            ])
        })(),
        SchedulerExitTrigger::ScxDumpState => skel
            .progs
            .ktstr_trigger_dump_fentry
            .attach_trace()
            .map(|link| vec![(link, "fentry/scx_dump_state".to_string())])
            .map_err(|e| e.to_string()),
    };
    match trigger_attach {
        Ok(trigger_links) => {
            tracing::debug!(
                trigger = exit_trigger.section(),
                "scheduler-exit trigger attached"
            );
            diag.trigger_type = exit_trigger.diagnostic_type().to_string();
            links.extend(trigger_links);
        }
        Err(e) => {
            let msg = format!(
                "auto-repro could not attach selected scheduler-exit trigger {}: {e}",
                exit_trigger.section()
            );
            tracing::error!(%msg, "trigger attach failed");
            diag.trigger_attach_error = Some(msg.clone());
            publish_phase_a(Err(format!(
                "probe trigger attach failed during Phase A: {msg}"
            )));
            return (None, diag, Vec::new());
        }
    }

    // Attach timeline programs (loaded by the skeleton but not
    // auto-attached — they need explicit attach_trace calls).
    if optional_programs_loaded {
        match skel.progs.ktstr_tl_switch.attach_trace() {
            Ok(link) => links.push((link, "tp_btf/sched_switch".to_string())),
            Err(e) => tracing::warn!(%e, "timeline sched_switch attach failed"),
        }
        match skel.progs.ktstr_tl_migrate.attach_trace() {
            Ok(link) => links.push((link, "tp_btf/sched_migrate_task".to_string())),
            Err(e) => tracing::warn!(%e, "timeline sched_migrate_task attach failed"),
        }
        match skel.progs.ktstr_tl_wakeup.attach_trace() {
            Ok(link) => links.push((link, "tp_btf/sched_wakeup".to_string())),
            Err(e) => tracing::warn!(%e, "timeline sched_wakeup attach failed"),
        }
    }

    // Attach optional programs. When the first load succeeded (all
    // targets present), they were loaded and just need attachment.
    // When the fallback path ran, they were not loaded — attach
    // returns an error that we silently absorb.
    if optional_programs_loaded {
        if let Ok(link) = skel.progs.ktstr_pi_fentry.attach_trace() {
            links.push((link, "fentry/rt_mutex_setprio".to_string()));
        }
        if let Ok(link) = skel.progs.ktstr_pi_fexit.attach_trace() {
            links.push((link, "fexit/rt_mutex_setprio".to_string()));
        }
        if let Ok(link) = skel.progs.ktstr_lock_contend.attach_trace() {
            links.push((link, "tp_btf/contention_begin".to_string()));
        }
        if let Ok(link) = skel.progs.ktstr_preempt_disable_tp.attach_trace() {
            links.push((link, "tp_btf/preempt_disable".to_string()));
        }
        if let Ok(link) = skel.progs.ktstr_preempt_enable_tp.attach_trace() {
            links.push((link, "tp_btf/preempt_enable".to_string()));
        }
    }

    // Set up ring buffer
    let events: std::sync::Arc<std::sync::Mutex<Vec<ProbeEvent>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let triggered = std::sync::Arc::new(AtomicBool::new(false));
    let triggered_clone = triggered.clone();

    // Ring buffer event layout matching probe_event in intf.h.
    // `str_val` + `has_str` + `str_param_idx` are kept in the wire
    // layout for ABI symmetry with `struct probe_entry` (the
    // kprobe-side hash map shares the field names) — the
    // EVENT_TRIGGER producer leaves them zeroed.
    #[repr(C)]
    struct RbEvent {
        type_: u32,
        tid: u32,
        func_idx: u32,
        ts: u64,
        args: [u64; 6],
        fields: [u64; 16],
        nr_fields: u32,
        kstack: [u64; 32],
        kstack_sz: u32,
        str_val: [u8; MAX_STR_LEN],
        has_str: u8,
        str_param_idx: u8,
    }

    let mut rb_builder = RingBufferBuilder::new();
    if let Err(e) = rb_builder.add(&skel.maps.ktstr_events, move |data: &[u8]| {
        if data.len() < std::mem::size_of::<RbEvent>() {
            return 0;
        }
        let raw: &RbEvent = unsafe { &*(data.as_ptr() as *const RbEvent) };

        if raw.type_ == EVENT_TRIGGER {
            triggered_clone.store(true, Ordering::Release);

            let kstack_sz = (raw.kstack_sz as usize).min(32);
            let event = ProbeEvent {
                func_idx: 0,
                task_ptr: raw.args[0],
                ts: raw.ts,
                args: raw.args,
                fields: vec![],
                kstack: raw.kstack[..kstack_sz].to_vec(),
                str_val: None,
                exit_fields: vec![],
                exit_ts: None,
            };

            events_clone.lock().unwrap().push(event);
        }

        0
    }) {
        tracing::error!(%e, "failed to register ring buffer callback");
        let error = format!("probe ring-buffer callback registration failed: {e}");
        diag.trigger_attach_error = Some(error.clone());
        publish_phase_a(Err(error));
        return (None, diag, Vec::new());
    }

    let rb = match rb_builder.build() {
        Ok(rb) => rb,
        Err(e) => {
            tracing::error!(%e, "failed to build ring buffer");
            let error = format!("probe ring-buffer build failed: {e}");
            diag.trigger_attach_error = Some(error.clone());
            publish_phase_a(Err(error));
            return (None, diag, Vec::new());
        }
    };

    // Enable is handled by the BPF program reading the volatile const.
    // Since we can't mutate rodata after load, the program starts enabled.
    // (ktstr_enabled defaults to false in BPF, but we always want probes
    // active once attached — remove the gate or set it before load.)

    tracing::debug!(
        funcs = func_ips.len(),
        links = links.len(),
        trigger_type = %diag.trigger_type,
        "polling for probe data",
    );

    // Signal Phase A probes attached (kprobes + kernel fexit +
    // trigger). When phase_b_rx is None, this means all probes.
    // When Some, BPF fentry is deferred to Phase B.
    publish_phase_a(Ok(()));

    // Phase B: receive BPF fentry targets and attach them while
    // polling the ring buffer. The channel is consumed once; after
    // that phase_b_done prevents re-checking.
    let mut phase_b_rx = phase_b_rx;
    let mut phase_b_done = false;
    // Accumulates BTF from Phase B functions so the readout phase
    // can resolve field keys for both Phase A and Phase B functions.
    let mut phase_b_btf: Vec<BtfFunc> = Vec::new();

    // Poll until trigger fires or stop requested.  When stop is
    // signaled, iterate all probe_data entries instead of waiting
    // for the trigger.
    loop {
        let _ = rb.poll(Duration::from_millis(100));

        // Check for Phase B input while polling.
        if !phase_b_done && let Some(ref rx) = phase_b_rx {
            match rx.try_recv() {
                Ok(pb) => {
                    tracing::debug!(
                        bpf_funcs = pb.functions.len(),
                        "Phase B: attaching BPF fentry/fexit"
                    );
                    // Save Phase B BTF for readout field key resolution.
                    phase_b_btf = pb.btf_funcs.clone();
                    // Attach Phase B probes: kernel callers (kprobes
                    // + kernel fexit) and BPF callbacks (fentry/fexit).
                    attach_phase_b_fentry(
                        &skel,
                        &pb,
                        &mut func_ips,
                        &mut fentry_links,
                        &mut fexit_links,
                        &mut links,
                        &mut probed_ips,
                        &mut diag,
                    );
                    pb.done.set();
                    phase_b_done = true;
                    // Drop the receiver to release the channel.
                    phase_b_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No Phase B input yet; if trigger already fired,
                    // break immediately — the crash happened before
                    // Phase B could attach.
                    if triggered.load(Ordering::Acquire) {
                        tracing::debug!("trigger fired during Phase B wait, skipping fentry");
                        phase_b_done = true;
                        phase_b_rx = None;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Channel sender dropped without delivering Phase B
                    // input. Without a Phase B payload there is nothing
                    // left for this probe loop to attach; if `stop` is
                    // also set, return empty diagnostics rather than
                    // poll an empty ringbuf for the rest of the timeout.
                    tracing::debug!("Phase B channel disconnected");
                    phase_b_done = true;
                    phase_b_rx = None;
                    if stop.load(Ordering::Relaxed) {
                        return (None, diag, Vec::new());
                    }
                }
            }
        }

        // Also check BSS err_exit_detected: stall exits skip the
        // ring buffer (bpf_get_current_task is unrelated to the
        // stall cause) but still latch the BSS flag. Volatile read
        // because the BPF program writes via the kernel-side mmap
        // and Rust's aliasing rules let the compiler hoist a normal
        // read out of the loop.
        let bss_triggered = skel.maps.bss_data.as_ref().is_some_and(|bss| unsafe {
            std::ptr::read_volatile(&bss.ktstr_err_exit_detected as *const u32) != 0
        });
        // Mirror the latch into the cross-thread atomic so scenario
        // code can synchronously classify SchedulerDied as Clean vs
        // Crashed without owning the skeleton. Monotonic 0→1→2 —
        // once a poll iteration completes we know the probe armed,
        // and the BSS latch only ever transitions 0→1 within a
        // single test run. Release pairs with the Acquire load in
        // `sched_exit_kind()`.
        PROBE_SCHED_EXIT_STATE.store(
            if bss_triggered {
                PROBE_EXIT_STATE_CRASHED
            } else {
                PROBE_EXIT_STATE_CLEAN
            },
            Ordering::Release,
        );
        // Snapshot `triggered` once: a second `triggered.load` for the
        // diag assignment would observe an unrelated state if a racing
        // trigger fires between the two reads, so the gate decision
        // and the recorded `trigger_fired` could disagree.
        let triggered_snapshot = triggered.load(Ordering::Acquire);
        if triggered_snapshot || bss_triggered || stop.load(Ordering::Acquire) {
            // Final ringbuf drain before breaking. BPF-side ordering:
            // the trigger handler does
            // `__sync_val_compare_and_swap(&ktstr_err_exit_detected, 0u, 1u)`
            // BEFORE `bpf_ringbuf_reserve` + `bpf_ringbuf_submit`
            // (see src/bpf/probe.bpf.c near line 687-731). A userspace
            // observer can therefore see `bss_triggered=true` (CAS
            // visible) while the ringbuf event is still in transit
            // (submit not yet visible to `rb.poll`). Without an
            // explicit drain here, breaking out of the loop on
            // `bss_triggered` would lose the trigger event — the
            // readout phase below would see `events.last() = None`
            // (or the prior probe's trailing event), drop into the
            // no-causal-tptr fallback path, and produce
            // grouped-by-frequency output instead of a verified
            // stitch even though the kernel did publish a real
            // causal task pointer.
            //
            // 100 ms is bounded by the same teardown budget the loop's
            // top-level `rb.poll(100ms)` uses; libbpf's
            // `RingBuffer::poll` returns as soon as events are
            // consumed (it uses epoll-edge under the hood), so the
            // worst case is one full timeout window when no event
            // arrives — acceptable since we already know the trigger
            // fired (bss_triggered is true).
            let _ = rb.poll(Duration::from_millis(100));
            diag.trigger_fired = triggered_snapshot || bss_triggered;

            // Read BPF-side diagnostic counters from BSS. The hot
            // counters live in the per-CPU `ktstr_pcpu_counters`
            // 2D array (`[MAX_CPUS][KTSTR_PCPU_NR]`); each per-CPU
            // slot is a cacheline-aligned `pcpu_counter { long
            // value; }`. Sum across CPUs to recover the cumulative
            // count — see `enum ktstr_pcpu_idx` in
            // src/bpf/probe.bpf.c for the slot indices.
            //
            // Every read below uses [`std::ptr::read_volatile`].
            // The BSS struct is mapped to userspace via the BPF
            // map's mmap region; the BPF program writes through
            // its own kernel-side mapping concurrently with these
            // reads. Without `read_volatile`, the userspace
            // compiler is free to hoist the loads (Rust's
            // aliasing rules: the compiler does not know a
            // `&types::bss` reference is shared with a kernel
            // writer through an unrelated mapping), miss the
            // post-trigger updates, and leave the diagnostic
            // counters / miss log / first-trigger timestamp
            // showing pre-trigger zeroes. Mirrors the existing
            // `ktstr_err_exit_detected` `read_volatile` site
            // upstream — same hazard, same fix, applied to every
            // BPF-mutated field the diag block reads.
            if let Some(bss) = skel.maps.bss_data.as_ref() {
                // Slot indices must match `enum ktstr_pcpu_idx`. A
                // reorder in the BPF source breaks every reader; the
                // explicit constants here surface that drift at the
                // call site instead of silently aliasing two
                // counters.
                const PCPU_PROBE_COUNT: usize = 0;
                const PCPU_KPROBE_RETURNS: usize = 1;
                const PCPU_META_MISS: usize = 2;
                const PCPU_RINGBUF_DROPS: usize = 3;
                const PCPU_TIMELINE_COUNT: usize = 4;
                const PCPU_TIMELINE_DROPS: usize = 5;
                const PCPU_TRIGGER_COUNT: usize = 14;
                let counters = &bss.ktstr_pcpu_counters;
                // SAFETY: each `pcpu_counter::value` is a plain
                // 64-bit integer at a stable BSS offset; the BPF
                // side updates it via `__sync_add_and_fetch`
                // (atomic add). A volatile load reads whatever
                // the kernel-side mmap currently shows — torn
                // reads are impossible because aligned 64-bit
                // loads are atomic on every supported arch
                // (x86_64, aarch64). The volatile qualifier is
                // what prevents the compiler from hoisting the
                // load out of the `sum` reduction across the
                // outer poll loop.
                let sum_pcpu = |idx: usize| -> u64 {
                    counters
                        .iter()
                        .map(|cpu_slots| unsafe {
                            std::ptr::read_volatile(&cpu_slots[idx].value as *const _) as u64
                        })
                        .sum()
                };
                diag.bpf_kprobe_fires = sum_pcpu(PCPU_PROBE_COUNT);
                diag.bpf_kprobe_returns = sum_pcpu(PCPU_KPROBE_RETURNS);
                diag.bpf_trigger_fires = sum_pcpu(PCPU_TRIGGER_COUNT);
                diag.bpf_meta_misses = sum_pcpu(PCPU_META_MISS);
                // SAFETY: `ktstr_miss_log_idx` is a `u32` written
                // via `__sync_fetch_and_add` from the BPF side
                // (see `src/bpf/probe.bpf.c` near the
                // `ktstr_miss_log[idx] = ip;` line). Aligned u32
                // loads are atomic on x86_64/aarch64. The BPF
                // writer increments-then-stores; a volatile read
                // observes either the pre- or post-update value
                // — both are bounded by the array length, so the
                // subsequent `.min(ktstr_miss_log.len())` keeps
                // the slice safe even if the kernel-side write
                // races this read.
                let miss_idx = unsafe {
                    std::ptr::read_volatile(&bss.ktstr_miss_log_idx as *const u32) as usize
                };
                let n = miss_idx.min(bss.ktstr_miss_log.len());
                // Element-wise volatile reads of the miss-log
                // entries that fall within `n`. A bulk
                // `to_vec()` over the `bss.ktstr_miss_log[..n]`
                // slice would let the compiler vectorise the
                // copy and elide the volatile semantics; pulling
                // each `u64` through `read_volatile` keeps every
                // load ordered against the BPF-side write.
                //
                // SAFETY: each entry is a 64-bit IP value the
                // BPF writer stores after its CAS-like
                // increment of `ktstr_miss_log_idx`. Aligned
                // u64 loads are atomic on every supported
                // arch; the BPF write order
                // (increment of `ktstr_miss_log_idx` via
                // `__sync_fetch_and_add` BEFORE the store
                // `ktstr_miss_log[idx] = ip`) means a
                // volatile read of `[..miss_idx]` may cover a
                // slot whose index was already reserved but
                // whose store has not yet landed. We tolerate
                // that race: a stale-zero entry is harmless
                // diagnostic noise compared with the
                // alternative (compiler-hoisted loads of
                // pre-trigger zeroes).
                diag.bpf_miss_ips = (0..n)
                    .map(|i| unsafe {
                        std::ptr::read_volatile(&bss.ktstr_miss_log[i] as *const u64)
                    })
                    .collect();
                diag.bpf_ringbuf_drops = sum_pcpu(PCPU_RINGBUF_DROPS);
                // SAFETY: `ktstr_last_trigger_ts` is a `u64`
                // written by the BPF trigger handler via
                // `bpf_ktime_get_ns()` (see
                // `src/bpf/probe.bpf.c::ktstr_last_trigger_ts`).
                // Aligned u64 loads are atomic; the volatile
                // qualifier prevents hoisting across the outer
                // poll loop so the userspace reader observes the
                // post-trigger timestamp instead of a cached
                // pre-trigger zero.
                diag.bpf_first_trigger_ns =
                    unsafe { std::ptr::read_volatile(&bss.ktstr_last_trigger_ts as *const u64) };
                // SAFETY: `ktstr_exit_kind_snap` is a `u32` written
                // by the BPF trigger handler in the same publishing
                // CAS sequence as `ktstr_err_exit_detected` (see
                // `src/bpf/probe.bpf.c::ktstr_exit_kind_snap`).
                // Aligned u32 loads are atomic on every supported
                // arch; the volatile qualifier prevents the compiler
                // from hoisting the load across the outer poll loop
                // so userspace observes the post-trigger SCX_EXIT_*
                // value rather than the pre-trigger zero.
                diag.bpf_exit_kind_snap =
                    unsafe { std::ptr::read_volatile(&bss.ktstr_exit_kind_snap as *const u32) };
                diag.bpf_timeline_count = sum_pcpu(PCPU_TIMELINE_COUNT);
                diag.bpf_timeline_drops = sum_pcpu(PCPU_TIMELINE_DROPS);
            }

            let key_size = std::mem::size_of::<types::probe_key>();
            let mut probe_events = Vec::new();
            let mut total_keys = 0u32;
            let mut unmatched_ips = 0u32;

            // Build IP → (func_idx, display_name) lookup once. The
            // event-drain loop below would otherwise scan every entry
            // in `func_ips` per probe_data key — O(events × funcs).
            // With thousands of funcs and tens of thousands of events
            // on a normal run, the linear scan dominates dump time.
            // HashMap turns the per-event lookup into O(1).
            let func_ips_by_ip: std::collections::HashMap<u64, (u32, &str)> = func_ips
                .iter()
                .map(|(idx, ip, name)| (*ip, (*idx, name.as_str())))
                .collect();

            // Pre-compute per-function `field_keys_hints` once. The
            // previous code recomputed `build_field_keys` for every
            // event, even when many events share the same function —
            // O(events × funcs) field-key construction. Building the
            // map keyed by function name turns the per-event work
            // into a single HashMap lookup.
            let field_keys_by_func: std::collections::HashMap<&str, Vec<(String, RenderHint)>> =
                btf_funcs
                    .iter()
                    .chain(phase_b_btf.iter())
                    .map(|f| (f.name.as_str(), build_field_keys(f)))
                    .collect();

            for key_bytes in skel.maps.probe_data.keys() {
                if key_bytes.len() < key_size {
                    continue;
                }
                total_keys += 1;
                let key: &types::probe_key =
                    unsafe { &*(key_bytes.as_ptr() as *const types::probe_key) };

                // Find which function this IP belongs to.
                let (func_idx, display_name) = match func_ips_by_ip.get(&key.func_ip) {
                    Some(&(idx, name)) => (idx, name),
                    None => {
                        unmatched_ips += 1;
                        continue;
                    }
                };

                if let Ok(Some(val_bytes)) = skel.maps.probe_data.lookup(&key_bytes, MapFlags::ANY)
                {
                    let entry: &types::probe_entry =
                        unsafe { &*(val_bytes.as_ptr() as *const types::probe_entry) };
                    if entry.ts == 0 {
                        continue;
                    }

                    // Borrow the pre-computed hints for this function;
                    // an empty slice for unknown funcs preserves the
                    // previous `unwrap_or_default()` behaviour.
                    let empty: Vec<(String, RenderHint)> = Vec::new();
                    let field_keys_hints: &Vec<(String, RenderHint)> =
                        field_keys_by_func.get(display_name).unwrap_or(&empty);

                    let nr = (entry.nr_fields as usize).min(16);
                    let fields: Vec<(String, u64)> = entry.fields[..nr]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &val)| {
                            field_keys_hints.get(i).map(|(k, _)| (k.clone(), val))
                        })
                        .collect();

                    let str_val = if entry.has_str != 0 {
                        let s = &entry.str_val;
                        let bytes: Vec<u8> = s.iter().map(|&b| b as u8).collect();
                        let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                        let text = std::str::from_utf8(&bytes[..len]).unwrap_or("").to_string();
                        if text.is_empty() { None } else { Some(text) }
                    } else {
                        None
                    };

                    // Extract exit-side fields if fexit fired.
                    let (exit_fields, exit_ts) = if entry.has_exit != 0 {
                        let nr_exit = (entry.nr_exit_fields as usize).min(16);
                        let ef: Vec<(String, u64)> = entry.exit_fields[..nr_exit]
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &val)| {
                                field_keys_hints.get(i).map(|(k, _)| (k.clone(), val))
                            })
                            .collect();
                        (ef, Some(entry.exit_ts))
                    } else {
                        (Vec::new(), None)
                    };

                    probe_events.push(ProbeEvent {
                        func_idx,
                        task_ptr: key.task_ptr,
                        ts: entry.ts,
                        args: entry.args,
                        fields,
                        kstack: vec![],
                        str_val,
                        exit_fields,
                        exit_ts,
                    });
                }
            }

            probe_events.sort_by_key(|e| e.ts);

            diag.probe_data_keys = total_keys;
            diag.probe_data_unmatched_ips = unmatched_ips;
            diag.events_before_stitch = probe_events.len() as u32;

            tracing::debug!(
                events = probe_events.len(),
                total_keys,
                unmatched_ips,
                "probe_data readout",
            );

            if probe_events.is_empty() {
                return (None, diag, Vec::new());
            }

            // Stitch by task_struct pointer. Build a map of func_idx ->
            // task_struct param index from BPF_OP_CALLERS and BTF, then
            // filter events to those referencing the same task_struct
            // pointer as the causal task.
            //
            // The selected BPF trigger handler sets args[0] to
            // bpf_get_current_task() ONLY for
            // SCX_EXIT_ERROR_BPF (1025), where a BPF scheduler
            // callback faulted in the running task's context — so
            // `current` IS the causal task. For SCX_EXIT_ERROR
            // (1024), args[0] is 0 because that exit can fire from
            // kworker context (async unregistration, sysrq), where
            // `current` is the worker thread rather than the task
            // that triggered the exit. For SCX_EXIT_ERROR_STALL
            // (1026) the trigger handler returns early without
            // submitting an event at all (watchdog/timer context).
            // The filter below therefore drops args[0] == 0 to
            // suppress non-causal probe output: no causal task
            // means no useful stitch chain.
            let task_param_idx = build_task_param_idx(&func_ips, btf_funcs, &phase_b_btf);

            // Extract tptr and kstack from the trigger event in one
            // lock acquisition. When the trigger did not fire (stop-
            // signaled) or the exit kind lacks a causal task, probe
            // output is suppressed.
            let (target_tptr, trigger_kstack) = {
                let guard = events.lock().unwrap();
                let tptr = guard.last().and_then(|e| causal_tptr(e.task_ptr));
                let kstack = guard.last().map(|e| e.kstack.clone()).unwrap_or_default();
                (tptr, kstack)
            };

            let Some(tptr) = target_tptr else {
                // No causal task identified — the trigger fired with
                // args[0]==0 (kind=STALL or generic ERROR from kworker
                // context), or never fired at all (probe lifecycle
                // race, scheduler clean-exited).
                //
                // Best-effort fallback: instead of returning empty,
                // group the captured events by task_struct pointer
                // (resolved via task_param_idx for events with a
                // task_struct param, otherwise key.task_ptr from the
                // probe map) and pick the top-N most-frequent
                // pointers. The events that fired during the actual
                // crash window ARE causal data — the host renderer
                // will mark the output as "trigger absent —
                // best-effort grouped by frequency" so the operator
                // doesn't mistake the candidate chain for a verified
                // stitch. Cap at 3 candidates to keep the output
                // readable; one task usually dominates a single
                // crash window.
                use std::collections::HashMap;
                let mut counts: HashMap<u64, u32> = HashMap::new();
                for ev in &probe_events {
                    let key = if let Some(&pidx) = task_param_idx.get(&ev.func_idx) {
                        ev.args[pidx]
                    } else {
                        ev.task_ptr
                    };
                    if key != 0 {
                        *counts.entry(key).or_default() += 1;
                    }
                }
                if counts.is_empty() {
                    tracing::debug!(
                        "no causal tptr and no candidate task_ptrs — suppressing probe output"
                    );
                    return (None, diag, Vec::new());
                }
                let mut sorted: Vec<(u64, u32)> = counts.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                const MAX_CANDIDATES: usize = 3;
                sorted.truncate(MAX_CANDIDATES);
                let candidate_set: std::collections::HashSet<u64> =
                    sorted.iter().map(|(k, _)| *k).collect();
                let before = probe_events.len();
                probe_events.retain(|e| {
                    let key = if let Some(&pidx) = task_param_idx.get(&e.func_idx) {
                        e.args[pidx]
                    } else {
                        e.task_ptr
                    };
                    candidate_set.contains(&key)
                });
                tracing::debug!(
                    candidates = sorted.len(),
                    kept = probe_events.len(),
                    total = before,
                    "stitched by frequency (fallback — no causal tptr)"
                );
                diag.events_after_stitch = probe_events.len() as u32;
                diag.stitch_fallback_used = true;
                let fnames: Vec<(u32, String)> = func_ips
                    .iter()
                    .map(|(idx, _, name)| (*idx, name.clone()))
                    .collect();
                return (Some(probe_events), diag, fnames);
            };

            let before = probe_events.len();
            probe_events.retain(|e| {
                if let Some(&pidx) = task_param_idx.get(&e.func_idx) {
                    e.args[pidx] == tptr
                } else {
                    e.task_ptr == tptr // no task_struct param — match on current
                }
            });
            tracing::debug!(
                tptr = format_args!("0x{tptr:x}"),
                kept = probe_events.len(),
                total = before,
                "stitched by task_struct arg",
            );

            diag.events_after_stitch = probe_events.len() as u32;

            // Attach trigger kstack if available.
            if let Some(last) = probe_events.last_mut() {
                last.kstack = trigger_kstack;
            }

            let fnames: Vec<(u32, String)> = func_ips
                .iter()
                .map(|(idx, _, name)| (*idx, name.clone()))
                .collect();
            return (Some(probe_events), diag, fnames);
        }
    }
}

/// Attach Phase B probes after the scheduler starts.
///
/// Handles both kernel callers (kprobes + kernel fexit) and BPF
/// callbacks (fentry/fexit). Uses `pb.func_idx_offset` for all
/// func_idx values to avoid collisions with Phase A indices.
///
/// Kprobes are attached via the Phase A kprobe skeleton (`skel`),
/// which stays alive throughout. BPF fentry/fexit use separate
/// skeleton loads that share `probe_data` and `func_meta_map` via
/// `reuse_fd`.
///
/// `probed_ips` carries the addresses Phase A already planted a kprobe
/// on; targets landing on one of them get their `func_meta` but no
/// second kprobe.
#[allow(clippy::too_many_arguments)]
fn attach_phase_b_fentry(
    skel: &crate::bpf_skel::ProbeSkel<'_>,
    pb: &PhaseBInput,
    func_ips: &mut Vec<(u32, u64, String)>,
    fentry_links: &mut Vec<libbpf_rs::Link>,
    fexit_links: &mut Vec<libbpf_rs::Link>,
    links: &mut Vec<(libbpf_rs::Link, String)>,
    probed_ips: &mut HashSet<u64>,
    diag: &mut ProbeDiagnostics,
) {
    use libbpf_rs::MapCore;
    use libbpf_rs::MapFlags;
    use libbpf_rs::skel::{OpenSkel, SkelBuilder};

    const FENTRY_BATCH: usize = 4;

    struct FentryTarget<'a> {
        slot: usize,
        fd: i32,
        idx: u32,
        name: &'a str,
        ok: bool,
        is_kernel: bool,
    }

    let offset = pb.func_idx_offset;

    // Pre-load vmlinux BTF once for the Phase B kprobe + fentry
    // loops below. Same rationale as the Phase A path in
    // `run_probe_skeleton`: per-call `resolve_field_specs(_, None)`
    // would re-parse the multi-MB vmlinux BTF on every iteration,
    // dominating Phase B attach time on a kernel with thousands of
    // probed functions.
    //
    // `cached_vmlinux_btf` shares the process-global memo with Phase
    // A so a single auto-repro VM pays one parse for both phases, and
    // a multi-VM nextest run pays one parse total.
    let cached_btf = crate::monitor::btf_offsets::cached_vmlinux_btf();

    // --- Phase B kernel functions: kprobes + func_meta ---
    //
    // Two-pass shape: first populate `func_meta_map` for every
    // resolvable target and stage the (raw_name, idx, ip,
    // display_name) tuples, then run all kprobe attaches in
    // parallel via [`parallel_attach_kprobes`]. Splitting the loop
    // matters because the attach is the slow syscall pair (one
    // `perf_event_open` + one `BPF_LINK_CREATE` each); meta
    // population is a quick map update. Doing them as a single
    // sequential loop forces the slow attach to gate the next
    // iteration's meta update for no good reason.
    struct PhaseBKprobeTarget {
        raw_name: String,
        display_name: String,
        idx: u32,
        ip: u64,
    }
    let mut kprobe_targets: Vec<PhaseBKprobeTarget> = Vec::new();
    for (i, func) in pb.functions.iter().enumerate() {
        if func.is_bpf {
            continue;
        }
        let idx = offset + i as u32;
        let ip = match resolve_func_ip(&func.raw_name) {
            Some(ip) => ip,
            None => {
                tracing::warn!(func = %func.raw_name, "phase_b: could not resolve function IP");
                diag.kprobe_resolve_failed.push(func.raw_name.clone());
                continue;
            }
        };

        let mut meta = types::func_meta {
            func_idx: idx,
            ..Default::default()
        };

        if let Some(btf_func) = pb.btf_funcs.iter().find(|f| f.name == func.raw_name) {
            let field_specs = match cached_btf.as_ref() {
                Some(btf) => super::btf::resolve_field_specs_with_btf(btf_func, btf),
                None => Vec::new(),
            };
            populate_field_specs(&mut meta, &field_specs);
            meta.str_param_idx = detect_str_param(btf_func);
        }

        let key_bytes = ip.to_ne_bytes();
        let meta_bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const _ as *const u8,
                std::mem::size_of::<types::func_meta>(),
            )
        };

        if let Err(e) = skel
            .maps
            .func_meta_map
            .update(&key_bytes, meta_bytes, MapFlags::ANY)
        {
            tracing::warn!(%e, func = %func.raw_name, "phase_b: failed to update func_meta_map");
            continue;
        }

        kprobe_targets.push(PhaseBKprobeTarget {
            raw_name: func.raw_name.clone(),
            display_name: func.display_name.clone(),
            idx,
            ip,
        });
    }

    // Skip any target Phase A already has a kprobe on: that kprobe runs
    // the same `ktstr_probe` program and keys off the `func_meta` written
    // above, so the Phase B entry capture is unaffected, while a second
    // kprobe on the address would only aggregate them in the kernel. See
    // the Phase A attach loop.
    let mut attach_slots: Vec<usize> = Vec::new();
    let mut attach_input: Vec<String> = Vec::new();
    for (slot, target) in kprobe_targets.iter().enumerate() {
        if !probed_ips.insert(target.ip) {
            tracing::debug!(
                func = %target.raw_name,
                ip = target.ip,
                "phase_b: kprobe already attached at ip, skipping",
            );
            continue;
        }
        attach_slots.push(slot);
        attach_input.push(target.raw_name.clone());
    }
    // Kernel fexit must mirror the set that resolved and received func_meta,
    // not the raw Phase-B plan. Feeding an absent generation-specific alias to
    // set_attach_target can poison the whole batch even though the matching
    // kprobe was already rejected above.
    struct KernelFexitTarget {
        idx: u32,
        name: String,
    }
    let kernel_fexit_targets: Vec<KernelFexitTarget> = kprobe_targets
        .iter()
        .map(|target| KernelFexitTarget {
            idx: target.idx,
            name: target.raw_name.clone(),
        })
        .collect();
    let attach_results = parallel_attach_kprobes(&skel.progs.ktstr_probe, &attach_input);
    // Scatter the results back onto their targets. `attach_results` is in
    // the same order as `attach_input`, which `attach_slots` maps onto
    // `kprobe_targets`, so the post-attach bookkeeping (links push,
    // func_ips push, counter bumps) still runs in target order. A `None`
    // slot is a target skipped by the dedupe above.
    let mut results_by_slot: Vec<Option<libbpf_rs::Result<libbpf_rs::Link>>> =
        (0..kprobe_targets.len()).map(|_| None).collect();
    for (slot, (raw_name, result)) in attach_slots.into_iter().zip(attach_results.into_iter()) {
        debug_assert_eq!(kprobe_targets[slot].raw_name, raw_name);
        results_by_slot[slot] = Some(result);
    }
    for (target, result) in kprobe_targets.into_iter().zip(results_by_slot.into_iter()) {
        match result {
            Some(Ok(link)) => {
                links.push((link, target.raw_name));
                diag.kprobe_attached += 1;
            }
            Some(Err(e)) => {
                // Nothing planted here after all — release the address.
                probed_ips.remove(&target.ip);
                tracing::warn!(%e, func = %target.raw_name, "phase_b kprobe attach failed");
                diag.kprobe_attach_failed
                    .push((target.raw_name, e.to_string()));
            }
            None => {}
        }
        diag.kprobe_resolved += 1;
        func_ips.push((target.idx, target.ip, target.display_name));
    }

    // --- Phase B kernel function fexit batches (fd=0 = vmlinux BTF) ---
    use std::os::unix::io::AsFd;
    for chunk in kernel_fexit_targets.chunks(FENTRY_BATCH) {
        let names: Vec<(u32, &str)> = chunk.iter().map(|kt| (kt.idx, kt.name.as_str())).collect();
        if let Err((e, targeted)) = load_kernel_fexit_batch(
            &names,
            FENTRY_BATCH,
            skel.maps.probe_data.as_fd(),
            skel.maps.func_meta_map.as_fd(),
            fexit_links,
            "phase_b kernel fexit",
        ) {
            tracing::warn!(
                %e,
                kind = ?e.kind(),
                attempts = KERNEL_FEXIT_LOAD_ATTEMPTS,
                targets = ?targeted,
                "phase_b kernel fexit: batch load failed",
            );
            for name in targeted {
                diag.fentry_attach_failed
                    .push((name, format!("batch load: {e}")));
            }
        }
    }

    // --- Phase B BPF callback fentry/fexit batches ---
    let valid_bpf: Vec<(u32, &StackFunction)> = pb
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| f.bpf_prog_id.is_some())
        .map(|(i, f)| (offset + i as u32, f))
        .collect();
    diag.fentry_candidates += valid_bpf.len() as u32;

    for chunk in valid_bpf.chunks(FENTRY_BATCH) {
        let mut targets: Vec<FentryTarget<'_>> = Vec::new();
        for (slot, (idx, func)) in chunk.iter().enumerate() {
            let prog_id = func.bpf_prog_id.unwrap();
            let fd = if let Some(&pre_fd) = pb.bpf_prog_fds.get(&prog_id) {
                let dup_fd = unsafe { libc::dup(pre_fd) };
                if dup_fd < 0 {
                    tracing::warn!(prog_id, func = %func.display_name, "phase_b fentry: dup failed");
                    diag.fentry_attach_failed.push((
                        func.display_name.clone(),
                        format!("dup(pre_fd={pre_fd}) failed"),
                    ));
                    continue;
                }
                dup_fd
            } else {
                let fd = unsafe { libbpf_rs::libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
                if fd < 0 {
                    tracing::warn!(prog_id, func = %func.display_name, "phase_b fentry: failed to get fd");
                    diag.fentry_attach_failed.push((
                        func.display_name.clone(),
                        format!("bpf_prog_get_fd_by_id({prog_id}) returned {fd}"),
                    ));
                    continue;
                }
                fd
            };
            targets.push(FentryTarget {
                slot,
                fd,
                idx: *idx,
                name: &func.display_name,
                ok: false,
                is_kernel: false,
            });
        }
        if targets.is_empty() {
            continue;
        }

        use crate::bpf_skel::fentry::*;
        let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
        let fentry_builder = FentryProbeSkelBuilder::default();
        let mut fentry_open = match fentry_builder.open(&mut fentry_open_obj) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "phase_b fentry skeleton open failed");
                for t in &targets {
                    unsafe { libc::close(t.fd) };
                }
                continue;
            }
        };

        if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
            rodata.ktstr_enabled = true;
            for t in &targets {
                set_rodata_slot(rodata, t.slot, t.idx, t.is_kernel);
            }
        }

        for t in targets.iter_mut() {
            let Some(fentry_prog) = fentry_prog_mut_by_slot(&mut fentry_open, t.slot) else {
                continue;
            };
            match fentry_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                Ok(()) => {
                    t.ok = true;
                }
                Err(e) => {
                    tracing::warn!(slot = t.slot, func = t.name, %e, "phase_b fentry: set_attach_target failed");
                    diag.fentry_attach_failed
                        .push((t.name.to_string(), format!("set_attach_target: {e}")));
                    continue;
                }
            }
            let Some(fexit_prog) = fexit_prog_mut_by_slot(&mut fentry_open, t.slot) else {
                continue;
            };
            if let Err(e) = fexit_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                tracing::debug!(slot = t.slot, func = t.name, %e, "phase_b fexit: set_attach_target failed (entry-only)");
                fexit_prog.set_autoload(false);
            }
        }

        if !targets.iter().any(|t| t.ok) {
            for t in &targets {
                unsafe { libc::close(t.fd) };
            }
            continue;
        }

        let used_slots: std::collections::HashSet<usize> =
            targets.iter().filter(|t| t.ok).map(|t| t.slot).collect();
        for slot in 0..FENTRY_BATCH {
            if !used_slots.contains(&slot) {
                disable_slot_programs(&mut fentry_open, slot);
            }
        }

        use std::os::unix::io::AsFd;
        if let Err(e) = fentry_open
            .maps
            .probe_data
            .reuse_fd(skel.maps.probe_data.as_fd())
        {
            tracing::warn!(%e, "phase_b fentry: probe_data reuse_fd failed");
        }
        if let Err(e) = fentry_open
            .maps
            .func_meta_map
            .reuse_fd(skel.maps.func_meta_map.as_fd())
        {
            tracing::warn!(%e, "phase_b fentry: func_meta_map reuse_fd failed");
        }

        let fentry_skel = match fentry_open.load() {
            Ok(s) => {
                for t in &targets {
                    unsafe { libc::close(t.fd) };
                }
                s
            }
            Err(e) => {
                tracing::warn!(%e, "phase_b fentry: batch load failed");
                for t in &targets {
                    if t.ok {
                        diag.fentry_attach_failed
                            .push((t.name.to_string(), format!("batch load: {e}")));
                    }
                    unsafe { libc::close(t.fd) };
                }
                continue;
            }
        };

        for t in &targets {
            if !t.ok {
                continue;
            }

            let sentinel_ip = (t.idx as u64) | (1u64 << 63);
            let mut meta = crate::bpf_skel::types::func_meta {
                func_idx: t.idx,
                ..Default::default()
            };

            if let Some(btf_func) = pb.btf_funcs.iter().find(|f| f.name == t.name) {
                let mut field_specs = match cached_btf.as_ref() {
                    Some(btf) => super::btf::resolve_field_specs_with_btf(btf_func, btf),
                    None => Vec::new(),
                };
                if field_specs.is_empty()
                    && let Some(prog_id) = pb
                        .functions
                        .iter()
                        .find(|f| f.display_name == t.name)
                        .and_then(|f| f.bpf_prog_id)
                {
                    field_specs = super::btf::resolve_bpf_field_specs(btf_func, prog_id);
                }
                populate_field_specs(&mut meta, &field_specs);
                meta.str_param_idx = detect_str_param(btf_func);
            }

            let Some(result) = attach_fentry_by_slot(&fentry_skel, t.slot) else {
                continue;
            };
            let link = match result {
                Ok(link) => {
                    tracing::debug!(func = t.name, "phase_b fentry attached");
                    link
                }
                Err(e) => {
                    tracing::warn!(%e, func = t.name, "phase_b fentry attach failed");
                    diag.fentry_attach_failed
                        .push((t.name.to_string(), e.to_string()));
                    continue;
                }
            };

            // func_meta_map.update + func_ips.push run AFTER the
            // fentry attach succeeds. See the matching ordering
            // rationale at the phase A site above: reversing the
            // order would orphan map entries and func_ip tuples on
            // attach failure. If the map update fails after the
            // attach succeeded, drop the Link so post-attach state
            // matches what func_ips reports.
            let key_bytes = sentinel_ip.to_ne_bytes();
            let meta_bytes = unsafe {
                std::slice::from_raw_parts(
                    &meta as *const _ as *const u8,
                    std::mem::size_of::<crate::bpf_skel::types::func_meta>(),
                )
            };
            if let Err(e) = skel
                .maps
                .func_meta_map
                .update(&key_bytes, meta_bytes, MapFlags::ANY)
            {
                tracing::warn!(%e, func = t.name, "phase_b fentry: failed to update func_meta_map; dropping attached link");
                drop(link);
                continue;
            }
            fentry_links.push(link);
            func_ips.push((t.idx, sentinel_ip, t.name.to_string()));
            let Some(fexit_result) = attach_fexit_by_slot(&fentry_skel, t.slot) else {
                continue;
            };
            match fexit_result {
                Ok(link) => {
                    tracing::debug!(func = t.name, "phase_b fexit attached");
                    fexit_links.push(link);
                }
                Err(e) => {
                    tracing::debug!(%e, func = t.name, "phase_b fexit attach failed (entry-only)");
                }
            }
        }

        drop(fentry_skel);
    }

    diag.fentry_attached = fentry_links.len() as u32;
    tracing::debug!(
        fentry = fentry_links.len(),
        fexit = fexit_links.len(),
        bpf_targets = valid_bpf.len(),
        kernel_targets = kernel_fexit_targets.len(),
        "Phase B probes attached",
    );
}

#[cfg(test)]
#[path = "process_tests.rs"]
mod tests;
