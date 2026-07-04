//! Auto-repro and BPF probe pipeline for `#[ktstr_test]`.
//!
//! When a scheduler crash is observed in a ktstr_test VM, the framework
//! boots a second "repro" VM with BPF kprobes/fentries attached to the
//! functions that appeared in the crash stack. Probe output is
//! serialized on the guest (bulk port, via the stdout forwarder) and
//! deserialized + formatted on the host where DWARF is available.
//!
//! Probe attachment runs in two phases:
//! - **Phase A** ([`start_probe_phase_a`]) attaches kprobes, fexits,
//!   and the tracepoint trigger to kernel functions before the scheduler
//!   starts. Needed because kprobes must be in place before the
//!   first call to each traced function.
//! - **Phase B** ([`maybe_dispatch_vm_test_with_phase_a`]) discovers
//!   BPF symbols from the running scheduler and attaches fentries to
//!   BPF callbacks. Runs after the scheduler has loaded.
//!
//! The single-phase path ([`maybe_dispatch_vm_test_with_args`]) is used
//! when the kernel doesn't support Phase A; all probes attach after the
//! scheduler is up.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::assert::AssertResult;

use super::args::{
    extract_probe_stack_arg, extract_test_fn_arg, extract_variant_hash_arg, extract_work_type_arg,
    resolve_cgroup_root,
};
use super::entry::find_test;
use super::output::{extract_sched_ext_dump, print_assert_result};
use super::profraw::try_flush_profraw;
use super::runtime::{config_content_parts, config_file_parts, verbose};
use super::{KtstrTestEntry, TopoOverride};
use crate::verifier::{
    SCHED_OUTPUT_END, SCHED_OUTPUT_START, parse_sched_output, parse_sched_output_partial,
};

/// Sentinel value for `--ktstr-probe-stack` when no crash stack functions
/// were extracted. Triggers the guest-side probe path so
/// `discover_bpf_symbols()` can dynamically find the scheduler's BPF
/// programs. `filter_traceable` drops it (not in kallsyms).
const DISCOVER_SENTINEL: &str = "__discover__";

/// Propagate `RUST_BACKTRACE` and `RUST_LOG` from the guest kernel
/// cmdline into the process environment.
///
/// # Safety invariant
///
/// Performs `std::env::set_var`, which is unsound on Linux unless the
/// process is provably single-threaded. glibc mutates the global
/// `__environ` array without locks, so a concurrent reader or another
/// `set_var` produces UB. The two callers
/// ([`ktstr_test_early_dispatch`](super::dispatch::ktstr_test_early_dispatch)
/// and the `ktstr_guest_init` boot path in `vmm::rust_init`) both
/// invoke this before any probe / workload / test thread is spawned.
pub(crate) fn propagate_rust_env_from_cmdline() {
    let Ok(cmdline) = std::fs::read_to_string("/proc/cmdline") else {
        return;
    };
    for (key, val) in parse_rust_env_from_cmdline(&cmdline) {
        // SAFETY: called from ktstr_guest_init before any probe /
        // workload thread is spawned; single-threaded mutation of
        // `__environ` is sound.
        unsafe { std::env::set_var(key, val) };
    }
}

/// Pure parser for the cmdline side of `propagate_rust_env_from_cmdline`.
/// Returns `(key, value)` pairs for every `RUST_BACKTRACE=...`,
/// `RUST_LOG=...`, or `KTSTR_SIDECAR_DIR=...` token found in
/// whitespace-split `cmdline`, in the order they appear. Split from
/// the env-mutating wrapper so the parse logic is testable without
/// touching the process environment.
///
/// `KTSTR_SIDECAR_DIR` propagation: the host writes its resolved
/// `sidecar_dir()` into the kernel cmdline at boot so the guest's
/// `sidecar_dir()` returns the same override path. Without this,
/// host and guest each compute the run directory independently —
/// the host's `gix::discover` walks the workspace tree and produces
/// `{kernel}-{commit}` while the guest's cwd is `/` and falls back
/// to `unknown-unknown`. With the override propagated, both sides
/// agree on the path so a guest scenario reading
/// `sidecar_dir().join(...)` resolves to the same string the host's
/// freeze coordinator writes to (the file itself still lives on the
/// host filesystem; this only aligns the path computation, not
/// cross-process file access).
fn parse_rust_env_from_cmdline(cmdline: &str) -> Vec<(&'static str, &str)> {
    let mut out = Vec::new();
    let sidecar_prefix = format!("{}=", crate::KTSTR_SIDECAR_DIR_ENV);
    for token in cmdline.split_whitespace() {
        if let Some(val) = token.strip_prefix("RUST_BACKTRACE=") {
            out.push(("RUST_BACKTRACE", val));
        } else if let Some(val) = token.strip_prefix("RUST_LOG=") {
            out.push(("RUST_LOG", val));
        } else if let Some(val) = token.strip_prefix(sidecar_prefix.as_str()) {
            out.push((crate::KTSTR_SIDECAR_DIR_ENV, val));
        }
    }
    out
}

/// Delimiters for probe output, emitted by emit_probe_payload via println! to
/// the bulk-port stdout forwarder.
pub(crate) const PROBE_OUTPUT_START: &str = "===PROBE_OUTPUT_START===";
pub(crate) const PROBE_OUTPUT_END: &str = "===PROBE_OUTPUT_END===";

/// Format the last `n` lines of `text` under a `--- header ---` delimiter.
/// Returns `None` if `text` is empty.
fn format_tail(text: &str, n: usize, header: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let start = lines.len().saturating_sub(n);
    Some(format!("--- {header} ---\n{}", lines[start..].join("\n")))
}

/// The literal tracepoint-output marker for `sched_ext_dump` lines in
/// both wire formats the host captures. The trailing colon is part of
/// the marker:
///
/// - trace_pipe (ftrace): `<task>-<pid>  [<cpu>]  <ts>: sched_ext_dump: <body>`
/// - dmesg printk:        `[<ts>] sched_ext_dump: <body>`
///
/// Both formats place `: ` immediately after the event name, so the
/// colon anchors the marker against unrelated printk content that
/// merely mentions the substring `sched_ext_dump` (e.g. a future
/// kernel message documenting the tracepoint by name).
pub(crate) const SCHED_EXT_DUMP_MARKER: &str = "sched_ext_dump:";

/// Render the repro VM's stderr as a `--- repro VM dmesg ---` tail
/// after stripping `sched_ext_dump` lines (those land in their own
/// section via [`extract_sched_ext_dump`]). Always returns a string —
/// every input path produces some operator-readable output: the
/// pointer diagnostic, the corruption diagnostic, or the
/// `format_tail` rendering of remaining content. An empty stderr
/// surfaces the "scheduler crashed before kernel printk" diagnostic;
/// a stderr containing only `sched_ext_dump` lines surfaces the
/// pointer to the dump section.
///
/// Three cases distinguished:
///
/// 1. The filter dropped one or more `sched_ext_dump` lines AND the
///    post-filter text is empty or matches a corruption shape. The
///    VM produced real dump output that is already rendered in the
///    `--- repro VM sched_ext dump ---` section above; the dmesg
///    section emits a pointer-to-that diagnostic. This covers two
///    sub-cases:
///    - every non-dump byte was whitespace (clean run, dump lines
///      only);
///    - non-dump bytes existed but matched
///      [`classify_dmesg_corruption`] (0xFF / U+FFFD / control
///      chars) — surfacing the classifier's "scheduler crashed
///      before kernel printk" message here would contradict the
///      proven-real dump section above. Point at the dump section
///      so the operator does not chase a phantom crash.
/// 2. The filter dropped nothing and the post-filter text matches
///    a corruption shape — surface [`classify_dmesg_corruption`]'s
///    diagnostic.
/// 3. The post-filter text has real content — defer to
///    [`format_tail`] for the standard `n`-line tail render.
fn render_dmesg_tail(stderr: &str, tail_lines: usize) -> String {
    let mut filter_dropped_any = false;
    let filtered: String = stderr
        .lines()
        .filter(|l| {
            let drop = l.contains(SCHED_EXT_DUMP_MARKER);
            filter_dropped_any |= drop;
            !drop
        })
        .collect::<Vec<_>>()
        .join("\n");
    let post_filter_corrupt = classify_dmesg_corruption(&filtered);
    if filter_dropped_any && post_filter_corrupt.is_some() {
        // Filter saw real dump lines — never surface a "scheduler
        // crashed before kernel printk" diagnostic against the
        // residue of those lines. Point at the dump section above.
        return "--- repro VM dmesg ---\n(no kernel printk other than \
                sched_ext_dump — full dump in section above)"
            .to_string();
    }
    if let Some(diag) = post_filter_corrupt {
        return format!("--- repro VM dmesg ---\n{diag}");
    }
    // `filtered` is non-empty (classify returns Some for empty input)
    // so `format_tail` is guaranteed to return Some — unwrap with a
    // sentinel header rather than an `unwrap()` to preserve the
    // single-string return contract.
    format_tail(&filtered, tail_lines, "repro VM dmesg")
        .unwrap_or_else(|| "--- repro VM dmesg ---\n(unavailable)".to_string())
}

/// Detect dmesg corruption shapes that would render as opaque garbage
/// in a `format_tail` block, and return a single-line operator-readable
/// diagnostic instead.
///
/// Returns:
/// - `Some("empty ...")` when the input is empty or contains only
///   whitespace — no kernel printk reached the buffer.
/// - `Some("corrupt ...")` when every non-whitespace char is a
///   U+FFFD replacement character or a non-whitespace control
///   character ([`char::is_control`] — C0 range U+0000..=U+001F
///   excluding whitespace, DEL U+007F, and C1 range U+0080..=U+009F).
///   The UART buffer was uninitialized / trimmed / filled with NUL
///   or other control bytes — the scheduler likely crashed before
///   any kernel printk was written.
/// - `None` when at least one ordinary printable / extended-ASCII
///   character is present — the caller falls through to the
///   standard `format_tail` rendering.
///
/// The 0xFF byte is `vm-superio`'s convention for uninitialized read
/// positions in the UART ring buffer (see vm-superio/src/serial.rs);
/// a stream of 0xFF means the scheduler crashed before any kernel
/// printk reached the buffer, OR the COM1 capture buffer overflowed
/// and trimmed the relevant bytes. The host captures stderr via
/// [`String::from_utf8_lossy`] (see `vmm::console::output`), which
/// decodes every invalid UTF-8 byte (including a raw 0xFF) to U+FFFD.
/// Raw 0xFF bytes therefore arrive at this classifier as U+FFFD
/// chars, never as the Unicode codepoint U+00FF (which is the
/// legitimate Latin-1 letter `ÿ` — checking for U+00FF would false-
/// positive on valid printk content). The U+0000 NUL byte by
/// contrast is valid UTF-8 and arrives as the U+0000 char; covering
/// it (and other C0 control chars) keeps an uninitialized-NUL-byte
/// UART buffer from rendering as silent garbage.
fn classify_dmesg_corruption(text: &str) -> Option<&'static str> {
    if text.is_empty() {
        return Some("empty (scheduler crashed before kernel printk reached the UART buffer)");
    }
    // Walk every char. If we ever see something other than
    // whitespace / U+FFFD / non-whitespace control char, the text
    // has real content and we let format_tail render it.
    let mut saw_corrupt = false;
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        if c == '\u{fffd}' || c.is_control() {
            saw_corrupt = true;
            continue;
        }
        return None;
    }
    if saw_corrupt {
        Some(
            "corrupt or no readable text (UART buffer uninitialized or \
                 trimmed — scheduler likely crashed before any kernel printk)",
        )
    } else {
        // Whitespace only — same outcome as empty for the
        // operator: the kernel never wrote anything readable.
        Some("empty (scheduler crashed before kernel printk reached the UART buffer)")
    }
}

/// Read the repro VM's failure-dump JSON, parse it via
/// [`crate::monitor::dump::FailureDumpReportAny`], and emit the
/// Display rendering as a `--- repro VM failure dump ---` tail.
/// Returns `None` when the file is missing (no freeze fired during
/// repro), unreadable, or fails to parse — the JSON file itself
/// stays on disk for any downstream consumer that needs the
/// structured form. Schema-dispatch logic (single vs dual vs
/// degraded discriminant; absent or unknown schema rejection) lives
/// in `FailureDumpReportAny::from_json`; this helper is just the
/// file-IO + tail-header wrapper.
fn render_failure_dump_file(path: &std::path::Path) -> Option<String> {
    use crate::monitor::dump::FailureDumpReportAny;
    use std::fmt::Write;
    let json = std::fs::read_to_string(path).ok()?;
    let any = FailureDumpReportAny::from_json(&json)?;
    let mut buf = String::with_capacity(json.len());
    buf.push_str("--- repro VM failure dump ---\n");
    let _ = write!(buf, "{any}");
    Some(buf)
}

/// Prepend "PRIMARY DID NOT REACH WORKLOAD" to a repro VM verdict
/// when the primary VM failed before emitting its `PayloadStarting`
/// lifecycle frame — the marker that the test workload was reached.
/// When the primary DID reach the workload (`PayloadStarting` was
/// emitted) the verdict passes through unchanged: a clean repro run
/// is load-bearing evidence about reproducibility.
///
/// Uses [`primary_reached_workload`] directly rather than
/// piggybacking on [`classify_init_stage`]'s stage-string bucketing,
/// because that bucketing lumps `SchedulerNotAttached`
/// (pre-workload failure) into the same string as `PayloadStarting`
/// — the bucketed test would mislabel a "scheduler failed to attach"
/// primary as "reached workload."
///
/// The wrap preserves the original verdict on a following line so
/// operators can still see whether the repro VM itself booted
/// successfully — useful for distinguishing "primary AND repro both
/// failed to reach workload" from "primary failed to reach workload
/// but the repro completed cleanly" (the latter still confirms the
/// repro VM's boot path works, just on a topology/timing the primary
/// couldn't survive).
///
/// [`primary_reached_workload`]: crate::test_support::output::primary_reached_workload
/// [`classify_init_stage`]: crate::test_support::output::classify_init_stage
fn label_repro_verdict_when_workload_not_reached(
    primary_reached_workload: bool,
    repro_verdict: &str,
) -> String {
    if primary_reached_workload {
        repro_verdict.to_string()
    } else {
        format!(
            "PRIMARY DID NOT REACH WORKLOAD — auto-repro is not \
             load-bearing (the primary VM's failure prevented the bug \
             from being exercised, so the repro's verdict below should \
             not be read as evidence about bug reproducibility — the \
             bug was never exercised by either run)\n\
             {repro_verdict}"
        )
    }
}

/// Classify the repro VM outcome into a single human-readable status
/// line, used when the probe pipeline produced no events and the
/// caller needs to tell the user *why* the repro VM did not yield
/// probe data.
///
/// The ordering of branches matters. Each check eliminates a
/// distinct failure mode so the most specific match wins:
///
/// 1. `timed_out` — VM wall clock exceeded. No further signals are
///    meaningful; the run never reached a natural exit.
/// 2. A `SchedulerNotAttached` lifecycle frame in `guest_messages`
///    (matched by `extract_not_attached_reason`) — the scheduler
///    process stayed alive but never completed attachment (BPF
///    verifier reject, ops mismatch, sysfs absent). `rust_init` emits
///    this lifecycle frame with a reason suffix then force-reboots,
///    which is distinct from a scheduler crash. Checked *before* the
///    crash branch because the emission path prevents a subsequent
///    `SchedulerDied` frame in the same run.
/// 3. `has_crash_message`, or a `SchedulerDied` lifecycle frame in
///    `guest_messages` — the scheduler process crashed or was
///    reported dead by the guest's sched_exit monitor.
/// 4. Nonzero exit code — something exited abnormally, but the
///    guest did not emit a classification sentinel.
/// 5. Clean exit — scheduler ran to completion; the first VM's
///    crash did not reproduce.
fn classify_repro_vm_status(
    timed_out: bool,
    has_crash_message: bool,
    exit_code: i32,
    guest_messages: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> String {
    if timed_out {
        return "repro VM: timed out".to_string();
    }
    if let Some(reason) = extract_not_attached_reason(guest_messages) {
        return format!("repro VM: scheduler did not attach ({reason}) (exit code {exit_code})",);
    }
    let scheduler_died = guest_messages
        .map(|d| {
            d.entries.iter().any(|e| {
                e.msg_type == crate::vmm::wire::MSG_TYPE_LIFECYCLE
                    && e.crc_ok
                    && !e.payload.is_empty()
                    && crate::vmm::wire::LifecyclePhase::from_wire(e.payload[0])
                        == Some(crate::vmm::wire::LifecyclePhase::SchedulerDied)
            })
        })
        .unwrap_or(false);
    if has_crash_message || scheduler_died {
        // Describe qemu's exit disposition precisely: a detected crash
        // (crash_message, or a SchedulerDied lifecycle frame in
        // guest_messages) can coincide with qemu itself exiting 0 (guest
        // panic handler + orderly reboot), >0 (propagated non-zero),
        // -1 (VMM internal sentinel — the boot CPU's run loop seeds
        // `VmResult::exit_code = -1` and leaves it at -1 on error
        // paths that did not deliver the guest's final exit message;
        // watchdog-fire is caught earlier via `timed_out`, so a -1
        // reaching THIS branch indicates a code-unsetting error
        // path, not a signal-kill), or <-1 (signal-kill, rendered
        // via `ExitStatus::signal()` as a negative i32 on unix).
        // Labeling all four as "crashed (exit N)" conflates the
        // guest-scheduler failure with qemu's own exit, making the
        // qemu-clean and VMM-sentinel cases especially misleading.
        // The -1 clause is phrased in end-user terms: the internals
        // (boot-CPU run loop, scheduler-exit IPC message) belong in
        // the code comment, not in the output a test operator reads
        // at the console.
        let exit_clause = if exit_code == -1 {
            "VM host reported no final exit status (the scheduler did not \
             deliver an exit signal before the VM ended)"
                .to_string()
        } else if exit_code < 0 {
            format!("killed by signal ({exit_code})")
        } else if exit_code == 0 {
            "exited cleanly".to_string()
        } else {
            format!("exited with non-zero status ({exit_code})")
        };
        return format!("repro VM: scheduler crashed — {exit_clause}");
    }
    if exit_code != 0 {
        return format!("repro VM: exited abnormally (exit code {exit_code})");
    }
    "repro VM: scheduler ran normally (crash did not reproduce)".to_string()
}

/// Extract the reason suffix from the first
/// [`crate::vmm::wire::LifecyclePhase::SchedulerNotAttached`] frame
/// in the bulk-port drain. Returns `Some("timeout")` for a
/// timeout-on-attach emission, `Some("sched_ext sysfs absent")` for
/// the sysfs-absent emission, or `None` when no
/// `SchedulerNotAttached` lifecycle frame is present (or the frame
/// carries an empty reason).
///
/// Pre-bulk-port-migration: the emission lived as a
/// `"SCHEDULER_NOT_ATTACHED: <reason>"` COM2 line and the parser
/// split at the first colon. The reason now travels in the
/// `MSG_TYPE_LIFECYCLE` payload bytes after the 1-byte phase
/// header (see `vmm::guest_comms::send_lifecycle`), so the
/// extraction is a direct UTF-8 slice without any delimiter
/// parsing.
///
/// FIRST matching frame wins unconditionally — same semantics as
/// the prior line-walk. The caller handles `None` by routing to
/// the generic crashed / abnormal-exit branches, which already
/// surface exit code and crash-message diagnostics.
fn extract_not_attached_reason(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> Option<String> {
    use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE};
    let drain = drain?;
    for e in &drain.entries {
        if e.msg_type != MSG_TYPE_LIFECYCLE || !e.crc_ok || e.payload.is_empty() {
            continue;
        }
        if LifecyclePhase::from_wire(e.payload[0]) != Some(LifecyclePhase::SchedulerNotAttached) {
            continue;
        }
        let reason = String::from_utf8_lossy(&e.payload[1..]).trim().to_string();
        if reason.is_empty() {
            return None;
        }
        return Some(reason);
    }
    None
}

/// Persist the auto-repro VM's bulk-drain sidecar artifacts to
/// disk: the wprof Perfetto trace. The wprof path uses the
/// `.repro.wprof.pb` infix so it sits beside (not on top of) the
/// primary VM's `.wprof.pb` from the same test — the `.repro.`
/// infix matches the convention every other repro-VM sidecar in
/// the crate already uses (`.repro.failure-dump.json`,
/// `.repro.probe-payload.partial.json`).
///
/// Coverage profraw from the auto-repro run is NOT extracted here:
/// the repro VM runs via [`crate::vmm::KtstrVm::run`], which
/// persists every `Profraw` frame centrally (see
/// [`crate::test_support::persist_guest_profraw`]). Extracting it
/// here too would write the same payload twice and make
/// `llvm-profdata merge` double-count the counters.
///
/// Mirrors the primary VM's wprof reassembly in
/// `crate::test_support::eval::run_ktstr_test_inner_impl` — both call
/// [`crate::test_support::wprof::reassemble_wprof_trace`] over the full
/// bulk drain so a chunked trace is concatenated, not truncated to its
/// terminal frame. Only the wprof trace unique to this path is
/// persisted here — Stimulus and PayloadMetrics
/// frames from the auto-repro run are intentionally NOT extracted
/// because they're a duplicate of the primary's and the verdict
/// context only applies to the primary's drain.
///
/// CRC failures gate the write per arm — a corrupted frame's
/// payload is undecidable, so writing it would mask the corruption
/// rather than surface it.
///
/// Best-effort: every disk-write failure logs to stderr but never
/// aborts the auto-repro flow. The caller's job is to surface
/// repro verdicts; losing a sidecar artifact is observable on the
/// filesystem (file absent) but should not erase the verdict.
///
/// Cross-binary collision safety relies on cargo-nextest's per-test
/// binary isolation: each test binary runs as its own process under
/// nextest, so two test binaries with identically-named
/// `auto_repro=true` tests still write to distinct
/// `sidecar_dir()`'s by binary lineage. Within a single binary,
/// `KtstrTestEntry::name` is unique per test fn (the
/// `#[ktstr_test]` macro derives it from the fn's module-path).
///
/// Gated on `wprof`: the only artifact written here is the wprof trace, and
/// the reassembly helper lives in the wprof-gated `test_support::wprof`
/// module. A non-wprof build skips both the fn and its call site.
#[cfg(feature = "wprof")]
fn write_auto_repro_sidecar_artifacts(
    entry: &KtstrTestEntry,
    repro_result: &crate::vmm::result::VmResult,
) {
    let Some(drain) = repro_result.guest_messages.as_ref() else {
        return;
    };
    // Reassemble the (possibly chunked) wprof trace: the guest ships a trace
    // larger than one bulk frame as N-1 WprofTraceChunk slices followed by a
    // terminal WprofTrace; `reassemble_wprof_trace` concatenates them in
    // arrival order and returns `None` when no terminal frame arrived or the
    // trace is empty (skip-on-absent, matching the pre-chunking behavior).
    if let Some(pb) = crate::test_support::wprof::reassemble_wprof_trace(&drain.entries) {
        // Variant-keyed name matching VmResult::repro_wprof_pb_path
        // (the reader the expect_auto_repro inversion resolves).
        // repro_result.variant_hash is stamped by attempt_auto_repro
        // to the primary's host-authoritative hash; the
        // writer==reader invariant is pinned by a regression test.
        let wprof_path = crate::test_support::sidecar::sidecar_dir().join(format!(
            "{}-{:016x}.repro.wprof.pb",
            entry.name, repro_result.variant_hash
        ));
        if let Err(e) = std::fs::create_dir_all(
            wprof_path
                .parent()
                .expect("sidecar_dir join always has parent"),
        ) {
            eprintln!("ktstr_test: auto-repro: create sidecar dir for wprof trace: {e}",);
        } else if let Err(e) = std::fs::write(&wprof_path, &pb) {
            eprintln!(
                "ktstr_test: auto-repro: write wprof trace to {}: {e}",
                wprof_path.display(),
            );
        }
    }
}

/// Extra host-deadline budget the auto-repro VM gets beyond the workload
/// timeout, covering the post-trigger probe-drain tail: after the fentry
/// trigger fires, `run_probe_skeleton` reads the `probe_data` map, serializes
/// the per-function arg payload, and flushes it (framed by `PROBE_OUTPUT_END`)
/// over the bulk port before `force_reboot`. The base host timeout
/// ([`vm_timeout_from_entry`](super::runtime::vm_timeout_from_entry)) is sized
/// for the workload only, so without this grace a slow or large drain can hit
/// the deadline mid-flush — the host then extracts a truncated, terminator-less
/// payload, fails to parse it, and drops a fully-captured arg set as
/// "auto-repro: no probe data". Sized well above the worst-case drain of the
/// bounded `probe_data` map (so a slow host or a large captured-function set
/// still flushes the terminator in time); it is a CAP, not a fixed wait — the
/// guest force-reboots when the drain completes, so over-sizing costs nothing
/// on the success path and only bounds a genuinely hung guest.
pub(crate) const PROBE_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// Build and configure the auto-repro VM builder: resolve staged
/// schedulers, construct the base builder, point the failure-dump sink
/// at the `.repro` sibling path, enable the dual-snapshot freeze
/// coordinator, attach wprof (when requested), wire KernelBuiltin
/// enable/disable commands and monitor thresholds, and union the
/// include files / scheduler args. Returns the configured builder plus
/// the `.repro.failure-dump.json` path (needed to render the
/// failure-dump tail later), or `None` if wprof attach fails.
fn build_repro_vm_builder(
    entry: &KtstrTestEntry,
    kernel: &Path,
    scheduler: Option<&Path>,
    ktstr_bin: &Path,
    topo: Option<&TopoOverride>,
    guest_args: &[String],
) -> Option<(crate::vmm::KtstrVmBuilder, std::path::PathBuf)> {
    let cmdline_extra = super::runtime::build_cmdline_extra(entry);

    let (vm_topology, memory_mib) = super::runtime::resolve_vm_topology(entry, topo);

    let no_perf_mode = super::runtime::no_perf_mode_for_entry(entry);
    // Resolve staged schedulers for the auto-repro VM so any
    // scheduler-lifecycle ops in the replayed scenario can find
    // their staged binaries at the same /staging/schedulers/<name>/
    // paths the primary VM used. See crate::test_support::eval for the
    // resolve-loop rationale + KernelBuiltin/Eevdf skip semantics.
    //
    // Resolution errors here log + skip rather than propagate: the
    // auto-repro function returns Option<String> (best-effort), so
    // a staging-resolve failure for one staged scheduler should not
    // tear down the whole auto-repro path. The operator still gets
    // the warn in tracing; the primary VM's failure already landed
    // its own dump.
    let mut resolved_staged: Vec<(String, std::path::PathBuf, Vec<String>)> = Vec::new();
    for staged in entry.staged_schedulers {
        match super::eval::resolve_scheduler(&staged.binary) {
            Ok((Some(host_path), _src)) => {
                resolved_staged.push((
                    staged.name.to_string(),
                    host_path,
                    staged.sched_args.iter().map(|s| s.to_string()).collect(),
                ));
            }
            Ok((None, _)) => {} // KernelBuiltin / Eevdf — no binary
            Err(e) => {
                tracing::warn!(
                    staged_name = %staged.name,
                    error = %e,
                    "auto-repro: failed to resolve staged scheduler binary; skipping (Op::AttachScheduler / Op::ReplaceScheduler against this staged entry will fail at dispatch time in the repro VM)"
                );
            }
        }
    }
    let mut builder = super::runtime::build_vm_builder_base(
        entry,
        kernel,
        ktstr_bin,
        scheduler,
        &resolved_staged,
        vm_topology,
        memory_mib,
        &cmdline_extra,
        guest_args,
        no_perf_mode,
    );

    // The repro VM has a post-trigger probe-drain tail the primary's budget does
    // not cover (probe_data map readout + arg-payload serialize + PROBE_OUTPUT_END
    // flush over the bulk port + force_reboot). Override the base host deadline to
    // add a bounded drain grace so the watchdog cannot fire mid-flush and truncate
    // the captured-arg payload (which the host would then fail to parse and drop
    // as "auto-repro: no probe data"). See PROBE_DRAIN_GRACE.
    builder = builder.timeout(
        super::runtime::vm_timeout_from_entry(entry, vm_topology.total_cpus()) + PROBE_DRAIN_GRACE,
    );

    // Set the auto-repro failure-dump sink to a `.repro` sibling
    // of the primary's `{name}-{variant_hash}.failure-dump.json` so the auto-repro
    // VM's dump (if it fires again) lands alongside, not on top of,
    // the just-failed primary's dump. Both files survive in the
    // sidecar dir for primary-vs-repro comparison. The setter is
    // pure (no FS side effects); the repro path's stale-file
    // pre-clear happened earlier at `test_support::eval`'s
    // primary dispatch (which clears BOTH the primary AND the
    // repro path before the primary VM boots). This setter call
    // is therefore the only `failure_dump_path` invocation on the
    // auto-repro path: `build_vm_builder_base` deliberately does
    // not attach a path, so the primary dump is never touched
    // during auto-repro.
    // Same authoritative variant hash the primary uses (same entry +
    // resolved topology + work_type), so the repro dump is variant-keyed
    // to match its sidecar and a gauntlet preset's repro dump doesn't
    // clobber a sibling's.
    let variant_hash = super::sidecar::variant_hash_from_parts(
        entry,
        &vm_topology,
        &super::args::current_work_type(),
    );
    let repro_dump_path = super::sidecar::sidecar_dir().join(format!(
        "{}-{variant_hash:016x}.repro.failure-dump.json",
        entry.name
    ));
    builder = builder.failure_dump_path(&repro_dump_path);

    // Repro VM gets the dual-snapshot freeze coordinator. The
    // primary VM keeps the single-snapshot path (the primary's
    // failure-dump.json schema is `FailureDumpReport`, not
    // wrapped); flipping this on the repro builder is what tells
    // the freeze coord to run the per-CPU `runnable_at` scanner
    // and emit a `DualFailureDumpReport` at the repro path. The
    // gate is a builder field rather than a path-string match so
    // a future caller invoking the repro logic with a different
    // `.failure_dump_path()` keeps working without surprise.
    builder = builder
        .dual_snapshot(true)
        .performance_mode(entry.performance_mode);

    #[cfg(feature = "wprof")]
    {
        builder = match crate::test_support::runtime::attach_wprof_if_requested(
            builder,
            entry,
            "auto-repro",
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ktstr_test: {e:#}");
                return None;
            }
        };
    }

    if let crate::test_support::entry::SchedulerSpec::KernelBuiltin { enable, disable } =
        &entry.scheduler.binary
    {
        builder = builder.sched_enable_cmds(enable);
        builder = builder.sched_disable_cmds(disable);
    }

    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    if entry.scheduler.has_bpf_scheduler() {
        builder = builder.monitor_thresholds(merged_assert.monitor_thresholds());
    }

    {
        let mut args: Vec<String> = Vec::new();

        let declarative_specs: Vec<std::path::PathBuf> = entry
            .all_include_files()
            .into_iter()
            .map(std::path::PathBuf::from)
            .collect();
        let mut resolved_includes: Vec<(String, std::path::PathBuf, &'static str)> =
            if declarative_specs.is_empty() {
                Vec::new()
            } else {
                match crate::cli::resolve_include_files(&declarative_specs) {
                    Ok(v) => v.into_iter().map(|(a, h)| (a, h, "declarative")).collect(),
                    Err(e) => {
                        eprintln!("ktstr_test: auto-repro: include_files resolve: {e:#}");
                        Vec::new()
                    }
                }
            };
        if let Some((archive_path, host_path, guest_path)) = config_file_parts(entry) {
            resolved_includes.push((archive_path, host_path, "scheduler config_file"));
            args.push("--config".to_string());
            args.push(guest_path);
        }
        if let Some((archive_path, host_path, _guest_path, cfg_args)) = config_content_parts(entry)
        {
            resolved_includes.push((archive_path, host_path, "inline config_content"));
            args.extend(cfg_args);
        }
        match super::eval::dedupe_include_files(&resolved_includes) {
            Ok(unioned) if !unioned.is_empty() => {
                builder = builder.include_files(unioned);
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("ktstr_test: auto-repro: include_files dedupe: {e:#}");
            }
        }

        super::runtime::append_base_sched_args(entry, &mut args);
        if !args.is_empty() {
            builder = builder.sched_args(&args);
        }
    }

    Some((builder, repro_dump_path))
}

/// Attempt auto-repro: extract stack functions from the repro VM's scheduler
/// log or COM1 kernel console (fallback), boot a second VM with BPF probes
/// attached, and return formatted probe data. When no stack functions are
/// available (e.g. BPF text error without backtrace), falls back to
/// dynamic BPF program discovery in the repro VM.
/// `console_output` is COM1 kernel console text, used when the scheduler log
/// has no extractable functions (e.g. scheduler died before writing output).
///
/// Returns `None` if repro cannot be attempted or yields no data.
///
/// `too_many_arguments` allow: each parameter is independent test
/// fixture state (test entry, kernel/scheduler/ktstr binaries, the
/// two console captures, optional topology override). Bundling into
/// a struct would build a struct used at exactly one call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attempt_auto_repro(
    entry: &KtstrTestEntry,
    kernel: &Path,
    scheduler: Option<&Path>,
    ktstr_bin: &Path,
    first_vm_output: &str,
    console_output: &str,
    topo: Option<&TopoOverride>,
    primary_exit_kind: Option<u64>,
    primary_reached_workload: bool,
) -> Option<String> {
    use crate::probe::stack::extract_stack_functions_all;

    // Whole-function timer. Per-phase markers below sit at the boundaries
    // identified by the perf-repro investigation: VM build, VM run, and
    // the post-run formatting tail. Without these the 60s claim is
    // unverifiable from a single test invocation, and any future
    // regression in a single phase is invisible against the aggregate.
    let auto_repro_start = Instant::now();

    // Extract scheduler log from the repro VM's captured output.
    let has_sched_start = first_vm_output.contains(SCHED_OUTPUT_START);
    let has_sched_end = first_vm_output.contains(SCHED_OUTPUT_END);
    eprintln!(
        "ktstr_test: auto-repro: captured-output length={} has_sched_start={has_sched_start} has_sched_end={has_sched_end}",
        first_vm_output.len(),
    );
    // `parse_sched_output_partial` accepts a missing SCHED_OUTPUT_END
    // (scheduler crashed mid-run, never wrote the closing delimiter)
    // and falls back to the slice from SCHED_OUTPUT_START to end of
    // buffer. Discarding partial scheduler-log output and skipping straight to
    // COM1 would lose the crash stack the probe pipeline needs.
    let sched_output = parse_sched_output_partial(first_vm_output);

    // Extract function names from the scheduler log first, then
    // fall back to COM1 kernel console (which has kernel backtraces
    // including sched_ext_dump output).
    let stack_funcs = if let Some(sched) = sched_output {
        let funcs = extract_stack_functions_all(sched);
        if funcs.is_empty() {
            if has_sched_start && !has_sched_end {
                eprintln!(
                    "ktstr_test: auto-repro: no functions from partial scheduler log \
                     (missing SCHED_OUTPUT_END), trying COM1",
                );
            } else {
                eprintln!("ktstr_test: auto-repro: no functions from scheduler log, trying COM1");
            }
            extract_stack_functions_all(console_output)
        } else {
            if has_sched_start && !has_sched_end {
                eprintln!(
                    "ktstr_test: auto-repro: extracted {} functions from partial scheduler log \
                     (missing SCHED_OUTPUT_END)",
                    funcs.len(),
                );
            }
            funcs
        }
    } else {
        eprintln!(
            "ktstr_test: auto-repro: no scheduler output on the scheduler-log channel, trying COM1"
        );
        extract_stack_functions_all(console_output)
    };
    let func_names: Vec<String> = stack_funcs.iter().map(|f| f.raw_name.clone()).collect();

    // Stall exits have no causal task — the probe event chain is
    // always empty after stitch. Skip probe attachment entirely to
    // avoid the BPF discovery + kprobe/fentry attach overhead. The
    // repro VM still boots for the failure dump and diagnostic tails.
    let is_stall = primary_exit_kind == Some(crate::probe::scx_defs::EXIT_ERROR_STALL);

    // When no stack functions were extracted (e.g. BPF text error with no
    // backtrace), still boot the repro VM. The guest-side discover_bpf_symbols()
    // dynamically finds the scheduler's BPF programs. Pass a sentinel value
    // so extract_probe_stack_arg returns Some and the guest probe path activates.
    // Host-authoritative variant hash (same entry + resolved topology +
    // work_type as the primary VM → same hash). Threaded to the guest so
    // its in-VM Ctx repro paths match the host's, AND stamped onto
    // repro_result after the run (below) so the host-side repro sidecar
    // writers and the expect_auto_repro inversion resolve the same
    // `-{hash}` artifact names. resolve_vm_topology also runs inside
    // build_repro_vm_builder; it is a pure deterministic fn of the same
    // inputs, so the two computations agree.
    let (vm_topology, _) = super::runtime::resolve_vm_topology(entry, topo);
    let variant_hash = super::sidecar::variant_hash_from_parts(
        entry,
        &vm_topology,
        &super::args::current_work_type(),
    );
    let mut guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
        format!("--ktstr-variant-hash={variant_hash:016x}"),
    ];
    if !is_stall {
        let probe_arg = if func_names.is_empty() {
            eprintln!(
                "ktstr_test: auto-repro: no stack functions, using BPF discovery in repro VM"
            );
            format!("--ktstr-probe-stack={DISCOVER_SENTINEL}")
        } else {
            eprintln!(
                "ktstr_test: auto-repro: probing {} functions in second VM",
                func_names.len()
            );
            format!("--ktstr-probe-stack={}", func_names.join(","))
        };
        guest_args.push(probe_arg);
    } else {
        eprintln!("ktstr_test: auto-repro: stall exit — skipping probe attachment");
    }

    let (builder, repro_dump_path) =
        build_repro_vm_builder(entry, kernel, scheduler, ktstr_bin, topo, &guest_args)?;

    // VM build phase: KVM create, vCPU pinning, virtio device setup,
    // freeze-coord arming, ELF/BTF parses for monitor accessors. Any
    // regression here shows as a wider gap between the start of
    // attempt_auto_repro and the start of the run phase.
    let build_start = Instant::now();
    let vm = match builder.build() {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("ktstr_test: auto-repro: failed to build VM: {e:#}");
            tracing::info!(
                elapsed_ms = auto_repro_start.elapsed().as_millis() as u64,
                outcome = "build_failed",
                "auto_repro: total",
            );
            return None;
        }
    };
    tracing::info!(
        elapsed_ms = build_start.elapsed().as_millis() as u64,
        "auto_repro: vm_build",
    );

    // VM run phase: full guest lifecycle (boot, sched attach, Phase A
    // probe attach, Phase B BPF discovery + fentry attach, workload,
    // cleanup, virtio-console drain). The dominant share of auto-repro
    // wall time lives here; per-phase guest-side breakdown is emitted
    // by the probe pipeline inside the guest itself.
    let run_start = Instant::now();
    let mut repro_result = match vm.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ktstr_test: auto-repro: VM run failed: {e:#}");
            tracing::info!(
                elapsed_ms = auto_repro_start.elapsed().as_millis() as u64,
                outcome = "run_failed",
                "auto_repro: total",
            );
            return None;
        }
    };
    tracing::info!(
        elapsed_ms = run_start.elapsed().as_millis() as u64,
        guest_duration_ms = repro_result.duration.as_millis() as u64,
        "auto_repro: vm_run",
    );
    drop(vm);

    // Stamp the host-authoritative variant hash onto the repro result.
    // vm.run() builds the VmResult entry-agnostic (variant_hash=0), just
    // as the primary path does before the eval layer stamps it
    // (eval/mod.rs). Without this the repro sidecar writers + the
    // expect_auto_repro inversion would resolve `-0000000000000000`
    // names while the artifact carries the real `-{hash}`.
    repro_result.variant_hash = variant_hash;

    format_repro_output(
        entry,
        &repro_result,
        is_stall,
        kernel,
        primary_reached_workload,
        auto_repro_start,
        &repro_dump_path,
    )
}

/// Render the auto-repro VM's output: write its sidecar artifacts,
/// extract + format the probe section (or the crash-reproduction
/// verdict when probe data is absent), and append the diagnostic tails
/// (scheduler log, sched_ext dump, failure-dump JSON, dmesg). Returns
/// `None` when neither probe data nor any tail is available.
#[allow(clippy::too_many_arguments)]
fn format_repro_output(
    entry: &KtstrTestEntry,
    repro_result: &crate::vmm::result::VmResult,
    is_stall: bool,
    kernel: &Path,
    primary_reached_workload: bool,
    auto_repro_start: Instant,
    repro_dump_path: &Path,
) -> Option<String> {
    // Write the auto-repro VM's wprof Perfetto trace BEFORE
    // classify_repro_vm_status consumes the drain for lifecycle
    // signalling. Reassembles the (possibly chunked) trace like the
    // primary VM's `crate::test_support::eval` wprof handler but writes it
    // under `${entry.name}-${variant_hash}.repro.wprof.pb` so primary +
    // repro artifacts coexist on disk (matches the `.repro.` infix every
    // other repro-VM sidecar already uses). Without this hop the auto-repro
    // VM's wprof bytes would be silently dropped — the host already paid the
    // capture cost in the guest, throwing the data away would mask exactly
    // the bug the auto-repro VM was booted to reproduce. wprof-gated: the
    // only sidecar written here is the wprof trace (profraw from the repro
    // run is persisted centrally by `crate::vmm::KtstrVm::run`, not here).
    #[cfg(feature = "wprof")]
    write_auto_repro_sidecar_artifacts(entry, repro_result);

    // Forward guest stderr (COM1) and bulk-port stdout probe lines when verbose.
    if verbose() {
        eprintln!(
            "ktstr_test: auto-repro: COM1 stderr length={} stdout length={}",
            repro_result.stderr.len(),
            repro_result.output.len(),
        );
        for line in repro_result.stderr.lines() {
            eprintln!("  repro-vm-com1: {line}");
        }
        let mut in_probe = false;
        for line in repro_result.output.lines() {
            if line.contains("ktstr_test: probe:") {
                in_probe = true;
            }
            if in_probe {
                eprintln!("  repro-vm-stdout: {line}");
            }
        }
    }

    // Extract probe JSON from the repro VM and format on the host with
    // kernel_dir so blazesym can resolve source locations via vmlinux
    // DWARF. derive_kernel_dir handles both build-tree and cache-entry
    // layouts; for Local cache entries whose source tree is still on
    // disk, prefer_source_tree_for_dwarf re-routes blazesym to the
    // unstripped vmlinux in the source tree. Tarball/git cache entries
    // still can't recover file:line — stripped cache vmlinux is all
    // we have.
    let kernel_dir = crate::kernel_path::derive_kernel_dir(kernel)
        .map(|dir| crate::cache::prefer_source_tree_for_dwarf(&dir).unwrap_or(dir))
        .and_then(|p| p.to_str().map(String::from));
    let kernel_dir_str = kernel_dir.as_deref();
    // Sibling of `repro_dump_path` for the truncated-payload sink: when
    // the repro VM dies mid-`println!` of the probe payload (e.g. KVM
    // EFAULT, panic-before-PROBE_OUTPUT_END), `extract_probe_output`
    // writes the raw extracted JSON here so the operator can inspect
    // the truncated bytes. Pre-clear so a previous run's stale
    // partial doesn't get mistaken for this run's output.
    let probe_payload_partial_path = super::sidecar::sidecar_dir()
        .join(format!("{}.repro.probe-payload.partial.json", entry.name));
    let _ = std::fs::remove_file(&probe_payload_partial_path);
    let probe_section = if is_stall {
        tracing::debug!(
            "auto-repro: suppressing chain-to-failure for stall exit \
             (no causal task — probe events are always empty after stitch)",
        );
        None
    } else {
        extract_probe_output(
            &repro_result.output,
            kernel_dir_str,
            Some(probe_payload_partial_path.as_path()),
        )
    };

    // Build diagnostic tails from the repro VM's output.
    const REPRO_TAIL_LINES: usize = 40;

    let sched_log_tail = parse_sched_output(&repro_result.output).and_then(|log| {
        let collapsed = crate::verifier::collapse_cycles(log);
        format_tail(&collapsed, REPRO_TAIL_LINES, "repro VM scheduler log")
    });

    let dump_tail = extract_sched_ext_dump(&repro_result.stderr)
        .and_then(|dump| format_tail(&dump, REPRO_TAIL_LINES, "repro VM sched_ext dump"));

    // Filter sched_ext_dump lines from dmesg tail to avoid duplicating
    // the dump section. Only non-dump kernel console lines are shown.
    // See [`render_dmesg_tail`] for the corruption / filter-empty
    // disambiguation policy. The helper always returns a populated
    // string — wrap in `Some` so the `tails` array stays homogeneous
    // with the other tail builders that legitimately return `None`.
    let dmesg_tail = Some(render_dmesg_tail(&repro_result.stderr, REPRO_TAIL_LINES));

    // Inline-render the freeze coordinator's failure-dump JSON when
    // present. The freeze-coord writes a `FailureDumpReport` /
    // `DualFailureDumpReport` to `repro_dump_path` if any error-class
    // SCX exit (or the dual-snapshot half-way trigger) fired during
    // the repro VM run. Surfacing the Display rendering inline means a
    // CLI user sees the BPF map state, vCPU regs, and (for dual)
    // early/late jiffies metadata in the same tail block as the
    // sched_ext_dump and dmesg — no need to chase the separate
    // `.repro.failure-dump.json` sibling for the at-a-glance view.
    let failure_dump_tail = render_failure_dump_file(repro_dump_path);

    let tails: Vec<String> = [sched_log_tail, dump_tail, failure_dump_tail, dmesg_tail]
        .into_iter()
        .flatten()
        .collect();

    if probe_section.is_none() && tails.is_empty() {
        tracing::info!(
            elapsed_ms = auto_repro_start.elapsed().as_millis() as u64,
            outcome = "no_data",
            "auto_repro: total",
        );
        return None;
    }

    let has_probe = probe_section.is_some();
    let mut out = probe_section.unwrap_or_default();

    // Crash reproduction status when probe data is absent. Wrap the
    // verdict in a "PRIMARY DID NOT REACH WORKLOAD" prefix when the
    // primary VM failed before the workload could fire — without it,
    // a clean repro run reads as "bug is gone" when reality is "the
    // bug was never exercised on either run." The prefix preserves
    // the repro VM's own diagnostic on the following line so operators
    // can still see whether the repro VM itself booted successfully.
    if !has_probe {
        let verdict = classify_repro_vm_status(
            repro_result.timed_out,
            repro_result.crash_message.is_some(),
            repro_result.exit_code,
            repro_result.guest_messages.as_ref(),
        );
        out.push_str(&label_repro_verdict_when_workload_not_reached(
            primary_reached_workload,
            &verdict,
        ));
    }

    // Duration line before tails.
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(&format!(
        "repro VM duration: {:.1}s",
        repro_result.duration.as_secs_f64(),
    ));

    for tail in &tails {
        out.push_str("\n\n");
        out.push_str(tail);
    }
    tracing::info!(
        elapsed_ms = auto_repro_start.elapsed().as_millis() as u64,
        outcome = if has_probe {
            "probe_data"
        } else {
            "tails_only"
        },
        "auto_repro: total",
    );
    Some(out)
}

/// Extract probe JSON from the guest's bulk-port stdout, deserialize, and format on the
/// host where vmlinux (DWARF) is available for source locations.
///
/// `partial_dump_path` (when `Some`) names a file under the sidecar
/// dir where the raw extracted JSON gets written if deserialization
/// fails for any reason — truncation, syntax, or schema mismatch.
/// Truncation in particular is expected when the repro VM dies mid-
/// `println!` of the probe payload (e.g. KVM EFAULT, panic before
/// `PROBE_OUTPUT_END`); the partial-dump file lets the operator
/// inspect whatever the guest managed to write.
pub(crate) fn extract_probe_output(
    output: &str,
    kernel_dir: Option<&str>,
    partial_dump_path: Option<&Path>,
) -> Option<String> {
    let json = crate::probe::output::extract_section(output, PROBE_OUTPUT_START, PROBE_OUTPUT_END);
    if json.is_empty() {
        return None;
    }
    let payload = parse_probe_payload(&json, partial_dump_path)?;
    let mut out = String::new();

    // Append pipeline diagnostics if present.
    if let Some(ref diag) = payload.diagnostics {
        out.push_str(&format_probe_diagnostics(&diag.pipeline, &diag.skeleton));
    }

    if payload.events.is_empty() {
        if out.is_empty() {
            return None;
        }
        return Some(out);
    }
    out.push_str(&crate::probe::output::format_probe_events_with_bpf_locs(
        &payload.events,
        &payload.func_names,
        kernel_dir,
        &payload.bpf_source_locs,
        payload.nr_cpus,
        &payload.param_names,
        &payload.render_hints,
    ));
    Some(out)
}

/// Try to deserialize the probe payload, recovering as much data as
/// possible when the guest truncated mid-write.
///
/// Strategy:
/// 1. Strict `serde_json::from_str::<ProbeBytes>` succeeds → return
///    the full payload.
/// 2. EOF / truncation error (`Category::Eof`) → walk the JSON,
///    locate the `"events":[ ... ]` array, and parse leading
///    `ProbeEvent` objects until one fails to deserialize. Return a
///    `ProbeBytes` whose `events` field carries the recovered events
///    and every other field is empty / `None`. Empty `func_names`
///    means the formatter will print `unknown` for func names, which
///    is degraded but still surfaces ts/args/fields/kstack.
/// 3. Any non-EOF deserialize error (syntax, schema mismatch) →
///    return `None`.
///
/// On any failure path, when `partial_dump_path` is `Some` the raw
/// extracted JSON is written there verbatim so the operator can
/// inspect the truncated bytes manually. Write errors are logged but
/// do not affect the return value.
fn parse_probe_payload(json: &str, partial_dump_path: Option<&Path>) -> Option<ProbeBytes> {
    match serde_json::from_str::<ProbeBytes>(json) {
        Ok(payload) => Some(payload),
        Err(e) => {
            // Surface byte position + total length so the operator
            // sees how far the guest got before truncation.
            let total_len = json.len();
            let category = if e.is_eof() { "truncated" } else { "malformed" };
            eprintln!(
                "ktstr_test: probe payload {category}: {e} \
                 (line {}, column {}, total {total_len} bytes)",
                e.line(),
                e.column(),
            );
            // Always persist the raw payload when we have a sink, so
            // the operator can grep / re-parse / diff against the
            // emitter format. We write before attempting recovery:
            // recovery may itself fail, but the raw bytes are the
            // ground truth either way.
            if let Some(path) = partial_dump_path {
                match std::fs::write(path, json) {
                    Ok(()) => eprintln!(
                        "ktstr_test: probe payload: wrote raw bytes to {}",
                        path.display(),
                    ),
                    Err(write_err) => eprintln!(
                        "ktstr_test: probe payload: failed to write raw bytes to {}: {write_err}",
                        path.display(),
                    ),
                }
            }
            if !e.is_eof() {
                return None;
            }
            // EOF path: try to salvage events.
            let recovered = recover_partial_events(json);
            if recovered.is_empty() {
                return None;
            }
            eprintln!(
                "ktstr_test: probe payload: recovered {} event(s) from truncated input",
                recovered.len(),
            );
            Some(ProbeBytes {
                events: recovered,
                func_names: Vec::new(),
                bpf_source_locs: Default::default(),
                diagnostics: None,
                nr_cpus: None,
                param_names: Default::default(),
                render_hints: Default::default(),
            })
        }
    }
}

/// Walk a (possibly truncated) `ProbeBytes` JSON payload and parse
/// as many leading `ProbeEvent` objects from the `events` array as
/// will deserialize cleanly. Stops at the first parse failure or
/// when the array's closing `]` is reached.
///
/// The walk is depth-tracking byte scanning rather than full JSON
/// parsing: we locate `"events":` then `[`, and from there split the
/// remainder into balanced `{...}` chunks (honoring string escapes
/// so braces inside strings don't unbalance the count). Each chunk
/// is fed to `serde_json::from_str::<ProbeEvent>` — a partial
/// trailing event therefore gets dropped at the splitter (`None`
/// from `find_balanced_object_end`), or, if it is balanced but
/// internally malformed, at the `from_str` step.
fn recover_partial_events(json: &str) -> Vec<crate::probe::process::ProbeEvent> {
    // Locate the start of the events array. ProbeBytes is serialized
    // with `events` as the first field, so this match is at the
    // beginning of the payload in normal output, but `find` is robust
    // to any field order serde_json might choose in the future.
    let key = "\"events\":";
    let Some(key_idx) = json.find(key) else {
        return Vec::new();
    };
    let after_key = &json[key_idx + key.len()..];
    let Some(open_offset) = after_key.find('[') else {
        return Vec::new();
    };
    let mut events = Vec::new();
    let mut cur = &after_key[open_offset + 1..];
    loop {
        cur = cur.trim_start();
        // Optional comma between elements (not before the first).
        if let Some(rest) = cur.strip_prefix(',') {
            cur = rest.trim_start();
        }
        // Closing bracket means a clean array end — no truncation.
        // Empty / non-`{` head means we hit the truncation boundary
        // mid-event (or before the first event was emitted).
        if cur.is_empty() || cur.starts_with(']') || !cur.starts_with('{') {
            break;
        }
        let Some(end) = find_balanced_object_end(cur) else {
            break;
        };
        let chunk = &cur[..end];
        let Ok(ev) = serde_json::from_str::<crate::probe::process::ProbeEvent>(chunk) else {
            break;
        };
        events.push(ev);
        cur = &cur[end..];
    }
    events
}

/// Return the byte length of the leading balanced `{...}` object in
/// `s`, or `None` if `s` does not start with `{` or the object is
/// truncated. Honors JSON string escaping so quoted braces don't
/// unbalance the depth count.
///
/// Used only by [`recover_partial_events`]; the input is the
/// remainder of an events-array body, so the first non-whitespace
/// byte is `{` for any non-truncated event.
fn find_balanced_object_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth: u32 = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                // depth started at 0, was bumped to 1 by the leading
                // `{`, so depth==1 here means the matching close.
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Classify the "events captured but 0 after stitch" failure mode.
/// Returns a static description of WHY every event was dropped in
/// stitch, grounded in BPF-side counters (`bpf_trigger_fires` +
/// `bpf_exit_kind_snap`) so the explanation is causal, not
/// speculative.
///
/// Branch order matches the failure-mode taxonomy:
///   1. `bpf_trigger_fires == 0` — the `tp_btf/sched_ext_exit`
///      handler never executed. Either the scheduler clean-exited
///      (kind < SCX_EXIT_ERROR, handler early-returns at line 565
///      of probe.bpf.c) or the scheduler crashed before reaching
///      the tracepoint at all.
///   2. `bpf_trigger_fires > 0 && exit_kind_snap == ERROR_STALL` —
///      the handler fired but skipped the ringbuf submit (probe.bpf.c
///      line 699 explicit early-return for STALL). target_tptr is
///      None at the host, run_probe_skeleton suppresses the chain.
///   3. `bpf_trigger_fires > 0 && exit_kind_snap == ERROR (generic)` —
///      handler fired but `args[0] = 0` because the generic ERROR
///      exit can come from kworker context where `current` is the
///      worker thread, not the causal task.
///   4. `bpf_trigger_fires > 0 && exit_kind_snap == ERROR_BPF` —
///      handler fired with a real causal task in `args[0]`, but
///      stitch matched no events. Suspected `func_idx_offset` bug
///      or ID mismatch between Phase A and Phase B.
///   5. fallback — kind value we don't recognize (future kernel
///      version or value not yet wired up); surface the raw value
///      so the operator can map it via include/linux/sched/ext.h.
fn stitch_drop_cause(
    skeleton: &crate::probe::process::ProbeDiagnostics,
) -> std::borrow::Cow<'static, str> {
    use crate::probe::scx_defs::{EXIT_ERROR, EXIT_ERROR_BPF, EXIT_ERROR_STALL};
    if skeleton.bpf_trigger_fires == 0 {
        return std::borrow::Cow::Borrowed(
            "trigger never fired (timing race or scheduler clean-exited; \
             no error-class sched_ext_exit observed)",
        );
    }
    match skeleton.bpf_exit_kind_snap as u64 {
        EXIT_ERROR_STALL => {
            "trigger fired with kind=STALL (no causal task; pre-trigger events \
             suppressed because watchdog-context exit lacks a current task)"
        }
        EXIT_ERROR => {
            "trigger fired with kind=ERROR (no current task at exit time; \
             pre-trigger events suppressed because generic ERROR can fire \
             from kworker context where `current` is not the causal task)"
        }
        EXIT_ERROR_BPF => {
            "trigger fired with kind=BPF_ERROR but stitch found no matching \
             task_ptr (suspected ID mismatch or func_idx_offset bug — file a ticket)"
        }
        other => {
            return format!(
                "trigger fired but exit kind {other} is unrecognized; \
                 pre-trigger events suppressed because no causal task \
                 was identified (map value via include/linux/sched/ext.h)"
            )
            .into();
        }
    }
    .into()
}

/// Format probe pipeline diagnostics into a human-readable summary.
pub(crate) fn format_probe_diagnostics(
    pipeline: &PipelineDiagnostics,
    skeleton: &crate::probe::process::ProbeDiagnostics,
) -> String {
    let mut out = String::new();
    out.push_str("--- probe pipeline ---\n");

    // Stage 1: extraction
    out.push_str(&format!(
        "  extracted:   {} functions from crash backtrace\n",
        pipeline.stack_extracted,
    ));

    // Stage 2: filter
    //
    // Invariant from construction (see start_probe_phase_a /
    // ProbeHandle setup): every name in `filter_dropped` is a member
    // of the original `raw_functions` whose count became
    // `stack_extracted`, so `filter_dropped.len() <= stack_extracted`.
    // `PipelineDiagnostics` is serde-serialized into the probe payload (bulk-port stdout) though, and
    // the format runs on the failure-reporting path. `saturating_sub`
    // keeps a corrupt or partial payload from masking the real failure
    // with a subtract-with-overflow panic during diagnostic rendering.
    let passed = (pipeline.stack_extracted as usize).saturating_sub(pipeline.filter_dropped.len());
    if pipeline.filter_dropped.is_empty() {
        out.push_str(&format!("  traceable:   {passed} passed filter\n"));
    } else {
        out.push_str(&format!(
            "  traceable:   {passed} passed, {} dropped: {}\n",
            pipeline.filter_dropped.len(),
            pipeline.filter_dropped.join(", "),
        ));
    }

    // Stage 3: BPF discovery
    out.push_str(&format!(
        "  bpf_discover: {} programs found\n",
        pipeline.bpf_discovered,
    ));

    // Stage 4: expansion
    out.push_str(&format!(
        "  after_expand: {} total probe targets\n",
        pipeline.total_after_expand,
    ));

    // Stage 5: kprobe attach
    if skeleton.kprobe_attach_failed.is_empty() {
        out.push_str(&format!(
            "  kprobes:     {} attached\n",
            skeleton.kprobe_attached,
        ));
    } else {
        out.push_str(&format!(
            "  kprobes:     {} attached, {} failed: {}\n",
            skeleton.kprobe_attached,
            skeleton.kprobe_attach_failed.len(),
            skeleton
                .kprobe_attach_failed
                .iter()
                .map(|(n, e)| format!("{n} ({e})"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    if !skeleton.kprobe_resolve_failed.is_empty() {
        out.push_str(&format!(
            "  kprobe_miss: {} unresolved: {}\n",
            skeleton.kprobe_resolve_failed.len(),
            skeleton.kprobe_resolve_failed.join(", "),
        ));
    }

    // Stage 6: fentry attach
    if skeleton.fentry_candidates > 0 {
        if skeleton.fentry_attach_failed.is_empty() {
            out.push_str(&format!(
                "  fentry:      {} attached\n",
                skeleton.fentry_attached,
            ));
        } else {
            out.push_str(&format!(
                "  fentry:      {} attached, {} failed: {}\n",
                skeleton.fentry_attached,
                skeleton.fentry_attach_failed.len(),
                skeleton
                    .fentry_attach_failed
                    .iter()
                    .map(|(n, e)| format!("{n} ({e})"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }

    // Stage 7: trigger
    let trigger_type = if skeleton.trigger_type.is_empty() {
        "unknown"
    } else {
        &skeleton.trigger_type
    };
    if let Some(ref err) = skeleton.trigger_attach_error {
        out.push_str(&format!("  trigger:     attach failed ({err})\n"));
    } else {
        out.push_str(&format!(
            "  trigger:     {} ({})\n",
            if skeleton.trigger_fired {
                "fired"
            } else {
                "not fired"
            },
            trigger_type,
        ));
    }
    if let Some(ref panic_msg) = skeleton.host_thread_panic {
        // Render at the top of stage-7 output so an operator
        // grepping the probe summary sees the panic line before any
        // counter the panicking thread might have published mid-run.
        // Terminal failure: every other stat below this line is
        // suspect because the producer thread did not finish.
        out.push_str(&format!(
            "  ERROR:       probe-collection thread panicked: {panic_msg}\n"
        ));
    }

    // Stage 8: capture
    out.push_str(&format!(
        "  probe_data:  {} keys, {} unmatched IPs\n",
        skeleton.probe_data_keys, skeleton.probe_data_unmatched_ips,
    ));

    // Stage 9: events + stitching
    //
    // When events disappear during stitch (`events_before_stitch > 0`
    // but `events_after_stitch == 0`), surface a CONCRETE causal
    // explanation. The previous renderer emitted the bare counter
    // pair, leaving the operator to guess WHY the events dropped:
    //   - trigger never fired (no error-class exit, scheduler clean
    //     exit, or scheduler crashed before the tracepoint fired)
    //   - trigger fired with kind=STALL → BPF handler returns early
    //     without submitting a ringbuf event, so target_tptr is None
    //     and `run_probe_skeleton` suppresses the unstitched chain
    //   - trigger fired with kind=ERROR (generic) → args[0] = 0
    //     because exit can fire from kworker context where `current`
    //     is the worker thread, not the causal task
    //   - trigger fired with kind=ERROR_BPF but stitch found no
    //     matching task_ptr → suspected ID mismatch or
    //     func_idx_offset bug
    // Each branch reads `bpf_trigger_fires` and `bpf_exit_kind_snap`
    // from the skeleton diag, already plumbed via the BSS-read site
    // in `run_probe_skeleton`, so the explanation is grounded in
    // BPF-side ground truth, not host-side speculation.
    out.push_str(&format!(
        "  events:      {} captured, {} after stitch",
        skeleton.events_before_stitch, skeleton.events_after_stitch,
    ));
    if skeleton.events_before_stitch > 0 && skeleton.events_after_stitch == 0 {
        let cause = stitch_drop_cause(skeleton);
        out.push_str(" — ");
        out.push_str(&cause);
    } else if skeleton.stitch_fallback_used {
        // Best-effort fallback path: events after stitch is non-zero
        // but the chain was grouped by task_ptr frequency rather than
        // matched against a verified trigger task pointer. Mark the
        // output explicitly so the operator does not mistake the
        // candidate chain for a verified stitch.
        out.push_str(" — trigger absent, grouped by task_ptr frequency (best-effort)");
    }
    out.push('\n');

    // Stage 10: BPF-side counters
    if skeleton.bpf_kprobe_fires > 0
        || skeleton.bpf_trigger_fires > 0
        || skeleton.bpf_meta_misses > 0
    {
        out.push_str(&format!(
            "  bpf_counts:  {} kprobe fires, {} trigger fires, {} meta misses\n",
            skeleton.bpf_kprobe_fires, skeleton.bpf_trigger_fires, skeleton.bpf_meta_misses,
        ));
        if !skeleton.bpf_miss_ips.is_empty() {
            let ips: Vec<String> = skeleton
                .bpf_miss_ips
                .iter()
                .map(|ip| format!("0x{ip:x}"))
                .collect();
            out.push_str(&format!("  miss_ips:    {}\n", ips.join(", ")));
        }
    }

    out
}

/// Guest-side dispatch: check for `--ktstr-test-fn=NAME` in args, run the
/// registered function, write the result to SHM and stdout (bulk port),
/// and exit. Profraw data is flushed via `try_flush_profraw()`
/// inline on both the success and failure paths before
/// `std::process::exit()` is invoked.
///
/// Called from `ktstr_test_early_dispatch()` (ctor) before `main()`, or
/// from `ktstr_guest_init()` when running as PID 1.
///
/// When called from PID 1 context, args must be pre-loaded into the
/// process args (the caller reads `/args` from the initramfs).
/// Returns `Some(exit_code)` if dispatched, `None` if not an
/// ktstr_test invocation.
pub(crate) fn maybe_dispatch_vm_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    maybe_dispatch_vm_test_with_args(&args)
}

/// Guest-side scenario context prelude shared by every VM-dispatch
/// entry point in this module.
///
/// `maybe_dispatch_vm_test_with_args` and
/// `maybe_dispatch_vm_test_with_phase_a` both construct a
/// [`crate::scenario::Ctx`] around the same topology / cgroup /
/// sched_pid / assert-merge inputs — the only difference is whether
/// a probe thread is attached. Moving the inputs into a single
/// helper keeps the two dispatch paths in sync so a change to the
/// settle duration, assert merge chain, or sysfs fallback behaviour
/// lands in both without drift.
///
/// Returns `(topo, cgroups, sched_pid, merged_assert)`. The caller
/// owns the returned values for the lifetime of the `Ctx` it builds;
/// `Ctx` fields that borrow from them (`&topo`, `&cgroups`) stay
/// valid until the caller drops this tuple.
fn build_dispatch_ctx_parts(
    entry: &KtstrTestEntry,
    args: &[String],
) -> (
    crate::topology::TestTopology,
    crate::cgroup::CgroupManager,
    Option<libc::pid_t>,
    crate::assert::Assert,
) {
    // Sysfs is ground truth: CPUID, ACPI MADT, and MPTABLE all
    // express the VM's actual topology. Fall back to from_vm_topology
    // only when sysfs read fails.
    let topo = match crate::topology::TestTopology::from_system() {
        Ok(sys) => sys,
        Err(e) => {
            eprintln!("ktstr_test: topology from sysfs failed ({e}), using VM spec fallback");
            crate::topology::TestTopology::from_vm_topology(&entry.topology)
        }
    };
    let cgroup_root = resolve_cgroup_root(args);
    let cgroups = crate::cgroup::CgroupManager::new(&cgroup_root);
    // Setup is deferred to `apply_setup` in the scenario runtime: it
    // walks the test's CgroupDef declarations to compute the controller
    // set the test actually needs, then invokes `cgroups.setup(&controllers)`
    // with that exact set. Calling setup() here would either over-enable
    // controllers (a test that requires the absence of a controller
    // would fail) or under-enable them (the test's set_cpuset/set_memory
    // call would fail with bare ENOENT/EACCES at the knob-write site).
    // Read the scheduler PID from the atomic side channel published by
    // `vmm::rust_init::start_scheduler`. The previous consumer parsed
    // `std::env::var("SCHED_PID")`, which is unsound under the live
    // probe thread spawned by `start_probe_phase_a` — glibc mutates
    // `__environ` without locks, so a concurrent reader vs. writer
    // races. `sched_pid()` returns `None` on the `0` sentinel, matching
    // the `.filter(|&pid| pid != 0)` clause it replaces.
    let sched_pid = crate::vmm::rust_init::sched_pid();
    // Three-layer merge: default_checks → scheduler.assert → entry.assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    (topo, cgroups, sched_pid, merged_assert)
}

/// Like `maybe_dispatch_vm_test` but with explicit args. Used by
/// `ktstr_guest_init()` which reads args from `/args` in the initramfs.
///
/// The caller (`ktstr_guest_init`) must have invoked
/// [`propagate_rust_env_from_cmdline`] before any probe / workload
/// thread was spawned. Doing the env propagation here would race
/// with the probe thread `start_probe_phase_a` spawned in the split
/// path, so it lives in the caller instead.
pub(crate) fn maybe_dispatch_vm_test_with_args(args: &[String]) -> Option<i32> {
    let name = match extract_test_fn_arg(args) {
        Some(n) => n,
        None => {
            tracing::debug!("ktstr-init: no --ktstr-test-fn in args, skipping dispatch");
            return None;
        }
    };

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    // Parse --ktstr-probe-stack=func1,func2,... for auto-repro mode.
    let probe_stack = extract_probe_stack_arg(args);

    // Parse --ktstr-work-type=NAME for work type override.
    let work_type_override = extract_work_type_arg(args).and_then(|s| {
        crate::workload::WorkType::from_name(&s).or_else(|| {
            // `from_name` is exact-match on the PascalCase canonical
            // form. A user typo (`spinwait`, `SPINWAIT`) lands here;
            // call `WorkType::suggest` for the canonical spelling
            // and surface it in the diagnostic so the user doesn't
            // have to guess the correct casing.
            match crate::workload::WorkType::suggest(&s) {
                Some(canonical) => eprintln!(
                    "ktstr_test: unknown work type '{s}'; did you mean \
                     '{canonical}'? Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
                None => eprintln!(
                    "ktstr_test: unknown work type '{s}'. Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
            }
            None
        })
    });

    // Set up BPF probes if --ktstr-probe-stack was provided.
    let pipeline = ProbePipeline::new();
    let probe_stop = pipeline.stop.clone();
    let probe_handle: Option<ProbeHandle> = probe_stack
        .as_ref()
        .and_then(|stack_input| setup_probe_handle(stack_input, &pipeline));

    let (topo, cgroups, sched_pid, merged_assert) = build_dispatch_ctx_parts(entry, args);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .sched_pid(sched_pid)
        .settle(Duration::ZERO)
        .work_type_override(work_type_override)
        .assert(merged_assert)
        .wait_for_map_write(!entry.bpf_map_write.is_empty())
        .entry_name(entry.name)
        // Host-threaded authoritative variant hash (see
        // extract_variant_hash_arg); `0` only if the guest was invoked
        // without it, in which case the entry_name=Some bail does not
        // fire but the path simply carries the 0 suffix — production
        // always injects it (run_ktstr_test_inner_impl).
        .variant_hash(extract_variant_hash_arg(args).unwrap_or(0))
        .build();

    // Send SCENARIO_START so the host-side watchdog resets its hard
    // deadline to `now + workload_duration`. Tests that use
    // `scenario::run()` send this from `ops::run_inner`; tests that
    // call `(entry.func)(&ctx)` directly with `thread::sleep(duration)`
    // do not, so the watchdog's boot-relative deadline can fire mid-
    // test even when the test's own clock has not yet hit duration.
    // Sending it here covers every dispatch path uniformly.
    if crate::vmm::guest_comms::is_guest() {
        crate::vmm::guest_comms::send_scenario_start();
    }

    let mut result = match (entry.func)(&ctx) {
        Ok(r) => r,
        Err(e) => {
            let r = AssertResult::fail_msg(format!("{e:#}"));
            publish_result_and_collect(&r, probe_stop, probe_handle);
            return Some(1);
        }
    };
    // survives_storm: re-check scheduler liveness here so a non-execute_*
    // scenario that hand-rolls Op dispatch (running no in-hold liveness probe)
    // still fails if the scheduler died during the run.
    enforce_survives_storm_liveness(&mut result, entry.survives_storm);

    let exit_code = exit_code_for_result(&result);
    publish_result_and_collect(&result, probe_stop, probe_handle);
    Some(exit_code)
}

/// Build the probe thread + handle for `--ktstr-probe-stack`: load the
/// stack, filter to traceable functions, discover the running
/// scheduler's BPF programs, resolve kernel + BPF BTF signatures,
/// pre-open BPF program FDs, spawn the probe-skeleton thread, and wait
/// for probes to attach before returning the handle. Returns `None`
/// when the stack yields no traceable functions.
fn setup_probe_handle(stack_input: &str, pipeline: &ProbePipeline) -> Option<ProbeHandle> {
    use crate::probe::stack::load_probe_stack;

    eprintln!("ktstr_test: probe: loading probe stack from --ktstr-probe-stack");
    let mut pipe_diag = PipelineDiagnostics::default();
    let raw_functions = load_probe_stack(stack_input);
    pipe_diag.stack_extracted = raw_functions.len() as u32;
    let pre_filter: Vec<String> = raw_functions.iter().map(|f| f.raw_name.clone()).collect();
    let mut functions = crate::probe::stack::filter_traceable(raw_functions);
    // Record which functions were dropped by filter_traceable.
    for name in &pre_filter {
        if !functions.iter().any(|f| f.raw_name == *name) {
            pipe_diag.filter_dropped.push(name.clone());
        }
    }
    // Discover BPF scheduler functions from the running scheduler.
    // Stack-extracted BPF names have stale prog IDs from the first VM;
    // discover_bpf_symbols finds the current scheduler's programs.
    let stack_display_names: Vec<&str> = functions
        .iter()
        .filter(|f| f.is_bpf)
        .map(|f| f.display_name.as_str())
        .collect();
    let bpf_syms = crate::probe::btf::discover_bpf_symbols(&stack_display_names);
    pipe_diag.bpf_discovered = bpf_syms.len() as u32;
    if !bpf_syms.is_empty() {
        eprintln!(
            "ktstr_test: probe: {} BPF symbols discovered",
            bpf_syms.len()
        );
        functions.extend(bpf_syms);
    }
    // Expand BPF functions to kernel-side callers for bridge kprobes,
    // keeping BPF functions for fentry attachment.
    let functions = crate::probe::stack::expand_bpf_to_kernel_callers(functions);
    pipe_diag.total_after_expand = functions.len() as u32;
    if functions.is_empty() {
        eprintln!("ktstr_test: no traceable functions from --ktstr-probe-stack");
        return None;
    }

    eprintln!(
        "ktstr_test: probe: {} functions loaded, spawning probe thread",
        functions.len()
    );

    // Resolve BTF signatures for kernel functions so probe output
    // gets decoded field names instead of raw register values.
    let kernel_names: Vec<&str> = functions
        .iter()
        .filter(|f| !f.is_bpf)
        .map(|f| f.raw_name.as_str())
        .collect();
    let mut btf_funcs = crate::probe::btf::parse_btf_functions(&kernel_names, None);
    // Parse BPF function signatures from BPF program BTF.
    let bpf_btf_args: Vec<(&str, u32)> = functions
        .iter()
        .filter(|f| f.is_bpf)
        .filter_map(|f| Some((f.display_name.as_str(), f.bpf_prog_id?)))
        .collect();
    if !bpf_btf_args.is_empty() {
        btf_funcs.extend(crate::probe::btf::parse_bpf_btf_functions(&bpf_btf_args));
    }

    // Build func_names from the filtered list so indices match
    // the func_idx values assigned by run_probe_skeleton.
    let func_names: Vec<(u32, String)> = functions
        .iter()
        .enumerate()
        .map(|(i, f)| (i as u32, f.display_name.clone()))
        .collect();

    // Pre-open BPF program FDs while the scheduler is alive.
    // Holding these FDs keeps programs alive via kernel refcounting
    // even after the scheduler crashes.
    let bpf_fds = crate::probe::process::open_bpf_prog_fds(&functions);
    let pnames = crate::probe::output::build_param_names(&btf_funcs);
    let rhints = crate::probe::output::build_render_hints(&btf_funcs);
    let pnames_thread = pnames.clone();
    let rhints_thread = rhints.clone();
    let thread_pipeline = pipeline.clone();
    let funcs = functions.clone();
    let fn_names = func_names.clone();
    let pd = pipe_diag.clone();
    let handle = std::thread::spawn(move || {
        use crate::probe::process::run_probe_skeleton;
        let (events, diag, accumulated_fn_names) = run_probe_skeleton(
            &funcs,
            &btf_funcs,
            &thread_pipeline.stop,
            &bpf_fds,
            &thread_pipeline.probes_ready,
            None,
        );
        let emit_fn_names = if accumulated_fn_names.is_empty() {
            &fn_names
        } else {
            &accumulated_fn_names
        };
        // Serialize probe output after the trigger fires or stop
        // is signaled. Runs before the thread returns so output
        // reaches the host (bulk-port stdout) even if the main thread is blocked.
        emit_probe_payload(
            events.as_deref().unwrap_or(&[]),
            emit_fn_names,
            &pd,
            &diag,
            &pnames_thread,
            &rhints_thread,
        );
        thread_pipeline.output_done.set();
        (events, diag, accumulated_fn_names)
    });

    // Wait for probes to attach before starting the test function.
    // Without this, the test may crash the scheduler before probes
    // are active, resulting in 0 captured events.
    pipeline.probes_ready.wait();

    Some(ProbeHandle {
        thread: handle,
        func_names,
        pipeline_diag: pipe_diag,
        output_done: pipeline.output_done.clone(),
        param_names: pnames,
        render_hints: rhints,
    })
}

/// Result returned by the probe thread: collected events, skeleton
/// diagnostics, and accumulated function names from both phases.
type ProbeThreadResult = (
    Option<Vec<crate::probe::process::ProbeEvent>>,
    crate::probe::process::ProbeDiagnostics,
    Vec<(u32, String)>,
);

/// Probe-thread handle and associated state returned by the setup path.
///
/// Owns the join handle plus everything `collect_and_print_probe_data`
/// needs on the stop side: function-name registry for event rendering,
/// pipeline diagnostics captured before skeleton spawn, the
/// `output_done` flag the thread flips when it has already written
/// `PROBE_OUTPUT_START`/`PROBE_OUTPUT_END` to the bulk port (stdout), and the param-name / render-hint maps
/// used to pretty-print parameters.
struct ProbeHandle {
    thread: std::thread::JoinHandle<ProbeThreadResult>,
    func_names: Vec<(u32, String)>,
    pipeline_diag: PipelineDiagnostics,
    output_done: std::sync::Arc<crate::sync::Latch>,
    param_names: std::collections::HashMap<String, Vec<(String, String)>>,
    render_hints: std::collections::HashMap<String, crate::probe::btf::RenderHint>,
}

/// Cross-thread probe pipeline signals.
///
/// Groups the three signals the probe setup path has to hand to its
/// worker thread: `stop` (main thread asks the probe thread to shut
/// down), `output_done` (probe thread tells the main thread it has
/// already emitted `PROBE_OUTPUT_START`/`PROBE_OUTPUT_END`), and `probes_ready` (probe
/// thread signals the main thread that kprobes/kfentries have
/// attached). `stop` is an `AtomicBool` because the probe thread's
/// ring-buffer poll loop checks it via `load(Acquire)` between
/// events — a blocking wait would stall diagnostics collection.
/// `output_done` and `probes_ready` use [`crate::sync::Latch`] so
/// the dispatch path and the early-bail drain path block on a
/// condvar instead of sleep-polling. [`Clone`] is the expected way
/// to produce the thread-side view before calling
/// `std::thread::spawn` — each clone bumps refcounts only.
#[derive(Clone, Default)]
pub(crate) struct ProbePipeline {
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub output_done: std::sync::Arc<crate::sync::Latch>,
    pub probes_ready: std::sync::Arc<crate::sync::Latch>,
}

impl ProbePipeline {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Pre-skeleton pipeline diagnostics captured during guest probe setup.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PipelineDiagnostics {
    /// Functions from --ktstr-probe-stack before filter.
    pub stack_extracted: u32,
    /// Functions dropped by filter_traceable.
    pub filter_dropped: Vec<String>,
    /// BPF symbols discovered from running scheduler.
    pub bpf_discovered: u32,
    /// Functions after expand_bpf_to_kernel_callers.
    pub total_after_expand: u32,
}

/// State from Phase A probe attachment (before scheduler starts).
///
/// Returned by `start_probe_phase_a`. Contains the probe thread handle,
/// the channel to send Phase B input (BPF fentry targets), and metadata
/// needed by the readout phase.
pub(crate) struct ProbePhaseAState {
    pub handle: std::thread::JoinHandle<ProbeThreadResult>,
    pub phase_b_tx: std::sync::mpsc::Sender<crate::probe::process::PhaseBInput>,
    /// Shared pipeline atomics (`stop`, `output_done`, `probes_ready`)
    /// — grouped so Phase B consumers thread a single value through
    /// the join + publish tail rather than tracking each `Arc` by hand.
    pub pipeline: ProbePipeline,
    pub kernel_func_names: Vec<(u32, String)>,
    /// Number of functions in Phase A. Phase B uses this as func_idx_offset
    /// to avoid index collisions in the shared BPF maps.
    pub kernel_func_count: u32,
    pub pipe_diag: PipelineDiagnostics,
    pub param_names: std::collections::HashMap<String, Vec<(String, String)>>,
    pub render_hints: std::collections::HashMap<String, crate::probe::btf::RenderHint>,
}

/// Start Phase A of the probe pipeline (before scheduler starts).
///
/// Parses `--ktstr-probe-stack` from args, loads kernel functions,
/// attaches kprobes + trigger + kernel fexit, and spawns the probe
/// thread with a Phase B channel. Returns `None` if no probe stack
/// arg is present or no traceable functions remain.
pub(crate) fn start_probe_phase_a(args: &[String]) -> Option<ProbePhaseAState> {
    use crate::probe::stack::{filter_traceable, load_probe_stack};

    let stack_input = extract_probe_stack_arg(args)?;

    let phase_a_start = Instant::now();
    eprintln!("ktstr_test: probe phase_a: loading kernel functions");
    let mut pipe_diag = PipelineDiagnostics::default();
    let raw_functions = load_probe_stack(&stack_input);
    pipe_diag.stack_extracted = raw_functions.len() as u32;
    let pre_filter: Vec<String> = raw_functions.iter().map(|f| f.raw_name.clone()).collect();
    let functions = filter_traceable(raw_functions);
    for name in &pre_filter {
        if !functions.iter().any(|f| f.raw_name == *name) {
            pipe_diag.filter_dropped.push(name.clone());
        }
    }

    // Phase A only processes kernel functions (non-BPF). BPF functions
    // are handled in Phase B after the scheduler starts.
    let kernel_functions: Vec<crate::probe::stack::StackFunction> =
        functions.into_iter().filter(|f| !f.is_bpf).collect();

    // Resolve BTF for kernel functions.
    let kernel_names: Vec<&str> = kernel_functions
        .iter()
        .map(|f| f.raw_name.as_str())
        .collect();
    let btf_funcs = crate::probe::btf::parse_btf_functions(&kernel_names, None);

    let func_names: Vec<(u32, String)> = kernel_functions
        .iter()
        .enumerate()
        .map(|(i, f)| (i as u32, f.display_name.clone()))
        .collect();

    pipe_diag.total_after_expand = kernel_functions.len() as u32;

    let bpf_fds = std::collections::HashMap::new(); // No BPF FDs in Phase A
    let param_names = crate::probe::output::build_param_names(&btf_funcs);
    let render_hints = crate::probe::output::build_render_hints(&btf_funcs);

    let pipeline = ProbePipeline::new();

    let (phase_b_tx, phase_b_rx) = std::sync::mpsc::channel();

    let thread_pipeline = pipeline.clone();
    let funcs = kernel_functions.clone();
    let btf = btf_funcs.clone();
    let fn_names = func_names.clone();
    let pd = pipe_diag.clone();
    let pnames = param_names.clone();
    let rhints = render_hints.clone();

    let handle = std::thread::spawn(move || {
        let (events, diag, accumulated_fn_names) = crate::probe::process::run_probe_skeleton(
            &funcs,
            &btf,
            &thread_pipeline.stop,
            &bpf_fds,
            &thread_pipeline.probes_ready,
            Some(phase_b_rx),
        );
        let emit_fn_names = if accumulated_fn_names.is_empty() {
            &fn_names
        } else {
            &accumulated_fn_names
        };
        emit_probe_payload(
            events.as_deref().unwrap_or(&[]),
            emit_fn_names,
            &pd,
            &diag,
            &pnames,
            &rhints,
        );
        thread_pipeline.output_done.set();
        (events, diag, accumulated_fn_names)
    });

    // Wait for Phase A probes (kprobes + trigger + kernel fexit) to attach.
    pipeline.probes_ready.wait();

    tracing::info!(
        elapsed_ms = phase_a_start.elapsed().as_millis() as u64,
        kernel_functions = kernel_functions.len(),
        "auto_repro: phase_a_attach",
    );

    eprintln!(
        "ktstr_test: probe phase_a: {} kernel functions attached, waiting for Phase B",
        kernel_functions.len(),
    );

    let kernel_func_count = kernel_functions.len() as u32;

    Some(ProbePhaseAState {
        handle,
        phase_b_tx,
        pipeline,
        kernel_func_names: func_names,
        kernel_func_count,
        pipe_diag,
        param_names,
        render_hints,
    })
}

/// Complete the probe pipeline with Phase B (after scheduler starts).
///
/// Discovers BPF symbols from the running scheduler, opens BPF prog
/// FDs, sends Phase B input to the probe thread, waits for Phase B
/// attachment, then runs the test function and collects probe output.
///
/// Returns `Some(exit_code)` if dispatched, `None` if not.
pub(crate) fn maybe_dispatch_vm_test_with_phase_a(
    args: &[String],
    pa: ProbePhaseAState,
) -> Option<i32> {
    use crate::probe::btf::discover_bpf_symbols;

    // Env propagation cannot happen here: `pa` holds a live probe
    // thread spawned by `start_probe_phase_a`, so mutating
    // `std::env::__environ` now would race with that thread. The
    // caller (`ktstr_guest_init`) invokes `propagate_rust_env_from_cmdline`
    // before Phase A spawns the thread.
    let name = match extract_test_fn_arg(args) {
        Some(n) => n,
        None => {
            tracing::debug!("ktstr-init: no --ktstr-test-fn in args, skipping dispatch");
            return None;
        }
    };

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    let work_type_override = extract_work_type_arg(args).and_then(|s| {
        crate::workload::WorkType::from_name(&s).or_else(|| {
            // `from_name` is exact-match on the PascalCase canonical
            // form. A user typo (`spinwait`, `SPINWAIT`) lands here;
            // call `WorkType::suggest` for the canonical spelling
            // and surface it in the diagnostic so the user doesn't
            // have to guess the correct casing.
            match crate::workload::WorkType::suggest(&s) {
                Some(canonical) => eprintln!(
                    "ktstr_test: unknown work type '{s}'; did you mean \
                     '{canonical}'? Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
                None => eprintln!(
                    "ktstr_test: unknown work type '{s}'. Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
            }
            None
        })
    });

    // Destructure Phase A state up front so later branches (Phase B
    // send / drop, handle construction, stop propagation) operate on
    // owned locals. Keeping `pa` whole across `drop(pa.phase_b_tx)`
    // would partial-move the value and block the final `ProbeHandle`
    // build.
    let ProbePhaseAState {
        handle: pa_handle,
        phase_b_tx: pa_phase_b_tx,
        pipeline: pa_pipeline,
        kernel_func_names: pa_kernel_func_names,
        kernel_func_count: pa_kernel_func_count,
        pipe_diag: mut pa_pipe_diag,
        param_names: pa_param_names,
        render_hints: pa_render_hints,
    } = pa;

    // Phase B: discover BPF symbols from the running scheduler.
    //
    // Distinguish the two zero-result modes so the `bpf_discover` line
    // in the rendered probe pipeline summary explains itself: when
    // discovery returns zero AND the scheduler is still alive, the
    // walk genuinely found no struct_ops programs — surface a warn
    // so the operator notices the unexpected configuration. When the
    // scheduler exited before discovery (fast crash path), the empty
    // result is by-design and gets logged at info level so the
    // operator doesn't chase it as a bug. The pa_pipe_diag value
    // itself is updated below regardless so the host sees the actual
    // discovered count rather than the stale Phase A 0.
    eprintln!("ktstr_test: probe phase_b: discovering BPF symbols");
    let discover_start = Instant::now();
    let stack_display_names: Vec<&str> = Vec::new(); // Discovery uses empty hint list
    let bpf_syms = discover_bpf_symbols(&stack_display_names);
    if bpf_syms.is_empty() {
        let sched_alive = crate::vmm::rust_init::sched_pid()
            .is_some_and(|pid| unsafe { libc::kill(pid, 0) == 0 });
        if sched_alive {
            tracing::warn!(
                "phase_b: bpf_discover returned 0 programs while scheduler is \
                 still alive — verify ProgInfoIter access permissions or BTF \
                 (this is the unexpected case; the auto-repro pipeline is now \
                 attached to no BPF struct_ops callbacks)"
            );
        } else {
            tracing::info!(
                "phase_b: bpf_discover returned 0 programs — scheduler exited \
                 before the discovery window (expected for fast-crash paths)"
            );
        }
    }
    // Update Phase A's pipeline-diag counter with the Phase B
    // discovery result. Phase A only counts kernel functions; the
    // BPF discovery is intrinsically a Phase B event but the same
    // diag struct travels through the host-side renderer, so without
    // this assignment the rendered `bpf_discover: 0 programs found`
    // line would understate every successful run. The raw Phase B
    // discover count (NOT the post-`expand_bpf_to_kernel_callers`
    // count) goes here because the pipeline-diag invariant is
    // "discovered struct_ops + auxiliary programs", not "fentry
    // attach plan".
    pa_pipe_diag.bpf_discovered = bpf_syms.len() as u32;
    tracing::info!(
        elapsed_ms = discover_start.elapsed().as_millis() as u64,
        bpf_syms = bpf_syms.len(),
        "auto_repro: phase_b_discover",
    );
    eprintln!(
        "ktstr_test: probe phase_b: {} BPF symbols discovered",
        bpf_syms.len()
    );

    run_phase_b_attach(
        bpf_syms,
        pa_phase_b_tx,
        pa_kernel_func_count,
        &mut pa_pipe_diag,
    );

    let (topo, cgroups, sched_pid, merged_assert) = build_dispatch_ctx_parts(entry, args);
    // Settle is `Duration::ZERO` to match the single-phase
    // `maybe_dispatch_vm_test_with_args` path. The Phase A `probes_ready`
    // latch already synchronises probe attachment with the test start;
    // a host-side post-cgroup-creation sleep adds no correctness over
    // the latch, only auto-repro overhead. Tests that need cgroup
    // settle override `Ctx::settle` themselves; this default keeps
    // the auto-repro fast path on par with the non-probe path.
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .sched_pid(sched_pid)
        .settle(Duration::ZERO)
        .work_type_override(work_type_override)
        .assert(merged_assert)
        .wait_for_map_write(!entry.bpf_map_write.is_empty())
        .entry_name(entry.name)
        // Host-threaded authoritative variant hash (see
        // extract_variant_hash_arg); `0` only if the guest was invoked
        // without it, in which case the entry_name=Some bail does not
        // fire but the path simply carries the 0 suffix — production
        // always injects it (run_ktstr_test_inner_impl).
        .variant_hash(extract_variant_hash_arg(args).unwrap_or(0))
        .build();

    // Build the ProbeHandle up front from the destructured Phase A
    // locals — cheap (mostly Arc clones and already-owned Vecs) and
    // lets both the Ok and Err tails funnel through
    // `publish_result_and_collect` without re-assembling the handle.
    let stop = pa_pipeline.stop.clone();
    let handle = ProbeHandle {
        thread: pa_handle,
        func_names: pa_kernel_func_names,
        pipeline_diag: pa_pipe_diag,
        output_done: pa_pipeline.output_done,
        param_names: pa_param_names,
        render_hints: pa_render_hints,
    };
    // Send SCENARIO_START so the host-side watchdog resets its hard
    // deadline to `now + workload_duration`. See the matching site
    // in `maybe_dispatch_vm_test_with_args` for the full rationale.
    if crate::vmm::guest_comms::is_guest() {
        crate::vmm::guest_comms::send_scenario_start();
    }
    let mut result = match (entry.func)(&ctx) {
        Ok(r) => r,
        Err(e) => {
            let r = AssertResult::fail_msg(format!("{e:#}"));
            publish_result_and_collect(&r, stop, Some(handle));
            return Some(1);
        }
    };
    // survives_storm: re-check scheduler liveness here so a non-execute_*
    // scenario that hand-rolls Op dispatch (running no in-hold liveness probe)
    // still fails if the scheduler died during the run.
    enforce_survives_storm_liveness(&mut result, entry.survives_storm);

    let exit_code = exit_code_for_result(&result);
    publish_result_and_collect(&result, stop, Some(handle));
    Some(exit_code)
}

/// Phase B of the auto-repro probe pipeline: given the BPF symbols
/// discovered from the running scheduler, expand them to kernel-side
/// callers, open the BPF program FDs, parse Phase B BTF, and hand the
/// attach request to the probe thread, waiting for attachment to
/// complete. With no BPF symbols, drops `phase_b_tx` so the probe
/// thread's `try_recv` sees `Disconnected`. Rolls the post-expansion
/// target count into `pipe_diag.total_after_expand`.
fn run_phase_b_attach(
    bpf_syms: Vec<crate::probe::stack::StackFunction>,
    phase_b_tx: std::sync::mpsc::Sender<crate::probe::process::PhaseBInput>,
    kernel_func_count: u32,
    pipe_diag: &mut PipelineDiagnostics,
) {
    use crate::probe::stack::expand_bpf_to_kernel_callers;

    if !bpf_syms.is_empty() {
        // Expand BPF to kernel callers. Both BPF callbacks (for fentry)
        // and kernel callers (for additional kprobes) are included in
        // Phase B input.
        let phase_b_functions = expand_bpf_to_kernel_callers(bpf_syms);
        // Roll the post-expansion total into Phase A's diag. The
        // `after_expand` counter is the host-side render's sense of
        // "total probe targets the pipeline planned"; Phase A only
        // contributed kernel functions (already in
        // `pipe_diag.total_after_expand`), so add Phase B's
        // contribution here. Without this, the
        // `after_expand: N total probe targets` line undercounts by
        // every Phase B function attached.
        pipe_diag.total_after_expand = pipe_diag
            .total_after_expand
            .saturating_add(phase_b_functions.len() as u32);

        // Open BPF program FDs while the scheduler is alive.
        let bpf_fds = crate::probe::process::open_bpf_prog_fds(&phase_b_functions);

        // Parse BPF function signatures from BPF program BTF.
        let bpf_btf_args: Vec<(&str, u32)> = phase_b_functions
            .iter()
            .filter(|f| f.is_bpf)
            .filter_map(|f| Some((f.display_name.as_str(), f.bpf_prog_id?)))
            .collect();
        let mut phase_b_btf = if !bpf_btf_args.is_empty() {
            crate::probe::btf::parse_bpf_btf_functions(&bpf_btf_args)
        } else {
            Vec::new()
        };
        // Parse BTF for kernel callers added by expand_bpf_to_kernel_callers.
        let kernel_caller_names: Vec<&str> = phase_b_functions
            .iter()
            .filter(|f| !f.is_bpf)
            .map(|f| f.raw_name.as_str())
            .collect();
        if !kernel_caller_names.is_empty() {
            phase_b_btf.extend(crate::probe::btf::parse_btf_functions(
                &kernel_caller_names,
                None,
            ));
        }

        let phase_b_done = std::sync::Arc::new(crate::sync::Latch::new());
        let phase_b_done_clone = phase_b_done.clone();

        let n_phase_b_functions = phase_b_functions.len();
        let phase_b_input = crate::probe::process::PhaseBInput {
            functions: phase_b_functions,
            bpf_prog_fds: bpf_fds,
            btf_funcs: phase_b_btf,
            done: phase_b_done_clone,
            func_idx_offset: kernel_func_count,
        };

        let attach_start = Instant::now();
        if let Err(e) = phase_b_tx.send(phase_b_input) {
            eprintln!("ktstr_test: probe phase_b: failed to send: {e}");
        } else {
            // Wait for Phase B attachment to complete.
            phase_b_done.wait();
            tracing::info!(
                elapsed_ms = attach_start.elapsed().as_millis() as u64,
                phase_b_functions = n_phase_b_functions,
                "auto_repro: phase_b_attach",
            );
            eprintln!("ktstr_test: probe phase_b: BPF fentry attached");
        }
    } else {
        eprintln!("ktstr_test: probe phase_b: no BPF symbols, skipping fentry");
        // Drop the sender so the probe thread's try_recv sees Disconnected.
        drop(phase_b_tx);
    }
}

/// Serialized probe data sent from guest to host via the bulk port (stdout).
/// The host deserializes and formats with kernel_dir for source locations.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbeBytes {
    pub events: Vec<crate::probe::process::ProbeEvent>,
    pub func_names: Vec<(u32, String)>,
    pub bpf_source_locs: std::collections::HashMap<String, String>,
    pub diagnostics: Option<ProbeBytesDiagnostics>,
    /// Guest VM CPU count for cpumask masking. Populated by
    /// `emit_probe_payload` which runs inside the guest where
    /// sysfs reports the correct value.
    pub nr_cpus: Option<u32>,
    /// BTF-resolved parameter labels per function: func_name ->
    /// vec of (param_name, type_label). Used by the formatter to
    /// print named args instead of arg0/arg1.
    pub param_names: std::collections::HashMap<String, Vec<(String, String)>>,
    /// BTF-derived render hints for auto-discovered fields.
    /// Maps field key (e.g. `"ctx:task_ctx.data__sz"`) to display format.
    pub render_hints: std::collections::HashMap<String, crate::probe::btf::RenderHint>,
}

/// Combined diagnostics for the probe payload.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbeBytesDiagnostics {
    pub pipeline: PipelineDiagnostics,
    pub skeleton: crate::probe::process::ProbeDiagnostics,
}

/// Serialize probe payload to stdout (bulk port) between delimiters.
/// Resolves BPF source locations from loaded programs before serializing.
///
/// Skips the BPF symbol discovery + source-loc resolution walk when
/// `events` is empty: with no captured events, the source-loc map is
/// never read by the host renderer, but the walk over every loaded
/// BPF program (`ProgInfoIter` + per-prog BTF parse + per-prog
/// `bpf_obj_get_info_by_fd` line_info cross-reference) still pays the
/// full cost. Skipping when there is nothing to render trims that
/// overhead from the auto-repro readout path on every clean (no-
/// trigger) run and on stop-driven empty drains.
fn emit_probe_payload(
    events: &[crate::probe::process::ProbeEvent],
    func_names: &[(u32, String)],
    pipeline_diag: &PipelineDiagnostics,
    skeleton_diag: &crate::probe::process::ProbeDiagnostics,
    param_names: &std::collections::HashMap<String, Vec<(String, String)>>,
    render_hints: &std::collections::HashMap<String, crate::probe::btf::RenderHint>,
) {
    let bpf_source_locs = if events.is_empty() {
        // Nothing to render — skip the ProgInfoIter walk + per-prog
        // line_info resolution. The host-side formatter only reads
        // `bpf_source_locs` when iterating events, so an empty map is
        // semantically equivalent here.
        std::collections::HashMap::new()
    } else {
        let source_loc_names: Vec<&str> =
            func_names.iter().map(|(_, name)| name.as_str()).collect();
        let bpf_syms = crate::probe::btf::discover_bpf_symbols(&source_loc_names);
        let bpf_prog_ids: Vec<u32> = func_names
            .iter()
            .filter_map(|(_, name)| {
                bpf_syms
                    .iter()
                    .find(|s| s.display_name == *name)
                    .and_then(|s| s.bpf_prog_id)
            })
            .collect();
        crate::probe::btf::resolve_bpf_source_locs(&bpf_prog_ids)
    };

    let payload = ProbeBytes {
        events: events.to_vec(),
        func_names: func_names.to_vec(),
        bpf_source_locs,
        diagnostics: Some(ProbeBytesDiagnostics {
            pipeline: pipeline_diag.clone(),
            skeleton: skeleton_diag.clone(),
        }),
        nr_cpus: crate::probe::output::get_nr_cpus(),
        param_names: param_names.clone(),
        render_hints: render_hints.clone(),
    };
    println!("{PROBE_OUTPUT_START}");
    if let Ok(json) = serde_json::to_string(&payload) {
        println!("{json}");
    }
    println!("{PROBE_OUTPUT_END}");
}

/// Probe-collection state stashed at the end of dispatch and drained
/// after the guest's Phase 6 scheduler teardown. Holds the `stop`
/// signal and probe handle that [`collect_and_print_probe_data`]
/// consumes; deferring the consumption past `child.kill()` is what
/// keeps the `tp_btf/sched_ext_exit` listener attached while the
/// kernel's `scx_claim_exit` path fires the trigger.
///
/// Stored in [`DEFERRED_PROBE_COLLECT`]; [`take_deferred_probe`]
/// drains it.
struct DeferredProbe {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
}

/// SAFETY-relevant invariants:
/// - Single producer: each VM run executes exactly one dispatch
///   call (`maybe_dispatch_vm_test_with_args` or
///   `maybe_dispatch_vm_test_with_phase_a`) which calls
///   [`stash_deferred_probe`] at most once.
/// - Single consumer: only `ktstr_guest_init` Phase 6 calls
///   [`finalize_probe_after_unwind`], which calls
///   [`take_deferred_probe`].
/// - Process-local: every guest VM is a fresh process; the static
///   resets between runs because the process exits.
///
/// `Mutex<Option<DeferredProbe>>` is the standard "stash for later
/// consumer" shape and matches `BPF_MAP_WRITE_DONE_LATCH` /
/// similar one-shot statics elsewhere in the crate.
static DEFERRED_PROBE_COLLECT: std::sync::Mutex<Option<DeferredProbe>> =
    std::sync::Mutex::new(None);

/// Stash the probe stop signal + handle for deferred consumption.
/// Replaces any prior value (a re-entrant dispatch is not a supported
/// pattern; the previous stash would belong to a phantom run).
fn stash_deferred_probe(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    let mut guard = DEFERRED_PROBE_COLLECT.lock().unwrap();
    *guard = Some(DeferredProbe { stop, handle });
}

/// Drain the stashed probe stop+handle, if any. Returns `None` when
/// no deferred collection was stashed (single-phase ctor path, or no
/// probes attached). Called from `ktstr_guest_init` Phase 6.
fn take_deferred_probe() -> Option<DeferredProbe> {
    DEFERRED_PROBE_COLLECT.lock().unwrap().take()
}

/// Wait up to `timeout` for `/sys/kernel/sched_ext/state` to read
/// "disabled". Returns `true` when the transition is observed,
/// `false` on timeout or when the file is unreadable (kernels
/// without sched_ext or non-root probes can't read it).
///
/// `disabled` is set by `scx_set_enable_state(SCX_DISABLED)` at the
/// tail of `scx_root_disable`, which runs strictly after
/// `scx_claim_exit` (called from `scx_vexit`/`scx_disable`) fired
/// `trace_sched_ext_exit`. Polling for this transition is the most
/// reliable signal that the trigger tracepoint has fired (or never
/// will, in which case the timeout is the correct signal).
///
/// Polls every 50 ms — short enough to bound the post-test
/// finalisation latency, long enough that the per-iteration
/// `read_to_string` doesn't dominate. The poll loop with bounded
/// timeout is the coordinator-approved pattern for this signal:
/// sysfs has no eventfd-style notify, and inotify on `state`
/// would be marginally simpler in code but pull a dependency
/// for sub-second savings.
fn wait_for_sched_disabled(timeout: std::time::Duration) -> bool {
    wait_for_sched_disabled_at("/sys/kernel/sched_ext/state", timeout)
}

/// Path-parametric core of [`wait_for_sched_disabled`]. Production passes
/// the hardcoded `/sys/kernel/sched_ext/state`; tests pass a controlled
/// path to drive each arm deterministically, since the live sysfs file's
/// presence and contents vary by host — a kernel built with
/// `CONFIG_SCHED_CLASS_EXT` and no attached scheduler reads `"disabled"`,
/// so the production path is not reliably absent.
fn wait_for_sched_disabled_at(path: &str, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Ok(s) = std::fs::read_to_string(path) {
            if s.trim() == "disabled" {
                return true;
            }
        } else {
            // File unreadable: kernel without sched_ext, or sysfs
            // restriction. The probe-attach path already ran a
            // tp_btf/sched_ext_exit attach; if the file doesn't
            // exist, the tracepoint can't fire either. Treat as
            // "no-op wait" rather than spinning.
            return false;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// Finalise probes after the guest's Phase 6 scheduler teardown.
///
/// Called from `ktstr_guest_init` AFTER `child.kill()` /
/// `child.wait()` / `/sched_disable`. By the time the kernel
/// finishes `scx_disable_irq_workfn` (signalled by
/// `/sys/kernel/sched_ext/state` transitioning to `disabled`),
/// the probe's `tp_btf/sched_ext_exit` listener has had its one
/// guaranteed fire — the trigger event lands in the ring buffer
/// with a real `target_tptr`, the probe poll loop's BSS latch
/// check observes `ktstr_err_exit_detected != 0`, and the
/// readout phase stitches the kprobe events that fired during
/// the actual stall window.
///
/// Returns immediately when no probes were stashed (single-phase
/// ctor path, EEVDF runs, etc.) — a no-op in that case.
///
/// Bounds the wait at 5 s so a non-responding kernel cannot stall
/// guest teardown indefinitely; on timeout the existing
/// stop-then-join path runs and the diagnostic surfaces "trigger
/// never fired" to the host.
pub(crate) fn finalize_probe_after_unwind() {
    let Some(deferred) = take_deferred_probe() else {
        return;
    };
    // Skip the kernel-unwind wait when no probe handle is attached.
    // The wait exists exclusively to give the probe's
    // tp_btf/sched_ext_exit listener time to observe the trigger;
    // when there is no listener, the wait is wasted teardown
    // latency on every test (auto-repro is the only path that
    // installs a probe handle, and even then only when the primary
    // failed). The unconditional `collect_and_print_probe_data`
    // call below early-returns on `handle.is_none()`, so the no-op
    // path is preserved.
    if deferred.handle.is_some() {
        // Wait for the kernel to finish unwinding the scheduler.
        // The probe poll loop is still running; once
        // `ktstr_err_exit_detected` flips, the loop drains any
        // pending ringbuf events and breaks on its own. We wait
        // first so the subsequent stop signal can't pre-empt the
        // loop's BSS check.
        //
        // No grace sleep after `wait_for_sched_disabled` returns
        // true: the kernel's `scx_claim_exit` (called from
        // `scx_vexit`/`scx_disable`) fires `trace_sched_ext_exit`
        // BEFORE `scx_set_enable_state(SCX_DISABLED)` runs at the
        // tail of `scx_root_disable`, so a `state == disabled`
        // observation establishes a happens-after relationship
        // with the BPF handler's CAS
        // on `ktstr_err_exit_detected` and the ringbuf-event
        // commit. The probe poll loop already breaks on
        // `bss_triggered` without needing `stop` set; setting
        // `stop` immediately is correct and event-driven.
        let _ = wait_for_sched_disabled(std::time::Duration::from_secs(5));
    }
    collect_and_print_probe_data(deferred.stop, deferred.handle);
}

/// Map an [`AssertResult`] to a probe-dispatch exit code per the
/// `Fail > Inconclusive > Pass > Skip` lattice: Pass → 0, Inconclusive
/// → 2, every other state → 1 (Skip degenerates to 1 in the probe
/// path because a skipped probe test produced no signal and the
/// dispatch caller needs the failure signal to surface). Distinct
/// codes let CI tooling triage zero-denominator probe runs separately
/// from real probe failures.
fn exit_code_for_result(result: &AssertResult) -> i32 {
    if result.is_pass() {
        0
    } else if result.is_inconclusive() {
        2
    } else {
        1
    }
}

/// When `survives_storm` is asserted, actively re-check scheduler liveness at
/// the guest-side test-function choke point and fold a scheduler-death fail into
/// `result` if the scheduler died/went down and `result` does not already carry
/// one. Closes the vacuous-pass hole for `survives_storm` scenarios whose test
/// function hand-rolls `Op` dispatch without an `execute_*` driver: those run no
/// in-hold liveness probe, so a scheduler that died mid-test would otherwise
/// leave a passing `AssertResult` and the host's `survives_storm` marker (which
/// keys on a `Scheduler*` detail) would never attach. For `execute_*` scenarios
/// the in-hold probe already recorded the death, so the already-present guard
/// makes this a no-op (no double-record). Folding the fail makes
/// `result.is_pass()` false, which drops the dispatch exit code to 1 and lets
/// the host eval attach the `SurvivesStormViolated` marker.
fn enforce_survives_storm_liveness(result: &mut AssertResult, survives_storm: bool) {
    use crate::assert::DetailKind;
    if !survives_storm {
        return;
    }
    // An execute_* in-hold probe already attributed the death — don't double-record.
    let already_recorded = result.failure_details().any(|d| {
        matches!(
            d.kind,
            DetailKind::SchedulerCrashed
                | DetailKind::SchedulerExitedCleanly
                | DetailKind::SchedulerDiedUnknownReason
        )
    });
    if already_recorded {
        return;
    }
    if let Some(kind) = crate::scenario::ops::sched_liveness_failure_kind() {
        result.record_fail(crate::assert::AssertDetail::new(
            kind,
            crate::assert::format_sched_died_survives_storm(),
        ));
    }
}

/// Flush profraw, publish the assert result to guest stdout, then
/// either STASH the probe stop+handle for deferred collection (when
/// running as the guest VM's PID 1, where Phase 6 scheduler
/// teardown lives) or collect-and-emit immediately (ctor path on
/// the host, where there is no Phase 6).
///
/// The deferred path is what keeps the `tp_btf/sched_ext_exit`
/// listener attached while the kernel fires the trigger from
/// `scx_claim_exit` during scheduler unwind — without it, the probe
/// is detached before `child.kill()` runs and the stall-class
/// trigger fires into a void (146 captured kprobe events, 0
/// trigger fires, 0 after stitch).
///
/// The deferred collection runs from
/// [`finalize_probe_after_unwind`] in `ktstr_guest_init` Phase 6
/// after `child.wait()` + `/sched_disable`.
fn publish_result_and_collect(
    result: &AssertResult,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    try_flush_profraw();
    print_assert_result(result);
    if crate::vmm::guest_comms::is_guest() {
        stash_deferred_probe(stop, handle);
    } else {
        collect_and_print_probe_data(stop, handle);
    }
}

/// Stop probes, join the probe thread. The probe thread emits output
/// directly when the trigger fires; this function only needs to set
/// `stop` and join. If the probe thread already emitted output, this
/// is a no-op.
fn collect_and_print_probe_data(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    let Some(ph) = handle else {
        return;
    };

    stop.store(true, std::sync::atomic::Ordering::Release);
    let (events, skeleton_diag, accumulated_fn_names) = match ph.thread.join() {
        Ok((Some(events), diag, fnames)) => (events, diag, fnames),
        Ok((None, diag, fnames)) => (Vec::new(), diag, fnames),
        Err(payload) => {
            // Stamp the panic payload onto a fresh diagnostics
            // record. Without this, the empty events vec + default
            // diag emitted on the host via the bulk-port stdout channel is byte-for-byte
            // identical to a clean run where the trigger simply
            // never fired — the host can't tell that the probe
            // thread crashed and would silently record the test as
            // passing. Setting `host_thread_panic` is the
            // single signal the host parser uses to fail the run.
            // `panic!(...)` payloads in safe code are either
            // `&'static str` or `String`; other types fall through
            // to a sentinel so the field is always populated when
            // we reach this arm.
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic>".to_string()
            };
            let diag = crate::probe::process::ProbeDiagnostics {
                host_thread_panic: Some(msg),
                ..Default::default()
            };
            (Vec::new(), diag, Vec::new())
        }
    };

    // Prefer accumulated func_names (includes both Phase A and Phase B).
    let effective_fn_names = if accumulated_fn_names.is_empty() {
        &ph.func_names
    } else {
        &accumulated_fn_names
    };

    // The probe thread already emitted output on trigger/stop.
    // Only emit here if it somehow didn't (e.g. thread panicked
    // before reaching emit_probe_payload).
    if !ph.output_done.is_set() {
        emit_probe_payload(
            &events,
            effective_fn_names,
            &ph.pipeline_diag,
            &skeleton_diag,
            &ph.param_names,
            &ph.render_hints,
        );
    }
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
