//! Host-mode worker stuck detection via `/proc/<pid>/sched` polling.
//!
//! When a scenario runs in host-mode (no VM boot — `!is_guest()` AND
//! `!cargo_test_mode_active()`), the freeze coordinator / KVM-side
//! stall plumbing is unavailable. This module fills that gap by
//! polling every worker pid's `/proc/<pid>/sched` file from a
//! background thread and flagging a "task did not run" condition
//! when both `nr_switches` and `sum_exec_runtime` are unchanged
//! across a sliding window of W samples.
//!
//! # Signal
//!
//! `/proc/<pid>/sched` exposes both `nr_switches` (total context
//! switches the task has been involved in) and `sum_exec_runtime`
//! (cumulative on-CPU nanoseconds). Both are emitted unconditionally
//! by `kernel/sched/debug.c` regardless of `CONFIG_SCHEDSTATS`, so
//! the signal works on any production kernel.
//!
//! `se.statistics.wait_sum` would arguably be a STRONGER signal
//! (cumulative time waiting in the runqueue — a starved task
//! grows wait_sum monotonically while a sleeping task does not),
//! but it IS gated on `CONFIG_SCHEDSTATS=y`. The monitor sticks
//! with the unconditional `nr_switches + sum_exec_runtime` pair
//! so it stays useful on minimum-config production kernels;
//! schedstat-aware schedulers (sched_ext schedulers that read
//! `wait_sum` via BPF) supplement this with their own latency
//! probes via the `--ktstr-probe-stack` pipeline.
//!
//! Stuck heuristic: if `Δnr_switches == 0` AND
//! `Δsum_exec_runtime == 0` across W consecutive samples, the task
//! has neither been picked nor preempted for `W * poll_interval` —
//! a stronger signal than either counter alone (a busy-loop on one
//! CPU could leave `nr_switches` flat while `sum_exec_runtime`
//! climbs; a fully starved task pins both).
//!
//! # Cadence
//!
//! Default poll interval is 500 ms; window size W = 4 yields a 2 s
//! detection latency. The interval is overridable via the
//! [`crate::KTSTR_STUCK_POLL_MS_ENV`] env var (empty / unset / 0 /
//! unparseable falls back to the default).
//!
//! # Diagnostic capture
//!
//! When a stuck condition fires, the polling thread captures a one-shot
//! diagnostic snapshot from `/proc/<pid>/{wchan, syscall, status,
//! stack, cgroup}` and `/proc/<pid>/task/<pid>/stat`, plus the
//! host's `/proc/loadavg`. Each field is read independently and
//! gracefully degraded to `"[unreadable: <reason>]"` on EACCES /
//! ENOENT — `/proc/<pid>/stack` requires `CAP_SYS_ADMIN` and is
//! typically absent for unprivileged callers, so its absence is
//! not a failure.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::KTSTR_STUCK_POLL_MS_ENV;

/// Default poll cadence when [`KTSTR_STUCK_POLL_MS_ENV`] is unset /
/// empty / 0 / unparseable. 500 ms × W=4 yields a 2 s detection
/// latency — short enough to catch a stuck scheduler within a
/// typical ktstr test duration, long enough that procfs reads stay
/// O(workers) per second rather than swamping the host.
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 500;

/// Sliding-window size: number of consecutive flat samples that
/// flip the stuck predicate. W=4 with [`DEFAULT_POLL_INTERVAL_MS`]
/// = 2 s detection latency. Constant rather than env-tunable —
/// the operator already controls latency via the poll interval,
/// and a smaller W would false-positive on transient idle.
///
/// # False-positive on slow-period workloads
///
/// A worker that legitimately runs once per `interval * W`
/// (e.g. a periodic 10 s sleep tracker on a 2 s poll) will look
/// "stuck" for the full W-sample window because both counters
/// stay flat between the worker's wakeups. The fire is a true
/// "no forward progress" observation — the operator distinguishes
/// false-positive (intentional slow period) from a true stuck condition
/// (kernel-side hang) via the [`StuckDiagnostic::wchan`] field on
/// the report: a healthy slow-period worker shows a `do_nanosleep`
/// / `pipe_read` / `epoll_wait` wchan, while a true stuck condition shows
/// the offending kernel function. Per the no-silent-drops
/// policy the framework opts for the loud-fire path rather than
/// guessing the worker's intended cadence.
pub const STUCK_WINDOW: usize = 4;

/// Snapshot of the two scheduler counters this monitor watches.
///
/// Mirrors what `kernel/sched/debug.c::proc_sched_show_task`
/// writes to `/proc/<pid>/sched`. Both fields are cumulative since task
/// creation; the monitor tracks deltas between consecutive
/// samples rather than absolute values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SchedSample {
    /// `nr_switches` from `/proc/<pid>/sched` — total voluntary +
    /// involuntary context switches involving this task.
    pub nr_switches: u64,
    /// `sum_exec_runtime` from `/proc/<pid>/sched` — cumulative
    /// on-CPU runtime in nanoseconds.
    pub sum_exec_runtime_ns: u64,
    /// Instant the sample was captured. Stored alongside the
    /// counters so a downstream consumer can compute wall-clock
    /// deltas without re-deriving the poll cadence. Not serialized
    /// — Instant is a monotonic-clock opaque value with no portable
    /// wire form; serde consumers reading a sidecar dump will see
    /// `Instant::now()` as the default. The wall-clock context for
    /// the report is carried separately by [`StuckReport::captured_at`]
    /// (also `#[serde(skip)]` for the same reason; sidecar dump
    /// consumers anchoring across runs should pair the report with
    /// the run's start timestamp from elsewhere).
    #[serde(skip, default = "Instant::now")]
    pub captured_at: Instant,
}

/// One-shot diagnostic snapshot captured at stuck-trip time.
///
/// Each field is the contents of a `/proc/<pid>/<field>` file
/// (or a stand-in describing why the read failed). Field
/// extraction is best-effort: a missing `/proc/<pid>/stack`
/// (requires `CAP_SYS_ADMIN`) does NOT block the diagnostic;
/// every field carries its own `"[unreadable: <reason>]"` stand-in
/// so an operator triaging the stuck worker can tell apart "kernel
/// didn't expose it" from "monitor failed to read it".
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StuckDiagnostic {
    /// `/proc/<pid>/wchan` — kernel symbol the task is sleeping
    /// in, or `0` when the task is runnable / wchan is unresolvable.
    pub wchan: String,
    /// `/proc/<pid>/syscall` — first field is the syscall number
    /// the task is blocked in (or `running` when on-CPU).
    pub syscall: String,
    /// `/proc/<pid>/status` task-state line ("State: S
    /// (sleeping)"). Parsed down to the single-letter state
    /// code; the full status file is preserved in `status_full`.
    pub state: String,
    /// `/proc/<pid>/stack` — kernel-stack trace; commonly
    /// unreadable without `CAP_SYS_ADMIN`. `None` when absent
    /// (versus an unreadable string in the string fields).
    pub stack: Option<String>,
    /// `/proc/<pid>/status` raw — full text for fields beyond
    /// `State:` the operator may want to inspect (Cpus_allowed,
    /// Threads, ctxt counts).
    pub status_full: String,
    /// `/proc/<pid>/cgroup` — v2 path the task belongs to.
    pub cgroup: String,
    /// Host's `/proc/loadavg` at trip time — useful for ruling
    /// out a false "stuck" verdict caused by extreme host load rather than
    /// scheduler misbehavior.
    pub host_loadavg: String,
}

/// One stuck report: a worker pid plus the sample window that
/// triggered the stuck predicate plus the captured diagnostic.
///
/// Pushed onto `StuckMonitor::reports` by the polling thread the
/// moment the predicate fires. The thread continues running so
/// subsequent stuck events on the same pid (or other pids) also surface,
/// but a pid is "re-armed" only after observing forward progress
/// (any non-zero delta) so a permanently-stuck task fires once and
/// then stays silent until it moves again — preventing a single
/// stuck event from spamming the report list every poll cycle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StuckReport {
    /// The stuck worker's pid.
    pub pid: libc::pid_t,
    /// `comm` field from `/proc/<pid>/comm` at trip time — the
    /// task's name (`SpinWait_0`, etc.) so an operator can map
    /// the pid back to a scenario WorkSpec without inspecting
    /// the diagnostic's full status file.
    pub comm: String,
    /// The sample window that triggered the predicate. Length
    /// equals [`STUCK_WINDOW`] at the moment of fire; later
    /// samples are NOT appended (each report is a snapshot, not
    /// a rolling view).
    pub samples: Vec<SchedSample>,
    /// Wall-clock instant the predicate fired. Not serialized —
    /// see [`SchedSample::captured_at`] for the rationale.
    #[serde(skip, default = "Instant::now")]
    pub captured_at: Instant,
    /// Diagnostic snapshot captured immediately after the trip.
    pub diagnostic: StuckDiagnostic,
}

/// Background polling thread + shared report buffer.
///
/// Framework-internal — the test author never constructs a
/// `StuckMonitor` directly; the scenario engine spawns one via
/// [`spawn_monitor`] in `apply_setup` when running host-mode.
/// Drop the [`StuckMonitorHandle`] to stop polling and join the
/// thread.
pub(crate) struct StuckMonitor {
    /// Shutdown flag — flipped to `true` by [`StuckMonitorHandle::drop`]
    /// to terminate the polling loop at the next sleep boundary.
    shutdown: Arc<AtomicBool>,
    /// Accumulated stuck reports. The polling thread appends; the
    /// owner drains via [`StuckMonitorHandle::drain`].
    reports: Arc<Mutex<Vec<StuckReport>>>,
}

/// Owned handle returned by [`spawn_monitor`]. Dropping joins the
/// polling thread; calling [`Self::drain`] returns accumulated
/// reports before the drop point.
///
/// Framework-internal — held by `StepState.stuck_monitor` for the
/// per-step lifetime; test authors don't construct or hand out
/// handles.
#[must_use = "StuckMonitorHandle stops polling on Drop; bind it to a local for the scenario lifetime"]
pub(crate) struct StuckMonitorHandle {
    monitor: StuckMonitor,
    thread: Option<JoinHandle<()>>,
}

impl StuckMonitorHandle {
    /// Take ownership of every report accumulated so far.
    /// Returns an empty Vec when no stuck events have been observed.
    /// Safe to call any number of times.
    pub(crate) fn drain(&self) -> Vec<StuckReport> {
        let mut guard = self
            .monitor
            .reports
            .lock()
            .expect("stuck-monitor reports mutex poisoned");
        std::mem::take(&mut *guard)
    }
}

impl Drop for StuckMonitorHandle {
    fn drop(&mut self) {
        // Flip the shutdown flag so the polling loop exits at the
        // next sleep boundary, then join the thread. The thread
        // sleeps at most `poll_interval_ms`, so the join latency is
        // bounded.
        self.monitor.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            // Join errors only on thread panic; log and continue —
            // a panicked thread already has the shutdown flag
            // honored implicitly (it's no longer running).
            if let Err(e) = handle.join() {
                tracing::warn!(?e, "stuck-monitor polling thread panicked");
            }
        }
    }
}

/// Spawn a background polling thread that watches each pid in
/// `pids` for stuck workers. Returns a [`StuckMonitorHandle`] whose
/// [`StuckMonitorHandle::drain`] yields the accumulated reports
/// and whose `Drop` stops the thread.
///
/// The poll cadence is read from [`crate::KTSTR_STUCK_POLL_MS_ENV`]
/// (empty / unset / 0 / unparseable → [`DEFAULT_POLL_INTERVAL_MS`]).
///
/// Empty `pids` is accepted: the polling thread starts but its
/// per-iteration loop is a no-op. The caller is expected to gate
/// the `spawn_monitor` call on `!pids.is_empty()` for the common
/// case so no thread is spawned at all; the no-op branch exists so
/// the constructor itself is total.
pub(crate) fn spawn_monitor(pids: &[libc::pid_t]) -> Result<StuckMonitorHandle> {
    let interval = resolve_poll_interval();
    let pids: Vec<libc::pid_t> = pids.to_vec();
    let shutdown = Arc::new(AtomicBool::new(false));
    let reports: Arc<Mutex<Vec<StuckReport>>> = Arc::new(Mutex::new(Vec::new()));

    let thread_shutdown = Arc::clone(&shutdown);
    let thread_reports = Arc::clone(&reports);

    let thread = thread::Builder::new()
        .name("ktstr-stuck-mon".to_string())
        .spawn(move || {
            poll_loop(pids, interval, thread_shutdown, thread_reports);
        })
        .map_err(|e| anyhow::anyhow!("failed to spawn stuck-monitor thread: {e}"))?;

    Ok(StuckMonitorHandle {
        monitor: StuckMonitor { shutdown, reports },
        thread: Some(thread),
    })
}

/// Resolve the poll interval from [`KTSTR_STUCK_POLL_MS_ENV`].
///
/// Contract: empty / unset / `0` / unparseable falls back to
/// [`DEFAULT_POLL_INTERVAL_MS`]. Any positive `u64` value is honored
/// verbatim. Mirrors the empty-string-as-unset contract documented
/// on the env-var const.
fn resolve_poll_interval() -> Duration {
    let ms = std::env::var(KTSTR_STUCK_POLL_MS_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_POLL_INTERVAL_MS);
    Duration::from_millis(ms)
}

/// Polling loop body. Runs until `shutdown` flips. Each iteration
/// samples every pid in `pids`, advances per-pid sliding windows,
/// and appends a [`StuckReport`] when the predicate trips.
fn poll_loop(
    pids: Vec<libc::pid_t>,
    interval: Duration,
    shutdown: Arc<AtomicBool>,
    reports: Arc<Mutex<Vec<StuckReport>>>,
) {
    // Per-pid state: the sliding sample window plus an "armed"
    // flag that prevents a permanently-stuck pid from spamming the
    // report list every iteration. The pid stays "disarmed" until
    // forward progress is observed (any non-zero delta in either
    // counter), at which point it re-arms and can fire again on
    // the next stuck window.
    let mut windows: Vec<(libc::pid_t, VecDeque<SchedSample>, bool)> = pids
        .iter()
        .map(|&p| (p, VecDeque::with_capacity(STUCK_WINDOW), true))
        .collect();

    while !shutdown.load(Ordering::SeqCst) {
        for (pid, window, armed) in windows.iter_mut() {
            let sample = match read_sched_sample(*pid) {
                Some(s) => s,
                // pid has exited or /proc/<pid>/sched is unreadable
                // — clear the window so we don't false-fire on a
                // pre-exit flat tail, and continue.
                None => {
                    window.clear();
                    continue;
                }
            };
            if process_iteration(sample, window, armed) {
                let samples: Vec<SchedSample> = window.iter().copied().collect();
                let comm =
                    read_comm(*pid).unwrap_or_else(|reason| format!("[unreadable: {reason}]"));
                let diagnostic = capture_diagnostic(*pid);
                let report = StuckReport {
                    pid: *pid,
                    comm,
                    samples,
                    captured_at: Instant::now(),
                    diagnostic,
                };
                {
                    let mut guard = reports
                        .lock()
                        .expect("stuck-monitor reports mutex poisoned");
                    guard.push(report);
                }
            }
        }
        // Sleep `interval` total but honor shutdown promptly: chunk
        // the sleep into 50 ms slices so the worst-case shutdown
        // latency is ~50 ms instead of `interval`. The cap exists
        // because `thread::sleep` is uninterruptible from the
        // outside — a sole `thread::sleep(interval)` for an
        // interval of (say) 5 s would block shutdown for up to 5
        // seconds; the 50 ms cap keeps shutdown latency bounded
        // independently of the polling cadence.
        let chunk = Duration::from_millis(50).min(interval);
        let mut remaining = interval;
        while remaining > Duration::ZERO && !shutdown.load(Ordering::SeqCst) {
            let slice = chunk.min(remaining);
            thread::sleep(slice);
            remaining = remaining.saturating_sub(slice);
        }
    }
}

/// Per-pid polling-iteration state update.
///
/// Folds the four-step transition the poll loop performs for each
/// pid each iteration into a single testable function:
///
/// 1. Push `sample` onto `window`.
/// 2. Evict the oldest sample if the window now exceeds
///    [`STUCK_WINDOW`].
/// 3. Re-arm `armed` when the two most recent samples show
///    forward progress in either counter (`nr_switches` or
///    `sum_exec_runtime_ns`).
/// 4. When `armed` AND [`stuck_predicate`] fires, return `true`
///    (caller emits a [`StuckReport`]) and disarm.
///
/// Returns `true` when this iteration produced a stuck fire — the
/// caller is expected to construct a [`StuckReport`] from the
/// window contents and the pid. Returns `false` otherwise.
///
/// Extracted from [`poll_loop`] so unit tests can exercise the
/// re-arm + spawn-gate semantics without spinning up the polling
/// thread (which would require real PIDs + readable
/// `/proc/<pid>/sched`).
pub(crate) fn process_iteration(
    sample: SchedSample,
    window: &mut VecDeque<SchedSample>,
    armed: &mut bool,
) -> bool {
    window.push_back(sample);
    while window.len() > STUCK_WINDOW {
        window.pop_front();
    }
    // Re-arm whenever forward progress is observed between the
    // two most recent samples — even if armed already (cheap
    // branch).
    if window.len() >= 2 {
        let last = window[window.len() - 1];
        let prev = window[window.len() - 2];
        if last.nr_switches != prev.nr_switches
            || last.sum_exec_runtime_ns != prev.sum_exec_runtime_ns
        {
            *armed = true;
        }
    }
    if *armed && stuck_predicate(window.make_contiguous()) {
        *armed = false;
        true
    } else {
        false
    }
}

/// Stuck predicate: returns `true` when `samples.len() >=
/// STUCK_WINDOW` AND every consecutive pair has both `nr_switches`
/// delta == 0 AND `sum_exec_runtime_ns` delta == 0. A window
/// shorter than [`STUCK_WINDOW`] never fires (insufficient signal).
///
/// Extracted as a free function so a unit test can exercise the
/// predicate directly with synthetic [`SchedSample`] sequences
/// without spinning up the polling thread.
pub fn stuck_predicate(samples: &[SchedSample]) -> bool {
    if samples.len() < STUCK_WINDOW {
        return false;
    }
    for pair in samples.windows(2) {
        if pair[0].nr_switches != pair[1].nr_switches
            || pair[0].sum_exec_runtime_ns != pair[1].sum_exec_runtime_ns
        {
            return false;
        }
    }
    true
}

/// Parse `/proc/<pid>/sched` content into a [`SchedSample`].
///
/// The kernel format (per `kernel/sched/debug.c::proc_sched_show_task`)
/// is:
///
/// ```text
/// <comm> (<pid>, #threads: N)
/// ---------------------------
/// se.exec_start                  :       12345.6789
/// ...
/// nr_switches                    :              42
/// nr_voluntary_switches          :              30
/// ...
/// se.sum_exec_runtime            :        1234.567890
/// ...
/// ```
///
/// Each "key : value" line uses arbitrary whitespace around the
/// `:`. The parser extracts the two named keys and ignores
/// everything else; missing keys yield `None`.
///
/// `se.sum_exec_runtime` is emitted as fractional milliseconds by
/// `kernel/sched/debug.c::proc_sched_show_task` via the `PN` macro
/// (`SPLIT_NS` divides the ns value by 1_000_000 and formats with 6
/// decimal places). This parser reads it as an f64 in ms and
/// multiplies by 1_000_000 to recover nanoseconds.
pub fn parse_sched_file(content: &str) -> Option<(u64, u64)> {
    let mut nr_switches: Option<u64> = None;
    let mut sum_exec_runtime_ns: Option<u64> = None;
    for line in content.lines() {
        // Lines we care about are `<key> : <value>`. Skip the
        // header and dashed separator.
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "nr_switches" => {
                nr_switches = value.parse::<u64>().ok();
            }
            "se.sum_exec_runtime" => {
                // Kernel emits as fractional milliseconds (e.g.
                // `1234.567890`) per `kernel/sched/debug.c`'s
                // `SPLIT_NS` macro — that's `value_ns / 1_000_000`
                // formatted with 6 decimal places. Recover ns by
                // parsing as f64 and multiplying back.
                //
                // Key carries the `se.` prefix because the kernel
                // formats this field via the `PN(se.sum_exec_runtime)`
                // macro at `kernel/sched/debug.c`'s
                // `proc_sched_show_task` — `PN` stringifies the FULL
                // expression including the `se.` qualifier. Matching
                // the bare `sum_exec_runtime` (without prefix) would
                // silently drop every real-kernel sample.
                sum_exec_runtime_ns = value
                    .parse::<f64>()
                    .ok()
                    .map(|ms| (ms * 1_000_000.0) as u64);
            }
            _ => {}
        }
        if nr_switches.is_some() && sum_exec_runtime_ns.is_some() {
            break;
        }
    }
    match (nr_switches, sum_exec_runtime_ns) {
        (Some(n), Some(r)) => Some((n, r)),
        _ => None,
    }
}

/// Read `/proc/<pid>/sched` and return a [`SchedSample`]. Returns
/// `None` when the file is unreadable (pid exited, EACCES, parse
/// failure).
fn read_sched_sample(pid: libc::pid_t) -> Option<SchedSample> {
    let content = std::fs::read_to_string(format!("/proc/{pid}/sched")).ok()?;
    let (nr_switches, sum_exec_runtime_ns) = parse_sched_file(&content)?;
    Some(SchedSample {
        nr_switches,
        sum_exec_runtime_ns,
        captured_at: Instant::now(),
    })
}

/// Read `/proc/<pid>/comm` and return the task name with the
/// trailing newline stripped. Returns the IO error's `to_string`
/// when the read fails.
fn read_comm(pid: libc::pid_t) -> std::result::Result<String, String> {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|s| s.trim_end_matches('\n').to_string())
        .map_err(|e| e.to_string())
}

/// Capture a [`StuckDiagnostic`] for a stuck pid. Every field is
/// read independently; an unreadable field becomes a
/// `"[unreadable: <reason>]"` stand-in rather than aborting the
/// snapshot — `/proc/<pid>/stack` in particular is privileged and
/// commonly absent.
fn capture_diagnostic(pid: libc::pid_t) -> StuckDiagnostic {
    let wchan = read_proc_field(pid, "wchan");
    let syscall = read_proc_field(pid, "syscall");
    let status_full = read_proc_field(pid, "status");
    let state = extract_state_letter(&status_full);
    let cgroup = read_proc_field(pid, "cgroup");
    // `/proc/<pid>/stack` is privileged; treat absence as the
    // expected case (None) rather than a read failure.
    let stack = std::fs::read_to_string(format!("/proc/{pid}/stack")).ok();
    let host_loadavg = std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim_end_matches('\n').to_string())
        .unwrap_or_else(|e| format!("[unreadable: {e}]"));
    StuckDiagnostic {
        wchan,
        syscall,
        state,
        stack,
        status_full,
        cgroup,
        host_loadavg,
    }
}

/// Read `/proc/<pid>/<field>` and return its contents trimmed.
/// Failures (EACCES, ENOENT) become a `"[unreadable: <reason>]"`
/// stand-in so the diagnostic always carries a value per field —
/// the operator triaging a stuck worker can tell apart "kernel didn't
/// expose it" from "monitor failed to read it".
fn read_proc_field(pid: libc::pid_t, field: &str) -> String {
    match std::fs::read_to_string(format!("/proc/{pid}/{field}")) {
        Ok(s) => s.trim_end_matches('\n').to_string(),
        Err(e) => format!("[unreadable: {e}]"),
    }
}

/// Parse the `State:` line out of a `/proc/<pid>/status` body and
/// return just the single-letter state code (e.g. `"S"` for
/// sleeping, `"R"` for running, `"D"` for uninterruptible sleep).
/// Falls back to `"?"` when the line is absent or malformed.
fn extract_state_letter(status: &str) -> String {
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("State:") {
            let rest = rest.trim();
            // Format: `<letter> (<description>)` per
            // `fs/proc/array.c::task_state_array`. Take the first
            // whitespace-delimited token.
            if let Some(letter) = rest.split_whitespace().next() {
                return letter.to_string();
            }
        }
    }
    "?".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_sched_file` recovers both counters from a realistic
    /// `/proc/<pid>/sched` fragment containing the headers and
    /// extra fields the kernel emits. Pins the parser against the
    /// arbitrary-whitespace + interleaved-other-lines reality of
    /// kernel output, AND against the exact `se.sum_exec_runtime`
    /// key the kernel emits (prefix included).
    #[test]
    fn parse_sched_file_extracts_signals() {
        let content = "\
worker_0 (12345, #threads: 1)
-------------------------------------------------------------------
se.exec_start                                :        123456789.123456
se.vruntime                                  :              789.012345
se.sum_exec_runtime                          :          1234.567890
nr_migrations                                :                       7
nr_switches                                  :                      42
nr_voluntary_switches                        :                      30
nr_involuntary_switches                      :                      12
clock-delta                                  :                       0
";
        let parsed = parse_sched_file(content).expect("both fields present");
        assert_eq!(parsed.0, 42, "nr_switches");
        // 1234.567890 ms = 1_234_567_890 ns.
        assert_eq!(parsed.1, 1_234_567_890, "sum_exec_runtime in ns");
    }

    /// Defense-in-depth against kernel-key-format regression: read
    /// the LIVE `/proc/self/sched` for the test process and verify
    /// the parser extracts plausible counters. A future
    /// kernel-format change (e.g. dropping the `se.` prefix,
    /// renaming `nr_switches`, switching to a non-SPLIT_NS time
    /// format) would fail this test loudly instead of silently
    /// disabling the monitor.
    #[test]
    fn parse_sched_file_handles_live_proc_self_sched() {
        let Ok(content) = std::fs::read_to_string("/proc/self/sched") else {
            // /proc not available (sandboxed test runner without
            // procfs) — skip rather than fail. The fixture-based
            // test above already covers the parser; this test's
            // value is defense against real-kernel drift.
            return;
        };
        let parsed = parse_sched_file(&content)
            .expect("live /proc/self/sched MUST parse — kernel-format regression");
        // nr_switches MUST be >= 1 by the time this test runs (the
        // process has at least spawned + scheduled + read the file).
        assert!(
            parsed.0 >= 1,
            "live nr_switches must be >= 1, got {}",
            parsed.0
        );
        // sum_exec_runtime_ns MUST be > 0 — process has accumulated
        // some CPU time to reach this assertion.
        assert!(
            parsed.1 > 0,
            "live sum_exec_runtime_ns must be > 0, got {}",
            parsed.1
        );
    }

    fn s(nr: u64, ns: u64) -> SchedSample {
        SchedSample {
            nr_switches: nr,
            sum_exec_runtime_ns: ns,
            captured_at: Instant::now(),
        }
    }

    /// Window with W consecutive identical samples → predicate
    /// fires. Pins the core stuck detection contract.
    #[test]
    fn stuck_predicate_fires_after_w_samples_no_delta() {
        let samples: Vec<SchedSample> = (0..STUCK_WINDOW).map(|_| s(100, 5_000)).collect();
        assert!(
            stuck_predicate(&samples),
            "all-flat window of W samples must fire"
        );
    }

    /// Window with ANY consecutive-pair delta > 0 → predicate
    /// does NOT fire, even if every other pair is flat. Pins the
    /// "any progress disqualifies" semantic.
    #[test]
    fn stuck_predicate_skips_when_delta_present() {
        // Three flat samples then one with a switch — the
        // last-pair delta is non-zero so predicate stays false.
        let samples = vec![s(100, 5_000), s(100, 5_000), s(100, 5_000), s(101, 5_000)];
        assert!(
            !stuck_predicate(&samples),
            "any non-zero delta in any consecutive pair must keep predicate false",
        );
        // Same for an exec_runtime move.
        let samples = vec![s(100, 5_000), s(100, 5_000), s(100, 5_001), s(100, 5_001)];
        assert!(
            !stuck_predicate(&samples),
            "exec_runtime delta in any pair must keep predicate false",
        );
        // And a window shorter than W never fires regardless.
        let short: Vec<SchedSample> = (0..STUCK_WINDOW - 1).map(|_| s(100, 5_000)).collect();
        assert!(
            !stuck_predicate(&short),
            "window shorter than STUCK_WINDOW must not fire (insufficient signal)",
        );
    }

    /// `capture_diagnostic` against pid 0 (an invalid pid that
    /// fails every read) returns a populated [`StuckDiagnostic`]
    /// where every string field carries the `"[unreadable: ...]"`
    /// stand-in and `stack` is `None`. Pins the
    /// graceful-degradation contract: a privileged or missing
    /// field never aborts the snapshot.
    #[test]
    fn diagnostic_capture_skips_unreadable_fields() {
        // pid 0 is reserved and /proc/0/* never exists.
        let diag = capture_diagnostic(0);
        assert!(
            diag.wchan.starts_with("[unreadable:"),
            "wchan must degrade: got {:?}",
            diag.wchan,
        );
        assert!(
            diag.syscall.starts_with("[unreadable:"),
            "syscall must degrade: got {:?}",
            diag.syscall,
        );
        assert!(
            diag.status_full.starts_with("[unreadable:"),
            "status must degrade: got {:?}",
            diag.status_full,
        );
        assert!(
            diag.cgroup.starts_with("[unreadable:"),
            "cgroup must degrade: got {:?}",
            diag.cgroup,
        );
        // status_full degraded → state extraction yields "?".
        assert_eq!(diag.state, "?", "unreadable status → state = ?");
        // /proc/0/stack is also absent.
        assert!(diag.stack.is_none(), "missing stack must remain None");
        // host_loadavg should succeed on any healthy Linux host
        // even when the pid is bogus; either it reads or carries
        // the unreadable stand-in.
        assert!(
            !diag.host_loadavg.is_empty(),
            "host_loadavg must always populate (success OR stand-in)",
        );
    }

    /// Sliding window honors [`STUCK_WINDOW`]: once full, the
    /// oldest sample is evicted as new ones arrive, so the
    /// predicate always sees the most-recent W samples. Pins the
    /// "predicate sees last W" guarantee.
    #[test]
    fn ring_buffer_sliding_window_correctness() {
        // Simulate the poll loop's window management without
        // spinning the actual thread: push samples through a
        // VecDeque with the same eviction policy and verify the
        // contents.
        let mut window: VecDeque<SchedSample> = VecDeque::with_capacity(STUCK_WINDOW);
        // Push 6 samples (more than W=4): values 0..6.
        for i in 0..(STUCK_WINDOW + 2) {
            window.push_back(s(i as u64, i as u64 * 10));
            while window.len() > STUCK_WINDOW {
                window.pop_front();
            }
        }
        assert_eq!(
            window.len(),
            STUCK_WINDOW,
            "window size must stay at STUCK_WINDOW after overflow",
        );
        // The oldest two were evicted; the head should be sample
        // index 2 (out of 0..6 pushed).
        let head = window.front().expect("window non-empty");
        assert_eq!(
            head.nr_switches, 2,
            "oldest sample must be index 2 after 2 evictions"
        );
        let tail = window.back().expect("window non-empty");
        assert_eq!(
            tail.nr_switches,
            (STUCK_WINDOW + 1) as u64,
            "newest sample must be the last pushed (index W+1)",
        );
        // None of these consecutive pairs is flat (every pair
        // differs by 1), so the predicate must NOT fire even
        // though the window is full.
        let snap: Vec<SchedSample> = window.iter().copied().collect();
        assert!(
            !stuck_predicate(&snap),
            "monotonic samples must not trip predicate"
        );
    }

    // -- process_iteration tests --
    //
    // Pin the per-pid state machine extracted from poll_loop:
    // re-arm after stuck-then-resume, spawn-gate behavior
    // when the initial window is empty, and the fire-disarm
    // sequence so a permanently-stuck pid produces ONE report
    // per stuck window, not one per polling iteration.

    /// Spawn-gate behavior: an initial window starts armed but
    /// empty. The first STUCK_WINDOW samples can't fire (predicate
    /// requires >= STUCK_WINDOW samples). Only the STUCK_WINDOW'th
    /// flat sample can trip the predicate.
    #[test]
    fn process_iteration_spawn_gate_short_window_never_fires() {
        let mut window: VecDeque<SchedSample> = VecDeque::with_capacity(STUCK_WINDOW);
        let mut armed = true;
        // Feed STUCK_WINDOW - 1 identical samples; predicate
        // sees a short window every iteration and stays false.
        for _ in 0..(STUCK_WINDOW - 1) {
            assert!(
                !process_iteration(s(100, 5_000), &mut window, &mut armed),
                "short window must not fire (spawn-gate semantic)",
            );
        }
        assert!(armed, "no resume seen → stays armed");
        // The STUCK_WINDOW'th identical sample fills the window
        // → predicate fires.
        assert!(
            process_iteration(s(100, 5_000), &mut window, &mut armed),
            "Wth flat sample fills window AND trips predicate → fire",
        );
        assert!(!armed, "fire path disarms");
    }

    /// Re-arm after stuck-then-resume: a fired stuck event disarms
    /// the pid; resume (forward progress) re-arms; a subsequent
    /// flat window fires AGAIN. A regression that
    /// dropped the re-arm branch would never produce the second
    /// report, silently hiding recurrent stuck events.
    #[test]
    fn process_iteration_rearm_after_stuck_then_resume() {
        let mut window: VecDeque<SchedSample> = VecDeque::with_capacity(STUCK_WINDOW);
        let mut armed = true;
        // Fill window with flat samples → first stuck event fires.
        for _ in 0..STUCK_WINDOW {
            process_iteration(s(100, 5_000), &mut window, &mut armed);
        }
        assert!(!armed, "after first fire, disarmed");
        // Resume: feed a sample with forward progress.
        // Predicate stays false (most-recent pair has delta), and
        // the re-arm branch flips armed back to true.
        assert!(
            !process_iteration(s(101, 5_001), &mut window, &mut armed),
            "resume sample must not fire (last pair has delta)",
        );
        assert!(armed, "resume sample must re-arm");
        // Re-stuck: feed STUCK_WINDOW identical samples. Need to
        // overwrite the resume sample first, so STUCK_WINDOW more
        // flat samples are needed to refill the window.
        let mut second_fire_iter = None;
        for i in 0..STUCK_WINDOW {
            if process_iteration(s(101, 5_001), &mut window, &mut armed) {
                second_fire_iter = Some(i);
                break;
            }
        }
        assert!(
            second_fire_iter.is_some(),
            "second stuck window must fire after re-arm; got no fire across {} iters",
            STUCK_WINDOW,
        );
        assert!(!armed, "second fire disarms");
    }

    /// Permanently-stuck pid produces EXACTLY ONE report per
    /// stuck window, not one per polling iteration. Pins the
    /// "stays disarmed until forward progress" semantic so a
    /// hung process doesn't spam the reports vec across hundreds
    /// of iterations.
    #[test]
    fn process_iteration_permanent_stuck_fires_only_once() {
        let mut window: VecDeque<SchedSample> = VecDeque::with_capacity(STUCK_WINDOW);
        let mut armed = true;
        // Fill window + fire (one report).
        let mut fire_count = 0;
        for _ in 0..STUCK_WINDOW {
            if process_iteration(s(100, 5_000), &mut window, &mut armed) {
                fire_count += 1;
            }
        }
        assert_eq!(fire_count, 1, "first window MUST fire exactly once");
        // 100 more flat samples — disarmed, predicate keeps
        // matching but fire path is gated on `armed`.
        for _ in 0..100 {
            if process_iteration(s(100, 5_000), &mut window, &mut armed) {
                fire_count += 1;
            }
        }
        assert_eq!(
            fire_count, 1,
            "permanently-stuck pid must NOT spam reports — exactly one fire across many iters",
        );
    }
}
