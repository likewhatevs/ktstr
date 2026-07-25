//! Exit-phase admission-teardown telemetry (diagnostic-only, pure observation).
//!
//! Pins where a finished VM cell spends wall between `release_run_locks` and
//! process exit. Labelled stamps feed an in-memory aggregate (delta since
//! release, slowest single step) and write nothing themselves; one aggregate
//! line per process is appended at exit, so ~1100 cells/lane cost ~1100 lines
//! in ONE shared file instead of ~1100 per-pid step logs. The 30s coordinator
//! fallback accusation keeps its own line, deduplicated to the first block
//! per process.
//!
//! Entirely inert unless `KTSTR_BUILD_DIAGNOSTICS_DIR` is set (CI only), the same
//! idiom as [`super::AdmissionTiming`] and the coordinator wake counter: an
//! unset sink costs one cached env lookup plus a relaxed atomic load per stamp
//! and changes no wait/wake semantics. Lines land in
//! `${KTSTR_BUILD_DIAGNOSTICS_DIR}/exit-timing.log`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg_attr(test, allow(dead_code))]
const EXIT_TIMING_DIR_ENV: &str = "KTSTR_BUILD_DIAGNOSTICS_DIR";

/// Cached diagnostics sink; resolved once. `None` keeps the whole path inert.
/// Unreachable in test builds, where the test override is authoritative.
#[cfg_attr(test, allow(dead_code))]
static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
/// Process-relative instant (ns since process start) of run-locks release, or 0
/// before release. Ordinary [`stamp`] calls emit only once this is set: the seam
/// is by definition post-release.
static RELEASED_NS: AtomicU64 = AtomicU64::new(0);
/// Instant of the most recent stamp, for the aggregate `handles_done` field.
static LAST_STAMP_NS: AtomicU64 = AtomicU64::new(0);
/// Previous stamp's delta-since-release, so each step's own cost is
/// `delta - prev_delta` rather than the cumulative delta.
static PREV_DELTA_NS: AtomicU64 = AtomicU64::new(0);
/// Slowest single step observed so far: `(label, step_ns)`.
static SLOWEST: Mutex<(String, u64)> = Mutex::new((String::new(), 0));
static ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);
/// First-fallback-accusation dedupe: see [`stamp_fallback_block`].
static FALLBACK_BLOCK_EMITTED: AtomicBool = AtomicBool::new(false);

/// Test-only sink override so the emission path is exercisable without racing
/// the process-global `DIR` OnceLock (which cannot be reset between tests).
/// In test builds this override is authoritative: `None` means DISABLED, never
/// "fall through to the environment" — CI sets the diagnostics env var
/// ambiently for every test process, and consulting it here would make the
/// unit tests' inert-when-unset assertions environment-dependent.
#[cfg(test)]
static TEST_DIR_OVERRIDE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Resolve the diagnostics directory from an environment value. Empty or absent
/// disables the sink. Split out so the gating is unit-testable without touching
/// the process-global `OnceLock`.
fn resolve_dir(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
fn dir() -> Option<PathBuf> {
    // Tests inject a per-test sink without racing the write-once `DIR` OnceLock.
    TEST_DIR_OVERRIDE
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or(None)
}

#[cfg(not(test))]
fn dir() -> Option<PathBuf> {
    DIR.get_or_init(|| resolve_dir(std::env::var_os(EXIT_TIMING_DIR_ENV)))
        .clone()
}

fn now_ns() -> u64 {
    super::process_start_elapsed()
        .unwrap_or_default()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Wall-clock nanoseconds since the UNIX epoch, for cross-process ordering.
/// The fallback accusation uses this instead of [`now_ns`]: an acquisition-phase
/// coordinator has no `record_process_start` anchor (that is stamped later, in
/// the post-exec test harness), so its process-relative clock reads 0 and its
/// accusations would be untotally-ordered against one another.
fn wall_now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Coordinator fallback-tick accusation line. `wall_ns` is UNIX-epoch
/// wall-clock (see [`wall_now_ns`]) so accusations from different acquisition-
/// phase processes are totally orderable. Factored for shape tests.
fn format_fallback_block_line(pid: u32, wall_ns: u64, blocked_on: &str, holder: &str) -> String {
    format!(
        "exit-fallback-block: pid={pid} wall_ns={wall_ns} \
         blocked_on={blocked_on} holder={holder}\n"
    )
}

/// Aggregate summary line. Factored for shape tests.
fn format_aggregate_line(
    pid: u32,
    released_ns: u64,
    handles_done_ns: u64,
    exit_ns: u64,
    slowest_label: &str,
    slowest_step_ns: u64,
) -> String {
    let label = if slowest_label.is_empty() {
        "none"
    } else {
        slowest_label
    };
    format!(
        "exit-timing: pid={pid} released={released_ns} handles_done={handles_done_ns} \
         exit={exit_ns} slowest_step={label},{slowest_step_ns}\n"
    )
}

fn append(dir: &Path, name: &str, line: &str) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(name))
    {
        let _ = file.write_all(line.as_bytes());
    }
}

fn log_name() -> String {
    // One shared O_APPEND file for every process: each line is far below
    // PIPE_BUF, so concurrent appends stay whole, and the artifact set stays
    // one file per lane instead of one per cell.
    "exit-timing.log".to_string()
}

/// Record the run-locks release instant and arm the exit aggregate. Called once
/// from `release_run_locks`, the moment the physical run reservation drops.
/// Inert when the diagnostics sink is unset.
pub(crate) fn mark_released() {
    if dir().is_none() {
        return;
    }
    let ns = now_ns().max(1);
    // A cell releases exactly one run reservation; the first release wins.
    let _ = RELEASED_NS.compare_exchange(0, ns, Ordering::Relaxed, Ordering::Relaxed);
    LAST_STAMP_NS.store(ns, Ordering::Relaxed);
    register_atexit();
}

/// True only when the sink is set AND this cell has passed run-locks release —
/// the exact window where [`stamp`] emits. Now that every stamp carries only a
/// static label, no production caller has to pre-check it; the tests assert the
/// window through this predicate.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn post_release_active() -> bool {
    dir().is_some() && RELEASED_NS.load(Ordering::Relaxed) != 0
}

/// Stamp one labelled point in the post-release teardown. No-op before
/// [`mark_released`] or when the sink is unset.
pub(crate) fn stamp(label: &str) {
    // Cheapest gate first: `RELEASED_NS` is 0 both before this cell's release
    // AND whenever the sink is unset (`mark_released` never ran), so the common
    // no-op path is one relaxed load with no directory clone.
    let released = RELEASED_NS.load(Ordering::Relaxed);
    if released == 0 {
        return;
    }
    // Aggregate-only: stamps feed the slowest-step attribution the atexit
    // line reports and write no per-step lines of their own — the per-step
    // channel cost ~1100 tiny per-pid files per lane for a question the
    // aggregate answers.
    let now = now_ns();
    let delta = now.saturating_sub(released);
    let prev = PREV_DELTA_NS.swap(delta, Ordering::Relaxed);
    let step = delta.saturating_sub(prev);
    LAST_STAMP_NS.store(now, Ordering::Relaxed);
    if let Ok(mut slowest) = SLOWEST.lock()
        && step > slowest.1
    {
        *slowest = (label.to_string(), step);
    }
}

/// Accusation line for a coordinator fallback tick: name the blocking resources
/// and their holder so every 30s [`super::host_topology::protocol`] backstop
/// tick becomes attributable. Best-effort; sink-gated but NOT release-gated (a
/// coordinator rides fallbacks during acquisition, before this process reaches
/// its own release).
pub(crate) fn stamp_fallback_block(blocked_on: &str, holder: &str) {
    let Some(dir) = dir() else {
        return;
    };
    // One accusation per process: the first blocked tick names the wedge; a
    // long wedge would otherwise append a line every 30s from every blocked
    // coordinator.
    if FALLBACK_BLOCK_EMITTED.swap(true, Ordering::Relaxed) {
        return;
    }
    append(
        &dir,
        &log_name(),
        &format_fallback_block_line(std::process::id(), wall_now_ns(), blocked_on, holder),
    );
}

#[cfg(not(test))]
fn register_atexit() {
    if ATEXIT_REGISTERED.swap(true, Ordering::Relaxed) {
        return;
    }
    // SAFETY: `emit_aggregate` is a plain `extern "C"` function that performs
    // only best-effort file IO behind a panic guard and is registered exactly
    // once (the swap above). `atexit` merely stores the pointer.
    unsafe {
        libc::atexit(emit_aggregate);
    }
}

/// Tests drive [`emit_aggregate`] directly through [`reset_for_tests`];
/// registering a real process-exit hook from the test binary would fire against
/// a torn-down override sink, so this just marks the arm as taken.
#[cfg(test)]
fn register_atexit() {
    ATEXIT_REGISTERED.store(true, Ordering::Relaxed);
}

/// Reset all process-global state and point the sink at `dir` (or disable it
/// with `None`). Test-only: the emission path is otherwise unreachable from a
/// unit test because `DIR` is a write-once `OnceLock`.
#[cfg(test)]
fn reset_for_tests(dir: Option<PathBuf>) {
    RELEASED_NS.store(0, Ordering::Relaxed);
    FALLBACK_BLOCK_EMITTED.store(false, Ordering::Relaxed);
    LAST_STAMP_NS.store(0, Ordering::Relaxed);
    PREV_DELTA_NS.store(0, Ordering::Relaxed);
    ATEXIT_REGISTERED.store(false, Ordering::Relaxed);
    *SLOWEST.lock().expect("SLOWEST poisoned") = (String::new(), 0);
    *TEST_DIR_OVERRIDE.lock().expect("override poisoned") = dir;
}

/// atexit hook: emit the aggregate summary. `extern "C"`, so it must never
/// unwind — every fallible step is best-effort and a panic guard wraps the body.
extern "C" fn emit_aggregate() {
    let _ = std::panic::catch_unwind(|| {
        let Some(dir) = dir() else {
            return;
        };
        let released = RELEASED_NS.load(Ordering::Relaxed);
        if released == 0 {
            return;
        }
        let handles_done = LAST_STAMP_NS.load(Ordering::Relaxed);
        let exit = now_ns();
        let (label, step) = SLOWEST
            .lock()
            .map(|slowest| slowest.clone())
            .unwrap_or_default();
        append(
            &dir,
            &log_name(),
            &format_aggregate_line(
                std::process::id(),
                released,
                handles_done,
                exit,
                &label,
                step,
            ),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that mutate the process-global emission state.
    static STATE_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn stamps_feed_the_aggregate_without_per_step_lines() {
        let _serial = STATE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().expect("exit-timing tempdir");
        reset_for_tests(Some(tmp.path().to_path_buf()));
        let log = tmp.path().join(log_name());

        // Sink is set but release has not happened: ordinary stamps are inert.
        assert!(!post_release_active(), "not active before release");
        stamp("before_release");
        assert!(
            !log.exists(),
            "an ordinary stamp before mark_released must not emit",
        );

        // Release arms the seam; stamps feed the in-memory aggregate but
        // write no per-step lines — only the atexit aggregate reaches the
        // shared file.
        mark_released();
        assert!(post_release_active(), "active once released");
        stamp("run_locks_dropped");
        stamp("eval_done");
        assert!(!log.exists(), "per-step stamps must not write file lines");
        emit_aggregate();

        let body = std::fs::read_to_string(&log).expect("exit-timing log written");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1, "exactly one aggregate line: {body:?}");
        assert!(
            lines[0].starts_with("exit-timing: ") && lines[0].contains("slowest_step="),
            "the aggregate closes the log with the slowest-step field: {:?}",
            lines[0],
        );

        // Disabling the sink makes even a post-release stamp inert.
        reset_for_tests(None);
        mark_released();
        stamp("after_disable");
        assert!(!post_release_active(), "inert when the sink is unset");

        reset_for_tests(None);
    }

    #[test]
    fn fallback_block_is_sink_gated_but_not_release_gated() {
        let _serial = STATE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().expect("exit-timing tempdir");
        reset_for_tests(Some(tmp.path().to_path_buf()));
        let log = tmp.path().join(log_name());

        // No mark_released: a fallback accusation still emits (a coordinator
        // rides fallbacks during acquisition, before its own release), and a
        // repeated block from the same process is deduplicated to the first.
        assert!(!post_release_active());
        stamp_fallback_block("cpu=3,permit=7", "pid=42 ticket=9");
        stamp_fallback_block("cpu=3,permit=7", "pid=42 ticket=9");
        let body = std::fs::read_to_string(&log).expect("fallback line written");
        assert_eq!(
            body.matches("exit-fallback-block: ").count(),
            1,
            "repeated blocks dedupe to the first accusation: {body:?}",
        );
        assert!(
            body.contains("blocked_on=cpu=3,permit=7") && body.contains("holder=pid=42 ticket=9"),
            "fallback accusation emits pre-release: {body:?}",
        );

        reset_for_tests(None);
    }

    #[test]
    fn resolve_dir_gates_on_present_nonempty_env() {
        assert_eq!(resolve_dir(None), None, "absent env disables the sink");
        assert_eq!(
            resolve_dir(Some(std::ffi::OsString::from(""))),
            None,
            "empty env disables the sink",
        );
        assert_eq!(
            resolve_dir(Some(std::ffi::OsString::from("/tmp/diag"))),
            Some(PathBuf::from("/tmp/diag")),
            "a non-empty path enables the sink",
        );
    }

    #[test]
    fn fallback_block_line_shape() {
        assert_eq!(
            format_fallback_block_line(9, 2_000, "permit=41,permit=42", "pid=1234 ticket=56"),
            "exit-fallback-block: pid=9 wall_ns=2000 \
             blocked_on=permit=41,permit=42 holder=pid=1234 ticket=56\n",
        );
    }

    #[test]
    fn aggregate_line_shape_and_empty_label_fallback() {
        assert_eq!(
            format_aggregate_line(3, 100, 200, 900, "eval_done", 650),
            "exit-timing: pid=3 released=100 handles_done=200 exit=900 \
             slowest_step=eval_done,650\n",
        );
        assert_eq!(
            format_aggregate_line(3, 100, 100, 100, "", 0),
            "exit-timing: pid=3 released=100 handles_done=100 exit=100 slowest_step=none,0\n",
            "an empty slowest label renders as `none`",
        );
    }
}
