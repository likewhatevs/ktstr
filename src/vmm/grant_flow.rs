//! Grant-flow diagnostics (diagnostic-only, pure observation).
//!
//! The slim core that outlived the admission-wall investigation:
//! - `grants_issued`: WAITING→GRANTED transitions the coordinator scan makes.
//! - `grants_reached_held`: records published as a live HELD run claim.
//! - `grants_lost`: licensed grants a cell was told about but did not convert.
//! - `held_in_flight` / `distinct_held_cpus`: peak count of in-flight
//!   GRANTED+HELD claims and the distinct host CPUs they cover — the ramp
//!   indicator.
//! - `discover` count and wall-time: how often the placement contention-bail
//!   holder diagnostic runs and how long it takes. The bail probes only the
//!   contended LLC lock files with a non-blocking flock (no host-global
//!   `/proc/locks` walk), so `discover_ns` staying at syscall scale is the
//!   regression tripwire for that fix.
//!
//! The first three are independent rates, not a conserved flow: read them
//! against each other over time, never as `issued == reached_held + lost`. A
//! preparation intent is granted like any other waiter, but its grant converts
//! to a parked PENDING preparation rather than to HELD, and the later one-shot
//! activation that does publish HELD needs no fresh grant.
//!
//! The per-cause loss/probe/replan/guard/watermark breakdown counters this
//! module once carried answered their questions (the watermark-park and
//! EX-pin-rotation churn producers are fixed) and were removed with them.
//!
//! Entirely inert unless `KTSTR_BUILD_DIAGNOSTICS_DIR` is set (CI only): the
//! `note_*` calls are relaxed atomic updates and the expensive inputs
//! (distinct-CPU popcount, discover timing) are computed only behind
//! [`enabled`]. One aggregate line per process lands at
//! `${KTSTR_BUILD_DIAGNOSTICS_DIR}/grant-flow-<pid>.txt`, rewritten
//! periodically by the coordinator loop (outside the registry flock) and
//! finally at exit — the scan-running orchestrator's atexit line alone is
//! lost when that process is killed.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const DIR_ENV: &str = "KTSTR_BUILD_DIAGNOSTICS_DIR";

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

static GRANTS_ISSUED: AtomicU64 = AtomicU64::new(0);
static GRANTS_REACHED_HELD: AtomicU64 = AtomicU64::new(0);
static GRANTS_LOST: AtomicU64 = AtomicU64::new(0);
static HELD_IN_FLIGHT_MAX: AtomicU64 = AtomicU64::new(0);
static DISTINCT_HELD_CPUS_MAX: AtomicU64 = AtomicU64::new(0);
static DISCOVER_COUNT: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_SUM: AtomicU64 = AtomicU64::new(0);
static DISCOVER_NS_MAX: AtomicU64 = AtomicU64::new(0);
static ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);

fn dir() -> Option<&'static PathBuf> {
    DIR.get_or_init(|| {
        std::env::var_os(DIR_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

/// Whether the diagnostics sink is active. Callers gate any input computation
/// that is not itself free (distinct-CPU popcount, discover timing) on this.
pub(crate) fn enabled() -> bool {
    dir().is_some()
}

fn ensure_atexit() {
    if ATEXIT_REGISTERED.swap(true, Ordering::Relaxed) {
        return;
    }
    // SAFETY: `emit` is a plain extern "C" fn with no arguments, registered at
    // most once. `atexit` only stores the pointer.
    unsafe {
        libc::atexit(emit);
    }
}

fn bump_max(cell: &AtomicU64, value: u64) {
    let mut current = cell.load(Ordering::Relaxed);
    while value > current {
        match cell.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// One WAITING→GRANTED transition issued by the coordinator scan.
pub(crate) fn note_grant_issued() {
    GRANTS_ISSUED.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// One grant that converted into a live HELD run claim.
pub(crate) fn note_grant_reached_held() {
    GRANTS_REACHED_HELD.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// One licensed grant a cell received but did not convert to HELD and
/// requeued for.
pub(crate) fn note_grant_lost() {
    GRANTS_LOST.fetch_add(1, Ordering::Relaxed);
    ensure_atexit();
}

/// Peak in-flight GRANTED+HELD claims and the distinct host CPUs they cover,
/// sampled once per authoritative grant scan. Only called behind [`enabled`].
/// Deliberately no file IO here — this runs under the registry EX flock; the
/// coordinator loop persists the image periodically outside the lock (the
/// atexit line alone is lost when the scan-running orchestrator is killed).
pub(crate) fn note_held_in_flight(held_records: u64, distinct_cpus: u64) {
    bump_max(&HELD_IN_FLIGHT_MAX, held_records);
    bump_max(&DISTINCT_HELD_CPUS_MAX, distinct_cpus);
    ensure_atexit();
}

/// One placement contention-bail holder diagnostic, and its wall-time. Since
/// the bail now probes only the contended LLC lock files with a non-blocking
/// flock (no host-global `/proc/locks` walk), this both counts how often the
/// diagnostic runs and proves the walk is gone: `discover_ns` should sit at
/// per-needle syscall scale, not the hundreds of milliseconds the seq-file read
/// cost under a many-lane flock storm. Only called behind [`enabled`].
pub(crate) fn note_discover(elapsed_ns: u64) {
    DISCOVER_COUNT.fetch_add(1, Ordering::Relaxed);
    DISCOVER_NS_SUM.fetch_add(elapsed_ns, Ordering::Relaxed);
    bump_max(&DISCOVER_NS_MAX, elapsed_ns);
    ensure_atexit();
}

fn format_line(pid: u32) -> String {
    let discover_count = DISCOVER_COUNT.load(Ordering::Relaxed);
    let discover_ns_mean = DISCOVER_NS_SUM
        .load(Ordering::Relaxed)
        .checked_div(discover_count)
        .unwrap_or(0);
    format!(
        "grant-flow: pid={pid} grants_issued={} grants_reached_held={} grants_lost={} \
         held_in_flight_max={} distinct_held_cpus_max={} discover_count={} \
         discover_ns_mean={discover_ns_mean} discover_ns_max={}\n",
        GRANTS_ISSUED.load(Ordering::Relaxed),
        GRANTS_REACHED_HELD.load(Ordering::Relaxed),
        GRANTS_LOST.load(Ordering::Relaxed),
        HELD_IN_FLIGHT_MAX.load(Ordering::Relaxed),
        DISTINCT_HELD_CPUS_MAX.load(Ordering::Relaxed),
        discover_count,
        DISCOVER_NS_MAX.load(Ordering::Relaxed),
    )
}

/// Rewrite this process's per-pid counter file with the current image. The
/// file holds exactly one line; each persist (periodic from the coordinator
/// loop, and the final `atexit` image) replaces the previous one, so readers
/// always see one complete, most-recent line per process.
pub(crate) fn persist_now() {
    let Some(dir) = dir() else {
        return;
    };
    let pid = std::process::id();
    let line = format_line(pid);
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dir.join(format!("grant-flow-{pid}.txt")))
    {
        let _ = file.write_all(line.as_bytes());
    }
}

extern "C" fn emit() {
    let _ = std::panic::catch_unwind(persist_now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_shape_has_all_fields() {
        let line = format_line(42);
        for field in [
            "grant-flow: pid=42",
            "grants_issued=",
            "grants_reached_held=",
            "grants_lost=",
            "held_in_flight_max=",
            "distinct_held_cpus_max=",
            "discover_count=",
            "discover_ns_mean=",
            "discover_ns_max=",
        ] {
            assert!(line.contains(field), "missing {field} in {line}");
        }
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn max_accumulator_keeps_the_largest() {
        let cell = AtomicU64::new(0);
        bump_max(&cell, 5);
        bump_max(&cell, 3);
        bump_max(&cell, 9);
        bump_max(&cell, 1);
        assert_eq!(cell.load(Ordering::Relaxed), 9);
    }
}
