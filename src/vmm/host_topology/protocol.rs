//! Cross-invocation acquisition protocol — the "second-order
//! scheduler" under nextest.
//!
//! nextest is a pure spawner: it launches every test process at once
//! and retries failures. The lock dir is the actual scheduler: it
//! decides which cells EXECUTE against host capacity (the guest
//! scheduler under test being the third layer down). Every ktstr
//! invocation sharing a `KTSTR_LOCK_DIR` — disjoint processes,
//! colocated runner units, concurrent nextest runs — coordinates
//! through the files in that directory and nothing else, so each
//! participant acts on a point-in-time view of global state. This
//! module makes that partial knowledge harmless via four rules:
//!
//! 1. **Global head license.** A single queue flock
//!    ([`queue_lock_path`]) serializes CONTENDED acquirers across all
//!    invocations: whoever holds it (the "head") is the only agent
//!    anywhere allowed to hold a PARTIAL resource-lock set while
//!    waiting for more. Deadlock needs two holders-waiting; the queue
//!    forbids a second, so the head's hold-and-wait is safe by
//!    construction, and its acquisition completes because it only
//!    ever accumulates toward a live-planned target. This is the
//!    protocol's core safety invariant — everything else leans on it.
//! 2. **Head claim visibility.** The head PUBLISHES its current
//!    target set ([`publish_claim`]) so fast-path planners in other
//!    processes SUBTRACT it from their view of free capacity
//!    ([`read_live_claim`]) instead of sniping the very slots the
//!    head is waiting on. The claim is advisory and crash-safe: its
//!    liveness is an EX flock on a marker file
//!    ([`head_marker_path`]) that vanishes with the process, so a
//!    dead head's stale manifest is ignored. A fast-path caller
//!    acting on a stale read at worst bounces once (rule 3 bounds
//!    the damage).
//! 3. **Fast-path grabs are all-or-nothing in canonical order.**
//!    Non-contended acquirers try their whole planned set
//!    non-blocking, in the global lock order (LLC locks by index,
//!    then CPU locks by index), and release EVERYTHING on any bounce.
//!    No fast-path partial ever persists, so the only partials in the
//!    system are the head's (rule 1), and overlapping fast-path sets
//!    cannot half-block each other.
//! 4. **Bounce economics.** A bounce is one failed non-blocking
//!    flock plus a release — cheap. Each consumer bounds its
//!    fast-path attempts (the default path scans its offset
//!    candidates once, perf mode tries its fixed set once, the
//!    no-perf planner runs its short TOCTOU budget) and then joins
//!    the queue: persistent contention belongs in the queue, both to
//!    stop wasted work and because queued waiters are what the
//!    head-license fairness protects.
//!
//! The objective function is END-TO-END SUITE TIME plus TEST
//! EXECUTION QUALITY (every cell runs, with valid resources, and
//! produces a trustworthy verdict — no silent skips, no starvation
//! casualties, no retry storms). Queue order is arrival order because
//! it is the simplest thing the kernel's flock wait queue provides,
//! not because ordering is a goal in itself.
//!
//! **Waiting is event-driven, not polled.** A queued waiter sleeps in
//! the kernel's flock wait queue on the queue file, waking on a
//! per-tick heartbeat only to observe progress. The head sleeps on an
//! inotify watch of the lock dir ([`LockDirWatch`]): releasing any
//! lockfile closes its fd, which fires `IN_CLOSE_*` on that file, and
//! the head wakes, RE-PLANS AGAINST LIVE HOLDER STATE (a hard
//! requirement — plans are never cached across waits; the freed
//! resource may satisfy a different plan than the one the head would
//! have blocked on), and attempts acquisition again. Spurious wakes
//! (peer fast-path bounces also open/close lockfiles) are tolerated:
//! exactly one waiter re-plans per event, which is not a herd.
//!
//! **Patience is progress-based, not wall-based.** A queued waiter
//! keeps waiting as long as the queue ADVANCES (the queue lockfile's
//! holder identity changes — a head finished or a waiter ahead gave
//! up); the head keeps waiting as long as its acquisition GAINS locks.
//! Only [`ACQUIRE_NO_PROGRESS_PATIENCE`] of ZERO progress — a wedged
//! holder, genuinely pathological — produces the retryable
//! `ResourceContention` failure that nextest re-runs. This kills the
//! worst churn mode of a wall deadline: a cell expiring mid-queue and
//! re-entering at the tail on retry, paying the queue twice.
//!
//! Waiters are CHEAP by design: a queued cell is a process holding
//! its ticket-wait (a blocked flock), an inotify fd, and a handful of
//! lockfile fds — no guest memory, no vCPUs (acquisition runs before
//! VM setup; see `KtstrVm::run`). nextest can therefore admit every
//! cell at once and let this queue schedule them.
//!
//! The lock dir is already restricted to local filesystems (every
//! lockfile open routes through `crate::flock`'s remote-fs rejection),
//! which is also what inotify needs to be reliable.

use anyhow::Result;
use std::collections::BTreeSet;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::flock::{FlockMode, FlockWait, block_flock_step, try_flock};

/// No-progress window after which a waiter gives up with the
/// retryable `ResourceContention` failure. This is NOT a bound on
/// total wait time — a cell deep in a busy-but-advancing queue waits
/// far longer and is meant to (progress resets the clock). It only
/// fires when NOTHING moves: no queue turnover for a queued waiter,
/// no acquired lock for the head. That means a wedged holder — the
/// pathological case the in-cell watchdogs should have killed — and
/// the retryable-fail + nextest retry is then the correct escape.
/// 120 s comfortably exceeds any healthy cell's hold window (boot +
/// workload + teardown), so a healthy suite never trips it.
pub(crate) const ACQUIRE_NO_PROGRESS_PATIENCE: Duration = Duration::from_secs(120);

/// Heartbeat for the queued waiter's progress observation, expressed
/// as a fraction of the patience window (a quarter): the blocked
/// queue flock is interrupted this often to sample the queue holder.
/// DELIBERATELY COARSE — each interruption closes the waiter's fd and
/// re-enters the kernel wait queue at the tail, so frequent ticks
/// would churn queue order among waiters. At patience/4 (30 s in
/// production) the position loss is negligible while the patience
/// window still gets several holder samples. The wake path back to
/// execution is the kernel granting the flock (immediate), never this
/// tick.
fn queue_progress_tick() -> Duration {
    (patience() / 4).max(Duration::from_millis(100))
}

/// Fallback wake for the head's inotify sleep, bounding the staleness
/// window if an event is missed (e.g. a release racing the
/// drain-before-sleep gap). The real wake is the inotify event.
const HEAD_WAKE_FALLBACK: Duration = Duration::from_millis(500);

#[cfg(test)]
thread_local! {
    /// Test override for [`ACQUIRE_NO_PROGRESS_PATIENCE`] so patience
    /// expiry tests run in milliseconds instead of minutes.
    pub(crate) static PATIENCE_OVERRIDE: std::cell::RefCell<Option<Duration>> =
        const { std::cell::RefCell::new(None) };
}

/// Resolve the effective no-progress patience (test override aware).
pub(crate) fn patience() -> Duration {
    #[cfg(test)]
    {
        if let Some(p) = PATIENCE_OVERRIDE.with(|p| *p.borrow()) {
            return p;
        }
    }
    ACQUIRE_NO_PROGRESS_PATIENCE
}

/// Directory the protocol files live in — derived from the LLC
/// lockfile path so the test-only lock-prefix override isolates the
/// queue, marker, and claim files into the same per-test tempdir as
/// the resource lockfiles they coordinate.
fn protocol_dir() -> PathBuf {
    Path::new(&super::llc_lock_path(0))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// The acquisition-queue lockfile. Blocking `LOCK_EX` on it IS the
/// ticket queue: the kernel's flock wait queue orders contended
/// acquirers across every invocation sharing the lock dir, and a
/// waiter's ticket vanishes with its process (crash-safe). Named
/// outside the `ktstr-llc-*.lock` / `ktstr-cpu-*.lock` globs so the
/// `ktstr locks` listing's index parse never sees it.
pub(crate) fn queue_lock_path() -> String {
    protocol_dir()
        .join("ktstr-acquire-queue.lock")
        .display()
        .to_string()
}

/// The head-liveness marker. The head holds `LOCK_EX` on this file
/// for exactly as long as it is the head; readers treat the claim
/// manifest as live only while `/proc/locks` shows a holder here.
/// Flock-based so a crashed head's claim dies with it.
pub(crate) fn head_marker_path() -> String {
    protocol_dir()
        .join("ktstr-head-alive.lock")
        .display()
        .to_string()
}

/// The head's published claim manifest (JSON [`ClaimSet`]). Written
/// atomically (temp + rename) on every re-plan; advisory only —
/// correctness never depends on it, it exists so fast-path planners
/// stop sniping the head's target slots.
pub(crate) fn head_claim_path() -> String {
    protocol_dir()
        .join("ktstr-head-claim.json")
        .display()
        .to_string()
}

/// The head's published target set: which LLC locks and CPU locks it
/// is currently accumulating toward. Fast-path planners subtract
/// these from their view of free capacity.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ClaimSet {
    pub llcs: BTreeSet<usize>,
    pub cpus: BTreeSet<usize>,
}

impl ClaimSet {
    pub(crate) fn is_empty(&self) -> bool {
        self.llcs.is_empty() && self.cpus.is_empty()
    }
}

/// Read the LIVE head claim, or an empty set when there is no live
/// head. Liveness gates content: the manifest counts only while a
/// process holds the head marker flock (checked via `/proc/locks`,
/// which takes no lock and cannot perturb the queue). A stale
/// manifest from a crashed or finished head is therefore ignored, and
/// any read/parse failure degrades to "no claim" — the worst a wrong
/// answer costs a fast-path caller is one bounce.
pub(crate) fn read_live_claim() -> ClaimSet {
    // Cheap-out first: no manifest file (the completed head removed
    // it) or an empty one means no claim, without paying the
    // /proc/locks parse below. Fast paths call this once per
    // candidate probe, and the no-head case is the steady state.
    match std::fs::metadata(head_claim_path()) {
        Ok(m) if m.len() > 0 => {}
        _ => return ClaimSet::default(),
    }
    let holders = crate::flock::read_holders(Path::new(&head_marker_path())).unwrap_or_default();
    if holders.is_empty() {
        return ClaimSet::default();
    }
    match std::fs::read_to_string(head_claim_path()) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => ClaimSet::default(),
    }
}

/// Atomically publish `claim` as the head's manifest (temp file +
/// rename, so readers never observe a torn write). Caller must hold
/// the head marker flock — the manifest is meaningless without it.
fn publish_claim(claim: &ClaimSet) -> Result<()> {
    let path = head_claim_path();
    let tmp = format!("{path}.tmp-{}", std::process::id());
    let text = serde_json::to_string(claim)?;
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Remove the claim manifest (best-effort — liveness is the marker
/// flock, so a leftover manifest is already inert once the marker
/// releases; removing it just keeps the directory tidy).
fn clear_claim() {
    let _ = std::fs::remove_file(head_claim_path());
}

/// inotify watch over the lock dir: any lockfile release closes its
/// fd and fires `IN_CLOSE_WRITE` (lockfiles open `O_RDWR`, so the
/// close is a writable-fd close — pinned by
/// `flock_release_fires_in_close_write` in the locking tests). The
/// head sleeps here between acquisition attempts; peer fast-path
/// bounces fire the same events and cause spurious wakes, which are
/// tolerated — one waiter re-planning per event is not a herd.
pub(crate) struct LockDirWatch {
    ino: nix::sys::inotify::Inotify,
}

impl LockDirWatch {
    /// Watch the protocol directory. `IN_CLOSE_NOWRITE` is included
    /// alongside `IN_CLOSE_WRITE` defensively: the mask that fires
    /// depends on the open mode of the closing fd, and a future
    /// read-only lockfile open must not silently stop waking heads.
    pub(crate) fn new() -> Result<Self> {
        use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
        let dir = protocol_dir();
        std::fs::create_dir_all(&dir)?;
        let ino = Inotify::init(InitFlags::IN_NONBLOCK | InitFlags::IN_CLOEXEC)?;
        ino.add_watch(
            &dir,
            AddWatchFlags::IN_CLOSE_WRITE | AddWatchFlags::IN_CLOSE_NOWRITE,
        )?;
        Ok(LockDirWatch { ino })
    }

    /// Drain any queued events without blocking, discarding them.
    /// The head calls this right before sleeping so its OWN
    /// open/close churn from the attempt it just made does not wake
    /// it straight back up (a self-wake busy loop). An external
    /// release slipping into the drain-to-sleep gap is caught by the
    /// [`HEAD_WAKE_FALLBACK`] tick.
    pub(crate) fn drain(&self) {
        while let Ok(events) = self.ino.read_events() {
            if events.is_empty() {
                break;
            }
        }
    }

    /// Sleep until a lock-dir event arrives or `timeout` passes.
    /// Returns whether at least one event arrived (the caller
    /// re-plans either way; the flag is informational).
    pub(crate) fn wait(&self, timeout: Duration) -> Result<bool> {
        use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
        use std::os::fd::AsFd;
        let mut fds = [PollFd::new(self.ino.as_fd(), PollFlags::POLLIN)];
        let ms = u16::try_from(timeout.as_millis().clamp(1, u16::MAX as u128)).unwrap_or(u16::MAX);
        let n = poll(&mut fds, PollTimeout::from(ms))?;
        if n > 0 {
            self.drain();
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Wait for our turn at the head of the acquisition queue, with
/// progress-based patience.
///
/// Parks in the kernel's flock wait queue on [`queue_lock_path`]
/// (`LOCK_EX`, blocking) — arrival-order FIFO across every
/// invocation, tickets vanishing with their process. A
/// [`QUEUE_PROGRESS_TICK`] heartbeat interrupts the wait to sample
/// the queue holder via `/proc/locks`: a CHANGED holder identity
/// means the queue advanced (a head finished, or a waiter ahead gave
/// up) and the patience clock resets. Only [`patience`] of zero
/// turnover — a wedged head, itself bounded by ITS patience, so
/// cascade drain is guaranteed — returns `None` (the caller maps
/// that to the retryable contention failure).
///
/// Queued waiters hold NO resource locks (their fast-path attempts
/// were all-or-nothing and released everything), so parking here can
/// never feed a deadlock cycle.
pub(crate) fn wait_for_queue_turn() -> Result<Option<OwnedFd>> {
    let qpath = queue_lock_path();
    // Fast grab: empty queue is the common case.
    if let Some(fd) = try_flock(&qpath, FlockMode::Exclusive)? {
        return Ok(Some(fd));
    }
    let mut last_holders: Vec<u32> = holder_pids(&qpath);
    let mut deadline = Instant::now() + patience();
    loop {
        match block_flock_step(
            &qpath,
            FlockMode::Exclusive,
            deadline,
            queue_progress_tick(),
        )? {
            FlockWait::Granted(fd) => return Ok(Some(fd)),
            outcome @ (FlockWait::Tick | FlockWait::DeadlineExpired) => {
                let holders = holder_pids(&qpath);
                if holders != last_holders {
                    last_holders = holders;
                    deadline = Instant::now() + patience();
                } else if matches!(outcome, FlockWait::DeadlineExpired) {
                    return Ok(None);
                }
            }
        }
    }
}

/// Sorted holder-pid sample for progress comparison.
fn holder_pids(path: &str) -> Vec<u32> {
    let mut pids: Vec<u32> = crate::flock::read_holders(Path::new(path))
        .unwrap_or_default()
        .into_iter()
        .map(|h| h.pid)
        .collect();
    pids.sort_unstable();
    pids
}

/// The head's accumulated partial holds, keyed by lockfile path.
/// Only ever instantiated inside [`acquire_as_head`] — the global
/// head license (queue `LOCK_EX`) is what makes holding these while
/// waiting safe.
#[derive(Default)]
pub(crate) struct HeldLocks {
    map: std::collections::BTreeMap<String, OwnedFd>,
}

impl HeldLocks {
    /// Release every held lock whose path is not in `keep` — the
    /// release-partials-then-replan rule: a fresh plan keeps the
    /// partials it still needs (dropping them would be wasted
    /// makespan) and releases only the ones it abandoned.
    pub(crate) fn retain_paths(&mut self, keep: &BTreeSet<String>) {
        self.map.retain(|p, _| keep.contains(p));
    }

    /// Non-blocking sweep: acquire, in the given (canonical) order,
    /// every lock in `target` not already held. Returns how many NEW
    /// locks were gained. Never blocks — the head's waiting happens
    /// on the inotify watch, not inside flock.
    pub(crate) fn sweep(&mut self, target: &[(String, FlockMode)]) -> Result<usize> {
        let mut gained = 0;
        for (path, mode) in target {
            if self.map.contains_key(path) {
                continue;
            }
            if let Some(fd) = try_flock(path, *mode)? {
                self.map.insert(path.clone(), fd);
                gained += 1;
            }
        }
        Ok(gained)
    }

    /// All-or-nothing probe of `candidate` reusing held locks: any
    /// lock already held counts; missing ones are tried non-blocking.
    /// On success the candidate's fds are REMOVED from the held map
    /// and returned in candidate order. On failure the newly taken
    /// fds are released and previously held ones stay — the probe
    /// must not regress accumulated progress.
    pub(crate) fn probe_complete(
        &mut self,
        candidate: &[(String, FlockMode)],
    ) -> Result<Option<Vec<OwnedFd>>> {
        let mut fresh: Vec<(String, OwnedFd)> = Vec::new();
        for (path, mode) in candidate {
            if self.map.contains_key(path) {
                continue;
            }
            match try_flock(path, *mode)? {
                Some(fd) => fresh.push((path.clone(), fd)),
                None => {
                    // Bounce: drop only what THIS probe took.
                    drop(fresh);
                    return Ok(None);
                }
            }
        }
        for (path, fd) in fresh {
            self.map.insert(path, fd);
        }
        let mut out = Vec::with_capacity(candidate.len());
        for (path, _) in candidate {
            out.push(
                self.map
                    .remove(path)
                    .expect("probe_complete: candidate path must be held"),
            );
        }
        Ok(Some(out))
    }

    /// Whether every lock in `target` is currently held.
    pub(crate) fn covers(&self, target: &[(String, FlockMode)]) -> bool {
        target.iter().all(|(p, _)| self.map.contains_key(p))
    }

    /// Take the fds for `target` out of the map, in target order.
    /// Caller must have verified [`Self::covers`].
    pub(crate) fn take(&mut self, target: &[(String, FlockMode)]) -> Vec<OwnedFd> {
        target
            .iter()
            .map(|(p, _)| self.map.remove(p).expect("take: target path must be held"))
            .collect()
    }

    /// First lock in `target` not currently held (diagnostics).
    pub(crate) fn first_missing<'t>(&self, target: &'t [(String, FlockMode)]) -> Option<&'t str> {
        target
            .iter()
            .find(|(p, _)| !self.map.contains_key(p))
            .map(|(p, _)| p.as_str())
    }

    /// Paths currently held (for plan_live's overlap preference).
    pub(crate) fn held_paths(&self) -> BTreeSet<String> {
        self.map.keys().cloned().collect()
    }
}

/// One head-loop iteration's verdict, produced by the caller's step
/// closure (which owns the path-specific planning + probing logic).
pub(crate) enum HeadStep<T> {
    /// Acquisition complete — `T` carries the payload (plan + fds).
    Complete(T),
    /// Still waiting. `claim` is the freshly planned target to
    /// publish; `gained` is whether this iteration acquired at least
    /// one new lock (progress — resets patience); `stalled_on` names
    /// a missing lock for the eventual timeout diagnostic.
    Waiting {
        claim: ClaimSet,
        gained: bool,
        stalled_on: String,
    },
    /// The step decided acquisition cannot proceed at all (e.g. no
    /// plannable candidate remains). Terminal; not a timeout.
    Abort { reason: String },
}

/// Outcome of [`acquire_as_head`].
pub(crate) enum HeadOutcome<T> {
    Acquired(T),
    /// Patience expired with zero acquisition progress.
    TimedOut {
        stalled_on: String,
        waited: Duration,
    },
    Aborted {
        reason: String,
    },
}

/// Run the head loop: RE-PLAN ON EVERY WAKE (the step closure is
/// called afresh each iteration and must plan from live holder state
/// — plans are never cached across waits), publish the claim, sleep
/// on the lock-dir inotify watch, with progress-based patience
/// (resets on every newly gained lock).
///
/// The caller must hold the queue `LOCK_EX` (the head license). This
/// function additionally takes the head marker flock for claim
/// liveness and releases it (plus any partial holds inside `held`)
/// on every exit path.
pub(crate) fn acquire_as_head<T>(
    mut step: impl FnMut(&mut HeldLocks) -> Result<HeadStep<T>>,
) -> Result<HeadOutcome<T>> {
    let watch = LockDirWatch::new()?;
    // The marker should be free: at most one head exists (queue EX)
    // and a crashed head's marker flock died with it. A held marker
    // here means a protocol bug — surface it rather than wedge.
    let _marker = try_flock(head_marker_path(), FlockMode::Exclusive)?.ok_or_else(|| {
        anyhow::anyhow!(
            "acquisition protocol: head marker {} is held while the queue \
             lock was free to take — two heads must never coexist",
            head_marker_path()
        )
    })?;
    let start = Instant::now();
    let mut held = HeldLocks::default();
    let mut deadline = Instant::now() + patience();
    let mut last_claim: Option<ClaimSet> = None;
    let outcome = loop {
        match step(&mut held)? {
            HeadStep::Complete(t) => break HeadOutcome::Acquired(t),
            HeadStep::Abort { reason } => break HeadOutcome::Aborted { reason },
            HeadStep::Waiting {
                claim,
                gained,
                stalled_on,
            } => {
                if last_claim.as_ref() != Some(&claim) {
                    publish_claim(&claim)?;
                    last_claim = Some(claim);
                }
                if gained {
                    deadline = Instant::now() + patience();
                }
                let now = Instant::now();
                if now >= deadline {
                    break HeadOutcome::TimedOut {
                        stalled_on,
                        waited: start.elapsed(),
                    };
                }
                // Drain our own attempt's open/close churn so we sleep
                // on EXTERNAL events, then park until a lockfile
                // closes somewhere (a release — or a peer bounce,
                // which costs one spurious re-plan) or the fallback
                // tick bounds the staleness window.
                watch.drain();
                let timeout = HEAD_WAKE_FALLBACK.min(deadline - now);
                let _ = watch.wait(timeout)?;
            }
        }
    };
    clear_claim();
    Ok(outcome)
}

/// Canonical global lock order for a resource set: LLC locks by
/// ascending index, then CPU locks by ascending index. Every
/// acquirer — fast path and head alike — walks locks in this order
/// (protocol rule 3), which is what lets the head hold partials
/// safely and keeps overlapping fast-path sets from half-blocking
/// each other.
pub(crate) fn canonical_lock_order(
    llc_indices: &[usize],
    llc_mode: FlockMode,
    cpus: &[usize],
) -> Vec<(String, FlockMode)> {
    let mut llcs: Vec<usize> = llc_indices.to_vec();
    llcs.sort_unstable();
    llcs.dedup();
    let mut cpu_sorted: Vec<usize> = cpus.to_vec();
    cpu_sorted.sort_unstable();
    cpu_sorted.dedup();
    let mut out = Vec::with_capacity(llcs.len() + cpu_sorted.len());
    for idx in llcs {
        out.push((super::llc_lock_path(idx), llc_mode));
    }
    for cpu in cpu_sorted {
        out.push((super::cpu_lock_path(cpu), FlockMode::Exclusive));
    }
    out
}
