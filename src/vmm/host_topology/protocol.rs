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
//! the kernel's flock wait queue on the queue file. The head sleeps on
//! an inotify watch of the lock dir ([`LockDirWatch`]): releasing any
//! lockfile closes its fd, which fires `IN_CLOSE_*` on that file, and
//! the head wakes, RE-PLANS AGAINST LIVE HOLDER STATE (a hard
//! requirement — plans are never cached across waits; the freed
//! resource may satisfy a different plan than the one the head would
//! have blocked on), and attempts acquisition again. Spurious wakes
//! (peer fast-path bounces also open/close lockfiles) are tolerated:
//! exactly one waiter re-plans per event, which is not a herd.
//!
//! **Lifecycle bounds belong to lifecycle owners.** Neither a queued
//! waiter nor the queue head can infer that a live resource holder is
//! wedged from elapsed wall time: coverage instrumentation, host
//! preemption, and a legitimately long cell all make that inference
//! false. They therefore wait until the authoritative flock release.
//! A holder crash releases its flock in the kernel; an in-cell VM
//! watchdog bounds guest progress; and nextest's slow-timeout is the
//! final process-lifecycle rail. This prevents one slow but healthy
//! holder from making every waiter behind it fail and retry.
//!
//! Waiters are CHEAP by design: a queued cell is a process holding
//! one blocked flock fd. Only the head adds an inotify fd and its
//! partial lockfile fds — still no guest memory or vCPUs (acquisition
//! runs before VM setup; see `KtstrVm::run`). nextest can therefore
//! admit every cell at once and let this queue schedule them.
//!
//! The lock dir is already restricted to local filesystems (every
//! lockfile open routes through `crate::flock`'s remote-fs rejection),
//! which is also what inotify needs to be reliable.

use anyhow::Result;
use std::collections::BTreeSet;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::flock::{FlockMode, InterruptibleFlockWaiter, block_flock, try_flock};

/// Fallback wake for the head's inotify sleep, bounding the staleness
/// window if an event is missed (e.g. a release racing the
/// drain-before-sleep gap). The real wake is the inotify event.
const HEAD_WAKE_FALLBACK: Duration = Duration::from_millis(500);

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

/// Sharing mode of the LLC locks in a published [`ClaimSet`].
///
/// A claim is an advisory reservation, so its compatibility rules must
/// exactly match `flock`: a shared claim only fences an exclusive LLC
/// requester, while an exclusive claim fences both shared and exclusive
/// requesters. CPU claims are always exclusive.
///
/// `Exclusive` is the serde default for manifests written before claims
/// carried a mode. Treating a legacy, ambiguous claim conservatively keeps
/// mixed-version runs correct. Older readers ignore the added field and
/// continue to over-fence all LLC claims until every participant is updated;
/// that costs concurrency, never correctness.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ClaimLlcMode {
    #[default]
    Exclusive,
    Shared,
}

impl From<FlockMode> for ClaimLlcMode {
    fn from(mode: FlockMode) -> Self {
        match mode {
            FlockMode::Exclusive => Self::Exclusive,
            FlockMode::Shared => Self::Shared,
        }
    }
}

/// The head's published target set: which LLC locks (and at what sharing
/// mode) and CPU locks it is currently accumulating toward. Fast-path
/// planners subtract only incompatible targets from their view of free
/// capacity.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ClaimSet {
    pub llcs: BTreeSet<usize>,
    pub cpus: BTreeSet<usize>,
    #[serde(default)]
    pub llc_mode: ClaimLlcMode,
}

impl ClaimSet {
    pub(crate) fn new(
        llcs: impl IntoIterator<Item = usize>,
        cpus: impl IntoIterator<Item = usize>,
        llc_mode: FlockMode,
    ) -> Self {
        Self {
            llcs: llcs.into_iter().collect(),
            cpus: cpus.into_iter().collect(),
            llc_mode: llc_mode.into(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.llcs.is_empty() && self.cpus.is_empty()
    }

    /// Whether `request_mode` on `llc_idx` is incompatible with this live
    /// head claim. This is the same compatibility matrix as `flock`.
    pub(crate) fn conflicts_with_llc(&self, llc_idx: usize, request_mode: FlockMode) -> bool {
        self.llcs.contains(&llc_idx)
            && matches!(
                (self.llc_mode, request_mode),
                (ClaimLlcMode::Exclusive, _) | (ClaimLlcMode::Shared, FlockMode::Exclusive)
            )
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

/// RAII queue-head license. The flock is the authoritative ticket and
/// disappears automatically if the process crashes.
pub(crate) struct QueueTurn {
    // Keep the thread-directed wake registration alive while this turn owns
    // the head license. A cancellation signal can therefore interrupt either
    // the queue flock below or the head's inotify poll. It is deliberately the
    // first field: struct fields drop in declaration order, so registration
    // teardown completes before releasing the queue flock to the next head.
    _interrupt_waiter: Option<InterruptibleFlockWaiter>,
    _fd: OwnedFd,
}

fn interrupted() -> anyhow::Error {
    std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "ktstr resource acquisition interrupted",
    )
    .into()
}

fn check_interrupted(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(interrupted())
    } else {
        Ok(())
    }
}

/// Prefer the caller's authoritative cancellation verdict over an incidental
/// syscall error (most commonly the `EINTR` produced by the private wake).
fn check_result<T>(result: Result<T>, cancelled: Option<&AtomicBool>) -> Result<T> {
    match result {
        Ok(value) => {
            check_interrupted(cancelled)?;
            Ok(value)
        }
        Err(error) => {
            check_interrupted(cancelled)?;
            Err(error)
        }
    }
}

/// Wait for our turn at the head of the acquisition queue.
///
/// Parks one persistent fd in the kernel's flock wait queue on
/// [`queue_lock_path`] (`LOCK_EX`, blocking) — arrival-order FIFO
/// across every invocation, tickets vanishing with their process.
/// The wait has no private wall-clock verdict: the holder's watchdog
/// and nextest process rail own lifecycle bounds, while flock owns
/// crash cleanup. Keeping one blocking fd also preserves the waiter's
/// kernel queue position instead of periodically moving it to the
/// tail.
///
/// Queued waiters hold NO resource locks (their fast-path attempts
/// were all-or-nothing and released everything), so parking here can
/// never feed a deadlock cycle.
pub(crate) fn wait_for_queue_turn() -> Result<QueueTurn> {
    wait_for_queue_turn_impl(None, || {})
}

/// Cancellation-aware variant of [`wait_for_queue_turn`].
///
/// The waiter retains one kernel FIFO flock request throughout healthy
/// contention. The RT-wake broker remains idle until the caller publishes
/// `cancelled = true` and asks [`crate::flock::wake_interruptible_flock_waiter`]
/// to wake the exact registered generation.
pub(crate) fn wait_for_queue_turn_interruptible(cancelled: &AtomicBool) -> Result<QueueTurn> {
    wait_for_queue_turn_impl(Some(cancelled), || {})
}

fn wait_for_queue_turn_impl(
    cancelled: Option<&AtomicBool>,
    before_block: impl FnOnce(),
) -> Result<QueueTurn> {
    check_interrupted(cancelled)?;
    let interrupt_waiter = check_result(
        cancelled
            .map(|_| InterruptibleFlockWaiter::register())
            .transpose(),
        cancelled,
    )?;
    check_interrupted(cancelled)?;

    let qpath = queue_lock_path();
    // Fast grab: empty queue is the common case.
    let fast = match try_flock(&qpath, FlockMode::Exclusive) {
        Ok(fast) => fast,
        Err(error) => {
            check_interrupted(cancelled)?;
            return Err(error);
        }
    };
    if let Some(fd) = fast {
        let turn = QueueTurn {
            _interrupt_waiter: interrupt_waiter,
            _fd: fd,
        };
        check_interrupted(cancelled)?;
        return Ok(turn);
    }

    // This check plus the cancellation broker's repeated RT wake closes both
    // sides of the check/enter-flock race: a cancellation before the check
    // returns here, while one after it interrupts the syscall on a broker tick.
    check_interrupted(cancelled)?;
    before_block();
    let fd = match block_flock(&qpath, FlockMode::Exclusive) {
        Ok(fd) => fd,
        Err(error) => {
            check_interrupted(cancelled)?;
            return Err(error);
        }
    };
    let turn = QueueTurn {
        _interrupt_waiter: interrupt_waiter,
        _fd: fd,
    };
    check_interrupted(cancelled)?;
    Ok(turn)
}

/// Test seam for deterministically cancelling after the final userspace check
/// but before entering the blocking flock.
#[cfg(test)]
pub(crate) fn wait_for_queue_turn_interruptible_with_handoff(
    cancelled: &AtomicBool,
    before_block: impl FnOnce(),
) -> Result<QueueTurn> {
    wait_for_queue_turn_impl(Some(cancelled), before_block)
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
    /// Still waiting. `claim` is the freshly planned target to publish
    /// before sleeping for the next release event.
    Waiting { claim: ClaimSet },
    /// The step decided acquisition cannot proceed at all (e.g. no
    /// plannable candidate remains). Terminal; not a timeout.
    Abort { reason: String },
}

/// Outcome of [`acquire_as_head`].
pub(crate) enum HeadOutcome<T> {
    Acquired(T),
    Aborted { reason: String },
}

/// Claim manifest cleanup must precede marker-flock release on every exit,
/// including a closure error or an interrupt out of the inotify poll.
struct HeadClaimCleanup;

impl Drop for HeadClaimCleanup {
    fn drop(&mut self) {
        clear_claim();
    }
}

/// Run the head loop: RE-PLAN ON EVERY WAKE (the step closure is
/// called afresh each iteration and must plan from live holder state
/// — plans are never cached across waits), publish the claim, sleep
/// on the lock-dir inotify watch, and continue until the authoritative
/// lock release makes a plan complete.
///
/// The caller must hold the queue `LOCK_EX` (the head license). This
/// function additionally takes the head marker flock for claim
/// liveness and releases it (plus any partial holds inside `held`)
/// on every exit path.
pub(crate) fn acquire_as_head<T>(
    step: impl FnMut(&mut HeldLocks) -> Result<HeadStep<T>>,
) -> Result<HeadOutcome<T>> {
    acquire_as_head_impl(None, step)
}

/// Cancellation-aware variant of [`acquire_as_head`].
///
/// The [`QueueTurn`] returned by [`wait_for_queue_turn_interruptible`] keeps
/// the private wake registration alive, so cancellation interrupts the
/// inotify poll rather than waiting for its fallback tick.
pub(crate) fn acquire_as_head_interruptible<T>(
    cancelled: &AtomicBool,
    step: impl FnMut(&mut HeldLocks) -> Result<HeadStep<T>>,
) -> Result<HeadOutcome<T>> {
    acquire_as_head_impl(Some(cancelled), step)
}

fn acquire_as_head_impl<T>(
    cancelled: Option<&AtomicBool>,
    mut step: impl FnMut(&mut HeldLocks) -> Result<HeadStep<T>>,
) -> Result<HeadOutcome<T>> {
    check_interrupted(cancelled)?;
    let watch = check_result(LockDirWatch::new(), cancelled)?;
    // The marker should be free: at most one head exists (queue EX)
    // and a crashed head's marker flock died with it. A held marker
    // here means a protocol bug — surface it rather than wedge.
    let _marker = match try_flock(head_marker_path(), FlockMode::Exclusive) {
        Ok(Some(marker)) => marker,
        Ok(None) => {
            return Err(anyhow::anyhow!(
                "acquisition protocol: head marker {} is held while the queue \
                 lock was free to take — two heads must never coexist",
                head_marker_path()
            ));
        }
        Err(error) => {
            check_interrupted(cancelled)?;
            return Err(error);
        }
    };
    // Declared after `_marker` so Rust's reverse drop order removes the
    // manifest while marker liveness is still held.
    let _claim_cleanup = HeadClaimCleanup;
    check_interrupted(cancelled)?;
    let mut held = HeldLocks::default();
    let mut last_claim: Option<ClaimSet> = None;
    let outcome = loop {
        check_interrupted(cancelled)?;
        let next = match step(&mut held) {
            Ok(next) => next,
            Err(error) => {
                check_interrupted(cancelled)?;
                return Err(error);
            }
        };
        check_interrupted(cancelled)?;
        match next {
            HeadStep::Complete(t) => break HeadOutcome::Acquired(t),
            HeadStep::Abort { reason } => break HeadOutcome::Aborted { reason },
            HeadStep::Waiting { claim } => {
                if last_claim.as_ref() != Some(&claim) {
                    check_result(publish_claim(&claim), cancelled)?;
                    last_claim = Some(claim);
                }
                check_interrupted(cancelled)?;
                // Drain our own attempt's open/close churn so we sleep
                // on EXTERNAL events, then park until a lockfile
                // closes somewhere (a release — or a peer bounce,
                // which costs one spurious re-plan) or the fallback
                // tick bounds the staleness window.
                watch.drain();
                check_interrupted(cancelled)?;
                check_result(watch.wait(HEAD_WAKE_FALLBACK), cancelled)?;
            }
        }
    };
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
