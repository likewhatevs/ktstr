//! sched_ext dump capture, log forwarding, trace-pipe, and the sched-exit monitor.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;
use std::io::{Seek, SeekFrom};
use std::sync::atomic::AtomicUsize;

/// Maximum scheduler-log chunk emitted in a single
/// [`crate::vmm::guest_comms::try_send_sched_log`] frame. Sub-cap of
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably inside one TLV frame; chunks above this size are
/// split before emission.
const SCHED_LOG_CHUNK_BYTES: usize = 64 * 1024;

/// Total attempts made before a terminal scheduler event is allowed to
/// overtake an incomplete scheduler-log transaction.
///
/// There is deliberately no sleep between attempts. Each call already waits
/// for the bounded stdio-forwarder drain when necessary, while a failed bulk
/// write is itself the readiness probe for the next immediate reopen attempt.
const TERMINAL_SCHED_LOG_DUMP_ATTEMPTS: usize = 3;

#[derive(Debug)]
enum SchedLogWireState {
    Idle,
    Active { owner: String },
    Abandoned { owner: String },
}

#[derive(Debug)]
struct SchedLogWireCoordinator {
    state: std::sync::Mutex<SchedLogWireState>,
    changed: std::sync::Condvar,
    waiters: AtomicUsize,
}

impl SchedLogWireCoordinator {
    fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(SchedLogWireState::Idle),
            changed: std::sync::Condvar::new(),
            waiters: AtomicUsize::new(0),
        }
    }

    fn acquire(&self, owner: String) -> Result<SchedLogWireLease<'_>, String> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match &*state {
                SchedLogWireState::Idle => {
                    *state = SchedLogWireState::Active {
                        owner: owner.clone(),
                    };
                    return Ok(SchedLogWireLease {
                        coordinator: self,
                        owner,
                        finalized: false,
                        transaction_open: false,
                    });
                }
                SchedLogWireState::Active { .. } => {
                    self.waiters.fetch_add(1, Ordering::Release);
                    state = self
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    self.waiters.fetch_sub(1, Ordering::Release);
                }
                SchedLogWireState::Abandoned { owner } => return Err(owner.clone()),
            }
        }
    }

    #[cfg(test)]
    fn wait_for_waiter(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while self.waiters.load(Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for a scheduler-log wire contender"
            );
            std::thread::yield_now();
        }
    }
}

struct SchedLogWireLease<'a> {
    coordinator: &'a SchedLogWireCoordinator,
    owner: String,
    finalized: bool,
    transaction_open: bool,
}

impl SchedLogWireLease<'_> {
    fn mark_open(&mut self) {
        self.transaction_open = true;
    }

    fn mark_closed(&mut self) {
        self.transaction_open = false;
    }

    fn finish(mut self) {
        let mut state = self
            .coordinator
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(matches!(
            &*state,
            SchedLogWireState::Active { owner } if owner == &self.owner
        ));
        *state = if self.transaction_open {
            SchedLogWireState::Abandoned {
                owner: self.owner.clone(),
            }
        } else {
            SchedLogWireState::Idle
        };
        self.finalized = true;
        drop(state);
        self.coordinator.changed.notify_all();
    }
}

impl Drop for SchedLogWireLease<'_> {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        // START acceptance is recorded synchronously in the send path. A panic
        // before START can release the idle wire; a panic while START..END is
        // open must reserve it permanently rather than permit a splice.
        let mut state = self
            .coordinator
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = if self.transaction_open {
            SchedLogWireState::Abandoned {
                owner: self.owner.clone(),
            }
        } else {
            SchedLogWireState::Idle
        };
        drop(state);
        self.coordinator.changed.notify_all();
    }
}

static SCHED_LOG_WIRE_COORDINATOR: OnceLock<SchedLogWireCoordinator> = OnceLock::new();

fn sched_log_wire_coordinator() -> &'static SchedLogWireCoordinator {
    SCHED_LOG_WIRE_COORDINATOR.get_or_init(SchedLogWireCoordinator::new)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedLogDumpPhase {
    Idle,
    InProgress,
    Done,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SchedStreamDumpCursor {
    payload_offset: u64,
    complete: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SchedLogDumpCursor {
    stdout: SchedStreamDumpCursor,
    stderr: SchedStreamDumpCursor,
    start_sent: bool,
    payload_offset: u64,
    end_sent: bool,
}

#[derive(Debug)]
struct SchedLogDumpState {
    phase: SchedLogDumpPhase,
    cursor: SchedLogDumpCursor,
}

#[derive(Debug)]
struct SchedLogDumpSlot {
    state: std::sync::Mutex<SchedLogDumpState>,
    changed: std::sync::Condvar,
}

static SCHED_LOG_DUMP_SLOTS: OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, Arc<SchedLogDumpSlot>>>,
> = OnceLock::new();

struct SchedLogDumpLease {
    slot: Arc<SchedLogDumpSlot>,
    complete: bool,
}

impl SchedLogDumpLease {
    fn acquire(path: &str) -> Option<Self> {
        let normalized = normalized_sched_log_path(path);
        let slot = {
            let slots = SCHED_LOG_DUMP_SLOTS
                .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
            let mut slots = slots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slots
                .entry(normalized)
                .or_insert_with(|| {
                    Arc::new(SchedLogDumpSlot {
                        state: std::sync::Mutex::new(SchedLogDumpState {
                            phase: SchedLogDumpPhase::Idle,
                            cursor: SchedLogDumpCursor::default(),
                        }),
                        changed: std::sync::Condvar::new(),
                    })
                })
                .clone()
        };

        let mut state = slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match state.phase {
                SchedLogDumpPhase::Idle => {
                    state.phase = SchedLogDumpPhase::InProgress;
                    drop(state);
                    return Some(Self {
                        slot,
                        complete: false,
                    });
                }
                SchedLogDumpPhase::InProgress => {
                    state = slot
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                SchedLogDumpPhase::Done => return None,
            }
        }
    }

    fn cursor(&self) -> SchedLogDumpCursor {
        self.slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cursor
    }

    fn mark_start_sent(&self) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.phase, SchedLogDumpPhase::InProgress);
        state.cursor.start_sent = true;
    }

    fn stream_cursor(
        &self,
        stream: super::scheduler::SchedulerOutputStream,
    ) -> SchedStreamDumpCursor {
        let cursor = self.cursor();
        match stream {
            super::scheduler::SchedulerOutputStream::Stdout => cursor.stdout,
            super::scheduler::SchedulerOutputStream::Stderr => cursor.stderr,
        }
    }

    fn advance_stream_payload(
        &self,
        stream: super::scheduler::SchedulerOutputStream,
        bytes: usize,
    ) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.phase, SchedLogDumpPhase::InProgress);
        let cursor = match stream {
            super::scheduler::SchedulerOutputStream::Stdout => &mut state.cursor.stdout,
            super::scheduler::SchedulerOutputStream::Stderr => &mut state.cursor.stderr,
        };
        cursor.payload_offset = cursor
            .payload_offset
            .checked_add(bytes as u64)
            .expect("scheduler stream transmission offset overflowed u64");
    }

    fn mark_stream_complete(&self, stream: super::scheduler::SchedulerOutputStream) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.phase, SchedLogDumpPhase::InProgress);
        match stream {
            super::scheduler::SchedulerOutputStream::Stdout => {
                state.cursor.stdout.complete = true;
            }
            super::scheduler::SchedulerOutputStream::Stderr => {
                state.cursor.stderr.complete = true;
            }
        }
    }

    fn advance_payload(&self, bytes: usize) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.phase, SchedLogDumpPhase::InProgress);
        state.cursor.payload_offset = state
            .cursor
            .payload_offset
            .checked_add(bytes as u64)
            .expect("scheduler log transmission offset overflowed u64");
    }

    fn mark_end_sent(&self) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.phase, SchedLogDumpPhase::InProgress);
        state.cursor.end_sent = true;
    }

    fn complete(mut self) {
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.cursor.start_sent);
        debug_assert!(state.cursor.end_sent);
        state.phase = SchedLogDumpPhase::Done;
        self.complete = true;
        drop(state);
        self.slot.changed.notify_all();
    }
}

impl Drop for SchedLogDumpLease {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        // A panic during drain/send leaves the path retryable. Wake a waiter
        // only after restoring Idle under the same per-path mutex.
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.phase = SchedLogDumpPhase::Idle;
        drop(state);
        self.slot.changed.notify_one();
    }
}

fn normalized_sched_log_path(path: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(path);
    if let Ok(canonical) = fs::canonicalize(path) {
        return canonical;
    }

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("/"))
            .join(path)
    };
    let mut normalized = std::path::PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

/// Send the scheduler log to the host bracketed by
/// [`crate::verifier::SCHED_OUTPUT_START`] /
/// [`crate::verifier::SCHED_OUTPUT_END`] markers. Replaces the
/// prior COM2 dump path: the markers travel verbatim inside the
/// chunk bytes so the host's `parse_sched_output` walker (which
/// scans for the start/end pair after concatenating chunks) keeps
/// working unchanged. The BPF verifier section embedded in the
/// scheduler's stderr / stdout passes through byte-for-byte so a
/// scheduler author still sees the kernel's verifier rejection
/// text in the host-side failure render.
///
/// Every normalized path owns one resumable transaction process-wide.
/// Concurrent callers serialize through the path slot. A failed drain, read,
/// or bulk write leaves the first unsent frame in the cursor, so a later caller
/// resumes without duplicating an accepted START or payload prefix.
#[cfg(test)]
fn dump_sched_output_with(
    log_path: &str,
    wait_for_forwarders: impl FnOnce(&str) -> bool,
    send: impl FnMut(&[u8]) -> bool,
) -> bool {
    dump_sched_output_with_wire(
        log_path,
        std::time::Instant::now() + super::scheduler::SCHED_FORWARDER_DRAIN_BOUND,
        |path, _deadline| wait_for_forwarders(path),
        send,
        None,
    )
}

#[cfg(test)]
fn dump_sched_output_with_wire(
    log_path: &str,
    forwarder_deadline: std::time::Instant,
    wait_for_forwarders: impl FnOnce(&str, std::time::Instant) -> bool,
    send: impl FnMut(&[u8]) -> bool,
    wire: Option<&mut SchedLogWireLease<'_>>,
) -> bool {
    dump_sched_output_with_wire_and_streams(
        log_path,
        forwarder_deadline,
        wait_for_forwarders,
        send,
        |_| false,
        |_| false,
        wire,
    )
}

/// Replay one complete scheduler stream only when its forwarder retained an
/// authoritative sidecar after observing a failed live publication.
///
/// Missing sidecars are the common, successful hot path: every live frame
/// reached the host, so no terminal replay is sent. A present sidecar is
/// immutable after its atomic promotion at forwarder EOF. Accepted ranges are
/// recorded synchronously, allowing terminal retries to resume without
/// duplicating a chunk. The host selects this stream only after the embedded
/// offsets prove the full `0..total_len` transaction arrived.
fn dump_sched_stream_replay(
    log_path: &str,
    stream: super::scheduler::SchedulerOutputStream,
    lease: &SchedLogDumpLease,
    mut send: impl FnMut(&[u8]) -> bool,
) -> bool {
    let mut cursor = lease.stream_cursor(stream);
    if cursor.complete {
        return true;
    }
    let replay_path = super::scheduler::scheduler_stream_replay_path(log_path, stream);
    let mut file = match fs::File::open(&replay_path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return true,
        Err(error) => {
            tracing::warn!(
                log_path,
                replay_path = %replay_path.display(),
                %error,
                "scheduler stream replay open failed; retaining cursor"
            );
            return false;
        }
    };
    let total_len = match file.metadata() {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            tracing::warn!(
                log_path,
                replay_path = %replay_path.display(),
                %error,
                "scheduler stream replay metadata failed; retaining cursor"
            );
            return false;
        }
    };
    if total_len < cursor.payload_offset {
        tracing::warn!(
            log_path,
            replay_path = %replay_path.display(),
            total_len,
            payload_offset = cursor.payload_offset,
            "scheduler stream replay shrank behind its accepted cursor"
        );
        return false;
    }
    if let Err(error) = file.seek(SeekFrom::Start(cursor.payload_offset)) {
        tracing::warn!(
            log_path,
            replay_path = %replay_path.display(),
            %error,
            "scheduler stream replay seek failed; retaining cursor"
        );
        return false;
    }

    // An empty capture still needs one explicit frame to distinguish a
    // complete empty stream from an interrupted transaction.
    if total_len == 0 {
        let payload = crate::vmm::wire::encode_sched_stream_final_chunk(0, 0, &[]);
        if !send(&payload) {
            return false;
        }
        lease.mark_stream_complete(stream);
        return true;
    }

    let mut chunk = [0u8; SCHED_LOG_CHUNK_BYTES];
    while cursor.payload_offset < total_len {
        let remaining = (total_len - cursor.payload_offset) as usize;
        let read_len = remaining.min(chunk.len());
        let read = match file.read(&mut chunk[..read_len]) {
            Ok(0) => {
                tracing::warn!(
                    log_path,
                    replay_path = %replay_path.display(),
                    total_len,
                    payload_offset = cursor.payload_offset,
                    "scheduler stream replay ended before its recorded length"
                );
                return false;
            }
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                tracing::warn!(
                    log_path,
                    replay_path = %replay_path.display(),
                    %error,
                    "scheduler stream replay read failed; retaining cursor"
                );
                return false;
            }
        };
        let payload = crate::vmm::wire::encode_sched_stream_final_chunk(
            total_len,
            cursor.payload_offset,
            &chunk[..read],
        );
        if !send(&payload) {
            return false;
        }
        lease.advance_stream_payload(stream, read);
        cursor.payload_offset += read as u64;
    }
    lease.mark_stream_complete(stream);
    true
}

fn dump_sched_output_with_wire_and_streams(
    log_path: &str,
    forwarder_deadline: std::time::Instant,
    wait_for_forwarders: impl FnOnce(&str, std::time::Instant) -> bool,
    mut send: impl FnMut(&[u8]) -> bool,
    mut send_stdout_replay: impl FnMut(&[u8]) -> bool,
    mut send_stderr_replay: impl FnMut(&[u8]) -> bool,
    mut wire: Option<&mut SchedLogWireLease<'_>>,
) -> bool {
    let Some(lease) = SchedLogDumpLease::acquire(log_path) else {
        return true;
    };
    let mut cursor = lease.cursor();
    // END is the transaction's durable completion edge. If the prior caller
    // was interrupted between recording its successful send and flipping the
    // phase to Done, finalize without reopening a path that may since have
    // been rotated or appended.
    if cursor.end_sent {
        lease.complete();
        return true;
    }
    // The child's output reaches the log file through the live-stream
    // forwarder threads; every dump caller runs after the child was
    // reaped, but the forwarders may still be draining the final pipe
    // bytes. Wait (bounded) so the dumped file — and the live
    // SchedStdout/SchedStderr streams — carry the complete output.
    if !wait_for_forwarders(log_path, forwarder_deadline) {
        return false;
    }

    if !dump_sched_stream_replay(
        log_path,
        super::scheduler::SchedulerOutputStream::Stdout,
        &lease,
        &mut send_stdout_replay,
    ) {
        return false;
    }
    if !dump_sched_stream_replay(
        log_path,
        super::scheduler::SchedulerOutputStream::Stderr,
        &lease,
        &mut send_stderr_replay,
    ) {
        return false;
    }

    // Open as bytes only after all pipe readers reached EOF. Invalid UTF-8 is
    // ordinary scheduler/verifier output and must survive byte-for-byte.
    let mut file = match fs::File::open(log_path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(log_path, %error, "scheduler log dump open failed; retaining cursor");
            return false;
        }
    };
    let file_len = match file.metadata() {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            tracing::warn!(log_path, %error, "scheduler log dump metadata failed; retaining cursor");
            return false;
        }
    };
    if file_len < cursor.payload_offset {
        tracing::warn!(
            log_path,
            file_len,
            payload_offset = cursor.payload_offset,
            "scheduler log shrank behind its accepted transmission cursor"
        );
        return false;
    }

    if !cursor.start_sent {
        if !send(crate::verifier::SCHED_OUTPUT_START.as_bytes()) {
            return false;
        }
        if let Some(wire) = wire.as_mut() {
            wire.mark_open();
        }
        lease.mark_start_sent();
        cursor.start_sent = true;
    }
    if let Err(error) = file.seek(SeekFrom::Start(cursor.payload_offset)) {
        tracing::warn!(log_path, %error, "scheduler log dump seek failed; retaining cursor");
        return false;
    }
    let mut chunk = [0u8; SCHED_LOG_CHUNK_BYTES];
    loop {
        let read = match file.read(&mut chunk) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                tracing::warn!(log_path, %error, "scheduler log dump read failed; retaining cursor");
                return false;
            }
        };
        if read == 0 {
            break;
        }
        if !send(&chunk[..read]) {
            return false;
        }
        lease.advance_payload(read);
        cursor.payload_offset += read as u64;
    }
    if !cursor.end_sent {
        if !send(crate::verifier::SCHED_OUTPUT_END.as_bytes()) {
            return false;
        }
        if let Some(wire) = wire.as_mut() {
            wire.mark_closed();
        }
        lease.mark_end_sent();
    }
    lease.complete();
    true
}

#[derive(Debug, Clone, Copy)]
struct SchedLogDumpAttempt {
    complete: bool,
}

/// Complete a scheduler-log transaction before publishing a terminal event or
/// rebooting the guest.
///
/// A failed attempt retains its per-path cursor, so the next attempt starts at
/// the first unaccepted frame rather than duplicating START, payload bytes, or
/// END. Exhaustion emits exactly one direct serial diagnostic. It must not use
/// the bulk port: the scheduler-log transaction may still have an accepted
/// START without END, and an unrelated bulk diagnostic must not be interposed
/// in that byte stream.
#[must_use]
pub(crate) fn dump_sched_output_before_terminal(log_path: &str) -> bool {
    let owner = normalized_sched_log_path(log_path)
        .to_string_lossy()
        .into_owned();
    dump_sched_output_before_terminal_with(
        sched_log_wire_coordinator(),
        &owner,
        log_path,
        |wire, forwarder_deadline| {
            let complete =
                dump_sched_output_with_terminal_streams(log_path, forwarder_deadline, wire);
            SchedLogDumpAttempt { complete }
        },
        write_com2,
    )
}

#[must_use]
fn dump_sched_output_with_terminal_streams(
    log_path: &str,
    forwarder_deadline: std::time::Instant,
    wire: &mut SchedLogWireLease<'_>,
) -> bool {
    dump_sched_output_with_wire_and_streams(
        log_path,
        forwarder_deadline,
        super::scheduler::wait_sched_forwarders_drained,
        crate::vmm::guest_comms::try_send_sched_log,
        crate::vmm::guest_comms::try_send_sched_stdout_final_chunk,
        crate::vmm::guest_comms::try_send_sched_stderr_final_chunk,
        Some(wire),
    )
}

fn dump_sched_output_before_terminal_with(
    coordinator: &SchedLogWireCoordinator,
    owner: &str,
    log_path: &str,
    mut dump: impl FnMut(&mut SchedLogWireLease<'_>, std::time::Instant) -> SchedLogDumpAttempt,
    mut diagnose: impl FnMut(&str),
) -> bool {
    let mut lease = match coordinator.acquire(owner.to_owned()) {
        Ok(lease) => lease,
        Err(_abandoned_owner) => {
            // The owner which transitioned the coordinator to Abandoned
            // already emitted the one serial diagnostic. Later transactions
            // fail silently here so that one wire failure cannot produce an
            // unbounded diagnostic storm while teardown walks staged logs.
            return false;
        }
    };
    let forwarder_deadline =
        std::time::Instant::now() + super::scheduler::SCHED_FORWARDER_DRAIN_BOUND;
    for _ in 0..TERMINAL_SCHED_LOG_DUMP_ATTEMPTS {
        let attempt = dump(&mut lease, forwarder_deadline);
        if attempt.complete {
            lease.mark_closed();
            lease.finish();
            return true;
        }
    }

    let diagnostic = format!(
        "scheduler log dump '{log_path}' did not complete after \
         {TERMINAL_SCHED_LOG_DUMP_ATTEMPTS} attempts (send failure, I/O error, or \
         forwarder-drain timeout); accepted frames were retained without retransmission"
    );
    diagnose(&diagnostic);
    // Publish the serial failure before releasing or poisoning the wire. A
    // woken terminal waiter may immediately reboot the guest.
    lease.finish();
    false
}

fn staged_scheduler_log_sequence(path: &std::path::Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_prefix("sched_")?.strip_suffix(".log")?;
    let (scheduler_name, sequence) = stem.rsplit_once('_')?;
    if scheduler_name.is_empty() {
        return None;
    }
    sequence.parse().ok()
}

fn staged_scheduler_log_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("sched_") && name.ends_with(".log"))
        })
        .collect();
    paths.sort_by(|left, right| {
        match (
            staged_scheduler_log_sequence(left),
            staged_scheduler_log_sequence(right),
        ) {
            (Some(left_sequence), Some(right_sequence)) => left_sequence
                .cmp(&right_sequence)
                .then_with(|| left.cmp(right)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.cmp(right),
        }
    });
    paths
}

/// Walk `/tmp/sched_*.log` and emit each readable file as a
/// separate `SCHED_OUTPUT_START` / `SCHED_OUTPUT_END` frame.
/// Captures logs from Op-spawned schedulers (Attach, Replace,
/// Restart) that the boot path's single `dump_sched_output` call
/// cannot reach. The final underscore-delimited component is the
/// process-wide numeric spawn sequence from `staged_scheduler_log_path`;
/// sorting that number (rather than the whole filename) preserves temporal
/// order across scheduler names and across the 9 → 10 boundary. Matching
/// legacy/malformed names without a numeric suffix remain readable and sort
/// deterministically after sequenced logs.
pub(crate) fn dump_staged_scheduler_logs() {
    for path in staged_scheduler_log_paths(std::path::Path::new("/tmp")) {
        if let Some(path) = path.to_str() {
            let _ = dump_sched_output_before_terminal(path);
        }
    }
}

/// Send a synthetic scheduler failure as one globally serialized, resumable
/// START / text / END transaction.
///
/// This is the no-child-log counterpart to
/// [`dump_sched_output_before_terminal`]. It shares the same wire coordinator,
/// so a spawn failure cannot splice its markers into another scheduler's open
/// dump. The text is bounded to one ordinary scheduler-log chunk.
#[must_use]
pub(crate) fn send_synthetic_sched_output_before_terminal(text: &str) -> bool {
    let bytes = text.as_bytes();
    let cap = SCHED_LOG_CHUNK_BYTES.min(bytes.len());
    send_synthetic_sched_output_before_terminal_with(
        sched_log_wire_coordinator(),
        "synthetic scheduler spawn failure",
        &bytes[..cap],
        crate::vmm::guest_comms::try_send_sched_log,
        write_com2,
    )
}

fn send_synthetic_sched_output_before_terminal_with(
    coordinator: &SchedLogWireCoordinator,
    owner: &str,
    payload: &[u8],
    mut send: impl FnMut(&[u8]) -> bool,
    diagnose: impl FnMut(&str),
) -> bool {
    let frames = [
        crate::verifier::SCHED_OUTPUT_START.as_bytes(),
        payload,
        crate::verifier::SCHED_OUTPUT_END.as_bytes(),
    ];
    let mut next_frame = 0usize;
    dump_sched_output_before_terminal_with(
        coordinator,
        owner,
        owner,
        |wire, _forwarder_deadline| {
            while next_frame < frames.len() {
                if !send(frames[next_frame]) {
                    return SchedLogDumpAttempt { complete: false };
                }
                if next_frame == 0 {
                    wire.mark_open();
                } else if next_frame + 1 == frames.len() {
                    wire.mark_closed();
                }
                next_frame += 1;
            }
            SchedLogDumpAttempt { complete: true }
        },
        diagnose,
    )
}

/// Enable sched_ext_dump trace event and pipe trace_pipe to COM1 in a
/// background thread. Returns the stop flag and thread join handle.
///
/// The reader opens trace_pipe with `O_NONBLOCK` and uses `poll()` on
/// a 200ms cadence so the loop is responsive to `stop` even when the
/// kernel never emits a sched_ext_dump event. A blocking `read(2)` on
/// trace_pipe parks the task in `tracing_wait_pipe` (kernel/trace/trace.c);
/// once that wait is entered with `iter->pos == 0` (no event ever
/// dispatched into the iterator), the kernel re-enters `wait_on_pipe`
/// after every wake because the inner loop in `tracing_wait_pipe` only
/// breaks when `!tracer_tracing_is_on(tr) && iter->pos`. Writing 0 to
/// `tracing_on` does fire `ring_buffer_wake_waiters`, but the
/// trace_pipe path supplies `wait_pipe_cond` (not the default
/// `rb_wait_once`) and that condition only flips when `iter->closed`
/// or `iter->wait_index` change — neither is touched by the trace_pipe
/// fops, so the wake produces a spurious return into `tracing_wait_pipe`
/// which immediately re-sleeps. Going non-blocking sidesteps the kernel
/// wait entirely: every iteration the userspace thread checks the stop
/// flag, polls for data, and drains any pending events without ever
/// parking in the kernel.
pub(crate) fn start_trace_pipe() -> (Option<Arc<AtomicBool>>, Option<std::thread::JoinHandle<()>>) {
    if Path::new(TRACE_SCHED_EXT_DUMP_ENABLE).exists() {
        let _ = fs::write(TRACE_SCHED_EXT_DUMP_ENABLE, "1");

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = std::thread::Builder::new()
            .name("trace-pipe".into())
            .spawn(move || {
                use std::os::unix::fs::OpenOptionsExt;
                let Ok(mut trace) = fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(TRACE_PIPE)
                else {
                    return;
                };
                let Ok(mut com1) = fs::OpenOptions::new().write(true).open(COM1) else {
                    return;
                };
                let mut buf = [0u8; 4096];
                // Tier-2 (lossless dump): rolling tail so an exit-dump
                // marker split across two reads is still matched by
                // `scan_dump_markers`.
                let mut scan_tail: Vec<u8> = Vec::new();
                loop {
                    // Break promptly once teardown signals stop. The only
                    // ftrace event ktstr enables, `sched_ext_dump`, fires
                    // ONLY as the one-shot crash dump (kernel scx_dump_state,
                    // emitted contiguously before the disable workfn); the
                    // disable itself emits no further trace. So trace_pipe
                    // at stop holds only the residual TAIL of that one dump
                    // still draining — a per-task dump (scx_dump_state with
                    // dump_all_tasks) whose size scales with runnable-task
                    // count, forwarded byte-by-byte over the slow PIO COM1
                    // UART, so its drain time scales with task count — that
                    // byte-by-byte COM1 forwarding is the tens-of-seconds
                    // cost (the kernel disable itself is ms-scale), and the
                    // dump is NOT disable-emitted.
                    // Forwarding the whole tail pinned `trace_handle.join`
                    // (the prior `drain_deadline` was checked only between
                    // polls, never inside the inner drain). Dropping it on
                    // stop is safe: teardown sets `stop` only AFTER the
                    // dump-complete latch (end-marker already on COM1) or
                    // the `SCX_DUMP_CAPTURE_TIMEOUT` bound; on the bound
                    // this ftrace copy truncates, but the full dump is
                    // captured via the scheduler's stderr log
                    // (`dump_sched_output`, scx_utils' `ei->dump`) over the
                    // fast bulk port — the authoritative copy.
                    if stop_clone.load(Ordering::Acquire) {
                        break;
                    }

                    let mut pollfds = [PollFd::new(trace.as_fd(), PollFlags::POLLIN)];
                    match poll(&mut pollfds, PollTimeout::from(200u16)) {
                        Ok(0) => continue,
                        Ok(_) => {}
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                    }
                    if let Some(revents) = pollfds[0].revents() {
                        if revents.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                            break;
                        }
                        if !revents.contains(PollFlags::POLLIN) {
                            // POLLHUP without POLLIN means no buffered
                            // data to drain; with POLLIN, fall through
                            // to read first so events that arrived
                            // before hangup still reach COM1.
                            if revents.contains(PollFlags::POLLHUP) {
                                break;
                            }
                            continue;
                        }
                    }

                    // Drain the bytes poll reported ready, re-checking
                    // `stop` after each chunk (below) so a continuous read
                    // cannot pin the reader here. Inner-loop exits use
                    // `break` (not `return`) so the outer loop's
                    // prompt-stop check + poll fd-state handling
                    // (POLLHUP/POLLERR) run on the next iteration.
                    loop {
                        match trace.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                let _ = com1.write_all(&buf[..n]);
                                scan_dump_markers(&buf[..n], &mut scan_tail);
                                // Re-check stop mid-batch so a continuous
                                // stream cannot pin the reader here; break
                                // to the prompt-stop check at the outer top.
                                if stop_clone.load(Ordering::Acquire) {
                                    break;
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(_) => break,
                        }
                    }
                }
            })
            .ok();
        (Some(stop), handle)
    } else {
        (None, None)
    }
}

/// Trailing bytes of the previous trace_pipe chunk retained by
/// [`scan_dump_markers`] so an exit-dump marker split across a read
/// boundary is still matched. The longest marker
/// (`SCX_EV_SUB_BYPASS_DISPATCH`, 26 bytes) fits with margin.
const SCAN_TAIL_KEEP: usize = 32;

/// Fired by the trace_pipe reader when the `sched_ext_dump` tracepoint
/// emits its FIRST line this run — i.e. an exit dump started streaming.
/// Read by teardown to decide whether to wait for completion: clean runs
/// never start a dump, so they never pay the [`SCX_DUMP_CAPTURE_TIMEOUT`]
/// wait.
static SCX_DUMP_STARTED_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Fired by the trace_pipe reader when the exit dump's end-marker
/// (`SCX_EV_SUB_BYPASS_DISPATCH`, the last event-counter line, or
/// `~~~~ TRUNCATED ~~~~`) reaches it — the full dump is captured.
/// Awaited by teardown before disabling the dump tracepoint so a fast
/// crash teardown does not truncate the dump mid-emit.
static SCX_DUMP_COMPLETE_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

pub(crate) fn scx_dump_started_latch() -> Arc<Latch> {
    SCX_DUMP_STARTED_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

pub(crate) fn scx_dump_complete_latch() -> Arc<Latch> {
    SCX_DUMP_COMPLETE_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// True if `needle` occurs in `haystack`.
fn slice_find(haystack: &[u8], needle: &[u8]) -> bool {
    needle.len() <= haystack.len() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Scan a freshly-read trace_pipe chunk for the sched_ext exit-dump
/// start + end markers, firing [`scx_dump_started_latch`] /
/// [`scx_dump_complete_latch`]. `tail` carries the last
/// [`SCAN_TAIL_KEEP`] bytes of the previous chunk so a marker split
/// across a read boundary is still matched. No-op once the dump is
/// complete (the common case after one full dump).
pub(crate) fn scan_dump_markers(chunk: &[u8], tail: &mut Vec<u8>) {
    if scx_dump_complete_latch().is_set() {
        return;
    }
    tail.extend_from_slice(chunk);
    if !scx_dump_started_latch().is_set() && slice_find(tail, b"sched_ext_dump:") {
        scx_dump_started_latch().set();
    }
    if slice_find(tail, b"SCX_EV_SUB_BYPASS_DISPATCH") || slice_find(tail, b"~~~~ TRUNCATED ~~~~") {
        scx_dump_complete_latch().set();
        tail.clear();
        return;
    }
    let excess = tail.len().saturating_sub(SCAN_TAIL_KEEP);
    if excess > 0 {
        tail.drain(..excess);
    }
}

/// Process-wide latch fired by the guest's `hvc0_poll_loop` when the
/// host's `bpf-map-write` thread pushes `SIGNAL_BPF_WRITE_DONE` through
/// virtio-console RX.
///
/// Producer: [`hvc0_poll_loop`] (this file). Consumer: the scenario
/// executor's [`crate::scenario::Ctx::wait_for_map_write`] gate
/// (in `scenario::ops`). A test that declares `bpf_map_write` on
/// its `KtstrTestEntry` flips `wait_for_map_write=true`; the
/// scenario runner then blocks on this latch's
/// [`Latch::wait_timeout`] before starting the workload phase, so
/// the workload never observes a stale BPF map value.
///
/// `OnceLock` so the first caller materialises the [`Latch`] and
/// every subsequent caller (producer or consumer) shares the same
/// instance. `Arc` so callers can hold the latch across
/// thread-spawn boundaries without re-resolving the static.
static BPF_MAP_WRITE_DONE_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Shared `accessor_ready` latch — fired by `hvc0_poll_loop` on
/// `SIGNAL_ACCESSOR_READY`, awaited by
/// `scenario::ops::await_accessor_ready`. Mirrors
/// [`BPF_MAP_WRITE_DONE_LATCH`].
static ACCESSOR_READY_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Shared `probe_dump_ready` latch. The host fires this only after the
/// exact failure-dump decoder has read the probe's complete per-CPU
/// counter slab. An opt-in guest waits on it before scheduler launch,
/// keeping scheduler-relative fault timers behind diagnostic readiness.
static PROBE_DUMP_READY_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Lazily materialise and return the shared `bpf_map_write_done`
/// latch. Both the producer (`hvc0_poll_loop`) and consumer (scenario
/// `wait_for_map_write` gate) reach for this — the first caller
/// installs the [`Latch`] into [`BPF_MAP_WRITE_DONE_LATCH`], every
/// subsequent caller observes the same instance.
pub(crate) fn bpf_map_write_done_latch() -> Arc<Latch> {
    BPF_MAP_WRITE_DONE_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Lazily materialise and return the shared `accessor_ready` latch.
/// The producer (`hvc0_poll_loop`, on `SIGNAL_ACCESSOR_READY`) and the
/// consumer (`scenario::ops::await_accessor_ready`) both reach for this;
/// the first caller installs the [`Latch`], every subsequent caller
/// observes the same instance. Mirrors [`bpf_map_write_done_latch`].
pub(crate) fn accessor_ready_latch() -> Arc<Latch> {
    ACCESSOR_READY_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Lazily materialise and return the shared `probe_dump_ready` latch.
/// The producer is [`hvc0_poll_loop`] on `SIGNAL_PROBE_DUMP_READY`; the
/// consumer is the opt-in pre-scheduler gate in guest init.
pub(crate) fn probe_dump_ready_latch() -> Arc<Latch> {
    PROBE_DUMP_READY_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Shared `periodic_prereqs_ready` latch — fired by `hvc0_poll_loop` on
/// [`crate::vmm::virtio_console::SIGNAL_PERIODIC_READY`], awaited by the
/// dispatch path before `send_scenario_start` when the run declares
/// periodic captures (`KTSTR_AWAIT_PERIODIC_READY=1`). Mirrors
/// [`ACCESSOR_READY_LATCH`] but carries the KASLR-inclusive prereq
/// state so the capture window opens only once ALL prereqs hold.
static PERIODIC_PREREQS_READY_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Lazily materialise and return the shared `periodic_prereqs_ready`
/// latch. Mirrors [`accessor_ready_latch`].
pub(crate) fn periodic_prereqs_ready_latch() -> Arc<Latch> {
    PERIODIC_PREREQS_READY_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Shared wprof-artifact acknowledgement latch. The host fires this only
/// after it has drained the terminal trace frame and, for auto-repro, the
/// probe payload terminator. Guest init waits after probe finalisation and
/// before reboot, so the wait is outside the measured scenario and adds no
/// workload wakeups.
static WPROF_ARTIFACTS_RECEIVED_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

pub(crate) fn wprof_artifacts_received_latch() -> Arc<Latch> {
    WPROF_ARTIFACTS_RECEIVED_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HvcControlEvent {
    Signal(u8),
    AttachControl(crate::vmm::wire::AttachControlPacket),
}

#[derive(Default)]
struct HvcControlDecoder {
    attach_control: Vec<u8>,
}

impl HvcControlDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<HvcControlEvent> {
        let mut events = Vec::new();
        for &byte in bytes {
            self.push_byte(byte, &mut events);
        }
        events
    }

    fn push_byte(&mut self, byte: u8, events: &mut Vec<HvcControlEvent>) {
        use crate::vmm::wire::{
            ATTACH_CONTROL_PACKET_SIZE, decode_attach_control, is_attach_control_signal,
        };

        if self.attach_control.is_empty() {
            if is_attach_control_signal(byte) {
                self.attach_control.push(byte);
            } else {
                events.push(HvcControlEvent::Signal(byte));
            }
            return;
        }

        if byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase() {
            self.attach_control.push(byte);
            if self.attach_control.len() == ATTACH_CONTROL_PACKET_SIZE {
                match decode_attach_control(&self.attach_control) {
                    Some(packet) => events.push(HvcControlEvent::AttachControl(packet)),
                    None => tracing::warn!(
                        "ktstr-init: rejected malformed scheduler-attach control packet"
                    ),
                }
                self.attach_control.clear();
            }
            return;
        }

        // A malformed suffix does not swallow the byte which exposed it.
        // Re-run that byte through the idle state so a regular control signal
        // or a fresh typed packet prefix is still delivered.
        tracing::warn!("ktstr-init: discarded malformed scheduler-attach control prefix");
        self.attach_control.clear();
        self.push_byte(byte, events);
    }
}

/// Start the hvc0 wake-byte poll loop.
///
/// Spawns a background thread that polls `/dev/hvc0` for host→guest
/// wake bytes and dispatches SysRq-D / shutdown / bpf-map-write-done
/// based on the wake byte. Returns the thread's stop flag so callers
/// can request termination on teardown.
///
/// `trace_stop` is the trace_pipe reader's stop flag. The graceful
/// shutdown handler sets it so the reader enters drain mode.
///
/// `probe_drain` carries the probe pipeline's `stop` + `output_done`
/// handles (present only when a probe stack is attached — the
/// auto-repro repro VM). The graceful-shutdown handler drains it so any
/// crash-arg probe output captured so far is emitted before the guest
/// reboots. The payload travels the virtio bulk port, not COM2:
/// `emit_probe_payload` `println!`s it to stdout, which
/// `redirect_stdio_to_bulk_port` has dup2'd onto a forwarder pipe that
/// ships `MsgType::Stdout` frames over the bulk port; the host recovers
/// it when it drains that port at teardown. On a repro scenario that
/// HANGS inside the test function this is the ONLY drain site — the
/// trigger / Phase-6b / scheduler-death drains all require the run to
/// make progress — so without it the captured events are lost to the
/// watchdog reboot.
pub(crate) fn start_hvc0_poll(
    trace_stop: Option<Arc<AtomicBool>>,
    probe_drain: Option<super::scheduler::ProbeDrain>,
) -> std::io::Result<Arc<AtomicBool>> {
    let hvc0 = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(HVC0)?;
    let mut tio = nix::sys::termios::tcgetattr(&hvc0)
        .map_err(|error| std::io::Error::other(format!("tcgetattr({HVC0}): {error}")))?;
    nix::sys::termios::cfmakeraw(&mut tio);
    nix::sys::termios::tcsetattr(&hvc0, nix::sys::termios::SetArg::TCSANOW, &tio)
        .map_err(|error| std::io::Error::other(format!("tcsetattr({HVC0}): {error}")))?;
    super::scheduler::scheduler_attach_control_reader_started().map_err(|error| {
        std::io::Error::other(format!(
            "publish hvc0 scheduler-attach control reader start: {error}"
        ))
    })?;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    let spawn_result = std::thread::Builder::new()
        .name("hvc0-poll".into())
        .spawn(move || {
            let reason = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                hvc0_poll_loop(
                    hvc0,
                    &stop_clone,
                    trace_stop.as_deref(),
                    probe_drain.as_ref(),
                )
            })) {
                Ok(reason) => reason,
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(|message| (*message).to_owned())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "non-string panic payload".into());
                    format!("hvc0 control reader panicked: {detail}")
                }
            };
            if let Err(error) =
                super::scheduler::scheduler_attach_control_reader_terminated(reason.clone())
            {
                tracing::error!(
                    reason,
                    error = %error,
                    "ktstr-init: failed to publish hvc0 control-reader termination"
                );
            }
        });
    if let Err(error) = spawn_result {
        let reason = format!("spawn hvc0 control reader: {error}");
        if let Err(publish_error) =
            super::scheduler::scheduler_attach_control_reader_terminated(reason.clone())
        {
            tracing::error!(
                error = %publish_error,
                "ktstr-init: failed to publish hvc0 control-reader spawn failure"
            );
        }
        return Err(std::io::Error::other(reason));
    }

    Ok(stop)
}

/// Poll `/dev/hvc0` for host→guest wake bytes and dispatch SysRq-D /
/// shutdown / bpf-map-write-done based on the wake byte alone.
///
/// Wake source: opens `/dev/hvc0` non-blocking (`O_NONBLOCK`) and
/// `poll()`s the fd with `POLLIN` at a 1000 ms safety timeout. The
/// host pushes a byte via `VirtioConsole::queue_input` whenever it
/// requests a dump (`SIGNAL_VC_DUMP`), a graceful shutdown
/// (`SIGNAL_VC_SHUTDOWN`), or a `bpf-map-write`-complete notification
/// (`SIGNAL_BPF_WRITE_DONE`). The poll wakes within microseconds of
/// the push.
///
/// On any wake the loop:
///   1. scans every drained hvc0 byte for `SIGNAL_VC_DUMP`; on
///      observing one, triggers SysRq-D via `/proc/sysrq-trigger`.
///   2. scans every drained hvc0 byte for `SIGNAL_BPF_WRITE_DONE`;
///      on observing one, fires [`bpf_map_write_done_latch`] so the
///      scenario's `wait_for_map_write` gate resumes.
///   3. scans every drained hvc0 byte for `SIGNAL_VC_SHUTDOWN`; on
///      observing one, drives graceful shutdown (drain the probe
///      pipeline so it emits over the bulk port, set `trace_stop`,
///      disable tracing, flush stdio + serial) and breaks.
fn hvc0_poll_loop(
    hvc0: fs::File,
    stop: &AtomicBool,
    trace_stop: Option<&AtomicBool>,
    probe_drain: Option<&super::scheduler::ProbeDrain>,
) -> String {
    use std::os::unix::io::AsRawFd;

    // The caller opened the device and established raw mode synchronously
    // before spawning this thread. AttachStarted cannot be published until
    // that succeeds, so every host cancellation has a live reader.
    let poll_timeout_ms: PollTimeout = 1000u16.into();
    let mut decoder = HvcControlDecoder::default();

    loop {
        if stop.load(Ordering::Acquire) {
            return "hvc0 control reader stop flag set".into();
        }
        let borrowed = unsafe { BorrowedFd::borrow_raw(hvc0.as_raw_fd()) };
        let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
        match poll(&mut fds, poll_timeout_ms) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(error) => return format!("poll({HVC0}) failed: {error}"),
        }
        // Inspect revents before reading: a host-side virtio-console
        // disconnect raises POLLHUP/POLLERR permanently, and without
        // this guard the bare `read().unwrap_or(0)` below returns
        // Ok(0) every iteration, the next `poll()` returns
        // immediately because the hangup is still latched, and the
        // loop spins burning CPU until `stop` is set. Mirrors the
        // pattern in `start_trace_pipe` (above): break on
        // POLLERR/POLLNVAL, break on POLLHUP-without-POLLIN, and
        // skip the read on a wake without POLLIN.
        if let Some(revents) = fds[0].revents() {
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                return format!("poll({HVC0}) returned terminal revents {revents:?}");
            }
            if !revents.contains(PollFlags::POLLIN) {
                if revents.contains(PollFlags::POLLHUP) {
                    return format!("poll({HVC0}) returned hangup without readable data");
                }
                continue;
            }
        }
        let mut buf = [0u8; 64];
        let mut hvc_ref: &fs::File = &hvc0;
        // Retry on EINTR (the read was interrupted by a signal before
        // returning data). The previous `unwrap_or(0)` collapsed both
        // EINTR and EIO into 0 bytes, masking transient signal races
        // (drops a real wake byte) and permanent device errors (silent
        // hang in the next poll iteration). Treat:
        //   - Ok(n): consume n bytes and dispatch signals below. An
        //     `Ok(0)` after POLLIN means the control stream reached EOF,
        //     so terminate the reader and publish that exact health reason.
        //   - EINTR: retry the read inline; poll already confirmed
        //     POLLIN, so the wake byte is still in the device's RX
        //     queue waiting to be drained.
        //   - other Err: log via tracing::warn and terminate the reader
        //     with the exact I/O error. A non-EINTR error after POLLIN means
        //     the device is in an unrecoverable state (host-side
        //     disconnect that didn't surface as POLLHUP, kernel-side
        //     I/O error, fd revoked) and continuing would either
        //     spin on the same error or silently miss every wake
        //     byte for the rest of the run.
        let n = loop {
            match hvc_ref.read(&mut buf) {
                Ok(0) => return format!("read({HVC0}) returned EOF"),
                Ok(n) => break n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: hvc0 read failed; aborting poll loop"
                    );
                    return format!("read({HVC0}) failed: {e}");
                }
            }
        };
        let mut shutdown = false;
        for event in decoder.push(&buf[..n]) {
            match event {
                HvcControlEvent::AttachControl(packet) => {
                    let delivery = match packet {
                        crate::vmm::wire::AttachControlPacket::StartedAck { generation } => {
                            super::scheduler::acknowledge_scheduler_attach_started(generation)
                        }
                        crate::vmm::wire::AttachControlPacket::FinishedAck { generation } => {
                            super::scheduler::acknowledge_scheduler_attach_finished(generation)
                        }
                        crate::vmm::wire::AttachControlPacket::Cancel { generation, cause } => {
                            super::scheduler::cancel_scheduler_attach(generation, cause)
                        }
                    };
                    if let Err(error) = delivery {
                        tracing::error!(
                            ?packet,
                            error = %error,
                            "ktstr-init: failed to deliver scheduler-attach control packet"
                        );
                    }
                }
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_VC_DUMP) => {
                    let _ = fs::write("/proc/sysrq-trigger", "D");
                }
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE) => {
                    bpf_map_write_done_latch().set();
                }
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_ACCESSOR_READY) => {
                    accessor_ready_latch().set();
                }
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_PROBE_DUMP_READY) => {
                    probe_dump_ready_latch().set();
                }
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_PERIODIC_READY) => {
                    periodic_prereqs_ready_latch().set();
                }
                HvcControlEvent::Signal(
                    crate::vmm::virtio_console::SIGNAL_WPROF_ARTIFACTS_RECEIVED,
                ) => {
                    wprof_artifacts_received_latch().set();
                }
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_VC_SHUTDOWN) => {
                    shutdown = true;
                }
                HvcControlEvent::Signal(_) => {}
            }
        }
        if shutdown {
            tracing::info!("ktstr-init: shutdown request received, draining");
            // Drain the probe pipeline FIRST so any crash-arg probe events
            // captured so far are EMITTED (via println! -> the stdout
            // forwarder -> the bulk port) before this handler returns. On a
            // repro scenario that hung inside the test function this is the
            // only drain site (the trigger / Phase-6b / scheduler-death drains
            // all require progress). The subsequent stdout().flush() below
            // pushes the emitted bytes into the forwarder pipe; the forwarder
            // ships them over the bulk port while the BSP is still alive in the
            // soft window, and the host's teardown port-drain recovers any
            // residual. This handler does NOT force_reboot (it only breaks), so
            // the forwarder keeps running until the host kills the VM —
            // delivery must not be made reboot-dependent without first joining
            // the forwarder. The 2.5s bound fits inside the watchdog's 3s
            // soft-shutdown window so the flush + tcdrain below also run before
            // the hard deadline; on a wedged probe the watchdog hard deadline
            // (BSP kick + host port-drain) is the ultimate net, so the bound is
            // a courtesy cap, not the operative one.
            drain_probe_for_shutdown(probe_drain, std::time::Duration::from_millis(2500));
            if let Some(ts) = trace_stop {
                ts.store(true, Ordering::Release);
            }
            let _ = fs::write(TRACE_TRACING_ON, "0");
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            if let Ok(f) = fs::OpenOptions::new().write(true).open(COM1) {
                unsafe {
                    libc::tcdrain(std::os::unix::io::AsRawFd::as_raw_fd(&f));
                }
            }
            if let Ok(f) = fs::OpenOptions::new().write(true).open(COM2) {
                unsafe {
                    libc::tcdrain(std::os::unix::io::AsRawFd::as_raw_fd(&f));
                }
            }
            return "hvc0 control reader received host shutdown request".into();
        }
    }
}

/// Drain the probe pipeline on graceful shutdown: request the probe thread to
/// stop, then wait (bounded by `timeout`) for it to emit its payload and signal
/// `output_done`. Returns `true` if `output_done` was observed within the bound
/// (or no probe drain is attached — a no-op), `false` on timeout.
///
/// `output_done` means the probe thread's `emit_probe_payload` `println!`s
/// returned — i.e. the `PROBE_OUTPUT_*` payload is in the stdout forwarder pipe,
/// NOT that the host has received it. Delivery is asynchronous: the detached
/// stdout forwarder (`redirect_stdio_to_bulk_port`) ships the bytes over the
/// virtio bulk port, and the host recovers them when it drains that port at
/// teardown. This is correct here because the caller (the `SIGNAL_VC_SHUTDOWN`
/// handler) does NOT force_reboot, so the forwarder stays alive to ship the
/// chunk. We deliberately do NOT join the forwarder (unlike a reference VMM such
/// as libkrun, whose console teardown joins its tx thread) — the guest never
/// self-reboots on this path, so the host's teardown port-drain is the catch.
///
/// Bounded because this runs on the watchdog's soft-shutdown path: the watchdog
/// hard deadline is the outer safety net, so this uses a tight bound (2500ms)
/// fitting that 3s window. `drain_probe_pipeline` ALSO bounds its wait, but at
/// the larger `PROBE_DRAIN_GRACE` (30s) — the pre-dispatch early-bail paths have
/// no watchdog window, so they ride the host's VM-deadline grace instead. Here
/// the watchdog net is the very thing that triggered this drain, so the tighter
/// bound is required.
fn drain_probe_for_shutdown(
    probe_drain: Option<&super::scheduler::ProbeDrain>,
    timeout: std::time::Duration,
) -> bool {
    let Some(pd) = probe_drain else {
        return true;
    };
    pd.stop.store(true, Ordering::Release);
    pd.output_done.wait_timeout(timeout)
}

/// Terminal scheduler state observed through its exact pidfd.
///
/// This is also carried by [`SchedExitCommitError::Terminal`] when
/// [`SchedExitStop::commit_with`] rejects a provisional owner.
/// A scheduler which has already exited cannot be published as the current
/// owner, and a pidfd whose readiness can no longer be observed is equally
/// unsafe to publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SchedExitTerminal {
    Exited,
    ObserverError(String),
}

impl std::fmt::Display for SchedExitTerminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exited => write!(f, "scheduler exited before monitor commit"),
            Self::ObserverError(error) => {
                write!(
                    f,
                    "scheduler pidfd observation failed before monitor commit: {error}"
                )
            }
        }
    }
}

impl std::error::Error for SchedExitTerminal {}

/// Failure to publish a provisional scheduler owner and arm its exit monitor.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SchedExitCommitError<E> {
    /// The exact scheduler pidfd was already terminal, or its observer had
    /// already failed, before ordered publication began.
    Terminal(SchedExitTerminal),
    /// The caller's ordered success-publication step returned an error.
    Publication(E),
    /// The caller's ordered success-publication step unwound. The gate catches
    /// the panic while holding `Pending`, aborts publication, and returns the
    /// diagnostic rather than poisoning the gate mutex.
    PublicationPanicked(String),
    /// A handle was committed twice or committed after explicit abort.
    InvalidState(&'static str),
}

impl<E: std::fmt::Display> std::fmt::Display for SchedExitCommitError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Terminal(terminal) => terminal.fmt(f),
            Self::Publication(error) => write!(f, "scheduler success publication failed: {error}"),
            Self::PublicationPanicked(detail) => {
                write!(f, "scheduler success publication panicked: {detail}")
            }
            Self::InvalidState(state) => {
                write!(f, "scheduler exit monitor cannot commit from {state}")
            }
        }
    }
}

impl<E> std::error::Error for SchedExitCommitError<E> where E: std::fmt::Debug + std::fmt::Display {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedExitPublicationState {
    /// The monitor owns the pidfd but is not allowed to publish an exit.
    Pending,
    /// The process owner is installed and exit publication is armed.
    Committed,
    /// Orderly teardown won the publication race.
    Aborted,
    /// The monitor atomically claimed the one permitted publication.
    Published,
}

#[derive(Debug)]
struct SchedExitPublicationInner {
    state: SchedExitPublicationState,
    terminal: Option<SchedExitTerminal>,
}

#[derive(Debug)]
struct SchedExitPublicationGate {
    inner: std::sync::Mutex<SchedExitPublicationInner>,
}

impl SchedExitPublicationGate {
    fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(SchedExitPublicationInner {
                state: SchedExitPublicationState::Pending,
                terminal: None,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SchedExitPublicationInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Record a terminal observation without publishing it.
    ///
    /// The return value tells the monitor whether the owner was committed at
    /// the observation linearization point. A pending observation terminates
    /// the monitor immediately: commit will find the recorded terminal value
    /// and reject the process, and no log/probe drain is entered.
    fn record_terminal(&self, terminal: SchedExitTerminal) -> bool {
        let mut inner = self.lock();
        if inner.terminal.is_none() {
            inner.terminal = Some(terminal);
        }
        inner.state == SchedExitPublicationState::Committed
    }

    /// Claim the only permitted exit publication.
    ///
    /// Drains happen between [`Self::record_terminal`] and this call. An
    /// orderly stop can therefore move `Committed -> Aborted` while a drain is
    /// in progress and suppress the frame. Once this method moves the state to
    /// `Published`, publication won the race and must complete exactly once.
    fn claim_publication(&self) -> bool {
        let mut inner = self.lock();
        if inner.state == SchedExitPublicationState::Committed && inner.terminal.is_some() {
            inner.state = SchedExitPublicationState::Published;
            true
        } else {
            false
        }
    }

    fn abort(&self) {
        let mut inner = self.lock();
        if inner.state != SchedExitPublicationState::Published {
            inner.state = SchedExitPublicationState::Aborted;
        }
    }

    /// Arm publication only if the exact handed-off pidfd is still live.
    ///
    /// The nonblocking pidfd poll intentionally runs while the gate mutex is
    /// held. That closes the handoff race where the process exits but the
    /// monitor thread has not yet acquired this mutex to record the edge:
    ///
    /// * readiness before/during this poll is recorded and rejected here;
    /// * readiness after this poll can only be recorded after
    ///   `Pending -> Committed`, so the monitor publishes it.
    fn commit_with<T, E>(
        &self,
        pidfd: &OwnedFd,
        publish_success: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, SchedExitCommitError<E>> {
        let mut inner = self.lock();
        match inner.state {
            SchedExitPublicationState::Committed | SchedExitPublicationState::Published => {
                return Err(SchedExitCommitError::InvalidState("committed state"));
            }
            SchedExitPublicationState::Aborted => {
                return match inner.terminal.clone() {
                    Some(terminal) => Err(SchedExitCommitError::Terminal(terminal)),
                    None => Err(SchedExitCommitError::InvalidState("aborted state")),
                };
            }
            SchedExitPublicationState::Pending => {}
        }

        if let Some(terminal) = inner.terminal.clone() {
            inner.state = SchedExitPublicationState::Aborted;
            return Err(SchedExitCommitError::Terminal(terminal));
        }

        if let Some(terminal) = sched_exit_terminal_now(pidfd) {
            inner.terminal = Some(terminal.clone());
            inner.state = SchedExitPublicationState::Aborted;
            return Err(SchedExitCommitError::Terminal(terminal));
        }

        // Publication remains Pending for the complete caller-defined success
        // boundary. A racing monitor can poll the terminal pidfd, but it cannot
        // acquire this mutex to record or claim the exit until the closure has
        // either completed and we arm Committed below, or failed and we abort.
        let published = std::panic::catch_unwind(std::panic::AssertUnwindSafe(publish_success));
        let value = match published {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                inner.state = SchedExitPublicationState::Aborted;
                return Err(SchedExitCommitError::Publication(error));
            }
            Err(payload) => {
                let detail = payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".into());
                inner.state = SchedExitPublicationState::Aborted;
                return Err(SchedExitCommitError::PublicationPanicked(detail));
            }
        };
        inner.state = SchedExitPublicationState::Committed;
        Ok(value)
    }

    #[cfg(test)]
    fn commit(&self, pidfd: &OwnedFd) -> Result<(), SchedExitTerminal> {
        match self.commit_with(pidfd, || Ok::<(), std::convert::Infallible>(())) {
            Ok(()) => Ok(()),
            Err(SchedExitCommitError::Terminal(terminal)) => Err(terminal),
            Err(SchedExitCommitError::Publication(never)) => match never {},
            Err(SchedExitCommitError::PublicationPanicked(detail)) => {
                Err(SchedExitTerminal::ObserverError(format!(
                    "infallible scheduler monitor commit panicked: {detail}"
                )))
            }
            Err(SchedExitCommitError::InvalidState(state)) => {
                Err(SchedExitTerminal::ObserverError(format!(
                    "scheduler monitor commit called from {state}"
                )))
            }
        }
    }
}

/// Perform an authoritative nonblocking terminal-state probe on `pidfd`.
fn sched_exit_terminal_now(pidfd: &OwnedFd) -> Option<SchedExitTerminal> {
    loop {
        let mut pfd = libc::pollfd {
            fd: pidfd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is one initialized pollfd on the stack, `nfds`
        // matches, and timeout zero makes this an immediate readiness probe.
        let result = unsafe { libc::poll(&mut pfd, 1, 0) };
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Some(SchedExitTerminal::ObserverError(format!(
                "pidfd poll failed: {error}"
            )));
        }
        return match sched_exit_observed(pfd.revents) {
            Ok(true) => Some(SchedExitTerminal::Exited),
            Ok(false) => None,
            Err(error) => Some(SchedExitTerminal::ObserverError(error)),
        };
    }
}

/// Pending/committed stop handle for the sched-exit monitor. Carries the
/// `Arc<AtomicBool>` source-of-truth flag, an exact pidfd clone used by
/// [`Self::commit_with`], a writable eventfd handle the cleanup site uses to
/// wake the monitor thread out of `poll(2)` without waiting for the legacy
/// 250 ms cadence, and the monitor thread's `JoinHandle` so the cleanup site
/// can wait for the thread to actually exit before proceeding.
///
/// Cleanup contract: before any action that could be misinterpreted
/// by the monitor as an unexpected scheduler exit (e.g. `child.kill()`
/// on the scheduler), the cleanup site MUST call
/// [`SchedExitStop::stop_and_join`]. The method aborts publication under the
/// same mutex used by exit observation before it sets the stop flag and wakes
/// the thread; manually setting the flag is not an equivalent handoff.
/// Otherwise the monitor races: the kill makes the pidfd readable (including
/// while a zombie still has a `/proc/{pid}` entry), the monitor takes the
/// `if exited` branch, and emits `MSG_TYPE_SCHED_EXIT` to the host, which
/// terminates the VM before the orderly `MSG_TYPE_EXIT` frame can be sent.
///
/// The bool is the source of truth; the eventfd write delivers the
/// edge that pulls the thread out of an indefinite `poll`. The
/// eventfd is owned by this struct on the writer side and by the
/// monitor thread on the reader side; both sides drop their fds when
/// the run ends, so the kernel-side counter is reclaimed cleanly.
pub(crate) struct SchedExitStop {
    /// Stop flag the monitor thread loads under `Acquire` after every
    /// `poll(2)` wake and before emitting an exit frame. The eventfd below
    /// supplies the wake edge paired with this source-of-truth flag.
    stop: Arc<AtomicBool>,
    /// Independent descriptor for the same process identity watched by the
    /// monitor thread. Commit probes this descriptor while holding the
    /// publication-gate mutex, closing the exit-before-publication race without
    /// reopening a numeric pid.
    commit_pidfd: OwnedFd,
    publication: Arc<SchedExitPublicationGate>,
    /// Owned eventfd write side. `wake()` writes `1` here; the
    /// monitor's `poll(2)` returns within microseconds. `None` when
    /// `eventfd(2)` failed or its writer-side `dup(2)` failed at monitor
    /// spawn (the legacy 250 ms timeout still bounds wake latency in that
    /// degraded path).
    wake_fd: Option<OwnedFd>,
    /// Monitor thread join handle. Construction cannot succeed without a
    /// live monitor thread: `std::thread::Builder::spawn` errors are returned
    /// synchronously to the caller. Consumed by
    /// [`SchedExitStop::stop_and_join`].
    join_handle: Option<std::thread::JoinHandle<bool>>,
}

impl SchedExitStop {
    /// Publish the monitor as the current scheduler's exit observer.
    ///
    /// Until this succeeds, an exit is recorded privately and produces no log
    /// drain or `SchedExit` frame. The final pidfd probe and
    /// `Pending -> Committed` transition share the publication mutex, so there
    /// is no interval in which a dead process can be committed merely because
    /// its monitor thread has not yet run.
    #[cfg(test)]
    pub(crate) fn commit(&self) -> Result<(), SchedExitTerminal> {
        self.publication.commit(&self.commit_pidfd)
    }

    /// Publish the caller's ordered success boundary and arm this monitor as
    /// one indivisible gate transition.
    ///
    /// The closure runs while the gate mutex is held and the monitor remains
    /// `Pending`. A racing terminal pidfd edge therefore cannot emit before
    /// the closure completes. On closure success the state becomes
    /// `Committed` before unlock; on error or panic it becomes `Aborted` and
    /// can never publish.
    pub(crate) fn commit_with<T, E>(
        &self,
        publish_success: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, SchedExitCommitError<E>> {
        self.publication
            .commit_with(&self.commit_pidfd, publish_success)
    }

    /// Wake the monitor thread out of its `poll(2)` wait. Idempotent
    /// — eventfd in counter mode coalesces multiple writes into a
    /// single wake. EAGAIN under `EFD_NONBLOCK` (counter saturation —
    /// physically impossible with a single writer + 64-bit counter)
    /// is silently absorbed; the `Acquire`-loaded `stop` bool above
    /// remains the source of truth.
    pub(crate) fn wake(&self) {
        if let Some(ref fd) = self.wake_fd {
            // SAFETY: `fd` is the owned write side of an eventfd
            // created with `EFD_NONBLOCK`; a single 8-byte write of
            // a non-zero u64 advances the counter and edge-fires
            // every reader's `poll(POLLIN)`. The bytes pointer is a
            // 64-bit aligned local; `count` is exactly 8 as
            // eventfd(2) requires.
            let val: u64 = 1;
            let bytes = val.to_ne_bytes();
            let _ = unsafe {
                libc::write(
                    fd.as_raw_fd(),
                    bytes.as_ptr() as *const libc::c_void,
                    bytes.len(),
                )
            };
        }
    }

    /// Atomically request stop and wait for the monitor thread to
    /// exit. Sets `stop=true` (Release) and writes the wake eventfd
    /// so the monitor's `poll(2)` returns within microseconds, then
    /// joins the thread. Returns true when the monitor observed the
    /// scheduler's pidfd exit edge, including when that edge raced with the
    /// stop request and therefore produced no SchedExit frame. A monitor
    /// panic also returns true so verifier cleanup fails closed when liveness
    /// could not be observed.
    /// After this returns the monitor thread is gone, so the caller can
    /// intentionally kill the scheduler without generating a new
    /// `MSG_TYPE_SCHED_EXIT`. A frame for an earlier unexpected exit may
    /// already have been sent; the true return preserves that fact.
    ///
    /// `JoinHandle::join` propagates a panic from the monitor closure as
    /// `Err`; verifier cleanup treats that as lost liveness evidence.
    pub(crate) fn stop_and_join(mut self) -> bool {
        self.request_stop();
        self.join_handle
            .take()
            .expect("scheduler-exit monitor handle is consumed exactly once")
            .join()
            .unwrap_or(true)
    }

    fn request_stop(&self) {
        self.publication.abort();
        self.stop.store(true, Ordering::Release);
        self.wake();
    }
}

impl Drop for SchedExitStop {
    fn drop(&mut self) {
        let Some(join_handle) = self.join_handle.take() else {
            return;
        };
        // A dropped provisional/owned monitor must never detach and outlive
        // its process record. Abort publication before waking it, then fence
        // the thread exactly as the explicit teardown path does.
        self.request_stop();
        let _ = join_handle.join();
    }
}

/// Interpret a scheduler pidfd poll result.
///
/// The handed-in pidfd is the scheduler's stable process identity, and its
/// readiness remains authoritative after the numeric pid has been reaped or
/// reused. No `/proc/<pid>` observation participates in this decision.
pub(crate) fn sched_exit_observed(pidfd_revents: libc::c_short) -> Result<bool, String> {
    if pidfd_revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
        return Err(format!(
            "scheduler pidfd returned invalid poll events {pidfd_revents:#x}"
        ));
    }
    Ok(pidfd_revents & (libc::POLLIN | libc::POLLHUP) != 0)
}

/// Poll timeout for the exit monitor. An indefinite wait is safe only when
/// both sides of the stop eventfd exist; otherwise the stop flag needs the
/// legacy bounded recheck.
pub(crate) fn sched_exit_poll_timeout(
    stop_fd: libc::c_int,
    wake_writer_available: bool,
) -> libc::c_int {
    if stop_fd >= 0 && wake_writer_available {
        -1
    } else {
        250
    }
}

/// Monitor the scheduler child process for unexpected exit.
///
/// Blocks the monitor thread in `poll(2)` against the scheduler's
/// pidfd plus a stop-eventfd; the wait returns when either the
/// child exits (pidfd POLLIN edge from the kernel's `do_notify_pidfd`)
/// or the cleanup site fires the stop-eventfd. Pidfd readiness is
/// authoritative even after the task is reaped; numeric pid and procfs
/// observations never participate in exit detection. On the scheduler's exit
/// two things are gated independently:
///   - `suppress_sched_log` gates the scheduler-log dump only: false
///     (normal mode) completes the resumable log transaction over the bulk
///     port (`dump_sched_output_before_terminal`); true (probes active)
///     instead waits on the probe thread's `output_done` latch — keeping the
///     VM alive until the probe has emitted its payload — and skips the dump
///     (the probe pipeline handles crash detection via its
///     kernel-selected typed scheduler-exit trigger).
///   - the SCHED_EXIT signal (MSG_TYPE_SCHED_EXIT, which lets the host
///     terminate the VM early) is then sent UNLESS the `stop` flag is set
///     (a host-initiated kill, where the exit is expected). It is gated by
///     `stop`, NOT by `suppress_sched_log` — so with probes active and a
///     genuine crash it still fires, but only after the `output_done` wait
///     above, so the probe payload is already out.
///
/// Uses the pidfd opened at scheduler spawn rather than waitpid because most
/// guest-init phases run with SIGCHLD ignored (the kernel auto-reaps children,
/// making waitpid return ECHILD). Carrying that exact descriptor through
/// attach and into this monitor also closes the numeric-pid reuse gap between
/// scheduler readiness and monitor installation.
///
/// The returned [`SchedExitStop`] carries the `Arc<AtomicBool>` the
/// monitor reads, an eventfd the cleanup site writes via
/// [`SchedExitStop::wake`] to drop wake latency from 250 ms (legacy
/// poll timeout) to microseconds, and the monitor thread's
/// `JoinHandle` so [`SchedExitStop::stop_and_join`] can confirm the
/// thread has exited before the caller proceeds with actions
/// (e.g. `child.kill()`) the monitor would otherwise interpret as
/// an unexpected scheduler exit.
///
/// Thread creation is part of installation: a spawn failure is returned
/// synchronously and no stop handle is published.
type SchedExitMonitorTask = Box<dyn FnOnce() -> bool + Send + 'static>;
type SchedExitPublisher = Arc<dyn Fn(i32) + Send + Sync + 'static>;

fn spawn_sched_exit_monitor_thread(
    task: SchedExitMonitorTask,
) -> std::io::Result<std::thread::JoinHandle<bool>> {
    std::thread::Builder::new()
        .name("sched-exit-mon".into())
        .spawn(task)
}

struct SchedExitMonitorConfig<'a> {
    pid: u32,
    pidfd: OwnedFd,
    log_path: Option<&'a str>,
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
    publish_sched_exit: SchedExitPublisher,
    create_wake_writer: bool,
}

fn start_sched_exit_monitor_inner<F>(
    config: SchedExitMonitorConfig<'_>,
    spawn_thread: F,
) -> std::io::Result<SchedExitStop>
where
    F: FnOnce(SchedExitMonitorTask) -> std::io::Result<std::thread::JoinHandle<bool>>,
{
    let SchedExitMonitorConfig {
        pid,
        pidfd,
        log_path,
        suppress_sched_log,
        probe_output_done,
        publish_sched_exit,
        create_wake_writer,
    } = config;
    let log_path = log_path.map(|s| s.to_string());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    // The monitor thread and the publication handle each keep a descriptor for
    // the same kernel pid object. The handle's clone is probed at commit while
    // the publication mutex is held; no numeric-pid reopen participates in the
    // handoff.
    let commit_pidfd = pidfd.try_clone().map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!("clone scheduler pidfd for monitor commit: {error}"),
        )
    })?;
    let publication = Arc::new(SchedExitPublicationGate::new());
    let publication_clone = publication.clone();

    // Allocate a stop-eventfd. Two fds are needed: one owned by the
    // monitor thread (read + close on exit), one owned by the
    // [`SchedExitStop`] writer (`wake` writes here). `dup(2)` shares
    // the underlying counter so a write on either fd advances both
    // sides' visibility. EFD_NONBLOCK so a doubled cleanup path can't
    // stall behind a saturated counter; EFD_CLOEXEC so a future
    // `Command::new` from this thread doesn't leak the fd into a
    // child.
    //
    // `eventfd(2)` failure (extremely unlikely on KVM hosts — the
    // syscall is unconditionally available since kernel 2.6.22) falls
    // back to the legacy 250 ms `poll(2)` timeout: stop still works
    // via the `Acquire`-loaded bool, just with a worst-case 250 ms
    // wake latency instead of microseconds.
    let (monitor_fd, writer_fd): (Option<OwnedFd>, Option<OwnedFd>) = {
        let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if raw < 0 {
            let err = std::io::Error::last_os_error();
            tracing::warn!(
                err = %err,
                "ktstr-init: sched-exit-mon eventfd allocation failed; \
                 falling back to 250 ms stop poll cadence"
            );
            (None, None)
        } else {
            // SAFETY: `eventfd(2)` returned a fresh non-negative fd
            // owned by this caller. Wrapping in `OwnedFd` transfers
            // close-on-drop responsibility; `try_clone` issues a
            // `dup` so writer and monitor each carry an independent
            // fd that addresses the same kernel-side counter. A
            // dup failure leaves the monitor fd alive and disables
            // the wake path (degrades to the no-eventfd branch).
            let monitor_fd = unsafe { OwnedFd::from_raw_fd(raw) };
            match monitor_fd.try_clone() {
                Ok(writer_fd) if !create_wake_writer => {
                    drop(writer_fd);
                    (Some(monitor_fd), None)
                }
                Ok(writer_fd) => (Some(monitor_fd), Some(writer_fd)),
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: sched-exit-mon eventfd dup failed; \
                         falling back to 250 ms stop poll cadence"
                    );
                    (Some(monitor_fd), None)
                }
            }
        }
    };

    let wake_writer_available = writer_fd.is_some();
    let panic_stop = stop.clone();
    let panic_publication = publication.clone();
    let panic_publisher = publish_sched_exit.clone();
    let monitor_body: SchedExitMonitorTask = Box::new(move || {
        // The monitor-side stop fd's raw value, or `-1` when the
        // caller's eventfd allocation or dup failed. `-1` in a
        // pollfd entry is valid: the kernel ignores the slot
        // (returns revents=0), so the same `poll(2)` call works
        // on the degraded path with a finite timeout that
        // re-checks `stop` periodically.
        let stop_fd = monitor_fd.as_ref().map(|f| f.as_raw_fd()).unwrap_or(-1);
        // Poll timeout policy: an indefinite wait is safe only when the
        // monitor read fd and caller-owned writer both exist. If eventfd
        // allocation or writer duplication failed, the legacy 250 ms
        // cadence periodically rechecks the stop flag.
        let poll_timeout = sched_exit_poll_timeout(stop_fd, wake_writer_available);
        loop {
            let exited = {
                // pidfd POLLIN fires at child exit (kernel
                // `pidfd_poll` in `fs/pidfs.c` checks
                // `exit_state`, woken via `do_notify_pidfd`
                // from `exit_notify`). Adding the stop eventfd
                // alongside makes a stop request also wake the
                // poll, so cleanup latency drops from the
                // legacy 250 ms (re-checking `stop` after each
                // `poll` timeout) to the kernel's eventfd
                // wakeup latency (microseconds).
                //
                // Pidfd revents are the sole exit evidence, including
                // for zombies and tasks which have already been reaped.
                let mut pfds = [
                    libc::pollfd {
                        fd: pidfd.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                    libc::pollfd {
                        fd: stop_fd,
                        events: libc::POLLIN,
                        revents: 0,
                    },
                ];
                // SAFETY: pfds is a 2-element pollfd array on
                // the local stack; nfds matches. A `stop_fd`
                // value of `-1` is valid per poll(2) — the
                // kernel skips that slot. The pidfd's stable identity
                // remains authoritative independently of task reaping.
                let poll_result = unsafe {
                    libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, poll_timeout)
                };
                if poll_result < 0 {
                    let error = std::io::Error::last_os_error();
                    if error.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    tracing::error!(
                        pid,
                        err = %error,
                        "ktstr-init: sched-exit pidfd poll failed",
                    );
                    let terminal =
                        SchedExitTerminal::ObserverError(format!("pidfd poll failed: {error}"));
                    if !stop_clone.load(Ordering::Acquire)
                        && publication_clone.record_terminal(terminal)
                        && publication_clone.claim_publication()
                    {
                        publish_sched_exit(1);
                    }
                    return true;
                }
                match sched_exit_observed(pfds[0].revents) {
                    Ok(exited) => exited,
                    Err(error) => {
                        tracing::error!(
                            pid,
                            error = %error,
                            "ktstr-init: scheduler exit monitor lost pidfd observation"
                        );
                        let terminal = SchedExitTerminal::ObserverError(error);
                        if !stop_clone.load(Ordering::Acquire)
                            && publication_clone.record_terminal(terminal)
                            && publication_clone.claim_publication()
                        {
                            publish_sched_exit(1);
                        }
                        return true;
                    }
                }
            };
            if exited {
                // An orderly teardown may race the exit edge. Observe the
                // stop request before entering probe/log drains so monitor
                // shutdown cannot block behind work whose only purpose is
                // reporting an unexpected exit.
                if stop_clone.load(Ordering::Acquire) {
                    return true;
                }
                if !publication_clone.record_terminal(SchedExitTerminal::Exited) {
                    // A provisional scheduler exited before its owner was
                    // published. Commit observes the typed terminal state and
                    // rejects it; entering log/probe drains here would leak a
                    // result for a scheduler which never became current.
                    return true;
                }
                if suppress_sched_log.load(Ordering::Acquire) {
                    // Probes active: wait event-driven on the
                    // probe thread's `output_done` latch.
                    // Outer wall-clock VM timeout is the
                    // safety net for a hung probe — adding a
                    // local timer would cap teardown latency
                    // but also truncate slow-but-progressing
                    // probe drains, which is the exact bug
                    // we're avoiding here.
                    if let Some(ref done) = probe_output_done {
                        while !done.wait_timeout(std::time::Duration::from_millis(100)) {
                            if stop_clone.load(Ordering::Acquire) {
                                return true;
                            }
                        }
                    }
                } else if let Some(ref path) = log_path {
                    let _ = dump_sched_output_before_terminal(path);
                }
                // Suppress SchedExit when the host cleanup
                // initiated the kill (stop flag set before
                // child.kill). Without this gate, Phase 6
                // child.kill → pidfd POLLIN → monitor enters
                // this branch → sends SchedExit → host sets
                // kill=true → BSP exits with ExternalKill
                // before the guest reaches send_exit,
                // producing exit_code=-1 on a clean run.
                if stop_clone.load(Ordering::Acquire) {
                    return true;
                }
                if publication_clone.claim_publication() {
                    let exit_code: i32 = 1;
                    publish_sched_exit(exit_code);
                }
                // `pidfd` and `monitor_fd` are both `OwnedFd`s captured
                // by this task. They close exactly once when the task
                // returns; the writer side remains in SchedExitStop.
                return true;
            }
            // Prioritise a simultaneous pidfd edge over the orderly stop
            // edge. The caller also checks Child::try_wait after joining,
            // covering an exit immediately after this poll snapshot.
            if stop_clone.load(Ordering::Acquire) {
                return false;
            }
            // Drain any pending stop-eventfd reads so the next
            // `poll` doesn't immediately re-fire on the same
            // edge. The `stop` AtomicBool is the source of
            // truth (re-checked after each poll); the eventfd is purely
            // a wake-edge, so a missed read is benign — the next
            // iteration's poll wakes either way. EAGAIN under
            // EFD_NONBLOCK (counter
            // already 0 from a racing reader, or no edge
            // arrived) is the steady-state non-stop case.
            if stop_fd >= 0 {
                let mut buf = [0u8; 8];
                // SAFETY: `stop_fd` is the borrowed read side
                // of an eventfd, valid for the lifetime of
                // this thread (the OwnedFd is owned by the
                // closure's `monitor_fd` and not dropped
                // until the closure returns). `buf` is an
                // 8-byte stack slot matching eventfd(2)'s
                // 8-byte read requirement.
                let _ = unsafe {
                    libc::read(stop_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len())
                };
            }
        }
        // `pidfd` and `monitor_fd` drop here as the closure returns.
    });
    let monitor_task: SchedExitMonitorTask = Box::new(move || {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(monitor_body)) {
            Ok(observed) => observed,
            Err(payload) => {
                let detail = payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".into());
                tracing::error!(pid, detail, "ktstr-init: scheduler exit monitor panicked");
                let terminal = SchedExitTerminal::ObserverError(format!(
                    "scheduler exit monitor panicked: {detail}"
                ));
                if !panic_stop.load(Ordering::Acquire)
                    && panic_publication.record_terminal(terminal)
                    && panic_publication.claim_publication()
                {
                    panic_publisher(1);
                }
                true
            }
        }
    });
    let join_handle = spawn_thread(monitor_task)?;

    Ok(SchedExitStop {
        stop,
        commit_pidfd,
        publication,
        wake_fd: writer_fd,
        join_handle: Some(join_handle),
    })
}

/// Start a scheduler-exit monitor in the unpublishable pending state.
///
/// This is the scheduler-owner handoff primitive. The caller installs its
/// `Child`, retained original pidfd, log path, scheduler identity, and this
/// handle into the single current-process owner, then calls
/// [`SchedExitStop::commit_with`]. Until commit succeeds, scheduler exit or
/// pidfd observer failure is retained as a typed terminal value and emits
/// neither a log drain nor a `SchedExit` frame.
pub(crate) fn start_pending_sched_exit_monitor(
    pid: u32,
    pidfd: OwnedFd,
    log_path: Option<&str>,
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) -> std::io::Result<SchedExitStop> {
    start_sched_exit_monitor_inner(
        SchedExitMonitorConfig {
            pid,
            pidfd,
            log_path,
            suppress_sched_log,
            probe_output_done,
            publish_sched_exit: Arc::new(crate::vmm::guest_comms::send_sched_exit),
            create_wake_writer: true,
        },
        spawn_sched_exit_monitor_thread,
    )
}

/// Start and immediately publish a scheduler-exit monitor.
///
/// Kept for boot/legacy callers which do not have a provisional ownership
/// window. New scheduler handoffs use [`start_pending_sched_exit_monitor`]
/// and commit only after the process owner is installed. Even on this
/// immediate path, an already-terminal exact pidfd is rejected rather than
/// emitting a result for a scheduler the caller never successfully published.
#[cfg(test)]
pub(crate) fn start_sched_exit_monitor(
    pid: u32,
    pidfd: OwnedFd,
    log_path: Option<&str>,
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) -> std::io::Result<SchedExitStop> {
    let monitor = start_pending_sched_exit_monitor(
        pid,
        pidfd,
        log_path,
        suppress_sched_log,
        probe_output_done,
    )?;
    if let Err(terminal) = monitor.commit() {
        let _ = monitor.stop_and_join();
        return Err(std::io::Error::other(terminal));
    }
    Ok(monitor)
}

#[cfg(test)]
pub(crate) fn start_sched_exit_monitor_without_wake_writer_for_test(
    sched_pid: u32,
    pidfd: OwnedFd,
) -> std::io::Result<SchedExitStop> {
    let monitor = start_sched_exit_monitor_inner(
        SchedExitMonitorConfig {
            pid: sched_pid,
            pidfd,
            log_path: None,
            suppress_sched_log: Arc::new(AtomicBool::new(false)),
            probe_output_done: None,
            publish_sched_exit: Arc::new(crate::vmm::guest_comms::send_sched_exit),
            create_wake_writer: false,
        },
        spawn_sched_exit_monitor_thread,
    )?;
    if let Err(terminal) = monitor.commit() {
        let _ = monitor.stop_and_join();
        return Err(std::io::Error::other(terminal));
    }
    Ok(monitor)
}

#[cfg(test)]
pub(crate) fn start_sched_exit_monitor_with_spawn_failure_for_test(
    sched_pid: u32,
    pidfd: OwnedFd,
) -> std::io::Result<SchedExitStop> {
    start_sched_exit_monitor_inner(
        SchedExitMonitorConfig {
            pid: sched_pid,
            pidfd,
            log_path: None,
            suppress_sched_log: Arc::new(AtomicBool::new(false)),
            probe_output_done: None,
            publish_sched_exit: Arc::new(crate::vmm::guest_comms::send_sched_exit),
            create_wake_writer: true,
        },
        |_task| {
            Err(std::io::Error::other(
                "injected sched-exit monitor thread spawn failure",
            ))
        },
    )
}

/// Execute shell-script-like commands from a file.
///
/// Handles the patterns used by sched_enable/sched_disable scripts:
/// - `echo VALUE > /path` (write VALUE to a file)
/// - Lines starting with `#` are comments
/// - Empty lines are ignored
///
/// # Failure surface
///
/// File-not-found is a legitimate "no script" condition (the
/// sched_enable/sched_disable hooks are optional per
/// `ShellTestDescriptor`). Logged at debug level and returns
/// silently. All other read errors are logged at error level —
/// the file exists but couldn't be read (permission denied,
/// I/O error, etc.) is a real defect.
///
/// Per-line failures (file-write failures, unsupported commands)
/// are counted and reported via a single error-level summary at
/// the end. The script is not aborted on first failure —
/// sched_enable/sched_disable hooks are typically independent
/// settings (cpufreq governor, scheduler sysctl, tracing knobs),
/// so the operator gets partial-apply behavior with a loud
/// summary instead of silent partial-apply. Catches
/// silent-drop violations where a typo'd `/sys/`
/// path silently dropped before this rewrite.
#[tracing::instrument]
pub(crate) fn exec_shell_script(path: &str) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(path, "ktstr-init: exec_shell_script: no script (skipping)");
            return;
        }
        Err(e) => {
            tracing::error!(path, err = %e, "ktstr-init: exec_shell_script: read failed");
            return;
        }
    };

    let mut ok_count = 0u32;
    let mut fail_count = 0u32;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if exec_shell_line(line).is_ok() {
            ok_count += 1;
        } else {
            fail_count += 1;
        }
    }
    if fail_count > 0 {
        tracing::error!(
            path,
            ok_count,
            fail_count,
            "ktstr-init: exec_shell_script partial-apply: {fail_count} line(s) failed, {ok_count} line(s) ok"
        );
    }
}

/// Execute a single shell-like command line.
///
/// Supports:
/// - `echo VALUE > /path` — write VALUE followed by newline to /path
///
/// Returns `Err(())` on file-write failure or unsupported command
/// so the caller can count partial-apply failures and emit a
/// summary. The per-line error is logged here; the unit-typed
/// `Err` is only a counter signal.
pub(crate) fn exec_shell_line(line: &str) -> Result<(), ()> {
    if let Some(rest) = line.strip_prefix("echo ")
        && let Some((value, path)) = rest.split_once(" > ")
    {
        let value = value.trim();
        let path = path.trim();
        if let Err(e) = fs::write(path, format!("{value}\n")) {
            tracing::error!(value, path, err = %e, "ktstr-init: echo redirect failed");
            return Err(());
        }
        return Ok(());
    }
    tracing::error!(line, "ktstr-init: unsupported command");
    Err(())
}

#[cfg(test)]
mod tests {
    use super::super::scheduler::ProbeDrain;
    use super::{
        HvcControlDecoder, HvcControlEvent, SchedExitCommitError, SchedExitMonitorConfig,
        SchedExitPublicationGate, SchedExitTerminal, SchedLogDumpAttempt, SchedLogWireCoordinator,
        SchedLogWireLease, drain_probe_for_shutdown, dump_sched_output_before_terminal_with,
        dump_sched_output_with, dump_sched_output_with_wire,
        dump_sched_output_with_wire_and_streams, send_synthetic_sched_output_before_terminal_with,
        spawn_sched_exit_monitor_thread, staged_scheduler_log_paths,
        start_sched_exit_monitor_inner,
    };
    use crate::sync::Latch;
    use crate::vmm::wire::{AttachCancelCause, AttachControlPacket};
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    fn test_pidfd(pid: u32) -> OwnedFd {
        // SAFETY: `pidfd_open` with flags zero returns a fresh descriptor for
        // this test's child. OwnedFd assumes the close-on-drop obligation.
        let raw =
            unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0u32) as libc::c_int };
        assert!(
            raw >= 0,
            "pidfd_open({pid}) failed: {}",
            std::io::Error::last_os_error()
        );
        unsafe { OwnedFd::from_raw_fd(raw) }
    }

    fn start_counted_sched_exit_monitor(
        pid: u32,
        pidfd: OwnedFd,
        publications: Arc<AtomicUsize>,
    ) -> super::SchedExitStop {
        start_sched_exit_monitor_inner(
            SchedExitMonitorConfig {
                pid,
                pidfd,
                log_path: None,
                suppress_sched_log: Arc::new(AtomicBool::new(false)),
                probe_output_done: None,
                publish_sched_exit: Arc::new(move |_exit_code| {
                    publications.fetch_add(1, Ordering::SeqCst);
                }),
                create_wake_writer: true,
            },
            spawn_sched_exit_monitor_thread,
        )
        .expect("start counted scheduler-exit monitor")
    }

    fn wait_for_publications(publications: &AtomicUsize, expected: usize) {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while publications.load(Ordering::SeqCst) != expected {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {expected} publication(s); observed {}",
                publications.load(Ordering::SeqCst)
            );
            std::thread::yield_now();
        }
    }

    fn sched_log_path_attempt_for_test(
        path: &str,
        wire: &mut SchedLogWireLease<'_>,
        send: impl FnMut(&[u8]) -> bool,
    ) -> SchedLogDumpAttempt {
        let complete = dump_sched_output_with_wire(
            path,
            std::time::Instant::now(),
            |_, _| true,
            send,
            Some(wire),
        );
        SchedLogDumpAttempt { complete }
    }

    #[test]
    fn staged_scheduler_logs_sort_by_global_numeric_spawn_sequence() {
        let dir = tempfile::tempdir().expect("create staged scheduler-log tempdir");
        for name in [
            "sched_zeta_11.log",
            "sched_alpha_2.log",
            "sched_name_with_underscores_10.log",
            "sched_omega_1.log",
            "sched_numeric_tail_99_3.log",
            "sched_alpha_bad.log",
            "sched_legacy.log",
            "sched.log",
            "sched_ignored_4.txt",
            "other_0.log",
        ] {
            std::fs::write(dir.path().join(name), name).expect("write staged scheduler log");
        }

        let ordered: Vec<String> = staged_scheduler_log_paths(dir.path())
            .into_iter()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .expect("test filename is UTF-8")
                    .to_owned()
            })
            .collect();
        assert_eq!(
            ordered,
            [
                "sched_omega_1.log",
                "sched_alpha_2.log",
                "sched_numeric_tail_99_3.log",
                "sched_name_with_underscores_10.log",
                "sched_zeta_11.log",
                "sched_alpha_bad.log",
                "sched_legacy.log",
            ],
            "scheduler name and decimal width must not outrank the global spawn sequence"
        );
    }

    #[test]
    fn concurrent_normalized_log_dump_is_framed_once() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("sched.log");
        std::fs::write(&path, b"one scheduler log").expect("write scheduler log");
        let alias = dir.path().join(".").join("sched.log");
        let path = path.to_string_lossy().into_owned();
        let alias = alias.to_string_lossy().into_owned();
        let frames = Arc::new(std::sync::Mutex::new(Vec::<Vec<u8>>::new()));
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel::<()>(0);

        let first_frames = frames.clone();
        let first = std::thread::spawn(move || {
            dump_sched_output_with(
                &path,
                move |_| {
                    started_tx.send(()).expect("publish first dump ownership");
                    release_rx.recv().expect("release first dump");
                    true
                },
                move |chunk| {
                    first_frames
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(chunk.to_vec());
                    true
                },
            );
        });
        started_rx.recv().expect("wait for first dump ownership");

        let second_frames = frames.clone();
        let second = std::thread::spawn(move || {
            dump_sched_output_with(
                &alias,
                |_| panic!("same normalized path entered a second drain"),
                move |chunk| {
                    second_frames
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(chunk.to_vec());
                    true
                },
            );
        });
        release_tx.send(()).expect("complete first dump");
        first.join().expect("join first log dump");
        second.join().expect("join duplicate log dump");

        let frames = frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let expected = vec![
            crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
            b"one scheduler log".to_vec(),
            crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
        ];
        assert_eq!(
            frames.as_slice(),
            expected.as_slice(),
            "one complete marker pair and payload must win for every normalized path"
        );
    }

    #[test]
    fn scheduler_log_dump_retries_a_failed_forwarder_drain() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("drain-retry.log");
        std::fs::write(&path, b"complete after drain").expect("write scheduler log");
        let path = path.to_string_lossy().into_owned();
        let frames = Arc::new(std::sync::Mutex::new(Vec::<Vec<u8>>::new()));

        assert!(
            !dump_sched_output_with(
                &path,
                |_| false,
                |_| { panic!("no frame may precede a complete forwarder drain") }
            ),
            "an incomplete forwarder drain must leave the transaction retryable"
        );

        let retry_frames = frames.clone();
        assert!(dump_sched_output_with(
            &path,
            |_| true,
            move |chunk| {
                retry_frames
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
        ));
        assert_eq!(
            *frames
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![
                crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
                b"complete after drain".to_vec(),
                crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn missing_scheduler_log_is_retryable_before_start() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("appears-later.log");
        let path = path.to_string_lossy().into_owned();
        let calls = AtomicUsize::new(0);
        assert!(!dump_sched_output_with(
            &path,
            |_| true,
            |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                true
            },
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        std::fs::write(&path, b"late bytes").expect("create scheduler log");
        let mut frames = Vec::new();
        assert!(dump_sched_output_with(
            &path,
            |_| true,
            |chunk| {
                frames.push(chunk.to_vec());
                true
            },
        ));
        assert_eq!(
            frames,
            vec![
                crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
                b"late bytes".to_vec(),
                crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn scheduler_log_dump_retries_an_unaccepted_start_marker() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("start-retry.log");
        std::fs::write(&path, b"payload").expect("write scheduler log");
        let path = path.to_string_lossy().into_owned();
        assert!(!dump_sched_output_with(&path, |_| true, |_| false));

        let mut frames = Vec::new();
        assert!(dump_sched_output_with(
            &path,
            |_| true,
            |chunk| {
                frames.push(chunk.to_vec());
                true
            },
        ));
        assert_eq!(
            frames,
            vec![
                crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
                b"payload".to_vec(),
                crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn scheduler_log_dump_resumes_after_first_unsent_payload_frame() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("payload-retry.log");
        let mut payload = vec![0x5a; super::SCHED_LOG_CHUNK_BYTES];
        payload.extend_from_slice(b"tail");
        std::fs::write(&path, &payload).expect("write multi-frame scheduler log");
        let path = path.to_string_lossy().into_owned();
        let accepted = Arc::new(std::sync::Mutex::new(Vec::<Vec<u8>>::new()));
        let attempts = AtomicUsize::new(0);
        let first_accepted = accepted.clone();

        assert!(!dump_sched_output_with(
            &path,
            |_| true,
            |chunk| {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                if attempt == 2 {
                    return false;
                }
                first_accepted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
        ));

        let retry_accepted = accepted.clone();
        assert!(dump_sched_output_with(
            &path,
            |_| true,
            move |chunk| {
                retry_accepted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
        ));
        let accepted = accepted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(accepted.len(), 4);
        assert_eq!(
            accepted[0].as_slice(),
            crate::verifier::SCHED_OUTPUT_START.as_bytes()
        );
        assert_eq!(
            accepted[1].as_slice(),
            &payload[..super::SCHED_LOG_CHUNK_BYTES]
        );
        assert_eq!(accepted[2].as_slice(), b"tail");
        assert_eq!(
            accepted[3].as_slice(),
            crate::verifier::SCHED_OUTPUT_END.as_bytes()
        );
    }

    #[test]
    fn scheduler_log_dump_retries_only_an_unaccepted_end_marker() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("end-retry.log");
        std::fs::write(&path, b"payload").expect("write scheduler log");
        let path = path.to_string_lossy().into_owned();
        let accepted = Arc::new(std::sync::Mutex::new(Vec::<Vec<u8>>::new()));
        let first_accepted = accepted.clone();
        let attempts = AtomicUsize::new(0);

        assert!(!dump_sched_output_with(
            &path,
            |_| true,
            |chunk| {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                if attempt == 2 {
                    return false;
                }
                first_accepted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
        ));
        let retry_accepted = accepted.clone();
        assert!(dump_sched_output_with(
            &path,
            |_| true,
            move |chunk| {
                retry_accepted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
        ));
        assert_eq!(
            *accepted
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![
                crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
                b"payload".to_vec(),
                crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn scheduler_log_dump_preserves_non_utf8_bytes() {
        let dir = tempfile::tempdir().expect("create scheduler-log tempdir");
        let path = dir.path().join("opaque.log");
        let payload = b"\xff\xfeverifier\0\x80";
        std::fs::write(&path, payload).expect("write opaque scheduler log");
        let path = path.to_string_lossy().into_owned();
        let mut frames = Vec::new();
        assert!(dump_sched_output_with(
            &path,
            |_| true,
            |chunk| {
                frames.push(chunk.to_vec());
                true
            },
        ));
        assert_eq!(frames[1].as_slice(), payload);
    }

    #[test]
    fn terminal_stream_replay_is_absent_on_complete_live_and_resumable_after_drop() {
        let dir = tempfile::tempdir().expect("create scheduler-stream replay tempdir");

        let complete_live_path = dir.path().join("complete-live.log");
        std::fs::write(&complete_live_path, b"merged").expect("write complete-live merged log");
        let complete_live_path = complete_live_path.to_string_lossy().into_owned();
        let mut merged_frames = Vec::new();
        let mut stdout_replays = 0usize;
        let mut stderr_replays = 0usize;
        assert!(dump_sched_output_with_wire_and_streams(
            &complete_live_path,
            std::time::Instant::now() + Duration::from_secs(1),
            |_, _| true,
            |chunk| {
                merged_frames.push(chunk.to_vec());
                true
            },
            |_| {
                stdout_replays += 1;
                true
            },
            |_| {
                stderr_replays += 1;
                true
            },
            None,
        ));
        assert_eq!(
            (stdout_replays, stderr_replays),
            (0, 0),
            "fully delivered live streams must generate no terminal replay frames"
        );
        assert_eq!(merged_frames.len(), 3);

        let dropped_path = dir.path().join("dropped-live.log");
        std::fs::write(&dropped_path, b"merged after dropped live frame")
            .expect("write dropped-live merged log");
        let dropped_path = dropped_path.to_string_lossy().into_owned();
        let mut replay = vec![0x6du8; super::SCHED_LOG_CHUNK_BYTES];
        replay.extend_from_slice(b"tail");
        let stdout_replay_path = super::super::scheduler::scheduler_stream_replay_path(
            &dropped_path,
            super::super::scheduler::SchedulerOutputStream::Stdout,
        );
        std::fs::write(&stdout_replay_path, &replay).expect("write authoritative stdout replay");

        let accepted = Arc::new(std::sync::Mutex::new(Vec::<Vec<u8>>::new()));
        let first_accepted = accepted.clone();
        let mut first_frame = 0usize;
        assert!(!dump_sched_output_with_wire_and_streams(
            &dropped_path,
            std::time::Instant::now() + Duration::from_secs(1),
            |_, _| true,
            |_| panic!("merged log must wait for an incomplete stream replay"),
            |chunk| {
                let frame = first_frame;
                first_frame += 1;
                if frame == 1 {
                    return false;
                }
                first_accepted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
            |_| panic!("an absent stderr sidecar must not emit a replay"),
            None,
        ));

        let retry_accepted = accepted.clone();
        let mut retry_merged = Vec::new();
        assert!(dump_sched_output_with_wire_and_streams(
            &dropped_path,
            std::time::Instant::now() + Duration::from_secs(1),
            |_, _| true,
            |chunk| {
                retry_merged.push(chunk.to_vec());
                true
            },
            |chunk| {
                retry_accepted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(chunk.to_vec());
                true
            },
            |_| panic!("an absent stderr sidecar must not emit a replay"),
            None,
        ));

        let accepted = accepted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            accepted.len(),
            2,
            "accepted replay ranges must not be duplicated across retries"
        );
        let mut reconstructed = Vec::new();
        for (index, payload) in accepted.iter().enumerate() {
            let (total_len, offset, bytes) =
                crate::vmm::wire::decode_sched_stream_final_chunk(payload)
                    .expect("decode accepted terminal stream frame");
            assert_eq!(total_len, replay.len() as u64);
            assert_eq!(offset, reconstructed.len() as u64, "frame {index} offset");
            reconstructed.extend_from_slice(bytes);
        }
        assert_eq!(reconstructed, replay);
        assert_eq!(
            retry_merged,
            vec![
                crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
                b"merged after dropped live frame".to_vec(),
                crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn terminal_dump_retries_resume_without_duplicate_accepted_frames() {
        let dir = tempfile::tempdir().expect("create terminal scheduler-log tempdir");
        let path = dir.path().join("terminal-resume.log");
        let mut payload = vec![0x6b; super::SCHED_LOG_CHUNK_BYTES];
        payload.extend_from_slice(b"tail");
        std::fs::write(&path, &payload).expect("write multi-frame terminal scheduler log");
        let path = path.to_string_lossy().into_owned();
        let mut attempts = 0usize;
        let mut accepted = Vec::<Vec<u8>>::new();
        let mut diagnostics = Vec::<String>::new();
        let coordinator = SchedLogWireCoordinator::new();

        assert!(dump_sched_output_before_terminal_with(
            &coordinator,
            "terminal-resume",
            &path,
            |wire, forwarder_deadline| {
                let attempt = attempts;
                attempts += 1;
                let mut frame = 0usize;
                let complete = dump_sched_output_with_wire(
                    &path,
                    forwarder_deadline,
                    |_, _| true,
                    |chunk| {
                        let reject = (attempt == 0 && frame == 2) || (attempt == 1 && frame == 1);
                        frame += 1;
                        if reject {
                            false
                        } else {
                            accepted.push(chunk.to_vec());
                            true
                        }
                    },
                    Some(wire),
                );
                SchedLogDumpAttempt { complete }
            },
            |diagnostic| diagnostics.push(diagnostic.to_owned()),
        ));

        assert_eq!(attempts, 3, "success on the final attempt ends the budget");
        assert!(
            diagnostics.is_empty(),
            "a completed transaction must not emit a terminal diagnostic"
        );
        assert_eq!(
            accepted,
            [
                crate::verifier::SCHED_OUTPUT_START.as_bytes().to_vec(),
                payload[..super::SCHED_LOG_CHUNK_BYTES].to_vec(),
                b"tail".to_vec(),
                crate::verifier::SCHED_OUTPUT_END.as_bytes().to_vec(),
            ],
            "accepted START, payload ranges, and END must each appear exactly once"
        );
    }

    #[test]
    fn terminal_dump_exhaustion_is_bounded_and_diagnosed_once() {
        let path = "/tmp/sched_exhausted_18446744073709551615.log";
        let mut attempts = 0usize;
        let mut forwarder_deadlines = Vec::new();
        let mut diagnostics = Vec::<String>::new();
        let coordinator = SchedLogWireCoordinator::new();

        assert!(!dump_sched_output_before_terminal_with(
            &coordinator,
            "terminal-exhaustion",
            path,
            |_wire, forwarder_deadline| {
                attempts += 1;
                forwarder_deadlines.push(forwarder_deadline);
                SchedLogDumpAttempt { complete: false }
            },
            |diagnostic| diagnostics.push(diagnostic.to_owned()),
        ));

        assert_eq!(
            attempts,
            super::TERMINAL_SCHED_LOG_DUMP_ATTEMPTS,
            "terminal publication must not retry without a fixed bound"
        );
        assert!(
            forwarder_deadlines
                .windows(2)
                .all(|pair| pair[0] == pair[1]),
            "terminal retries must share one absolute drain deadline"
        );
        assert_eq!(
            diagnostics.len(),
            1,
            "only one serial diagnostic may follow an unfinished SchedLog transaction"
        );
        assert!(diagnostics[0].contains(path));
        assert!(
            diagnostics[0].contains(&format!(
                "{} attempts",
                super::TERMINAL_SCHED_LOG_DUMP_ATTEMPTS
            )),
            "diagnostic must name the exhausted attempt budget"
        );
    }

    #[test]
    fn post_start_exhaustion_suppresses_later_sched_log_emission() {
        let coordinator = SchedLogWireCoordinator::new();
        let mut first_diagnostics = Vec::<String>::new();
        assert!(!dump_sched_output_before_terminal_with(
            &coordinator,
            "open-path",
            "/tmp/sched_open_1.log",
            |wire, _forwarder_deadline| {
                wire.mark_open();
                SchedLogDumpAttempt { complete: false }
            },
            |diagnostic| first_diagnostics.push(diagnostic.to_owned()),
        ));
        assert_eq!(first_diagnostics.len(), 1);

        let mut sends = 0usize;
        let mut suppressed_diagnostics = Vec::<String>::new();
        assert!(!send_synthetic_sched_output_before_terminal_with(
            &coordinator,
            "synthetic-after-open-path",
            b"must not be emitted",
            |_| {
                sends += 1;
                true
            },
            |diagnostic| suppressed_diagnostics.push(diagnostic.to_owned()),
        ));
        assert_eq!(
            sends, 0,
            "an abandoned START..END transaction must suppress every later SchedLog byte"
        );
        assert!(
            suppressed_diagnostics.is_empty(),
            "the abandoning owner already emitted the one serial diagnostic"
        );
    }

    #[test]
    fn wire_lease_panic_releases_before_start_and_abandons_after_start() {
        let before_start = SchedLogWireCoordinator::new();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = dump_sched_output_before_terminal_with(
                &before_start,
                "panic-before-start",
                "/tmp/panic-before-start.log",
                |_wire, _forwarder_deadline| panic!("injected pre-START panic"),
                |_| panic!("panic path must not diagnose"),
            );
        }));
        assert!(panic.is_err());
        let mut before_frames = Vec::new();
        assert!(send_synthetic_sched_output_before_terminal_with(
            &before_start,
            "after-pre-start-panic",
            b"still available",
            |chunk| {
                before_frames.push(chunk.to_vec());
                true
            },
            |diagnostic| panic!("idle wire unexpectedly failed: {diagnostic}"),
        ));
        assert_eq!(before_frames.len(), 3);

        let after_start = SchedLogWireCoordinator::new();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = dump_sched_output_before_terminal_with(
                &after_start,
                "panic-after-start",
                "/tmp/panic-after-start.log",
                |wire, _forwarder_deadline| {
                    wire.mark_open();
                    panic!("injected post-START panic");
                },
                |_| panic!("panic path must not diagnose"),
            );
        }));
        assert!(panic.is_err());
        let mut sends = 0usize;
        assert!(!send_synthetic_sched_output_before_terminal_with(
            &after_start,
            "after-post-start-panic",
            b"must be suppressed",
            |_| {
                sends += 1;
                true
            },
            |_| panic!("abandoned owner already owns the diagnostic"),
        ));
        assert_eq!(sends, 0);
    }

    #[test]
    fn wire_coordinator_keeps_two_path_transactions_contiguous() {
        let dir = tempfile::tempdir().expect("create two-path scheduler-log tempdir");
        let first_path = dir.path().join("first.log");
        let second_path = dir.path().join("second.log");
        std::fs::write(&first_path, b"first payload").expect("write first scheduler log");
        std::fs::write(&second_path, b"second payload").expect("write second scheduler log");
        let first_path = first_path.to_string_lossy().into_owned();
        let second_path = second_path.to_string_lossy().into_owned();
        let coordinator = Arc::new(SchedLogWireCoordinator::new());
        let frames = Arc::new(std::sync::Mutex::new(Vec::<(&'static str, Vec<u8>)>::new()));
        let (first_started_tx, first_started_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let (release_first_tx, release_first_rx) = std::sync::mpsc::sync_channel::<()>(0);

        let first_coordinator = coordinator.clone();
        let first_frames = frames.clone();
        let first = std::thread::spawn(move || {
            assert!(dump_sched_output_before_terminal_with(
                &first_coordinator,
                "first-path",
                &first_path,
                |wire, _forwarder_deadline| {
                    sched_log_path_attempt_for_test(&first_path, wire, |chunk| {
                        first_frames
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(("first", chunk.to_vec()));
                        if chunk == crate::verifier::SCHED_OUTPUT_START.as_bytes() {
                            first_started_tx
                                .send(())
                                .expect("publish first path START attempt");
                            release_first_rx
                                .recv()
                                .expect("release first path transaction");
                        }
                        true
                    })
                },
                |diagnostic| panic!("first path unexpectedly failed: {diagnostic}"),
            ));
        });
        first_started_rx
            .recv()
            .expect("wait for first path START attempt");

        let second_coordinator = coordinator.clone();
        let second_frames = frames.clone();
        let second = std::thread::spawn(move || {
            assert!(dump_sched_output_before_terminal_with(
                &second_coordinator,
                "second-path",
                &second_path,
                |wire, _forwarder_deadline| {
                    sched_log_path_attempt_for_test(&second_path, wire, |chunk| {
                        second_frames
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(("second", chunk.to_vec()));
                        true
                    })
                },
                |diagnostic| panic!("second path unexpectedly failed: {diagnostic}"),
            ));
        });
        coordinator.wait_for_waiter();
        assert_eq!(
            frames
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1,
            "the second path must not emit while the first owns START..END"
        );
        release_first_tx
            .send(())
            .expect("release first scheduler-log path");
        first.join().expect("join first scheduler-log path");
        second.join().expect("join second scheduler-log path");

        let labels: Vec<&str> = frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(label, _)| *label)
            .collect();
        assert_eq!(
            labels,
            ["first", "first", "first", "second", "second", "second"]
        );
    }

    #[test]
    fn wire_coordinator_keeps_synthetic_transaction_after_open_path() {
        let dir = tempfile::tempdir().expect("create path-synthetic scheduler-log tempdir");
        let path = dir.path().join("path.log");
        std::fs::write(&path, b"path payload").expect("write scheduler log");
        let path = path.to_string_lossy().into_owned();
        let coordinator = Arc::new(SchedLogWireCoordinator::new());
        let frames = Arc::new(std::sync::Mutex::new(Vec::<(&'static str, Vec<u8>)>::new()));
        let (path_started_tx, path_started_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let (release_path_tx, release_path_rx) = std::sync::mpsc::sync_channel::<()>(0);

        let path_coordinator = coordinator.clone();
        let path_frames = frames.clone();
        let path_thread = std::thread::spawn(move || {
            assert!(dump_sched_output_before_terminal_with(
                &path_coordinator,
                "path",
                &path,
                |wire, _forwarder_deadline| {
                    sched_log_path_attempt_for_test(&path, wire, |chunk| {
                        path_frames
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(("path", chunk.to_vec()));
                        if chunk == crate::verifier::SCHED_OUTPUT_START.as_bytes() {
                            path_started_tx
                                .send(())
                                .expect("publish path START attempt");
                            release_path_rx.recv().expect("release path transaction");
                        }
                        true
                    })
                },
                |diagnostic| panic!("path unexpectedly failed: {diagnostic}"),
            ));
        });
        path_started_rx.recv().expect("wait for path START attempt");

        let synthetic_coordinator = coordinator.clone();
        let synthetic_frames = frames.clone();
        let synthetic = std::thread::spawn(move || {
            assert!(send_synthetic_sched_output_before_terminal_with(
                &synthetic_coordinator,
                "synthetic",
                b"synthetic payload",
                |chunk| {
                    synthetic_frames
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(("synthetic", chunk.to_vec()));
                    true
                },
                |diagnostic| panic!("synthetic transaction unexpectedly failed: {diagnostic}"),
            ));
        });
        coordinator.wait_for_waiter();
        assert_eq!(
            frames
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1,
            "synthetic markers must not enter an open path transaction"
        );
        release_path_tx
            .send(())
            .expect("release scheduler-log path");
        path_thread.join().expect("join scheduler-log path");
        synthetic
            .join()
            .expect("join synthetic scheduler-log transaction");

        let labels: Vec<&str> = frames
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(label, _)| *label)
            .collect();
        assert_eq!(
            labels,
            [
                "path",
                "path",
                "path",
                "synthetic",
                "synthetic",
                "synthetic",
            ]
        );
    }

    #[test]
    fn pending_sched_exit_is_private_and_rejected_at_commit() {
        let gate = SchedExitPublicationGate::new();
        assert!(
            !gate.record_terminal(SchedExitTerminal::Exited),
            "a pending terminal observation must not authorize publication"
        );
        assert!(
            !gate.claim_publication(),
            "pending terminal state must never produce a guest frame"
        );

        // The recorded terminal value outranks the descriptor probe, so an
        // ordinary owned fd is sufficient to prove the typed handoff result.
        let file = std::fs::File::open("/dev/null").expect("open harmless owned fd");
        let fd: OwnedFd = file.into();
        assert_eq!(gate.commit(&fd), Err(SchedExitTerminal::Exited));
        assert!(
            !gate.claim_publication(),
            "a rejected owner cannot publish after commit returns"
        );
    }

    #[test]
    fn committed_sched_exit_has_one_publication_claim() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn live scheduler stand-in");
        let pidfd = test_pidfd(child.id());
        let gate = SchedExitPublicationGate::new();
        gate.commit(&pidfd).expect("commit live scheduler");

        assert!(gate.record_terminal(SchedExitTerminal::Exited));
        assert!(gate.claim_publication(), "first claim must publish");
        assert!(
            !gate.claim_publication(),
            "the gate must reject duplicate publication"
        );
        gate.abort();
        assert!(
            !gate.claim_publication(),
            "an already-published terminal remains exactly once"
        );

        child.kill().expect("kill scheduler stand-in");
        child.wait().expect("reap scheduler stand-in");
    }

    #[test]
    fn abort_suppresses_a_committed_terminal_before_publication() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn live scheduler stand-in");
        let pidfd = test_pidfd(child.id());
        let gate = SchedExitPublicationGate::new();
        gate.commit(&pidfd).expect("commit live scheduler");
        assert!(gate.record_terminal(SchedExitTerminal::Exited));

        gate.abort();
        assert!(
            !gate.claim_publication(),
            "orderly teardown must suppress the pending guest frame"
        );

        child.kill().expect("kill scheduler stand-in");
        child.wait().expect("reap scheduler stand-in");
    }

    #[test]
    fn commit_poll_rejects_exit_not_yet_recorded_by_monitor_thread() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let pidfd = test_pidfd(child.id());
        child.kill().expect("kill scheduler stand-in");
        child.wait().expect("reap scheduler stand-in");

        let gate = SchedExitPublicationGate::new();
        assert_eq!(
            gate.commit(&pidfd),
            Err(SchedExitTerminal::Exited),
            "commit's mutex-held pidfd poll must close the observer scheduling race"
        );
        assert!(!gate.claim_publication());
    }

    #[test]
    fn pending_observer_failure_is_typed_and_never_published() {
        let terminal = SchedExitTerminal::ObserverError("injected pidfd failure".into());
        let gate = SchedExitPublicationGate::new();
        assert!(!gate.record_terminal(terminal.clone()));
        let file = std::fs::File::open("/dev/null").expect("open harmless owned fd");
        let fd: OwnedFd = file.into();
        assert_eq!(gate.commit(&fd), Err(terminal));
        assert!(!gate.claim_publication());
    }

    #[test]
    fn already_readable_pending_monitor_rejects_without_publication() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let pid = child.id();
        let pidfd = test_pidfd(pid);
        child.kill().expect("kill scheduler stand-in");
        child.wait().expect("reap scheduler stand-in");

        let publications = Arc::new(AtomicUsize::new(0));
        let monitor = start_counted_sched_exit_monitor(pid, pidfd, publications.clone());
        assert_eq!(monitor.commit(), Err(SchedExitTerminal::Exited));
        assert!(
            monitor.stop_and_join(),
            "the actual monitor must retain its terminal observation"
        );
        assert_eq!(
            publications.load(Ordering::SeqCst),
            0,
            "an unowned scheduler must not emit"
        );
    }

    #[test]
    fn aborting_a_live_pending_monitor_suppresses_later_exit() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let publications = Arc::new(AtomicUsize::new(0));
        let monitor = start_counted_sched_exit_monitor(
            child.id(),
            test_pidfd(child.id()),
            publications.clone(),
        );
        assert!(
            !monitor.stop_and_join(),
            "aborting a live pending monitor is not an observed exit"
        );

        child.kill().expect("kill scheduler stand-in");
        child.wait().expect("reap scheduler stand-in");
        assert_eq!(
            publications.load(Ordering::SeqCst),
            0,
            "the joined old monitor cannot publish a later kill"
        );
    }

    #[test]
    fn commit_exit_race_is_linearizable_and_publishes_at_most_once() {
        for _ in 0..32 {
            let child = std::process::Command::new("/bin/sleep")
                .arg("30")
                .spawn()
                .expect("spawn scheduler stand-in");
            let pid = child.id();
            let publications = Arc::new(AtomicUsize::new(0));
            let monitor =
                start_counted_sched_exit_monitor(pid, test_pidfd(pid), publications.clone());
            let barrier = Arc::new(Barrier::new(2));
            let killer_barrier = barrier.clone();
            let killer = std::thread::spawn(move || {
                let mut child = child;
                killer_barrier.wait();
                child.kill().expect("kill scheduler stand-in");
                child.wait().expect("reap scheduler stand-in");
            });

            barrier.wait();
            let commit = monitor.commit();
            killer.join().expect("join scheduler killer");
            match commit {
                Ok(()) => {
                    wait_for_publications(&publications, 1);
                    assert!(
                        monitor.stop_and_join(),
                        "committed exit must be observed by the monitor"
                    );
                    assert_eq!(
                        publications.load(Ordering::SeqCst),
                        1,
                        "a committed generation publishes exactly once"
                    );
                }
                Err(SchedExitTerminal::Exited) => {
                    assert!(
                        monitor.stop_and_join(),
                        "rejected exit must remain visible to cleanup"
                    );
                    assert_eq!(
                        publications.load(Ordering::SeqCst),
                        0,
                        "a precommit exit must never publish"
                    );
                }
                Err(error) => panic!("unexpected commit observer error: {error}"),
            }
        }
    }

    #[test]
    fn joined_old_generation_cannot_publish_as_new_generation() {
        let publications = Arc::new(AtomicUsize::new(0));

        let mut old_child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn old scheduler stand-in");
        let old_monitor = start_counted_sched_exit_monitor(
            old_child.id(),
            test_pidfd(old_child.id()),
            publications.clone(),
        );
        old_monitor.commit().expect("commit old scheduler stand-in");
        assert!(!old_monitor.stop_and_join());
        old_child.kill().expect("kill old scheduler stand-in");
        old_child.wait().expect("reap old scheduler stand-in");

        let mut new_child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn new scheduler stand-in");
        let new_monitor = start_counted_sched_exit_monitor(
            new_child.id(),
            test_pidfd(new_child.id()),
            publications.clone(),
        );
        new_monitor.commit().expect("commit new scheduler stand-in");
        new_child.kill().expect("kill new scheduler stand-in");
        new_child.wait().expect("reap new scheduler stand-in");
        wait_for_publications(&publications, 1);
        assert!(new_monitor.stop_and_join());
        assert_eq!(
            publications.load(Ordering::SeqCst),
            1,
            "only the committed new generation may publish"
        );
    }

    #[test]
    fn terminal_racing_commit_waits_for_all_success_markers() {
        let child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let pid = child.id();
        let publications = Arc::new(AtomicUsize::new(0));
        let monitor = start_counted_sched_exit_monitor(pid, test_pidfd(pid), publications.clone());
        let (kill_tx, kill_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let (killed_tx, killed_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let killer = std::thread::spawn(move || {
            let mut child = child;
            kill_rx.recv().expect("receive kill request");
            child.kill().expect("kill scheduler stand-in");
            child.wait().expect("reap scheduler stand-in");
            killed_tx.send(()).expect("publish terminal pidfd edge");
        });
        let mut markers = Vec::new();

        monitor
            .commit_with(|| {
                markers.push("Attached");
                kill_tx.send(()).expect("request scheduler exit");
                killed_rx.recv().expect("wait for terminal pidfd edge");
                assert_eq!(
                    publications.load(Ordering::SeqCst),
                    0,
                    "terminal observation cannot publish while success closure holds Pending"
                );
                markers.push("Settled");
                assert_eq!(
                    publications.load(Ordering::SeqCst),
                    0,
                    "Settled must complete before the monitor is armed"
                );
                Ok::<(), String>(())
            })
            .expect("publish success markers and arm monitor");
        assert_eq!(markers, ["Attached", "Settled"]);
        killer.join().expect("join scheduler killer");
        wait_for_publications(&publications, 1);
        assert!(monitor.stop_and_join());
        assert_eq!(publications.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_success_publication_aborts_racing_terminal() {
        let child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let pid = child.id();
        let publications = Arc::new(AtomicUsize::new(0));
        let monitor = start_counted_sched_exit_monitor(pid, test_pidfd(pid), publications.clone());
        let (kill_tx, kill_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let (killed_tx, killed_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let killer = std::thread::spawn(move || {
            let mut child = child;
            kill_rx.recv().expect("receive kill request");
            child.kill().expect("kill scheduler stand-in");
            child.wait().expect("reap scheduler stand-in");
            killed_tx.send(()).expect("publish terminal pidfd edge");
        });

        let result = monitor.commit_with(|| {
            kill_tx.send(()).expect("request scheduler exit");
            killed_rx.recv().expect("wait for terminal pidfd edge");
            Err::<(), _>("Settled publication failed")
        });
        assert_eq!(
            result,
            Err(SchedExitCommitError::Publication(
                "Settled publication failed"
            ))
        );
        killer.join().expect("join scheduler killer");
        assert!(
            monitor.stop_and_join(),
            "the terminal edge remains visible after publication failure"
        );
        assert_eq!(
            publications.load(Ordering::SeqCst),
            0,
            "failed success publication must permanently suppress the exit frame"
        );
    }

    #[test]
    fn panicking_success_publication_aborts_without_arming() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn scheduler stand-in");
        let pidfd = test_pidfd(child.id());
        let gate = SchedExitPublicationGate::new();
        let result = gate.commit_with(&pidfd, || -> Result<(), String> {
            panic!("injected Settled publication panic")
        });
        assert!(
            matches!(
                result,
                Err(SchedExitCommitError::PublicationPanicked(ref detail))
                    if detail.contains("injected Settled publication panic")
            ),
            "unexpected panic conversion: {result:?}"
        );
        assert!(
            !gate.record_terminal(SchedExitTerminal::Exited),
            "panicking success publication must leave the gate aborted"
        );
        assert!(!gate.claim_publication());
        child.kill().expect("kill scheduler stand-in");
        child.wait().expect("reap scheduler stand-in");
    }

    #[test]
    fn hvc_control_decoder_preserves_split_and_coalesced_typed_packets() {
        let started = AttachControlPacket::StartedAck {
            generation: 0x1234_abcd_9876_ef01,
        };
        let finished = AttachControlPacket::FinishedAck {
            generation: 0x1234_abcd_9876_ef01,
        };
        let cancelled = AttachControlPacket::Cancel {
            generation: 0xfeed_face_cafe_beef,
            cause: AttachCancelCause::ServiceBudget,
        };
        let started_packet = crate::vmm::wire::encode_attach_control(started);
        let finished_packet = crate::vmm::wire::encode_attach_control(finished);
        let cancelled_packet = crate::vmm::wire::encode_attach_control(cancelled);

        for (packet, expected) in [
            (started_packet, started),
            (finished_packet, finished),
            (cancelled_packet, cancelled),
        ] {
            for split in 1..packet.len() {
                let mut decoder = HvcControlDecoder::default();
                assert!(
                    decoder.push(&packet[..split]).is_empty(),
                    "split {split} emitted a partial {expected:?}"
                );
                assert_eq!(
                    decoder.push(&packet[split..]),
                    vec![HvcControlEvent::AttachControl(expected)],
                    "split {split} lost or changed {expected:?}"
                );
            }
        }

        let mut decoder = HvcControlDecoder::default();
        let mut coalesced = started_packet.to_vec();
        coalesced.extend_from_slice(&finished_packet);
        coalesced.extend_from_slice(&cancelled_packet);
        coalesced.push(crate::vmm::virtio_console::SIGNAL_VC_DUMP);
        assert_eq!(
            decoder.push(&coalesced),
            vec![
                HvcControlEvent::AttachControl(started),
                HvcControlEvent::AttachControl(finished),
                HvcControlEvent::AttachControl(cancelled),
                HvcControlEvent::Signal(crate::vmm::virtio_console::SIGNAL_VC_DUMP),
            ]
        );
    }

    #[test]
    fn malformed_attach_control_reprocesses_a_fresh_typed_prefix() {
        let finished = AttachControlPacket::FinishedAck {
            generation: 0x1122_3344_5566_7788,
        };
        let finished_packet = crate::vmm::wire::encode_attach_control(finished);
        let mut decoder = HvcControlDecoder::default();

        let mut bytes = vec![crate::vmm::wire::SIGNAL_ATTACH_STARTED_ACK, b'0', b'1'];
        bytes.extend_from_slice(&finished_packet);
        assert_eq!(
            decoder.push(&bytes),
            vec![HvcControlEvent::AttachControl(finished)]
        );

        for signal in [
            crate::vmm::virtio_console::SIGNAL_VC_DUMP,
            crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE,
            crate::vmm::virtio_console::SIGNAL_ACCESSOR_READY,
            crate::vmm::virtio_console::SIGNAL_PROBE_DUMP_READY,
            crate::vmm::virtio_console::SIGNAL_PERIODIC_READY,
            crate::vmm::virtio_console::SIGNAL_WPROF_ARTIFACTS_RECEIVED,
            crate::vmm::virtio_console::SIGNAL_VC_SHUTDOWN,
        ] {
            let mut decoder = HvcControlDecoder::default();
            assert!(
                decoder
                    .push(&[crate::vmm::wire::SIGNAL_ATTACH_STARTED_ACK, b'0'])
                    .is_empty()
            );
            assert_eq!(
                decoder.push(&[signal]),
                vec![HvcControlEvent::Signal(signal)],
                "malformed packet swallowed legacy signal {signal:#x}"
            );
        }
    }

    #[test]
    fn hvc_control_decoder_rejects_zero_and_uppercase_generations() {
        let mut decoder = HvcControlDecoder::default();
        let mut zero_packet = [b'0'; crate::vmm::wire::ATTACH_CONTROL_PACKET_SIZE];
        zero_packet[0] = crate::vmm::wire::SIGNAL_ATTACH_STARTED_ACK;
        assert!(decoder.push(&zero_packet).is_empty());

        assert!(
            decoder
                .push(&[crate::vmm::wire::SIGNAL_ATTACH_FINISHED_ACK, b'0', b'1',])
                .is_empty()
        );
        assert_eq!(decoder.push(b"A"), vec![HvcControlEvent::Signal(b'A')]);
    }

    #[test]
    fn hvc_control_decoder_preserves_coalesced_legacy_signals() {
        let signals = [
            crate::vmm::virtio_console::SIGNAL_VC_DUMP,
            crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE,
            crate::vmm::virtio_console::SIGNAL_ACCESSOR_READY,
            crate::vmm::virtio_console::SIGNAL_PROBE_DUMP_READY,
            crate::vmm::virtio_console::SIGNAL_PERIODIC_READY,
            crate::vmm::virtio_console::SIGNAL_WPROF_ARTIFACTS_RECEIVED,
            crate::vmm::virtio_console::SIGNAL_VC_SHUTDOWN,
        ];
        let mut decoder = HvcControlDecoder::default();
        assert_eq!(
            decoder.push(&signals),
            signals
                .into_iter()
                .map(HvcControlEvent::Signal)
                .collect::<Vec<_>>()
        );
    }

    /// `drain_probe_for_shutdown` is the bounded probe drain the
    /// graceful-shutdown (watchdog soft-shutdown) handler runs so captured
    /// crash-arg probe output reaches the host (emitted over the virtio bulk
    /// port) before the guest reboots — the only drain site for a repro
    /// scenario that hangs inside the test function.
    /// Host-runnable (no VM): drives a synthetic [`ProbeDrain`] so CI without
    /// `/dev/kvm` actively proves the drain logic instead of relying on the
    /// VM-gated repro e2e (which SKIPs there; see the "skipping e2e masks bugs"
    /// lesson). Pins: (1) no drain attached → no-op `true`; (2) output already
    /// emitted → returns `true` immediately AND requested stop; (3) probe wedged
    /// (`output_done` never set) → returns `false` within the bound but still
    /// requested stop (the probe thread is told to wind down regardless).
    #[test]
    fn drain_probe_for_shutdown_requests_stop_and_is_bounded() {
        // (1) no drain → no-op true.
        assert!(drain_probe_for_shutdown(None, Duration::from_secs(5)));

        // (2) output already emitted → immediate true + stop requested.
        let stop = Arc::new(AtomicBool::new(false));
        let output_done = Arc::new(Latch::new());
        output_done.set();
        let pd = ProbeDrain {
            stop: stop.clone(),
            output_done,
        };
        let start = std::time::Instant::now();
        assert!(drain_probe_for_shutdown(Some(&pd), Duration::from_secs(5)));
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "an already-set output_done returns immediately"
        );
        assert!(
            stop.load(Ordering::Acquire),
            "drain requested the probe thread to stop"
        );

        // (3) wedged probe (output_done never set) → false within the bound,
        //     stop still requested. A short bound keeps the test fast; the wait
        //     is an evented condvar wait_timeout, not a poll-sleep.
        let stop = Arc::new(AtomicBool::new(false));
        let pd = ProbeDrain {
            stop: stop.clone(),
            output_done: Arc::new(Latch::new()),
        };
        let start = std::time::Instant::now();
        assert!(!drain_probe_for_shutdown(
            Some(&pd),
            Duration::from_millis(50)
        ));
        assert!(
            start.elapsed() >= Duration::from_millis(40),
            "the wait actually blocked for the bound rather than returning early: {:?}",
            start.elapsed()
        );
        assert!(
            stop.load(Ordering::Acquire),
            "drain requests stop even when the probe is wedged"
        );
    }
}
