//! Guest-only typed senders for the host-bound bulk TLV stream.
//!
//! Every function in this module is callable ONLY from inside a
//! ktstr guest VM. Host-context invocations log a `tracing::warn!`
//! and no-op.
//!
//! Each function frames its payload with the corresponding
//! [`super::wire::MsgType`] so call sites do not pass raw u32 ids.
//! The frame format is the [`super::wire::ShmMessage`] header +
//! payload described on the [`super::wire`] module doc.
//!
//! # Backpressure
//!
//! The bulk channel uses the kernel virtio_console TX path: a full
//! virtqueue blocks the writer until the host's `add_used` rate
//! catches up. Callers that cannot block (panic hook, signal
//! handlers, anything called from a critical section) MUST write
//! directly to COM2 (`/dev/ttyS1`) — the 16550 UART PIO path
//! commits synchronously inside `KVM_RUN` and never blocks the
//! guest on host backpressure. The panic hook in
//! [`super::rust_init`] follows this discipline.

use crate::sync::MutexExt;
use crate::vmm::wire::{
    AttachAttemptEvent, KERNEL_OP_REPLY_MAX, KernelOpReplyPayload, KernelOpRequestPayload,
    KernelOpRequestResult, LifecyclePhase, MSG_TYPE_KERNEL_OP_REPLY, MSG_TYPE_SNAPSHOT_REPLY,
    MsgType, PORT1_NAME, SNAPSHOT_REASON_MAX, SNAPSHOT_STATUS_ERR, SNAPSHOT_STATUS_OK,
    SNAPSHOT_TAG_MAX, ShmMessage, SnapshotReplyPayload, SnapshotRequestPayload,
    SnapshotRequestResult,
};
use zerocopy::{FromBytes, IntoBytes};

/// Mutex serializing guest-side bulk-port writes. Every guest writer
/// (`write_msg`) takes this lock before submitting bytes to
/// the port advertised as [`PORT1_NAME`], so the in-stream order of bytes stays
/// `[header][payload]` regardless of which producer (step executor,
/// sched-exit-mon, profraw flusher) emitted the frame.
static GUEST_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Number of scheduler-attach transactions which currently reserve the bulk
/// stream for their generation-tagged control boundaries.
///
/// Scheduler stdout/stderr already lands in the merged log file before the
/// live-copy send is attempted. Dropping those best-effort live copies while
/// an attach transaction is open keeps a noisy scheduler from sitting ahead
/// of Started/Finished/Settled on the one virtio-console TX stream; the
/// terminal log transaction still ships the authoritative bytes afterwards.
static BULK_LIFECYCLE_PRIORITY: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// RAII reservation for a scheduler-attach control transaction.
pub(crate) struct BulkLifecyclePriorityGuard;

impl Drop for BulkLifecyclePriorityGuard {
    fn drop(&mut self) {
        let previous = BULK_LIFECYCLE_PRIORITY.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        debug_assert!(previous > 0, "bulk lifecycle priority underflow");
    }
}

/// Give scheduler-attach boundaries priority over best-effort bulk traffic.
pub(crate) fn reserve_bulk_lifecycle_priority() -> BulkLifecyclePriorityGuard {
    BULK_LIFECYCLE_PRIORITY.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    BulkLifecyclePriorityGuard
}

// ---------------------------------------------------------------------------
// is_guest detection
// ---------------------------------------------------------------------------

/// Detect whether the current process is running inside a ktstr guest
/// VM, by looking for the `KTSTR_GUEST=1` token on `/proc/cmdline`.
///
/// PID is NOT a reliable signal: the guest test code runs as forked
/// children of init (PID 1), not as PID 1 itself. The guest kernel
/// command line, populated by the host VMM, is the unique fingerprint.
///
/// The result is cached in a `OnceLock` — `/proc/cmdline` is read at
/// most once per process. False on the host (no cmdline match) and
/// false on any non-Linux platform that lacks `/proc/cmdline` (read
/// fails).
///
/// In test builds, the `IS_GUEST_TEST_OVERRIDE` thread-local takes
/// precedence over the `OnceLock`-cached natural detection; the
/// `OnceLock` is consulted only when no override is set on the
/// calling thread.
pub fn is_guest() -> bool {
    #[cfg(test)]
    {
        // Test-only override: tests run on the host but need to
        // exercise the guest-only path (write_msg). The override is
        // thread-local so parallel tests don't fight over it.
        if let Some(v) = IS_GUEST_TEST_OVERRIDE.with(|c| c.get()) {
            return v;
        }
    }
    static IS_GUEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *IS_GUEST.get_or_init(|| {
        std::fs::read_to_string("/proc/cmdline")
            .ok()
            .is_some_and(|c| c.split_whitespace().any(|tok| tok == "KTSTR_GUEST=1"))
    })
}

// Test-only thread-local override for `is_guest`. `None` means
// "consult /proc/cmdline"; `Some(b)` pins the result for the
// current thread. Per-thread so parallel tests cannot interfere.
#[cfg(test)]
thread_local! {
    static IS_GUEST_TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// RAII guard that overrides [`is_guest`] for the duration of its
/// scope on the current thread, and restores the previous value on
/// drop. Avoids leaking override state across tests sharing a thread
/// (e.g. via test-runner thread pools).
///
/// `pub(crate)` so other test modules in the crate can use the
/// fixture when they need to exercise guest-only paths.
#[cfg(test)]
pub(crate) struct IsGuestOverrideGuard {
    prev: Option<bool>,
}

#[cfg(test)]
impl IsGuestOverrideGuard {
    pub(crate) fn new(value: bool) -> Self {
        let prev = IS_GUEST_TEST_OVERRIDE.with(|c| c.replace(Some(value)));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for IsGuestOverrideGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        IS_GUEST_TEST_OVERRIDE.with(|c| c.set(prev));
    }
}

// ---------------------------------------------------------------------------
// Bulk-port writer (guest → host TLV)
// ---------------------------------------------------------------------------

/// Reject a call to a guest-only entry point when invoked from host
/// context. Returns `true` if the caller may proceed (we're inside a
/// guest VM); `false` after emitting a `tracing::warn!` that names the
/// caller and the message type, so a host-side caller surfaces in the
/// log instead of silently no-op'ing.
fn assert_guest_context(fn_name: &str, msg_type: u32) -> bool {
    if !is_guest() {
        tracing::warn!(
            msg_type = msg_type,
            "guest_comms::{fn_name} called from host context"
        );
        return false;
    }
    true
}

/// Sysfs class populated by the kernel virtio-console driver. Device
/// numbering (`vport0p1`, `vport1p1`, ...) depends on virtio device
/// probe order, so ktstr resolves ports by the stable names advertised
/// by the host instead of assuming a particular numeric prefix.
pub(crate) const VIRTIO_PORT_CLASS_DIR: &str = "/sys/class/virtio-ports";
const SYS_CLASS_DIR: &str = "/sys/class";

/// Map a host-advertised virtio-console port name to its guest device
/// node. `class_dir` and `dev_dir` are parameters so the filesystem
/// mapping can be covered without a booted VM.
fn named_virtio_port_path_in(
    class_dir: &std::path::Path,
    dev_dir: &std::path::Path,
    port_name: &str,
) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(class_dir).ok()?;
    for entry in entries.flatten() {
        let Ok(name) = std::fs::read_to_string(entry.path().join("name")) else {
            continue;
        };
        if name.trim() == port_name {
            return Some(dev_dir.join(entry.file_name()));
        }
    }
    None
}

/// Resolve a host-advertised virtio-console port name to `/dev`.
///
/// The host publishes [`PORT1_NAME`] / [`crate::vmm::wire::PORT2_NAME`]
/// through `PORT_NAME` control messages. Linux exposes those names at
/// `/sys/class/virtio-ports/vportNpM/name`; the directory basename is
/// the corresponding `/dev/vportNpM` node.
pub(crate) fn named_virtio_port_path(port_name: &str) -> Option<std::path::PathBuf> {
    named_virtio_port_path_in(
        std::path::Path::new(VIRTIO_PORT_CLASS_DIR),
        std::path::Path::new("/dev"),
        port_name,
    )
}

/// Existing directory whose inotify events can lead to a named port.
///
/// Before `virtio_console` registers its class, watch `/sys/class` for
/// the class directory itself. Once registered, watch the class for
/// port devices. This keeps early boot event-driven without assuming
/// the class already existed when init armed its first wait.
pub(crate) fn virtio_port_watch_dir() -> &'static std::path::Path {
    let class = std::path::Path::new(VIRTIO_PORT_CLASS_DIR);
    if class.is_dir() {
        class
    } else {
        std::path::Path::new(SYS_CLASS_DIR)
    }
}

/// Resolve the guest-to-host bulk channel by its stable advertised name.
pub(crate) fn bulk_port_path() -> Option<std::path::PathBuf> {
    named_virtio_port_path(PORT1_NAME)
}

/// Cached bulk-port writer. Opened lazily on the first
/// successful `write_to_bulk_port` call after the kernel's
/// virtio_console driver creates the device node (post multiport
/// handshake). `OnceLock<Option<...>>` so repeated open failures
/// (port not yet ready) do not pin the slot to None permanently —
/// instead we re-attempt until `try_open_bulk_port` succeeds, then
/// cache the file handle for the rest of the process.
static BULK_PORT_FD: std::sync::OnceLock<std::sync::Mutex<Option<std::fs::File>>> =
    std::sync::OnceLock::new();

/// Test-only counter of [`write_to_bulk_port`] entries. Incremented
/// on the FIRST line of `write_to_bulk_port`, which the `is_guest()`
/// gate in [`write_msg`] reaches only inside a guest context. A
/// host-context sender call short-circuits in `write_msg` and never
/// touches this counter — so a test that snapshots the count, calls a
/// void-returning sender from host context, and re-reads the count
/// can assert the write was SUPPRESSED (count unchanged), not merely
/// that the call did not panic.
#[cfg(test)]
pub(crate) static BULK_PORT_WRITE_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Try to resolve and open the named bulk port. Returns None while
/// the kernel has not yet received both PORT_ADD and PORT_NAME.
///
/// Open mode: read+write and nonblocking. O_RDWR is required because the
/// kernel's `port_fops_open` (drivers/char/virtio_console.c) sets
/// `guest_connected = true` on the first open and returns EBUSY on
/// any subsequent open of the same port. A write-only open would
/// block a later read-only open needed by `request_snapshot`'s
/// reply reader. The port-2 stats relay already uses O_RDWR
/// (rust_init/relay.rs `start_sched_stats_relay`). O_NONBLOCK lets the
/// lifecycle path make a bounded try-write instead of parking behind a full
/// virtqueue. Ordinary senders explicitly poll POLLOUT and retain their prior
/// blocking semantics.
fn try_open_bulk_port() -> Option<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    let path = bulk_port_path()?;
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .ok()
}

/// Clone the process's one O_RDWR bulk-port open description without holding
/// the cache mutex for the lifetime of a blocking reply read. `try_clone`
/// duplicates the fd and therefore does not trip virtio-console's one-open
/// EBUSY rule; writers can continue using the cached sibling while an RPC
/// waits for host RX.
fn clone_or_open_bulk_port() -> std::io::Result<std::fs::File> {
    let slot = BULK_PORT_FD.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = slot.lock_unpoisoned();
    if guard.is_none() {
        *guard = try_open_bulk_port();
    }
    guard
        .as_ref()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "named virtio-console port {PORT1_NAME:?} not yet open \
                     (multiport handshake still in flight)"
                ),
            )
        })?
        .try_clone()
}

fn invalidate_bulk_port() {
    let slot = BULK_PORT_FD.get_or_init(|| std::sync::Mutex::new(None));
    *slot.lock_unpoisoned() = None;
}

/// Write a TLV-framed message to the host through the bulk channel
/// (the virtio-console port named [`PORT1_NAME`]). The frame format is
/// 16-byte [`ShmMessage`] header + `payload.len()` bytes; the host
/// parses the same byte stream via [`super::host_comms::parse_tlv_stream`].
///
/// Returns `true` when the frame was fully written, `false` when the
/// bulk port is not yet open (multiport handshake still in flight),
/// the writev failed, or the call originated from host context. The
/// existing fire-and-forget callers (Exit, TestResult, PayloadMetrics,
/// Profraw, Stimulus, SchedExit, ScenarioStart,
/// ScenarioEnd, SnapshotRequest) discard the return at statement
/// position — only the [`send_sys_rdy`]/[`send_kern_addrs`] retry
/// loop in `vmm::rust_init::send_sys_rdy_with_retry` observes it.
///
/// Backpressure: the kernel's virtio_console TX path (`hvc_push` /
/// `port_fops_write`) blocks the writer until the host's
/// `add_used` rate catches up. There is no drop path; callers that
/// cannot block (panic hook, signal handlers, anything called from
/// a critical section) MUST write directly to COM2 (`/dev/ttyS1`).
///
/// `assert_guest_context` rejects host-context invocations with a
/// `tracing::warn` so a host-side caller surfaces in the log instead
/// of silently no-op'ing.
fn write_msg(msg_type: u32, payload: &[u8]) -> bool {
    if !assert_guest_context("write_msg", msg_type) {
        return false;
    }
    if lifecycle_priority_drops(msg_type) {
        return false;
    }
    let _guard = GUEST_WRITE_LOCK.lock_unpoisoned();
    // Recheck after taking the mutex: an attach reservation may have arrived
    // while this ordinary sender was queued behind a previous writer.
    if lifecycle_priority_drops(msg_type) {
        return false;
    }
    write_to_bulk_port(msg_type, payload, BulkWriteMode::Blocking)
}

fn lifecycle_priority_drops(msg_type: u32) -> bool {
    BULK_LIFECYCLE_PRIORITY.load(std::sync::atomic::Ordering::Acquire) != 0
        && matches!(
            MsgType::from_wire(msg_type),
            Some(MsgType::SchedStdout | MsgType::SchedStderr)
        )
}

/// Nonblocking control-frame attempt. Contention, port backpressure, or a
/// transient fd error returns `false` immediately; the caller owns retry
/// cadence. A partial prefix is safe because the host assembler resynchronizes
/// at the next complete CRC-valid known frame.
fn try_write_msg(msg_type: u32, payload: &[u8]) -> bool {
    if !assert_guest_context("try_write_msg", msg_type) {
        return false;
    }
    let Ok(_guard) = GUEST_WRITE_LOCK.try_lock() else {
        return false;
    };
    write_to_bulk_port(msg_type, payload, BulkWriteMode::Immediate)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BulkWriteMode {
    Blocking,
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BulkWriteProgress {
    complete: bool,
    written: usize,
    failure: Option<BulkWriteFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BulkWriteFailure {
    Backpressured,
    Fatal,
}

/// Drive one contiguous frame through an injected writer.
///
/// The helper is deliberately independent of the virtio fd so tests can force
/// positive partial writes followed by EAGAIN/EIO deterministically. Blocking
/// mode invokes `wait_writable` on EAGAIN; immediate mode returns at once.
fn write_frame_with(
    frame: &[u8],
    mode: BulkWriteMode,
    mut write: impl FnMut(&[u8]) -> std::io::Result<usize>,
    mut wait_writable: impl FnMut() -> std::io::Result<()>,
) -> BulkWriteProgress {
    let mut written = 0usize;
    while written < frame.len() {
        match write(&frame[written..]) {
            Ok(0) => {
                return BulkWriteProgress {
                    complete: false,
                    written,
                    failure: Some(BulkWriteFailure::Fatal),
                };
            }
            Ok(n) => {
                let remaining = frame.len() - written;
                if n > remaining {
                    return BulkWriteProgress {
                        complete: false,
                        written,
                        failure: Some(BulkWriteFailure::Fatal),
                    };
                }
                written += n;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if mode == BulkWriteMode::Immediate || wait_writable().is_err() {
                    return BulkWriteProgress {
                        complete: false,
                        written,
                        failure: Some(if mode == BulkWriteMode::Immediate {
                            BulkWriteFailure::Backpressured
                        } else {
                            BulkWriteFailure::Fatal
                        }),
                    };
                }
            }
            Err(_) => {
                return BulkWriteProgress {
                    complete: false,
                    written,
                    failure: Some(BulkWriteFailure::Fatal),
                };
            }
        }
    }
    BulkWriteProgress {
        complete: true,
        written,
        failure: None,
    }
}

/// Try to write a TLV-framed message to the named bulk port. Returns
/// true when the message was fully written, false when the bulk
/// port is not yet available or the write failed.
///
/// Lazy-open semantics: the multiport handshake completes
/// asynchronously during kernel virtio_console init, so the device
/// node may appear any time after the first `write_msg` call. We
/// retry the open on every call until it succeeds; once cached,
/// subsequent writes go through the cached `File`.
///
/// Submission shape: header and payload are concatenated into one immutable
/// frame buffer before the first syscall. Every positive partial write resumes
/// at that buffer's exact unwritten suffix. The kernel virtio_console driver
/// caps each write at 32 KiB (`port_fops_write` does
/// `count = min(32*1024, count)`), so a larger frame necessarily spans
/// multiple syscalls. If a later syscall fails, a lifecycle retry starts a
/// complete fresh frame and the host's CRC-valid-known-FourCC resynchronizer
/// discards the abandoned prefix.
fn write_to_bulk_port(msg_type: u32, payload: &[u8], mode: BulkWriteMode) -> bool {
    // Test-only: record that the guest-only write path was entered.
    // Reached only after `write_msg`'s `is_guest()` gate passes, so a
    // host-context sender call never bumps this. See
    // [`BULK_PORT_WRITE_ATTEMPTS`].
    #[cfg(test)]
    BULK_PORT_WRITE_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let slot = BULK_PORT_FD.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = match mode {
        BulkWriteMode::Blocking => slot.lock_unpoisoned(),
        BulkWriteMode::Immediate => {
            let Ok(guard) = slot.try_lock() else {
                return false;
            };
            guard
        }
    };
    if guard.is_none() {
        match try_open_bulk_port() {
            Some(f) => *guard = Some(f),
            None => return false,
        }
    }
    let f = guard.as_mut().expect("bulk port handle just installed");
    let Ok(length_u32) = u32::try_from(payload.len()) else {
        tracing::warn!(
            len = payload.len(),
            msg_type,
            "write_to_bulk_port: payload exceeds u32::MAX; dropping"
        );
        return false;
    };
    let msg = ShmMessage {
        msg_type,
        length: length_u32,
        crc32: crc32fast::hash(payload),
        _pad: 0,
    };
    let header_bytes = msg.as_bytes();
    let mut frame = Vec::with_capacity(header_bytes.len() + payload.len());
    frame.extend_from_slice(header_bytes);
    frame.extend_from_slice(payload);
    let fd = std::os::unix::io::AsRawFd::as_raw_fd(f);
    let progress = write_frame_with(
        &frame,
        mode,
        |remaining| {
            // SAFETY: `remaining` is a live byte slice and `fd` remains owned
            // by `guard` for the entire helper call.
            let result = unsafe {
                libc::write(
                    fd,
                    remaining.as_ptr().cast::<libc::c_void>(),
                    remaining.len(),
                )
            };
            if result < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(result as usize)
            }
        },
        || wait_bulk_port_writable(fd),
    );
    if progress.complete {
        return true;
    }

    // A hard error may arrive after a positive partial write. The next
    // lifecycle retry deliberately starts a fresh complete frame; the host's
    // CRC-validated resynchronizer discards this abandoned prefix. Preserve a
    // live fd across ordinary nonblocking backpressure, but invalidate it for
    // zero writes and terminal I/O failures so a later attempt can reopen.
    if progress.failure == Some(BulkWriteFailure::Fatal) {
        *guard = None;
    }
    if progress.failure == Some(BulkWriteFailure::Backpressured) {
        return false;
    }
    tracing::warn!(
        msg_type,
        len = payload.len(),
        written = progress.written,
        mode = ?mode,
        "write_to_bulk_port: frame was not completed"
    );
    false
}

fn wait_bulk_port_writable(fd: libc::c_int) -> std::io::Result<()> {
    loop {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        // SAFETY: one borrowed live fd; an infinite poll preserves the former
        // blocking semantics for ordinary traffic without making the fd itself
        // blocking for lifecycle try-writes.
        let result = unsafe { libc::poll(&mut pfd, 1, -1) };
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if pfd.revents & libc::POLLOUT != 0 {
            return Ok(());
        }
        if pfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(std::io::Error::other(format!(
                "bulk port writable poll returned terminal events {:#x}",
                pfd.revents
            )));
        }
    }
}

// ---------------------------------------------------------------------------
// Typed senders
// ---------------------------------------------------------------------------

/// Send the guest exit code to the host. Payload: 4-byte LE i32.
///
/// Frames the exit code with [`MsgType::Exit`] and routes through
/// the bulk port. The host's `collect_results` reads the latest
/// `Exit` entry to override the BSP run-loop sentinel.
pub fn send_exit(code: i32) {
    write_msg(MsgType::Exit.wire_value(), &code.to_le_bytes());
}

/// What to emit for an `AssertResult` over a `max`-byte bulk frame — the pure
/// graceful-degradation decision, factored out of [`send_test_result`] (which
/// owns the `write_msg` side effect) so the size-class branch selection is
/// unit-testable without a guest wire.
#[derive(Debug)]
enum TestResultWire {
    /// The encoded result fits the frame as-is.
    Raw(Vec<u8>),
    /// Over-frame: `dropped` per_cgroup raw samples were dropped (the encoded
    /// `bytes` already carry the info note); the sample-free encoding now fits.
    Stripped { bytes: Vec<u8>, dropped: usize },
    /// Even the sample-free verdict overruns the frame; emit a truncated FAIL.
    /// `offending` is the size of the payload that actually overran — the
    /// sample-free size when a strip happened, else the original (incl. the rare
    /// re-serialize-error sub-case, which keeps the pre-strip size as best-effort).
    Truncated { offending: usize },
}

/// Classify how to emit `result` over a `max`-byte bulk frame. Pure apart from
/// `postcard` encoding; returns `None` only on a postcard encode error (the
/// caller logs and drops).
///
/// This is the first path to put RAW per-cgroup sample vectors on the bulk port,
/// and a many-cgroup × many-step scenario accumulates non-merging carriers that
/// can overrun the frame even with each carrier reservoir-capped. On overflow
/// the three sample pools (wake_latencies_ns / run_delays_ns / off_cpu_pcts) are
/// dropped — so the wake p99/median/CV and mean/worst run-delay re-pools DEGRADE
/// to the cross-cgroup max of the surviving per-cgroup `CgroupStats` reductions
/// (a worst-cgroup proxy, via `populate_run_distribution_metrics`), not vanish;
/// off-CPU% has no run-level re-pool consumer so only its per-phase render
/// is lost. The verdict, outcomes, and all scalar/counter telemetry are
/// PRESERVED (never a PASS→FAIL flip). The truncated FAIL is reached only if the
/// sample-free verdict ALONE overruns.
fn classify_test_result(
    result: &crate::assert::AssertResult,
    max: usize,
) -> Option<TestResultWire> {
    let bytes = postcard::to_stdvec(result).ok()?;
    if bytes.len() <= max {
        return Some(TestResultWire::Raw(bytes));
    }
    let mut stripped = result.clone();
    let dropped = stripped.strip_phase_cgroup_samples();
    // Sample-free size once a strip happened — the payload that actually overruns
    // on the Truncated path. On a re-serialize error (practically unreachable:
    // stripping only shrinks) it stays None and the fallback reports the pre-strip
    // size as best-effort.
    let mut sample_free_size: Option<usize> = None;
    if dropped > 0 {
        stripped.note(format!(
            "per_cgroup raw samples ({dropped}) dropped: AssertResult postcard size {} \
             exceeded bulk port limit {max}; verdict and reduced telemetry preserved",
            bytes.len(),
        ));
        if let Ok(small) = postcard::to_stdvec(&stripped) {
            sample_free_size = Some(small.len());
            if small.len() <= max {
                return Some(TestResultWire::Stripped {
                    bytes: small,
                    dropped,
                });
            }
        }
    }
    Some(TestResultWire::Truncated {
        offending: sample_free_size.unwrap_or(bytes.len()),
    })
}

/// Send a test result to the host. Payload: postcard-encoded
/// [`crate::assert::AssertResult`].
///
/// Frames with [`MsgType::TestResult`]. Guest and host both use
/// `postcard` so layout never diverges; the host's
/// `crate::test_support::output::parse_assert_result_from_drain`
/// decodes with the same library.
///
/// Required: `result` MUST round-trip through postcard without
/// erroring — every field is owned `String` / `bool` / nested
/// `serde::Serialize` derives, so the only failure path is OOM
/// during the `Vec<u8>` allocation, which the surrounding eprintln
/// guards against silent loss.
pub fn send_test_result(result: &crate::assert::AssertResult) {
    let max = crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD as usize;
    match classify_test_result(result, max) {
        Some(TestResultWire::Raw(bytes)) => {
            write_msg(MsgType::TestResult.wire_value(), &bytes);
        }
        Some(TestResultWire::Stripped { bytes, dropped }) => {
            tracing::warn!(
                stripped = bytes.len(),
                dropped_samples = dropped,
                max,
                "AssertResult exceeded bulk frame; dropped per_cgroup raw samples, verdict preserved"
            );
            write_msg(MsgType::TestResult.wire_value(), &bytes);
        }
        Some(TestResultWire::Truncated { offending }) => {
            tracing::error!(
                offending_size = offending,
                max,
                "AssertResult exceeds bulk port frame limit even after dropping samples, sending truncated verdict"
            );
            let truncated = crate::assert::AssertResult::fail(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!(
                    "AssertResult postcard size {offending} exceeded bulk port limit {max}; \
                     original details dropped",
                ),
            ));
            if let Ok(small) = postcard::to_stdvec(&truncated) {
                write_msg(MsgType::TestResult.wire_value(), &small);
            }
        }
        None => {
            eprintln!("ktstr: postcard-encode AssertResult for bulk-port emit failed");
        }
    }
}

/// Send per-payload-invocation metrics to the host. Payload:
/// postcard-encoded [`crate::test_support::PayloadMetrics`].
///
/// Frames with [`MsgType::PayloadMetrics`]. Returns `true` when the
/// frame was fully written, `false` when the bulk port was not open
/// (handshake in flight), the write failed, the postcard encode
/// failed, OR the call originated from host context (the
/// [`write_msg`] -> [`assert_guest_context`] early-return). The
/// fire-and-forget caller in
/// `crate::scenario::payload_run::emit_payload_metrics` discards
/// the return at statement position; the boolean exists so the
/// host-context no-op is observable to a test rather than swallowed.
pub fn send_payload_metrics(metrics: &crate::test_support::PayloadMetrics) -> bool {
    match postcard::to_stdvec(metrics) {
        Ok(bytes) => write_msg(MsgType::PayloadMetrics.wire_value(), &bytes),
        Err(e) => {
            eprintln!("ktstr: postcard-encode PayloadMetrics for bulk-port emit: {e}");
            false
        }
    }
}

/// Send a coverage profraw blob to the host. Payload: raw `.profraw`
/// bytes serialized by `__llvm_profile_write_buffer`.
///
/// Frames with [`MsgType::Profraw`]. Gated on `cfg(any(test, coverage))`
/// because the only callers are the coverage-build guest flush
/// (`try_flush_profraw`, `cfg(coverage)`) and its host-context
/// suppression unit test (`cfg(test)`); a plain lib build has none.
#[cfg(any(test, coverage))]
pub fn send_profraw(buf: &[u8]) {
    write_msg(MsgType::Profraw.wire_value(), buf);
}

/// Plan the bulk frames for a wprof trace of `buf`, splitting at `cap`
/// bytes per frame: all but the last slice are tagged
/// [`MsgType::WprofTraceChunk`], the last is the terminal
/// [`MsgType::WprofTrace`]. An empty `buf` yields exactly one empty terminal
/// `WprofTrace` frame (matching the pre-chunking single-frame behavior). A
/// `buf` that fits in one `cap`-byte frame yields a lone terminal
/// `WprofTrace` (no chunks), so the host reassembly
/// ([`crate::test_support::wprof::reassemble_wprof_trace`]) is a no-op for it.
///
/// Split out from [`send_wprof_trace`] with `cap` as a parameter so the
/// chunk→reassemble roundtrip is tested against the exact production split —
/// production passes [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`]; the
/// roundtrip test passes a small cap for cheap multi-frame coverage plus the
/// real cap at its boundary.
///
/// `cap` MUST be > 0 (`slice::chunks(0)` panics). Not reachable from any
/// caller — production passes the non-zero `MAX_BULK_FRAME_PAYLOAD` const and
/// tests pass fixed non-zero caps — so this is a `debug_assert`, not a guest-
/// reachable check (`cap` is host-chosen, never guest input).
#[cfg(feature = "wprof")]
pub(crate) fn wprof_trace_frames(buf: &[u8], cap: usize) -> Vec<(u32, &[u8])> {
    debug_assert!(
        cap > 0,
        "wprof_trace_frames: cap must be > 0 (chunks(0) panics)"
    );
    let mut it = buf.chunks(cap).peekable();
    if it.peek().is_none() {
        // Empty trace: one empty terminal frame.
        return vec![(MsgType::WprofTrace.wire_value(), &buf[..0])];
    }
    let mut frames = Vec::new();
    while let Some(chunk) = it.next() {
        let msg_type = if it.peek().is_none() {
            MsgType::WprofTrace.wire_value()
        } else {
            MsgType::WprofTraceChunk.wire_value()
        };
        frames.push((msg_type, chunk));
    }
    frames
}

/// Send a wprof Perfetto-format trace blob to the host.
///
/// A wprof `.pb` larger than the single-frame bulk-port ceiling
/// ([`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`], 16 MiB) is split by
/// [`wprof_trace_frames`] into `WprofTraceChunk` slices + a terminal
/// `WprofTrace`; the host concatenates them to reconstruct the `.pb`. CHUNK,
/// never cap — no trace bytes are dropped (an oversized single frame would
/// otherwise be silently dropped by the host `HostAssembler`).
#[cfg(feature = "wprof")]
pub fn send_wprof_trace(buf: &[u8]) -> bool {
    let cap = crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD as usize;
    for (msg_type, chunk) in wprof_trace_frames(buf, cap) {
        if !write_msg(msg_type, chunk) {
            eprintln!(
                "ktstr: send_wprof_trace: bulk-port write failed at a {}-byte frame — wprof trace truncated",
                chunk.len()
            );
            return false;
        }
    }
    true
}

/// Send a stimulus event from the guest step executor.
///
/// Payload: byte-serialised [`crate::vmm::wire::StimulusPayload`]
/// (24 bytes, `IntoBytes`-derived). Frames with
/// [`MsgType::Stimulus`].
pub fn send_stimulus(payload: &[u8]) {
    write_msg(MsgType::Stimulus.wire_value(), payload);
}

/// Send a per-step END frame from the guest step executor. Reuses the
/// same byte-serialised [`crate::vmm::wire::StimulusPayload`] body as
/// [`send_stimulus`] (24 bytes) but frames with [`MsgType::StepEnd`] so
/// the host can pair `StepStart[k]` -> `StepEnd[k]` for step-local
/// throughput. Emitted at each step's end-of-hold while its workers are
/// still alive.
pub fn send_step_end(payload: &[u8]) {
    write_msg(MsgType::StepEnd.wire_value(), payload);
}

/// Send a scheduler-process exit notification. Payload: 4-byte LE i32
/// containing the scheduler's exit code.
///
/// Frames with [`MsgType::SchedExit`]. The host's freeze coordinator
/// promotes a SchedExit message into the run-wide kill flag so the
/// test ends promptly instead of waiting for the watchdog.
pub fn send_sched_exit(code: i32) {
    write_msg(MsgType::SchedExit.wire_value(), &code.to_le_bytes());
}

/// Send a scheduler-swap notification to the host. Payload: empty.
///
/// Emitted by `kill_current_scheduler` (Op::DetachScheduler /
/// RestartScheduler / ReplaceScheduler) AFTER `wait_for_scx_disabled`
/// returns — so the kernel has already NULLed `*scx_root`
/// (`RCU_INIT_POINTER(scx_root, NULL)` precedes
/// `scx_set_enable_state(SCX_DISABLED)` in kernel/sched/ext.c) and the
/// prior scx_sched object is unlinked (`*scx_root` NULLed) and its
/// slab is subject to RCU-grace-period reuse. The host's freeze
/// coordinator decodes a CRC-valid frame and synchronously invalidates
/// the periodic-capture accessor (so the accessor-init worker rebuilds
/// against the next scheduler) rather than waiting up to one
/// SCAN_INTERVAL for its scx_root watchpoint poll to notice the rebind.
///
/// A lost frame is non-fatal: the watchpoint poll still tears the stale
/// accessor down within one SCAN_INTERVAL — the notify only collapses
/// that defer window. Retries up to 5×100 ms to ride out a transient
/// bulk-port hiccup, matching [`send_scenario_start`]; the port is
/// long-open by Op-dispatch time so a retry rarely fires.
///
/// The 100 ms backoff is a bounded poll-retry, not an evented wait, and
/// that is deliberate: the normal TX path blocks in the kernel's
/// `wait_port_writable` (evented) and succeeds on the first attempt, so
/// the loop only spins when `write_msg` returns false — i.e. the bulk-port
/// node is not yet openable/connected or its cached fd was invalidated.
/// Neither is a condition the guest can epoll on (device-node + multiport
/// connection state), so a bounded re-open-and-retry is the only recovery
/// — the same poll shape [`super::rust_init::send_sys_rdy_with_retry`]
/// uses for the initial handshake.
pub fn send_sched_swap_notify() {
    for attempt in 0..5 {
        if write_msg(MsgType::SchedSwapNotify.wire_value(), &[]) {
            return;
        }
        if attempt + 1 < 5 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    tracing::warn!(
        "send_sched_swap_notify: 5 retries failed — bulk port write never \
         succeeded; the host watchpoint poll will tear the stale \
         periodic-capture accessor down within one SCAN_INTERVAL instead"
    );
}

/// Send a scenario-start marker.
///
/// `MSG_TYPE_SCENARIO_START` is load-bearing: the host's freeze
/// coordinator gates the entire periodic-capture pipeline on the
/// first CRC-valid arrival (stamps `scenario_start_ns`, which the
/// capture loop reads as the anchor for boundary computation). A
/// silent loss here means `periodic_fired` stays at 0 regardless
/// of how many boundaries the workload should have crossed — the
/// failure mode this guards against.
///
/// `send_sys_rdy` already retries until the bulk-port multiport
/// handshake completes, so by Phase 5 the port is normally
/// already open. The retry here is belt-and-braces for the rare
/// case where the cached File handle was invalidated between
/// `send_sys_rdy` and this call (process restart, fd close from
/// an unrelated path) — a fresh `try_open_bulk_port` on each
/// retry recovers transparently. 5 retries × 100 ms = 500 ms
/// total budget, an order of magnitude under the periodic
/// capture's typical inter-boundary spacing so retries don't
/// shift downstream timing measurably.
///
/// The 100 ms backoff is a bounded poll-retry, not an evented wait —
/// deliberate, for the same reason as [`send_sched_swap_notify`]: the
/// normal TX blocks in the kernel's `wait_port_writable` (evented) and
/// succeeds first try, so this loop only spins when `write_msg` returns
/// false, i.e. the port is not yet openable/connected or its cached fd
/// was invalidated — neither a state the guest can epoll on, so a bounded
/// re-open-and-retry is the only recovery.
pub fn send_scenario_start() {
    for attempt in 0..5 {
        if write_msg(MsgType::ScenarioStart.wire_value(), &[]) {
            return;
        }
        if attempt + 1 < 5 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    tracing::warn!(
        "send_scenario_start: 5 retries failed — bulk port write never \
         succeeded; periodic captures will see scenario_anchor=0 and \
         silently 0-fire"
    );
}

/// Publish the instrumentation-ready edge for a configured host BPF-map
/// write. The guest calls this only when the host put
/// `KTSTR_AWAIT_BPF_MAP_WRITE_READY=1` on the cmdline. By Phase 5 the bulk
/// port is already established, but use the same bounded reopen/retry policy
/// as [`send_scenario_start`] so a transient invalidated fd cannot silently
/// turn a requested crash injection into an uninstrumented race.
pub fn send_bpf_map_write_ready() -> bool {
    for attempt in 0..5 {
        if write_msg(MsgType::BpfMapWriteReady.wire_value(), &[]) {
            return true;
        }
        if attempt + 1 < 5 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    tracing::warn!(
        "send_bpf_map_write_ready: 5 retries failed — host map injection remains safely gated"
    );
    false
}

/// Send a scenario-end marker. Payload: two LE u64s —
/// `elapsed_ms` (since scenario start) followed by
/// `total_iterations`, the final cumulative worker iteration count
/// summed across every live handle at the LAST step's end. The host preserves
/// both values as terminal boundary telemetry; CPU-denominated iteration rates
/// come from guest per-cgroup CPU-time carriers. Parsed host-side by
/// [`crate::vmm::wire::parse_scenario_end`].
pub fn send_scenario_end(elapsed_ms: u64, total_iterations: u64) {
    let mut payload = [0u8; crate::vmm::wire::SCENARIO_END_PAYLOAD_SIZE];
    payload[0..8].copy_from_slice(&elapsed_ms.to_le_bytes());
    payload[8..16].copy_from_slice(&total_iterations.to_le_bytes());
    write_msg(MsgType::ScenarioEnd.wire_value(), &payload);
}

/// Publish an ordered teardown barrier and wait for the host to
/// acknowledge it over port 1 RX.
///
/// This is deliberately stronger than a successful [`write_msg`]:
/// virtio-console returns once the guest has offered a descriptor, not
/// once the host has consumed it. The progress-watchdog fault injector
/// enters an infinite spin immediately after this call, so it needs proof
/// that the coordinator has already persisted every earlier bulk frame
/// (notably `TestResult` and `ScenarioEnd`) and promoted the lifecycle to
/// Teardown. The host sends a zero-payload
/// [`MsgType::TeardownBarrierAck`] only after those effects are complete.
///
/// Shares [`SNAPSHOT_REQUEST_LOCK`] with the other port-1 RPCs because
/// their replies use the same O_RDWR file description and cannot be
/// demultiplexed safely by concurrent readers.
///
/// Unlike ordinary request/reply RPCs, this rendezvous intentionally has no
/// guest-side wall-clock deadline. Its caller is about to inject a watchdog
/// fixture, and a wall deadline here would race the very host contention that
/// the progress watchdog is designed to tolerate. The blocking virtio read
/// consumes no guest CPU while the host coordinator is descheduled; the VM
/// watchdog and nextest's outer rail remain the owners of lifecycle bounds.
pub fn send_teardown_barrier_and_wait() -> Result<(), String> {
    if !is_guest() {
        return Err("send_teardown_barrier_and_wait called from host context \
             (virtio-console port 1 is reachable only from inside the guest)"
            .into());
    }
    let _rpc_guard = SNAPSHOT_REQUEST_LOCK.lock_unpoisoned();
    if !write_msg(MsgType::TeardownBarrier.wire_value(), &[]) {
        return Err(format!(
            "failed to publish TeardownBarrier on named virtio-console port {PORT1_NAME:?}"
        ));
    }

    let mut read_file = clone_or_open_bulk_port().map_err(|error| error.to_string())?;
    let (msg_type, payload) = match read_bulk_port_frame(&mut read_file, 0, None) {
        Ok(frame) => frame,
        Err(e) => {
            drop(read_file);
            invalidate_bulk_port();
            return Err(format!("teardown barrier acknowledgement failed: {e}"));
        }
    };
    if msg_type != MsgType::TeardownBarrierAck.wire_value() || !payload.is_empty() {
        return Err(format!(
            "unexpected teardown barrier reply: msg_type=0x{msg_type:08x}, \
             payload_len={} (expected empty TDBA)",
            payload.len()
        ));
    }
    Ok(())
}

pub fn send_scenario_pause() {
    write_msg(MsgType::ScenarioPause.wire_value(), &[]);
}

pub fn send_scenario_resume() {
    write_msg(MsgType::ScenarioResume.wire_value(), &[]);
}

/// Send a workload-progress heartbeat carrying the guest's own
/// scenario-elapsed time in milliseconds (see
/// [`crate::vmm::wire::MSG_TYPE_WORKLOAD_PROGRESS`]). Emitted from the
/// scenario driver's existing scheduler-liveness wakeups so the host can
/// place periodic captures on the guest clock without a dedicated heartbeat
/// thread; a no-op when the bulk port is not yet open.
pub fn send_workload_progress(elapsed_ms: u64) {
    write_msg(
        crate::vmm::wire::MSG_TYPE_WORKLOAD_PROGRESS,
        &elapsed_ms.to_le_bytes(),
    );
}

/// Send the boot-complete signal to the host. Payload: empty.
/// Returns `true` when the frame was fully written, `false` when the
/// bulk port is not yet open (the multiport handshake completes
/// asynchronously during kernel virtio_console init, so
/// the named bulk-port device may not exist on the first call after
/// `mount_filesystems()` returns) or the write failed.
///
/// Frames an empty payload with [`MsgType::SysRdy`] and routes
/// through the bulk port. The host's freeze coordinator promotes
/// a CRC-valid SYS_RDY frame into the monitor's boot-complete
/// eventfd, releasing the monitor's pre-sample epoll wait. Called
/// from `vmm::rust_init::send_sys_rdy_with_retry` after
/// `mount_filesystems` completes, so the host's first sample
/// observes a fully-booted guest with `setup_per_cpu_areas` and
/// KASLR randomization already done.
///
/// The boolean return lets the caller retry on transient
/// not-yet-open failures: the multiport handshake completes
/// independently of `mount_filesystems`'s devtmpfs mount, so a
/// single call right after the mount can race the handshake. The
/// retry loop in `send_sys_rdy_with_retry` polls until success or
/// budget exhaustion, ensuring the host eventually observes the
/// signal rather than silently dropping the boot-complete event.
pub fn send_sys_rdy() -> bool {
    write_msg(MsgType::SysRdy.wire_value(), &[])
}

/// Send the typed [`crate::vmm::wire::KernAddrs`] payload to the
/// host so the monitor can translate kernel virtual addresses
/// without walking guest page tables. Called from
/// `vmm::rust_init::send_sys_rdy_with_retry` once the console bulk
/// port is ready (its evented node-appearance wait fires), and
/// LATCHED: once this succeeds the retry loop skips it while it keeps
/// attempting [`send_sys_rdy`].
///
/// The wire layout, the per-field encoding (including the +1
/// bias on present-bit slots), and the host-side decode contract
/// all live on the typed struct — see
/// [`crate::vmm::wire::KernAddrs`] for the full reference. This
/// helper is a thin transport wrapper that delegates to
/// [`crate::vmm::wire::KernAddrs::to_payload`] and ships the
/// bytes through the host_comms TLV channel.
///
/// The runtime `_text` KVA in the payload powers the
/// cross-architecture virt-KASLR derive at
/// `src/vmm/freeze_coord/dispatch.rs`'s KERN_ADDRS arm:
/// `virt_kaslr = _text_runtime - _text_link`, where the link-time
/// KVA comes from the host's vmlinux parse
/// (`KernelSymbols::kernel_text_kva` at `src/monitor/symbols.rs`).
/// `_text` is defined in `vmlinux.lds.S` on every Linux build so
/// the derivation works on both x86_64 and aarch64.
///
/// Two independent paths feed the same
/// `Arc<AtomicU64> kern_virt_kaslr` on the host: this guest-side
/// derivation (cross-arch), and the BSP-side
/// `KVM_GET_MSRS(MSR_LSTAR)` readback
/// (`src/vmm/x86_64/msr_kaslr::read_and_derive`, x86_64-only).
/// Either is sufficient on x86_64; on aarch64 only the guest
/// channel and the `nokaslr` cmdline gate participate.
pub fn send_kern_addrs(addrs: &super::wire::KernAddrs) -> bool {
    let payload = addrs.to_payload();
    write_msg(super::wire::MSG_TYPE_KERN_ADDRS, &payload)
}

/// Send the running kernel's GNU build-id to the host
/// (`MSG_TYPE_KERN_BUILD_ID`). Returns whether the frame was written
/// (`false` on an empty build-id or a closed port). Empty is a benign
/// "could not read" — the host skips the cache-consistency check rather
/// than treating an absent build-id as a mismatch.
pub fn send_kern_build_id(build_id: &[u8]) -> bool {
    if build_id.is_empty() {
        return false;
    }
    write_msg(super::wire::MSG_TYPE_KERN_BUILD_ID, build_id)
}

/// Read the running kernel's GNU build-id from `/sys/kernel/notes`.
///
/// `/sys/kernel/notes` exposes the raw bytes of the kernel image's
/// `SHT_NOTE` sections. Walk the ELF-note records (each
/// `namesz|descsz|type` header + 4-byte-aligned name + 4-byte-aligned
/// desc) and return the descriptor of the `NT_GNU_BUILD_ID` (type 3)
/// note whose owner name is `"GNU"`. Returns `None` when sysfs is
/// absent, the file is unreadable, or no build-id note is present — the
/// caller treats that as "no publish", and the host skips its check.
/// The descriptor is truncated to `KERN_BUILD_ID_MAX` defensively.
pub fn read_kernel_build_id() -> Option<Vec<u8>> {
    let bytes = std::fs::read("/sys/kernel/notes").ok()?;
    parse_gnu_build_id_note(&bytes)
}

/// Pure ELF-note walk over a `SHT_NOTE` blob (native endianness, the
/// only layout `/sys/kernel/notes` exposes): return the descriptor of
/// the first `NT_GNU_BUILD_ID` (type 3) note owned by `"GNU"`, truncated
/// to [`super::wire::KERN_BUILD_ID_MAX`]. `None` on absence or a
/// malformed/truncated record. Split out from [`read_kernel_build_id`]
/// so the parse is unit-testable without a live sysfs.
fn parse_gnu_build_id_note(bytes: &[u8]) -> Option<Vec<u8>> {
    const NT_GNU_BUILD_ID: u32 = 3;
    let align4 = |n: usize| n.div_ceil(4) * 4;
    let mut off = 0usize;
    while off + 12 <= bytes.len() {
        let namesz = u32::from_ne_bytes(bytes[off..off + 4].try_into().ok()?) as usize;
        let descsz = u32::from_ne_bytes(bytes[off + 4..off + 8].try_into().ok()?) as usize;
        let ntype = u32::from_ne_bytes(bytes[off + 8..off + 12].try_into().ok()?);
        let name_off = off + 12;
        let desc_off = name_off + align4(namesz);
        let next = desc_off + align4(descsz);
        // Malformed/over-long record: stop rather than risk a runaway
        // or an out-of-bounds slice on a truncated read.
        if next > bytes.len() || next <= off {
            break;
        }
        if ntype == NT_GNU_BUILD_ID
            && namesz >= 3
            && &bytes[name_off..name_off + 3] == b"GNU"
            && descsz > 0
        {
            let end = desc_off + descsz.min(super::wire::KERN_BUILD_ID_MAX);
            return Some(bytes[desc_off..end].to_vec());
        }
        off = next;
    }
    None
}

#[cfg(test)]
mod build_id_tests {
    use super::parse_gnu_build_id_note;

    /// Encode one ELF note (native-endian header + 4-byte-aligned name
    /// and desc), matching the `/sys/kernel/notes` layout.
    fn note(name: &[u8], ntype: u32, desc: &[u8]) -> Vec<u8> {
        let pad = |v: &mut Vec<u8>| {
            while !v.len().is_multiple_of(4) {
                v.push(0)
            }
        };
        let mut out = Vec::new();
        out.extend_from_slice(&(name.len() as u32).to_ne_bytes());
        out.extend_from_slice(&(desc.len() as u32).to_ne_bytes());
        out.extend_from_slice(&ntype.to_ne_bytes());
        out.extend_from_slice(name);
        pad(&mut out);
        out.extend_from_slice(desc);
        pad(&mut out);
        out
    }

    #[test]
    fn extracts_gnu_build_id_and_skips_other_notes() {
        let id = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03];
        let mut blob = note(b"GNU\0", 1, b"other"); // wrong type, skipped
        blob.extend(note(b"Linux\0\0\0", 3, b"xxxx")); // right type, wrong owner
        blob.extend(note(b"GNU\0", 3, &id)); // the real build-id
        assert_eq!(parse_gnu_build_id_note(&blob).as_deref(), Some(&id[..]));
    }

    #[test]
    fn none_when_absent_or_truncated() {
        assert_eq!(parse_gnu_build_id_note(&[]), None);
        assert_eq!(parse_gnu_build_id_note(&note(b"GNU\0", 1, b"x")), None);
        // A header claiming a desc longer than the blob is rejected, not
        // read out of bounds.
        let mut torn = 4u32.to_ne_bytes().to_vec(); // namesz
        torn.extend_from_slice(&999u32.to_ne_bytes()); // descsz (overlong)
        torn.extend_from_slice(&3u32.to_ne_bytes()); // NT_GNU_BUILD_ID
        torn.extend_from_slice(b"GNU\0");
        assert_eq!(parse_gnu_build_id_note(&torn), None);
    }

    #[test]
    fn caps_overlong_build_id() {
        let long = vec![0xab; crate::vmm::wire::KERN_BUILD_ID_MAX + 16];
        let got = parse_gnu_build_id_note(&note(b"GNU\0", 3, &long)).unwrap();
        assert_eq!(got.len(), crate::vmm::wire::KERN_BUILD_ID_MAX);
    }
}

/// Read the runtime virtual address of `_text` (the kernel image
/// start symbol) from `/proc/kallsyms`.
///
/// Returns `Some(kva)` when the symbol is present AND the address
/// is non-zero (kallsyms masks addresses to `0000000000000000`
/// when `kernel.kptr_restrict >= 1` and the reader lacks
/// `CAP_SYSLOG`). `rust_init` runs as PID 1 with all caps including
/// `CAP_SYSLOG`, so the read sees real addresses regardless of the
/// `kptr_restrict` sysctl default.
///
/// The kernel writes the post-relocation KVA into the symbol table
/// via `handle_relocations` in `arch/x86/boot/compressed/misc.c`
/// (x86_64) and via the kallsyms relocation pass in
/// `init/main.c::__init` (aarch64) before userspace boots, so by
/// the time guest userland can read `/proc/kallsyms` the entry
/// already reflects the runtime virt-KASLR slide. `_text` is
/// defined in `vmlinux.lds.S` on every Linux build, so this
/// returns a meaningful value on both x86_64 and aarch64 — and on
/// any other architecture ktstr might target in future.
pub fn read_kernel_text_from_kallsyms() -> Option<u64> {
    read_kallsyms_symbol_kva("_text", &["T", "t"])
}

/// Read the runtime virtual address of `page_offset_base` (the
/// CONFIG_RANDOMIZE_MEMORY direct-map slide global) from
/// `/proc/kallsyms`. Companion to [`read_kernel_text_from_kallsyms`].
///
/// `page_offset_base` is declared `unsigned long page_offset_base
/// __ro_after_init` at `arch/x86/kernel/head64.c:63` (kernel-PhD
/// confirmed); type letter is `D`/`d` (writable data section, even
/// though `__ro_after_init` lives in `.data..ro_after_init` which
/// the linker places in the RO image post-mark_rodata_ro). This
/// reader returns the RUNTIME KVA of the symbol; the VALUE stored
/// at that KVA — the direct-map base produced by
/// `kernel_randomize_memory` at boot — must be read separately by
/// the host via `text_kva_to_pa_with_base(kva, START_KERNEL_MAP,
/// phys_base)` followed by `read_u64`. Returns `None` when the
/// symbol is absent (CONFIG_RANDOMIZE_MEMORY=n, arm64) OR when
/// kptr_restrict masks the address to 0 (kernel-PhD: default is 0,
/// ktstr-init runs as PID 1 with CAP_SYSLOG so this is rare).
pub fn read_kernel_page_offset_base_from_kallsyms() -> Option<u64> {
    read_kallsyms_symbol_kva("page_offset_base", &["D", "d"])
}

/// Shared `/proc/kallsyms` symbol-KVA reader. Both
/// [`read_kernel_text_from_kallsyms`] (type `T`/`t`) and
/// [`read_kernel_page_offset_base_from_kallsyms`] (type `D`/`d`)
/// dispatch through here. The 16-char hex format is fixed-width per
/// `kernel/kallsyms.c::s_show` on 64-bit kernels (kernel-PhD
/// confirmed); leading zeros are not suppressed. Returns `None` on
/// (a) `/proc/kallsyms` unreadable, (b) symbol absent, (c)
/// `addr == 0` (kptr_restrict elevated or symbol stripped — caller
/// MUST treat as "not readable", not as a legitimate zero KVA).
fn read_kallsyms_symbol_kva(name: &str, allowed_types: &[&str]) -> Option<u64> {
    let kallsyms = std::fs::read_to_string("/proc/kallsyms").ok()?;
    for line in kallsyms.lines() {
        let mut parts = line.split_ascii_whitespace();
        let addr = parts.next()?;
        let typ = parts.next()?;
        let sym = parts.next()?;
        if sym == name && allowed_types.contains(&typ) {
            let kva = u64::from_str_radix(addr, 16).ok()?;
            if kva != 0 {
                return Some(kva);
            }
        }
    }
    None
}

/// Derive the KASLR physical displacement from `/proc/iomem`.
///
/// The published value's CONTRACT is "the displacement the host adds
/// to its text-KVA → PA translations" (`text_kva_to_pa_with_base`'s
/// `phys_base` argument) — the offset of the kernel IMAGE BASE
/// (`_text`) from its default load PA.
///
/// x86_64: the "Kernel code" resource starts at `__pa_symbol(_text)`
/// (arch/x86/kernel/setup.c), and the default load PA is
/// `LOAD_PHYSICAL_ADDR` (0x100_0000, CONFIG_PHYSICAL_START):
/// `phys_base = code_pa - 0x100_0000`.
///
/// aarch64: the resource start is VERSION-DEPENDENT —
/// `__pa_symbol(_stext)` up to v6.14 and `__pa_symbol(_text)` on
/// newer kernels (arch/arm64/kernel/setup.c:217 in each) — and
/// `_stext - _text` is the `.head.text` + SEGMENT_ALIGN gap
/// (0x10000 on our 6.14 4K-granule builds, absorbed into `_text`
/// on ≥6.15-era layouts). Naively publishing `code_pa − ram_start`
/// therefore reported the `_stext` bias, NOT a load displacement:
/// an observed arm64/6.14 guest published `0x10000` and the host's
/// per-tick adoption displaced every text translation by it,
/// reading a stable-garbage `__per_cpu_offset[]` page and keeping
/// the evidence channels dead all run (7.1 kernels publish 0 and
/// were unaffected — the version split that made the failure
/// 6.14-only). Subtract the same-boot `_stext − _text` gap from
/// kallsyms so the resource-start convention cancels on every
/// kernel generation: with `KERNEL_LOAD_ADDR == DRAM_START` by VMM
/// construction (aarch64/kvm.rs) and no physical self-relocation on
/// arm64 (KASLR randomizes the VIRTUAL mapping only), the debiased
/// value is 0 today — matching the accessor-init worker's
/// documented `phys_base = 0` decouple invariant
/// (`freeze_coord/mod.rs`) — and stays correct if a future VMM
/// loads the Image displaced. When kallsyms is unreadable the
/// aarch64 arm returns `None` rather than a possibly-biased guess
/// (the host's CR3-walk fallback and per-tick adoption tolerate an
/// absent publish; a WRONG publish poisons every translation).
pub fn read_phys_base_from_iomem() -> Option<u64> {
    let iomem = std::fs::read_to_string("/proc/iomem").ok()?;
    #[cfg(target_arch = "x86_64")]
    {
        for line in iomem.lines() {
            let line = line.trim();
            if line.ends_with(": Kernel code") {
                let range = line.split(':').next()?.trim();
                let start = range.split('-').next()?.trim();
                let phys_load = u64::from_str_radix(start, 16).ok()?;
                return Some(phys_load.wrapping_sub(0x100_0000));
            }
        }
        None
    }
    #[cfg(target_arch = "aarch64")]
    {
        // First "System RAM" entry = lowest-addressed DRAM region.
        // KERNEL_LOAD_ADDR == DRAM_START by construction in our VMM,
        // so the kernel always loads at this base. Parse the "Kernel
        // code" START (the displacement-bearing quantity) and its END
        // (the convention discriminator — see below).
        let mut ram_start: Option<u64> = None;
        let mut code_start: Option<u64> = None;
        let mut code_end: Option<u64> = None;
        for line in iomem.lines() {
            let line = line.trim();
            if ram_start.is_none() && line.ends_with(": System RAM") {
                let range = line.split(':').next()?.trim();
                let start = range.split('-').next()?.trim();
                ram_start = Some(u64::from_str_radix(start, 16).ok()?);
            }
            if line.ends_with(": Kernel code") {
                let range = line.split(':').next()?.trim();
                let mut parts = range.split('-');
                code_start = Some(u64::from_str_radix(parts.next()?.trim(), 16).ok()?);
                code_end = Some(u64::from_str_radix(parts.next()?.trim(), 16).ok()?);
            }
        }
        let delta = code_start?.wrapping_sub(ram_start?);
        // Debias the resource-start convention CONDITIONALLY. The
        // resource starts at `__pa_symbol(_stext)` on ≤6.14 arm64 and
        // `__pa_symbol(_text)` on newer kernels
        // (arch/arm64/kernel/setup.c:217 in each), while the SYMBOL gap
        // `_stext − _text` (the `.head.text` + SEGMENT_ALIGN pad,
        // 0x10000 on our 4K-granule builds) exists on BOTH layouts — an
        // earlier unconditional subtraction underflowed the 7.1-era
        // value to −0x10000 and poisoned every host text translation.
        //
        // Discriminator: the resource SIZE. The end is `__init_begin−1`
        // on both generations, so
        //   size == __init_begin − _stext  ⇔ the start is `_stext`
        //   size == __init_begin − _text   ⇔ the start is `_text`
        // Symbol DIFFERENCES from the same-boot kallsyms are exact
        // (every symbol carries the same virtual KASLR slide) and a
        // physical load displacement shifts start and end together,
        // never the size — so this discriminates correctly even under
        // real physical KASLR, where the naive `delta == gap` test is
        // ambiguous (delta = displacement + maybe-bias). When the gap
        // is 0 the conventions coincide and no bias exists. An
        // unrecognized size (neither candidate — unexpected layout, or
        // `__init_begin` missing from kallsyms) publishes nothing
        // rather than a guess: the host tolerates an absent publish
        // (CR3-walk fallback + per-tick adoption), while a WRONG one
        // poisons every translation.
        let stext = read_kallsyms_symbol_kva("_stext", &["T", "t"])?;
        let text = read_kallsyms_symbol_kva("_text", &["T", "t"])?;
        let stext_gap = stext.wrapping_sub(text);
        if stext_gap == 0 {
            return Some(delta);
        }
        let init_begin = read_kallsyms_symbol_kva("__init_begin", &["D", "d", "T", "t", "R", "r"])?;
        let size = code_end?.wrapping_sub(code_start?).wrapping_add(1);
        let size_if_stext = init_begin.wrapping_sub(stext);
        let size_if_text = init_begin.wrapping_sub(text);
        if size == size_if_stext {
            Some(delta.wrapping_sub(stext_gap))
        } else if size == size_if_text {
            Some(delta)
        } else {
            None
        }
    }
}

/// Send a stdout chunk to the host. Payload: opaque UTF-8 bytes.
///
/// Frames with [`MsgType::Stdout`]. Replaces the prior COM2
/// stdout redirect: the guest's stdout pipe forwarder (set up in
/// `redirect_stdio_to_bulk_port`) reads chunks from the pipe
/// read-end and feeds them through this sender. The host
/// concatenates chunks in arrival order to reconstruct the
/// stream. Each chunk SHOULD fit comfortably under
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`]; oversized chunks
/// are rejected by `write_to_bulk_port`'s `u32::try_from` length
/// guard plus the host-side per-frame cap and are logged.
///
/// Required: caller MUST split chunks at sub-cap boundaries. The
/// pipe forwarder uses 4 KiB reads which is well under the cap.
///
/// Optional: a not-yet-open bulk port returns `false` and the
/// chunk is dropped. The forwarder thread continues reading the
/// pipe — early-init bytes (before the multiport handshake
/// completes) are lost, mirroring the existing COM2 fallback's
/// "first bytes may not reach the host" caveat.
pub fn send_stdout_chunk(buf: &[u8]) -> bool {
    write_msg(MsgType::Stdout.wire_value(), buf)
}

/// Send a stderr chunk to the host. Payload: opaque UTF-8 bytes.
///
/// Frames with [`MsgType::Stderr`]. Same chunked semantics as
/// [`send_stdout_chunk`].
pub fn send_stderr_chunk(buf: &[u8]) -> bool {
    write_msg(MsgType::Stderr.wire_value(), buf)
}

/// Send a scheduler-log chunk to the host. Payload: opaque bytes from the
/// scheduler child process's captured log; verifier output is not required to
/// be valid UTF-8.
///
/// Frames with [`MsgType::SchedLog`]. The host concatenates
/// chunks in arrival order and the embedded `SCHED_OUTPUT_START` /
/// `SCHED_OUTPUT_END` delimiters travel verbatim inside the chunk
/// bytes, so the existing `parse_sched_output` walker (verifier
/// module) keeps slicing the log without changes. Replaces the
/// prior COM2 dump path in `dump_sched_output`.
///
/// Fallible scheduler-log frame send used by the idempotent dump transaction.
/// A `true` result means the complete TLV frame reached the kernel port write.
///
/// Required: caller chunks at sub-cap boundaries; same constraint as
/// [`send_stdout_chunk`].
pub(crate) fn try_send_sched_log(buf: &[u8]) -> bool {
    write_msg(MsgType::SchedLog.wire_value(), buf)
}

/// Try to send a scheduler-STDOUT chunk to the host. Payload: opaque UTF-8
/// bytes read live from the scheduler child's stdout pipe.
///
/// Frames with [`MsgType::SchedStdout`]. The scheduler-spawn forwarder
/// thread (in `try_spawn_scheduler`) ships each pipe read chunk through
/// this sender as it arrives, so the stdout stream survives a watchdog
/// timeout that never reaches the teardown `dump_sched_output`.
///
/// This live copy must never wait behind virtio backpressure while holding
/// [`GUEST_WRITE_LOCK`]: the forwarder also writes a complete per-stream
/// sidecar, and teardown replays that sidecar whenever any live attempt
/// fails. Immediate publication therefore preserves the fast/watchdog-visible
/// path when capacity is available without letting redundant traffic block a
/// required lifecycle boundary. Unlike [`try_send_sched_log`] the payload
/// carries no `SCHED_OUTPUT_START/END` framing — it is the raw child stream.
pub fn try_send_sched_stdout_chunk(buf: &[u8]) -> bool {
    try_write_msg(MsgType::SchedStdout.wire_value(), buf)
}

/// Try to send a scheduler-STDERR chunk to the host. Payload: opaque UTF-8
/// bytes read live from the scheduler child's stderr pipe.
///
/// Frames with [`MsgType::SchedStderr`]. Same live streaming semantics
/// as [`try_send_sched_stdout_chunk`], applied to the child's stderr (where
/// libbpf / log-crate output — including the BPF verifier log region —
/// typically lands).
pub fn try_send_sched_stderr_chunk(buf: &[u8]) -> bool {
    try_write_msg(MsgType::SchedStderr.wire_value(), buf)
}

/// Send an authoritative terminal scheduler-STDOUT chunk to the host.
///
/// The caller supplies the completion-aware payload produced by
/// [`crate::vmm::wire::encode_sched_stream_final_chunk`] after the scheduler
/// child and its stdout forwarder have exited. Unlike the best-effort live
/// stream, these frames are not suppressed by lifecycle priority.
pub(crate) fn try_send_sched_stdout_final_chunk(buf: &[u8]) -> bool {
    write_msg(MsgType::SchedStdoutFinal.wire_value(), buf)
}

/// Send an authoritative terminal scheduler-STDERR chunk to the host.
///
/// Same terminal replay semantics as
/// [`try_send_sched_stdout_final_chunk`], applied to stderr.
pub(crate) fn try_send_sched_stderr_final_chunk(buf: &[u8]) -> bool {
    write_msg(MsgType::SchedStderrFinal.wire_value(), buf)
}

/// Send a lifecycle phase event to the host. Payload: 1-byte
/// [`LifecyclePhase`] discriminant followed by a UTF-8 reason
/// suffix (only `SchedulerNotAttached` populates `reason`; every
/// other phase passes `""`).
///
/// Frames with [`MsgType::Lifecycle`]. Replaces the prior
/// `KTSTR_INIT_STARTED` / `KTSTR_PAYLOAD_STARTING` /
/// `SCHEDULER_DIED` / `SCHEDULER_NOT_ATTACHED` COM2 sentinel
/// strings. Host classifies init failure stages by walking the
/// per-VM lifecycle bucket instead of substring-matching on COM2
/// output.
///
/// Required: `phase` is a recognised [`LifecyclePhase`] discriminant
/// (currently wire values 1..=5). The 0 byte is reserved as the
/// host-side "unknown" sentinel and is rejected by
/// [`LifecyclePhase::from_wire`].
pub fn send_lifecycle(phase: LifecyclePhase, reason: &str) {
    let _ = try_send_lifecycle(phase, reason);
}

fn lifecycle_payload(phase: LifecyclePhase, reason: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + reason.len());
    buf.push(phase.wire_value());
    buf.extend_from_slice(reason.as_bytes());
    buf
}

fn try_send_lifecycle(phase: LifecyclePhase, reason: &str) -> bool {
    let payload = lifecycle_payload(phase, reason);
    write_msg(MsgType::Lifecycle.wire_value(), &payload)
}

const REQUIRED_FRAME_ATTEMPTS: usize = 5;

fn send_required_frame_with(mut send: impl FnMut() -> bool, mut wait: impl FnMut()) -> bool {
    for attempt in 0..REQUIRED_FRAME_ATTEMPTS {
        if send() {
            return true;
        }
        if attempt + 1 < REQUIRED_FRAME_ATTEMPTS {
            wait();
        }
    }
    false
}

/// Reliably publish a lifecycle proof which must precede an armed asynchronous
/// observer. Returns an error after bounded reopen retries instead of silently
/// allowing a later `SchedExit` frame to overtake a missing proof.
pub(crate) fn send_lifecycle_required(phase: LifecyclePhase, reason: &str) -> Result<(), String> {
    // Keep the attach transaction's priority continuously through the first
    // SchedulerAttached send. `SchedulerAttachAttempt::settle` consumes its
    // own guard immediately before this callback runs, and otherwise a noisy
    // scheduler forwarder can win that handoff race.
    let _priority = reserve_bulk_lifecycle_priority();
    let payload = lifecycle_payload(phase, reason);
    if send_required_frame_with(
        || try_write_msg(MsgType::Lifecycle.wire_value(), &payload),
        || std::thread::sleep(std::time::Duration::from_millis(100)),
    ) {
        Ok(())
    } else {
        Err(format!(
            "required lifecycle frame {phase:?} was not delivered after \
             {REQUIRED_FRAME_ATTEMPTS} attempts"
        ))
    }
}

/// Publish one generation-tagged scheduler attach boundary.
///
/// Returns the underlying bulk-write result so the spawn path can fail
/// closed if the host never received the `Started` boundary. All terminal
/// paths for a generation must send the matching `Finished` event; the host
/// ignores stale or mismatched generations rather than letting a delayed
/// completion close a newer lifecycle-Op attempt.
pub fn send_attach_attempt(event: AttachAttemptEvent) -> bool {
    try_write_msg(MsgType::AttachAttempt.wire_value(), &event.to_payload())
}

/// Send a shell-exec exit code to the host. Payload: 4-byte LE
/// i32 carrying the exec'd process's exit code.
///
/// Frames with [`MsgType::ExecExit`]. Replaces the prior COM2
/// `KTSTR_EXEC_EXIT=N` sentinel line emitted by `cargo ktstr
/// shell --exec <cmd>`.
pub fn send_exec_exit(code: i32) {
    write_msg(MsgType::ExecExit.wire_value(), &code.to_le_bytes());
}

/// Send a kernel ring-buffer dump to the host. Payload: opaque
/// UTF-8 bytes from `rmesg::logs_raw`.
///
/// Frames with [`MsgType::Dmesg`]. Sent on the
/// initramfs-extraction failure path so the host sees the kernel
/// OOM messages without scraping COM2.
pub fn send_dmesg(buf: &[u8]) {
    write_msg(MsgType::Dmesg.wire_value(), buf);
}

/// Send a probe-pipeline JSON output chunk to the host. Payload:
/// opaque UTF-8 bytes from the probe output stream.
///
/// Frames with [`MsgType::ProbeOutput`]. Replaces the prior COM2
/// ProbeDrain path so probe output and scheduler-log dumps stop
/// interleaving on the same serial port.
///
/// Required: caller chunks at sub-cap boundaries; same constraint
/// as [`send_stdout_chunk`].
#[allow(dead_code)]
pub fn send_probe_output(buf: &[u8]) {
    write_msg(MsgType::ProbeOutput.wire_value(), buf);
}

// ---------------------------------------------------------------------------
// Snapshot request (guest → host) + reply read-back
// ---------------------------------------------------------------------------

/// Monotonic guest-side request id counter. Bumped by every call to
/// [`request_snapshot`] before publishing the request frame.
/// `AtomicU32` so concurrent requests from different guest threads do
/// not produce duplicate ids. Wraparound past `u32::MAX` is
/// theoretically possible after billions of requests; the host's
/// reply pairing tolerates it because the comparison is equality
/// against the issuer's most-recent value, not a monotonicity check.
static SNAPSHOT_REQUEST_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1);

/// Mutex serialising guest-side request/reply RPCs over the
/// port-1 transport — both [`request_snapshot`] and
/// [`request_kernel_op`] take it before publishing. Without it two
/// guest threads issuing concurrent requests would interleave their
/// TX writes and race for each other's replies on the shared read fd
/// (only one open is permitted per port, so the snapshot reader and
/// the kernel-op reader share the same `BULK_PORT_FD` handle). The
/// freeze coordinator's `on_demand_in_flight` latch already collapses
/// doorbell floods to one capture per thaw on the host side; this
/// lock keeps the guest-side request/reply pairing well-defined for
/// every RPC kind too.
static SNAPSHOT_REQUEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Monotonic guest-side request id counter for
/// [`request_kernel_op`]. Kept separate from
/// [`SNAPSHOT_REQUEST_COUNTER`] so the two RPC kinds have independent
/// id sequences — the reply's [`MsgType`] distinguishes which counter
/// the id pairs against, but separate counters keep on-the-wire ids
/// monotonic per request kind which simplifies host-side logs.
static KERNEL_OP_REQUEST_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1);

/// Number of fast-poll iterations at the start of
/// [`bounded_read_exact`] before escalating to the slow-poll cadence.
/// Four iterations of 100µs gives ~400µs of fast-path coverage,
/// enough to absorb a host reply that lands in the virtqueue while
/// the guest is still entering `ppoll`, without burning more than
/// a hundred microseconds of cumulative wake-up budget.
const SNAPSHOT_FAST_POLL_ITERS: u32 = 4;
/// Per-iteration ppoll timeout for the first
/// [`SNAPSHOT_FAST_POLL_ITERS`] iterations (100µs). Sub-millisecond
/// granularity is the reason this path uses `ppoll` rather than
/// `poll(2)` (which only takes millisecond timeouts).
const SNAPSHOT_FAST_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_micros(100);
/// Per-iteration ppoll timeout after the fast-poll preamble (5ms).
/// Bounds the worst-case extra latency when virtio_console's
/// `port_fops_poll` does not deliver an early wake, while keeping
/// vCPU-thread wake-up cost low across the full snapshot deadline.
const SNAPSHOT_SLOW_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(5);

/// Read exactly `buf.len()` bytes from `f`, bounded by `deadline`.
/// Uses `ppoll(POLLIN)` between reads to wait without blocking past
/// the deadline. Returns `ErrorKind::TimedOut` when the deadline
/// expires before the read completes.
///
/// Each `ppoll` call's timeout is capped at an adaptive interval, not
/// the full remaining deadline:
///
/// * The first [`SNAPSHOT_FAST_POLL_ITERS`] iterations use a
///   [`SNAPSHOT_FAST_POLL_INTERVAL`] timeout (100µs). On the common
///   path the host's reply is already buffered in the virtqueue by
///   the time the guest enters `ppoll`, so a sub-millisecond bound
///   keeps wake-up latency low without burning CPU on the vCPU
///   thread.
/// * Subsequent iterations escalate to [`SNAPSHOT_SLOW_POLL_INTERVAL`]
///   (5ms), bounding the per-iteration wakeup cost while still
///   guaranteeing prompt deadline checks across the outer loop.
///
/// Each interval is further capped against the remaining deadline so
/// the loop never overshoots the caller's timeout.
fn bounded_read_exact(
    f: &mut std::fs::File,
    buf: &mut [u8],
    deadline: std::time::Instant,
) -> std::io::Result<()> {
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    let fd = f.as_raw_fd();
    let mut filled = 0usize;
    let mut iter: u32 = 0;
    while filled < buf.len() {
        let now = std::time::Instant::now();
        if now >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "snapshot reply deadline elapsed after reading {filled} of {} header/payload bytes",
                    buf.len()
                ),
            ));
        }
        let remaining = deadline - now;
        let interval = if iter < SNAPSHOT_FAST_POLL_ITERS {
            SNAPSHOT_FAST_POLL_INTERVAL
        } else {
            SNAPSHOT_SLOW_POLL_INTERVAL
        };
        // Cap the per-iteration sleep at min(interval, remaining) so
        // the last iteration before the deadline does not overshoot.
        let slice = remaining.min(interval);
        let ts = libc::timespec {
            tv_sec: slice.as_secs() as libc::time_t,
            tv_nsec: slice.subsec_nanos() as libc::c_long,
        };
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pfd is a valid &mut to a single pollfd; nfds is 1.
        // `ts` is a local timespec passed by const pointer. sigmask
        // is null so the caller's signal mask applies unchanged.
        // Every poll outcome (ready, timeout, EINTR, error) loops
        // back to the read attempt; EINTR is harmless because the
        // outer loop re-evaluates the deadline on every iteration.
        let pr = unsafe { libc::ppoll(&mut pfd, 1, &ts, std::ptr::null()) };
        iter = iter.saturating_add(1);
        if pr < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if pr == 0 {
            // ppoll timeout — re-check deadline at the loop head.
            continue;
        }
        match f.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "snapshot reply read returned 0 after {filled} of {} bytes",
                        buf.len()
                    ),
                ));
            }
            Ok(n) => {
                filled += n;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Read a single TLV frame (16-byte header + payload bytes) from
/// the named bulk port. Returns the parsed message type and payload on
/// success.
///
/// When `deadline` is `Some`, reads the header and payload with
/// [`bounded_read_exact`]. When it is `None`, uses the port's blocking
/// read mode and lets the VM lifecycle watchdog own the bound. On any
/// I/O failure (premature EOF, EINTR, etc.) the cached handle is
/// dropped so a subsequent call retries the open.
///
/// `max_payload_size` caps the payload allocation against a hostile
/// or corrupted host that frames an oversized length. Callers pass
/// the upper bound of any payload they expect to read on this
/// transport (e.g. `size_of::<SnapshotReplyPayload>()` for snapshot
/// replies, [`KERNEL_OP_REPLY_MAX`] for postcard-encoded kernel-op
/// replies); a length above the cap is rejected with `InvalidData`
/// BEFORE the `vec![0u8; length]` allocation so a forged
/// `length = u32::MAX` cannot OOM the guest's PID 1 init.
fn read_bulk_port_frame(
    f: &mut std::fs::File,
    max_payload_size: usize,
    deadline: Option<std::time::Instant>,
) -> std::io::Result<(u32, Vec<u8>)> {
    fn read_exact(
        f: &mut std::fs::File,
        buf: &mut [u8],
        deadline: Option<std::time::Instant>,
    ) -> std::io::Result<()> {
        match deadline {
            Some(deadline) => bounded_read_exact(f, buf, deadline),
            None => unbounded_poll_read_exact(f, buf),
        }
    }

    let mut header = [0u8; std::mem::size_of::<ShmMessage>()];
    read_exact(f, &mut header, deadline)?;
    let msg = ShmMessage::read_from_bytes(&header).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ShmMessage::read_from_bytes failed (header underflow)",
        )
    })?;
    let length = msg.length as usize;
    if length > max_payload_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "TLV length {length} exceeds max payload {max_payload_size} for port-1 RX; \
                 rejecting before allocation to avoid guest OOM"
            ),
        ));
    }
    let mut payload = vec![0u8; length];
    if length > 0 {
        read_exact(f, &mut payload, deadline)?;
    }
    let computed = crc32fast::hash(&payload);
    if computed != msg.crc32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "TLV CRC mismatch: header crc=0x{:08x} computed=0x{computed:08x} length={length}",
                msg.crc32
            ),
        ));
    }
    Ok((msg.msg_type, payload))
}

/// Event-driven exact read for the lifecycle fence, whose bound is owned by
/// the outer VM watchdog rather than a local wall clock. The cached bulk fd is
/// O_NONBLOCK so terminal control writes can fail immediately; an indefinite
/// `poll(POLLIN)` restores blocking semantics here without holding any shared
/// userspace mutex.
fn unbounded_poll_read_exact(f: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read;
    use std::os::fd::AsRawFd;

    let mut filled = 0usize;
    while filled < buf.len() {
        let mut pfd = libc::pollfd {
            fd: f.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` describes the live borrowed file descriptor and stays
        // valid for the duration of this one-fd poll call.
        let ready = unsafe { libc::poll(&mut pfd, 1, -1) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if pfd.revents & libc::POLLIN == 0
            && pfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "bulk port read poll returned terminal events {:#x}",
                    pfd.revents
                ),
            ));
        }
        match f.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "bulk port read returned 0 after {filled} of {} bytes",
                        buf.len()
                    ),
                ));
            }
            Ok(n) => filled += n,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Request a host-driven snapshot. Publishes a snapshot request via
/// the virtio-console port-1 TLV stream and blocks reading port 1 RX
/// until a matching [`MsgType::SnapshotReply`] arrives (or `timeout`
/// elapses).
///
/// `kind` selects the dispatch path on the host:
/// [`crate::vmm::wire::SNAPSHOT_KIND_CAPTURE`] for a capture-now
/// request, [`crate::vmm::wire::SNAPSHOT_KIND_WATCH`] for a
/// hardware-watchpoint registration.
///
/// `tag` is copied into the request payload's tag buffer up to
/// [`SNAPSHOT_TAG_MAX`] bytes. Longer tags are truncated.
///
/// Returns one of [`SnapshotRequestResult`] variants. The serialised
/// guest lock ensures only one in-flight request per process — this
/// matches the host coordinator's `on_demand_in_flight` invariant.
pub fn request_snapshot(
    kind: u32,
    tag: &str,
    timeout: std::time::Duration,
) -> SnapshotRequestResult {
    if !is_guest() {
        return SnapshotRequestResult::TransportError {
            reason: "request_snapshot called from host context (virtio-console port 1 \
                     is reachable only from inside the guest)"
                .into(),
        };
    }
    let _guard = SNAPSHOT_REQUEST_LOCK.lock_unpoisoned();
    // Allocate a request id. Skip 0 so the wait loop's `reply.request_id
    // == request_id` check cannot accidentally match a zero-initialised
    // reply payload from an earlier protocol version.
    let mut request_id = SNAPSHOT_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    if request_id == 0 {
        request_id = SNAPSHOT_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    // Build the request payload.
    let tag_bytes = tag.as_bytes();
    let tag_len = tag_bytes.len().min(SNAPSHOT_TAG_MAX);
    let mut tag_buf = [0u8; SNAPSHOT_TAG_MAX];
    tag_buf[..tag_len].copy_from_slice(&tag_bytes[..tag_len]);
    let payload = SnapshotRequestPayload {
        request_id,
        kind,
        tag: tag_buf,
    };
    // Send via the existing port-1 TX writer. `write_msg` already
    // takes `GUEST_WRITE_LOCK` internally, so this serialises with
    // every other guest TLV producer.
    let bytes = payload.as_bytes();
    write_msg(MsgType::SnapshotRequest.wire_value(), bytes);
    // Read replies from the same O_RDWR fd used for writes.
    // The kernel's port_fops_open allows only one concurrent open
    // per port (EBUSY on second open), so a separate read-only
    // open would fail. The write fd is opened O_RDWR by
    // try_open_bulk_port.
    let mut read_file = match clone_or_open_bulk_port() {
        Ok(file) => file,
        Err(error) => {
            return SnapshotRequestResult::TransportError {
                reason: error.to_string(),
            };
        }
    };
    // Read TLV reply frames until we observe one whose payload
    // request_id matches ours. Frames addressed to other request ids
    // (none in current protocol — the host only writes replies in
    // response to a specific request) or unknown msg_types are
    // logged + dropped.
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            return SnapshotRequestResult::TransportError {
                reason: format!(
                    "host did not deliver matching snapshot reply within {timeout:?} \
                     (request_id={request_id}, kind={kind})"
                ),
            };
        }
        let frame = match read_bulk_port_frame(
            &mut read_file,
            std::mem::size_of::<SnapshotReplyPayload>(),
            Some(deadline),
        ) {
            Ok(frame) => frame,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return SnapshotRequestResult::TransportError {
                    reason: format!(
                        "snapshot reply deadline elapsed before frame complete \
                         (request_id={request_id}, kind={kind}): {e}"
                    ),
                };
            }
            Err(e) => {
                // I/O error on the read fd — drop the cached
                // handle so the next call retries the open and
                // surface the failure to the caller.
                drop(read_file);
                invalidate_bulk_port();
                return SnapshotRequestResult::TransportError {
                    reason: format!("snapshot reply read failed (request_id={request_id}): {e}"),
                };
            }
        };
        let (msg_type, frame_payload) = frame;
        if msg_type != MSG_TYPE_SNAPSHOT_REPLY {
            tracing::warn!(
                msg_type,
                len = frame_payload.len(),
                request_id,
                "request_snapshot: ignoring unexpected TLV on port 1 RX (only \
                 SnapshotReply is expected on this transport in current protocol)"
            );
            continue;
        }
        if frame_payload.len() != std::mem::size_of::<SnapshotReplyPayload>() {
            tracing::warn!(
                request_id,
                got = frame_payload.len(),
                want = std::mem::size_of::<SnapshotReplyPayload>(),
                "request_snapshot: malformed reply payload size; ignoring"
            );
            continue;
        }
        let reply = match SnapshotReplyPayload::read_from_bytes(&frame_payload) {
            Ok(r) => r,
            Err(_) => {
                tracing::warn!(
                    request_id,
                    "request_snapshot: SnapshotReplyPayload::read_from_bytes failed; ignoring"
                );
                continue;
            }
        };
        if reply.request_id != request_id {
            tracing::warn!(
                expected = request_id,
                got = reply.request_id,
                "request_snapshot: stale reply id (likely a leftover from a prior \
                 request that timed out on the guest side); ignoring"
            );
            continue;
        }
        return match reply.status {
            SNAPSHOT_STATUS_OK => SnapshotRequestResult::Ok,
            SNAPSHOT_STATUS_ERR => {
                let len = reply
                    .reason
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(SNAPSHOT_REASON_MAX);
                let reason = String::from_utf8_lossy(&reply.reason[..len]).to_string();
                SnapshotRequestResult::HostError { reason }
            }
            other => SnapshotRequestResult::TransportError {
                reason: format!(
                    "host reply with unknown status {other} \
                     (expected OK={SNAPSHOT_STATUS_OK} or ERR={SNAPSHOT_STATUS_ERR})"
                ),
            },
        };
    }
}

/// Request a host-driven kernel-memory op (`Op::WriteKernel{Hot,Cold}`
/// / `Op::ReadKernel{Hot,Cold}`). Publishes a postcard-encoded
/// [`KernelOpRequestPayload`] via the virtio-console port-1 TLV
/// stream and blocks reading port 1 RX until a matching
/// [`MsgType::KernelOpReply`] arrives (or `timeout` elapses).
///
/// The supplied `request` carries the full op intent — mode
/// (hot/cold), direction (write/read), tag (for read replies and
/// diagnostics), and the ordered batch of `(target, value)` entries.
/// The function stamps a fresh `request_id` into the payload before
/// publishing (overriding whatever the caller put there) so the
/// reply pairing stays well-defined; the returned reply mirrors that
/// id back in [`KernelOpReplyPayload::request_id`].
///
/// Returns one of [`KernelOpRequestResult`] variants. Distinct from
/// [`SnapshotRequestResult`]: the "host completed but op failed"
/// carrier is [`KernelOpReplyPayload::success`] = false +
/// [`KernelOpReplyPayload::reason`], not a separate enum arm,
/// because postcard-encoded replies can carry per-entry result data
/// (e.g. read values) that an enum arm would erase.
///
/// Shares [`SNAPSHOT_REQUEST_LOCK`] with [`request_snapshot`]: only
/// one in-flight guest→host RPC per process, regardless of kind —
/// the shared `BULK_PORT_FD` read handle cannot safely demux two
/// concurrent reply streams.
///
/// The reply reader uses a dup of the cached O_RDWR file description, so it
/// does not hold the `BULK_PORT_FD` cache mutex while waiting (up to `timeout`,
/// normally 30 s for a cold-path freeze rendezvous). Concurrent guest writers
/// therefore retain normal port-level backpressure instead of serializing
/// behind an unrelated userspace mutex.
pub fn request_kernel_op(
    request: KernelOpRequestPayload,
    timeout: std::time::Duration,
) -> KernelOpRequestResult {
    if !is_guest() {
        return KernelOpRequestResult::TransportError {
            reason: "request_kernel_op called from host context (virtio-console port 1 \
                     is reachable only from inside the guest)"
                .into(),
        };
    }
    let _guard = SNAPSHOT_REQUEST_LOCK.lock_unpoisoned();
    // Allocate a request id. Skip 0 so the wait loop's `reply.request_id
    // == request_id` check cannot accidentally match a zero-initialised
    // reply payload from an earlier protocol version.
    let mut request_id =
        KERNEL_OP_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    if request_id == 0 {
        request_id = KERNEL_OP_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    // Stamp the freshly-allocated id into a clone of the request
    // payload (the caller's `request_id` field is overwritten — the
    // function owns id allocation per the doc contract).
    let stamped = KernelOpRequestPayload {
        request_id,
        ..request
    };
    let payload_bytes = match postcard::to_allocvec(&stamped) {
        Ok(b) => b,
        Err(e) => {
            return KernelOpRequestResult::TransportError {
                reason: format!(
                    "request_kernel_op: postcard encode failed (request_id={request_id}): {e}"
                ),
            };
        }
    };
    // Send via the existing port-1 TX writer. `write_msg` already
    // takes `GUEST_WRITE_LOCK` internally, so this serialises with
    // every other guest TLV producer.
    write_msg(MsgType::KernelOpRequest.wire_value(), &payload_bytes);
    // Read replies from the same O_RDWR fd used for writes. See
    // `request_snapshot` for the bulk-port handle lifecycle notes;
    // both helpers share `BULK_PORT_FD`.
    let mut read_file = match clone_or_open_bulk_port() {
        Ok(file) => file,
        Err(error) => {
            return KernelOpRequestResult::TransportError {
                reason: error.to_string(),
            };
        }
    };
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            return KernelOpRequestResult::TransportError {
                reason: format!(
                    "host did not deliver matching kernel-op reply within {timeout:?} \
                     (request_id={request_id})"
                ),
            };
        }
        let frame = match read_bulk_port_frame(&mut read_file, KERNEL_OP_REPLY_MAX, Some(deadline))
        {
            Ok(frame) => frame,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return KernelOpRequestResult::TransportError {
                    reason: format!(
                        "kernel-op reply deadline elapsed before frame complete \
                         (request_id={request_id}): {e}"
                    ),
                };
            }
            Err(e) => {
                drop(read_file);
                invalidate_bulk_port();
                return KernelOpRequestResult::TransportError {
                    reason: format!("kernel-op reply read failed (request_id={request_id}): {e}"),
                };
            }
        };
        let (msg_type, frame_payload) = frame;
        if msg_type != MSG_TYPE_KERNEL_OP_REPLY {
            tracing::warn!(
                msg_type,
                len = frame_payload.len(),
                request_id,
                "request_kernel_op: ignoring non-KernelOpReply TLV on port 1 RX (likely a \
                 stale snapshot reply from a prior request that timed out on the guest side)"
            );
            continue;
        }
        let reply: KernelOpReplyPayload = match postcard::from_bytes(&frame_payload) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    request_id,
                    error = %e,
                    "request_kernel_op: postcard decode failed; ignoring"
                );
                continue;
            }
        };
        if reply.request_id != request_id {
            tracing::warn!(
                expected = request_id,
                got = reply.request_id,
                "request_kernel_op: stale reply id (likely a leftover from a prior \
                 request that timed out on the guest side); ignoring"
            );
            continue;
        }
        return KernelOpRequestResult::Ok(reply);
    }
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the typed sender wrappers.
    //!
    //! Every guest_comms helper routes through `write_msg`
    //! which gates on `is_guest()`. The host-context check
    //! rejects every call from these tests — verifying that gate
    //! holds is the safest unit-test scope: it confirms the wrappers
    //! do not write to the host's bulk-port path.
    //!
    //! The void-returning senders give the caller no return value to
    //! inspect, so "did not write" is observed via
    //! [`BULK_PORT_WRITE_ATTEMPTS`]: [`assert_no_bulk_write`] snapshots
    //! the global write-path entry counter, runs the sender, and
    //! asserts the counter did not advance — the concrete no-op
    //! observation (suppression), not merely "did not panic". The
    //! `is_guest()` override is thread-local, so the host-context gate
    //! these tests exercise cannot be perturbed by parallel tests;
    //! and no test calls a sender in guest context, so the global
    //! counter is never advanced concurrently with these snapshots.
    //!
    //! End-to-end transport (guest → bulk port → host drain → TLV
    //! parse) is exercised by the integration test suite under
    //! `tests/`.

    use super::*;

    #[test]
    fn named_virtio_port_resolves_by_advertised_name_not_device_index() {
        let tmp = tempfile::tempdir().unwrap();
        let class_dir = tmp.path().join("sys/class/virtio-ports");
        let dev_dir = tmp.path().join("dev");
        std::fs::create_dir_all(class_dir.join("vport0p0")).unwrap();
        std::fs::create_dir_all(class_dir.join("vport1p1")).unwrap();
        std::fs::create_dir_all(class_dir.join("vport1p2")).unwrap();
        std::fs::create_dir_all(&dev_dir).unwrap();
        std::fs::write(class_dir.join("vport0p0/name"), "com.redhat.spice.0\n").unwrap();
        std::fs::write(class_dir.join("vport1p1/name"), PORT1_NAME).unwrap();
        std::fs::write(
            class_dir.join("vport1p2/name"),
            crate::vmm::wire::PORT2_NAME,
        )
        .unwrap();

        assert_eq!(
            named_virtio_port_path_in(&class_dir, &dev_dir, PORT1_NAME),
            Some(dev_dir.join("vport1p1")),
        );
        assert_eq!(
            named_virtio_port_path_in(&class_dir, &dev_dir, crate::vmm::wire::PORT2_NAME),
            Some(dev_dir.join("vport1p2")),
        );
    }

    #[test]
    fn named_virtio_port_ignores_entries_without_readable_name() {
        let tmp = tempfile::tempdir().unwrap();
        let class_dir = tmp.path().join("class");
        let dev_dir = tmp.path().join("dev");
        std::fs::create_dir_all(class_dir.join("vport0p0")).unwrap();
        std::fs::create_dir_all(class_dir.join("vport2p1")).unwrap();
        std::fs::write(class_dir.join("vport2p1/name"), PORT1_NAME).unwrap();

        assert_eq!(
            named_virtio_port_path_in(&class_dir, &dev_dir, PORT1_NAME),
            Some(dev_dir.join("vport2p1")),
        );
    }

    /// Run `f` (a host-context sender call) and assert it did NOT
    /// enter `write_to_bulk_port` — i.e. the `is_guest()` gate in
    /// `write_msg` suppressed the write. Snapshots
    /// [`BULK_PORT_WRITE_ATTEMPTS`] before and after; a host-context
    /// sender must leave it unchanged.
    fn assert_no_bulk_write(label: &str, f: impl FnOnce()) {
        use std::sync::atomic::Ordering;
        let before = BULK_PORT_WRITE_ATTEMPTS.load(Ordering::SeqCst);
        f();
        let after = BULK_PORT_WRITE_ATTEMPTS.load(Ordering::SeqCst);
        assert_eq!(
            after, before,
            "{label}: host-context call must NOT reach write_to_bulk_port; \
             the is_guest() gate failed to suppress the write \
             (before={before}, after={after})",
        );
    }

    /// `send_exit` from host context must suppress the bulk-port write.
    #[test]
    fn send_exit_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_exit(0)", || send_exit(0));
        assert_no_bulk_write("send_exit(-1)", || send_exit(-1));
    }

    /// `send_test_result` from host context suppresses the write.
    #[test]
    fn send_test_result_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_test_result", || {
            send_test_result(&crate::assert::AssertResult::pass())
        });
    }

    /// `send_payload_metrics` from host context is a no-op and
    /// reports it by returning `false` (the `assert_guest_context`
    /// early-return inside `write_msg`), mirroring `send_sys_rdy`.
    #[test]
    fn send_payload_metrics_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        let pm = crate::test_support::PayloadMetrics {
            payload_index: 0,
            metrics: vec![],
            exit_code: 0,
        };
        assert!(
            !send_payload_metrics(&pm),
            "host-context send must return false (no frame written)"
        );
    }

    /// `send_profraw` from host context suppresses the write.
    #[test]
    fn send_profraw_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_profraw", || send_profraw(b"\x01\x02\x03"));
    }

    /// `send_stimulus` from host context suppresses the write.
    #[test]
    fn send_stimulus_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_stimulus", || send_stimulus(&[0u8; 24]));
    }

    /// `send_sched_exit` from host context suppresses the write.
    #[test]
    fn send_sched_exit_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_sched_exit(0)", || send_sched_exit(0));
        assert_no_bulk_write("send_sched_exit(-1)", || send_sched_exit(-1));
    }

    /// `send_scenario_start` from host context suppresses the write.
    /// `send_scenario_start` retries the write up to 5 times on a
    /// not-yet-open port; the host-context gate must short-circuit
    /// EVERY attempt, so the write-path counter stays put despite the
    /// retry loop.
    #[test]
    fn send_scenario_start_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_scenario_start", send_scenario_start);
    }

    /// The ordered teardown barrier is a guest-only RPC. Host context
    /// must fail before touching the bulk-port writer or trying to wait
    /// for a reply that can never arrive.
    #[test]
    fn teardown_barrier_from_host_context_is_rejected_without_write() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_teardown_barrier_and_wait", || {
            let err = send_teardown_barrier_and_wait()
                .expect_err("host-context teardown barrier must be rejected");
            assert!(err.contains("host context"), "unexpected error: {err}");
        });
    }

    /// `send_sched_swap_notify` from host context suppresses the write.
    /// Like `send_scenario_start` it retries the write up to 5 times
    /// (4 × 100 ms sleeps) on a not-yet-open port; the host-context
    /// gate must short-circuit EVERY attempt, so the write-path counter
    /// stays put despite the retry loop.
    #[test]
    fn send_sched_swap_notify_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_sched_swap_notify", send_sched_swap_notify);
    }

    /// `send_scenario_end` from host context suppresses the write.
    #[test]
    fn send_scenario_end_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_scenario_end(0,0)", || send_scenario_end(0, 0));
        assert_no_bulk_write("send_scenario_end(MAX,MAX)", || {
            send_scenario_end(u64::MAX, u64::MAX)
        });
    }

    /// `send_sys_rdy` from host context returns false (no-op +
    /// failure indicator for the retry caller).
    #[test]
    fn send_sys_rdy_from_host_context_returns_false() {
        let _g = IsGuestOverrideGuard::new(false);
        assert!(
            !send_sys_rdy(),
            "host-context call must return false so the guest's \
             retry loop can distinguish 'wrote' from 'noop'"
        );
    }

    /// `send_stdout_chunk` from host context returns false
    /// (no-op + failure indicator), mirroring `send_sys_rdy`.
    #[test]
    fn send_stdout_chunk_from_host_context_returns_false() {
        let _g = IsGuestOverrideGuard::new(false);
        assert!(!send_stdout_chunk(b"hello"));
    }

    /// `send_stderr_chunk` from host context returns false.
    #[test]
    fn send_stderr_chunk_from_host_context_returns_false() {
        let _g = IsGuestOverrideGuard::new(false);
        assert!(!send_stderr_chunk(b"oops"));
    }

    /// `try_send_sched_log` from host context returns false without writing.
    #[test]
    fn try_send_sched_log_from_host_context_returns_false() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("try_send_sched_log", || {
            assert!(!try_send_sched_log(b"---SCHED_OUTPUT_START---\n"));
        });
    }

    /// `send_lifecycle` from host context suppresses the write for
    /// every phase, including the reason-bearing variant.
    #[test]
    fn send_lifecycle_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_lifecycle(InitStarted)", || {
            send_lifecycle(LifecyclePhase::InitStarted, "")
        });
        assert_no_bulk_write("send_lifecycle(PayloadStarting)", || {
            send_lifecycle(LifecyclePhase::PayloadStarting, "")
        });
        assert_no_bulk_write("send_lifecycle(SchedulerDied)", || {
            send_lifecycle(LifecyclePhase::SchedulerDied, "")
        });
        assert_no_bulk_write("send_lifecycle(SchedulerNotAttached)", || {
            send_lifecycle(LifecyclePhase::SchedulerNotAttached, "verifier rejected")
        });
    }

    #[test]
    fn required_frame_failure_is_bounded_and_reported() {
        let attempts = std::cell::Cell::new(0usize);
        let waits = std::cell::Cell::new(0usize);
        assert!(
            !send_required_frame_with(
                || {
                    attempts.set(attempts.get() + 1);
                    false
                },
                || waits.set(waits.get() + 1),
            ),
            "an undelivered required boundary must fail closed"
        );
        assert_eq!(attempts.get(), REQUIRED_FRAME_ATTEMPTS);
        assert_eq!(waits.get(), REQUIRED_FRAME_ATTEMPTS - 1);
    }

    #[test]
    fn lifecycle_priority_suppresses_only_redundant_scheduler_live_copies() {
        assert!(!lifecycle_priority_drops(MsgType::SchedStdout.wire_value()));
        let priority = reserve_bulk_lifecycle_priority();
        assert!(lifecycle_priority_drops(MsgType::SchedStdout.wire_value()));
        assert!(lifecycle_priority_drops(MsgType::SchedStderr.wire_value()));
        assert!(
            !lifecycle_priority_drops(MsgType::SchedStdoutFinal.wire_value()),
            "completion-proven stdout recovery must bypass live-copy suppression"
        );
        assert!(
            !lifecycle_priority_drops(MsgType::SchedStderrFinal.wire_value()),
            "completion-proven stderr recovery must bypass live-copy suppression"
        );
        assert!(
            !lifecycle_priority_drops(MsgType::Lifecycle.wire_value()),
            "required lifecycle traffic must bypass its own reservation"
        );
        assert!(
            !lifecycle_priority_drops(MsgType::TestResult.wire_value()),
            "authoritative non-scheduler frames must not be collateral drops"
        );
        drop(priority);
        assert!(!lifecycle_priority_drops(MsgType::SchedStdout.wire_value()));
    }

    #[test]
    fn immediate_frame_write_reports_partial_backpressure_without_waiting() {
        let calls = std::cell::Cell::new(0usize);
        let waits = std::cell::Cell::new(0usize);
        let progress = write_frame_with(
            b"complete-frame",
            BulkWriteMode::Immediate,
            |remaining| {
                let call = calls.get();
                calls.set(call + 1);
                if call == 0 {
                    Ok(4.min(remaining.len()))
                } else {
                    Err(std::io::Error::from(std::io::ErrorKind::WouldBlock))
                }
            },
            || {
                waits.set(waits.get() + 1);
                Ok(())
            },
        );
        assert_eq!(
            progress,
            BulkWriteProgress {
                complete: false,
                written: 4,
                failure: Some(BulkWriteFailure::Backpressured),
            }
        );
        assert_eq!(waits.get(), 0, "immediate lifecycle writes never poll");
    }

    #[test]
    fn blocking_frame_write_resumes_same_suffix_after_backpressure() {
        let calls = std::cell::Cell::new(0usize);
        let waits = std::cell::Cell::new(0usize);
        let mut observed_suffixes = Vec::new();
        let progress = write_frame_with(
            b"complete-frame",
            BulkWriteMode::Blocking,
            |remaining| {
                observed_suffixes.push(remaining.to_vec());
                let call = calls.get();
                calls.set(call + 1);
                match call {
                    0 => Ok(4),
                    1 => Err(std::io::Error::from(std::io::ErrorKind::WouldBlock)),
                    _ => Ok(remaining.len()),
                }
            },
            || {
                waits.set(waits.get() + 1);
                Ok(())
            },
        );
        assert_eq!(
            progress,
            BulkWriteProgress {
                complete: true,
                written: b"complete-frame".len(),
                failure: None,
            }
        );
        assert_eq!(waits.get(), 1);
        assert_eq!(&observed_suffixes[1], b"lete-frame");
        assert_eq!(&observed_suffixes[2], b"lete-frame");
    }

    #[test]
    fn partial_hard_failure_is_distinct_from_retryable_backpressure() {
        let calls = std::cell::Cell::new(0usize);
        let progress = write_frame_with(
            b"complete-frame",
            BulkWriteMode::Immediate,
            |remaining| {
                let call = calls.get();
                calls.set(call + 1);
                if call == 0 {
                    Ok(3.min(remaining.len()))
                } else {
                    Err(std::io::Error::other("injected hard write failure"))
                }
            },
            || panic!("hard failures must not wait"),
        );
        assert_eq!(
            progress,
            BulkWriteProgress {
                complete: false,
                written: 3,
                failure: Some(BulkWriteFailure::Fatal),
            }
        );
    }

    /// `send_exec_exit` from host context suppresses the write.
    #[test]
    fn send_exec_exit_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_exec_exit(0)", || send_exec_exit(0));
        assert_no_bulk_write("send_exec_exit(-1)", || send_exec_exit(-1));
    }

    /// `send_dmesg` from host context suppresses the write.
    #[test]
    fn send_dmesg_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_dmesg", || {
            send_dmesg(b"[    0.000000] Linux version 6.16.0\n")
        });
    }

    /// `send_probe_output` from host context suppresses the write.
    #[test]
    fn send_probe_output_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        assert_no_bulk_write("send_probe_output", || send_probe_output(b"{}\n"));
    }

    /// `request_snapshot` from host context returns `TransportError`.
    #[test]
    fn request_snapshot_from_host_context_returns_transport_error() {
        let _g = IsGuestOverrideGuard::new(false);
        let r = request_snapshot(0, "tag", std::time::Duration::from_millis(0));
        match r {
            SnapshotRequestResult::TransportError { .. } => {}
            other => panic!("expected TransportError from host context, got {other:?}"),
        }
    }

    /// `request_kernel_op` from host context returns
    /// `TransportError` (mirrors `request_snapshot`'s host-context
    /// gate). The virtio-console port-1 transport is reachable only
    /// from inside the guest; a host-context call must not silently
    /// no-op or panic.
    #[test]
    fn request_kernel_op_from_host_context_returns_transport_error() {
        let _g = IsGuestOverrideGuard::new(false);
        let request = crate::vmm::wire::KernelOpRequestPayload {
            request_id: 0,
            mode: crate::vmm::wire::KernelOpMode::Hot,
            direction: crate::vmm::wire::KernelOpDirection::Write,
            tag: String::new(),
            entries: vec![],
        };
        let r = request_kernel_op(request, std::time::Duration::from_millis(0));
        match r {
            crate::vmm::wire::KernelOpRequestResult::TransportError { .. } => {}
            other => panic!("expected TransportError from host context, got {other:?}"),
        }
    }

    /// `read_bulk_port_frame` rejects a payload whose `length`
    /// exceeds the caller-supplied `max_payload_size` cap. Pins
    /// the parameterized cap introduced for the kernel-op reply
    /// path — a callers passes its own limit and the function
    /// must honour it, NOT the old hardcoded
    /// `size_of::<SnapshotReplyPayload>()` value.
    #[test]
    fn read_bulk_port_frame_respects_caller_supplied_cap() {
        use std::os::unix::io::FromRawFd;
        let mut fds = [0i32; 2];
        // SAFETY: standard pipe(2) call; fds is a valid &mut to a
        // 2-element i32 array. Returning <0 indicates failure.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(r, 0, "pipe(2) failed: {}", std::io::Error::last_os_error());
        // SAFETY: pipe(2) just returned the fds; both are open and
        // owned by this scope. From_raw_fd takes ownership so the
        // File closes them on drop.
        let mut read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let mut write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        // Frame a header with length = 200 but cap at 100. The
        // function must reject WITHOUT reading the (forged) payload.
        let header = ShmMessage {
            msg_type: MSG_TYPE_KERNEL_OP_REPLY,
            length: 200,
            crc32: 0,
            _pad: 0,
        };
        use std::io::Write;
        write_end
            .write_all(header.as_bytes())
            .expect("write forged header");
        drop(write_end);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let err = read_bulk_port_frame(&mut read_end, 100, Some(deadline))
            .expect_err("cap=100 must reject length=200");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("exceeds max payload 100"),
            "error must cite the caller-supplied cap, got: {msg}"
        );
    }

    /// `read_bulk_port_frame` must reject a header whose `length`
    /// exceeds `size_of::<SnapshotReplyPayload>()` BEFORE allocating
    /// the payload buffer. A hostile or corrupted host could otherwise
    /// frame `length = u32::MAX` and cause `vec![0u8; u32::MAX]` to
    /// OOM the guest's PID 1 init, panicking the kernel.
    #[test]
    fn read_bulk_port_frame_rejects_oversized_length_before_alloc() {
        use std::os::unix::io::FromRawFd;
        // Build a pipe, write a forged 16-byte header with
        // length = u32::MAX, then call read_bulk_port_frame on the
        // read side. The function must return InvalidData without
        // attempting to read or allocate the (huge) payload.
        let mut fds = [0i32; 2];
        // SAFETY: standard pipe(2) call; fds is a valid &mut to a
        // 2-element i32 array. Returning <0 indicates failure.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(r, 0, "pipe(2) failed: {}", std::io::Error::last_os_error());
        // SAFETY: pipe(2) just returned the fds; both are open and
        // owned by this scope. From_raw_fd takes ownership so the
        // File closes them on drop.
        let mut read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let mut write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };

        let header = ShmMessage {
            msg_type: MSG_TYPE_SNAPSHOT_REPLY,
            length: u32::MAX,
            crc32: 0,
            _pad: 0,
        };
        use std::io::Write;
        write_end
            .write_all(header.as_bytes())
            .expect("write forged header");
        // Drop the writer so the reader observes EOF after the
        // header rather than blocking forever on the missing payload.
        drop(write_end);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let err = read_bulk_port_frame(
            &mut read_end,
            std::mem::size_of::<SnapshotReplyPayload>(),
            Some(deadline),
        )
        .expect_err("oversized length must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("exceeds max payload"),
            "error must explain the cap, got: {msg}"
        );
    }

    /// `read_bulk_port_frame` must accept a length that exactly
    /// matches `size_of::<SnapshotReplyPayload>()` — the cap is an
    /// upper bound, not a strict-less-than check. This pins the
    /// boundary so a future tightening of the cap would force a
    /// deliberate test update rather than silently breaking the
    /// snapshot-reply path.
    #[test]
    fn read_bulk_port_frame_accepts_exact_max_payload() {
        use std::os::unix::io::FromRawFd;
        let mut fds = [0i32; 2];
        // SAFETY: pipe(2) on a freshly-zeroed 2-element i32 array.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(r, 0, "pipe(2) failed: {}", std::io::Error::last_os_error());
        // SAFETY: pipe just returned both fds; ownership transfers
        // to the File handles which close on drop.
        let mut read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let mut write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };

        let payload = vec![0u8; std::mem::size_of::<SnapshotReplyPayload>()];
        let header = ShmMessage {
            msg_type: MSG_TYPE_SNAPSHOT_REPLY,
            length: payload.len() as u32,
            crc32: crc32fast::hash(&payload),
            _pad: 0,
        };
        use std::io::Write;
        write_end.write_all(header.as_bytes()).expect("header");
        write_end.write_all(&payload).expect("payload");
        drop(write_end);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let (msg_type, body) = read_bulk_port_frame(
            &mut read_end,
            std::mem::size_of::<SnapshotReplyPayload>(),
            Some(deadline),
        )
        .expect("exact-size payload must succeed");
        assert_eq!(msg_type, MSG_TYPE_SNAPSHOT_REPLY);
        assert_eq!(body.len(), std::mem::size_of::<SnapshotReplyPayload>());
    }

    /// An ordered lifecycle fence has no local wall-clock deadline:
    /// it must remain blocked until the host supplies the frame instead
    /// of turning host descheduling into a transport timeout.
    #[test]
    fn read_bulk_port_frame_without_deadline_waits_for_host_frame() {
        use std::io::Write;
        use std::os::unix::io::FromRawFd;

        let mut fds = [0i32; 2];
        // SAFETY: standard pipe(2) call with a valid two-fd output array.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(r, 0, "pipe(2) failed: {}", std::io::Error::last_os_error());
        // SAFETY: pipe(2) returned two owned descriptors; each transfers
        // to exactly one File and is closed on drop.
        let mut read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let mut write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        // Match the production bulk-port open disposition. The no-deadline
        // reader must provide blocking behavior through poll, not by relying
        // on a blocking file descriptor.
        let flags = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
        assert!(
            flags >= 0,
            "F_GETFL failed: {}",
            std::io::Error::last_os_error()
        );
        let set = unsafe { libc::fcntl(fds[0], libc::F_SETFL, flags | libc::O_NONBLOCK) };
        assert_eq!(
            set,
            0,
            "F_SETFL failed: {}",
            std::io::Error::last_os_error()
        );

        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let result = read_bulk_port_frame(&mut read_end, 0, None);
            result_tx.send(result).expect("publish read result");
        });

        assert!(
            result_rx
                .recv_timeout(std::time::Duration::from_millis(25))
                .is_err(),
            "no-deadline read returned before the host frame arrived"
        );

        let header = ShmMessage {
            msg_type: MsgType::TeardownBarrierAck.wire_value(),
            length: 0,
            crc32: crc32fast::hash(&[]),
            _pad: 0,
        };
        write_end.write_all(header.as_bytes()).expect("write ack");
        drop(write_end);

        let (msg_type, payload) = result_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("blocking reader did not wake for host frame")
            .expect("read host frame");
        reader.join().expect("reader thread panicked");
        assert_eq!(msg_type, MsgType::TeardownBarrierAck.wire_value());
        assert!(payload.is_empty());
    }

    #[test]
    fn is_guest_override_round_trips_through_thread_local() {
        // Toggling override should affect is_guest() result.
        {
            let _g = IsGuestOverrideGuard::new(false);
            assert!(!is_guest());
        }
        {
            let _g = IsGuestOverrideGuard::new(true);
            assert!(is_guest());
        }
    }

    #[test]
    fn is_guest_override_guards_nest_correctly() {
        let _outer = IsGuestOverrideGuard::new(true);
        assert!(is_guest());
        {
            let _inner = IsGuestOverrideGuard::new(false);
            assert!(!is_guest());
        }
        // Inner dropped — outer's value is restored.
        assert!(is_guest());
    }

    /// send_test_result's overflow branch selection (the AGGREGATE payload bound
    /// across non-merging per-cgroup-per-step carriers). Drives the pure
    /// classify_test_result at each size class and pins verdict integrity: a PASS
    /// with only oversized samples is degraded (samples dropped, verdict kept),
    /// NEVER flipped to a synthetic FAIL.
    #[test]
    fn classify_test_result_selects_branch_by_size() {
        use crate::assert::{AssertResult, PhaseBucket, PhaseCgroupStats};
        let mk = |n: u64| {
            let mut pc = std::collections::BTreeMap::new();
            pc.insert(
                "cg".to_string(),
                PhaseCgroupStats {
                    wake_latencies_ns: (0..n).collect(),
                    wake_sample_total: n,
                    ..Default::default()
                },
            );
            let mut r = AssertResult::pass();
            r.stats.phases = vec![PhaseBucket {
                step_index: 1,
                label: "Step[0]".to_string(),
                start_ms: 0,
                end_ms: 1,
                sample_count: 0,
                metrics: std::collections::BTreeMap::new(),
                per_cgroup: pc,
            }];
            r
        };
        let r = mk(1000);
        let full = postcard::to_stdvec(&r).unwrap().len();

        // (a) fits as-is -> Raw.
        match classify_test_result(&r, full) {
            Some(TestResultWire::Raw(b)) => assert_eq!(b.len(), full),
            other => panic!("expected Raw, got {other:?}"),
        }
        // (b) sample_free <= max < full -> Stripped: verdict PRESERVED, samples gone.
        let max = full - 1;
        match classify_test_result(&r, max) {
            Some(TestResultWire::Stripped { bytes, dropped }) => {
                assert_eq!(dropped, 1000);
                assert!(bytes.len() <= max);
                let decoded: AssertResult = postcard::from_bytes(&bytes).unwrap();
                assert!(decoded.is_pass(), "verdict PRESERVED — no PASS->FAIL flip");
                assert!(
                    decoded.stats.phases[0].per_cgroup["cg"]
                        .wake_latencies_ns
                        .is_empty(),
                    "only the samples were dropped",
                );
            }
            other => panic!("expected Stripped, got {other:?}"),
        }
        // (c) even the sample-free verdict overruns -> Truncated. The reported
        // `offending` is the POST-strip (sample-free, incl. the dropped-samples
        // info note) size, NOT the pre-strip original — so it is strictly less
        // than `full`. (max=1 guarantees the stripped result still overruns.)
        match classify_test_result(&r, 1) {
            Some(TestResultWire::Truncated { offending }) => {
                assert!(offending > 1, "offending overran max=1");
                assert!(
                    offending < full,
                    "offending {offending} is the post-strip size, not the pre-strip original {full}",
                );
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
        // (d) no samples to drop + over max -> Truncated, offending == original size.
        let ns = mk(0);
        let ns_full = postcard::to_stdvec(&ns).unwrap().len();
        match classify_test_result(&ns, ns_full - 1) {
            Some(TestResultWire::Truncated { offending }) => assert_eq!(offending, ns_full),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }
}
