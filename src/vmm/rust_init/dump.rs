//! sched_ext dump capture, log forwarding, trace-pipe, and the sched-exit monitor.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

/// Maximum scheduler-log chunk emitted in a single
/// [`crate::vmm::guest_comms::send_sched_log`] frame. Sub-cap of
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably inside one TLV frame; chunks above this size are
/// split before emission.
const SCHED_LOG_CHUNK_BYTES: usize = 64 * 1024;

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
pub(crate) fn dump_sched_output(log_path: &str) {
    // The child's output reaches the log file through the live-stream
    // forwarder threads; every dump caller runs after the child was
    // reaped, but the forwarders may still be draining the final pipe
    // bytes. Wait (bounded) so the dumped file — and the live
    // SchedStdout/SchedStderr streams — carry the complete output.
    super::scheduler::wait_sched_forwarders_drained(log_path);
    crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_START.as_bytes());
    send_sched_log_file(log_path);
    crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_END.as_bytes());
}

/// Walk `/tmp/sched_*.log` and emit each non-empty file as a
/// separate `SCHED_OUTPUT_START` / `SCHED_OUTPUT_END` frame.
/// Captures logs from Op-spawned schedulers (Attach, Replace,
/// Restart) that the boot path's single `dump_sched_output` call
/// cannot reach. Sorted by filename so the host sees logs in
/// spawn order (the monotonic seq suffix from
/// `staged_scheduler_log_path` guarantees lexicographic = temporal).
pub(crate) fn dump_staged_scheduler_logs() {
    let Ok(entries) = fs::read_dir("/tmp") else {
        return;
    };
    let mut paths: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("sched_") && n.ends_with(".log"))
        })
        .collect();
    paths.sort();
    for p in paths {
        if let Some(s) = p.to_str() {
            dump_sched_output(s);
        }
    }
}

/// Read the scheduler log file and emit it to the host as one or
/// more [`crate::vmm::wire::MsgType::SchedLog`] TLV chunks bounded
/// by [`SCHED_LOG_CHUNK_BYTES`]. Empty / missing file is a silent
/// no-op (mirrors the prior `dump_file_to_com2` behaviour where an
/// `Err` from `read_to_string` skipped the dump rather than
/// emitting a partial marker pair).
fn send_sched_log_file(path: &str) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let bytes = content.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        let end = (start + SCHED_LOG_CHUNK_BYTES).min(bytes.len());
        crate::vmm::guest_comms::send_sched_log(&bytes[start..end]);
        start = end;
    }
}

/// Send a fixed text snippet (e.g. a "failed to spawn" diagnostic)
/// to the host as a single [`crate::vmm::wire::MsgType::SchedLog`]
/// TLV chunk. The snippet is bounded by `SCHED_LOG_CHUNK_BYTES`
/// like every other chunk; oversized snippets would be rejected
/// by the host-side per-frame cap and are guarded here by
/// truncating the input before the call.
pub(crate) fn send_sched_log_text(s: &str) {
    let bytes = s.as_bytes();
    let cap = SCHED_LOG_CHUNK_BYTES.min(bytes.len());
    crate::vmm::guest_comms::send_sched_log(&bytes[..cap]);
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
) -> Option<Arc<AtomicBool>> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    std::thread::Builder::new()
        .name("hvc0-poll".into())
        .spawn(move || {
            hvc0_poll_loop(&stop_clone, trace_stop.as_deref(), probe_drain.as_ref());
        })
        .ok();

    Some(stop)
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
    stop: &AtomicBool,
    trace_stop: Option<&AtomicBool>,
    probe_drain: Option<&super::scheduler::ProbeDrain>,
) {
    use std::os::unix::io::AsRawFd;

    // Open the virtio-console wake fd. Failure here used to be
    // `.expect()`d, which panicked the worker thread; the
    // process-wide panic hook installed at PID-1 entry calls
    // `force_reboot()`, so a transient open failure (e.g. devtmpfs
    // not yet populated when the thread spawns) tore the VM down
    // before any test could dispatch. Log + return instead so the
    // poll loop simply doesn't deliver wake bytes for this boot —
    // tests that rely on `bpf_map_write` notification will time out
    // on their `wait_for_map_write` latch with a recoverable error
    // instead of a forced reboot.
    let hvc0 = match fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(HVC0)
    {
        Ok(f) => f,
        Err(e) => {
            write_com2(&format!(
                "ktstr-init: hvc0 poll loop disabled — open {HVC0}: {e}"
            ));
            return;
        }
    };
    // /dev/hvc0 is an hvc tty whose kernel-default termios has ICANON
    // set. N_TTY canonical mode only makes input visible to
    // poll(POLLIN)/read() on a line boundary: __receive_buf returns
    // before the EPOLLIN wake when `icanon && !L_EXTPROC`, and
    // input_available_p reports readable only once canon_head advances
    // (on a `\n`/EOL). The host's wake bytes (SIGNAL_VC_DUMP,
    // SIGNAL_BPF_WRITE_DONE, SIGNAL_ACCESSOR_READY, SIGNAL_VC_SHUTDOWN)
    // are single non-newline bytes, so in canonical mode they sit in the
    // canon buffer and never wake this loop. hvc registers
    // TTY_DRIVER_REAL_RAW, but n_tty_set_termios forces real_raw=0 while
    // ICANON is set, so the driver flag is inert until userspace clears
    // ICANON. Put the fd in raw mode (clearing ICANON) so every byte is
    // immediately poll-visible — mirrors the COM2 raw-mode setup in
    // `modes.rs`. The host-side virtio-console delivery is already
    // correct (RX descriptors are posted at probe, the byte is written
    // and the guest IRQ is raised); this gate is purely the guest
    // reader's line discipline, so the fix belongs here, not host-side.
    // A tcgetattr/tcsetattr failure is logged rather than swallowed:
    // without raw mode the loop reverts to canonical gating and silently
    // misses single-byte wakes — the exact silent hang this exists to
    // kill — so the failure must be diagnosable.
    match nix::sys::termios::tcgetattr(&hvc0) {
        Ok(mut tio) => {
            nix::sys::termios::cfmakeraw(&mut tio);
            if let Err(e) =
                nix::sys::termios::tcsetattr(&hvc0, nix::sys::termios::SetArg::TCSANOW, &tio)
            {
                write_com2(&format!(
                    "ktstr-init: hvc0 raw-mode tcsetattr failed: {e}; poll loop \
                     stays canonical and may miss single-byte wake signals"
                ));
            }
        }
        Err(e) => write_com2(&format!(
            "ktstr-init: hvc0 raw-mode tcgetattr failed: {e}; poll loop stays \
             canonical and may miss single-byte wake signals"
        )),
    }
    let poll_timeout_ms: PollTimeout = 1000u16.into();

    while !stop.load(Ordering::Acquire) {
        let borrowed = unsafe { BorrowedFd::borrow_raw(hvc0.as_raw_fd()) };
        let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
        match poll(&mut fds, poll_timeout_ms) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
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
                break;
            }
            if !revents.contains(PollFlags::POLLIN) {
                if revents.contains(PollFlags::POLLHUP) {
                    break;
                }
                continue;
            }
        }
        let mut buf = [0u8; 16];
        let mut hvc_ref: &fs::File = &hvc0;
        // Retry on EINTR (the read was interrupted by a signal before
        // returning data). The previous `unwrap_or(0)` collapsed both
        // EINTR and EIO into 0 bytes, masking transient signal races
        // (drops a real wake byte) and permanent device errors (silent
        // hang in the next poll iteration). Treat:
        //   - Ok(n): consume n bytes and dispatch signals below. An
        //     `Ok(0)` here is rare (poll already confirmed POLLIN)
        //     but harmless — the byte-contains checks no-op and the
        //     outer loop iterates normally, same as the original
        //     `unwrap_or(0)` behaviour for that case.
        //   - EINTR: retry the read inline; poll already confirmed
        //     POLLIN, so the wake byte is still in the device's RX
        //     queue waiting to be drained.
        //   - other Err: log via tracing::warn and break the outer
        //     poll loop. A non-EINTR read error after POLLIN means
        //     the device is in an unrecoverable state (host-side
        //     disconnect that didn't surface as POLLHUP, kernel-side
        //     I/O error, fd revoked) and continuing would either
        //     spin on the same error or silently miss every wake
        //     byte for the rest of the run.
        let n = 'read_retry: loop {
            match hvc_ref.read(&mut buf) {
                Ok(n) => break 'read_retry Some(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: hvc0 read failed; aborting poll loop"
                    );
                    break 'read_retry None;
                }
            }
        };
        let Some(n) = n else { break };
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP) {
            let _ = fs::write("/proc/sysrq-trigger", "D");
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE) {
            bpf_map_write_done_latch().set();
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_ACCESSOR_READY) {
            accessor_ready_latch().set();
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_VC_SHUTDOWN) {
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
            break;
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

/// Stop handle for the sched-exit monitor. Carries the
/// `Arc<AtomicBool>` source-of-truth flag, a writable eventfd handle
/// the cleanup site uses to wake the monitor thread out of `poll(2)`
/// without waiting for the legacy 250 ms cadence, and the monitor
/// thread's `JoinHandle` so the cleanup site can wait for the
/// thread to actually exit before proceeding.
///
/// Cleanup contract: before any action that could be misinterpreted
/// by the monitor as an unexpected scheduler exit (e.g. `child.kill()`
/// on the scheduler), the cleanup site MUST call
/// [`SchedExitStop::stop_and_join`] (or its equivalent of
/// `store(true, Release)` + [`SchedExitStop::wake`] + joining the
/// thread). Otherwise the monitor races: it sees `/proc/{pid}` gone
/// after the kill, takes the `if exited` branch, and emits
/// `MSG_TYPE_SCHED_EXIT` to the host, which terminates the VM
/// before the orderly `MSG_TYPE_EXIT` frame can be sent.
///
/// The bool is the source of truth; the eventfd write delivers the
/// edge that pulls the thread out of an indefinite `poll`. The
/// eventfd is owned by this struct on the writer side and by the
/// monitor thread on the reader side; both sides drop their fds when
/// the run ends, so the kernel-side counter is reclaimed cleanly.
pub(crate) struct SchedExitStop {
    /// Stop flag the monitor thread polls under `Acquire` ordering at
    /// every loop iteration. Setting `true` is the only way to make
    /// the thread exit through its top-of-loop early-return arm; the
    /// eventfd below is the wake-edge that pairs with this store.
    pub(crate) stop: Arc<AtomicBool>,
    /// Owned eventfd write side. `wake()` writes `1` here; the
    /// monitor's `poll(2)` returns within microseconds. `None` when
    /// `eventfd(2)` failed at monitor spawn (legacy 250 ms timeout
    /// still bounds wake latency in that degraded path).
    wake_fd: Option<OwnedFd>,
    /// Monitor thread join handle. `None` when
    /// `std::thread::Builder::spawn` failed (the monitor never
    /// started; nothing to join). Consumed by
    /// [`SchedExitStop::stop_and_join`].
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl SchedExitStop {
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
    /// joins the thread. After this returns, the monitor has
    /// observed `stop=true` at the top of its loop and exited
    /// without sending `MSG_TYPE_SCHED_EXIT` — making it safe for
    /// the caller to proceed with actions (like killing the
    /// scheduler child) that the monitor would otherwise interpret
    /// as an unexpected scheduler exit.
    ///
    /// `JoinHandle::join` propagates a panic from the monitor closure
    /// as `Err`; it is consumed and ignored — a panicked monitor is
    /// already dead and there is no recovery path during teardown.
    pub(crate) fn stop_and_join(self) {
        self.stop.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.join_handle {
            let _ = handle.join();
        }
    }
}

/// Monitor the scheduler child process for unexpected exit.
///
/// Blocks the monitor thread in `poll(2)` against the scheduler's
/// pidfd plus a stop-eventfd; the wait returns when either the
/// child exits (pidfd POLLIN edge from the kernel's `do_notify_pidfd`)
/// or the cleanup site fires the stop-eventfd. `/proc/{pid}` is
/// re-checked post-wake to catch the rare "pidfd opened after kernel
/// reaped" race. On the scheduler's exit two things are gated
/// independently:
///   - `suppress_sched_log` gates the scheduler-log dump only: false
///     (normal mode) dumps the log over the bulk port (`dump_sched_output`
///     -> `send_sched_log`); true (probes active) instead waits on the
///     probe thread's `output_done` latch — keeping the VM alive until the
///     probe has emitted its payload — and skips the dump (the probe
///     pipeline handles crash detection via tp_btf/sched_ext_exit).
///   - the SCHED_EXIT signal (MSG_TYPE_SCHED_EXIT, which lets the host
///     terminate the VM early) is then sent UNLESS the `stop` flag is set
///     (a host-initiated kill, where the exit is expected). It is gated by
///     `stop`, NOT by `suppress_sched_log` — so with probes active and a
///     genuine crash it still fires, but only after the `output_done` wait
///     above, so the probe payload is already out.
///
/// Uses procfs instead of waitpid because SIGCHLD is SIG_IGN (the kernel
/// auto-reaps children, making waitpid return ECHILD).
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
/// Returns None when no scheduler is running.
pub(crate) fn start_sched_exit_monitor(
    sched_pid: Option<u32>,
    log_path: Option<&str>,
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) -> Option<SchedExitStop> {
    let pid = sched_pid?;
    let proc_path = format!("/proc/{pid}");
    let log_path = log_path.map(|s| s.to_string());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

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

    let join_handle = std::thread::Builder::new()
        .name("sched-exit-mon".into())
        .spawn(move || {
            // pidfd_open lets us block on SIGCHLD-equivalent
            // notification for the scheduler process exit instead
            // of polling /proc/{pid} on a sleep cadence.
            // SAFETY: pid is the scheduler's stable pid for the
            // run; pidfd_open(2) accepts any process the caller
            // can signal (we are pid 1). pidfd_open has been
            // available since kernel 5.3 (2019); ktstr targets
            // 6.16+ where it is unconditionally present, so the
            // procfs fallback is dead code. A failure here means
            // the kernel rejected the syscall entirely (sandbox /
            // seccomp filter); abort the monitor rather than
            // fabricate a polling fallback that hides the
            // configuration error.
            let pidfd = unsafe {
                libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int
            };
            if pidfd < 0 {
                tracing::error!(
                    pid,
                    err = %std::io::Error::last_os_error(),
                    "ktstr-init: pidfd_open failed for sched — sched exit monitor disabled",
                );
                return;
            }
            // The monitor-side stop fd's raw value, or `-1` when the
            // caller's eventfd allocation or dup failed. `-1` in a
            // pollfd entry is valid: the kernel ignores the slot
            // (returns revents=0), so the same `poll(2)` call works
            // on the degraded path with a finite timeout that
            // re-checks `stop` periodically.
            let stop_fd = monitor_fd.as_ref().map(|f| f.as_raw_fd()).unwrap_or(-1);
            // Poll timeout policy: when the stop eventfd is live
            // (`stop_fd >= 0`), a stop request fires the eventfd
            // edge and the wait returns within microseconds — so an
            // indefinite `-1` timeout is correct; the loop never has
            // to wake just to re-check `stop`. When the eventfd
            // allocation degraded to `None`, the legacy 250 ms
            // cadence is the only path that pulls the thread out
            // of the wait, so we fall back to that timeout.
            let poll_timeout: i32 = if stop_fd >= 0 { -1 } else { 250 };
            while !stop_clone.load(Ordering::Acquire) {
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
                    // Re-checking proc_path post-`poll` is a
                    // belt-and-suspenders against the rare
                    // "pidfd was opened but the kernel reaped
                    // before we entered poll" race — an exited
                    // child's pidfd POLLIN may already be latched
                    // by the time we add it to the poll set;
                    // checking proc_path independently catches
                    // that case.
                    let mut pfds = [
                        libc::pollfd {
                            fd: pidfd,
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
                    // kernel skips that slot. Return value not
                    // consulted — the loop re-checks the stop
                    // flag and the proc path each iteration
                    // regardless.
                    let _ = unsafe {
                        libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, poll_timeout)
                    };
                    !Path::new(&proc_path).exists()
                };
                if exited {
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
                            done.wait();
                        }
                    } else if let Some(ref path) = log_path {
                        dump_sched_output(path);
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
                        unsafe {
                            libc::close(pidfd);
                        }
                        return;
                    }
                    let exit_code: i32 = 1;
                    crate::vmm::guest_comms::send_sched_exit(exit_code);
                    // SAFETY: pidfd is owned by this thread
                    // and is no longer used after close.
                    unsafe {
                        libc::close(pidfd);
                    }
                    // `monitor_fd` (Option<OwnedFd>) drops here on
                    // function return — the OwnedFd's Drop closes
                    // the read side of the stop eventfd. The
                    // writer-side `OwnedFd` lives on the
                    // SchedExitStop returned to the caller.
                    return;
                }
                // Drain any pending stop-eventfd reads so the next
                // `poll` doesn't immediately re-fire on the same
                // edge. The `stop` AtomicBool is the source of
                // truth (re-checked at the top of the loop); the
                // eventfd is purely a wake-edge, so a missed read
                // is benign — the next iteration's poll wakes
                // either way. EAGAIN under EFD_NONBLOCK (counter
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
            // SAFETY: same as above — close on exit path.
            unsafe {
                libc::close(pidfd);
            }
            // `monitor_fd` drops here as the closure returns.
        })
        .ok();

    Some(SchedExitStop {
        stop,
        wake_fd: writer_fd,
        join_handle,
    })
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
    use super::drain_probe_for_shutdown;
    use crate::sync::Latch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

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
