//! The sched-stats socket relay (guest stats socket -> host port).
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

/// Path of the scheduler-stats Unix socket inside the guest. Owned
/// by the running scx_* scheduler binary (created via
/// `scx_utils::stats::ScxStatsServer`). Empty when no scheduler is
/// running.
const SCHED_STATS_SOCKET: &str = "/var/run/scx/root/stats";

/// Path of the guest-side stats relay's port-2 device node. The
/// kernel virtio-console driver creates this when the multiport
/// PORT_NAME control message lands ahead of PORT_OPEN; see
/// [`crate::vmm::wire::PORT2_NAME`].
const SCHED_STATS_PORT_DEV: &str = "/dev/vport0p2";

/// Per-iteration scratch buffer size. Matches
/// [`crate::vmm::sched_stats::MAX_REQUEST_BYTES`] (256 KiB) so a
/// single legitimate request or response fits in one read. Larger
/// payloads span multiple loop iterations.
const RELAY_BUFFER_BYTES: usize = 256 * 1024;

/// Parent directory of the scheduler-stats Unix socket. The relay
/// creates this directory if it doesn't exist (the scheduler
/// userspace creates it before bind, but we may race) and watches
/// it via inotify for the `stats` socket file's `IN_CREATE` event.
const SCHED_STATS_SOCKET_DIR: &str = "/var/run/scx/root";

/// File name (final component) of the scheduler-stats Unix socket
/// inside [`SCHED_STATS_SOCKET_DIR`]. Matched against
/// [`nix::sys::inotify::InotifyEvent::name`] entries to detect
/// the scheduler's bind without polling.
const SCHED_STATS_SOCKET_NAME: &str = "stats";

/// Inline JSON error response the relay writes back to the host
/// when it has not yet connected to (or has lost connection to)
/// the scheduler's Unix socket. The host's
/// [`crate::vmm::sched_stats::SchedStatsClient`] parses the
/// `ktstr_relay_error` field into a typed
/// [`crate::vmm::sched_stats::SchedStatsError::NoScheduler`]. The
/// trailing `\n` matches scx_stats's line-delimited wire format.
const SCHED_STATS_RELAY_NO_SCHEDULER_REPLY: &[u8] =
    b"{\"ktstr_relay_error\":\"no scheduler available\"}\n";

/// Stop signal for the scheduler-stats relay thread. Carries an
/// `AtomicBool` source-of-truth flag plus an `EventFd` wake fd so
/// callers in phase-6 cleanup can interrupt a relay that is parked
/// in `poll(2)` without waiting for any timeout. The relay
/// registers the eventfd in its poll set and re-checks the
/// AtomicBool at every wake.
pub(crate) struct RelayStopSignal {
    flag: Arc<AtomicBool>,
    evt: Arc<vmm_sys_util::eventfd::EventFd>,
}

impl RelayStopSignal {
    /// Flip the source-of-truth flag and write the eventfd. The
    /// flag is set with `Release` before the fd write so a relay
    /// that wakes on the eventfd edge observes `true` on its
    /// `Acquire` load. Errors from the eventfd write are silently
    /// ignored — the AtomicBool is authoritative and a saturated
    /// counter (or torn fd) just means the relay's next natural
    /// wake re-checks the flag.
    pub(crate) fn signal_stop(&self) {
        self.flag.store(true, Ordering::Release);
        let _ = self.evt.write(1);
    }
}

/// Spawn the scheduler-stats relay thread.
///
/// Event-driven design: the relay opens [`SCHED_STATS_PORT_DEV`]
/// once (no retry — `redirect_stdio_to_bulk_port` already proved
/// the multiport handshake completed by the time this is called),
/// then runs an outer loop that:
///
/// 1. Waits for the scheduler's Unix socket to appear via inotify
///    (no sleep loop).
/// 2. Connects, then poll(2)s on the port fd, the socket fd, and
///    the stop eventfd. On port→socket data arriving, forwards
///    the bytes; on socket→host data, forwards back; on socket
///    EOF/error, writes the inline error envelope to the port and
///    falls back to inotify wait; on stop, returns.
///
/// Returns a [`RelayStopSignal`] the caller flips on teardown.
/// The thread is detached; the kernel reboot path tears down both
/// device nodes synchronously, so the relay exits when its
/// blocking I/O returns EBADF/EOF.
pub(crate) fn start_sched_stats_relay() -> RelayStopSignal {
    use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
    let flag = Arc::new(AtomicBool::new(false));
    let evt = match EventFd::new(EFD_NONBLOCK) {
        Ok(e) => Arc::new(e),
        Err(err) => {
            tracing::error!(
                error = %err,
                "stats relay: eventfd create failed; relay disabled \
                 (host SchedStatsClient calls will hang on shutdown)"
            );
            // Return a flag-only signal; the relay never spawns.
            return RelayStopSignal {
                flag,
                evt: Arc::new(EventFd::new(0).unwrap_or_else(|_| {
                    // Last-resort: try without EFD_NONBLOCK. If
                    // even this fails, the host is in a degraded
                    // state where no relay can run anyway.
                    panic!("stats relay: cannot create any eventfd")
                })),
            };
        }
    };
    let flag_for_thread = flag.clone();
    let evt_for_thread = evt.clone();
    let _ = std::thread::Builder::new()
        .name("ktstr-sched-stats-relay".into())
        .spawn(move || {
            sched_stats_relay_loop(flag_for_thread, evt_for_thread);
        });
    RelayStopSignal { flag, evt }
}

/// Inner loop for the stats relay thread. Opens the port-2 device
/// node once (single open — the multiport handshake completed
/// before this function was called) and drives the outer
/// inotify-wait → connect → poll-relay-session cycle until `stop`
/// flips.
/// Maximum consecutive `PortEof` returns from the inner functions
/// (`wait_for_stats_socket` and `run_relay_session`) we tolerate
/// before declaring the virtio-console port dead and exiting the
/// relay thread. Any non-`PortEof` exit (`RelaySessionExit::Other`,
/// `WaitSocketResult::Connected`, `WaitSocketResult::Stopped`)
/// resets the counter.
///
/// B14: the stats-port reader can return Ok(0) when the host
/// hasn't connected its end of `/dev/vport0p2` yet, when the host
/// closes its console connection, or when the kernel virtio-console
/// driver hits a transient disconnect. A single Ok(0) is recoverable
/// (the inner functions exit cleanly, the outer loop re-arms via
/// inotify and the scheduler/host can re-establish the link).
/// But a port that's permanently closed produces back-to-back Ok(0)
/// returns indefinitely — re-arming inotify, getting woken by the
/// race-free initial probe (which can succeed against a still-bound
/// socket file even though the port itself is dead), running a
/// micro-session that immediately exits on Ok(0), and looping. This
/// busy-loop wastes CPU and produces a log flood. After three
/// consecutive Ok(0) returns the relay thread exits — the host
/// loses scheduler-stats relay (no automatic recovery) but the
/// guest's CPU bill stops.
const SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF: u32 = 3;

/// Return value of [`run_relay_session`] / [`wait_for_stats_socket`]
/// signalling why the inner function exited so the outer
/// [`sched_stats_relay_loop`] can count consecutive port EOFs and
/// bail when the virtio-console port is persistently dead. See
/// [`SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF`] for the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelaySessionExit {
    /// `port.read` returned Ok(0). Counts toward the consecutive
    /// EOF budget the outer loop tracks.
    PortEof,
    /// Any other clean exit (socket EOF, scheduler error, stop_evt
    /// fired). Resets the consecutive EOF counter.
    Other,
}

fn sched_stats_relay_loop(stop: Arc<AtomicBool>, stop_evt: Arc<vmm_sys_util::eventfd::EventFd>) {
    let mut port = match fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(SCHED_STATS_PORT_DEV)
    {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = SCHED_STATS_PORT_DEV,
                "stats relay: open vport0p2 failed; relay disabled"
            );
            return;
        }
    };

    // B14: count consecutive `port.read` Ok(0) outcomes from the
    // inner functions. A single Ok(0) is recoverable; after
    // `SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF` in a row we
    // assume the virtio-console port is permanently dead and exit.
    let mut consecutive_port_eof: u32 = 0;

    // Outer loop: wait for socket via inotify, connect, run
    // session, fall back to inotify on socket failure. Stops only
    // when stop_evt fires (signal_stop flipped the flag and woke
    // every blocked syscall).
    while !stop.load(Ordering::Acquire) {
        let wait_exit = wait_for_stats_socket(&mut port, &stop, &stop_evt);
        match wait_exit {
            WaitSocketResult::Connected(socket) => {
                // A successful connect refreshes the
                // consecutive-EOF budget. Without this reset, a
                // run of inotify-wait Ok(0)s could leave the
                // counter near the cap; if the next session
                // happens to return PortEof once it would push
                // past the cap and exit even though the port
                // proved live enough to deliver a connect-edge
                // and run a session.
                consecutive_port_eof = 0;
                let exit = run_relay_session(&mut port, socket, &stop, &stop_evt);
                match exit {
                    RelaySessionExit::PortEof => {
                        consecutive_port_eof += 1;
                    }
                    RelaySessionExit::Other => {
                        consecutive_port_eof = 0;
                    }
                }
                if consecutive_port_eof >= SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF {
                    tracing::warn!(
                        consecutive_port_eof,
                        "stats relay: vport0p2 returned Ok(0) on \
                         {SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF} consecutive \
                         relay sessions — assuming the port is permanently dead and \
                         exiting the relay thread to avoid a busy-loop"
                    );
                    return;
                }
            }
            WaitSocketResult::PortEof => {
                consecutive_port_eof += 1;
                if consecutive_port_eof >= SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF {
                    tracing::warn!(
                        consecutive_port_eof,
                        "stats relay: vport0p2 returned Ok(0) on \
                         {SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF} consecutive \
                         inotify-wait drains — assuming the port is permanently \
                         dead and exiting the relay thread to avoid a busy-loop"
                    );
                    return;
                }
                // Continue the outer loop to re-arm inotify; the
                // count keeps climbing until it hits the cap or a
                // non-EOF event resets it.
            }
            WaitSocketResult::Stopped => {
                // wait_for_stats_socket returned None only when
                // stop flipped or inotify itself errored. Either
                // way, exit.
                return;
            }
        }
    }
}

/// Result of [`wait_for_stats_socket`]: distinguishes a successful
/// connect from the two clean-exit paths so the outer loop can
/// classify them correctly. B14: `PortEof` (port read returned
/// Ok(0)) feeds the consecutive-EOF counter; `Stopped` (stop_evt
/// fired or inotify errored) terminates the loop unconditionally.
enum WaitSocketResult {
    /// Scheduler socket connected; the relay can run a session.
    Connected(std::os::unix::net::UnixStream),
    /// `port.read` returned Ok(0) while waiting for the scheduler
    /// to bind. Counts toward the outer loop's consecutive-EOF
    /// budget — see
    /// [`SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF`].
    PortEof,
    /// `stop_evt` fired or inotify itself errored — exit
    /// unconditionally.
    Stopped,
}

/// Block (event-driven) until the scheduler's Unix socket exists,
/// then connect and return the stream. Uses inotify on the parent
/// directory to receive a `IN_CREATE` event when the scheduler
/// binds. Returns `Stopped` when `stop_evt` fires or inotify
/// itself errors out, `PortEof` when the host-side port read
/// reports Ok(0).
///
/// Race-free initial check: after setting up the watch, attempt
/// to connect once. If the socket already exists (scheduler
/// finished binding before we created the watch) the connect
/// succeeds and we return without ever reading from inotify.
fn wait_for_stats_socket(
    port: &mut std::fs::File,
    stop: &Arc<AtomicBool>,
    stop_evt: &Arc<vmm_sys_util::eventfd::EventFd>,
) -> WaitSocketResult {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
    use std::ffi::OsStr;
    use std::os::unix::io::AsFd;

    // Best-effort: ensure the parent directory exists so the
    // inotify watch can attach. The scheduler creates this
    // directory before bind, but we may race; pre-creating is
    // idempotent.
    let _ = fs::create_dir_all(SCHED_STATS_SOCKET_DIR);

    let inotify = match Inotify::init(InitFlags::IN_CLOEXEC | InitFlags::IN_NONBLOCK) {
        Ok(i) => i,
        Err(e) => {
            tracing::warn!(error = %e, "stats relay: inotify_init failed");
            return WaitSocketResult::Stopped;
        }
    };
    // B2 fix: include IN_ATTRIB so a chmod-on-listen (some
    // schedulers tighten perms after listen()) wakes us; include
    // IN_OPEN so any client that successfully connects (including
    // ourselves on a retry) re-fires the watch even if the
    // initial CREATE-then-connect race already lost. The broader
    // mask catches more edges than IN_CREATE alone, so a connect
    // that fails with ECONNREFUSED post-CREATE has additional
    // events to wake on rather than wedging.
    if let Err(e) = inotify.add_watch(
        SCHED_STATS_SOCKET_DIR,
        AddWatchFlags::IN_CREATE
            | AddWatchFlags::IN_MOVED_TO
            | AddWatchFlags::IN_ATTRIB
            | AddWatchFlags::IN_OPEN,
    ) {
        tracing::warn!(
            error = %e,
            dir = SCHED_STATS_SOCKET_DIR,
            "stats relay: inotify add_watch failed"
        );
        return WaitSocketResult::Stopped;
    }

    // Race-free initial probe: socket may already exist before the
    // watch was added. Try connect; on success skip the loop.
    if stop.load(Ordering::Acquire) {
        return WaitSocketResult::Stopped;
    }
    if let Ok(s) = std::os::unix::net::UnixStream::connect(SCHED_STATS_SOCKET) {
        tracing::debug!("stats relay: connected to scheduler socket (race-free initial probe)");
        return WaitSocketResult::Connected(s);
    }

    // Park on poll(inotify_fd, port_fd, stop_evt). Each wake:
    //   - inotify edge: re-read events; on any event in the
    //     watched dir, retry connect (the B2 expanded mask plus
    //     this any-event retry policy guards against the
    //     IN_CREATE-then-listen() race that left the prior code
    //     waiting on a CREATE-only edge that never came again).
    //   - port edge: B3 fix — host pushed a request before the
    //     scheduler came up. Drain it and reply with the inline
    //     error envelope so the host's request_raw wakes
    //     immediately with NoScheduler instead of waiting for the
    //     scheduler to appear.
    //   - stop_evt edge: shutdown.
    let target = OsStr::new(SCHED_STATS_SOCKET_NAME);
    let mut buf = vec![0u8; RELAY_BUFFER_BYTES];
    loop {
        if stop.load(Ordering::Acquire) {
            return WaitSocketResult::Stopped;
        }
        let inotify_fd = inotify.as_fd();
        let port_fd = port.as_fd();
        // SAFETY: `stop_evt` is held by the surrounding `Arc`, so
        // the raw fd is valid for the whole loop body.
        let stop_evt_fd =
            unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stop_evt.as_raw_fd()) };
        let mut fds = [
            PollFd::new(inotify_fd, PollFlags::POLLIN),
            PollFd::new(port_fd, PollFlags::POLLIN),
            PollFd::new(stop_evt_fd, PollFlags::POLLIN),
        ];
        match poll(&mut fds, PollTimeout::NONE) {
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(e) => {
                tracing::warn!(error = %e, "stats relay: poll on inotify failed");
                return WaitSocketResult::Stopped;
            }
        };
        let inotify_ready = fds[0]
            .revents()
            .is_some_and(|r| r.contains(PollFlags::POLLIN));
        let port_ready = fds[1]
            .revents()
            .is_some_and(|r| r.contains(PollFlags::POLLIN));
        let stop_ready = fds[2]
            .revents()
            .is_some_and(|r| r.contains(PollFlags::POLLIN));

        // Stop-fd ready? Drain and exit. The only writer is
        // `RelayStopSignal::signal_stop`.
        if stop_ready {
            let _ = stop_evt.read();
            return WaitSocketResult::Stopped;
        }

        // B3: host pushed a request while we're still waiting for
        // the scheduler. Drain whatever bytes are available and
        // reply with the inline error envelope so the request
        // surfaces NoScheduler immediately. A burst that exceeds
        // RELAY_BUFFER_BYTES gets one error reply per drain; the
        // host's request_raw will see the first envelope and
        // return — subsequent envelopes are harmless because
        // they sit in the response_buf as stale bytes that the
        // next request clears.
        if port_ready {
            match port.read(&mut buf) {
                Ok(0) => {
                    // B14: this Ok(0) feeds the outer loop's
                    // consecutive-EOF counter via the PortEof
                    // return. A single Ok(0) is recoverable
                    // (the outer re-arms inotify); after the
                    // configured cap the relay thread exits.
                    tracing::debug!(
                        "stats relay: port read EOF in inotify wait; \
                         returning to outer loop for EOF accounting"
                    );
                    return WaitSocketResult::PortEof;
                }
                Ok(n) => {
                    tracing::debug!(
                        bytes = n,
                        "stats relay: host pushed request while waiting for scheduler; \
                         emitting no-scheduler error envelope"
                    );
                    if let Err(e) = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY) {
                        tracing::warn!(
                            error = %e,
                            "stats relay: port write failed in inotify wait; exiting"
                        );
                        return WaitSocketResult::Stopped;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "stats relay: port read error in inotify wait; exiting"
                    );
                    return WaitSocketResult::Stopped;
                }
            }
        }

        // inotify fd ready: drain events and try connect. B2:
        // try connect on ANY event in the watched directory, not
        // just IN_CREATE for our target — the bind-without-listen
        // window means a CREATE-only check would miss the
        // listen-edge that follows.
        if inotify_ready {
            let events = match inotify.read_events() {
                Ok(e) => e,
                Err(nix::errno::Errno::EINTR) => continue,
                Err(nix::errno::Errno::EAGAIN) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "stats relay: inotify read_events failed");
                    return WaitSocketResult::Stopped;
                }
            };
            // If any event names our target — or if any event
            // names anything (the dir's only legitimate occupant
            // is our socket plus possible peer scheduler files) —
            // attempt connect. The connect itself is the
            // synchronisation primitive: ECONNREFUSED means we'll
            // wait for the next inotify edge.
            let saw_target_or_any = events
                .iter()
                .any(|ev| ev.name.as_deref() == Some(target) || ev.name.is_some());
            if !saw_target_or_any {
                continue;
            }
            match std::os::unix::net::UnixStream::connect(SCHED_STATS_SOCKET) {
                Ok(s) => {
                    tracing::debug!("stats relay: connected to scheduler socket via inotify edge");
                    return WaitSocketResult::Connected(s);
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "stats relay: socket appeared but connect failed (likely \
                         bind-without-listen race); will retry on next inotify edge"
                    );
                }
            }
        }
    }
}

/// On socket loss, drain whatever request bytes the host has
/// already pushed onto port 2 and answer each readable batch with
/// the inline error envelope. Without this, B12: a request that
/// the host wrote AFTER we forwarded the prior request to the
/// (now-dead) socket would otherwise be carried over into the
/// next relay session, where it would be forwarded to a fresh
/// scheduler — meaning the host's old request gets answered by
/// the new scheduler's stats, not by an error.
///
/// Uses non-blocking poll-with-zero-timeout via PollTimeout::ZERO
/// to drain only what's already queued, then returns. Each drained
/// batch gets one error envelope; the host's request_raw observes
/// the first envelope and surfaces NoScheduler — the rest sit as
/// stale bytes in response_buf that the next request clears.
fn drain_port_emit_errors(port: &mut std::fs::File) {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use std::io::ErrorKind;
    use std::os::unix::io::AsFd;

    let mut buf = vec![0u8; RELAY_BUFFER_BYTES];
    loop {
        let port_ready = {
            let port_fd = port.as_fd();
            let mut fds = [PollFd::new(port_fd, PollFlags::POLLIN)];
            match poll(&mut fds, PollTimeout::ZERO) {
                Ok(_) => fds[0]
                    .revents()
                    .is_some_and(|r| r.contains(PollFlags::POLLIN)),
                Err(_) => false,
            }
        };
        if !port_ready {
            break;
        }
        match port.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => {
                if port
                    .write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY)
                    .is_err()
                {
                    break;
                }
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

/// Run a single port-↔-socket relay session. Returns
/// [`RelaySessionExit::PortEof`] when `port.read` returned Ok(0)
/// (the outer loop counts these toward the busy-loop budget — see
/// [`SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF`]) and
/// [`RelaySessionExit::Other`] for every other clean exit (socket
/// EOF, scheduler error, stop_evt fired, port write error). Uses
/// poll(2) on (port_fd, socket_fd, stop_evt) so the thread blocks
/// in the kernel until exactly one of those fds is readable — no
/// spinning, no timeouts, and `stop_evt` interrupts any blocked
/// I/O within microseconds.
///
/// Single-thread serialization: the relay is the only writer and
/// the only reader of `/dev/vport0p2` inside the guest, so no
/// userspace mutex around the port fd is required. scx_stats
/// requests are strictly request/response on a single socket
/// connection — no req-id multiplexing — so the natural ordering
/// of the relay's per-iteration loop (read host → write socket
/// → read socket → write host) preserves the protocol semantics.
fn run_relay_session(
    port: &mut std::fs::File,
    mut socket: std::os::unix::net::UnixStream,
    stop: &Arc<AtomicBool>,
    stop_evt: &Arc<vmm_sys_util::eventfd::EventFd>,
) -> RelaySessionExit {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use std::io::ErrorKind;
    use std::os::unix::io::AsFd;

    let mut buf = vec![0u8; RELAY_BUFFER_BYTES];
    // B6: track socket health across poll iterations. Set false the
    // moment POLLHUP/POLLERR is observed on the socket fd; gate
    // every `socket.write_all` on this flag so we never write into
    // a HUP'd socket (which fails with EPIPE/SIGPIPE and surfaces
    // as a noisy error path). When POLLHUP and POLLIN both arrive
    // in the same poll, drain the buffered POLLIN data first so the
    // host sees the scheduler's last response before we declare
    // the session dead — the kernel keeps already-queued data
    // readable across the half-close, so reading after POLLHUP
    // is well-defined.
    let mut socket_healthy = true;

    while !stop.load(Ordering::Acquire) {
        // Wait for one of: host pushed bytes (port readable),
        // scheduler emitted bytes (socket readable), or shutdown
        // (stop_evt readable). Wrap the poll call in an inner
        // scope so the `fds` array (and the immutable borrows on
        // port + socket it holds) drops before we try to read or
        // write either of them.
        let (port_ready, socket_in, socket_hup_seen, stop_ready) = {
            let port_fd = port.as_fd();
            let socket_fd = socket.as_fd();
            // SAFETY: `stop_evt` is held by the surrounding `Arc`,
            // so the raw fd is valid for the whole inner scope.
            let stop_evt_fd =
                unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stop_evt.as_raw_fd()) };
            let mut fds = [
                PollFd::new(port_fd, PollFlags::POLLIN),
                PollFd::new(socket_fd, PollFlags::POLLIN),
                PollFd::new(stop_evt_fd, PollFlags::POLLIN),
            ];
            match poll(&mut fds, PollTimeout::NONE) {
                Ok(_) => {}
                Err(nix::errno::Errno::EINTR) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "stats relay: poll failed; exiting session");
                    return RelaySessionExit::Other;
                }
            }
            // Snapshot the revents and drop the borrows.
            let port_rev = fds[0].revents();
            let socket_rev = fds[1].revents();
            let stop_rev = fds[2].revents();
            let port_ready = port_rev.is_some_and(|r| r.contains(PollFlags::POLLIN));
            let socket_in = socket_rev.is_some_and(|r| r.contains(PollFlags::POLLIN));
            // B6: any POLLHUP or POLLERR on the socket — with or
            // without POLLIN — is a permanent transition to
            // unhealthy. POLLHUP+POLLIN means buffered data is
            // still drainable; the same-iteration drain of the
            // socket POLLIN happens below after `socket_healthy`
            // is flipped, because reading from a HUP'd socket with
            // buffered data is well-defined — only WRITES need the
            // gate.
            let socket_hup_seen = socket_rev
                .is_some_and(|r| r.contains(PollFlags::POLLHUP) || r.contains(PollFlags::POLLERR));
            let stop_ready = stop_rev.is_some_and(|r| r.contains(PollFlags::POLLIN));
            (port_ready, socket_in, socket_hup_seen, stop_ready)
        };

        // Flip `socket_healthy` to false IMMEDIATELY when
        // POLLHUP/POLLERR is observed in the current iteration —
        // before any port-read processing. Earlier code flipped
        // the flag at the END of the loop body, which raced when
        // POLLHUP and port-POLLIN arrived in the same revents:
        // the port arm at `if !socket_healthy` saw stale `true`
        // and forwarded the host's request into the HUP'd socket
        // (EPIPE / SIGPIPE). The socket-POLLIN drain below still
        // runs because reading buffered scheduler responses
        // across a half-close remains well-defined; only the
        // WRITE side needs gating.
        if socket_hup_seen {
            socket_healthy = false;
        }

        // Stop edge: drain and exit.
        if stop_ready {
            let _ = stop_evt.read();
            return RelaySessionExit::Other;
        }

        // Host→guest port readable: read bytes and forward to
        // socket. The socket forward is a blocking write — bounded
        // by the kernel's Unix-socket buffer, not by any user
        // timeout. B6: skip the write_all entirely when the socket
        // is already known unhealthy (POLLHUP seen on a prior
        // iteration). Reading the port still drains the host's
        // queued request bytes so they don't pile up in the kernel
        // buffer; we answer with the inline error envelope and
        // exit so the host's pending request_raw wakes with
        // NoScheduler instead of timing out.
        if port_ready {
            let n = match port.read(&mut buf) {
                Ok(0) => {
                    // B14: this Ok(0) is the busy-loop trigger —
                    // surface it through the typed return so the
                    // outer `sched_stats_relay_loop` can count
                    // consecutive port-EOF exits and bail when the
                    // budget is exhausted.
                    tracing::debug!(
                        "stats relay: port read EOF; returning to outer loop \
                         for EOF accounting"
                    );
                    return RelaySessionExit::PortEof;
                }
                Ok(n) => n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "stats relay: port read error; exiting session");
                    return RelaySessionExit::Other;
                }
            };
            if !socket_healthy {
                tracing::debug!(
                    bytes = n,
                    "stats relay: port→socket forward skipped (socket already \
                     unhealthy); emitting error envelopes and reconnecting"
                );
                let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                drain_port_emit_errors(port);
                return RelaySessionExit::Other;
            }
            if let Err(e) = socket.write_all(&buf[..n]) {
                tracing::debug!(
                    error = %e,
                    "stats relay: socket write failed; emitting error envelopes and reconnecting"
                );
                // B12: the host may have additional queued
                // requests on the port that we haven't read yet —
                // the failed write_all means we're abandoning the
                // socket without forwarding them. Answer the
                // request that triggered this write_all PLUS any
                // already-queued follow-up requests with error
                // envelopes so they don't survive into the next
                // session and get forwarded to a fresh scheduler.
                // No need to mutate `socket_healthy` here — every
                // arm that reaches a write-failure path returns
                // immediately and the local goes out of scope.
                let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                drain_port_emit_errors(port);
                return RelaySessionExit::Other;
            }
        }

        // Scheduler→host socket readable: read response bytes and
        // forward to port. B6: read POLLIN data even when POLLHUP
        // arrived in the same poll — buffered scheduler responses
        // remain readable across the half-close until the kernel
        // socket buffer drains. `socket_healthy` was already
        // flipped at the top of the loop body if POLLHUP/POLLERR
        // appeared in the same revents; the `!socket_healthy`
        // reconnect block below catches that case after this drain.
        if socket_in {
            let m = match socket.read(&mut buf) {
                Ok(0) => {
                    tracing::debug!(
                        "stats relay: socket EOF; emitting error envelopes and reconnecting"
                    );
                    let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                    drain_port_emit_errors(port);
                    return RelaySessionExit::Other;
                }
                Ok(m) => m,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "stats relay: socket read error; emitting error envelopes and reconnecting"
                    );
                    let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                    drain_port_emit_errors(port);
                    return RelaySessionExit::Other;
                }
            };
            if let Err(e) = port.write_all(&buf[..m]) {
                tracing::warn!(error = %e, "stats relay: port write failed; exiting session");
                return RelaySessionExit::Other;
            }
        }

        // With `socket_healthy` already flipped at the top
        // of the loop body when POLLHUP/POLLERR arrived, the
        // post-drain reconnect just checks the flag. Reaching
        // this point with `!socket_healthy` means either (a)
        // POLLHUP arrived alone — we exit immediately so we don't
        // spin re-arming poll on a dead fd; or (b) POLLHUP+POLLIN
        // arrived together — we drained the buffered scheduler
        // responses above and now exit. Both cases share the
        // reconnect path: emit the inline error envelope plus
        // drain any queued port requests.
        if !socket_healthy {
            tracing::debug!(
                drained_in = socket_in,
                "stats relay: socket POLLHUP/POLLERR; reconnecting after draining"
            );
            let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
            drain_port_emit_errors(port);
            return RelaySessionExit::Other;
        }
    }
    // Reached only when the outer `stop` flag is observed at the
    // top of the loop — an ordinary clean shutdown.
    RelaySessionExit::Other
}
