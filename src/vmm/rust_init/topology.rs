//! CPU-topology + online-CPU parsing, wprof spawn, and the sys-ready handshake.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

/// Print the topology line for the shell MOTD.
///
/// Parses KTSTR_TOPO=N,L,C,T from /proc/cmdline (passed by the host).
/// Falls back to counting online CPUs via /sys/devices/system/cpu/online.
pub(crate) fn print_topology_line() {
    if let Some((n, l, c, t)) = parse_topo_from_cmdline() {
        let total = l * c * t;
        if n > 1 {
            println!(
                "  topology:  {n} NUMA nodes, {l} LLC{}, {c} core{}, {t} thread{} ({total} vCPU{})",
                if l == 1 { "" } else { "s" },
                if c == 1 { "" } else { "s" },
                if t == 1 { "" } else { "s" },
                if total == 1 { "" } else { "s" },
            );
        } else {
            println!(
                "  topology:  {l} LLC{}, {c} core{}, {t} thread{} ({total} vCPU{})",
                if l == 1 { "" } else { "s" },
                if c == 1 { "" } else { "s" },
                if t == 1 { "" } else { "s" },
                if total == 1 { "" } else { "s" },
            );
        }
    } else if let Some(count) = count_online_cpus() {
        println!(
            "  topology:  {count} vCPU{}",
            if count == 1 { "" } else { "s" }
        );
    }
}

/// Parse KTSTR_TOPO=N,L,C,T from /proc/cmdline.
pub(crate) fn parse_topo_from_cmdline() -> Option<(u32, u32, u32, u32)> {
    let val = cmdline_val("KTSTR_TOPO")?;
    let parts: Vec<&str> = val.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let n: u32 = parts[0].parse().ok()?;
    let l: u32 = parts[1].parse().ok()?;
    let c: u32 = parts[2].parse().ok()?;
    let t: u32 = parts[3].parse().ok()?;
    Some((n, l, c, t))
}

#[cfg(feature = "wprof")]
const WPROF_READY_MARKER: &[u8] = b"Running...";

#[cfg(feature = "wprof")]
struct WprofStartupSignal {
    sender: Option<std::sync::mpsc::SyncSender<bool>>,
    suffix: Vec<u8>,
}

#[cfg(feature = "wprof")]
impl WprofStartupSignal {
    fn new(sender: std::sync::mpsc::SyncSender<bool>) -> Self {
        Self {
            sender: Some(sender),
            suffix: Vec::with_capacity(WPROF_READY_MARKER.len() - 1),
        }
    }

    fn observe(&mut self, bytes: &[u8]) {
        if self.sender.is_none() {
            return;
        }

        self.suffix.extend_from_slice(bytes);
        if self
            .suffix
            .windows(WPROF_READY_MARKER.len())
            .any(|window| window == WPROF_READY_MARKER)
        {
            if let Some(sender) = self.sender.take() {
                let _ = sender.send(true);
            }
            self.suffix.clear();
            return;
        }

        let keep = self.suffix.len().min(WPROF_READY_MARKER.len() - 1);
        self.suffix.drain(..self.suffix.len() - keep);
    }

    fn finish_unready(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(false);
        }
    }
}

#[cfg(feature = "wprof")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WprofWait {
    Exited,
    TimedOut,
    Failed,
}

#[cfg(feature = "wprof")]
fn set_nonblocking(fd: libc::c_int) -> std::io::Result<()> {
    // SAFETY: `fd` is the live ChildStderr owned by the calling thread.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: preserving the existing status flags and adding O_NONBLOCK is
    // valid for a pipe read end. The fd remains owned by ChildStderr.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(feature = "wprof")]
fn drain_wprof_stderr(
    child_stderr: &mut std::process::ChildStderr,
    guest_stderr: &mut std::io::Stderr,
    startup: &mut WprofStartupSignal,
) -> std::io::Result<()> {
    let mut buf = [0u8; 4096];
    loop {
        match child_stderr.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => {
                let bytes = &buf[..n];
                startup.observe(bytes);
                // Preserve the previous inherited-stderr behavior: wprof's
                // diagnostics remain visible on the guest console as they
                // arrive, including startup errors before the ready edge.
                let _ = guest_stderr.write_all(bytes);
                let _ = guest_stderr.flush();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(feature = "wprof")]
fn wait_wprof_evented(
    child: &mut std::process::Child,
    child_stderr: &mut std::process::ChildStderr,
    startup: &mut WprofStartupSignal,
    timeout: std::time::Duration,
) -> WprofWait {
    let pid = child.id() as libc::pid_t;
    // SAFETY: pidfd_open(2) on this thread's live child, flags 0. The returned
    // descriptor is wrapped in OwnedFd immediately below.
    let raw_pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0u32) as libc::c_int };
    if raw_pidfd < 0 {
        startup.finish_unready();
        return WprofWait::Failed;
    }
    // SAFETY: raw_pidfd is a newly-created owned descriptor and is consumed
    // exactly once by this OwnedFd.
    let pidfd = unsafe { OwnedFd::from_raw_fd(raw_pidfd) };

    if let Err(e) = set_nonblocking(child_stderr.as_raw_fd()) {
        eprintln!("ktstr: failed to make wprof stderr evented: {e}");
        startup.finish_unready();
        return WprofWait::Failed;
    }

    let deadline = std::time::Instant::now() + timeout;
    let mut guest_stderr = std::io::stderr();
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            startup.finish_unready();
            return WprofWait::TimedOut;
        }
        let remaining = deadline - now;
        let timeout_ms = remaining
            .as_millis()
            .saturating_add(u128::from(remaining.subsec_nanos() % 1_000_000 != 0))
            .min(i32::MAX as u128) as libc::c_int;
        let mut pollfds = [
            libc::pollfd {
                fd: child_stderr.as_raw_fd(),
                events: libc::POLLIN | libc::POLLHUP,
                revents: 0,
            },
            libc::pollfd {
                fd: pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: both pollfds borrow live descriptors for the duration of
        // this call. poll only writes their revents fields.
        let rc = unsafe { libc::poll(pollfds.as_mut_ptr(), pollfds.len() as _, timeout_ms) };
        if rc < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            eprintln!("ktstr: wprof event wait failed: {e}");
            startup.finish_unready();
            return WprofWait::Failed;
        }
        if rc == 0 {
            startup.finish_unready();
            return WprofWait::TimedOut;
        }

        let stderr_events = pollfds[0].revents;
        if stderr_events & (libc::POLLIN | libc::POLLHUP) != 0
            && let Err(e) = drain_wprof_stderr(child_stderr, &mut guest_stderr, startup)
        {
            eprintln!("ktstr: reading wprof stderr failed: {e}");
            startup.finish_unready();
            return WprofWait::Failed;
        }
        if stderr_events & libc::POLLNVAL != 0 {
            startup.finish_unready();
            return WprofWait::Failed;
        }

        let pidfd_events = pollfds[1].revents;
        if pidfd_events & (libc::POLLIN | libc::POLLHUP) != 0 {
            // A process can write its final stderr bytes immediately before
            // exit. Drain once more after pidfd readiness so the ready marker
            // cannot be lost when both edges coalesce into one poll wake.
            let _ = drain_wprof_stderr(child_stderr, &mut guest_stderr, startup);
            startup.finish_unready();
            return WprofWait::Exited;
        }
        if pidfd_events & (libc::POLLERR | libc::POLLNVAL) != 0 {
            startup.finish_unready();
            return WprofWait::Failed;
        }
    }
}

#[cfg(feature = "wprof")]
pub(crate) struct WprofCapture {
    startup: std::sync::mpsc::Receiver<bool>,
    startup_timeout: std::time::Duration,
    handle: std::thread::JoinHandle<Option<Vec<u8>>>,
}

#[cfg(feature = "wprof")]
impl WprofCapture {
    pub(crate) fn wait_until_ready(&self) -> bool {
        match self
            .startup
            .recv_timeout(self.startup_timeout + std::time::Duration::from_secs(1))
        {
            Ok(true) => true,
            Ok(false) => {
                eprintln!("ktstr: wprof exited or timed out before becoming ready");
                false
            }
            Err(e) => {
                eprintln!("ktstr: wprof readiness wait failed: {e}");
                false
            }
        }
    }

    pub(crate) fn join(self) -> std::thread::Result<Option<Vec<u8>>> {
        self.handle.join()
    }
}

#[cfg(feature = "wprof")]
/// Spawn `/bin/wprof` in a background thread if the host set
/// `KTSTR_WPROF_ARGS` on the kernel cmdline. Returns a capture handle
/// that exposes wprof's event-driven startup edge and whose `.join()`
/// yields `Some(Vec<u8>)` (the `.pb` trace bytes) on success, or `None`
/// on failure / no-op.
///
/// The spawned thread:
/// 1. Parses `KTSTR_WPROF_ARGS` from `/proc/cmdline`
/// 2. Runs `/bin/wprof <args> -T /tmp/wprof.pb -D /tmp/wprof.data`
/// 3. Waits for the process to exit
/// 4. Reads `/tmp/wprof.pb` and returns the bytes
///
/// If `KTSTR_WPROF_ARGS` is absent or `/bin/wprof` doesn't exist,
/// returns `None` (no thread spawned, no-op). The caller joins the
/// handle after the test workload dispatch returns and ships the
/// bytes via [`crate::vmm::guest_comms::send_wprof_trace`].
pub(crate) fn spawn_wprof_if_configured() -> Option<WprofCapture> {
    let args_str = cmdline_val("KTSTR_WPROF_ARGS")?;
    let wprof_bin = std::path::Path::new("/bin/wprof");
    if !wprof_bin.exists() {
        tracing::warn!("KTSTR_WPROF_ARGS set but /bin/wprof missing from initramfs");
        return None;
    }
    let mut cmd_args: Vec<String> = args_str.split('\x1f').map(String::from).collect();
    cmd_args.extend([
        "-T".to_string(),
        "/tmp/wprof.pb".to_string(),
        "-D".to_string(),
        "/tmp/wprof.data".to_string(),
    ]);
    let capture_ms = cmd_args
        .iter()
        .position(|a| a == "-d")
        .and_then(|i| cmd_args.get(i + 1))
        .and_then(|d| d.parse::<u64>().ok())
        .unwrap_or(500);
    let deadline =
        std::time::Duration::from_millis(capture_ms) + std::time::Duration::from_secs(10);
    let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel(1);
    let handle = std::thread::Builder::new()
            .name("wprof-capture".into())
            .spawn(move || {
                // Host encodes args with ASCII Unit Separator (\x1F)
                // via `WprofConfig::args_cmdline` because kernel
                // cmdline tokenization would truncate a space-joined
                // value at the first space. Split on the same
                // delimiter here to recover the per-arg vec.
                let mut startup = WprofStartupSignal::new(startup_tx);
                tracing::debug!(args = ?cmd_args, "spawning /bin/wprof");
                // Bounded wait for the wprof child. A wprof stranded by a
                // crashing sched_ext scheduler — notably the 6.14
                // bypass-drain stall after scx_bpf_error, where runnable
                // tasks (including wprof's own) are not reliably migrated to
                // fair — never reaches its capture-window exit, so a
                // blocking `.status()` here would wedge guest teardown: no
                // reboot, no wprof ship, no SCHED_EXIT, the guest just hangs
                // to the host watchdog. Spawn + wait on the child's pidfd up
                // to a deadline sized off the `-d` capture window (ms) plus a
                // generous processing margin, kept under the host
                // `WPROF_SHIP_GRACE` (30s) so the guest kills a wedged wprof
                // and reboots within the host's grace. On the cap the trace
                // is dropped (loudly) and teardown proceeds — a clean
                // failure, never a hang. Mirrors `reap_child_bounded`.
                let mut child = match std::process::Command::new("/bin/wprof")
                    .args(&cmd_args)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                {
                    Ok(c) => c,
                    Err(e) => {
                        startup.finish_unready();
                        tracing::warn!(%e, "spawn /bin/wprof failed");
                        return None;
                    }
                };
                let Some(mut child_stderr) = child.stderr.take() else {
                    startup.finish_unready();
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                };
                // Determine wprof's exit disposition, then ship the trace on
                // FILE-PRESENCE (a non-empty /tmp/wprof.pb) regardless of exit
                // disposition. wprof fflushes the COMPLETE .pb as the last step
                // before main returns, so a wprof that finished its post-window
                // emit but is slow or nonzero to EXIT — e.g. its userspace emit
                // thread lost CPU to the concurrent auto-repro probe pipeline
                // during the crash-bypass window — still yields a valid trace
                // (the host validates shape via assert_wprof_pb_shape). On a
                // timeout, SIGTERM first (wprof's handler sets its `exiting`
                // flag, letting an in-progress emit flush), then a bounded
                // secondary wait, then SIGKILL, so a nearly-done emit is not
                // truncated. This is the reliability backstop; it perturbs no
                // scheduling priority.
                let pid = child.id() as libc::pid_t;
                match wait_wprof_evented(
                    &mut child,
                    &mut child_stderr,
                    &mut startup,
                    deadline,
                ) {
                    WprofWait::Exited => {
                        let _ = child.wait();
                    }
                    WprofWait::TimedOut => {
                        // wprof did not exit within the window — its userspace
                        // emit thread lost CPU (e.g. to the concurrent auto-repro
                        // probe pipeline during the crash-bypass window). SIGTERM
                        // first: wprof's handler sets its `exiting` flag, letting
                        // an in-progress emit fflush the .pb, then reap with a
                        // bounded second wait, then hard-kill. The .pb read below
                        // still recovers a completed-but-slow-to-exit emit.
                        eprintln!(
                            "ktstr: wprof exceeded {deadline:?}; SIGTERM to let its \
                             emit flush, then reap"
                        );
                        // SAFETY: `pid` is this thread's own child; SIGTERM to a
                        // live pid is well-defined and ESRCH on an already-reaped
                        // pid is harmless (return value dropped).
                        unsafe {
                            libc::kill(pid, libc::SIGTERM);
                        }
                        match wait_wprof_evented(
                            &mut child,
                            &mut child_stderr,
                            &mut startup,
                            std::time::Duration::from_secs(8),
                        ) {
                            WprofWait::Exited => {
                                let _ = child.wait();
                            }
                            _ => {
                                let _ = child.kill();
                                let _ = child.wait();
                            }
                        }
                    }
                    WprofWait::Failed => {
                        if !matches!(child.try_wait(), Ok(Some(_))) {
                            let _ = child.kill();
                        }
                        let _ = child.wait();
                    }
                }
                match std::fs::read("/tmp/wprof.pb") {
                    Ok(bytes) if !bytes.is_empty() => {
                        tracing::debug!(pb_bytes = bytes.len(), "wprof trace captured");
                        Some(bytes)
                    }
                    Ok(_) => {
                        eprintln!("ktstr: wprof produced no trace this run (/tmp/wprof.pb empty)");
                        None
                    }
                    Err(e) => {
                        eprintln!("ktstr: wprof produced no trace this run (/tmp/wprof.pb absent/unreadable: {e})");
                        None
                    }
                }
            })
            .expect("spawn wprof-capture thread");
    Some(WprofCapture {
        startup: startup_rx,
        startup_timeout: deadline,
        handle,
    })
}

#[cfg(all(test, feature = "wprof"))]
mod wprof_startup_tests {
    use super::*;

    #[test]
    fn ready_marker_is_detected_across_read_boundaries() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut signal = WprofStartupSignal::new(tx);

        signal.observe(b"Preparing...\nRun");
        assert!(matches!(
            rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        signal.observe(b"ning...\n");
        assert_eq!(rx.recv().unwrap(), true);

        signal.finish_unready();
        assert!(matches!(
            rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn exit_before_ready_reports_failure() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut signal = WprofStartupSignal::new(tx);
        signal.observe(b"startup failed\n");
        signal.finish_unready();
        assert_eq!(rx.recv().unwrap(), false);
    }
}

/// Loop-iteration counter for [`send_sys_rdy_with_retry`], bumped once
/// per retry iteration. Exists so a test can pin that the fast-fail
/// retry path THROTTLES (bounded iterations) rather than hot-spinning:
/// with the guard-rail sleep, a port-exists + always-failing-send run
/// makes roughly `budget / 100ms` iterations, whereas a regression that
/// dropped the throttle would make thousands. Mirrors the
/// observability-counter pattern of
/// [`crate::vmm::guest_comms::BULK_PORT_WRITE_ATTEMPTS`], including its
/// `#[cfg(test)]` gating: the counter and its increment compile only
/// under test, never into the production guest-init binary.
#[cfg(test)]
pub(crate) static SEND_SYS_RDY_RETRY_ITERS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Wait (event-driven) for the virtio-console bulk port, then deliver
/// the KERN_ADDRS + SYS_RDY frames to the host.
///
/// The kernel virtio_console driver's multiport handshake
/// (DEVICE_READY → PORT_ADD → PORT_READY → PORT_OPEN, see
/// `drivers/char/virtio_console.c`) completes asynchronously in two
/// stages this function waits on WITHOUT polling:
///
/// 1. **Node appearance.** `/dev/vport0p1` is created by `add_port`'s
///    `device_create` when PORT_ADD arrives. We block on
///    `kernfs_evented_wait` over an inotify watch of the port's
///    parent directory (`IN_CREATE` / `IN_MOVED_TO`) so we wake on the
///    exact devtmpfs create edge, with a 1 s guard-rail cadence
///    bounding wake latency if the best-effort inotify source misses.
///    This replaces a former 100 ms existence poll whose stacked sleep
///    latency on a cold / oversubscribed boot delayed KERN_ADDRS (and
///    thus the host's BPF-accessor build) well past the workload.
///
/// 2. **Host-connected.** `host_connected` flips true only when
///    PORT_OPEN arrives later. The send path's blocking writev is
///    itself kernel-evented: it parks in `wait_port_writable` →
///    `wait_event_freezable(port->waitqueue, !will_write_block)` (no
///    timeout) until PORT_OPEN's `wake_up_interruptible` fires. So no
///    poll/sleep is needed for stage 2 — `send_kern_addrs` /
///    `send_sys_rdy` simply block until writable. (`poll(POLLOUT)` is
///    NOT usable: `port_fops_poll` reports `EPOLLHUP` while
///    `!host_connected`, so a POLLOUT wait would busy-return.)
///
/// The deadline is captured INSIDE the function so guest-init setup
/// (mounts, kallsyms reads) does not eat the handshake's budget.
///
/// Both KERN_ADDRS and SYS_RDY are required for the host. KERN_ADDRS
/// is latched: once a `send_kern_addrs` call returns true the retry
/// skips it. The early-return condition is `kern_addrs_sent &&
/// send_sys_rdy()` — we never exit until BOTH have been delivered (a
/// successful sys_rdy on a re-opened FD after a kern_addrs failure must
/// not leave kern_addrs unsent, because the host's KERN_ADDRS arm is
/// the only virt-KASLR publisher on aarch64; see
/// `src/vmm/freeze_coord/dispatch.rs`'s KERN_ADDRS handler). A send
/// that fails re-enters stage 1 — whose fast-path returns immediately
/// while the node exists — and retries, bounded by the deadline. Most
/// send failures block first (the writev parks in `wait_port_writable`
/// until PORT_OPEN), but the open-error and post-connect fast-error
/// paths (a `try_open` failure, or an ENOMEM / add-outbuf error after
/// host_connected) return WITHOUT parking, so a bounded throttle guards
/// the retry against hot-spinning the guest init thread (matching the
/// sibling `send_sched_swap_notify` / `send_scenario_start` loops).
///
/// Host idempotency for KERN_ADDRS retries (matters when the latch
/// is reset by a failed write that cleared the cached FD):
/// `kern_phys_base` uses `.store(Release)` (overwrites every
/// CRC-valid frame) and `kern_virt_kaslr` uses CAS-once. The
/// payload bytes are identical across retries (built once from
/// `KernAddrs::new`), so repeated stores and a CAS-success-then-
/// no-op-on-equal-existing both produce the same final state.
///
/// On budget exhaustion (or a kernel with no evented source available
/// for stage 1) the function emits a structured WARN with fields
/// `budget_ms`, `vcpus`, `elapsed_ms` (loop wall time), `port_exists`
/// (sampled once before WARN), and `kern_addrs_sent`. The guest then
/// continues — the host monitor's `data_valid` gate keeps reads safe
/// without SYS_RDY, and the freeze coordinator's `Option::take` makes a
/// late SYS_RDY harmless (fire-once). See
/// `doc/guide/src/troubleshooting.md#send_sys_rdy-timeout` for the
/// operator-facing diagnosis flow.
pub(crate) fn send_sys_rdy_with_retry(
    budget: std::time::Duration,
    vcpus: u32,
    kern_addrs: &crate::vmm::wire::KernAddrs,
    port_path: &std::path::Path,
) {
    use crate::vmm::freeze_coord::evented_wait::{KernfsWaitOutcome, kernfs_evented_wait};
    use nix::sys::inotify::AddWatchFlags;

    let loop_t0 = std::time::Instant::now();
    let deadline = loop_t0 + budget;
    // Guard-rail cadence bounding wake latency ONLY if the best-effort
    // inotify source misses the device-create edge; the real wake is
    // the inotify event. Not a poll — see `kernfs_evented_wait`.
    let cadence = std::time::Duration::from_secs(1);
    // The port node lives directly under /dev; watch that directory for
    // its creation. `/dev` is the parent of `/dev/vport0p1`.
    let watch_dir = port_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("/dev"));

    // Structured WARN shared by the timeout and no-evented-source exits.
    // Snapshots `port_exists` so the field reports the last-attempt
    // state, not a fresh stat that could observe the port appearing in
    // the gap between the final wait and the WARN call.
    //
    // The budget bounds how long BOOT blocks on the handshake — it must
    // NOT cancel the publishes themselves. KERN_ADDRS is the host's only
    // virt-KASLR / phys_base publisher on aarch64; the previous
    // warn-and-abandon exit left the host's evidence channels dead for
    // the whole run on starved boots (observed on arm64/6.14: a narrow
    // cell whose 10.15 s budget expired mid-boot at a few % host CPU
    // share reached Teardown with `kaslr_raw=0`, while the wide sibling
    // — whose per-vCPU-scaled budget was ~2x larger — published fine in
    // the same leg). Every budget-exhaustion exit therefore hands the
    // unsent frames to [`spawn_background_publish_retry`], which keeps
    // the SAME evented loop going without a deadline: boot proceeds
    // immediately (unchanged blocking semantics) and the publish lands
    // whenever the port comes up — the host adopts it per monitor tick
    // (`RqRefresh::{kaslr_offset,kern_phys_base}` re-reads), so a late
    // publish still lights the evidence channels. The WARN remains the
    // operator signal that the budget was exceeded.
    let warn_timeout = |kern_addrs_sent: bool| {
        let port_exists_snapshot = port_path.exists();
        tracing::warn!(
            budget_ms = budget.as_millis() as u64,
            vcpus,
            elapsed_ms = loop_t0.elapsed().as_millis() as u64,
            port_exists = port_exists_snapshot,
            kern_addrs_sent,
            "ktstr-init: send_sys_rdy failed within boot budget; \
             boot continues and a detached background retry keeps \
             publishing KERN_ADDRS/SYS_RDY — \
             see https://ktstr.dev/guide/troubleshooting.html#send_sys_rdy-timeout",
        );
    };

    let mut kern_addrs_sent = false;
    let mut first_attempt_logged = false;
    // The booted kernel's build-id, read once (a boot constant). Sent
    // best-effort alongside KERN_ADDRS so the host can prove the vmlinux
    // it introspects is the same build as this running kernel; an empty
    // read (no sysfs / no note) simply skips the host-side check.
    let build_id = crate::vmm::guest_comms::read_kernel_build_id().unwrap_or_default();
    let mut build_id_sent = build_id.is_empty();
    loop {
        #[cfg(test)]
        SEND_SYS_RDY_RETRY_ITERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Stage 1: block (evented) until `/dev/vport0p1` appears. On a
        // send-failure retry the node already exists, so the fast-path
        // predicate returns immediately.
        match kernfs_evented_wait(
            watch_dir,
            AddWatchFlags::IN_CREATE | AddWatchFlags::IN_MOVED_TO,
            None::<&std::path::Path>,
            cadence,
            deadline,
            || port_path.exists().then_some(()),
        ) {
            KernfsWaitOutcome::Done(()) => {
                // Stage 2: the sends' blocking writev parks on the port
                // waitqueue until PORT_OPEN (host_connected) — kernel-
                // evented, no poll/sleep needed here.
                if !kern_addrs_sent {
                    // Pre-attempt crumb, once: `send_kern_addrs`'s blocking
                    // writev parks in the driver's `wait_port_writable`
                    // (no timeout) until the host's PORT_OPEN — a park that
                    // never wakes would swallow every post-attempt crumb, so
                    // "attempting" with no following outcome line IS the
                    // park-forever fingerprint.
                    if !first_attempt_logged {
                        crate::vmm::rust_init::console_breadcrumb(
                            "ktstr-init: attempting first KERN_ADDRS send",
                        );
                    }
                    kern_addrs_sent = crate::vmm::guest_comms::send_kern_addrs(kern_addrs);
                    // Publish the build-id once the port is writable (same
                    // gate as KERN_ADDRS). Best-effort: a failed write just
                    // retries next iteration; the host skips its check if it
                    // never arrives.
                    if !build_id_sent {
                        build_id_sent = crate::vmm::guest_comms::send_kern_build_id(&build_id);
                    }
                    // Send-side breadcrumb (port-independent serial channel —
                    // see `console_breadcrumb`): the first attempt's outcome,
                    // once. A field run with the wire + dispatcher exonerated
                    // (kill-time frames=0) left this segment unlit.
                    if !first_attempt_logged {
                        first_attempt_logged = true;
                        crate::vmm::rust_init::console_breadcrumb(if kern_addrs_sent {
                            "ktstr-init: KERN_ADDRS first send ok"
                        } else {
                            "ktstr-init: KERN_ADDRS first send FAILED; retrying"
                        });
                    }
                }
                if kern_addrs_sent && crate::vmm::guest_comms::send_sys_rdy() {
                    crate::vmm::rust_init::console_breadcrumb(
                        "ktstr-init: KERN_ADDRS + SYS_RDY delivered (foreground)",
                    );
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    warn_timeout(kern_addrs_sent);
                    crate::vmm::rust_init::console_breadcrumb(if kern_addrs_sent {
                        "ktstr-init: publish budget expired (KERN_ADDRS sent, SYS_RDY \
                         pending); background retry spawned"
                    } else {
                        "ktstr-init: publish budget expired with KERN_ADDRS UNSENT; \
                         background retry spawned"
                    });
                    spawn_background_publish_retry(kern_addrs_sent, *kern_addrs, port_path);
                    return;
                }
                // The send failed and reset the cached FD. Most send
                // failures block first (the writev parks in
                // wait_port_writable until PORT_OPEN), but the
                // open-error and post-connect fast-error paths (a
                // try_open failure, or an ENOMEM / add-outbuf error
                // after host_connected) return WITHOUT parking. Throttle
                // before retrying so a persistent fast-fail cannot
                // hot-spin the guest init thread — a bounded guard-rail
                // capped by the remaining budget, matching the sibling
                // bulk-port retry loops (send_sched_swap_notify /
                // send_scenario_start). Stage 1's node-appearance wait
                // stays fully evented.
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                std::thread::sleep(std::time::Duration::from_millis(100).min(remaining));
            }
            KernfsWaitOutcome::Timeout | KernfsWaitOutcome::NoEventedSource => {
                warn_timeout(kern_addrs_sent);
                crate::vmm::rust_init::console_breadcrumb(
                    "ktstr-init: publish budget expired waiting for the port node; \
                     background retry spawned",
                );
                spawn_background_publish_retry(kern_addrs_sent, *kern_addrs, port_path);
                return;
            }
        }
    }
}

/// Continue the KERN_ADDRS / SYS_RDY publish on a detached background
/// thread after [`send_sys_rdy_with_retry`]'s boot budget expired.
///
/// Rationale on the `warn_timeout` doc above: the budget bounds BOOT
/// blocking, not the publish. This loop mirrors the foreground one —
/// evented node wait, blocking sends, 1 s throttle between fast-fail
/// retries — with no deadline: it parks in the inotify wait / the
/// port's `wait_port_writable` queue between attempts, burning no CPU,
/// and exits on success. The thread dies with the VM otherwise (init
/// never reaps it; a still-parked sender at teardown is harmless — a
/// late SYS_RDY is fire-once on the host and KERN_ADDRS stores are
/// idempotent). No-op when both frames already went out.
///
/// `#[cfg(not(test))]`: the host-side unit tests drive
/// `send_sys_rdy_with_retry` against paths that never become ports and
/// assert on the retry-iteration counter; a detached forever-thread
/// would leak retries into unrelated assertions.
#[cfg(not(test))]
fn spawn_background_publish_retry(
    kern_addrs_sent: bool,
    kern_addrs: crate::vmm::wire::KernAddrs,
    port_path: &std::path::Path,
) {
    use crate::vmm::freeze_coord::evented_wait::{KernfsWaitOutcome, kernfs_evented_wait};
    use nix::sys::inotify::AddWatchFlags;

    let port_path = port_path.to_path_buf();
    let spawn_res = std::thread::Builder::new()
        .name("ktstr-publish-retry".into())
        .spawn(move || {
            let watch_dir = port_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("/dev"))
                .to_path_buf();
            let cadence = std::time::Duration::from_secs(1);
            let mut kern_addrs_sent = kern_addrs_sent;
            let retry_t0 = std::time::Instant::now();
            let mut attempts: u64 = 0;
            let mut last_crumb = retry_t0;
            loop {
                // Far-future deadline: the wait itself is evented and the
                // loop is exit-on-success; a finite horizon only guards
                // against Instant arithmetic overflow.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3600);
                match kernfs_evented_wait(
                    &watch_dir,
                    AddWatchFlags::IN_CREATE | AddWatchFlags::IN_MOVED_TO,
                    None::<&std::path::Path>,
                    cadence,
                    deadline,
                    || port_path.exists().then_some(()),
                ) {
                    KernfsWaitOutcome::Done(()) => {
                        attempts += 1;
                        if !kern_addrs_sent {
                            kern_addrs_sent = crate::vmm::guest_comms::send_kern_addrs(&kern_addrs);
                        }
                        if kern_addrs_sent && crate::vmm::guest_comms::send_sys_rdy() {
                            // Serial, not tracing/stdout: by now stdio rides
                            // the very bulk port under diagnosis.
                            crate::vmm::rust_init::console_breadcrumb(
                                "ktstr-init: background publish retry delivered \
                                 KERN_ADDRS + SYS_RDY",
                            );
                            return;
                        }
                        // Coarse-interval liveness breadcrumb (~30 s): the
                        // retry is alive and how many attempts it has burned
                        // — with the first-send and spawn crumbs this carries
                        // the whole send-side story to the host log. Coarse
                        // formatting only (attempt count bucketed) to keep
                        // the line cheap and the volume run-bounded.
                        if last_crumb.elapsed() >= std::time::Duration::from_secs(30) {
                            last_crumb = std::time::Instant::now();
                            let msg = if attempts < 10 {
                                "ktstr-init: publish retry alive (attempts < 10)"
                            } else if attempts < 100 {
                                "ktstr-init: publish retry alive (attempts 10-99)"
                            } else {
                                "ktstr-init: publish retry alive (attempts >= 100)"
                            };
                            crate::vmm::rust_init::console_breadcrumb(msg);
                        }
                        // Fast-fail throttle, mirroring the foreground loop.
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    KernfsWaitOutcome::Timeout | KernfsWaitOutcome::NoEventedSource => {
                        // Node still absent after an hour-scale horizon (or
                        // no inotify available): re-arm and keep waiting —
                        // the loop stays evented either way.
                    }
                }
            }
        });
    if let Err(e) = spawn_res {
        tracing::warn!(
            err = %e,
            "ktstr-init: could not spawn the background publish retry; \
             KERN_ADDRS/SYS_RDY stay unsent for this run"
        );
    }
}

/// Test builds: no background thread (see the `#[cfg(not(test))]`
/// sibling's doc); the WARN alone records the exhaustion.
#[cfg(test)]
fn spawn_background_publish_retry(
    _kern_addrs_sent: bool,
    _kern_addrs: crate::vmm::wire::KernAddrs,
    _port_path: &std::path::Path,
) {
}

/// Count online CPUs from /sys/devices/system/cpu/online.
///
/// The file contains a range list like "0-3" or "0-1,3". Parse and
/// count individual CPUs.
pub(crate) fn count_online_cpus() -> Option<u32> {
    let content = fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    parse_online_cpus(&content)
}

/// Parse a cpulist string (kernel `/sys/.../online` format) and
/// return the total count of CPUs it covers. Comma-separated tokens,
/// each either a single index or a `start-end` inclusive range.
/// Returns `None` on any unparseable token, inverted range, or
/// completely empty content. The `sys_rdy` budget caller at
/// [`count_online_cpus`]'s primary use defaults to 1 vCPU on `None`
/// (safe degradation to the single-vCPU budget); the topology-print
/// caller skips the MOTD line instead of substituting a default.
pub(crate) fn parse_online_cpus(content: &str) -> Option<u32> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut count = 0u32;
    for range in trimmed.split(',') {
        if let Some((start, end)) = range.split_once('-') {
            let s: u32 = start.parse().ok()?;
            let e: u32 = end.parse().ok()?;
            count = count.checked_add(e.checked_sub(s)?.checked_add(1)?)?;
        } else {
            let _: u32 = range.parse().ok()?;
            count = count.checked_add(1)?;
        }
    }
    Some(count)
}

/// Kernel cpulists are bounded by `CONFIG_NR_CPUS`. Reject any index or
/// range endpoint wildly larger so a corrupt sysfs read cannot balloon
/// the materialised id vec in [`parse_cpu_list`] — the parallel to the
/// `checked_add` overflow guard in [`parse_online_cpus`].
const CPU_ID_CEILING: u32 = 1 << 16;

/// Parse a cpulist string (kernel `/sys/.../{online,possible}` format)
/// into the explicit, ascending set of CPU IDs it covers. Same grammar
/// as [`parse_online_cpus`] (comma-separated single indices or inclusive
/// `start-end` ranges), but materialises the IDs instead of only
/// counting them, so callers can diff two lists. Returns `None` on any
/// unparseable token, inverted range, empty content, or an id/endpoint
/// at or beyond [`CPU_ID_CEILING`].
pub(crate) fn parse_cpu_list(content: &str) -> Option<Vec<u32>> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut ids = Vec::new();
    for range in trimmed.split(',') {
        if let Some((start, end)) = range.split_once('-') {
            let s: u32 = start.parse().ok()?;
            let e: u32 = end.parse().ok()?;
            if e < s || e >= CPU_ID_CEILING {
                return None;
            }
            ids.extend(s..=e);
        } else {
            let id: u32 = range.parse().ok()?;
            if id >= CPU_ID_CEILING {
                return None;
            }
            ids.push(id);
        }
    }
    Some(ids)
}

/// Outcome of diffing the kernel's `possible` and `online` cpulists.
pub(crate) struct OfflineCpus {
    /// Possible-but-offline CPU IDs, ascending. Empty when every
    /// possible CPU is online — the healthy case.
    pub missing: Vec<u32>,
    /// Total online CPUs.
    pub online: u32,
    /// Total possible CPUs.
    pub possible: u32,
}

/// Diff a `possible` cpulist against an `online` cpulist, returning the
/// possible-but-offline CPU IDs (ascending) plus both totals. `None`
/// when either list is unparseable — the caller degrades to skipping
/// the check rather than firing on a procfs hiccup.
pub(crate) fn offline_possible_cpus(possible: &str, online: &str) -> Option<OfflineCpus> {
    let possible = parse_cpu_list(possible)?;
    let online: std::collections::BTreeSet<u32> = parse_cpu_list(online)?.into_iter().collect();
    let missing: Vec<u32> = possible
        .iter()
        .copied()
        .filter(|c| !online.contains(c))
        .collect();
    Some(OfflineCpus {
        missing,
        online: online.len() as u32,
        possible: possible.len() as u32,
    })
}

/// Verify every *possible* CPU is *online* before the topology is handed
/// to the scheduler. On oversubscribed hosts an AP can miss its INIT-SIPI
/// alive-window (the kernel's `cpuhp_wait_for_sync_state` gives it 10 s of
/// guest WALL-CLOCK, which host stalls burn while the AP thread gets no
/// cycles) and land present-but-offline; a sched_ext scheduler then aborts
/// on the resulting CPU-ID gap with a cryptic error ("Holes in CPU IDs
/// detected") that surfaces as a scheduler failure rather than the infra
/// fault it is. Detection only — deliberately NO hotplug re-online here:
/// CPU hotplug is rare on real hardware, and a hotplug-assembled topology
/// could send scheduler developers chasing virtualization-only artifacts.
/// The caller's PANIC + reboot instead hands recovery to the host, which
/// can re-run the whole boot so the scheduler only ever sees a topology
/// assembled by a clean cold boot.
///
/// Returns `Err(message)` naming the missing CPUs and the online/possible
/// tally. Degrades to `Ok(())` when either sysfs list is missing or
/// unparseable — the same conservative stance as [`count_online_cpus`]'s
/// fallback. Every ktstr scenario boots with all vCPUs online (no
/// maxcpus / nr_cpus / offline path), so a non-empty gap is always a
/// fault.
pub(crate) fn all_possible_cpus_online() -> Result<(), String> {
    let (Ok(possible), Ok(online)) = (
        fs::read_to_string("/sys/devices/system/cpu/possible"),
        fs::read_to_string("/sys/devices/system/cpu/online"),
    ) else {
        return Ok(());
    };
    let Some(report) = offline_possible_cpus(&possible, &online) else {
        return Ok(());
    };
    if report.missing.is_empty() {
        return Ok(());
    }
    Err(format_ap_gap_message(
        &report.missing,
        report.online,
        report.possible,
    ))
}

/// Format the AP-bring-up-gap panic message that the caller PANICs with.
///
/// Built around [`crate::test_support::AP_BRINGUP_GAP_MARKER`] so the
/// host's boot-retry detection (`run_vm_with_ap_gap_retry`, which keys on
/// `crash_message.contains(MARKER)`) stays in lockstep with this format:
/// a reword here that dropped the marker would silently disable the
/// retry. Factored out of [`all_possible_cpus_online`] so a unit test can
/// pin that sync without a sysfs read.
pub(crate) fn format_ap_gap_message(missing: &[u32], online: u32, possible: u32) -> String {
    format!(
        "CPUs {missing:?} {}; {online}/{possible} online)",
        crate::test_support::AP_BRINGUP_GAP_MARKER,
    )
}

/// Print the include-files line for the shell MOTD.
///
/// Scans /include-files/ and lists each entry. Executable files
/// are marked with "(executable)".
pub(crate) fn print_includes_line() {
    let include_dir = Path::new("/include-files");
    if !include_dir.is_dir() {
        return;
    }
    let mut files: Vec<(String, bool)> = Vec::new();
    // Walk recursively to discover files in nested directories.
    for entry in walkdir::WalkDir::new(include_dir)
        .min_depth(1)
        .sort_by_file_name()
    {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(include_dir)
            .unwrap_or(entry.path());
        let name = rel.to_string_lossy().to_string();
        let executable = entry
            .metadata()
            .map(|m| {
                use std::os::unix::fs::PermissionsExt;
                m.permissions().mode() & 0o111 != 0
            })
            .unwrap_or(false);
        files.push((name, executable));
    }
    if files.is_empty() {
        return;
    }
    for (i, (name, executable)) in files.iter().enumerate() {
        let marker = if *executable { " (executable)" } else { "" };
        let path = format!("/include-files/{name}{marker}");
        if i == 0 {
            println!("  includes:  {path}");
        } else {
            println!("             {path}");
        }
    }
}
