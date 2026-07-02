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
/// Spawn `/bin/wprof` in a background thread if the host set
/// `KTSTR_WPROF_ARGS` on the kernel cmdline. Returns a join handle
/// whose `.join()` yields `Some(Vec<u8>)` (the `.pb` trace bytes)
/// on success, or `None` on failure / no-op.
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
pub(crate) fn spawn_wprof_if_configured() -> Option<std::thread::JoinHandle<Option<Vec<u8>>>> {
    let args_str = cmdline_val("KTSTR_WPROF_ARGS")?;
    let wprof_bin = std::path::Path::new("/bin/wprof");
    if !wprof_bin.exists() {
        tracing::warn!("KTSTR_WPROF_ARGS set but /bin/wprof missing from initramfs");
        return None;
    }
    Some(
        std::thread::Builder::new()
            .name("wprof-capture".into())
            .spawn(move || {
                // Host encodes args with ASCII Unit Separator (\x1F)
                // via `WprofConfig::args_cmdline` because kernel
                // cmdline tokenization would truncate a space-joined
                // value at the first space. Split on the same
                // delimiter here to recover the per-arg vec.
                let mut cmd_args: Vec<String> = args_str.split('\x1f').map(String::from).collect();
                cmd_args.extend([
                    "-T".to_string(),
                    "/tmp/wprof.pb".to_string(),
                    "-D".to_string(),
                    "/tmp/wprof.data".to_string(),
                ]);
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
                let capture_ms = cmd_args
                    .iter()
                    .position(|a| a == "-d")
                    .and_then(|i| cmd_args.get(i + 1))
                    .and_then(|d| d.parse::<u64>().ok())
                    .unwrap_or(500);
                let deadline = std::time::Duration::from_millis(capture_ms)
                    + std::time::Duration::from_secs(10);
                let mut child = match std::process::Command::new("/bin/wprof")
                    .args(&cmd_args)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(%e, "spawn /bin/wprof failed");
                        return None;
                    }
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
                match crate::sync::pidfd_poll_exited(pid, deadline) {
                    crate::sync::PidfdWait::Exited => {
                        let _ = child.wait();
                    }
                    crate::sync::PidfdWait::TimedOut => {
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
                        match crate::sync::pidfd_poll_exited(
                            pid,
                            std::time::Duration::from_secs(8),
                        ) {
                            crate::sync::PidfdWait::Exited => {
                                let _ = child.wait();
                            }
                            _ => {
                                let _ = child.kill();
                                let _ = child.wait();
                            }
                        }
                    }
                    crate::sync::PidfdWait::NoPidfd => {
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
            .expect("spawn wprof-capture thread"),
    )
}

/// Poll for the virtio-console bulk port and deliver the
/// KERN_ADDRS + SYS_RDY frames to the host.
///
/// The kernel virtio_console driver's multiport handshake
/// (DEVICE_READY → PORT_ADD → PORT_READY → PORT_OPEN, see
/// `drivers/char/virtio_console.c`) completes asynchronously.
/// `/dev/vport0p1` is created by `add_port` when PORT_ADD arrives
/// (via `device_create` + devtmpfs `wait_for_completion`);
/// `host_connected` flips true only when PORT_OPEN arrives later in
/// the same `control_work_handler` batch. So a writev between port
/// creation and host_connected blocks inside `wait_port_writable`
/// (see `wait_event_freezable` at `include/linux/wait.h` — no
/// timeout argument). The loop polls the device-node existence at
/// 100 ms cadence with a wall-clock deadline so blocking writev
/// time counts against the budget. The deadline is captured INSIDE
/// the function so guest-init setup (mounts, kallsyms reads) does
/// not eat the handshake's budget.
///
/// Both KERN_ADDRS and SYS_RDY are required for the host. KERN_ADDRS
/// is latched: once a `send_kern_addrs` call returns true the loop
/// skips it on subsequent iterations. The early-return condition is
/// `kern_addrs_sent && send_sys_rdy()` — we never exit until BOTH
/// have been delivered (a successful sys_rdy on the re-opened FD
/// after a kern_addrs failure must not leave kern_addrs unsent,
/// because the host's KERN_ADDRS arm is the only virt-KASLR
/// publisher on aarch64; see `src/vmm/freeze_coord/dispatch.rs`'s
/// KERN_ADDRS handler).
///
/// Host idempotency for KERN_ADDRS retries (matters when the latch
/// is reset by a failed write that cleared the cached FD):
/// `kern_phys_base` uses `.store(Release)` (overwrites every
/// CRC-valid frame) and `kern_virt_kaslr` uses CAS-once. The
/// payload bytes are identical across retries (built once from
/// `KernAddrs::new`), so repeated stores and a CAS-success-then-
/// no-op-on-equal-existing both produce the same final state.
///
/// On budget exhaustion the function emits a structured WARN with
/// fields `budget_ms`, `vcpus`, `elapsed_ms` (loop wall time),
/// `port_exists` (sampled once before WARN), and `kern_addrs_sent`.
/// The guest then continues — the host monitor's `data_valid` gate
/// keeps reads safe without SYS_RDY, and the freeze coordinator's
/// `Option::take` makes a late SYS_RDY harmless (fire-once). See
/// `doc/guide/src/troubleshooting.md#send_sys_rdy-timeout` for the
/// operator-facing diagnosis flow.
pub(crate) fn send_sys_rdy_with_retry(
    budget: std::time::Duration,
    vcpus: u32,
    kern_addrs: &crate::vmm::wire::KernAddrs,
    port_path: &std::path::Path,
) {
    let loop_t0 = std::time::Instant::now();
    let deadline = loop_t0 + budget;
    let mut kern_addrs_sent = false;
    loop {
        if port_path.exists() {
            if !kern_addrs_sent {
                kern_addrs_sent = crate::vmm::guest_comms::send_kern_addrs(kern_addrs);
            }
            if kern_addrs_sent && crate::vmm::guest_comms::send_sys_rdy() {
                return;
            }
        }
        if std::time::Instant::now() >= deadline {
            // Snapshot before WARN so the field reports the
            // last-attempt state, not a fresh stat that could
            // observe a port appearing in the gap between the
            // final loop iteration and the WARN call.
            let port_exists_snapshot = port_path.exists();
            tracing::warn!(
                budget_ms = budget.as_millis() as u64,
                vcpus,
                elapsed_ms = loop_t0.elapsed().as_millis() as u64,
                port_exists = port_exists_snapshot,
                kern_addrs_sent,
                "ktstr-init: send_sys_rdy failed within boot budget; \
                 see https://ktstr.dev/guide/troubleshooting.html#send_sys_rdy-timeout",
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
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
