//! stdio redirection, shell mode, and the disk-template build mode.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;


/// Redirect stdout and stderr through bulk-port forwarder threads.
///
/// Pre-bulk-port-migration: dup2'd `/dev/ttyS1` over fd 1 and fd 2 so
/// every `println!` / `eprintln!` reached the host as a stream of
/// COM2 bytes.  The bulk-port migration replaces COM2 with one
/// `MsgType::Stdout` / `MsgType::Stderr` TLV frame per chunk:
///
///   1. Open a pair of `pipe(2)` pipes (one for stdout, one for
///      stderr).
///   2. `dup2` each pipe's write end over fd 1 / fd 2 so every
///      `println!` / `eprintln!` lands in the pipe.
///   3. Spawn one reader thread per pipe.  Each thread reads up to
///      [`STDIO_CHUNK_BYTES`] at a time from the pipe's read end and
///      ships the chunk via
///      [`crate::vmm::guest_comms::send_stdout_chunk`] /
///      [`crate::vmm::guest_comms::send_stderr_chunk`].
///
/// The threads are detached: they exit cleanly when fd 1 / fd 2 are
/// closed (process exit / `force_reboot`) because the read end then
/// returns EOF.
///
/// Panic diagnostics still go to COM2 — the panic hook in
/// [`ktstr_guest_init`] writes directly to `/dev/ttyS1` because the
/// hook cannot block on virtio backpressure.  Every other guest
/// stream now travels over the bulk port.
///
/// On any pipe / dup2 / thread-spawn failure the function logs via
/// the tracing subscriber (which writes to stderr; fd 2 is still
/// attached to the kernel console at the failure point, so the
/// operator sees the misroute) and returns — stdout/stderr stay
/// attached to whatever fd they pointed at on entry.
pub(crate) fn redirect_stdio_to_bulk_port() {
    use std::io::Read;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    fn make_pipe() -> Option<(std::fs::File, std::fs::File)> {
        let mut fds = [0i32; 2];
        // SAFETY: `fds` is a valid `&mut [i32; 2]`; `pipe(2)` writes
        // exactly two file descriptors on success.  Passing `O_CLOEXEC`
        // would belong on `pipe2`, but we deliberately want the pipe
        // ends to survive across any forks the test may perform — the
        // dup2'd write end carries fd 1 / fd 2 across exec/fork, which
        // is the entire point.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        if r < 0 {
            return None;
        }
        // SAFETY: `pipe(2)` just returned with the two fds populated.
        // `from_raw_fd` takes ownership of each side; both close on
        // drop.  Held by `File` for the natural Read/Write impls.
        let read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        Some((read_end, write_end))
    }

    fn spawn_forwarder(mut read_end: std::fs::File, name: &'static str, sender: fn(&[u8]) -> bool) {
        let _ = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let mut buf = [0u8; STDIO_CHUNK_BYTES];
                loop {
                    match read_end.read(&mut buf) {
                        Ok(0) => break, // EOF — fd 1/2 closed.
                        Ok(n) => {
                            // Fire-and-forget.  `send_*_chunk`
                            // returns false when the bulk port is
                            // not yet ready; bytes emitted before
                            // the multiport handshake completes are
                            // dropped.  Same caveat as the prior
                            // COM2 path's pre-handshake byte loss.
                            let _ = sender(&buf[..n]);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
            });
    }

    let Some((stdout_r, stdout_w)) = make_pipe() else {
        tracing::error!("ktstr-init: redirect_stdio_to_bulk_port: pipe(stdout) failed");
        return;
    };
    let Some((stderr_r, stderr_w)) = make_pipe() else {
        tracing::error!("ktstr-init: redirect_stdio_to_bulk_port: pipe(stderr) failed");
        return;
    };

    // Capture errno via `last_os_error` BEFORE any subsequent libc
    // call: errno is per-thread but every libc call may clobber it.
    let (rc1, err1, rc2, err2) = unsafe {
        let r1 = libc::dup2(stdout_w.as_raw_fd(), 1);
        let e1 = std::io::Error::last_os_error();
        let r2 = libc::dup2(stderr_w.as_raw_fd(), 2);
        let e2 = std::io::Error::last_os_error();
        (r1, e1, r2, e2)
    };
    // The dup2 above duplicated each pipe's write end onto fd 1 / fd 2;
    // the originals (`stdout_w` / `stderr_w`) close on this scope's
    // exit.  Without that close, the read end of each pipe would see
    // EOF only after the test process holding fd 1 / fd 2 also dropped
    // those file descriptors — but we want the EOF condition to fire
    // when fd 1 / fd 2 reach their natural close-on-exit, not when
    // some other holder of `stdout_w` closes too.  Letting the
    // originals drop here is correct because `dup2` increments the
    // file's refcount.
    if rc1 < 0 {
        tracing::error!(err = %err1, "ktstr-init: redirect_stdio_to_bulk_port: dup2(stdout) failed");
    }
    if rc2 < 0 {
        tracing::error!(err = %err2, "ktstr-init: redirect_stdio_to_bulk_port: dup2(stderr) failed");
    }

    spawn_forwarder(stdout_r, "ktstr-stdout-fwd", |b| {
        crate::vmm::guest_comms::send_stdout_chunk(b)
    });
    spawn_forwarder(stderr_r, "ktstr-stderr-fwd", |b| {
        crate::vmm::guest_comms::send_stderr_chunk(b)
    });
}

/// Check kernel cmdline for KTSTR_MODE=shell.
pub(crate) fn shell_mode_requested() -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|c| cmdline_contains_token(&c, "KTSTR_MODE=shell"))
        .unwrap_or(false)
}

/// Check kernel cmdline for `KTSTR_MODE=disk_template`. The host
/// asserts this when booting a one-shot template-build VM (see
/// [`crate::vmm::disk_template`]).
pub(crate) fn disk_template_mode_requested() -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|c| cmdline_contains_token(&c, "KTSTR_MODE=disk_template"))
        .unwrap_or(false)
}

/// Pure-function cmdline-token check, factored out of
/// [`shell_mode_requested`] / [`disk_template_mode_requested`] so
/// the precedence-and-multiplicity behavior can be tested without
/// mocking `/proc/cmdline`. Whitespace-separated, exact match (the
/// kernel passes cmdline tokens verbatim — no quoting, no escapes).
pub(crate) fn cmdline_contains_token(cmdline: &str, token: &str) -> bool {
    cmdline.split_whitespace().any(|s| s == token)
}

/// Disk-template build dispatch: exec `/bin/mkfs.btrfs /dev/vda`
/// (the host packed `mkfs.btrfs` into the initramfs at this path),
/// wait for it, return its exit code so the caller emits the exit
/// sentinel on COM2 before rebooting. Returns `0` on success and
/// the binary's exit code (or `1` on spawn failure) otherwise.
///
/// The disk image at `/dev/vda` is the host-side staging file
/// (sparse, sized to the requested capacity); after this function
/// returns and the VM reboots, the host's [`crate::vmm::disk_template::store_atomic`]
/// publishes the now-formatted image into the cache.
///
/// The host never execs `mkfs.btrfs` against a real backing file —
/// driving the format through this guest-side dispatch keeps the
/// kernel under test as the on-disk-format authority, so any btrfs
/// feature regression in that kernel surfaces as a guest format
/// failure here instead of as a host/guest mkfs disagreement that
/// would slip past testing.
pub(crate) fn run_disk_template_mode() -> i32 {
    redirect_stdio_to_bulk_port();
    // The mkfs.btrfs binary is packed at `bin/mkfs.btrfs` by
    // [`crate::vmm::disk_template::build_template_via_vm`] via
    // `include_files`; that function — not `ensure_template` — is
    // the host-side site that assembles the template-VM
    // initramfs.
    const MKFS: &str = "/bin/mkfs.btrfs";
    // `-f` forces overwrite of any existing signature so a leftover
    // ext4 magic from a host that recycled the staging file does
    // not block formatting. mkfs runs verbose (no `--quiet`) so its
    // own diagnostics land in the guest stderr, which the host's
    // publish gate surfaces on a magic-check failure. `/dev/vda` is
    // the singleton virtio-blk device the host attached.
    //
    // No `--metadata DUP` override: btrfs picks DUP metadata by
    // default on a single-device fs, which is the desired
    // production format. The 256 MiB minimum capacity (see
    // VIRTIO_BLK_DEFAULT_CAPACITY_BYTES doc) accommodates DUP.
    tracing::info!(mkfs = MKFS, target = "/dev/vda", "running mkfs.btrfs");
    // SIGCHLD is `SIG_IGN` for the rest of this process (installed by
    // [`ktstr_guest_init`] for zombie prevention). `Command::output()`
    // calls `waitpid(2)` internally; under `SIG_IGN` the kernel
    // auto-reaps the child before `waitpid` runs, so the syscall
    // returns `ECHILD`, the std-lib maps it to
    // `Err(io::Error::ECHILD)`, and the original `match` branch fell
    // into the `Err(_) => 1` arm — surfacing a fixed `1` exit code for
    // every successful `mkfs.btrfs` run. The host would then see
    // "template build failed" for a perfectly formatted image. Restore
    // `SIG_DFL` for the closure's lifetime so `waitpid` reaps and
    // reports the real status; the post-closure restore re-installs
    // `SIG_IGN` for any future child this process spawns.
    //
    // The build VM's stdio forwarders are NOT joined before
    // `force_reboot`'s `RB_AUTOBOOT`, so bytes still in the
    // stdout/stderr pipes at reboot are lost (see the dispatch site),
    // and the host does not drain the bulk-port stdout/stderr frames
    // for a fast-exiting build VM either — only the EXIT frame and COM2
    // survive. So capture mkfs's output in-process via
    // `Command::output()` and surface it on the two channels that DO
    // survive: the exit code (a pass/no-op verdict) and COM2
    // (`/dev/ttyS1`, the full text) — so the publish gate can show WHY
    // a build produced no filesystem.
    let mut diag = format!(
        "MKFS_DIAG /dev/vda exists={} /sys/class/block/vda exists={}\n",
        std::path::Path::new("/dev/vda").exists(),
        std::path::Path::new("/sys/class/block/vda").exists(),
    );
    match std::fs::metadata(MKFS) {
        Ok(m) => diag.push_str(&format!("MKFS_DIAG {MKFS} size={}\n", m.len())),
        Err(e) => diag.push_str(&format!("MKFS_DIAG {MKFS} metadata failed: {e}\n")),
    }
    let output = with_sigchld_default(|| Command::new(MKFS).args(["-f", "/dev/vda"]).output());
    let mut code = match output {
        Ok(o) => {
            diag.push_str(&format!(
                "MKFS_DIAG exit={:?}\nMKFS_STDOUT:\n{}\nMKFS_STDERR:\n{}\n",
                o.status.code(),
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr),
            ));
            o.status.code().unwrap_or(1)
        }
        Err(e) => {
            diag.push_str(&format!("MKFS_DIAG mkfs spawn failed: {e}\n"));
            1
        }
    };
    const BTRFS_DEV_MAGIC: u64 = 0x4D5F_5366_5248_425F;
    let magic = read_dev_vda_magic();
    let magic_present = magic.as_ref().is_ok_and(|&m| m == BTRFS_DEV_MAGIC);
    diag.push_str(&format!(
        "MKFS_DIAG in-guest /dev/vda magic@0x10040 = {}\n",
        match &magic {
            Ok(m) => format!("0x{m:016x}"),
            Err(e) => format!("read failed: {e}"),
        },
    ));
    // Only sync + run the no-op verdict on a clean mkfs exit; a non-zero
    // mkfs exit propagates through the host's clean-exit gate
    // (build_template_via_vm), which surfaces mkfs's stderr tail.
    if code == 0 {
        if let Some(flush_err) = flush_template_disk() {
            diag.push_str(&format!("MKFS_DIAG flush: {flush_err}\n"));
        }
        // A clean mkfs exit that leaves /dev/vda WITHOUT the btrfs magic
        // — in the guest's own page-cache view — is a silent no-op.
        // Override to a distinct exit code so the host's clean-exit check
        // reports it (the exit code is the only build-VM channel proven
        // to survive force_reboot, so the verdict lands even if the COM2
        // diag below is lost). Distinguishes a guest-side no-op from a
        // host-side write loss (which Layer B's host-magic check catches).
        if !magic_present {
            diag.push_str(
                "MKFS_DIAG VERDICT: mkfs exited 0 but /dev/vda has no btrfs magic \
                 in-guest — silent no-op (exit overridden to 64)\n",
            );
            code = 64;
        }
    }
    // Mirror the diag to COM2 (`/dev/ttyS1`) — the host's
    // fault-diagnostic serial channel (the panic hook uses it),
    // captured into `result.output` via `com2_bytes` in
    // `collect_results`. COM2 is synchronous and survives the immediate
    // `force_reboot`, unlike the bulk-port stdio forwarders.
    if let Ok(mut com2) = std::fs::OpenOptions::new().write(true).open("/dev/ttyS1") {
        use std::io::Write;
        let _ = com2.write_all(diag.as_bytes());
        let _ = com2.flush();
    }
    code
}

/// Read the 8-byte little-endian btrfs superblock magic at offset
/// `0x10040` from `/dev/vda` as the guest sees it through the
/// block-device page cache. Returns the raw `u64` (the caller compares
/// it against the expected magic for the no-op verdict — a value
/// compare, not a fragile string compare) or the io::Error if the
/// device can't be opened/read. The build-VM diagnostic formats it to
/// distinguish "mkfs wrote the magic but the host backing lost it" from
/// "mkfs reported success without writing anything".
fn read_dev_vda_magic() -> std::io::Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open("/dev/vda")?;
    f.seek(SeekFrom::Start(0x1_0040))?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Flush `/dev/vda` to the host backing before `force_reboot`'s
/// `RB_AUTOBOOT` (which performs no sync). Belt-and-braces:
/// `bdev_release` already `sync_blockdev`s the page cache when the
/// last opener closes (block/bdev.c), so a normal buffered write is
/// already durable on the host by the time mkfs exits — but a guest
/// path that bypasses the bdev page cache (mmap, an O_DIRECT skip)
/// would otherwise leave writes unflushed at reboot, and `RB_AUTOBOOT`
/// abandons them (virtio_blk has no `.shutdown` op that drains IO).
///
/// Returns `None` on success or `Some(msg)` describing the failure.
/// The caller folds the message into the COM2-mirrored diag rather
/// than logging it: build-VM stderr is dropped at `force_reboot` (the
/// bulk-port forwarders are not joined before the reboot), so a
/// `tracing::error!` here would never reach the host.
fn flush_template_disk() -> Option<String> {
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/vda")
    {
        Ok(f) => f
            .sync_all()
            .err()
            .map(|e| format!("sync_all(/dev/vda) failed: {e}")),
        Err(e) => Some(format!("open(/dev/vda) for flush failed: {e}")),
    }
}

/// Read /exec_cmd from the initramfs if present.
/// The host writes this file via build_suffix when --exec is used.
pub(crate) fn shell_exec_cmd() -> Option<String> {
    fs::read_to_string("/exec_cmd")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Extract a KEY=value pair from the kernel cmdline.
pub(crate) fn cmdline_val(key: &str) -> Option<String> {
    let cmdline = fs::read_to_string("/proc/cmdline").ok()?;
    let prefix = format!("{key}=");
    cmdline
        .split_whitespace()
        .find_map(|s| s.strip_prefix(&prefix))
        .map(|s| s.to_string())
}

/// Build PATH with /include-files directories containing executables.
///
/// Walks /include-files recursively, collects directories that contain
/// at least one executable file, prepends them all to PATH. This makes
/// included binaries runnable by name regardless of subdirectory depth
/// (e.g. `-i ../scx/target/release` → `scx_cake` works directly).
pub(crate) fn build_include_path() -> String {
    use std::collections::BTreeSet;
    use std::os::unix::fs::PermissionsExt;
    let include_dir = std::path::Path::new("/include-files");
    let mut dirs = BTreeSet::new();

    if include_dir.is_dir() {
        for entry in walkdir::WalkDir::new(include_dir).follow_links(true) {
            let Ok(entry) = entry else { continue };
            if entry.file_type().is_file()
                && entry
                    .metadata()
                    .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
                && let Some(parent) = entry.path().parent()
            {
                dirs.insert(parent.to_string_lossy().to_string());
            }
        }
    }

    let mut path_parts: Vec<String> = dirs.into_iter().collect();
    path_parts.push("/bin".to_string());
    path_parts.join(":")
}

/// Redirect stdin, stdout, and stderr to the given device with O_RDWR.
///
/// Shell mode needs all three fds on the console device: stdin for
/// reading input, stdout/stderr for writing output.
///
/// `dup2` failures are logged via `tracing::error!`. A failing `dup2`
/// leaves the target fd unchanged, so the diagnostic still reaches
/// the pre-redirect stderr (kernel console / COM1) through the
/// tracing subscriber and the operator sees the misroute rather than
/// the failing path silently writing to a wrong device.
pub(crate) fn redirect_all_stdio_to(path: &str) {
    use std::os::unix::io::AsRawFd;

    let Ok(dev) = fs::OpenOptions::new().read(true).write(true).open(path) else {
        return;
    };
    let fd = dev.as_raw_fd();
    // Capture errno per call before the next libc call clobbers
    // it. Run all three syscalls sequentially without aborting on
    // a partial failure — fd 0 redirect failing should not stop us
    // from at least getting stdout/stderr onto the console.
    let (rc0, err0, rc1, err1, rc2, err2) = unsafe {
        let r0 = libc::dup2(fd, 0);
        let e0 = std::io::Error::last_os_error();
        let r1 = libc::dup2(fd, 1);
        let e1 = std::io::Error::last_os_error();
        let r2 = libc::dup2(fd, 2);
        let e2 = std::io::Error::last_os_error();
        (r0, e0, r1, e1, r2, e2)
    };
    if rc0 < 0 {
        tracing::error!(path, err = %err0, "ktstr-init: redirect_all_stdio_to: dup2(stdin) failed");
    }
    if rc1 < 0 {
        tracing::error!(path, err = %err1, "ktstr-init: redirect_all_stdio_to: dup2(stdout) failed");
    }
    if rc2 < 0 {
        tracing::error!(path, err = %err2, "ktstr-init: redirect_all_stdio_to: dup2(stderr) failed");
    }
}

/// Select the console device for shell mode.
/// Prefers /dev/hvc0 (virtio-console) when available, falls back to COM2.
pub(crate) fn shell_console_device() -> &'static str {
    if Path::new(HVC0).exists() { HVC0 } else { COM2 }
}

/// Mount devpts at /dev/pts for PTY allocation.
///
/// Required before `openpty()` — the C library opens `/dev/ptmx` and
/// the slave device lives under `/dev/pts/N`.
pub(crate) fn mount_devpts() {
    mkdir_p("/dev/pts");
    let result = mount(
        Some("devpts"),
        "/dev/pts",
        Some("devpts"),
        MsFlags::empty(),
        None::<&str>,
    );
    if let Err(e) = result {
        tracing::error!(err = %e, "ktstr-init: mount devpts on /dev/pts failed");
    }
}

/// Spawn busybox sh with a PTY as its controlling terminal.
///
/// Allocates a PTY pair via `openpty()`, spawns sh with the slave as
/// stdin/stdout/stderr and `setsid` + `TIOCSCTTY` in `pre_exec` so sh
/// gets a controlling terminal (job control). The parent proxies data
/// between COM2 (fd 0/1) and the PTY master until the child exits.
///
/// SIGCHLD remains SIG_IGN (set earlier for zombie prevention), so
/// waitpid returns ECHILD after the kernel auto-reaps the child.
/// This is expected and suppressed.
pub(crate) fn spawn_shell_with_pty() {
    let pty = match openpty(None, None) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(err = %e, "ktstr-init: openpty failed");
            return;
        }
    };

    let slave_fd = pty.slave.as_raw_fd();

    // Set PTY size from host terminal dimensions passed via cmdline.
    if let (Some(cols), Some(rows)) = (cmdline_val("KTSTR_COLS"), cmdline_val("KTSTR_ROWS"))
        && let (Ok(cols), Ok(rows)) = (cols.parse::<u16>(), rows.parse::<u16>())
    {
        let ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSWINSZ, &ws);
        }
    }

    // Set terminal type from host. Default to "linux" if not passed.
    let term = cmdline_val("KTSTR_TERM").unwrap_or_else(|| "linux".to_string());
    let colorterm = cmdline_val("KTSTR_COLORTERM");

    let child = unsafe {
        let mut cmd = Command::new("/bin/busybox");
        cmd.arg("sh")
            .env("TERM", &term)
            .env("PS1", "\x1b[2m^Ax=quit\x1b[0m \\w # ");
        if let Some(ref ct) = colorterm {
            cmd.env("COLORTERM", ct);
        }
        cmd.stdin(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .stdout(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .stderr(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .pre_exec(move || {
                // Create a new session so sh becomes session leader.
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // Acquire a controlling terminal.
                if libc::ioctl(slave_fd, libc::TIOCSCTTY, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            })
            .spawn()
    };

    // Close slave in parent — the child has its own copies.
    drop(pty.slave);

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(err = %e, "ktstr-init: spawn shell failed");
            return;
        }
    };

    let child_pid = child.id();

    // Set COM2 serial (fd 0) to raw mode so the kernel line discipline
    // passes bytes through without processing. Without this, special
    // characters like tab (0x09) are consumed by the line discipline
    // instead of being forwarded through the proxy to the PTY.
    let stdin_fd = unsafe { BorrowedFd::borrow_raw(0) };
    if let Ok(mut termios) = tcgetattr(stdin_fd) {
        cfmakeraw(&mut termios);
        let _ = tcsetattr(stdin_fd, SetArg::TCSANOW, &termios);
    }

    // Proxy between COM2 (fd 0 for input, fd 1 for output) and PTY master.
    proxy_serial_pty(&pty.master, child_pid);

    // SIGCHLD is SIG_IGN so the kernel auto-reaps the child. waitpid
    // returns ECHILD — expected, not an error.
    match child.wait() {
        Ok(status) => {
            tracing::debug!(?status, "shell exited");
        }
        Err(e) if e.raw_os_error() == Some(libc::ECHILD) => {}
        Err(e) => {
            tracing::warn!(err = %e, "ktstr-init: wait for shell failed");
        }
    }

    // No guest-side exit message — the host prints "Connection to VM
    // closed." after the VM shuts down. Printing here too would
    // duplicate it, and writing to COM2 in raw mode after PTY teardown
    // leaks garbage bytes.
}

/// Proxy data between COM2 serial (fd 0/1) and a PTY master fd.
///
/// Uses poll(2) to multiplex reads from both fds. Exits when the PTY
/// master returns EOF (child closed the slave side) or the child process
/// no longer exists.
fn proxy_serial_pty(master: &OwnedFd, child_pid: u32) {
    let stdin_fd = unsafe { BorrowedFd::borrow_raw(0) };
    let stdout_fd = unsafe { BorrowedFd::borrow_raw(1) };
    let master_fd = master.as_fd();

    let mut buf = [0u8; 4096];

    loop {
        let mut pollfds = [
            PollFd::new(stdin_fd, PollFlags::POLLIN),
            PollFd::new(master_fd, PollFlags::POLLIN),
        ];

        match poll(&mut pollfds, PollTimeout::from(200u16)) {
            Ok(0) => {
                // Timeout — check if child is still alive.
                if !Path::new(&format!("/proc/{child_pid}")).exists() {
                    break;
                }
                continue;
            }
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
        }

        // Serial input -> PTY master (user typing).
        if let Some(revents) = pollfds[0].revents() {
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(stdin_fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = nix::unistd::write(master_fd, &buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(_) => break,
                }
            }
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP) {
                break;
            }
        }

        // PTY master -> serial output (shell output).
        // Check POLLHUP/POLLERR before POLLIN: when the shell exits,
        // both flags can arrive in the same poll iteration. Reading
        // after the slave closes produces partial/garbage bytes from
        // the PTY teardown (manifests as a raw U+FFFD on the terminal).
        if let Some(revents) = pollfds[1].revents() {
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP) {
                break;
            }
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(master_fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = nix::unistd::write(stdout_fd, &buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(_) => break,
                }
            }
        }
    }
}
