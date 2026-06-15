//! The PID-1 entry point `ktstr_guest_init` — mounts, then dispatches a test or a shell.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;


/// Full guest init lifecycle. Called from the ctor when PID 1 is
/// detected. Mounts filesystems, then either runs the test lifecycle
/// (scheduler + dispatch + reboot) or drops into an interactive
/// shell. Never returns.
pub(crate) fn ktstr_guest_init() -> ! {
    let t0 = std::time::Instant::now();

    // Crash diagnostic capture has two arms because they have
    // disjoint trigger surfaces:
    //
    // 1. Native fatal signals (`install_fatal_signal_handlers`,
    //    installed first): SIGSEGV / SIGBUS / SIGILL invoke the
    //    kernel's `do_coredump` under SIG_DFL — they bypass the
    //    panic hook entirely. Without a sigaction handler the
    //    kernel terminates init, which the parent kernel observes
    //    as "init exited" and force-reboots without any guest-side
    //    diagnostic reaching the host. Installing this arm before
    //    the panic hook minimises the window where an early fault
    //    (heap setup, mount syscalls, anything before the hook
    //    registers) escapes capture.
    // 2. Rust panic hook (below): fires on `panic!`, `unwrap`,
    //    assertion failures, and any other invocation of the Rust
    //    panic machinery (both `panic = "unwind"` and
    //    `panic = "abort"` runtimes invoke the hook before
    //    unwinding/aborting).
    //
    // Both arms write a `PANIC:`-prefixed line to COM2 (and COM1)
    // so the host-side `extract_panic_message` picks them up
    // through the same code path. COM2 is the canonical crash-
    // diagnostic transport, surviving a wedged virtio port: the
    // bulk-virtio path is intentionally NOT used here because the
    // kernel `virtio_console` TX can block on host backpressure
    // and blocking inside a fault handler would deadlock the
    // guest before the diagnostic reached the host. COM2 (16550
    // UART) PIO writes commit synchronously inside `KVM_RUN`
    // before userspace returns, so the host's serial capture
    // sees every byte even on a wedged guest.
    install_fatal_signal_handlers();
    std::panic::set_hook(Box::new(|info| {
        // Write the `PANIC:` header FIRST — cheap, no symbolization —
        // so the diagnostic reaches the host even when the subsequent
        // backtrace symbolization (which faults in the binary's DWARF,
        // hundreds of MiB for a debuginfo-heavy test binary) allocates
        // beyond a memory-pressured guest's headroom and aborts. The
        // host's `extract_panic_message` keys on this `PANIC:` prefix.
        let head = format!("PANIC: {info}\n");
        let _ = fs::write(COM2, &head);
        let _ = fs::write(COM1, &head);
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!("{bt}\n");
        // COM2 / COM1 serial. COM2 is the canonical crash log
        // destination for the host's serial-capture path; the
        // host parses the `PANIC:` prefix via
        // `extract_panic_message` to reconstruct the crash
        // diagnostic.
        let _ = fs::write(COM2, &msg);
        let _ = fs::write(COM1, &msg);
        // Push any buffered Rust-side bytes into the underlying pipe
        // before reboot. After stdio redirect, fd 1 / fd 2 are
        // pipe write ends drained by `redirect_stdio_to_bulk_port`'s
        // forwarder threads — `tcdrain` is unavailable here (the
        // pipe is not a tty, the syscall returns ENOTTY silently).
        // `flush()` is the equivalent: it commits any
        // BufWriter-buffered bytes into the pipe's kernel buffer
        // where the forwarder thread can pick them up. The
        // forwarder threads are not joined before `force_reboot`;
        // bytes that have not yet been read out of the pipe and
        // shipped over the bulk port at the moment of reboot are
        // lost — see the queue task on joining the forwarders for
        // the residual gap. The COM1/COM2 `fs::write` above remains
        // the synchronous-PIO path that guarantees the panic
        // diagnostic itself reaches the host before reboot.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        force_reboot();
    }));

    // Ignore SIGCHLD so child processes don't become zombies.
    // PID 1 is the reaper — without this, zombie processes accumulate.
    unsafe {
        libc::signal(libc::SIGCHLD, libc::SIG_IGN);
    }

    // Phase 1: Mounts.
    mount_filesystems();
    let t_mounts = t0.elapsed();

    // Install the tracing subscriber as early as possible — right after
    // `mount_filesystems()` so /proc is available for the RUST_LOG
    // cmdline extraction below, and BEFORE the rest of guest init runs
    // so every subsequent `tracing::*` call is captured. Earlier
    // versions installed the subscriber after `redirect_stdio_to_bulk_port`,
    // which silently dropped every tracing event before the redirect.
    //
    // EnvFilter respects RUST_LOG when set; default is `warn` so
    // teardown diagnostics (`tracing::warn!`, `tracing::error!`)
    // surface without requiring RUST_LOG to be plumbed through the
    // guest cmdline. `from_default_env()` alone would collapse to
    // the implicit `error` level and swallow warn-level output —
    // exactly the diagnostics needed to debug teardown failures.
    if let Ok(cmdline) = fs::read_to_string("/proc/cmdline")
        && let Some(val) = cmdline
            .split_whitespace()
            .find(|s| s.starts_with("RUST_LOG="))
            .and_then(|s| s.strip_prefix("RUST_LOG="))
    {
        // SAFETY: single-threaded PID 1 context.
        unsafe { std::env::set_var("RUST_LOG", val) };
    }
    let t_pre_subscriber = t0.elapsed();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let t_subscriber = t0.elapsed();

    // Verify initramfs extraction completed. The sentinel file is the
    // last entry written by build_initramfs_base — its absence means
    // the kernel ran out of memory during cpio extraction. The memory
    // formula should prevent this; hitting it indicates an estimation bug.
    if !Path::new("/.ktstr_init_ok").exists() {
        // Dump dmesg to serial so the host sees the kernel OOM messages.
        if let Ok(raw) = rmesg::logs_raw(rmesg::Backend::Default, false) {
            let _ = fs::write(COM2, &raw);
            let _ = fs::write(COM1, &raw);
        }
        let msg = "FATAL: initramfs extraction incomplete — kernel ran out of \
                   memory during cpio extraction. This indicates a bug in ktstr's \
                   memory estimation. Please report this issue. As a workaround, \
                   try `--memory N` with a larger value.";
        let _ = fs::write(COM2, msg);
        let _ = fs::write(COM1, msg);
        tracing::error!("{msg}");
        force_reboot();
    }

    // Boot-complete signal. The host monitor's pre-sample
    // `epoll_wait` blocks on a sys_rdy eventfd; the freeze
    // coordinator's bulk-drain dispatch promotes a CRC-valid
    // `MSG_TYPE_SYS_RDY` frame into that eventfd. Sending here —
    // after `mount_filesystems()` brought up devtmpfs and the
    // initramfs-extraction sentinel confirms userspace is sound —
    // guarantees the host's first sample observes a fully-booted
    // guest with `setup_per_cpu_areas` populated and KASLR
    // randomization already complete (both kernel-boot
    // prerequisites for the monitor's `__per_cpu_offset[]` /
    // `page_offset_base` reads). Replaces the earlier trigger that
    // fired on the first port-0 TX byte (kernel printk via
    // `/dev/hvc0`), which depended on incidental console traffic
    // rather than an explicit readiness signal.
    //
    // `/dev/vport0p1` may not yet exist at this point: the kernel
    // virtio_console driver's multiport handshake (DEVICE_READY →
    // PORT_ADD → PORT_READY → PORT_OPEN, see
    // `drivers/char/virtio_console.c`) completes asynchronously
    // and is independent of devtmpfs being mounted. The retry
    // protocol, wall-clock deadline, and failure diagnostics live
    // in [`send_sys_rdy_with_retry`].
    let kern_phys_base = crate::vmm::guest_comms::read_phys_base_from_iomem().unwrap_or(0);
    // Runtime KVA of `_text`, the kernel image start symbol.
    // Powers the host-side virt-KASLR derive at
    // `src/vmm/freeze_coord/dispatch.rs::dispatch_bulk_message`'s
    // KERN_ADDRS arm: subtracting the link-time KVA (from the
    // host's vmlinux parse) yields the virt-KASLR slide so the
    // monitor and dump pipelines can resolve per-CPU `rq` /
    // `kernel_cpustat` / `kstat` KVAs under
    // `CONFIG_RANDOMIZE_BASE=y`. `_text` is defined in
    // `vmlinux.lds.S` on every Linux build, so this works on
    // both x86_64 and aarch64. `None` only when the symbol is
    // masked (kptr_restrict + non-CAP_SYSLOG, which we are not —
    // PID 1 has all caps) or `/proc/kallsyms` is unreadable; the
    // host's KERN_ADDRS handler treats `None` as "guest could
    // not derive" and leaves the slot at its prior value (the
    // BSP MSR_LSTAR path may still publish on x86_64).
    let kern_text_kva = crate::vmm::guest_comms::read_kernel_text_from_kallsyms();
    // `page_offset_base` slot — derive the runtime KVA of the
    // `page_offset_base` global from /proc/kallsyms (kernel-PhD-
    // confirmed it lives in `.data..ro_after_init`, declared at
    // `arch/x86/kernel/head64.c:63`). The KVA here is the symbol's
    // ADDRESS — the host reads the runtime VALUE (the direct-map
    // base) by translating this KVA to PA via
    // `monitor::symbols::text_kva_to_pa_with_base` (using
    // `kern_phys_base`) and `read_u64`-ing at that PA. Returns
    // `None` on arm64 (no `page_offset_base` global — `PAGE_OFFSET`
    // is compile-time per `arch/arm64/include/asm/memory.h:43-45`)
    // and when CONFIG_RANDOMIZE_MEMORY=n (symbol absent). The
    // wire field is `u64`, so `None` collapses to 0 — host treats
    // 0 as "use DEFAULT_PAGE_OFFSET fallback" (matching the
    // historical pre-derivation behavior).
    let kern_page_offset_base_kva =
        crate::vmm::guest_comms::read_kernel_page_offset_base_from_kallsyms().unwrap_or(0);
    let kern_addrs =
        crate::vmm::wire::KernAddrs::new(kern_phys_base, kern_page_offset_base_kva, kern_text_kva);
    // `count_online_cpus()` reads /sys/devices/system/cpu/online which
    // `mount_filesystems()` mounted earlier. Fallback to 1 yields the
    // single-vCPU budget (base + 1×per-vCPU) if the read fails —
    // preserves the original single-CPU default rather than panicking
    // on a procfs hiccup.
    let vcpus = count_online_cpus().unwrap_or(1);
    let budget = std::time::Duration::from_millis(crate::test_support::sys_rdy_budget_ms(vcpus));
    send_sys_rdy_with_retry(
        budget,
        vcpus,
        &kern_addrs,
        std::path::Path::new(crate::vmm::guest_comms::BULK_PORT_DEV),
    );

    // Phase 1.5: Auto-mount the user data disk at /mnt/disk0 if the
    // host pre-formatted it (KTSTR_DISK0_FS=<tag> on the cmdline).
    // Runs BEFORE `disk_template_mode_requested()` is checked below
    // — but the template-build cmdline never carries
    // `KTSTR_DISK0_FS` (the host emits it only for non-Raw disks
    // and the template-build VM attaches a Raw disk because the
    // whole point is to format it), so this call is a no-op
    // during template-build and the build path is unaffected.
    auto_mount_data_disks();
    // Enable per-program BPF runtime stats (cnt, nsecs). The kernel
    // only populates bpf_prog_stats when bpf_stats_enabled_key is set.
    let _ = fs::write("/proc/sys/kernel/bpf_stats_enabled", "1");

    // Phase 2: Lifecycle event + stdio redirect. The lifecycle frame
    // is for the test harness on the host; shell mode doesn't need it
    // and would route the InitStarted phase into the operator's
    // bulk-port-backed transcript otherwise.
    if !shell_mode_requested() {
        crate::vmm::guest_comms::send_lifecycle(crate::vmm::wire::LifecyclePhase::InitStarted, "");
    }
    redirect_stdio_to_bulk_port();
    let t_stdio = t0.elapsed();

    // Phase 2c: spawn the scheduler-stats relay UNCONDITIONALLY.
    // Event-driven: the relay uses inotify to wait for the
    // scheduler's `/var/run/scx/root/stats` socket to appear, and
    // poll(2) to multiplex between the port fd, the socket fd, and
    // a stop eventfd. No timeouts, no retry sleeps — the only
    // wakeups are real I/O events or the stop edge written by
    // phase-6 cleanup.
    //
    // By this point `redirect_stdio_to_bulk_port` has run (line
    // above) and the bulk port has been opened, which proves the
    // multiport handshake completed; `/dev/vport0p2` is already
    // present, so the relay's first port-2 open succeeds without
    // retry.
    let stats_relay_stop = start_sched_stats_relay();

    tracing::debug!(
        mount_ms = t_mounts.as_millis() as u64,
        stdio_ms = t_stdio.as_millis() as u64,
        pre_subscriber_ms = t_pre_subscriber.as_millis() as u64,
        subscriber_ms = t_subscriber.as_millis() as u64,
        "guest_init_timing",
    );

    // Set environment variables.
    // SAFETY: single-threaded context — PID 1 before any threads spawn.
    unsafe {
        std::env::set_var("PATH", build_include_path());
        // Mark this process tree as running under guest init (PID 1).
        // Workers forked inside the guest legitimately have
        // `getppid() == 1` because init IS their parent, so the
        // host-side orphan-detection fast-path in `workload.rs` must
        // skip the `_exit(0)` branch when this variable is present.
        // The variable is inherited across fork/exec, so every
        // descendant of guest init (including workloads that re-exec
        // /init to run scenarios) observes it.
        std::env::set_var(crate::KTSTR_GUEST_INIT_ENV, "1");
    }

    // Disk-template build mode: format /dev/vda with the embedded
    // mkfs binary, then reboot. No scheduler load, no test dispatch,
    // no shell. Must run before shell_mode_requested() so a future
    // operator-facing shell command cannot accidentally trip the
    // template path. See [`crate::vmm::disk_template`] for the host
    // side that drives this mode.
    if disk_template_mode_requested() {
        let _span = tracing::debug_span!("disk_template_mode").entered();
        let code = run_disk_template_mode();
        // Match the post-test exit semantics: push buffered stdio
        // bytes into the pipe (the forwarder threads then ship them
        // over the bulk port), emit the binary exit code over the
        // bulk data port so the host knows we're done, reboot.
        // `flush()` replaces the broken `tcdrain(1/2)`
        // which returned ENOTTY against the pipe write ends; the
        // forwarder threads aren't joined here, so bytes still in
        // the pipe at reboot time are lost — see the queue task
        // for forwarder-join plumbing.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        crate::vmm::guest_comms::send_exit(code);
        // The bulk-port write inside `send_exit` commits via MMIO
        // before userspace returns from KVM_RUN — the EXIT frame is
        // in the host's port-1 RX buffer the moment `send_exit`
        // returns. No additional wait needed before reboot.
        force_reboot();
    }

    // Shell mode: interactive busybox shell instead of test dispatch.
    if shell_mode_requested() {
        let _shell_span = tracing::debug_span!("shell_mode").entered();
        let console_dev = shell_console_device();
        redirect_all_stdio_to(console_dev);

        // Create busybox applet symlinks.
        {
            let _s = tracing::debug_span!("busybox_install").entered();
            let _ = Command::new("/bin/busybox")
                .args(["--install", "-s", "/bin"])
                .status();
        }

        // Mount devpts so PTY allocation works.
        mount_devpts();

        // Run scheduler enable cmds (from `--ktstr-shell-test=NAME`'s
        // ShellTestDescriptor.scheduler_enable_cmds — Phase B of the
        // KernelBuiltin lifecycle, packed into /sched_enable by the
        // VM builder). Idempotent / safe when the file doesn't exist
        // (returns Ok(())). Mirrors the test-mode wire-up at L1329 so
        // the shell-mode operator drops into the SAME scheduler-loaded
        // environment a test would see — without this, the shell falls
        // through to whatever scheduler the kernel boots with and the
        // banner's "running N enable cmd(s)" claim would be a lie.
        exec_shell_script("/sched_enable");

        // --exec mode: run a command non-interactively instead of
        // dropping into an interactive shell. Inherits stdio from init
        // which redirect_all_stdio_to() already pointed at the console
        // device (virtio-console /dev/hvc0 when available, COM2
        // otherwise). The host stdout writer thread drains virtio TX.
        // Checked before MOTD so exec output is not polluted.
        if let Some(cmd) = shell_exec_cmd() {
            tracing::debug!(cmd = %cmd, "shell exec mode");
            // Disable OPOST on stdout so the tty layer does not
            // convert \n to \r\n. Without this, every newline in
            // command output gains a spurious \r visible to the host.
            let stdout_fd = unsafe { BorrowedFd::borrow_raw(1) };
            if let Ok(mut termios) = tcgetattr(stdout_fd) {
                termios
                    .output_flags
                    .remove(nix::sys::termios::OutputFlags::OPOST);
                let _ = tcsetattr(stdout_fd, SetArg::TCSANOW, &termios);
            }
            // [`with_sigchld_default`] flips SIGCHLD to SIG_DFL
            // for the closure body so `Command::status()` (which
            // calls `waitpid(2)`) reaps the child and reports the
            // real exit code. The `SIG_IGN` disposition installed
            // earlier in [`ktstr_guest_init`] for zombie
            // prevention is restored on closure return — and on
            // panic unwind, via the helper's RAII guard.
            let status = with_sigchld_default(|| {
                Command::new("/bin/busybox")
                    .args(["sh", "-c", &cmd])
                    .status()
            });
            let code = match status {
                Ok(s) => s.code().unwrap_or(1),
                Err(e) => {
                    tracing::error!(err = %e, "ktstr-init: exec failed");
                    1
                }
            };
            // Exit code travels via the bulk data port so it does
            // not pollute captured command output on stdout.
            crate::vmm::guest_comms::send_exec_exit(code as i32);
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            // tcdrain is synchronous on the vCPU exit: when these
            // syscalls return, every byte is already in the host's
            // serial writer Vec (or virtio-console TX path). No
            // additional wait needed before reboot.
            unsafe {
                libc::tcdrain(1);
            }
            unsafe {
                libc::tcdrain(2);
            }
            // Run scheduler disable cmds before reboot — symmetric
            // bracket with /sched_enable above; idempotent when the
            // file doesn't exist.
            exec_shell_script("/sched_disable");
            // Drain stdout/stderr after /sched_disable so any
            // stdout/stderr writes from the disable script (e.g.
            // the `echo > /proc/1/fd/1` marker pattern used by
            // the shell-mode lifecycle e2e fixture) reach host
            // capture before force_reboot triggers
            // device_shutdown. tcdrain bounds the TTY FIFO drain
            // in userspace; the virtio TX ring drain itself
            // happens during the kernel's hvc_close path in
            // device_shutdown. Sysfs-only disable scripts (e.g.
            // `echo 0 > /sys/...`) don't write to the TTY FIFO;
            // tcdrain is a harmless no-op for them. No prior
            // Rust stdout/stderr flush is needed because
            // exec_shell_line writes via fs::write, bypassing
            // Rust's BufWriter. Symmetric with the post-payload
            // drain above that protects the /sched_enable +
            // --exec output bracket.
            unsafe {
                libc::tcdrain(1);
            }
            unsafe {
                libc::tcdrain(2);
            }
            force_reboot();
        }

        // MOTD (printed to console before PTY proxy takes over).
        // Skipped in exec mode (handled above).
        let kernel_version = fs::read_to_string("/proc/version")
            .ok()
            .and_then(|v| v.split_whitespace().nth(2).map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        let mem_mib = fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kib| kib.parse::<u64>().ok())
                .map(|kib| kib / 1024)
        });
        println!("ktstr shell");
        println!("  kernel:    {kernel_version}");
        if let Some(mib) = mem_mib {
            println!("  memory:    {mib} MiB");
        }
        print_topology_line();
        print_includes_line();
        println!("  tools:     busybox (ls, ps, top, dmesg, ip, vi, ...)");
        println!("  mounts:    /proc /sys /dev /sys/fs/cgroup /sys/fs/bpf /tmp");
        println!("             /sys/kernel/debug /sys/kernel/tracing /dev/pts");
        println!("  type `exit` for clean shutdown, Ctrl+A X to force-kill");
        let _ = std::io::stdout().flush();

        // Allocate a PTY pair so busybox sh gets a controlling terminal
        // (required for job control: Ctrl+Z, bg, fg).
        tracing::debug!("spawning interactive shell with PTY");
        spawn_shell_with_pty();

        // Run scheduler disable cmds before reboot — symmetric
        // bracket with /sched_enable. Runs after the operator types
        // `exit` (spawn_shell_with_pty returns when the shell exits).
        exec_shell_script("/sched_disable");
        // Drain stdout/stderr after /sched_disable so any
        // stdout/stderr writes from the disable script reach
        // host capture before force_reboot triggers
        // device_shutdown. The interactive-shell path shares
        // the same race + drain semantics as the exec-mode
        // path above (see that comment for the TTY FIFO vs
        // virtio TX ring + fs::write bypass rationale).
        unsafe {
            libc::tcdrain(1);
        }
        unsafe {
            libc::tcdrain(2);
        }
        force_reboot();
    }

    // Read test args from /args early so Phase 2b can parse
    // --ktstr-probe-stack for probe setup before the scheduler starts.
    let args: Vec<String> = {
        let content = fs::read_to_string("/args").unwrap_or_default();
        let mut a = vec!["/init".to_string()];
        a.extend(content.lines().map(|s| s.to_string()));
        a
    };
    tracing::debug!(args = ?args, "parsed /args");

    // Propagate RUST_BACKTRACE and RUST_LOG from the kernel cmdline to
    // the process environment BEFORE Phase A spawns its probe thread.
    // `std::env::set_var` mutates glibc's `__environ` without locking;
    // calling it while the probe thread is live is UB on Linux.
    crate::test_support::propagate_rust_env_from_cmdline();

    // Phase 2b: Probe Phase A (before scheduler starts).
    // Attaches kprobes + trigger + kernel fexit so the one-shot
    // sched_ext_exit tracepoint is captured even if the scheduler
    // crashes immediately on startup.
    let _s_phase2b = tracing::debug_span!("phase2b_probe_phase_a").entered();
    let probe_phase_a = crate::test_support::start_probe_phase_a(&args);
    let probes_active = probe_phase_a.is_some();
    drop(_s_phase2b);

    // Phase 3: Cgroup parent + Scheduler.
    // Create the cgroup parent directory before starting the scheduler
    // so it exists when the scheduler looks for it.
    let _s_phase3 = tracing::debug_span!("phase3_scheduler_start").entered();
    // Per-test workload-cgroup root. Sourced from
    // `KtstrTestEntry::workload_root_cgroup`. The framework owns
    // this slot; the scheduler never sees it.
    create_workload_root_cgroup_from_file();
    // Per-scheduler cgroup the scheduler process is placed in.
    // Sourced from `Scheduler::cgroup_parent`. mkdir + enable
    // controllers here so the tree is ready when `start_scheduler`
    // spawns the child. Distinct from
    // `create_cgroup_parent_from_sched_args` (which fires only
    // when `--cell-parent-cgroup` is present in `/sched_args` for
    // cell-aware schedulers).
    create_scheduler_cgroup_parent_from_file();
    create_cgroup_parent_from_sched_args();
    exec_shell_script("/sched_enable");
    // Plumb the probe pipeline's `stop` + `output_done` into
    // `start_scheduler` so the early-bail paths (Died / not
    // attached / spawn error) can drain probe JSON to COM2 before
    // calling `force_reboot()`. Without the drain, every path that
    // crashes the scheduler before the test dispatches loses its
    // probe payload to the reboot — exactly the diagnostic the
    // probes were attached to capture.
    let probe_drain = probe_phase_a.as_ref().map(|pa| ProbeDrain {
        stop: pa.pipeline.stop.clone(),
        output_done: pa.pipeline.output_done.clone(),
    });
    let (mut sched_child, sched_log_path) = start_scheduler(probe_drain);
    drop(_s_phase3);

    // Phase 4: hvc0 polling + trace pipe (background threads).
    let _s_phase4 = tracing::debug_span!("phase4_vc_poll").entered();
    let (trace_stop, trace_handle) = start_trace_pipe();
    let vc_poll_stop = start_hvc0_poll(trace_stop.clone());
    drop(_s_phase4);

    // Phase 4b: Scheduler death monitor.
    // Spawn a thread that polls /proc/{pid}. If the scheduler exits during
    // the test, the thread writes MSG_TYPE_SCHED_EXIT via bulk port so the host
    // can detect early death without waiting for the watchdog.
    //
    // When probes are active, suppress COM2 log dump to avoid
    // interleaving with probe JSON output on the same serial port.
    let suppress_com2 = Arc::new(AtomicBool::new(probes_active));
    let probe_output_done = probe_phase_a
        .as_ref()
        .map(|pa| pa.pipeline.output_done.clone());
    // Install the boot-time scheduler-exit monitor handle into
    // the module-level slot via `install_initial_sched_exit_monitor`
    // so the scheduler-lifecycle Op dispatcher in
    // `src/scenario/ops/mod.rs` can swap the monitor across
    // Op::AttachScheduler / DetachScheduler / RestartScheduler /
    // ReplaceScheduler. The earlier local-binding pattern held
    // the SchedExitStop in this stack frame, which made it
    // unreachable from the Op dispatch path. The shutdown cascade
    // below calls `stop_sched_exit_monitor` instead of the
    // pre-refactor local `stop_and_join`. Cloning the Arcs is
    // cheap and the boot start_sched_exit_monitor call retains
    // its original semantics — the only difference is the
    // ownership chain after spawn.
    let boot_stop = start_sched_exit_monitor(
        sched_child.as_ref().map(|c| c.id()),
        sched_log_path.as_deref(),
        suppress_com2.clone(),
        probe_output_done.clone(),
    );
    install_initial_sched_exit_monitor(boot_stop, suppress_com2, probe_output_done);

    // Phase 5: Dispatch.
    let _s_phase5 = tracing::debug_span!("phase5_dispatch").entered();
    tracing::debug!("dispatching test");
    crate::vmm::guest_comms::send_lifecycle(crate::vmm::wire::LifecyclePhase::PayloadStarting, "");
    crate::vmm::guest_comms::send_scenario_start();

    #[cfg(feature = "wprof")]
    let wprof_handle = spawn_wprof_if_configured();

    unsafe { libc::signal(libc::SIGCHLD, libc::SIG_DFL) };
    let code = if let Some(pa) = probe_phase_a {
        crate::test_support::maybe_dispatch_vm_test_with_phase_a(&args, pa).unwrap_or(1)
    } else {
        crate::test_support::maybe_dispatch_vm_test_with_args(&args).unwrap_or(1)
    };
    unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };
    crate::vmm::guest_comms::send_scenario_pause();

    #[cfg(feature = "wprof")]
    if let Some(handle) = wprof_handle
        && let Ok(Some(pb_bytes)) = handle.join()
    {
        crate::vmm::guest_comms::send_wprof_trace(&pb_bytes);
    }

    drop(_s_phase5);

    // Flush test output before teardown. Rust's BufWriter on stdout
    // holds data until flushed; without this the host may not see the
    // test result before reboot.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    crate::test_support::try_flush_profraw();

    // Phase 6: Scheduler cleanup.
    let _s_phase6 = tracing::debug_span!("phase6_cleanup").entered();

    // Stop the sched-exit monitor BEFORE killing the scheduler.
    // Without this ordering, child.kill() makes the scheduler
    // exit, the monitor's pidfd poll wakes, it sees /proc/{pid}
    // gone and emits MSG_TYPE_SCHED_EXIT on the bulk port, the
    // host promotes kill=true, and the BSP exits with ExternalKill
    // before the guest reaches send_exit — producing exit_code=-1
    // on an otherwise clean run.
    //
    // `stop_and_join` sets stop=true (Release), writes the wake
    // eventfd to drop poll wake latency from 250 ms to
    // microseconds, then joins the monitor thread. Joining is
    // event-driven: the monitor's loop checks stop at the top,
    // exits cleanly after `poll(2)` returns, and the join
    // returns. After this call the monitor is guaranteed to have
    // exited without sending MSG_TYPE_SCHED_EXIT, so the
    // subsequent child.kill() cannot trigger the race.
    // Stop the live sched_exit_monitor (whichever scheduler PID it
    // was last installed for — boot or post-Op::Replace) before
    // tearing down the scheduler child below. The slot may be
    // empty if the test ran Op::DetachScheduler without a
    // re-attach; the helper handles that case as a no-op.
    stop_sched_exit_monitor();

    if let Some(ref mut child) = sched_child {
        // On a crash the scheduler is shutting down and flushing its
        // userspace diagnostics to its stderr log. Give it a brief
        // BOUNDED grace to finish writing and exit on its own BEFORE the
        // hard kill, so SIGKILL doesn't truncate that output
        // (`dump_sched_output` below reads the stderr log). Gated on
        // dump_started (the `sched_ext_dump:` tracepoint fires only on an
        // error exit) so clean runs pay nothing; the grace returns early
        // the moment the scheduler exits, and is bounded
        // (`SCHED_KILL_GRACE`) so a userspace hang can't wedge teardown.
        let exited_in_grace =
            scx_dump_started_latch().is_set() && reap_child_bounded(child, SCHED_KILL_GRACE);
        if !exited_in_grace {
            let _ = child.kill();
            // Bounded, evented reap. A SIGKILL'd scheduler normally exits
            // <<1s — post-crash bypass keeps it CFS-schedulable and it is
            // not held in the kernel disable (see `SCHED_REAP_TIMEOUT`).
            // The bound caps the rare case where the process can't take its
            // pending SIGKILL promptly; the VM reboot below reaps any
            // straggler, so cap the wait rather than risk blocking teardown.
            if !reap_child_bounded(child, SCHED_REAP_TIMEOUT) {
                tracing::warn!(
                    ?SCHED_REAP_TIMEOUT,
                    "scheduler did not exit within the reap bound after SIGKILL \
                     (still uninterruptible — unexpected); leaving it for VM reboot to reap"
                );
            }
        }
        if let Some(ref log_path) = sched_log_path {
            dump_sched_output(log_path);
        }
    }
    dump_staged_scheduler_logs();
    exec_shell_script("/sched_disable");

    // Phase 6b: probe finalisation. Now that the scheduler is
    // killed and `/sched_disable` has run, the kernel's
    // `scx_disable_irq_workfn` path runs `scx_claim_exit` which
    // fires `trace_sched_ext_exit`. The probe's tp_btf listener is
    // STILL attached at this point because
    // [`crate::test_support::probe::publish_result_and_collect`]
    // stashed the probe stop+handle into a deferred slot rather
    // than detaching at end-of-dispatch. Draining now means the
    // trigger event lands in the ring buffer, the BSS latch flips,
    // the probe poll loop sees `ktstr_err_exit_detected != 0`, and
    // the readout phase stitches the kprobe events that fired
    // during the actual stall window.
    //
    // The drain is bounded internally (5 s wait for
    // `/sys/kernel/sched_ext/state == disabled`, plus a one-shot
    // `rb.poll(100 ms)` final ringbuf drain inside the probe loop
    // when `bss_triggered` is observed); a non-responding kernel
    // cannot stall teardown. When no probes were stashed
    // (single-phase ctor path or EEVDF runs), the call is a no-op.
    crate::test_support::finalize_probe_after_unwind();

    // Stop remaining background threads.
    if let Some(ref stop) = vc_poll_stop {
        stop.store(true, Ordering::Release);
    }
    stats_relay_stop.signal_stop();

    // Flush COM1 trace data before reboot. The reader thread runs on
    // a poll(POLLIN, 200ms) cadence over a non-blocking trace_pipe fd
    // (see start_trace_pipe), so setting `stop` is what bounds
    // `handle.join()` — the thread observes the flag at the next poll
    // wake and enters its 5s drain window. Effective shutdown latency
    // is up to ~5.2s in the worst case: the 200ms poll cadence elapses
    // before the thread notices the stop flag, then the 5s drain
    // deadline begins. Disabling the tracepoint and writing 0 to
    // `tracing_on` first quiesces the producer side so the drain
    // window terminates promptly: no new events are recorded into the
    // ring buffer, the reader sees POLLIN until the buffer is empty,
    // then poll returns 0 each cycle and the drain_deadline elapses
    // cleanly. Trace events arriving after the 5s deadline are dropped
    // by design — bounded drain is the explicit tradeoff that
    // guarantees cleanup completes (a faulty producer that never
    // pauses cannot wedge teardown).
    //
    // tracing_on=0 alone does NOT wake a trace_pipe reader stuck at
    // `iter->pos == 0` — the kernel wake fires `ring_buffer_wake_waiters`
    // but the trace_pipe wait uses `wait_pipe_cond` (not
    // `rb_wait_once`), and that condition only flips when `iter->closed`
    // or `iter->wait_index` change. The non-blocking + poll design
    // sidesteps this by never blocking in the kernel wait at all.
    // Tier-2 (best-effort ftrace dump): if a sched_ext exit dump started
    // streaming this run, hold the dump tracepoint open until the reader
    // has forwarded its end-marker to COM1 (or the bound elapses) BEFORE
    // disabling it below, so a fast teardown does not disable the
    // tracepoint mid-emit. Only paid when a dump is in flight — clean runs
    // never start one. Returns immediately for a small dump; the bound
    // (`SCX_DUMP_CAPTURE_TIMEOUT`) caps the wait for a LARGE dump, whose
    // per-task content can take tens of seconds to forward over the slow
    // PIO COM1 UART. On that bound this ftrace copy is truncated — but the
    // full dump content is captured independently via the scheduler's
    // stderr log (`dump_sched_output`; scx_utils reads the same kernel
    // `ei->dump`) over the fast bulk port, which is the authoritative
    // copy. Best-effort, not lossless.
    if scx_dump_started_latch().is_set()
        && !scx_dump_complete_latch().wait_timeout(SCX_DUMP_CAPTURE_TIMEOUT)
    {
        tracing::warn!(
            ?SCX_DUMP_CAPTURE_TIMEOUT,
            "sched_ext exit dump did not reach its end-marker within the capture bound \
             before tracepoint teardown; the rendered dump may be truncated"
        );
    }
    let _ = fs::write(TRACE_SCHED_EXT_DUMP_ENABLE, "0");
    if let Some(ref stop) = trace_stop {
        stop.store(true, Ordering::Release);
    }
    let _ = fs::write(TRACE_TRACING_ON, "0");
    if let Some(handle) = trace_handle {
        let _ = handle.join();
    }
    if let Ok(com1) = fs::OpenOptions::new().write(true).open(COM1) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com1.as_raw_fd());
        }
    }

    // Phase 7: Exit.
    // Push buffered stdout/stderr bytes into the pipe write ends so
    // the bulk-port forwarder threads can ship them before reboot.
    // After stdio redirect, fd 1 / fd 2 are pipe write ends
    // (not the COM2 UART) so `tcdrain(1)` would return ENOTTY
    // silently — `flush()` is the equivalent for pipes. The
    // forwarder threads are not joined before `force_reboot`; bytes
    // still resident in the pipe buffer at reboot time are lost
    // (see the queue task for forwarder-join plumbing).
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    // Write exit code via the typed guest API on the bulk data
    // port. The legacy COM2 `SENTINEL_EXIT_PREFIX` fallback is gone
    // — bulk-port backpressure guarantees delivery and the host's
    // `collect_results` walks `guest_messages` for a binary
    // `MSG_TYPE_EXIT` frame as the sole authoritative source.
    crate::vmm::guest_comms::send_exit(code as i32);

    // Drain COM2 UART for any panic-hook bytes that may still be
    // in flight (the panic hook is the one remaining COM2 writer).
    // tcdrain is synchronous on the vCPU exit: when it returns,
    // every byte is already in the host's COM2 writer Vec.
    if let Ok(com2) = fs::OpenOptions::new().write(true).open(COM2) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com2.as_raw_fd());
        }
    }

    force_reboot()
}
