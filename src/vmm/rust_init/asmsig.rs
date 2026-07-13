//! Async-signal-safe formatting helpers and the fatal-signal handler install.
//!
//! Split from rust_init.rs. Self-contained — references `libc` and
//! `std` by fully-qualified path, so no glob from the parent module
//! is needed.

/// Async-signal-safe rendering of `value` as lowercase hex (no `0x`
/// prefix, no leading-zero trim) filling all 16 bytes of `buf`.
/// Returns `&buf[..]`, the full 16-byte slice.
///
/// Used by [`fatal_signal_handler`], where every libc allocator
/// boundary is forbidden — `format!`, `write!`, and even
/// `core::fmt::Display` formatters can pull in heap or thread-local
/// state. A hand-rolled nibble walk over a stack buffer is the only
/// AS-safe way to surface the faulting address.
///
/// 16 hex digits cover the full `u64` range. The caller passes a
/// `[u8; 16]` and uses the returned subslice (always exactly 16
/// bytes) directly.
fn u64_to_hex_asm(value: u64, buf: &mut [u8; 16]) -> &[u8] {
    static HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, slot) in buf.iter_mut().enumerate() {
        let nibble = (value >> ((15 - i) * 4)) & 0xf;
        *slot = HEX[nibble as usize];
    }
    &buf[..]
}

/// AS-safe write of every byte in `bytes` to fd `fd`. Loops on partial
/// writes; bails on the first error or zero-byte return so a closed/
/// faulted fd cannot wedge the handler.
fn write_all_asm(fd: libc::c_int, bytes: &[u8]) {
    let mut off = 0;
    while off < bytes.len() {
        // SAFETY: `write(2)` is async-signal-safe per signal-safety(7)
        // on Linux. `bytes.as_ptr().add(off)` is in-bounds because
        // `off < bytes.len()`. The write is best-effort — any
        // failure short-circuits the loop and the handler proceeds
        // to `reboot(2)`.
        let n = unsafe {
            libc::write(
                fd,
                bytes.as_ptr().add(off) as *const libc::c_void,
                bytes.len() - off,
            )
        };
        if n <= 0 {
            return;
        }
        off += n as usize;
    }
}

/// Async-signal-safe handler for SIGSEGV / SIGBUS / SIGILL.
///
/// The Rust panic hook installed in [`crate::vmm::rust_init::ktstr_guest_init`] does NOT
/// fire for native CPU faults: the kernel raises these signals with
/// `SIG_DFL` disposition, which calls `do_coredump` and terminates
/// the process. Inside guest init that means PID 1 dies, the kernel
/// observes "init exited", and the host sees the VM force-reboot
/// without any guest-side diagnostic on COM2.
///
/// This handler closes the gap by emitting a `PANIC:`-prefixed line
/// — matching the prefix `extract_panic_message` anchors on — that
/// names the signal and the faulting address before driving
/// [`crate::vmm::rust_init::force_reboot`]. The host crash-classification pipeline then
/// surfaces native faults through the same code path as Rust panics.
///
/// Constraints, all enforced inside the handler:
///
/// - Async-signal-safety per `signal-safety(7)`. No `fs::write`, no
///   `format!`, no `Backtrace::force_capture` — all of those touch
///   the heap, locks, or per-thread formatter state. Only `open(2)`,
///   `write(2)`, `tcdrain(2)`, and `reboot(2)` (all in the AS-safe
///   list) are invoked, plus pure stack arithmetic.
/// - No thread-local state. Worker threads spawned later
///   (`hvc0_poll_loop`, `start_trace_pipe`) inherit the parent's
///   sigaction disposition because Linux signal dispositions are
///   process-wide; this handler runs on whichever thread faulted.
/// - Bounded recursion. `SA_RESETHAND` is set so a fault inside this
///   handler reverts to `SIG_DFL`, which terminates immediately
///   instead of looping.
unsafe extern "C" fn fatal_signal_handler(
    sig: libc::c_int,
    info: *mut libc::siginfo_t,
    ctx: *mut libc::c_void,
) {
    // Static prefixes per signal. Hard-coded because signal-name
    // formatting via `strsignal(3)` allocates / touches locale
    // state and is not AS-safe.
    let prefix: &[u8] = match sig {
        libc::SIGSEGV => b"PANIC: fatal signal SIGSEGV at addr 0x",
        libc::SIGBUS => b"PANIC: fatal signal SIGBUS at addr 0x",
        libc::SIGILL => b"PANIC: fatal signal SIGILL at addr 0x",
        _ => b"PANIC: fatal signal (unknown) at addr 0x",
    };

    // Faulting address from `siginfo_t.si_addr`. `siginfo_t` field
    // access in Rust requires going through the libc bindings;
    // `si_addr()` is the canonical accessor that handles the union
    // layout differences between glibc and musl. Falls back to 0
    // when `info` is null. Defensive null check; Linux always
    // populates info for SA_SIGINFO handlers (see kernel/signal.c
    // `force_sig_fault_to_task` → `force_sig_info_to_task` and the
    // arch `setup_rt_frame` paths, which unconditionally pass
    // `&frame->info` to the handler).
    let addr: u64 = if info.is_null() {
        0
    } else {
        // SAFETY: `info` is non-null here; `si_addr()` reads the
        // address-fault arm of the siginfo union, which is the
        // valid arm for SIGSEGV / SIGBUS / SIGILL per the kernel's
        // `force_sig_fault` path (`kernel/signal.c`).
        let p = unsafe { (*info).si_addr() };
        p as u64
    };

    let mut hex_buf = [0u8; 16];
    let hex = u64_to_hex_asm(addr, &mut hex_buf);

    // Faulting INSTRUCTION POINTER from the ucontext, plus this
    // handler's own runtime address as a relocation anchor. `si_addr`
    // alone names the datum (0x0 / a small struct offset for a null
    // deref) but not the CODE that faulted — observed CI init crashes
    // (`SIGSEGV at addr 0x0` / `0x140` before init's first output)
    // were unattributable without the IP. The binary is PIE, so the
    // raw IP is load-slid; printing the runtime address of
    // `fatal_signal_handler` alongside lets one offline lookup
    // recover the link-time IP:
    //   link_ip = nm(binary, "fatal_signal_handler") + (ip - handler)
    // Pure memory reads off the ucontext — AS-safe. gnu-libc layout
    // only (the dual-role binary ships as *-unknown-linux-gnu on both
    // arches); other envs print ip=0, still leaving the anchor.
    let ip: u64 = if ctx.is_null() {
        0
    } else {
        #[cfg(all(target_arch = "x86_64", target_env = "gnu"))]
        // SAFETY: for SA_SIGINFO handlers the third argument is a
        // `ucontext_t *` (sigaction(2)); non-null checked above.
        unsafe {
            (*(ctx as *const libc::ucontext_t)).uc_mcontext.gregs[libc::REG_RIP as usize] as u64
        }
        #[cfg(all(target_arch = "aarch64", target_env = "gnu"))]
        // SAFETY: as above; aarch64 exposes the faulting PC directly.
        unsafe {
            (*(ctx as *const libc::ucontext_t)).uc_mcontext.pc
        }
        #[cfg(not(all(
            any(target_arch = "x86_64", target_arch = "aarch64"),
            target_env = "gnu"
        )))]
        {
            0
        }
    };
    let mut ip_buf = [0u8; 16];
    let ip_hex = u64_to_hex_asm(ip, &mut ip_buf);
    let mut anchor_buf = [0u8; 16];
    let anchor_hex = u64_to_hex_asm(
        fatal_signal_handler as *const () as usize as u64,
        &mut anchor_buf,
    );

    // Open COM2 first (canonical destination), then COM1. Both with
    // `O_WRONLY | O_NONBLOCK` so the open and the `write_all_asm`
    // loop never block on guest-side flow control. `tcdrain(2)`
    // does NOT honor `O_NONBLOCK` — it is a separate ioctl that
    // waits for the kernel tty layer's write queue to drain — but
    // the wait is bounded by UART FIFO drain time (microseconds at
    // worst) because PIO commits each byte inside `KVM_RUN` before
    // userspace returns; the kernel sees its own output queue empty
    // almost immediately after the final `write(2)` returns.
    //
    // SAFETY: `open(2)`, `write(2)`, `tcdrain(2)`, and `close(2)`
    // are all in the signal-safety(7) AS-safe set. The path
    // strings are static C strings with explicit NUL terminators.
    for path in [c"/dev/ttyS1", c"/dev/ttyS0"] {
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_NONBLOCK) };
        if fd < 0 {
            continue;
        }
        write_all_asm(fd, prefix);
        write_all_asm(fd, hex);
        // IP + anchor breadcrumb on the same PANIC line (the host's
        // `extract_panic_message` captures the whole line): see the
        // `ip`/anchor derivation above for the offline attribution
        // formula.
        write_all_asm(fd, b" ip=0x");
        write_all_asm(fd, ip_hex);
        write_all_asm(fd, b" handler=0x");
        write_all_asm(fd, anchor_hex);
        write_all_asm(fd, b"\n");
        // Seal the contract: tcdrain waits for the kernel's output
        // queue to drain before we issue `reboot(2)`. PIO commits
        // per byte so the wait is effectively immediate; tcdrain
        // ignores `O_NONBLOCK` but the drain time is bounded by
        // UART FIFO depth, not by host-side back-pressure.
        unsafe {
            libc::tcdrain(fd);
            libc::close(fd);
        }
    }

    // `reboot(LINUX_REBOOT_CMD_RESTART)` is the AS-safe analogue of
    // `force_reboot()`'s nix wrapper. The syscall does not return
    // on success; if it somehow does (CAP_SYS_BOOT missing,
    // already rebooting), `_exit(1)` ensures the handler does
    // NOT fall through to user code with a corrupt stack /
    // mid-fault state.
    unsafe {
        libc::reboot(libc::LINUX_REBOOT_CMD_RESTART);
        libc::_exit(1);
    }
}

/// Install [`fatal_signal_handler`] for SIGSEGV, SIGBUS, and SIGILL.
///
/// `SA_SIGINFO` makes the handler receive the `siginfo_t *` whose
/// `si_addr` carries the faulting address. `SA_RESETHAND` reverts
/// the disposition to `SIG_DFL` after the first delivery so a fault
/// inside the handler terminates cleanly instead of looping. `SA_ONSTACK`
/// directs the kernel to run the handler on the alternate stack
/// registered via `sigaltstack(2)` below — without it a stack-overflow
/// SIGSEGV faults again on the overflowed stack and the kernel
/// terminates the process before any diagnostic reaches the host.
///
/// `sa_mask` adds SIGSEGV / SIGBUS / SIGILL so that while one fatal-
/// signal handler is executing, the other two cannot interrupt it.
/// Cross-signal nesting (e.g. SIGBUS arriving while the SIGSEGV
/// handler is mid-write to COM2) would scribble interleaved bytes
/// onto the serial output and lose the diagnostic. The signal being
/// delivered is also masked by default; combined with `SA_RESETHAND`
/// a re-fault of the same signal terminates under `SIG_DFL` instead
/// of looping back into this handler.
///
/// Failures are silently ignored: if `sigaction(2)` rejects the
/// install (returns -1), the previous disposition (typically
/// `SIG_DFL`) remains in place — which is exactly the pre-fix
/// behavior. There's no user-visible regression on failure, just
/// the unchanged gap the panic hook also doesn't cover. `mmap(2)` /
/// `sigaltstack(2)` failures are similarly tolerated: the handler
/// stays installed without `SA_ONSTACK`, which only loses the
/// stack-overflow diagnostic — every other fatal-signal path keeps
/// working.
pub(crate) fn install_fatal_signal_handlers() {
    // SAFETY: `std::mem::zeroed::<libc::sigaction>()` produces a
    // valid all-zero `sigaction` (all libc fields are integer or
    // pointer-typed, zero is valid for all of them). The
    // `sa_sigaction` field is then set to a function pointer with
    // the correct `extern "C"` signature, and `sa_flags` is set
    // to a valid combination of POSIX `SA_*` constants.
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    act.sa_sigaction = fatal_signal_handler as *const () as usize;
    act.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
    // Initialize the mask, then add every fatal signal so that one
    // handler in flight cannot be interrupted by another fatal
    // signal — see fn doc for why interleaved handlers corrupt the
    // diagnostic.
    unsafe {
        libc::sigemptyset(&mut act.sa_mask);
        libc::sigaddset(&mut act.sa_mask, libc::SIGSEGV);
        libc::sigaddset(&mut act.sa_mask, libc::SIGBUS);
        libc::sigaddset(&mut act.sa_mask, libc::SIGILL);
    }

    // Allocate and register a signal alternate stack so a stack-
    // overflow SIGSEGV runs the handler on a separate stack instead
    // of faulting again on the overflowed one. `SIGSTKSZ` is the
    // platform's recommended minimum; clamp to 64 KiB so older libc
    // headers (where SIGSTKSZ is 8 KiB) still leave headroom for the
    // backtrace-free handler frame plus `write(2)` / `tcdrain(2)` /
    // `reboot(2)` syscall trampolines.
    //
    // `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` is the AS-safe-allocation
    // analogue to a heap allocation: pages are zero-initialised on
    // first touch, so no separate clear is needed. The mapping is
    // intentionally leaked — `sigaltstack` keeps the kernel pointing
    // at it for the lifetime of the process, and PID 1 never returns
    // from `ktstr_guest_init`.
    //
    // SAFETY: `mmap(2)` and `sigaltstack(2)` are both POSIX-defined
    // syscalls. The pointers / lengths supplied are well-formed
    // (NULL hint, fd=-1 for anonymous mappings, offset=0). Failure
    // returns `MAP_FAILED`; on that path we skip `sigaltstack` and
    // leave `SA_ONSTACK` unset on `sa_flags` — see fn doc for the
    // failure-mode rationale.
    let stack_size = libc::SIGSTKSZ.max(65536);
    let stack = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            stack_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if stack != libc::MAP_FAILED {
        let ss = libc::stack_t {
            ss_sp: stack,
            ss_flags: 0,
            ss_size: stack_size,
        };
        // SAFETY: `ss` is a fully-initialised `stack_t` with a
        // valid mmap'd buffer and matching size. Passing
        // `null_mut()` for `oss` discards the previous alternate
        // stack — PID 1 has no prior alternate stack at this
        // call site (signal handling has not been touched yet).
        unsafe {
            libc::sigaltstack(&ss, std::ptr::null_mut());
        }
        act.sa_flags |= libc::SA_ONSTACK;
    }

    for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL] {
        // SAFETY: `sigaction(2)` with a valid `struct sigaction`
        // and a NULL old-action pointer is well-defined.
        // Failures are silently swallowed (see fn doc).
        let _ = unsafe { libc::sigaction(sig, &act, std::ptr::null_mut()) };
    }
}
