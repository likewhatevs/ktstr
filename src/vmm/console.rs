/// 16550A UART emulation via vm-superio.
///
/// Thin wrapper around vm-superio::Serial providing port-based I/O
/// dispatch and output capture. On x86_64, UARTs are accessed via port
/// I/O at COM1 (0x3F8) and COM2 (0x2F8). On aarch64, UARTs are
/// MMIO-mapped. COM1 carries the kernel console, COM2 carries
/// application stdout/stderr.
///
/// x86_64 dispatch uses `handle_in`/`handle_out` (port I/O); aarch64
/// dispatch uses `inner_read`/`inner_write` (MMIO offset).
pub(crate) const COM1_BASE: u16 = 0x3F8;
pub(crate) const COM2_BASE: u16 = 0x2F8;

/// ISA IRQ lines for the two COM ports (x86_64 port I/O only).
#[cfg(target_arch = "x86_64")]
pub(crate) const COM1_IRQ: u32 = 4;
#[cfg(target_arch = "x86_64")]
pub(crate) const COM2_IRQ: u32 = 3;

#[cfg(any(target_arch = "x86_64", test))]
const NUM_REGS: u16 = 8;

/// Hard cap on the captured-output buffer (per Serial). A hostile or
/// runaway guest can drive COM1/COM2 in a tight `outb` loop, pushing
/// one byte per PIO exit into the inner Vec writer. Without a cap the
/// writer grows without bound and the host process eventually OOMs —
/// every Serial maps a single mutex-protected Vec, so even modest
/// guest-side spam (a `yes` loop redirected to /dev/console) doubles
/// the writer's allocation on every realloc step.
///
/// 4 MiB is large enough that no benign workload reaches the cap in
/// practice: a verbose kernel boot log is on the order of 100–500 KiB,
/// and the heaviest scheduler-trace tests observe peak captures under
/// 2 MiB. Guests that exceed this cap are exhibiting unbounded
/// behaviour the host must not allow to consume host memory.
const OUTPUT_CAP_BYTES: usize = 4 * 1024 * 1024;
/// Target length to trim back to once [`OUTPUT_CAP_BYTES`] is reached.
/// Trimming to a value strictly below the cap (rather than exactly to
/// the cap) amortises the O(N) `Vec::drain(0..N)` cost across the next
/// `OUTPUT_CAP_BYTES - OUTPUT_TRIM_TARGET` byte writes — without a
/// gap, every subsequent byte would trigger another drain.
const OUTPUT_TRIM_TARGET: usize = 3 * 1024 * 1024;

/// EventFd-based interrupt trigger. Signals the eventfd on each
/// interrupt, which KVM's irqfd mechanism delivers to the guest as
/// the configured IRQ line assertion.
struct EventFdTrigger(vmm_sys_util::eventfd::EventFd);

impl vm_superio::Trigger for EventFdTrigger {
    type E = std::io::Error;
    fn trigger(&self) -> Result<(), Self::E> {
        self.0.write(1)
    }
}

/// Serial wrapper around vm-superio::Serial with output capture.
///
/// The captured-output writer is capped at [`OUTPUT_CAP_BYTES`].
/// When the cap is reached, [`Serial::enforce_output_cap`] drains
/// the oldest bytes back down to [`OUTPUT_TRIM_TARGET`]. This is the
/// host-side defence against a hostile or runaway guest spamming
/// COM1/COM2 — without the cap the inner `Vec<u8>` writer would
/// grow unboundedly until the host process OOMs.
pub struct Serial {
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    base: u16,
    inner: vm_superio::Serial<EventFdTrigger, vm_superio::serial::NoEvents, Vec<u8>>,
    /// Optional notifier for "captured-output buffer grew." When set,
    /// every register write that pushes a byte into the inner writer
    /// (DATA register stores, sans DLAB) bumps this eventfd's counter
    /// so an external consumer (e.g. the interactive dmesg drain
    /// thread) can `epoll_wait` for new bytes instead of sleep-polling
    /// `drain_output`. Spurious wakes are harmless: a consumer that
    /// observes the eventfd then drains an empty buffer simply
    /// re-blocks. `None` when no consumer has installed a notifier
    /// (the run_vm path that consumes output via the COM2 stdout
    /// writer thread does not need the eventfd).
    data_evt: Option<std::sync::Arc<vmm_sys_util::eventfd::EventFd>>,
    /// Cursor for [`Self::output_contains`]: `(needle, scanned_len)`
    /// where `scanned_len` is the writer-buffer length that was
    /// already searched for `needle` without a hit. The next
    /// `output_contains(needle)` call resumes from
    /// `scanned_len.saturating_sub(needle.len() - 1)` so a needle
    /// straddling the prior cursor still matches. Reset on
    /// [`Self::drain_output`] / `Self::clear` (writer shrinks) and
    /// invalidated when the caller passes a different needle.
    ///
    /// Without this cache the freeze-coordinator's post-thaw COM2
    /// marker poll re-scanned the whole writer buffer on every
    /// iteration of the grace-window loop — O(N²) over the buffer
    /// growth pattern produced by a guest spamming COM2.
    contains_cursor: Option<(Vec<u8>, usize)>,
    /// First guest kernel-panic line, latched at ingest. Set by
    /// [`Self::latch_panic_if_present`] the instant a "Kernel panic" /
    /// "Attempted to kill init" marker appears in a DATA store — BEFORE
    /// [`Self::enforce_output_cap`] can trim it — so a panic that later
    /// scrolls past the cap is never lost. Polled+taken by `run_bsp_loop`
    /// (via [`Self::take_panic`]) to abort the run fast with the cause,
    /// instead of waiting out the watchdog or the 24h interactive-shell
    /// timeout (which produced cryptic "test timed out" failures on guest
    /// OOM). Only COM1 (the kernel console, `console=ttyS0`) latches this;
    /// COM2 carries app output, not kernel panics. This assumes the kernel
    /// console is COM1 (the default cmdline); a test that reroutes it (e.g.
    /// `console=hvc0` to the virtio-console) would bypass the latch.
    panic_latch: Option<String>,
    /// Captured-output cap in bytes. Initialized to
    /// [`OUTPUT_CAP_BYTES`] by every constructor; a test-only
    /// constructor (`new_with_cap`) overrides it so the
    /// cap/trim logic can be exercised at KiB scale without filling
    /// the production 4 MiB cap one byte at a time. Read by
    /// [`Self::enforce_output_cap`].
    output_cap_bytes: usize,
    /// Trim target in bytes -- the writer is drained back to this
    /// length when it exceeds [`Self::output_cap_bytes`].
    /// Initialized to [`OUTPUT_TRIM_TARGET`] by every constructor.
    output_trim_target: usize,
}

impl Default for Serial {
    /// Constructs a `Serial` bound to `COM1_BASE` (ISA port 0x3F8 —
    /// the standard PC/AT primary-serial address that ktstr's
    /// guest kernel cmdline routes its console to). Delegates to
    /// [`Self::new`] for the eventfd + buffer wiring; see that
    /// constructor for the I/O-port semantics that drive every
    /// kernel `console=ttyS0` write through this device.
    fn default() -> Self {
        Self::new(COM1_BASE)
    }
}

impl Serial {
    pub fn new(base: u16) -> Self {
        let evt = vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
            .expect("failed to create serial eventfd");
        Serial {
            base,
            inner: vm_superio::Serial::new(EventFdTrigger(evt), Vec::new()),
            data_evt: None,
            contains_cursor: None,
            panic_latch: None,
            output_cap_bytes: OUTPUT_CAP_BYTES,
            output_trim_target: OUTPUT_TRIM_TARGET,
        }
    }

    /// Test-only constructor with an overridable output cap + trim
    /// target, so the cap/trim/cursor logic can be validated at KiB
    /// scale instead of filling the production [`OUTPUT_CAP_BYTES`]
    /// (4 MiB) one byte per `handle_out`. The cap math is
    /// threshold-independent (relative drain to `trim_target`), so a
    /// small cap exercises the identical code path. `cap > trim` must
    /// hold (the same invariant the prod consts satisfy).
    #[cfg(test)]
    fn new_with_cap(base: u16, cap: usize, trim: usize) -> Self {
        debug_assert!(cap > trim, "cap must exceed trim target");
        let mut s = Self::new(base);
        s.output_cap_bytes = cap;
        s.output_trim_target = trim;
        s
    }

    /// Install (or replace) the captured-output notifier. The returned
    /// eventfd is bumped each time a byte is appended to the inner
    /// writer, letting a consumer block in `epoll_wait` instead of
    /// sleep-polling. Counter mode (not semaphore) — a single read
    /// returns the accumulated count and resets it.
    pub fn install_data_evt(
        &mut self,
    ) -> std::io::Result<std::sync::Arc<vmm_sys_util::eventfd::EventFd>> {
        let evt = std::sync::Arc::new(vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)?);
        self.data_evt = Some(std::sync::Arc::clone(&evt));
        Ok(evt)
    }

    /// Return the interrupt eventfd for registering with KVM's irqfd.
    pub fn irq_evt(&self) -> &vmm_sys_util::eventfd::EventFd {
        &self.inner.interrupt_evt().0
    }

    /// Bump the captured-output eventfd if the writer grew during
    /// the most recent inner.write call. Helper to keep the offset/
    /// register-decoding ladder out of every call site.
    #[inline]
    fn signal_if_writer_grew(&self, pre_len: usize) {
        if self.inner.writer().len() > pre_len
            && let Some(evt) = &self.data_evt
        {
            // Discarded Result: this is `EventFd::write(1)`. Counter
            // mode (libc::EFD_NONBLOCK | counter, see
            // [`Self::install_data_evt`]) — the only failure modes
            // are (a) counter overflow at u64::MAX - 1, which would
            // require the guest to write 2^64 bytes between
            // consumer reads, and (b) EBADF if the consumer dropped
            // its end. Both are recoverable: the consumer will see
            // the buffer growth on its next `output()`/`drain_output`
            // call regardless of the eventfd bump.
            let _ = evt.write(1);
        }
    }

    /// Enforce [`OUTPUT_CAP_BYTES`] on the captured-output writer.
    ///
    /// Called after every byte that vm-superio appends to the inner
    /// Vec writer (the `handle_out` and `inner_write` paths). When
    /// the writer exceeds the cap, drains the oldest
    /// `writer.len() - OUTPUT_TRIM_TARGET` bytes via
    /// `Vec::drain(0..N)`, leaving `OUTPUT_TRIM_TARGET` bytes of
    /// the most recent output in place.
    ///
    /// **Cursor invalidation.** [`Self::contains_cursor`] caches
    /// `(needle, scanned_len)` where `scanned_len` is an absolute
    /// writer-buffer offset. A drain shifts every retained byte's
    /// position backwards by the drained count, so the cached
    /// offset no longer corresponds to any position in the new
    /// buffer. Reset the cursor to `None`, matching the invariant
    /// pinned by [`Self::drain_output`] and `Self::clear`.
    ///
    /// **Visibility.** The trim is logged at debug level (matching
    /// the pattern used elsewhere in this file for guest-driven
    /// anomalies) with the byte count, so a host operator can
    /// observe runaway guest output without flooding the log: each
    /// trim retires `OUTPUT_CAP_BYTES - OUTPUT_TRIM_TARGET` bytes
    /// of capture, so the log rate is bounded to one entry per
    /// `OUTPUT_CAP_BYTES - OUTPUT_TRIM_TARGET` bytes of overflow.
    #[inline]
    fn enforce_output_cap(&mut self) {
        let cap = self.output_cap_bytes;
        let target = self.output_trim_target;
        let writer = self.inner.writer_mut();
        let len = writer.len();
        if len <= cap {
            return;
        }
        let drop_count = len - target;
        writer.drain(0..drop_count);
        self.contains_cursor = None;
        tracing::debug!(
            base = self.base,
            cap = cap,
            target = target,
            dropped = drop_count,
            "captured-output buffer exceeded cap; oldest bytes dropped",
        );
    }

    /// Latch the first guest kernel-panic line seen in freshly-captured
    /// output. Called from both DATA-store paths (`handle_out` on
    /// x86, `inner_write` on aarch64) AFTER the byte is appended
    /// and BEFORE [`Self::enforce_output_cap`] — so the latch is set at
    /// ingest and a panic that later scrolls past the cap is never lost.
    ///
    /// Scans only the recent tail: a kernel panic banner is emitted in one
    /// printk burst, so the marker lands in the freshly-written bytes. The
    /// scan is therefore a bounded constant per byte (not O(buffer)), and
    /// a no-op once latched. Matches "Kernel panic - not syncing" (the
    /// `panic()` banner prefix from kernel/panic.c — covers every kernel
    /// panic) or "Attempted to kill init" (the init-death case, typically
    /// guest OOM); the canonical OOM banner "Kernel panic - not syncing:
    /// Attempted to kill init! exitcode=0x…" contains both (`do_exit` in
    /// kernel/exit.c appends the ` exitcode=0x%08x` tail before the '\n').
    /// The first marker is anchored to the `- not syncing` form so an
    /// unrelated log line mentioning the bare words "Kernel panic" cannot
    /// false-trigger an abort.
    ///
    /// We scrape the console for the cause rather than emulating a pvpanic
    /// device (cloud-hypervisor's mechanism): pvpanic signals only a binary
    /// panic event — not the cause STRING this harness reports — does not
    /// self-abort the VM, and would add a guest `CONFIG_PVPANIC` dependency.
    /// Reusing the serial console already wired needs none of that.
    ///
    /// Only a newline-TERMINATED marker line is latched. The guest emits the
    /// console one byte per exit, so the byte at which the marker first
    /// completes has only the line prefix in the buffer — latching there
    /// would record "Kernel panic - not syncing" and drop the actionable
    /// "Attempted to kill init" tail of the OOM banner. `run_bsp_loop` polls
    /// `take_panic` every exit and aborts on the first `Some`, so that prefix
    /// would become the reported cause. Waiting for the following '\n'
    /// captures the whole line. A panic that emits the marker but no trailing
    /// '\n' (e.g. a fault mid-printk) is not fast-latched — the run falls
    /// back to the watchdog timeout.
    ///
    /// Bound: the marker AND its terminating '\n' must lie within the same
    /// ~256 B tail. A panic line longer than that (marker >256 B before its
    /// '\n') is NOT fast-latched — by the time the '\n' is buffered the
    /// marker has scrolled out of the tail, so no terminated marker segment
    /// is found and the run falls back to the watchdog. The canonical OOM
    /// banner is ~70 B, well under 256 B, so only a pathological panic
    /// message hits this.
    #[inline]
    fn latch_panic_if_present(&mut self, pre_len: usize) {
        if self.panic_latch.is_some() {
            return;
        }
        let writer = self.inner.writer();
        // No DATA byte was appended (a register/DLAB store) — nothing new
        // to scan. Mirrors the writer-grew gate in `signal_if_writer_grew`.
        if writer.len() <= pre_len {
            return;
        }
        const TAIL: usize = 256;
        const MARKERS: [&[u8]; 2] = [b"Kernel panic - not syncing", b"Attempted to kill init"];
        let start = writer.len().saturating_sub(TAIL);
        let tail = &writer[start..];
        let hit = MARKERS
            .iter()
            .any(|m| tail.windows(m.len()).any(|w| w == *m));
        if !hit {
            return;
        }
        // Latch the marker line for a human-readable cause, but only once it
        // is newline-TERMINATED (see doc): the byte completing the marker has
        // only the line prefix ("...not syncing") buffered, and
        // `run_bsp_loop` would report that truncated cause. A marker segment
        // followed by another segment has its terminating '\n' in the tail —
        // capture the whole line then; otherwise wait for the rest.
        let mut segs = tail.split(|&b| b == b'\n').peekable();
        while let Some(seg) = segs.next() {
            if !MARKERS
                .iter()
                .any(|m| seg.windows(m.len()).any(|w| w == *m))
            {
                continue;
            }
            let terminated = segs.peek().is_some();
            if terminated {
                let line = String::from_utf8_lossy(seg).trim().to_string();
                self.panic_latch = Some(if line.is_empty() {
                    "guest kernel panic".to_string()
                } else {
                    line
                });
            }
            return;
        }
    }

    /// Take the latched guest kernel-panic line, if any (clears the latch).
    /// Polled by `run_bsp_loop` each iteration to abort the run fast with
    /// the cause rather than waiting out the timeout.
    pub fn take_panic(&mut self) -> Option<String> {
        self.panic_latch.take()
    }

    /// Handle a port I/O write from the guest. Returns true if the port
    /// is in this serial's range.
    ///
    /// 16550A registers are byte-wide. The Linux serial driver issues
    /// `outb` (one byte per access), so KVM hands us `data.len() == 1`
    /// in normal operation. Multi-byte writes (`outw`/`outl`) to a UART
    /// register are anomalous: real hardware would step the access across
    /// adjacent registers in an implementation-defined way, and we have
    /// no way to recover correct semantics from a guest that violates
    /// the kernel's own driver contract. Drop the access rather than
    /// silently corrupting register state.
    ///
    /// Reference VMM behaviour for non-byte-width PIO:
    /// - firecracker (`BusDevice::write` for `SerialWrapper`, `devices/legacy/serial.rs`):
    ///   bumps `METRICS.missed_write_count` (silent metric, no log).
    /// - cloud-hypervisor (`BusDevice::write` for `Serial`, `devices/src/legacy/serial.rs`):
    ///   silent early-return, no metric, no log.
    /// - libkrun (`BusDevice::write` for `Serial`, x86_64): silent early-return, no
    ///   metric, no log.
    /// - qemu (`hw/char/serial.c`): the `serial_io_ops` MemoryRegionOps
    ///   declare `.impl.min_access_size = .impl.max_access_size = 1`,
    ///   so qemu's memory dispatch decomposes a wide access into N
    ///   one-byte calls before reaching the device. qemu's serial
    ///   never observes a multi-byte access; the kernel decomposes
    ///   rather than rejecting (a different design choice — qemu
    ///   would happily service `outw` to a UART by stepping it across
    ///   two adjacent registers).
    ///
    /// We diverge from all four references: a guest issuing a
    /// non-byte access has violated the Linux driver contract, and we
    /// emit a debug-level trace rather than a silent counter. Logging
    /// is debug (not warn) because the port number is guest-driven —
    /// a hostile guest issuing wide PIO in a tight loop would otherwise
    /// flood the host log.
    #[cfg(any(target_arch = "x86_64", test))]
    pub fn handle_out(&mut self, port: u16, data: &[u8]) -> bool {
        let Some(offset) = self.offset(port) else {
            return false;
        };
        if data.len() != 1 {
            tracing::debug!(
                base = self.base,
                port,
                offset,
                len = data.len(),
                "serial PIO write with non-byte width dropped",
            );
            return true;
        }
        let pre = self.inner.writer().len();
        // Discarded Result: vm-superio's `Serial::write` returns
        // `Result<(), Error<EventFdTrigger::E>>` (vm-superio-0.8.1
        // src/serial.rs::write). The two error variants reachable
        // from this call site are both recoverable / unactionable:
        //
        // - `Error::IOError(io::Error)` arises from the inner
        //   writer's `write_all` / `flush` (non-loopback DATA write
        //   path). The writer is `Vec<u8>` (constructed in
        //   `Serial::new` above); std's `impl Write for Vec<u8>`
        //   always returns `Ok` and only fails by panicking on
        //   allocation failure (OOM). The [`Self::enforce_output_cap`]
        //   call below bounds the writer's capacity to
        //   `OUTPUT_CAP_BYTES`, so allocation pressure is
        //   host-controlled rather than guest-controlled.
        // - `Error::Trigger(io::Error)` arises from the THRE
        //   eventfd write inside `thr_empty_interrupt` (DATA writes)
        //   or the RDA interrupt inside `received_data_interrupt`
        //   (loopback DATA writes). Counter-mode eventfds only fail
        //   on counter overflow (would require the guest to pend
        //   2^64 unacked interrupts) or EBADF (KVM dropped the
        //   irqfd). Both are recoverable: the guest re-polls IIR
        //   and the next interrupt fires on the subsequent write.
        //
        // `Error::FullFifo` is unreachable from `Serial::write` —
        // vm-superio guards loopback DATA pushes with
        // `in_buffer.len() < FIFO_SIZE` and silently drops on
        // overflow rather than returning the variant; `FullFifo`
        // only escapes from `enqueue_raw_bytes`, which we do not
        // call here.
        let _ = self.inner.write(offset, data[0]);
        self.signal_if_writer_grew(pre);
        self.latch_panic_if_present(pre);
        self.enforce_output_cap();
        true
    }

    /// Handle a port I/O read from the guest. Returns true if the port
    /// is in this serial's range.
    ///
    /// 16550A registers are byte-wide. The Linux serial driver issues
    /// `inb` (one byte per access). Multi-byte reads (`inw`/`inl`) to a
    /// UART register are anomalous: real hardware would step the access
    /// across adjacent registers in an implementation-defined way, and
    /// some UART register reads have side effects we cannot replay
    /// coherently across a stepped access. In vm-superio's `Serial::read`
    /// (vm-superio-0.8.1, src/serial.rs):
    /// - DATA (offset 0): pops `in_buffer.pop_front()` — RX FIFO byte
    ///   consumed.
    /// - IIR (offset 2): calls `reset_iir()` — clears the pending
    ///   interrupt identification.
    /// - LSR (offset 5), MSR (offset 6): vm-superio returns the
    ///   stored register value with no clear-on-read mutation. The
    ///   16550A datasheet specifies LSR error-bits and MSR delta-bits
    ///   are cleared on read, but vm-superio does not implement that
    ///   side effect.
    ///
    /// Servicing only the first register would fire DATA/IIR side
    /// effects on the wrong access and feed the guest one byte of real
    /// state followed by `data[1..]` bytes the guest treats as
    /// adjacent-register reads but that we never sourced. Drop the
    /// access rather than risk that.
    ///
    /// Reference VMM behaviour for non-byte-width PIO:
    /// - firecracker (`BusDevice::read` for `SerialWrapper`, `devices/legacy/serial.rs`):
    ///   bumps `METRICS.missed_read_count` (silent metric, no log).
    /// - cloud-hypervisor (`BusDevice::read` for `Serial`, `devices/src/legacy/serial.rs`):
    ///   silent early-return, no metric, no log.
    /// - libkrun (`BusDevice::read` for `Serial`, x86_64): silent early-return, no
    ///   metric, no log.
    /// - qemu (`hw/char/serial.c`): the `serial_io_ops` MemoryRegionOps
    ///   declare `.impl.min_access_size = .impl.max_access_size = 1`,
    ///   so qemu's memory dispatch decomposes a wide access into N
    ///   one-byte calls before reaching the device — qemu services
    ///   the access by stepping it across registers (a different
    ///   design choice from the reject-and-drop pattern above).
    ///
    /// We diverge from all four references: emit a debug-level trace
    /// rather than a silent counter. Logging is debug (not warn)
    /// because the port number is guest-driven — a hostile guest
    /// issuing wide PIO in a tight loop would otherwise flood the
    /// host log.
    #[cfg(any(target_arch = "x86_64", test))]
    pub fn handle_in(&mut self, port: u16, data: &mut [u8]) -> bool {
        let Some(offset) = self.offset(port) else {
            return false;
        };
        if data.len() != 1 {
            tracing::debug!(
                base = self.base,
                port,
                offset,
                len = data.len(),
                "serial PIO read with non-byte width dropped",
            );
            return true;
        }
        data[0] = self.inner.read(offset);
        true
    }

    /// Write a byte to a register at the given offset.
    /// Used by MMIO dispatch where the offset is computed externally.
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn inner_write(&mut self, offset: u8, byte: u8) {
        let pre = self.inner.writer().len();
        // Discarded Result: same rationale as the x86_64 `handle_out`
        // call site above. vm-superio's `Serial::write` over a
        // `Vec<u8>` writer is effectively infallible — `IOError`
        // only arises from allocation panic (host-controlled by
        // [`Self::enforce_output_cap`] below) and `Trigger` is a
        // recoverable eventfd-bump failure that the guest will
        // re-observe on the next interrupt poll.
        let _ = self.inner.write(offset, byte);
        self.signal_if_writer_grew(pre);
        self.latch_panic_if_present(pre);
        self.enforce_output_cap();
    }

    /// Read a byte from a register at the given offset.
    /// Used by MMIO dispatch where the offset is computed externally.
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn inner_read(&mut self, offset: u8) -> u8 {
        self.inner.read(offset)
    }

    /// Test helper — queue input bytes for host->guest
    /// communication.
    #[cfg(test)]
    pub fn queue_input(&mut self, bytes: &[u8]) {
        let _ = self.inner.enqueue_raw_bytes(bytes);
    }

    /// Return and clear accumulated output. O(1) via buffer swap.
    pub fn drain_output(&mut self) -> Vec<u8> {
        // Drain shrinks the writer to zero — invalidate the
        // output_contains cursor so the next search starts fresh
        // rather than skipping bytes that the new buffer hasn't
        // accumulated yet.
        self.contains_cursor = None;
        std::mem::take(self.inner.writer_mut())
    }

    /// Get all captured output as a string.
    pub fn output(&self) -> String {
        String::from_utf8_lossy(self.inner.writer()).to_string()
    }

    /// Return true when the captured output contains `needle` as a
    /// contiguous byte sequence. Resumes from the prior cursor so
    /// repeat polls amortize to O(N) over the buffer growth instead
    /// of O(N²) per call.
    ///
    /// Used by the freeze coordinator's post-thaw COM2 marker poll,
    /// which calls this in a tight loop while the writer grows from
    /// guest emissions during the grace window. The cursor caches
    /// `(needle, scanned_len)` after a miss; the next call with the
    /// same needle skips the prefix already scanned and only
    /// inspects the newly-appended tail (plus a `needle.len() - 1`
    /// byte overlap so a needle straddling the cursor still
    /// matches).
    ///
    /// A different needle invalidates the cache (we must scan the
    /// full buffer for the new pattern). [`Self::drain_output`] /
    /// `Self::clear` also reset the cursor — both shrink the
    /// writer, after which the absolute byte offsets in the cursor
    /// no longer correspond to any positions in the new buffer.
    #[allow(dead_code)]
    pub fn output_contains(&mut self, needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        let writer: &[u8] = self.inner.writer();
        if writer.len() < needle.len() {
            // Below-needle-length means no match is possible; do not
            // touch the cursor — the writer will grow on subsequent
            // guest writes and the next call will scan from zero
            // (cursor stays None).
            return false;
        }
        // Resume position: `cursor` is the writer length already
        // scanned for THIS needle. Start `(needle.len() - 1)` bytes
        // earlier so a needle straddling the cursor (last byte after
        // it, first byte before it) still matches. Saturate at zero
        // for the empty/first-call case.
        let resume_from = match &self.contains_cursor {
            Some((cached, scanned_len)) if cached.as_slice() == needle => {
                scanned_len.saturating_sub(needle.len() - 1)
            }
            _ => 0,
        };
        // Defensive bound: a stale cursor (caller wrote into the
        // writer through a path that didn't go through the public
        // API, then the writer somehow shrank) could push
        // resume_from past writer.len(). Clamp so the slice index
        // never panics.
        let resume_from = resume_from.min(writer.len().saturating_sub(needle.len() - 1));
        let tail = &writer[resume_from..];
        let found = tail.windows(needle.len()).any(|w| w == needle);
        if found {
            // Cursor is no longer useful — a hit short-circuits any
            // future poll. Clear it so a subsequent call with a
            // different needle doesn't carry stale state.
            self.contains_cursor = None;
        } else {
            // Cache the writer length we just scanned so the next
            // call resumes from there.
            self.contains_cursor = Some((needle.to_vec(), writer.len()));
        }
        found
    }

    /// Clear captured output.
    #[cfg(test)]
    pub fn clear(&mut self) {
        // Mirror drain_output: clearing shrinks the writer, so the
        // output_contains cursor is no longer valid.
        self.contains_cursor = None;
        self.inner.writer_mut().clear();
    }

    #[cfg(any(target_arch = "x86_64", test))]
    fn offset(&self, port: u16) -> Option<u8> {
        if port >= self.base && port < self.base + NUM_REGS {
            Some((port - self.base) as u8)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Register offsets for test readability
    const DATA: u16 = 0;
    const IER: u16 = 1;
    const IIR: u16 = 2;
    const FCR: u16 = 2;
    const LCR: u16 = 3;
    const MCR: u16 = 4;
    const LSR: u16 = 5;
    const MSR: u16 = 6;
    const SCR: u16 = 7;

    // Bit constants for tests
    const IER_RDI: u8 = 0x01;
    const IER_THRI: u8 = 0x02;
    const IIR_FIFO_BITS: u8 = 0xC0;
    const IIR_NONE: u8 = 0x01;
    const IIR_THRI: u8 = 0x02;
    const IIR_RDI: u8 = 0x04;
    const LCR_DLAB: u8 = 0x80;
    const LSR_DR: u8 = 0x01;
    const LSR_THRE: u8 = 0x20;
    const LSR_TEMT: u8 = 0x40;
    const MCR_DTR: u8 = 0x01;
    const MCR_RTS: u8 = 0x02;
    const MCR_OUT1: u8 = 0x04;
    const MCR_OUT2: u8 = 0x08;
    const MCR_LOOP: u8 = 0x10;
    const MSR_CTS: u8 = 0x10;
    const MSR_DSR: u8 = 0x20;
    const MSR_RI: u8 = 0x40;
    const MSR_DCD: u8 = 0x80;
    const FIFO_SIZE: usize = 0x40;

    #[test]
    fn new_serial_defaults() {
        let s = Serial::default();
        assert!(s.output().is_empty());
    }

    #[test]
    fn write_thr_captures() {
        let mut s = Serial::default();
        assert!(s.handle_out(COM1_BASE, b"H"));
        assert!(s.handle_out(COM1_BASE, b"i"));
        assert_eq!(s.output(), "Hi");
    }

    #[test]
    fn write_thr_multi_byte_drops() {
        let mut s = Serial::default();
        // Multi-byte PIO writes to a UART register are anomalous —
        // the Linux serial driver only issues byte-wide accesses.
        // handle_out drops the access (after logging at debug level)
        // so that bytes 1..N never get silently mapped to the wrong
        // register offset. Diverges from firecracker (bumps a silent
        // metric) and cloud-hypervisor / libkrun (silent return) by
        // emitting a trace instead.
        s.handle_out(COM1_BASE, b"Hello");
        assert_eq!(s.output(), "");
    }

    #[test]
    fn read_multi_byte_drops() {
        let mut s = Serial::default();
        // Multi-byte PIO reads (inw/inl) to a UART register are
        // anomalous — DATA pops the RX FIFO and IIR clears the pending
        // interrupt, side effects that cannot be coherently stepped
        // across adjacent registers. handle_in drops the access
        // (after logging at debug level) and returns true.
        //
        // Use DATA so the test exercises a register with an observable
        // side effect: queue a byte, attempt a 2-byte read, then
        // perform a single-byte DATA read. If the multi-byte read
        // were serviced, the queued byte would be popped from the RX
        // FIFO and the subsequent 1-byte read would return 0 instead
        // of 0x42. Reading from LSR (the prior version of this test)
        // would not distinguish rejection from servicing because
        // vm-superio's LSR read has no side effects.
        s.queue_input(&[0x42]);
        let mut buf = [0xCDu8; 2];
        assert!(s.handle_in(COM1_BASE + DATA, &mut buf));
        assert_eq!(
            buf,
            [0xCD, 0xCD],
            "dropped read must not write any byte of the buffer",
        );
        let mut single = [0u8; 1];
        assert!(s.handle_in(COM1_BASE + DATA, &mut single));
        assert_eq!(
            single[0], 0x42,
            "FIFO byte must remain after a dropped multi-byte read",
        );
    }

    #[test]
    fn read_lsr_thre_temt() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut buf);
        assert_ne!(buf[0] & LSR_THRE, 0, "THR should report empty");
        assert_ne!(buf[0] & LSR_TEMT, 0, "transmitter should report empty");
    }

    #[test]
    fn iir_no_interrupt_at_start() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut buf);
        assert_eq!(buf[0] & 0x0F, IIR_NONE, "no interrupt pending");
        assert_ne!(buf[0] & IIR_FIFO_BITS, 0, "FIFO bits should be set");
    }

    #[test]
    fn iir_read_resets_all_interrupts() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + IER, &[IER_RDI | IER_THRI]);
        s.handle_out(COM1_BASE + DATA, b"X");
        s.queue_input(&[0x42]);

        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & IIR_NONE, 0, "should have interrupt pending");

        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_NONE, "cleared after read");
    }

    #[test]
    fn write_read_lcr() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + LCR, &[0x1B]);
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + LCR, &mut buf);
        assert_eq!(buf[0], 0x1B);
    }

    #[test]
    fn write_read_scr() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + SCR, &[0x42]);
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + SCR, &mut buf);
        assert_eq!(buf[0], 0x42);
    }

    #[test]
    fn out_of_range_port() {
        let mut s = Serial::default();
        assert!(!s.handle_out(0x100, b"x"));
        let mut buf = [0u8; 1];
        assert!(!s.handle_in(0x100, &mut buf));
    }

    #[test]
    fn clear_output() {
        let mut s = Serial::default();
        for c in b"test" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert_eq!(s.output(), "test");
        s.clear();
        assert!(s.output().is_empty());
    }

    #[test]
    fn dlab_baud_divisor() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + LCR, &[0x03 | LCR_DLAB]);
        s.handle_out(COM1_BASE + DATA, &[0x01]);
        s.handle_out(COM1_BASE + IER, &[0x00]);
        let mut lo = [0u8; 1];
        let mut hi = [0u8; 1];
        s.handle_in(COM1_BASE + DATA, &mut lo);
        s.handle_in(COM1_BASE + IER, &mut hi);
        assert_eq!(lo[0], 0x01);
        assert_eq!(hi[0], 0x00);
        s.handle_out(COM1_BASE + LCR, &[0x03]);
        s.handle_out(COM1_BASE + DATA, b"X");
        assert_eq!(s.output(), "X");
    }

    #[test]
    fn loopback_mode() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP]);
        s.handle_out(COM1_BASE + DATA, &[0xAA]);
        assert!(s.output().is_empty(), "loopback should not produce output");
        let mut lsr = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0] & LSR_DR, 0, "data should be ready");
        let mut data = [0u8; 1];
        s.handle_in(COM1_BASE + DATA, &mut data);
        assert_eq!(data[0], 0xAA);
    }

    #[test]
    fn loopback_msr_mapping() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP]);
        let mut msr = [0u8; 1];
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0], 0x00, "no MCR outputs set");

        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP | MCR_OUT2 | MCR_RTS]);
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0], MSR_DCD | MSR_CTS, "OUT2->DCD, RTS->CTS");

        s.handle_out(
            COM1_BASE + MCR,
            &[MCR_LOOP | MCR_DTR | MCR_RTS | MCR_OUT1 | MCR_OUT2],
        );
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0], MSR_DSR | MSR_CTS | MSR_RI | MSR_DCD);
    }

    #[test]
    fn thr_interrupt_lifecycle() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + IER, &[IER_THRI]);
        s.handle_out(COM1_BASE + DATA, b"A");
        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_THRI, "THR empty interrupt");
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_NONE, "cleared after read");
    }

    #[test]
    fn receive_interrupt() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + IER, &[IER_RDI]);
        s.queue_input(&[0x42]);
        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_RDI, "receive data interrupt");
        let mut data = [0u8; 1];
        s.handle_in(COM1_BASE + DATA, &mut data);
        assert_eq!(data[0], 0x42);
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_NONE, "cleared");
    }

    #[test]
    fn queue_input_respects_fifo_size() {
        let mut s = Serial::default();
        let big = vec![0xAA; FIFO_SIZE + 10];
        s.queue_input(&big);
        // vm-superio caps at FIFO_SIZE
        assert_eq!(s.inner.fifo_capacity(), 0);
    }

    #[test]
    fn kernel_autoconfig_ier_test() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_out(COM1_BASE + IER, &[0x00]);
        s.handle_in(COM1_BASE + IER, &mut buf);
        assert_eq!(buf[0] & 0x0F, 0x00);
        s.handle_out(COM1_BASE + IER, &[0x0F]);
        s.handle_in(COM1_BASE + IER, &mut buf);
        assert_eq!(buf[0] & 0x0F, 0x0F);
    }

    #[test]
    fn kernel_autoconfig_loopback_test() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP | MCR_OUT2 | MCR_RTS]);
        let mut msr = [0u8; 1];
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0] & 0xF0, MSR_DCD | MSR_CTS);
    }

    #[test]
    fn kernel_lsr_safety_check() {
        let mut s = Serial::default();
        let mut lsr = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0], 0xFF, "LSR must not be 0xFF");
    }

    #[test]
    fn kernel_fifo_detection() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + FCR, &[0x01]);
        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0xC0, 0xC0, "16550A FIFO");
    }

    #[test]
    fn com2_write_captures() {
        let mut s = Serial::new(COM2_BASE);
        assert!(s.handle_out(COM2_BASE, b"H"));
        assert!(s.handle_out(COM2_BASE, b"i"));
        assert_eq!(s.output(), "Hi");
    }

    #[test]
    fn com2_rejects_com1_port() {
        let mut s = Serial::new(COM2_BASE);
        assert!(!s.handle_out(COM1_BASE, b"x"));
        let mut buf = [0u8; 1];
        assert!(!s.handle_in(COM1_BASE, &mut buf));
    }

    #[test]
    fn com1_rejects_com2_port() {
        let mut s = Serial::default();
        assert!(!s.handle_out(COM2_BASE, b"x"));
        let mut buf = [0u8; 1];
        assert!(!s.handle_in(COM2_BASE, &mut buf));
    }

    #[test]
    fn com2_loopback_msr() {
        let mut s = Serial::new(COM2_BASE);
        s.handle_out(COM2_BASE + MCR, &[MCR_LOOP | MCR_OUT2 | MCR_RTS]);
        let mut msr = [0u8; 1];
        s.handle_in(COM2_BASE + MSR, &mut msr);
        assert_eq!(msr[0], MSR_DCD | MSR_CTS);
    }

    #[test]
    fn com2_lsr_defaults() {
        let mut s = Serial::new(COM2_BASE);
        let mut lsr = [0u8; 1];
        s.handle_in(COM2_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0] & LSR_THRE, 0);
        assert_ne!(lsr[0] & LSR_TEMT, 0);
    }

    #[test]
    fn dual_serial_isolation() {
        let mut com1 = Serial::new(COM1_BASE);
        let mut com2 = Serial::new(COM2_BASE);
        com1.handle_out(COM1_BASE, b"A");
        com2.handle_out(COM2_BASE, b"B");
        assert_eq!(com1.output(), "A");
        assert_eq!(com2.output(), "B");
    }

    #[test]
    fn queue_input_overflow_caps() {
        let mut s = Serial::new(COM1_BASE);
        let data = vec![0x42; FIFO_SIZE + 100];
        s.queue_input(&data);
        assert_eq!(s.inner.fifo_capacity(), 0);
    }

    #[test]
    fn dlab_rapid_transitions() {
        let mut s = Serial::new(COM1_BASE);
        s.handle_out(COM1_BASE + LCR, &[0x80]);
        s.handle_out(COM1_BASE + DATA, &[0x01]);
        s.handle_out(COM1_BASE + LCR, &[0x03]);
        s.handle_out(COM1_BASE + DATA, b"Z");
        assert_eq!(s.output(), "Z");
    }

    #[test]
    fn write_multi_byte_output() {
        let mut s = Serial::new(COM1_BASE);
        for c in b"HELLO" {
            s.handle_out(COM1_BASE + DATA, &[*c]);
        }
        assert_eq!(s.output(), "HELLO");
    }

    #[test]
    fn msr_default_not_loopback() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + MSR, &mut buf);
        // vm-superio defaults MSR to DEFAULT_MODEM_STATUS (DCD).
        // Kernel autoconfig probes MSR in loopback mode, so default doesn't matter.
        assert_eq!(buf[0] & 0x80, 0x80);
    }

    #[test]
    fn mcr_default_value() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + MCR, &mut buf);
        // vm-superio defaults MCR to MCR_OUT2 (0x08).
        assert_eq!(buf[0], MCR_OUT2);
    }

    #[test]
    fn drain_output_returns_and_clears() {
        let mut s = Serial::default();
        for c in b"hello" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        let drained = s.drain_output();
        assert_eq!(drained, b"hello");
        assert!(s.output().is_empty(), "buffer should be empty after drain");
    }

    #[test]
    fn drain_output_empty() {
        let mut s = Serial::default();
        let drained = s.drain_output();
        assert!(drained.is_empty());
    }

    #[test]
    fn drain_output_incremental() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE, b"A");
        s.handle_out(COM1_BASE, b"B");
        let first = s.drain_output();
        assert_eq!(first, b"AB");

        s.handle_out(COM1_BASE, b"C");
        let second = s.drain_output();
        assert_eq!(second, b"C");
    }

    #[test]
    fn queue_input_pub_accessible() {
        let mut s = Serial::default();
        s.queue_input(&[0x41, 0x42]);
        let mut lsr = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0] & LSR_DR, 0, "data should be ready after queue_input");
    }

    #[test]
    fn output_contains_empty_buffer() {
        let mut s = Serial::default();
        assert!(!s.output_contains(b"x"));
        // Empty needle is vacuously contained.
        assert!(s.output_contains(b""));
    }

    #[test]
    fn output_contains_finds_marker() {
        let mut s = Serial::default();
        for c in b"prefix===END===suffix" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(s.output_contains(b"===END==="));
        assert!(s.output_contains(b"prefix"));
        assert!(s.output_contains(b"suffix"));
        assert!(!s.output_contains(b"missing"));
    }

    #[test]
    fn output_contains_needle_longer_than_buffer() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE, b"a");
        assert!(!s.output_contains(b"abcdef"));
    }

    #[test]
    fn output_contains_resumes_after_growth() {
        // Cursor pattern: first call misses, then bytes arrive that
        // contain the needle, then the second call must find it.
        // This is the freeze-coord polling shape — repeat scans while
        // the buffer grows from guest emissions.
        let mut s = Serial::default();
        for c in b"prelude " {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(!s.output_contains(b"MARKER"));
        for c in b"MARKER appears" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(s.output_contains(b"MARKER"));
    }

    #[test]
    fn output_contains_finds_needle_straddling_cursor() {
        // Adversarial case: the cursor advanced just into the
        // beginning of what becomes the needle. The next call must
        // back up `needle.len() - 1` bytes so the straddle is
        // detected even though the prior scan scanned past the
        // first byte.
        let mut s = Serial::default();
        for c in b"abcdMA" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        // First poll: writer is "abcdMA"; needle "MARKER" is not
        // present. Cursor caches scanned_len=6.
        assert!(!s.output_contains(b"MARKER"));
        for c in b"RKER!" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        // Writer is now "abcdMARKER!". Resume from
        // 6 - (6 - 1) = 1, scan "bcdMARKER!" — must hit.
        assert!(s.output_contains(b"MARKER"));
    }

    #[test]
    fn output_contains_different_needle_invalidates_cursor() {
        // A different needle must scan the full buffer, not skip a
        // prefix that was scanned for an earlier (different) needle.
        let mut s = Serial::default();
        for c in b"foobarbaz" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        // Prime the cursor with one needle (miss).
        assert!(!s.output_contains(b"missing"));
        // A different needle that lives in the prefix must be found.
        assert!(s.output_contains(b"foo"));
    }

    #[test]
    fn output_contains_drain_resets_cursor() {
        // drain_output shrinks the writer to zero. A subsequent
        // output_contains call must scan the new buffer from byte 0,
        // not from a stale cursor that points past the (empty)
        // buffer.
        let mut s = Serial::default();
        for c in b"abcdef" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(!s.output_contains(b"XYZ"));
        let _ = s.drain_output();
        // Re-fill with bytes that contain XYZ. The cached cursor
        // would have scanned_len=6; without invalidation, the new
        // search would skip into out-of-range territory.
        for c in b"...XYZ..." {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(s.output_contains(b"XYZ"));
    }

    #[test]
    fn output_contains_clear_resets_cursor() {
        // clear() shrinks the writer the same way drain_output does;
        // sibling test pinning the same invariant for the
        // test-only entry point.
        let mut s = Serial::default();
        for c in b"abcdef" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(!s.output_contains(b"XYZ"));
        s.clear();
        for c in b"...XYZ..." {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(s.output_contains(b"XYZ"));
    }

    #[test]
    fn output_contains_repeat_hit_stable() {
        // A needle that is already present must still be reported as
        // present on a second call. The cursor-clearing logic on hit
        // means the second call rescans from zero — verify that
        // produces the same answer.
        let mut s = Serial::default();
        for c in b"prefix MARKER suffix" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(s.output_contains(b"MARKER"));
        assert!(s.output_contains(b"MARKER"));
    }

    /// Writer length must stay bounded by its cap no matter
    /// how many bytes the guest pushes. Drives `handle_out` past the
    /// cap and asserts the post-write length never exceeds the cap.
    /// Pre-fix, the inner `Vec<u8>` would grow to the full byte count
    /// the test loop wrote -- a hostile guest in production would push
    /// the host into OOM.
    #[test]
    fn output_cap_bounds_writer_length() {
        // Small cap (4:3 ratio mirrors the prod 4MiB:3MiB) so the
        // threshold-independent trim logic is exercised in ~6K writes
        // instead of filling the 4 MiB prod cap one byte at a time.
        const CAP: usize = 4096;
        const TRIM: usize = 3072;
        let mut s = Serial::new_with_cap(COM1_BASE, CAP, TRIM);
        // Push enough bytes to trigger several drain cycles. Crossing
        // the cap twice exercises the case where the trimmed writer
        // grows back up to the cap and trims again. Each handle_out
        // call writes a single DATA byte through the THR path.
        let total = CAP + 2 * (CAP - TRIM);
        for i in 0..total {
            assert!(s.handle_out(COM1_BASE, b"x"));
            assert!(
                s.inner.writer().len() <= CAP,
                "writer must never exceed cap; got {} at iter {}",
                s.inner.writer().len(),
                i,
            );
        }
        // After the burst, the writer is between TRIM and CAP -- the
        // most recent trim left exactly TRIM bytes, then we wrote more.
        let final_len = s.inner.writer().len();
        assert!(
            final_len >= TRIM,
            "final length {} below trim target {}",
            final_len,
            TRIM,
        );
        assert!(
            final_len <= CAP,
            "final length {} above cap {}",
            final_len,
            CAP,
        );
    }

    /// Cap enforcement must drop OLDEST bytes (FIFO), keeping the most
    /// recent guest output. Use a unique pattern at the start of the
    /// stream and a different pattern at the tail; after the cap is
    /// breached, the head pattern must be gone and the tail pattern
    /// must remain.
    #[test]
    fn output_cap_drops_oldest_bytes() {
        const CAP: usize = 4096;
        const TRIM: usize = 3072;
        let mut s = Serial::new_with_cap(COM1_BASE, CAP, TRIM);
        // Head marker -- write CAP - TRIM bytes of "H" so a single
        // drain will remove all of them.
        let head_count = CAP - TRIM;
        for _ in 0..head_count {
            assert!(s.handle_out(COM1_BASE, b"H"));
        }
        // Body filler -- write TRIM bytes of "B" so the writer is
        // exactly at CAP afterwards.
        for _ in 0..TRIM {
            assert!(s.handle_out(COM1_BASE, b"B"));
        }
        assert_eq!(
            s.inner.writer().len(),
            CAP,
            "writer should be exactly at the cap before the trigger byte",
        );
        // One more byte triggers the trim: len is now CAP+1, so the
        // drain removes len-TRIM bytes off the front -- all the "H"s
        // plus one "B" -- leaving TRIM bytes: the tail of "B"s plus
        // the trigger byte.
        assert!(s.handle_out(COM1_BASE, b"T"));
        let writer = s.inner.writer();
        assert!(
            writer.len() <= CAP,
            "post-trim length must be bounded by cap",
        );
        assert!(
            !writer.contains(&b'H'),
            "all 'H' bytes (oldest) should have been drained",
        );
        assert_eq!(
            *writer.last().expect("buffer must be non-empty"),
            b'T',
            "trigger byte must be retained at the tail",
        );
    }

    /// Cap enforcement must invalidate `contains_cursor` because the
    /// cursor's `scanned_len` is an absolute writer-buffer offset that
    /// no longer corresponds to any position once bytes are drained
    /// from the front. Without invalidation, a subsequent
    /// `output_contains` call would resume from a stale offset and
    /// could miss a needle that lives in the retained tail.
    #[test]
    fn output_cap_invalidates_contains_cursor() {
        const CAP: usize = 4096;
        const TRIM: usize = 3072;
        let mut s = Serial::new_with_cap(COM1_BASE, CAP, TRIM);
        // Prime the cursor with a miss so it caches scanned_len.
        for c in b"prelude " {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert!(!s.output_contains(b"NEEDLE"));
        assert!(
            s.contains_cursor.is_some(),
            "miss should populate the cursor",
        );
        // Now push past the cap to force a trim. The cursor's
        // scanned_len would be ~8 (length of "prelude "); after the
        // trim, the writer's leading bytes are different content but
        // the same absolute offsets, so the stale cursor would skip
        // the wrong region. Cap enforcement clears the cursor.
        let total = CAP + (CAP - TRIM);
        for _ in 0..total {
            assert!(s.handle_out(COM1_BASE, b"x"));
        }
        assert!(
            s.contains_cursor.is_none(),
            "cap enforcement must clear the contains_cursor",
        );
    }

    // ---- guest panic latch (BspExitReason::GuestPanic surface) -------

    #[test]
    fn panic_latch_captures_oom_banner() {
        // The canonical guest-OOM banner contains both markers; the latch
        // must capture the marker line verbatim (trimmed).
        let mut s = Serial::default();
        // Mirror the real wire: do_exit() appends " exitcode=0x%08x" before
        // the '\n' (kernel/exit.c), so the line the latch sees is the full
        // banner including the exitcode tail.
        for &b in
            b"[ 0.5] Kernel panic - not syncing: Attempted to kill init! exitcode=0x00000009\n"
        {
            assert!(s.handle_out(COM1_BASE, &[b]));
        }
        assert_eq!(
            s.take_panic().as_deref(),
            Some("[ 0.5] Kernel panic - not syncing: Attempted to kill init! exitcode=0x00000009"),
        );
    }

    #[test]
    fn panic_latch_anchor_rejects_bare_words() {
        // A log line mentioning the bare words "Kernel panic" (without
        // "- not syncing") must NOT latch — the anchor prevents a false
        // abort of a healthy run.
        let mut s = Serial::default();
        for &b in b"note: the Kernel panic handler was registered\n" {
            assert!(s.handle_out(COM1_BASE, &[b]));
        }
        assert_eq!(s.take_panic(), None);
    }

    #[test]
    fn panic_latch_survives_output_cap_trim() {
        // Cap-immunity (the load-bearing claim): the latch is set at
        // ingest BEFORE enforce_output_cap, so a panic that later scrolls
        // past the cap window is still recoverable even after
        // the writer trims the panic bytes away.
        let mut s = Serial::new_with_cap(COM1_BASE, 4096, 3072);
        for &b in b"Kernel panic - not syncing: early oops\n" {
            assert!(s.handle_out(COM1_BASE, &[b]));
        }
        // Grow the writer past the cap (bulk -- handle_out is 1-byte/call)
        // then trim; the trim drops the OLDEST bytes (the panic line at
        // the front), proving the writer no longer holds it.
        s.inner.writer_mut().extend(std::iter::repeat_n(b'.', 4097));
        s.enforce_output_cap();
        assert!(
            !s.output().contains("Kernel panic"),
            "the panic bytes must have been trimmed out of the writer",
        );
        assert_eq!(
            s.take_panic().as_deref(),
            Some("Kernel panic - not syncing: early oops"),
            "the latch must survive the cap trim",
        );
    }

    #[test]
    fn panic_latch_take_clears() {
        let mut s = Serial::default();
        for &b in b"Kernel panic - not syncing: boom\n" {
            assert!(s.handle_out(COM1_BASE, &[b]));
        }
        assert!(s.take_panic().is_some());
        assert_eq!(s.take_panic(), None, "take clears the latch");
    }
}
