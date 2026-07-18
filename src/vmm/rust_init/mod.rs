/// Rust init (PID 1) for the VM guest.
///
/// When the test binary is
/// packed as `/init` in the initramfs, `ktstr_guest_init()` is called
/// from the ctor when PID 1 is detected.
/// It never returns — it mounts filesystems, then either dispatches
/// a test (start scheduler, run test, reboot) or drops into an
/// interactive shell (when `KTSTR_MODE=shell` is on the kernel
/// cmdline).
pub(crate) use std::fs;
pub(crate) use std::io::{Read, Write};
pub(crate) use std::os::unix::fs::OpenOptionsExt;
pub(crate) use std::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
pub(crate) use std::os::unix::process::{CommandExt, ExitStatusExt};
pub(crate) use std::path::Path;
pub(crate) use std::process::{Child, Command, Stdio};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::OnceLock;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(crate) use crate::sync::Latch;

pub(crate) use nix::mount::{MsFlags, mount};
pub(crate) use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
pub(crate) use nix::pty::openpty;
pub(crate) use nix::sys::reboot::{RebootMode, reboot};
pub(crate) use nix::sys::termios::{SetArg, cfmakeraw, tcgetattr, tcsetattr};

/// COM2 device path for sentinel and diagnostic output.
const COM2: &str = "/dev/ttyS1";
/// COM1 device path for kernel console / trace output.
const COM1: &str = "/dev/ttyS0";
/// Virtio-console device path. Used for shell I/O when available.
const HVC0: &str = "/dev/hvc0";

/// tracefs enable gate for the `sched_ext_dump` tracepoint. Writing
/// `"1"` activates the event, `"0"` deactivates it.
const TRACE_SCHED_EXT_DUMP_ENABLE: &str =
    "/sys/kernel/tracing/events/sched_ext/sched_ext_dump/enable";
/// Global tracefs on/off switch. Writing `"0"` stops new events from
/// being recorded into the ring buffer (`ring_buffer_record_off`); the
/// userspace trace_pipe reader still has to drain whatever is already
/// buffered before reboot. Disabling the producer side first is what
/// makes the reader's drain window terminate — once no new events
/// arrive, poll eventually returns 0 and the drain_deadline elapses.
const TRACE_TRACING_ON: &str = "/sys/kernel/tracing/tracing_on";
/// tracefs streaming endpoint for the active trace. The trace_pipe
/// reader opens this once per boot and forwards every line to COM1.
const TRACE_PIPE: &str = "/sys/kernel/tracing/trace_pipe";

/// sysfs attribute exposing the active sched_ext root scheduler's
/// name. Empty / absent when no scheduler is registered; populated
/// (with a trailing newline) when registration has completed.
/// Kernel-side owner: `kernel/sched/ext.c` creates this via
/// `kobject_init_and_add` under the `sched_ext` kset after
/// `sch->ops.name` is set.
const SYSFS_SCHED_EXT_ROOT_OPS: &str = "/sys/kernel/sched_ext/root/ops";

/// Maximum bytes per `MsgType::Stdout` / `MsgType::Stderr` TLV
/// chunk emitted by the pipe forwarder threads. 4 KiB matches a
/// page-size pipe read; well under the host-side per-frame cap
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably in one frame even with the 16-byte header.
const STDIO_CHUNK_BYTES: usize = 4 * 1024;

/// Bound on [`CurrentSchedulerProcess::reap_bounded_status`]: how long
/// teardown waits for a SIGKILL'd scheduler to exit before giving up and
/// letting the VM reboot reap it. A SIGKILL'd scheduler normally exits <<1s —
/// post-crash bypass
/// keeps it CFS-schedulable, and it is NOT held in the kernel scx disable:
/// its `struct_ops` detach (`bpf_scx_unreg`) only `kthread_flush_work`s
/// the `scx_root_disable` the crash irq_work already kicked, which is
/// ms-scale (bypass + per-task reclass + one `synchronize_rcu` + the BPF
/// `ops.exit`, all fast for these schedulers). The bound is a defensive
/// cap — only a pathological multi-second `ops.exit` or RCU stall could
/// approach it — so teardown caps the wait rather than risk adding such a
/// stall to every crashed-scheduler teardown.
const SCHED_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Grace given to a CRASHED scheduler to finish flushing its userspace
/// diagnostics to stderr and exit on its own BEFORE the hard SIGKILL, so
/// the kill doesn't truncate that output (`dump_sched_output` reads it).
/// Bounded so a userspace hang can't wedge teardown; returns early the
/// moment the scheduler exits. Applied on a crash, detected as EITHER the
/// kernel exit dump having started (dump_started) OR the scheduler process
/// still being alive once sched_ext has gone down (`scx_down` with its pid
/// still owned).
/// Sized for the USERSPACE flush, not the kernel dump: the kernel's scx
/// exit dump is bounded and truncated in-kernel, but the scheduler's
/// userspace flush of it to stderr (plus libbpf teardown) can run past a
/// shorter window, and the SIGKILL then truncates the tail of THAT output.
const SCHED_KILL_GRACE: std::time::Duration = std::time::Duration::from_millis(3000);

/// Bound on how long teardown waits for the exit dump's end-marker to be
/// forwarded to COM1 before disabling the `sched_ext_dump` tracepoint.
/// The kernel builds+emits the whole dump synchronously at crash time
/// (the `scx_disable_irq_workfn` irq path), so a SMALL dump's marker is
/// forwarded well before teardown and the wait returns at once. A LARGE
/// dump (many runnable tasks → scx_dump_state(dump_all_tasks) builds a
/// per-task dump) can take tens of seconds to forward byte-by-byte over
/// the slow PIO COM1 UART; this bound caps that so a big crash dump
/// cannot wedge teardown. On the bound the ftrace copy is truncated; the
/// authoritative full dump is the scheduler stderr log
/// (`dump_sched_output`) over the fast bulk port.
const SCX_DUMP_CAPTURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

mod asmsig;
mod dump;
mod init;
mod kmod;
mod modes;
mod mounts;
mod process;
mod relay;
mod scheduler;
mod topology;
mod verifier_workload;

pub(crate) use asmsig::*;
pub(crate) use dump::*;
pub(crate) use init::*;
pub(crate) use kmod::*;
pub(crate) use modes::*;
pub(crate) use mounts::*;
pub(crate) use process::*;
pub(crate) use relay::*;
pub(crate) use scheduler::*;
pub(crate) use topology::*;

#[cfg(test)]
mod tests;
