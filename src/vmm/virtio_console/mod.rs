//! Three-port virtio-console with inline MMIO transport.
//!
//! Eight virtqueues per virtio-v1.2 §5.3.5 with `VIRTIO_CONSOLE_F_MULTIPORT`:
//!   q0 in0  — host→guest, port 0 (console / hvc0 stdin)
//!   q1 out0 — guest→host, port 0 (console / hvc0 stdout)
//!   q2 c_ivq — host→guest control (PORT_ADD, PORT_OPEN, etc.)
//!   q3 c_ovq — guest→host control (DEVICE_READY, PORT_READY, PORT_OPEN ack)
//!   q4 in1  — host→guest, port 1 (snapshot reply payloads from the
//!            freeze coordinator; see `queue_input_port1`)
//!   q5 out1 — guest→host, port 1 (bulk TLV stream)
//!   q6 in2  — host→guest, port 2 (scheduler-stats requests from the
//!            host's [`super::sched_stats::SchedStatsClient`]; see
//!            `queue_input_port2`)
//!   q7 out2 — guest→host, port 2 (scheduler-stats responses; raw
//!            byte passthrough, no TLV)
//!
//! Port 0 carries the interactive console (stdout/stdin via `/dev/hvc0`).
//! Port 1 carries the TLV stream written by
//! `guest_comms::send_*` — exit code, test result, per-payload
//! metrics, raw payload outputs, profraw, scheduler exit
//! notifications, stimulus events, scenario start/end markers.
//! Port 2 is a transparent byte pipe: the host pushes scx_stats
//! request bytes; the guest's relay thread forwards them to
//! `/var/run/scx/root/stats` and pumps the socket response back via
//! port 2 TX. scx_stats is already newline-delimited JSON so no
//! framing layer is added.
//! Crash payloads travel over COM2. Backpressure is asymmetric:
//!   * Guest→host TX (port 1, port 2): the host's `add_used` rate
//!     gates the guest's writes; when the host lags, the guest
//!     blocks in `wait_port_writable` instead of dropping. Per-call
//!     drain is also capped (`TX_PER_CALL_MAX`) so a hostile guest
//!     cannot grow the host accumulator without bound on a single
//!     notify.
//!   * Host→guest RX (port 0 + port 1 + port 2): the per-port
//!     `pending_rx` accumulators are unbounded by design — the host
//!     alone produces these bytes (kernel scheduler signals, terminal
//!     paste, snapshot replies, stats requests), so a hostile guest
//!     cannot grow them; losing a host→guest byte would silently
//!     strand a wake signal or truncate a reply, which is worse than
//!     a host-side OOM. The per-call CHAIN drain is capped
//!     (`RX_CHAINS_PER_CALL_MAX`) so a hostile guest publishing many
//!     zero-progress descriptor chains cannot hold the vCPU MMIO
//!     handler in `drain_pending_rx` for an unbounded number of
//!     iterations on a single notify.
//!
//! Features: `VIRTIO_F_VERSION_1 | VIRTIO_CONSOLE_F_MULTIPORT`.
//! Config space: `cols=0, rows=0, max_nr_ports=3, emerg_wr=0` (cols/rows
//! valid only with F_SIZE which we do not advertise; the kernel reads
//! `max_nr_ports` via `virtio_cread_feature(F_MULTIPORT, max_nr_ports)`,
//! offset 4 in `struct virtio_console_config`).
//!
//! MMIO register layout per virtio-v1.2 §4.2.2. Interrupt delivery via
//! irqfd (eventfd → KVM GSI). TX data on port 0 or port 1 signals
//! `tx_evt`; TX data on port 2 signals a separate `stats_tx_evt` so
//! the host's [`super::sched_stats::SchedStatsClient`] wakes only on
//! its own port without contending with the freeze coordinator's
//! port-1 drain path.

use std::collections::VecDeque;

use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FAILED, VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_ids::VIRTIO_ID_CONSOLE;

/// Multiport feature bit per `include/uapi/linux/virtio_console.h`.
/// virtio-bindings 0.2.7 does not expose virtio_console.h constants
/// (the crate's per-arch `bindings/` tree only carries blk/config/
/// gpu/ids/input/mmio/net/ring/scsi), so the spec-defined value lives
/// here as a single source of truth.
const VIRTIO_CONSOLE_F_MULTIPORT: u32 = 1;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INTERRUPT_STATUS,
    VIRTIO_MMIO_MAGIC_VALUE, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, VIRTIO_MMIO_QUEUE_AVAIL_LOW,
    VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY,
    VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX, VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH, VIRTIO_MMIO_QUEUE_USED_LOW,
    VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};
use virtio_queue::{Queue, QueueT};
use vm_memory::{Bytes, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;
use zerocopy::{FromBytes, IntoBytes};

const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page).
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// RX wake byte: host requested a SysRq-D dump. The guest's
/// `hvc0_poll_loop` blocks on `/dev/hvc0`, scans every drained byte
/// for this value, and triggers SysRq-D directly via
/// `/proc/sysrq-trigger` when it is observed. Distinct from
/// `SIGNAL_VC_SHUTDOWN` and `SIGNAL_BPF_WRITE_DONE` so stack traces
/// and tcpdump-style captures can distinguish the trigger source.
pub const SIGNAL_VC_DUMP: u8 = 0xD1;

/// RX wake byte: host pushed a graceful-shutdown request through
/// the virtio-console RX queue.
pub const SIGNAL_VC_SHUTDOWN: u8 = 0xD3;

/// RX wake byte: host's `bpf-map-write` thread finished applying
/// every queued `bpf_map_write` to the BPF maps inside the guest's
/// kernel. The guest's `hvc0_poll_loop` recognises the byte and
/// sets the `bpf_map_write_done` latch so a scenario blocked on
/// [`crate::scenario::Ctx::wait_for_map_write`] resumes. Replaces
/// the legacy SHM signal-slot rendezvous (host writes slot 0, guest
/// blocks on slot 0) with a virtio-console wake byte. Host side:
/// `host_comms::request_bpf_map_write_done`.
pub const SIGNAL_BPF_WRITE_DONE: u8 = 0xBF;

/// RX wake byte: the host freeze coordinator has ADOPTED its
/// kernel-symbol accessor (`owned_accessor` is now `Some`), so a
/// failure dump captured from this point renders real BPF map values
/// instead of placeholders. The guest's `hvc0_poll_loop` recognises
/// the byte and sets the `accessor_ready` latch so a scenario blocked
/// in [`crate::scenario::ops::await_accessor_ready`] resumes and
/// triggers its stall only once the dump path is fully armed. Host side:
/// `host_comms::request_accessor_ready`, pushed at the coordinator's
/// accessor-adoption point.
pub const SIGNAL_ACCESSOR_READY: u8 = 0xAC;

/// RX wake byte: the host freeze coordinator has stamped its periodic-
/// capture PREREQS ready — KASLR published AND both accessors adopted,
/// the exact triad the periodic boundary window anchors on
/// (`periodic_prereqs_ready_ns`). A periodic-capture guest (one whose
/// host set `KTSTR_AWAIT_PERIODIC_READY=1` from `num_snapshots > 0`)
/// blocks its `send_scenario_start` on the matching latch so the
/// capture WINDOW opens only once prereqs hold — otherwise, on a slow
/// host where the KASLR publish (guest KERN_ADDRS handshake / its
/// background retry) lands mid-workload, the window
/// `[max(anchor, prereqs_ready), anchor + duration]` collapses from the
/// start side and few/no boundaries fire. Distinct from
/// [`SIGNAL_ACCESSOR_READY`] (0xAC), which fires on accessor adoption
/// ALONE — insufficient here because it can precede the KASLR publish
/// on aarch64 (the accessor build is MMU-only, decoupled from
/// KERN_ADDRS). Host side: `host_comms::request_periodic_prereqs_ready`.
pub const SIGNAL_PERIODIC_READY: u8 = 0xAD;

/// RX wake byte: the host has drained every artifact required for the
/// current wprof run. A normal run requires the terminal WprofTrace frame;
/// an auto-repro run additionally requires PROBE_OUTPUT_END. Guest init waits
/// on the matching latch after probe finalisation and before reboot, closing
/// the virtio-reset race where a successful bulk write was discarded before
/// the host consumed its descriptor.
pub const SIGNAL_WPROF_ARTIFACTS_RECEIVED: u8 = 0xAE;

// `NUM_PORTS` lives in [`super::wire`]; re-exported here so existing
// call sites keep working. Port 0 = console (hvc0); port 1 = named bulk
// TLV stream; port 2 = named scheduler-stats relay. Three ports →
// eight queues per virtio-v1.2 §5.3.5
// (`2 + 2 * num_ports`).
pub use super::wire::NUM_PORTS;

const NUM_QUEUES: usize = 2 + 2 * NUM_PORTS as usize;
const QUEUE_MAX_SIZE: u16 = 256;

// Per port_id_to_queue_idx in libkrun (mirrored here):
//   port 0: rx=0, tx=1
//   control: c_ivq=2 (host→guest), c_ovq=3 (guest→host)
//   port N>=1: rx = 2+2N, tx = 2+2N+1
// So port 1: rx=4, tx=5; port 2: rx=6, tx=7.
const PORT0_RXQ: usize = 0;
const PORT0_TXQ: usize = 1;
const C_IVQ: usize = 2; // host pushes control msgs to guest
const C_OVQ: usize = 3; // guest sends control msgs to host
const PORT1_RXQ: usize = 4;
const PORT1_TXQ: usize = 5;
const PORT2_RXQ: usize = 6;
const PORT2_TXQ: usize = 7;

/// Maximum bytes accepted from a single TX descriptor. The kernel's
/// virtio-console driver sends PAGE_SIZE chunks; this cap prevents a
/// malformed descriptor (len=0xFFFFFFFF) from triggering a ~4GB alloc.
const TX_DESC_MAX: usize = 32 * 1024;

/// Maximum cumulative bytes accepted by a single `process_tx`
/// call. The per-descriptor `TX_DESC_MAX` cap bounds individual
/// descriptors, but a hostile guest can publish thousands of valid
/// descriptors back-to-back and grow the per-port `tx_buf` without
/// bound. Capping the per-call drain at 256 KiB keeps the per-vCPU
/// MMIO-handler latency budget bounded — once the cap is hit we
/// stop popping chains and let the next QUEUE_NOTIFY drain the
/// rest. Backpressure on the guest's TX queue is the natural
/// consequence: a chain that has not been add_used yet stays in
/// the avail ring for the next call.
const TX_PER_CALL_MAX: usize = 256 * 1024;

/// Maximum control-queue chains drained per `process_control_tx`
/// call. The c_ovq's payload is a fixed 8-byte
/// `VirtioConsoleControl` frame — a hostile guest publishing
/// thousands of small chains would otherwise let one notify hold the
/// vCPU thread in `process_control_tx` for an unbounded duration and
/// grow the `events` Vec without bound. Mirrors the TX byte-cap
/// pattern: chains beyond the cap stay in the avail ring for the
/// next QUEUE_NOTIFY. 32 is enough headroom for the legitimate
/// handshake (DEVICE_READY + per-port PORT_READY + per-port
/// PORT_OPEN = ~5 events) with margin while still bounding the
/// adversarial case.
const CONTROL_CHAINS_PER_CALL_MAX: usize = 32;

/// Maximum host→guest RX chains drained per `drain_pending_rx`
/// call. Unlike TX (byte-driven via `TX_PER_CALL_MAX`), RX progress
/// is chain-shaped: each chain absorbs
/// `min(pending_rx_len, sum_of_write_only_desc_lens)` bytes. A
/// hostile guest publishing many zero-length write-only descriptors
/// (or chains lacking any write-only desc — unusual but legal)
/// makes `consumed_offset` stay 0; the `drain(..0)` at the bottom
/// of the loop is then a no-op and the outer
/// `while !pending_rx.is_empty()` reissues `pop_descriptor_chain`
/// without progress until the avail ring is exhausted. With
/// `QUEUE_MAX_SIZE = 256` chains × 256 descriptors per chain that's
/// ~65k iterations per notify, parked on a vCPU thread that is
/// expected to bound MMIO-handler latency. Cap drains at 64 chains
/// per call: legitimate traffic posts a small number of multi-KB
/// chains (kernel virtio-console driver allocates PAGE_SIZE buffers
/// per chain for hvc0, larger for the bulk port); 64 is well
/// above any single-notify legitimate fan-out while still bounding
/// the adversarial latency. Remaining chains stay in the avail
/// ring for the next QUEUE_NOTIFY (or the next host-side push).
const RX_CHAINS_PER_CALL_MAX: usize = 64;

/// Status bits required before each phase.
const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// ----- virtio-console control protocol -----------------------------
//
// `VirtioConsoleControl` and its u16 event discriminants live in
// [`super::wire`]. The constants here are convenience aliases for
// `ControlEvent::*.wire_value()` so the existing call sites read
// the same as the kernel uapi names; new code should prefer the
// typed `ControlEvent` enum directly. The wire format is 8 bytes
// little-endian: id (u32), event (u16), value (u16). LE on the
// wire is x86_64 / aarch64 native.

pub use super::wire::VirtioConsoleControl;

pub const VIRTIO_CONSOLE_DEVICE_READY: u16 = super::wire::ControlEvent::DeviceReady.wire_value();
pub const VIRTIO_CONSOLE_PORT_ADD: u16 = super::wire::ControlEvent::PortAdd.wire_value();
// PORT_REMOVE and RESIZE are kernel uapi event ids the lib does not
// yet generate or consume — kept as named constants so the public
// surface mirrors `enum virtio_console_event` 1:1. `#[allow(dead_code)]`
// matches the `KVM_INTERESTING_STATS` pattern in `result.rs`.
#[allow(dead_code)]
pub const VIRTIO_CONSOLE_PORT_REMOVE: u16 = super::wire::ControlEvent::PortRemove.wire_value();
pub const VIRTIO_CONSOLE_PORT_READY: u16 = super::wire::ControlEvent::PortReady.wire_value();
pub const VIRTIO_CONSOLE_CONSOLE_PORT: u16 = super::wire::ControlEvent::ConsolePort.wire_value();
#[allow(dead_code)]
pub const VIRTIO_CONSOLE_RESIZE: u16 = super::wire::ControlEvent::Resize.wire_value();
pub const VIRTIO_CONSOLE_PORT_OPEN: u16 = super::wire::ControlEvent::PortOpen.wire_value();
pub const VIRTIO_CONSOLE_PORT_NAME: u16 = super::wire::ControlEvent::PortName.wire_value();

const VC_CONTROL_SIZE: usize = std::mem::size_of::<VirtioConsoleControl>();
const _: () = assert!(VC_CONTROL_SIZE == 8);

// `PORT1_NAME` and `PORT2_NAME` live in [`super::wire`]; re-exported
// here for the existing call sites in this module.
pub use super::wire::PORT1_NAME;
pub use super::wire::PORT2_NAME;

/// Port-0 device-name advertised to the guest. The kernel's
/// `handle_control_message` PORT_NAME case
/// (drivers/char/virtio_console.c) creates the sysfs
/// `/sys/class/virtio-ports/vport0p0/name` attribute when the host
/// sends PORT_NAME; without that emission the attribute does not
/// exist and tooling that scans `/sys/class/virtio-ports/*/name` to
/// disambiguate port 0 (console) from port 1 (bulk) cannot
/// distinguish them. QEMU's `add_port` (hw/char/virtio-console.c)
/// sets a name on the chardev (`chardev-id` derived) and the
/// virtio-serial PORT_NAME emission in
/// `virtio_serial_post_load_timer_cb` / `send_control_event`
/// emits it for every port that has one — including the console
/// port. Mirror that here.
pub const PORT0_NAME: &str = "ktstr-console";

/// Outbound (host→guest) control payload kinds. The host serialises
/// these into 8-byte wire frames (plus optional name bytes) for the
/// c_ivq.
#[derive(Debug, Clone)]
enum ControlOut {
    /// Fixed 8-byte command.
    Cmd(VirtioConsoleControl),
    /// 8-byte PORT_NAME header followed by name bytes and a trailing
    /// NUL terminator. QEMU's PORT_NAME emitter
    /// (hw/char/virtio-serial-bus.c, `buffer_len = sizeof(cpkt) +
    /// strlen(port->name) + 1; ... buffer[buffer_len - 1] = 0;`)
    /// includes the NUL; the kernel parser
    /// (drivers/char/virtio_console.c `handle_control_message`
    /// PORT_NAME case) computes `name_size = buf->len - buf->offset
    /// - sizeof(*cpkt) + 1` and `strscpy`s into a kmalloc'd buffer,
    ///   which works either way but expects the QEMU layout. Sending
    ///   the NUL keeps the wire format byte-identical to QEMU so any
    ///   downstream tooling that snoops the frame sees the same shape.
    Name { id: u32, name: &'static str },
}

impl ControlOut {
    fn len(&self) -> usize {
        match self {
            ControlOut::Cmd(_) => VC_CONTROL_SIZE,
            // +1 for the trailing NUL terminator (see Name doc).
            ControlOut::Name { name, .. } => VC_CONTROL_SIZE + name.len() + 1,
        }
    }

    fn write_into(&self, dst: &mut Vec<u8>) {
        match self {
            ControlOut::Cmd(c) => dst.extend_from_slice(c.as_bytes()),
            ControlOut::Name { id, name } => {
                let hdr = VirtioConsoleControl {
                    id: *id,
                    event: VIRTIO_CONSOLE_PORT_NAME,
                    value: 1, // libkrun / qemu both pass value=1 here.
                };
                dst.extend_from_slice(hdr.as_bytes());
                dst.extend_from_slice(name.as_bytes());
                // Trailing NUL — matches QEMU's wire layout.
                dst.push(0);
            }
        }
    }
}

/// Per-port state for the three virtio-console ports. Indexed by
/// `port_id` (0 = console / hvc0, 1 = named bulk TLV stream,
/// 2 = named scheduler-stats relay).
///
/// `tx_buf` accumulates guest→host TX bytes pending host drain;
/// `pending_rx` accumulates host→guest RX bytes pending delivery
/// into the guest's RX ring. Both are unbounded by design — a host
/// alone produces RX bytes (so a hostile guest cannot grow
/// `pending_rx`), and TX bytes are bounded per-call by
/// `TX_PER_CALL_MAX`.
struct Port {
    /// Accumulated guest→host TX output. `VecDeque` so port 1's
    /// [`VirtioConsole::push_back_bulk`] can prepend the freeze
    /// coordinator's `bulk_assembler` residual via `push_front` in
    /// O(bytes); other ports drain via `Vec::from(VecDeque)` (no
    /// reallocation; at worst an O(N) rotate when the ring is split).
    tx_buf: VecDeque<u8>,
    /// Pending host→guest RX bytes. Drained into write-only
    /// descriptors on the matching RX queue notify (or on a
    /// PORT_OPEN closed→open transition for ports 1 and 2).
    pending_rx: VecDeque<u8>,
    /// Set when the guest sends `PORT_OPEN(value=1)` on c_ovq for
    /// this port. The RX drain gates on this flag for ports 1 and 2 —
    /// pushing bytes before the guest opens the port lets the kernel
    /// discard them with no userspace reader. Port 0 starts implicitly
    /// open (the kernel's hvc-console path does not require a
    /// control-protocol open before TX/RX).
    opened: bool,
    /// Set when the guest sends `PORT_READY(value=1)` on c_ovq for
    /// this port. Gates the host-side CONSOLE_PORT / PORT_OPEN /
    /// PORT_NAME enqueues — repeat PORT_READY messages from a
    /// hostile guest would otherwise grow `control_out` without
    /// bound, exhausting host memory. Each port may be readied
    /// exactly once per device lifecycle; `reset()` clears this
    /// back to false.
    readied: bool,
    /// Device name advertised to the guest via the PORT_NAME
    /// control message. Becomes the sysfs
    /// `/sys/class/virtio-ports/vport0pN/name` attribute that udev
    /// rules consume to symlink the port.
    name: &'static str,
}

impl Port {
    const fn new(name: &'static str) -> Self {
        Port {
            tx_buf: VecDeque::new(),
            pending_rx: VecDeque::new(),
            opened: false,
            readied: false,
            name,
        }
    }
}

/// Map a queue index to (port_id, is_tx). Returns None for the
/// control queues (C_IVQ / C_OVQ) and any out-of-range index.
const fn queue_to_port(queue_idx: usize) -> Option<(usize, bool)> {
    match queue_idx {
        PORT0_RXQ => Some((0, false)),
        PORT0_TXQ => Some((0, true)),
        PORT1_RXQ => Some((1, false)),
        PORT1_TXQ => Some((1, true)),
        PORT2_RXQ => Some((2, false)),
        PORT2_TXQ => Some((2, true)),
        _ => None,
    }
}

/// Map a port id to its (rxq, txq) queue indices. Inverse of
/// `queue_to_port` for the data direction.
const fn port_queues(port_id: usize) -> (usize, usize) {
    match port_id {
        0 => (PORT0_RXQ, PORT0_TXQ),
        1 => (PORT1_RXQ, PORT1_TXQ),
        2 => (PORT2_RXQ, PORT2_TXQ),
        _ => panic!("port_queues: port id out of range"),
    }
}

/// Static log-friendly label for a port id. Used in tracing fields
/// across the TX / RX / reset-drain paths so structured logs carry a
/// stable port identifier without per-call allocation.
const fn port_label(port_id: usize) -> &'static str {
    match port_id {
        0 => "port0",
        1 => "port1",
        2 => "port2",
        _ => "port?",
    }
}

/// Three-port virtio-console MMIO device.
///
/// Single-struct state — no separate transport layer. Caller holds
/// this in a `PiMutex` and dispatches MMIO reads/writes from the vCPU
/// run loop.
pub struct VirtioConsole {
    queues: [Queue; NUM_QUEUES],
    queue_select: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    device_status: u32,
    interrupt_status: u32,
    config_generation: u32,
    /// Eventfd for KVM irqfd — signals guest interrupt.
    irq_evt: EventFd,
    /// Eventfd signaled when TX data is available on port 0 or port 1.
    /// The host's stdout drain thread polls this to wake on port-0
    /// console bytes; the freeze coordinator's TOKEN_TX handler reads
    /// `ports[1].tx_buf` after a generic notification (the eventfd
    /// does not carry per-port granularity between ports 0 and 1, but
    /// the cost of an extra empty drain is negligible). Port 2 TX is
    /// signaled separately via [`Self::stats_tx_evt`] so the stats
    /// client wakes only on its own port.
    tx_evt: EventFd,
    /// Eventfd signaled when TX data is available on port 2 (scheduler
    /// stats relay). Distinct from [`Self::tx_evt`] so the host's
    /// [`super::sched_stats::SchedStatsClient`] poll wakes only on a
    /// stats-response edge — the freeze coordinator's TOKEN_TX
    /// handler does not contend on this fd, and the stats client
    /// does not get spurious wakes from port-0 console / port-1 bulk
    /// traffic.
    stats_tx_evt: EventFd,
    /// Guest memory reference. Set before starting vCPUs.
    mem: Option<GuestMemoryMmap>,
    /// Per-port state, indexed by port id. Replaces the prior
    /// `port{0,1,2}_tx_buf`, `port{0,1,2}_pending_rx`,
    /// `port_opened`, `port_readied` per-port fields with a single
    /// indexed array. See [`Port`] for field semantics.
    ports: [Port; NUM_PORTS as usize],
    /// Scratch staging for TX descriptor reads. `read_slice`
    /// writes into a contiguous `&mut [u8]`; the per-port `tx_buf`
    /// is a `VecDeque` so we read into this scratch first, then
    /// `extend` the deque from it. Shared mutex with the rest of
    /// the device (no concurrent TX/RX), so a single per-device
    /// scratch is safe and avoids per-descriptor heap churn.
    tx_scratch: Vec<u8>,
    /// Per-device reusable scratch for RX delivery; same justification
    /// as the original single-port impl — avoids per-descriptor heap
    /// churn under high paste rates.
    rx_scratch: Vec<u8>,
    /// Outbound control queue: messages waiting for the next c_ivq
    /// descriptor chain (PORT_ADD, PORT_OPEN, CONSOLE_PORT, PORT_NAME).
    /// Serviced FIFO; the host pushes during the
    /// DEVICE_READY/PORT_READY handshake, the guest publishes c_ivq
    /// buffers and we copy one message per chain.
    control_out: VecDeque<ControlOut>,
    /// True once the guest has sent `DEVICE_READY(value=1)` on c_ovq.
    /// Gates the host-side PORT_ADD enqueues — emitting them before
    /// DEVICE_READY would be ignored by the kernel and a per-port
    /// PORT_READY handshake would never start.
    device_ready: bool,
}

impl Default for VirtioConsole {
    /// Constructs a virtio-console device with the framework's three
    /// fixed ports (`PORT0_NAME` / `PORT1_NAME` / `PORT2_NAME`) and
    /// the 8 virtqueues the multi-port protocol requires (2 RX/TX
    /// per port + the control RX/TX pair). `device_ready` starts
    /// `false` and flips `true` only after the guest driver's
    /// `DEVICE_READY(value=1)` arrives on the control queue — the
    /// host must defer PORT_ADD enqueues until then or the guest
    /// will ignore them. Delegates to [`Self::new`] for the eventfd
    /// + queue construction; subsequent VM-builder calls bind
    ///   per-port destinations (e.g. bulk-data port to the test
    ///   framework's output directory) without adding new ports
    ///   beyond the three.
    fn default() -> Self {
        Self::new()
    }
}

mod device;
