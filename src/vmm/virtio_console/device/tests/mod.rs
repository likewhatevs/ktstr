#[allow(unused_imports)]
use super::super::*;
use std::os::unix::io::AsRawFd;
use virtio_bindings::bindings::virtio_ring::{VRING_DESC_F_NEXT, VRING_DESC_F_WRITE};
use virtio_bindings::virtio_mmio::VIRTIO_MMIO_INT_CONFIG;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;
use vm_memory::{Address, GuestAddress};

fn read_reg(dev: &VirtioConsole, offset: u32) -> u32 {
    let mut buf = [0u8; 4];
    dev.mmio_read(offset as u64, &mut buf);
    u32::from_le_bytes(buf)
}

fn write_reg(dev: &mut VirtioConsole, offset: u32, val: u32) {
    dev.mmio_write(offset as u64, &val.to_le_bytes());
}

/// Drive the device through the full init sequence up to DRIVER_OK,
/// negotiating both VIRTIO_F_VERSION_1 and F_MULTIPORT (the kernel
/// always negotiates every feature bit it supports that we
/// advertise).
fn init_device(dev: &mut VirtioConsole) {
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
    // F_MULTIPORT (bit 1) lives in the low 32-bit page.
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << VIRTIO_CONSOLE_F_MULTIPORT,
    );
    // VIRTIO_F_VERSION_1 (bit 32) lives in the high 32-bit page.
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
}

/// Build a guest memory map sized to host the chain test rings
/// (one or two queues at low GPAs) plus per-test data buffers
/// (high GPAs). 2 MiB is generous enough that buffers up to
/// 64 KiB (the oversize-truncation test) fit above the queue
/// regions without collision. Queue rings live near GPA 0; the
/// per-test descriptor data addresses sit at 0x10000+ which
/// leaves the entire first 64 KiB free for ring placement.
fn make_chain_test_mem() -> GuestMemoryMmap {
    GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 2 << 20)])
        .expect("create chain test guest mem")
}

/// Walk the FSM up to FEATURES_OK, configure `queue_idx` to
/// point at `mock`'s desc/avail/used rings with size 16, mark
/// the queue ready, then transition to DRIVER_OK. Mirrors
/// `virtio_blk::testing::wire_device_to_mock` adapted to the
/// 3-port / 8-queue console layout.
///
/// `init_device` walks the FSM all the way to DRIVER_OK with no
/// queue config — the post-DRIVER_OK gate then rejects any
/// queue address writes (`queue_config_allowed` requires
/// `S_FEAT && !DRIVER_OK`). For chain-level tests we must
/// configure the queue BEFORE DRIVER_OK lands, so this helper
/// stops at S_FEAT to install the queue addresses then advances
/// to S_OK once.
///
/// The MockSplitQueue size (16) matches the queue size we tell
/// the device via `QUEUE_NUM`; the mock's ring layout
/// (desc table → avail → used) is what the device's
/// `pop_descriptor_chain` walks once it sees the
/// driver-published avail.idx.
fn wire_console_queue_to_mock(
    dev: &mut VirtioConsole,
    mock: &MockSplitQueue<GuestMemoryMmap>,
    queue_idx: u32,
) {
    write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
    // F_MULTIPORT (bit 1) low half + VIRTIO_F_VERSION_1 (bit 32)
    // high half — same negotiation as `init_device`.
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << VIRTIO_CONSOLE_F_MULTIPORT,
    );
    write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
    write_reg(
        dev,
        VIRTIO_MMIO_DRIVER_FEATURES,
        1 << (VIRTIO_F_VERSION_1 - 32),
    );
    write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);

    write_reg(dev, VIRTIO_MMIO_QUEUE_SEL, queue_idx);
    write_reg(dev, VIRTIO_MMIO_QUEUE_NUM, 16);
    let desc = mock.desc_table_addr().0;
    let avail = mock.avail_addr().0;
    let used = mock.used_addr().0;
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
    write_reg(dev, VIRTIO_MMIO_QUEUE_READY, 1);
    write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
    // Sanity: a regression in feature negotiation that wedged
    // the FSM at FEATURES_OK would otherwise produce confusing
    // "process_tx sees an empty queue" failures from every
    // chain test downstream.
    assert_eq!(
        dev.device_status, S_OK,
        "wire_console_queue_to_mock: FSM did not reach DRIVER_OK \
         (got {:#x}) — feature negotiation likely regressed",
        dev.device_status,
    );
    assert!(
        dev.queues[queue_idx as usize].ready(),
        "wire_console_queue_to_mock: queue {queue_idx} did not \
         become ready after QUEUE_READY=1",
    );
}

// ----------------------------------------------------------------
// Chain-level MockSplitQueue tests for `drain_pending_rx` (port 1).
//
// These pin the host->guest port-1 RX path: snapshot reply payloads
// the freeze coordinator queues via `queue_input_port1` get
// delivered into guest write-only descriptors on q4 (PORT1_RXQ).
//
// Two concerns covered:
//
//   1. Deferral gates. Each early-return must hold pending bytes
//      in `port1_pending_rx` without touching the queue, the used
//      ring, or `interrupt_status`. A regression that lifted any
//      gate would either:
//        - publish a chain the guest has not committed (pre-
//          DRIVER_OK / queue-not-ready);
//        - publish bytes the kernel discards (port not yet
//          opened by the guest's PORT_OPEN handshake);
//        - walk a queue that does not exist for the negotiated
//          features (F_MULTIPORT not negotiated).
//
//   2. Torn-write recovery. When a multi-descriptor write-only
//      chain has one descriptor pointing at unmapped guest memory,
//      `mem.write_slice` fails mid-chain. The function must:
//        - publish the chain head with `len=0` (so the guest
//          reclaims the descriptor; without this the head leaks
//          from the avail ring until reset);
//        - leave bytes in `port1_pending_rx` unchanged (the
//          per-chain `drain(..consumed_offset)` is in the
//          non-torn branch, so a torn chain does NOT consume
//          bytes from the deque);
//        - break out of the drain loop (further chains for this
//          notify are NOT processed; the next notify retries).
//
// The torn-write fixture exploits write_slice's bounds check by
// pointing one descriptor at a GPA past the 2 MiB
// `make_chain_test_mem` map.
// ----------------------------------------------------------------

/// Wire q4 (PORT1_RXQ) to a MockSplitQueue and drive the FSM to
/// DRIVER_OK with F_MULTIPORT negotiated. Variant of
/// `wire_console_queue_to_mock` that targets q4 specifically; used
/// by every drain_port1 test below.
fn wire_port1_rxq_to_mock(dev: &mut VirtioConsole, mock: &MockSplitQueue<GuestMemoryMmap>) {
    wire_console_queue_to_mock(dev, mock, PORT1_RXQ as u32);
}

/// Mark port 1 as opened by sending PORT_OPEN(value=1) on c_ovq.
/// This sets `ports[1].opened = true` so the gate in
/// `drain_pending_rx` lets bytes through. The same call
/// invokes `drain_pending_rx` internally on the open
/// transition, but with no pending bytes that is a no-op — the
/// caller pushes pending bytes AFTER opening the port (or relies
/// on the open handler's own deferred-drain trigger).
fn open_port1(dev: &mut VirtioConsole) {
    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    });
    assert!(
        dev.ports[1].opened,
        "open_port1 helper precondition: PORT_OPEN(value=1) must \
         set ports[1].opened"
    );
}

// ----------------------------------------------------------------
// Port 2 chain-level tests (scheduler-stats relay).
//
// Port 2 mirrors port 1's TX/RX paths line-for-line in production
// — only the queue index, the buffer field, and the wake eventfd
// differ. Without explicit port-2 coverage the queue_idx → buffer
// routing match in `process_tx` (via `port_queues`), the
// `ports[2].opened` gate in `drain_pending_rx`, and the
// `stats_tx_evt` vs `tx_evt`
// dispatch could regress to incorrect targets and only surface in
// an end-to-end VM run. These tests pin every divergence between
// port 2 and the other two ports.
// ----------------------------------------------------------------

/// Mark port 2 as opened by sending PORT_OPEN(value=1) on c_ovq.
/// Mirrors `open_port1` for the port-2 RX gate in
/// `drain_pending_rx`.
fn open_port2(dev: &mut VirtioConsole) {
    dev.handle_control_event(VirtioConsoleControl {
        id: 2,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    });
    assert!(
        dev.ports[2].opened,
        "open_port2 helper precondition: PORT_OPEN(value=1) must \
         set ports[2].opened"
    );
}

/// Wire q6 (PORT2_RXQ) to a MockSplitQueue and drive the FSM to
/// DRIVER_OK with F_MULTIPORT negotiated. Variant of
/// `wire_port1_rxq_to_mock` for port 2.
fn wire_port2_rxq_to_mock(dev: &mut VirtioConsole, mock: &MockSplitQueue<GuestMemoryMmap>) {
    wire_console_queue_to_mock(dev, mock, PORT2_RXQ as u32);
}

mod control_events;
mod mmio_config;
mod mmio_read;
mod port2;
mod port_tx;
mod rx_drain;
mod status_fsm;
