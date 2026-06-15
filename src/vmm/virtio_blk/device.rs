//! Device-side virtio-block: MMIO dispatch, FSM, request state, and
//! `Drop`. The handler bodies live in `handlers.rs`, the request-queue
//! drain bracket lives in `drain.rs`, and the per-device counter type
//! lives in `counters.rs` — see the parent module's submodule-layout
//! doc for the full split rationale.
//!
//! See the parent module `super` for the full execution-model and
//! "why" doc — the module-level rationale for why
//! `add_used` is gated on status-write success, why throttle stalls
//! roll back the chain and arm a timerfd, and the backing-speed
//! caveat — lives there.

pub(crate) use std::fs::File;
// `AsRawFd` for the eventfd raw-fd plumbing (stop_fd / kick_fd
// `as_raw_fd` in the worker-engine drop and respawn paths), which is
// `cfg(not(test))` (the test build uses the inline engine and has no
// worker fds). The backing's vectored IO no longer needs it — that
// routes through the `Backing` trait's `preadv`/`pwritev` (the
// impl-for-File holds the `as_raw_fd` call) rather than a raw fd at
// the call site.
#[cfg(not(test))]
pub(crate) use std::os::unix::io::AsRawFd;
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::OnceLock;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
// `std::thread` is used only by the `#[cfg(not(test))]` SpawnedEngine handle;
// the worker spawn/join + mpsc moved with control.rs/lifecycle.rs.
#[cfg(not(test))]
use std::thread;
pub(crate) use std::time::Duration;

pub(crate) use virtio_bindings::virtio_blk::{
    VIRTIO_BLK_F_BLK_SIZE, VIRTIO_BLK_F_FLUSH, VIRTIO_BLK_F_RO, VIRTIO_BLK_F_SEG_MAX,
    VIRTIO_BLK_F_SIZE_MAX, VIRTIO_BLK_ID_BYTES, VIRTIO_BLK_S_IOERR, VIRTIO_BLK_S_OK,
    VIRTIO_BLK_S_UNSUPP, VIRTIO_BLK_T_FLUSH, VIRTIO_BLK_T_GET_ID, VIRTIO_BLK_T_IN,
    VIRTIO_BLK_T_OUT,
};
pub(crate) use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FAILED, VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_CONFIG_S_NEEDS_RESET,
    VIRTIO_F_VERSION_1,
};
pub(crate) use virtio_bindings::virtio_ids::VIRTIO_ID_BLOCK;
// `VIRTIO_MMIO_INT_CONFIG` and `VIRTIO_MMIO_INT_VRING` are consumed by
// `drain.rs` directly (its own per-name imports) and by `cfg(test)`
// test sub-files via the `super::*;` glob — neither path is visible
// to `clippy --lib`, so the re-export looks unused without the allow.
#[allow(unused_imports)]
pub(crate) use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INTERRUPT_ACK,
    VIRTIO_MMIO_INTERRUPT_STATUS, VIRTIO_MMIO_MAGIC_VALUE, VIRTIO_MMIO_QUEUE_AVAIL_HIGH,
    VIRTIO_MMIO_QUEUE_AVAIL_LOW, VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW,
    VIRTIO_MMIO_QUEUE_NOTIFY, VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX,
    VIRTIO_MMIO_QUEUE_READY, VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH,
    VIRTIO_MMIO_QUEUE_USED_LOW, VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};
pub(crate) use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
// `VirtioQueueError` is matched on in `drain.rs` and in `cfg(test)`
// test sub-files via `super::*;`; clippy --lib doesn't see those.
#[allow(unused_imports)]
pub(crate) use virtio_queue::Error as VirtioQueueError;
#[cfg(test)]
use virtio_queue::Queue;
// `QueueOwnedT::iter` is invoked by `drain.rs` and the test sub-files
// via `super::*;`; clippy --lib doesn't see those.
#[allow(unused_imports)]
pub(crate) use virtio_queue::QueueOwnedT;
#[cfg(not(test))]
use virtio_queue::QueueSync;
pub(crate) use virtio_queue::QueueT;
pub(crate) use vm_memory::{ByteValued, Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

// `VirtioBlkCounters` lives in `counters.rs`; reach it via `super::*`
// (sourced from `mod.rs`'s `pub(crate) use counters::*;`).
use super::VirtioBlkCounters;
use super::{Backing, advance_iovecs};
// `EpollEvent` / `EventSet` are re-exported because tests for the
// always-compiled `worker_dispatch_event` helper construct EventSet
// values directly via `super::*`, and the helper itself accepts an
// EventSet argument. clippy --lib doesn't see the test consumers,
// so the re-export looks unused without the allow.
#[allow(unused_imports)]
pub(crate) use vmm_sys_util::epoll::{EpollEvent, EventSet};
pub(crate) use vmm_sys_util::eventfd::EventFd;

pub(crate) use super::super::disk_config::DiskThrottle;

pub(crate) const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
pub(crate) const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
pub(crate) const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KiB (one page).
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

/// Single request queue. virtio-spec §5.2.2 declares one request
/// queue plus optional multiqueue (`VIRTIO_BLK_F_MQ`); MQ deferred.
pub(crate) const NUM_QUEUES: usize = 1;
pub(crate) const QUEUE_MAX_SIZE: u16 = 256;
pub(crate) const REQ_QUEUE: usize = 0;

/// Queue type used for the request virtqueue. Production uses
/// `QueueSync` (`Arc<Mutex<Queue>>` internally) so the vCPU thread
/// (MMIO config writes — set_size/ready/desc/avail/used addresses)
/// and the dedicated worker thread (drain_bracket_impl — pop, add_used,
/// needs_notification) can share the same queue state safely. Tests
/// run drain_bracket_impl inline on the caller thread so the bare `Queue`
/// (single-threaded, no internal lock) is sufficient and avoids
/// changing the test surface that drives `Queue` methods directly
/// (`disable_notification`, `set_avail_ring_address`, etc.).
///
/// The `QueueT` trait is the single API both implementations honour;
/// every drain-side method this file calls (pop_descriptor_chain,
/// add_used, disable/enable_notification, needs_notification,
/// avail_ring, event_idx_enabled) is part of `QueueT`. Generic
/// helpers like `publish_completion` are bound by `QueueT` so they
/// compile against either alias without further indirection.
#[cfg(not(test))]
pub(crate) type BlkQueue = QueueSync;
#[cfg(test)]
pub(crate) type BlkQueue = Queue;

/// Logical block size advertised to the guest. 512 bytes matches
/// the virtio spec default.
pub const VIRTIO_BLK_SECTOR_SIZE: u32 = 512;

/// Default capacity (256 MiB) used by virtio_blk tests. Mirrors the
/// 256-MiB default in `super::disk_config::DiskConfig::default`.
///
/// Sized for `mkfs.btrfs` minimum without `--mixed`: btrfs needs
/// ~109 MiB for single-profile metadata and ~256 MiB if it picks DUP
/// metadata (which is the default on a single-device fs). Sized
/// below 256 MiB risks `mkfs.btrfs` failing at template-build time.
///
/// `dead_code` allow: only consumed by `#[cfg(test)]` modules
/// (every virtio_blk test fixture passes this as the device's
/// capacity); clippy --lib doesn't see those references.
#[allow(dead_code)]
pub const VIRTIO_BLK_DEFAULT_CAPACITY_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum number of data segments per request the device supports.
/// virtio-v1.2 §5.2.4: `seg_max` is the max scatter-gather buffer
/// count, exclusive of the header and status descriptors. Without
/// `F_SEG_MAX` the guest defaults `max_segments` to 1, which forces
/// `bio_split` and serializes large requests; advertising 128 is the
/// firecracker default and ample for the small files this device
/// targets.
pub(crate) const VIRTIO_BLK_SEG_MAX: u32 = 128;

/// Maximum size in bytes of a single descriptor's data buffer.
/// virtio-v1.2 §5.2.4 (`size_max`): caps per-descriptor length so a
/// guest can't submit a single 4 GB descriptor and force the device
/// to allocate a matching `Vec<u8>` for `read_at`/`write_at`. 1 MiB
/// matches firecracker's default and is far above what the guest's
/// blk-mq layer typically generates (max_sectors_kb defaults to
/// 512 KiB). Without `F_SIZE_MAX` the guest treats per-descriptor
/// length as unbounded — host OOM hazard on a hostile guest.
pub(crate) const VIRTIO_BLK_SIZE_MAX: u32 = 1 << 20;

/// Linux's maximum iovec count (`UIO_MAXIOV`) for a single
/// `preadv`/`pwritev`: passing `iovcnt > IOV_MAX` returns `EINVAL`.
/// The vectored-IO paths build at most `SEG_MAX * 2 = 256` iovecs
/// (see the SAFETY comments at the preadv/pwritev sites for the
/// structural derivation), well under this cap; the `debug_assert`s
/// there trip in test/debug builds if a future change ever breaks
/// that bound.
const IOV_MAX: usize = 1024;

/// Cap on consecutive `EINTR` re-issues of a single blk vectored IO
/// (`preadv`/`pwritev`) before giving up with `S_IOERR`. `EINTR` on a
/// buffered regular-file path implies a PENDING FATAL signal (the
/// wait is `TASK_KILLABLE`, so only `SIGKILL`/group-exit interrupts),
/// which does NOT clear — an unbounded retry would spin until the task
/// is reaped, delaying the freeze-rendezvous failure dump. 16 is
/// generous for any (rare) catchable-signal case while bounding the
/// fatal-signal spin to a handful of iterations.
///
/// Divergence (we get this right): firecracker / vm-memory's
/// `retry_eintr!` loops on `EINTR` UNBOUNDED — on a pending fatal
/// signal that is a latent spin. We bound it: a spinning IO thread
/// delays the freeze-rendezvous dump (the project's vCPU-blocking
/// budget).
const MAX_EINTR_RETRIES: u32 = 16;

/// Device serial number returned by `VIRTIO_BLK_T_GET_ID`. Per
/// virtio-v1.2 §5.2.6.4 (and `virtio_blk.h` `VIRTIO_BLK_ID_BYTES`)
/// the kernel driver passes a 20-byte buffer (`virtblk_get_id` →
/// `blk_rq_map_kern(req, id_str, VIRTIO_BLK_ID_BYTES, GFP_KERNEL)`,
/// drivers/block/virtio_blk.c). The string is exposed at
/// `/sys/block/<dev>/serial` after `serial_show` reads it from the
/// device. The 16-byte payload `ktstr-virtio-blk` is null-padded to
/// 20 bytes; the trailing zeros let `serial_show`'s
/// `strlen(buf)` (after the kernel's `buf[VIRTIO_BLK_ID_BYTES] =
/// '\0'` sentinel) terminate at the first NUL.
pub(crate) const VIRTIO_BLK_SERIAL: [u8; VIRTIO_BLK_ID_BYTES as usize] =
    *b"ktstr-virtio-blk\0\0\0\0";

/// Request out-header. virtio-v1.2 §5.2.6: every request chain
/// starts with a device-readable, 16-byte header carrying the
/// request type, ioprio (ignored), and starting sector. The struct
/// matches virtio_bindings::virtio_blk::virtio_blk_outhdr field for
/// field — it is redeclared here so we can attach `ByteValued` (the
/// bindings struct does not implement it) and use `Bytes::read_obj`
/// directly. `repr(C)` + integer-only fields satisfy the
/// `ByteValued` invariants (§ vm-memory bytes.rs trait docs).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioBlkOutHdr {
    /// `VIRTIO_BLK_T_*`. LE per virtio-v1.2 §5.2.6.
    pub(crate) type_: u32,
    /// I/O priority, ignored on this device.
    pub(crate) _ioprio: u32,
    /// Starting sector (512-byte units).
    pub(crate) sector: u64,
}

// SAFETY: VirtioBlkOutHdr is `repr(C)`, contains only `u32` and `u64`
// (themselves `ByteValued`), has no padding (4+4+8 = 16, all aligned),
// and any byte pattern is a valid value (the type/ioprio fields are
// validated separately by the request dispatcher; sector is just a
// number). All `ByteValued` requirements are met.
unsafe impl vm_memory::ByteValued for VirtioBlkOutHdr {}

/// Header size for `VirtioBlkOutHdr`. virtio-v1.2 §5.2.6:
/// type:u32, ioprio:u32, sector:u64.
pub(crate) const VIRTIO_BLK_OUTHDR_SIZE: usize = std::mem::size_of::<VirtioBlkOutHdr>();

/// Legacy CHS geometry sub-struct of `VirtioBlkConfig`, gated on
/// `VIRTIO_BLK_F_GEOMETRY`. Mirrors the kernel uapi
/// `struct virtio_blk_geometry` (cylinders:u16, heads:u8, sectors:u8 —
/// 4 bytes total) at config-space offset 0x10. We don't advertise
/// `F_GEOMETRY` so the field is left zero; the guest driver reads it
/// via `virtio_cread_feature`, which returns `-ENOENT` when the
/// feature bit is not negotiated and the read is skipped.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioBlkGeometry {
    pub(crate) cylinders: u16,
    pub(crate) heads: u8,
    pub(crate) sectors: u8,
}

/// Block-device config space (virtio-v1.2 §5.2.4). Mirrors the kernel
/// uapi `struct virtio_blk_config` field-for-field up through
/// `blk_size` (the last field whose feature bit this device
/// advertises). Trailing fields (topology, MQ, discard, write-zeroes,
/// secure-erase, zoned) are gated on feature bits we don't advertise,
/// so the guest driver's `virtio_cread_feature` returns `-ENOENT` for
/// those reads and never depends on the device-side bytes — we serve
/// zeros for any read past `size_of::<VirtioBlkConfig>()`, matching
/// virtio-v1.2 §4.2.2.2 ("reads past the populated config layout
/// return zero").
///
/// The kernel struct is `__attribute__((packed))` (see
/// `include/uapi/linux/virtio_blk.h`), so this redeclaration uses
/// `repr(C, packed)` to match the wire layout byte-for-byte. Without
/// the `packed` attribute the compiler would insert padding after
/// `seg_max` to align `geometry` (which contains a `u16`) — that
/// padding would shift `blk_size` from offset 0x14 to 0x18 and serve
/// the guest a wrong block-size value silently.
#[repr(C, packed)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct VirtioBlkConfig {
    /// Capacity in 512-byte sectors. Always populated; the kernel
    /// driver reads this unconditionally (no feature bit gates it).
    pub(crate) capacity: u64,
    /// Maximum per-descriptor data length, gated on
    /// `VIRTIO_BLK_F_SIZE_MAX`.
    pub(crate) size_max: u32,
    /// Maximum scatter-gather segments per request, gated on
    /// `VIRTIO_BLK_F_SEG_MAX`.
    pub(crate) seg_max: u32,
    /// Legacy CHS geometry, gated on `VIRTIO_BLK_F_GEOMETRY`. We
    /// don't advertise that bit so this field is left zero.
    pub(crate) geometry: VirtioBlkGeometry,
    /// Logical block size, gated on `VIRTIO_BLK_F_BLK_SIZE`.
    pub(crate) blk_size: u32,
}

// SAFETY: `VirtioBlkConfig` and `VirtioBlkGeometry` are
// `repr(C, packed)`. With `packed` the alignment is 1 and there is no
// inter-field padding by definition (every field is byte-aligned). All
// fields are integer types (`u8`, `u16`, `u32`, `u64`) for which every
// bit pattern is a valid value, so reading arbitrary bytes into the
// struct yields a well-defined value. The struct is `Copy`, `Send`,
// and `Sync` (all primitives), satisfying the `ByteValued` supertrait
// bounds. Total size is verified against the kernel uapi layout by
// the `VIRTIO_BLK_CONFIG_SIZE` const assertion below.
unsafe impl vm_memory::ByteValued for VirtioBlkConfig {}
// SAFETY: same justification as `VirtioBlkConfig`. `VirtioBlkGeometry`
// is `repr(C, packed)` with three integer fields (`u16`, `u8`, `u8`),
// no padding, all bit patterns valid, `Copy + Send + Sync`.
unsafe impl vm_memory::ByteValued for VirtioBlkGeometry {}

/// Size of the populated portion of block config space (24 bytes:
/// capacity 8 + size_max 4 + seg_max 4 + geometry 4 + blk_size 4).
/// Reads at config-space offsets `>= VIRTIO_BLK_CONFIG_SIZE` return
/// zero per virtio-v1.2 §4.2.2.2.
pub(crate) const VIRTIO_BLK_CONFIG_SIZE: usize = std::mem::size_of::<VirtioBlkConfig>();
// Compile-time check that the struct layout matches the kernel uapi
// byte budget (8+4+4+4+4 = 24). A mismatch here means either Rust's
// `repr(C, packed)` introduced a divergence from the kernel's
// `__attribute__((packed))` layout, or a field was added/removed —
// in either case the guest would read garbage from a misaligned
// field. Failing to compile is preferable to silently serving wrong
// bytes.
const _: () = assert!(VIRTIO_BLK_CONFIG_SIZE == 24);
// Field-offset checks: the kernel driver reads each field at a
// specific offset via `virtio_cread`. If `repr(C, packed)` ever
// drifts from the kernel's `__attribute__((packed))` layout, these
// asserts catch it at compile time before a wrong-offset bug ships
// to the guest.
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, capacity) == 0x00);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, size_max) == 0x08);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, seg_max) == 0x0C);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, geometry) == 0x10);
const _: () = assert!(std::mem::offset_of!(VirtioBlkConfig, blk_size) == 0x14);

/// One descriptor from a virtio request chain. Used uniformly for
/// every chain role — header, data segments, and status — so the
/// chain-walk code can collect all descriptors into one buffer and
/// then index by position (first = header, middle = data, last =
/// status).
///
/// `is_write_only` mirrors `desc.is_write_only()` from
/// `virtio_queue` (i.e. VRING_DESC_F_WRITE set in the descriptor):
/// device-writable for read data segments and the status byte;
/// device-readable for the request header and write data segments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainDescriptor {
    pub(crate) addr: GuestAddress,
    pub(crate) len: u32,
    pub(crate) is_write_only: bool,
}

/// Status bits required before each phase (mirrors virtio_console).
pub(crate) const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
pub(crate) const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
pub(crate) const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
/// Test helper — terminal state bits with DRIVER_OK set.
#[cfg(test)]
pub(crate) const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

// Token-bucket throttle primitives live in `throttle`. Pulled out as
// a submodule so the throttle's adversarial test surface (which is
// the most-exercised piece of the device) sits next to its tests
// rather than scattered through the device + worker code. See
// `throttle.rs` for the full type-level rationale.
use super::throttle::*;

/// Publish a chain completion: write the status byte and, on
/// success, mark the chain used. Returns `true` if the device
/// should signal the guest (used-ring index advanced); `false`
/// otherwise.
///
/// Status-write-success gate: `add_used` is called ONLY after a
/// successful status-byte write. Publishing a completion the guest
/// can't observe (status write failed but used-ring advanced) would
/// let the guest's `virtblk_done` read its `vbr->in_hdr.status`
/// byte that's stale from prior blk-mq tag use (initially zero from
/// `__GFP_ZERO` at allocation, stale on reuse) as `BLK_STS_OK`
/// (drivers/block/virtio_blk.c `virtblk_vbr_status` +
/// `virtblk_result(0)`) — silent data corruption for reads, silent
/// dropped writes for writes. On status-write failure the chain
/// stays in the avail ring; virtio_blk has no `mq_ops->timeout`
/// (drivers/block/virtio_blk.c `virtio_mq_ops`), so the guest hangs
/// on this request until `kernel.hung_task_timeout_secs` (default
/// 120 s) fires or a higher-layer retries.
///
/// `used_len` is what `add_used` records as the "bytes written by
/// the device into guest memory". Error paths pass `1` (just the
/// status byte). The success path passes the data-bytes-written
/// total + 1 (for reads) or `1` (for writes/flushes — the device
/// wrote no data back).
///
/// `label` is included in any tracing::warn from this function so
/// operators can identify which gate triggered the publish.
///
/// `too_many_arguments` allow: every parameter is independent
/// per-request state (queue/memory binding, head index, status
/// address+byte, used-len, label) sourced from a different point
/// in the chain-handling pipeline. Bundling would build a struct
/// for one call seam.
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_completion<Q: QueueT>(
    mem: &GuestMemoryMmap,
    q: &mut Q,
    counters: &VirtioBlkCounters,
    head: u16,
    status_addr: GuestAddress,
    status_byte: u8,
    used_len: u32,
    label: &'static str,
) -> bool {
    if mem.write_slice(&[status_byte], status_addr).is_err() {
        // Status-byte write failed — the chain stays in the avail
        // ring. virtio_blk has no `mq_ops->timeout`, so the guest
        // hangs on this request until `kernel.hung_task_timeout_secs`
        // (default 120 s) fires or a higher-layer retries.
        // Bump io_errors so the host operator sees a counter for
        // every silent-stall event. Error-site callers also bump
        // io_errors before reaching here; the double-count is
        // intentional under hostile-guest scenarios — a guest
        // constructing chains with systematically unmapped
        // status_addr will double-count every request, but the
        // silent-stall it prevents on the success path is the
        // worse failure mode. Silent-swallow on the success path
        // (FLUSH or T_IN/T_OUT/T_GET_ID succeeded but the status
        // descriptor itself is unmapped) would otherwise produce
        // a host-side silent stall — virtio_blk has no
        // `mq_ops->timeout` callback, so blk-mq alone never
        // surfaces the unpublished request as an error; only the
        // guest's hung-task watchdog fires (default 120 s) — and
        // without this counter bump operators would have no
        // host-side signal until the watchdog message hits dmesg.
        counters.record_io_error();
        return false;
    }
    match q.add_used(mem, head, used_len) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(head, %e, label, "virtio-blk add_used failed");
            counters.record_io_error();
            false
        }
    }
}

// `VirtioBlkCounters` and its `record_*` mutators / `pub fn` readers
// live in `counters.rs`; reach them via the `super::*;` glob (which
// sources from `mod.rs`'s `pub(crate) use counters::*;`). Pulled out
// for module locality so the per-helper invariants and the
// failure-dump-renderer-relevant counter taxonomy doc sit together.

// ----------------------------------------------------------------------------
// Device struct
// ----------------------------------------------------------------------------

/// Worker-thread-owned mutable state. In production this lives on
/// the dedicated worker thread for the device's lifetime; in test
/// builds it lives directly inside `BlkWorker` (Inline mode) so the
/// existing test surface — which calls `process_requests`
/// synchronously and immediately reads back state via
/// `dev.worker.state_mut().ops_bucket` etc. — keeps working.
///
/// The MMIO-side state (interrupt_status, irq_evt, mem, FSM bits)
/// stays on `VirtioBlk` and is shared with the worker via Arc.
pub(crate) struct BlkWorkerState {
    /// Backing file. The worker reads and writes sectors via
    /// `pread`/`pwrite` and never inspects the on-disk contents.
    pub(crate) backing: Box<dyn Backing>,
    /// Token-bucket for ops/sec.
    pub(crate) ops_bucket: TokenBucket,
    /// Token-bucket for bytes/sec.
    pub(crate) bytes_bucket: TokenBucket,
    /// Reusable scratch for the descriptor-walk in `drain_bracket_impl`.
    /// Allocated once at construction with capacity
    /// `VIRTIO_BLK_SEG_MAX + 2` and `clear()`-ed each iteration so
    /// the underlying capacity is reused. Avoids one Vec
    /// allocation per request on the hot path. The push loop in
    /// `drain_bracket_impl` is hard-capped at this capacity — a
    /// hostile guest submitting a chain longer than `seg_max + 2`
    /// gets dropped before status-extraction (an oversized chain
    /// cannot be reliably IOERR-completed because the capped view
    /// loses sight of the chain's true status descriptor). The
    /// data-segment slice given to the handlers is borrowed
    /// directly from `&state.all_descs_scratch[1..chain_len - 1]`
    /// once `status_addr` has been validated — no second Vec, no
    /// copy.
    pub(crate) all_descs_scratch: Vec<ChainDescriptor>,
    /// Reusable per-segment IO buffer. Sized by `resize(len, 0)`
    /// per segment in the read/write handlers. Allocated once and
    /// reused across all segments of all requests; the underlying
    /// `Vec`'s capacity grows monotonically up to
    /// `VIRTIO_BLK_SIZE_MAX` (the per-descriptor cap we advertise),
    /// at which point all subsequent IO is amortized to zero
    /// allocation.
    pub(crate) io_buf_scratch: Vec<u8>,
    /// Capacity in bytes. Computed once at construction
    /// (`capacity_sectors * VIRTIO_BLK_SECTOR_SIZE`) and threaded
    /// into handlers so the multiply isn't repeated per request and
    /// can never overflow on a malicious sector value (the multiply
    /// happens once on host-trusted input).
    pub(crate) capacity_bytes: u64,
    /// Read-only mode. When `true`, the device advertises
    /// `VIRTIO_BLK_F_RO`. `VIRTIO_BLK_T_OUT` requests are rejected
    /// with `VIRTIO_BLK_S_IOERR`; `VIRTIO_BLK_T_FLUSH` requests
    /// return `VIRTIO_BLK_S_OK` (a no-op flush — there's no dirty
    /// data to flush, and a guest issuing a precautionary flush
    /// during mount-readonly should not see an error). Per
    /// virtio-v1.2 §5.2.5.1, when `F_RO` is negotiated the device
    /// is read-only and the guest driver SHOULD treat the device
    /// as read-only; how the driver chooses to do that is
    /// driver business — on Linux `open(2)` is not gated and
    /// writes fail with `EPERM` (no `open`-time error). The
    /// in-device rejection is defense
    /// against a malicious or buggy guest that ignores the
    /// negotiated feature bit.
    pub(crate) read_only: bool,
    /// Counters. `Arc` so external monitor observers can read them
    /// without holding any device borrow; the worker mutates via
    /// the same `Arc`.
    pub(crate) counters: Arc<VirtioBlkCounters>,
    /// Per-worker "is the head-of-queue chain currently stalled?"
    /// flag. Owned by `drain_bracket_impl`; the flag transitions
    /// gate the live gauge updates on the shared `counters` Arc:
    ///
    /// - `false → true` (transition into stall): bump
    ///   `currently_throttled_gauge` via
    ///   `counters.record_throttle_pending_inc()`.
    /// - `true → false` (transition out of stall): decrement via
    ///   `counters.record_throttle_pending_dec()`.
    /// - `true → true` (idempotent re-stall on the same head):
    ///   no gauge update; only `throttled_count` (events)
    ///   advances.
    /// - `false → false` (normal completion without prior stall):
    ///   no gauge update.
    ///
    /// Lives on `BlkWorkerState` (not on the shared counters Arc)
    /// because the transition logic is per-worker — only the
    /// thread that owns the drain knows which transition just
    /// happened. Reading the AtomicU64 gauge alone could not
    /// distinguish "first stall" from "re-stall on the same head"
    /// without per-worker state. Cfg-independent so both Inline
    /// and Spawned engines maintain the same invariant.
    pub(crate) currently_stalled: bool,
    /// Sticky "the queue is structurally broken; stop draining"
    /// flag. Set when the avail-ring iterator returns
    /// `Error::InvalidAvailRingIndex` — the avail.idx the guest
    /// published is more than `queue.size` ahead of the device's
    /// `next_avail`, which the virtio spec forbids
    /// (virtio-v1.2 §2.7.13.3: avail.idx advances monotonically
    /// at most `queue.size` ahead of the device-side cursor; an
    /// excursion beyond that distance is the structural-invariant
    /// violation `iter()` reports as `InvalidAvailRingIndex`).
    /// Without this flag, every subsequent `pop_descriptor_chain`
    /// would re-trip the same error and `enable_notification`
    /// would re-arm immediately, looping the worker forever
    /// against a hostile guest at full vCPU/host CPU cost.
    ///
    /// Once set, `drain_bracket_impl` short-circuits to `Done`
    /// without touching the queue at all — no
    /// `disable_notification`, no `iter`, no `enable_notification`.
    /// The flag clears only on a full virtio reset
    /// (`reset_engine_inline` / `respawn_worker` rebuilds the
    /// state with `queue_poisoned: false`), matching the device's
    /// `VIRTIO_CONFIG_S_NEEDS_RESET` (virtio-v1.2 §2.1.1 bit 0x40)
    /// behaviour: the device tells the guest "I need a reset before
    /// I can service IO" and the only escape is a STATUS=0 MMIO
    /// write. This converges with cloud-hypervisor's NEEDS_RESET
    /// path on hostile-guest queue corruption (NOT the FAILED status
    /// = 0x80, which is the orthogonal "driver gives up" exit per
    /// virtio-v1.2 §2.1.1 bit 0x80 — the framework is signalling
    /// "device needs reset", not "driver gave up").
    ///
    /// Per-worker (not on the shared counters Arc) because only
    /// the drain thread mutates it. Cfg-independent so both
    /// Inline and Spawned engines maintain the same invariant.
    pub(crate) queue_poisoned: bool,
}

/// Wraps the request-processing engine. In Inline mode (cfg(test))
/// the state lives in-line and `process_requests` runs the drain
/// synchronously on the caller thread — preserving the existing
/// 113-test surface that calls `process_requests` then immediately
/// reads back queue + counter state without crossing a thread
/// boundary. In Spawned mode (production) a dedicated worker thread
/// owns the state and is woken by `kick_fd`; the MMIO QUEUE_NOTIFY
/// handler writes 1 to `kick_fd` and returns immediately so the
/// vCPU thread is never blocked by the IO syscall.
///
/// `read_only` and `counters` are duplicated outside the engine so
/// MMIO accessors (`device_features` reads `read_only`, `counters()`
/// returns the Arc) can reach them without coordinating with the
/// worker. They are immutable after construction in Spawned mode and
/// kept in sync with the Inline branch's `BlkWorkerState`.
///
/// The shared resources the worker needs to drive the drain
/// (`Arc<BlkQueue>` queue, `Arc<EventFd>` irq_evt,
/// `Arc<AtomicU32>` interrupt_status, `Arc<OnceLock<GuestMemoryMmap>>` mem)
/// are stored on `VirtioBlk` and cloned into the worker thread at
/// spawn time; the worker holds independent Arc handles for the
/// duration of its run.
pub(crate) struct BlkWorker {
    pub(crate) queues: [BlkQueue; NUM_QUEUES],
    /// `read_only` flag, mirrored on the device side for
    /// `device_features` and direct test inspection
    /// (`dev.worker.read_only`). Set once at construction and never
    /// mutated.
    pub(crate) read_only: bool,
    /// Counters Arc shared with the worker thread; mirrored on the
    /// device side for `counters()` and direct test inspection.
    pub(crate) counters: Arc<VirtioBlkCounters>,
    /// Engine-mode-specific state.
    pub(crate) engine: WorkerEngine,
}

/// Implementation strategy for the request-processing engine.
pub(crate) enum WorkerEngine {
    /// Synchronous in-thread mode (cfg(test)). The drain runs on the
    /// caller thread when `process_requests` is called.
    #[cfg(test)]
    Inline(InlineEngine),
    /// Production mode: a dedicated worker thread owns the state
    /// and drives the drain on receipt of a kick eventfd write.
    #[cfg(not(test))]
    Spawned(SpawnedEngine),
}

/// Inline-mode engine state (cfg(test) only). Holds `BlkWorkerState`
/// directly so the existing test surface that reaches into
/// `dev.worker.<state field>` keeps compiling without renames.
#[cfg(test)]
pub(crate) struct InlineEngine {
    pub(crate) state: BlkWorkerState,
}

/// Test-only accessors: in `cfg(test)` the `BlkWorkerState` lives in
/// the Inline engine; tests reach in via `dev.worker.state_mut()` /
/// `dev.worker.state()` rather than walking the engine enum on every
/// access. The `match` is exhaustive against the single-variant cfg
/// — there is no Spawned variant to handle in test builds.
#[cfg(test)]
impl BlkWorker {
    pub(crate) fn state(&self) -> &BlkWorkerState {
        let WorkerEngine::Inline(engine) = &self.engine;
        &engine.state
    }
    pub(crate) fn state_mut(&mut self) -> &mut BlkWorkerState {
        let WorkerEngine::Inline(engine) = &mut self.engine;
        &mut engine.state
    }
}

/// Spawned-mode engine state (production only). The mutable
/// `BlkWorkerState` lives entirely on the worker thread; the device
/// retains only a kick eventfd, a stop eventfd, and the join handle.
/// The pause-eventfd write side lives on `VirtioBlk::pause_evt`
/// (cfg-independent) so `pause()` / `resume()` compile in `cfg(test)`
/// without an engine match — the worker's read clone is taken at
/// spawn time and consumed by `worker_thread_main`'s frame.
#[cfg(not(test))]
pub(crate) struct SpawnedEngine {
    /// Eventfd written by `mmio_write(QUEUE_NOTIFY, …)`; the worker
    /// epoll-waits on it and runs one drain iteration per signal.
    /// Counter-mode (no `EFD_SEMAPHORE` flag) so coalesced kicks
    /// produce one wakeup. Configured `EFD_NONBLOCK` so neither the
    /// vCPU `write(1)` nor the worker `read()` ever blocks.
    pub(crate) kick_fd: EventFd,
    /// Eventfd written by `Drop::drop`; worker reads it and exits.
    /// Counter-mode + `EFD_NONBLOCK`. The worker checks both fds in
    /// the same `epoll_wait` call so a stop signal supersedes any
    /// pending kick.
    pub(crate) stop_fd: EventFd,
    /// Worker thread join handle. Wrapped in `Option` so `Drop`
    /// and `reset()` can `take()` and `join()` it. None after the
    /// thread has been joined.
    ///
    /// The `BlkWorkerState` payload is yielded by
    /// `worker_thread_main` on STOP_TOKEN: `reset()` recovers it
    /// to rebuild fresh throttle buckets and re-spawn a worker
    /// against the post-`q.reset()` queue. `Drop` discards the
    /// returned state with `let _ = handle.join()`. Both paths
    /// observe the same return value; only the consumer differs.
    pub(crate) handle: Option<thread::JoinHandle<BlkWorkerState>>,
    /// State reclaimed from a quiesced worker, awaiting respawn at
    /// the next DRIVER_OK transition. `Some(_)` between
    /// `reset_engine_spawned` (which joins the old worker, captures
    /// its state, and stashes it here) and the guest's subsequent
    /// `STATUS = DRIVER_OK` MMIO write (which `set_status` consumes
    /// to re-spawn a fresh worker). `None` in all other steady
    /// states.
    ///
    /// # Why deferred
    ///
    /// Between `reset()` and DRIVER_OK the guest is rebinding —
    /// queue addresses are zeroed, `QUEUE_READY` is false, and any
    /// kick that lands hits the `queues[REQ_QUEUE].ready()` early
    /// return in `drain_bracket_impl`. A worker spawned eagerly in
    /// `reset()` would sit in `epoll_wait` consuming a thread for
    /// an indeterminate window — the guest's rebind sequence may
    /// take milliseconds to seconds depending on driver
    /// implementation. Deferring the spawn until DRIVER_OK lifts
    /// the cost only when there is real work to service. This
    /// matches cloud-hypervisor's "kill on reset, respawn on
    /// DRIVER_OK" pattern.
    ///
    /// # Race-free invariant
    ///
    /// Both `reset_engine_spawned` and `set_status` execute on the
    /// vCPU thread that received the MMIO write — `reset()` from
    /// `STATUS = 0` and `set_status` from `STATUS = …`. The two
    /// run sequentially within a single vCPU thread context, so
    /// the `respawn_pending` field has no concurrent reader/writer.
    /// A regression that moved either path off the vCPU thread
    /// would need to add explicit synchronisation.
    ///
    /// # Failure consequences
    ///
    /// If `reset_engine_spawned` populated `respawn_pending` but
    /// `respawn_worker` (called from `set_status` on DRIVER_OK)
    /// fails to construct fresh fds or spawn the thread, the
    /// device enters the same permanent-workerless state described
    /// in `respawn_worker`'s "Failure consequences" section. A
    /// reset that produces `respawn_pending = None` (the
    /// `stop_worker_and_reclaim_state` non-Joined outcomes) means
    /// no state to respawn from; the device is permanently dead.
    /// In either case `set_status` clears `respawn_pending` to
    /// avoid a stale state holding scratch buffers and the
    /// backing-file `File` handle alive past the device's
    /// effective lifetime.
    pub(crate) respawn_pending: Option<BlkWorkerState>,
}

/// Process-wide monotonic counter for VirtioBlk instance IDs. Used
/// to derive `instance_id` at construction so tracing logs name the
/// device with a stable small integer instead of a raw heap pointer.
/// Heap pointers expose ASLR offsets and process-layout details
/// (the `host_resource_snapshot` doc treats this kind of detail as
/// environment leakage); a per-process counter preserves the
/// "uniquely identify the device within this process run" property
/// that the diagnostics depend on without leaking the address.
pub(crate) static VIRTIO_BLK_INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Virtio-block MMIO device.
pub struct VirtioBlk {
    pub(crate) queue_select: u32,
    pub(crate) device_features_sel: u32,
    pub(crate) driver_features_sel: u32,
    pub(crate) driver_features: u64,
    /// FSM state bits per virtio-v1.2 §3.1.1 plus the
    /// `VIRTIO_CONFIG_S_NEEDS_RESET` bit set by `drain_bracket_impl`
    /// when the avail ring becomes structurally invalid (the
    /// queue-poison path). `Arc<AtomicU32>` so the worker thread can
    /// fetch_or the NEEDS_RESET bit alongside its INT_CONFIG +
    /// `irq_evt.write(1)` poison-signal sequence; the vCPU thread
    /// reads `STATUS` via `load(Acquire)` from `mmio_read` and writes
    /// it via the FSM in `set_status` / `reset`. Atomic ordering
    /// taxonomy: `set_status` uses
    /// `compare_exchange(_, _, Release, Acquire)` for race-safe
    /// FSM advance against the worker's concurrent
    /// `fetch_or(NEEDS_RESET)` (the only RMW write site on
    /// device_status from the vCPU thread); `reset` uses
    /// `store(0, Release)`; vCPU reads use `load(Acquire)`
    /// (mmio_read, queue_config_allowed, features_write_allowed);
    /// the worker uses `fetch_or(NEEDS_RESET, SeqCst)` on the
    /// queue-poison path. Mirrors the [`Self::interrupt_status`]
    /// shape and rationale.
    pub(crate) device_status: Arc<AtomicU32>,
    /// Worker may be on a separate thread (production cfg) and the
    /// vCPU MMIO reader may race the worker's bit-set, so the value
    /// is wrapped in an `Arc` and updated with atomic ops. Worker
    /// writes the bit via `fetch_or(VIRTIO_MMIO_INT_VRING, Release)`
    /// alongside its `add_used` publish; vCPU `mmio_read` of
    /// `INTERRUPT_STATUS` does `load(Acquire)`; `INTERRUPT_ACK`
    /// clears bits via `fetch_and(!val, AcqRel)`. The Release/Acquire
    /// pair orders the bit set vs. the used-ring `add_used` (which
    /// itself publishes `used.idx` with Release internally), so a
    /// vCPU reading the bit set is guaranteed to also observe the
    /// freshly-published used.idx — no torn observation where the
    /// bit appears before the ring update.
    pub(crate) interrupt_status: Arc<AtomicU32>,
    /// `AtomicU32` for consistency with `interrupt_status`; v0 bumps
    /// only from `reset()` on the vCPU thread, not from any other
    /// thread (the worker thread does not touch config space). The
    /// atomic shape is defense-in-depth for future runtime config
    /// changes that might add a non-vCPU writer.
    pub(crate) config_generation: AtomicU32,
    /// Eventfd for KVM irqfd. Shared `Arc` so the worker thread
    /// (production cfg) can call `write(1)` to fire the IRQ without
    /// taking ownership away from the device. Tests run inline so
    /// the same Arc is read directly via `dev.irq_evt.read()`.
    pub(crate) irq_evt: Arc<EventFd>,
    /// Guest memory reference. Set before starting vCPUs via
    /// `set_mem`. Wrapped in `Arc<OnceLock<…>>` so the worker
    /// thread (production) can pick up `mem` post-construction
    /// without locking on every drain. `set_mem` is the only
    /// writer and KVM wiring guarantees it runs before any vCPU
    /// runs (i.e. before any QUEUE_NOTIFY can fire), so the
    /// reader-side `OnceLock::get` is lock-free in steady state
    /// and returns `&GuestMemoryMmap` directly — no clone needed.
    /// `reset()` does NOT clear `mem`: the same guest memory map
    /// is re-used across re-binds (matching the behaviour of the
    /// original `Mutex<Option<…>>` field, which `set_mem` only
    /// overwrote at boot).
    pub(crate) mem: Arc<OnceLock<GuestMemoryMmap>>,
    /// Capacity in 512-byte sectors. Determines what the guest sees
    /// in the config space's `capacity` field.
    pub(crate) capacity_sectors: u64,
    /// Request-processing state. In production a worker thread owns
    /// the underlying `BlkWorkerState`; in `cfg(test)` the state is
    /// inline so existing tests can read it back synchronously.
    pub(crate) worker: BlkWorker,
    /// One-shot guard so the "queue notify before set_mem"
    /// warning fires at most once per device instance.
    /// Without this, a buggy caller that issues N notifies before
    /// `set_mem` would flood the log with N copies of the same
    /// message. Latched with Relaxed because the order of the
    /// log message vs. other operations doesn't affect
    /// correctness. `Arc` so the worker thread can also
    /// access-and-latch the same flag (production: the warn fires
    /// the first time the worker observes the unset mem during a
    /// drain).
    pub(crate) mem_unset_warned: Arc<AtomicBool>,
    /// Original throttle configuration. Stored so `reset()` can
    /// rebuild fresh `TokenBucket`s on the respawned worker. Per
    /// virtio-v1.2 §2.1 a reset returns the device to its initial
    /// state, which includes the throttle bucket fill: an
    /// adversarial guest must not be able to drain the bucket and
    /// then issue a reset to bypass the rate limit. `DiskThrottle`
    /// is `Copy` (a pair of `Option<NonZeroU64>`) so this is cheap
    /// to keep around.
    pub(crate) throttle: DiskThrottle,
    /// Stable per-device monotonic identifier from
    /// [`VIRTIO_BLK_INSTANCE_COUNTER`]. Replaces the previous
    /// `self as *const _ as usize` heap-pointer field for tracing
    /// log correlation: pointers fingerprint the host's ASLR
    /// layout, an integer counter does not.
    pub(crate) instance_id: u64,
    /// Pause eventfd (host-side handle). [`Self::pause`] writes 1 to
    /// signal the worker; the worker reads the counter and parks on
    /// [`Self::paused`]. Shared `Arc` because the worker owns a clone
    /// for its epoll registration and the device retains this handle
    /// for `pause()`/`resume()` calls from the freeze coordinator.
    /// Cfg-independent so [`Self::pause`] / [`Self::resume`] compile
    /// in `cfg(test)` builds (where the inline engine is a no-op
    /// because the worker thread does not exist).
    pub(crate) pause_evt: Arc<EventFd>,
    /// Worker-parked indicator. Set to `true` by the worker thread
    /// after it drains `pause_fd` and is parked in the
    /// `park_timeout`-loop; the freeze coordinator polls this with
    /// `load(Acquire)` to confirm the worker has reached its parked
    /// state before reading guest memory. Cleared by [`Self::resume`]
    /// (Release-store of `false`); the worker's `park_timeout(10ms)`
    /// observes the clear within 10 ms and resumes its `epoll_wait`
    /// loop.
    pub(crate) paused: Arc<AtomicBool>,
    /// Optional shared parked_evt the worker writes to alongside
    /// the `paused.store(true, Release)` so the freeze
    /// coordinator's rendezvous wakes within microseconds of the
    /// last parker rather than spinning. `None` when no freeze
    /// coordinator is plumbed (test paths). The freeze coordinator
    /// sets this on every device via [`Self::set_parked_evt`]
    /// before the first `pause()` call. Counter-mode EventFd
    /// (NOT EFD_SEMAPHORE): a single drain absorbs any number of
    /// coalesced parker writes.
    pub(crate) parked_evt: Arc<std::sync::Mutex<Option<Arc<EventFd>>>>,
    /// Per-thread CPU placement applied at the top of
    /// `worker_thread_main` before the worker enters its `epoll_wait`
    /// loop. Mirrors the host topology's perf-mode (`pin_target`) and
    /// `--cpu-cap` no-perf (`no_perf_cpus`) split: at most one of the
    /// two is `Some`, both `None` means inherit the parent thread's
    /// affinity (no placement applied). Set via
    /// [`Self::set_worker_placement`] after `with_options`; defaults
    /// to all-`None` so the device works in test fixtures and call
    /// sites that don't supply topology data.
    pub(crate) worker_placement: WorkerPlacement,
}

/// CPU placement for the virtio-blk worker thread. Threaded into
/// `worker_thread_main` and applied via `pin_current_thread` /
/// `set_thread_cpumask` at the top of the worker before entering
/// `epoll_wait`. Mutually exclusive: perf-mode picks a single CPU,
/// `--cpu-cap` no-perf picks an LLC mask, both `None` means inherit
/// the parent thread's affinity (the test/inline path).
// Fields are read by `worker_thread_main` which is itself
// `#[cfg(not(test))]` (worker.rs), so under `cargo check
// --tests` no reader exists and the fields look dead. The
// production path consumes them — keep the allow.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct WorkerPlacement {
    /// Single CPU pin (perf-mode). Equivalent to
    /// `pin_current_thread(cpu, "virtio-blk worker")`.
    pub service_cpu: Option<usize>,
    /// CPU mask (no-perf + `--cpu-cap`). Equivalent to
    /// `set_thread_cpumask(cpus, "virtio-blk worker")`.
    pub no_perf_cpus: Option<Vec<usize>>,
}

impl VirtioBlk {
    /// Create a new virtio-block device.
    ///
    /// `backing` is an open File for read+write at sector
    /// granularity (the host formatted it before VM boot).
    /// `capacity_bytes` is the disk capacity advertised to the
    /// guest (rounded down to sector boundary). `throttle` carries
    /// optional IOPS / bandwidth limits.
    ///
    /// `dead_code` allow: only consumed by `#[cfg(test)]` modules;
    /// production callers go through [`Self::with_options`] to set
    /// the read-only flag explicitly.
    #[allow(dead_code)]
    pub fn new(backing: File, capacity_bytes: u64, throttle: DiskThrottle) -> Self {
        Self::with_options(backing, capacity_bytes, throttle, false)
    }

    /// Like [`Self::new`] plus a `read_only` knob. When `read_only`
    /// is `true`, the device advertises `VIRTIO_BLK_F_RO` and
    /// rejects writes regardless of guest behaviour (defense
    /// against a guest that ignores the negotiated feature bit).
    ///
    /// `capacity_bytes` smaller than one sector is clamped to
    /// `capacity_sectors = 0`, producing a 0-sector disk that
    /// IOERRs every request. The device cannot represent a
    /// fractional sector, so a sub-sector allocation is operator
    /// error — log it, continue, and let the existing zero-capacity
    /// reject path surface the failure to the guest.
    pub fn with_options(
        backing: File,
        capacity_bytes: u64,
        throttle: DiskThrottle,
        read_only: bool,
    ) -> Self {
        let irq_evt = Arc::new(
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk irq eventfd"),
        );
        if capacity_bytes < VIRTIO_BLK_SECTOR_SIZE as u64 && capacity_bytes != 0 {
            tracing::warn!(
                capacity_bytes,
                sector_size = VIRTIO_BLK_SECTOR_SIZE,
                "virtio-blk capacity_bytes smaller than one sector; clamping \
                 capacity_sectors to 0 (every IO will be rejected)"
            );
        }
        let capacity_sectors = capacity_bytes / VIRTIO_BLK_SECTOR_SIZE as u64;
        let capacity_bytes = capacity_sectors * VIRTIO_BLK_SECTOR_SIZE as u64;
        let (ops_bucket, bytes_bucket) = buckets_from_throttle(throttle);
        let counters = Arc::new(VirtioBlkCounters::default());

        let state = BlkWorkerState {
            // Wrap the production `File` in the `Backing` seam (unsizing
            // coercion `Box<File>` -> `Box<dyn Backing>`). Tests swap
            // `state.backing` for a fault-injecting mock to exercise
            // the write retry / IOERR paths a real File rarely hits.
            backing: Box::new(backing),
            ops_bucket,
            bytes_bucket,
            all_descs_scratch: Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2),
            io_buf_scratch: Vec::new(),
            capacity_bytes,
            read_only,
            counters: Arc::clone(&counters),
            currently_stalled: false,
            queue_poisoned: false,
        };

        let interrupt_status = Arc::new(AtomicU32::new(0));
        let device_status = Arc::new(AtomicU32::new(0));
        let mem = Arc::new(OnceLock::new());
        let mem_unset_warned = Arc::new(AtomicBool::new(false));
        // Pause primitives (failure-dump rendezvous). The
        // `pause_evt` host handle is kept on the `VirtioBlk` for
        // `pause()`/`resume()`; in production a clone of its read
        // side becomes the `pause_fd` registered in the worker's
        // epoll. `paused` is the worker-set / coordinator-cleared
        // ack flag the freeze rendezvous polls. Both Arcs are
        // cfg-independent so the test-mode `pause`/`resume`
        // accessors compile without engine-conditional plumbing
        // (the test-mode worker is inline, so they observe the
        // same eventfd state without an active worker thread).
        let pause_evt = Arc::new(
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk pause eventfd"),
        );
        // Initialise to `true` so the freeze coordinator's
        // `is_paused()` poll passes vacuously while no worker is
        // alive — the initial spawn is deferred to DRIVER_OK
        // (see `respawn_pending` engine plumbing below), so any
        // freeze that fires between `with_options` and the first
        // DRIVER_OK MMIO write would otherwise time out at
        // FREEZE_RENDEZVOUS_TIMEOUT (30 s) waiting for a worker
        // that does not exist. The worker's first action inside
        // `worker_thread_main` (after affinity setup, before
        // entering `epoll_wait`) is a Release-store of `false`,
        // which makes the rendezvous start observing real
        // worker state from the moment the worker is genuinely
        // ready to service kicks. Cloud-hypervisor uses the
        // same "paused on construction, cleared by activate"
        // invariant in epoll_helper.rs.
        let paused = Arc::new(AtomicBool::new(true));

        // Build the queue. Production uses `QueueSync` (Arc<Mutex<Queue>>
        // internally) so the vCPU MMIO config writes and the worker
        // thread's drain can share the same queue state. Tests use the
        // bare `Queue` so the existing test surface that drives queue
        // methods directly via `dev.worker.queues[REQ_QUEUE].…` keeps
        // working without a runtime lock.
        let queues = [BlkQueue::new(QUEUE_MAX_SIZE).expect("valid queue size")];

        // Build the engine. cfg(test) keeps the state inline so the
        // existing test surface drives drain_bracket_impl synchronously;
        // cfg(not(test)) spawns a dedicated worker thread that owns
        // the state and waits for kick eventfd writes from
        // `process_requests`.
        #[cfg(test)]
        let engine = WorkerEngine::Inline(InlineEngine { state });

        #[cfg(not(test))]
        let engine = {
            // Counter-mode eventfds (no EFD_SEMAPHORE). EFD_NONBLOCK so
            // the vCPU `write(1)` to kick_fd never blocks even under
            // pathological backpressure (the worker has fallen behind
            // by more than u64::MAX-1 kicks — implausible under any
            // realistic workload, but the non-blocking flag keeps the
            // failure mode "EAGAIN, drop the spurious kick" instead of
            // "vCPU thread blocks on eventfd write").
            let kick_fd =
                EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk kick eventfd");
            let stop_fd =
                EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-blk stop eventfd");
            // Defer the initial worker spawn to the guest's first
            // DRIVER_OK transition (set_status → consume_pending_respawn
            // → respawn_worker). Stashing the seed `BlkWorkerState` in
            // `respawn_pending` collapses the initial-spawn path into
            // the existing respawn path, which `respawn_worker` already
            // implements correctly (placement applied via
            // `self.worker_placement` clone, fresh kick/stop/pause fds
            // built per spawn). This gives `set_worker_placement` a
            // race-free window between construction and DRIVER_OK in
            // which to override the default placement; without
            // deferral the initial worker would spawn with the default
            // placement before setup.rs's setter call could land.
            //
            // Pre-DRIVER_OK kicks land on the now-detached `kick_fd`
            // and accumulate harmlessly; the first post-DRIVER_OK
            // worker observes the queue's `ready()` flag and processes
            // any pre-existing chain state. The kernel's virtio-mmio
            // bind sequence (drivers/virtio/virtio_mmio.c
            // `virtio_mmio_probe` → `vp_finalize_features` →
            // `vm_setup_vq` → `STATUS=DRIVER_OK`) does not fire
            // QUEUE_NOTIFY before DRIVER_OK, so accumulation is
            // bounded at zero in the production path.
            WorkerEngine::Spawned(SpawnedEngine {
                kick_fd,
                stop_fd,
                handle: None,
                respawn_pending: Some(state),
            })
        };

        let worker = BlkWorker {
            queues,
            read_only,
            counters,
            engine,
        };

        VirtioBlk {
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status,
            interrupt_status,
            config_generation: AtomicU32::new(0),
            irq_evt,
            mem,
            capacity_sectors,
            worker,
            mem_unset_warned,
            throttle,
            instance_id: VIRTIO_BLK_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed),
            pause_evt,
            paused,
            parked_evt: Arc::new(std::sync::Mutex::new(None)),
            worker_placement: WorkerPlacement::default(),
        }
    }

    /// Plumb the freeze coordinator's shared parked_evt into this
    /// device. The worker writes to this fd alongside its
    /// `paused.store(true, Release)` so the coordinator's
    /// rendezvous wakes within microseconds of the worker
    /// parking. Called once by `run_vm` before any pause()/resume()
    /// fires; subsequent worker respawns pick up the same fd via
    /// the shared `Arc`.
    ///
    /// `None` is the default — test paths and any future device
    /// without a freeze coordinator skip the wake. The worker
    /// reads through this slot lazily so a setter call AFTER worker
    /// spawn (e.g. plumbing arrives late) still takes effect on
    /// the next pause cycle.
    pub fn set_parked_evt(&self, evt: Arc<EventFd>) {
        if let Ok(mut guard) = self.parked_evt.lock() {
            *guard = Some(evt);
        }
    }

    /// Configure the per-thread CPU placement applied at the top of
    /// the worker's main loop. Mirrors the `set_mem` setter pattern:
    /// called once after `with_options` / `new`, before the device
    /// starts servicing kicks. The placement is captured by the
    /// next worker-thread spawn — and because the initial spawn is
    /// DEFERRED to the guest's first `STATUS = DRIVER_OK` MMIO
    /// write (the seed `BlkWorkerState` lives in `respawn_pending`
    /// until then), a setter call between `with_options` and that
    /// DRIVER_OK transition lands on the very first worker. After
    /// the worker has started, calling this has no effect on the
    /// running thread — only respawned workers pick up the new
    /// placement, matching cloud-hypervisor's "topology applied at
    /// thread start" pattern.
    ///
    /// `WorkerPlacement::service_cpu` and `no_perf_cpus` are mutually
    /// exclusive — the topology layer (perf-mode vs `--cpu-cap`)
    /// produces at most one. Both `None` means inherit the parent
    /// thread's affinity (the test/inline path and the no-topology
    /// fallback for ad-hoc fixtures).
    pub fn set_worker_placement(&mut self, placement: WorkerPlacement) {
        self.worker_placement = placement;
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    ///
    /// Stores the memory inside the device's shared `Arc<OnceLock<…>>`
    /// so the worker thread (production cfg) can observe the
    /// reference on its next drain via a lock-free
    /// `OnceLock::get`. Returning before the worker observes the
    /// value is safe because the worker only consults `mem` in
    /// response to a kick (driven by `mmio_write` of QUEUE_NOTIFY),
    /// and KVM wiring guarantees `set_mem` runs before any vCPU
    /// runs (i.e. before any QUEUE_NOTIFY can fire).
    ///
    /// `OnceLock::set` returns `Err` if the slot is already
    /// populated. The current production wiring (mod.rs `init_virtio_blk`)
    /// calls `set_mem` exactly once per device, so the `Err` branch
    /// is unreachable in normal operation; `reset()` does NOT clear
    /// `mem`, matching the prior `Mutex<Option<…>>` semantics where
    /// the slot was only written at boot. Log on `Err` rather than
    /// panic so a future re-wire bug surfaces as a warning instead
    /// of aborting (a panic here could land mid-teardown when the
    /// caller is already unwinding).
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        if self.mem.set(mem).is_err() {
            tracing::warn!(
                "virtio-blk: set_mem called on already-initialised \
                 device; guest memory binding unchanged (mem is set \
                 once at boot and preserved across reset())"
            );
        }
    }

    /// Advertised capacity in 512-byte sectors.
    ///
    /// `dead_code` allow: used by tests to read back the
    /// rounded-to-sector capacity; the lib pipeline consumes
    /// the `capacity_bytes` input directly through the
    /// config-space rendering path, so the accessor would
    /// otherwise appear unused at lib-build time.
    #[allow(dead_code)]
    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    /// Cloneable handle to the host-observability counters. The
    /// monitor thread holds an Arc to read counters without locking
    /// the device.
    pub fn counters(&self) -> Arc<VirtioBlkCounters> {
        Arc::clone(&self.worker.counters)
    }

    /// Cloneable handle to the worker's parked-state flag. The
    /// freeze coordinator holds an `Arc<AtomicBool>` and polls it in
    /// the post-thaw barrier and timeout-diagnostic paths without
    /// taking the device's `PiMutex`. Reading via the device handle
    /// ([`Self::is_paused`]) requires `Arc<PiMutex<VirtioBlk>>::lock`,
    /// which contends with every concurrent device operation —
    /// `mmio_read`/`mmio_write` from the vCPU thread and any other
    /// freeze-coord call site holding the lock. Since the
    /// underlying field is already `Arc<AtomicBool>`, exposing it
    /// directly lets the rendezvous loop poll it lock-free; the
    /// Acquire/Release ordering on `paused` provides the same
    /// happens-before edges with the worker's parked-state writes
    /// that [`Self::is_paused`] does.
    pub fn paused_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.paused)
    }

    pub(crate) fn device_features(&self) -> u64 {
        // VIRTIO_F_VERSION_1: modern virtio.
        // VIRTIO_BLK_F_BLK_SIZE: config carries the logical
        //   block size at offset 0x14 (4 bytes LE).
        // VIRTIO_BLK_F_SEG_MAX: config carries the per-request
        //   max scatter-gather segment count at offset 0x0C.
        //   Without this bit the guest defaults max_segments to 1
        //   and a multi-segment bio gets split serially —
        //   throughput-cripplingly slow for our use case.
        // VIRTIO_BLK_F_SIZE_MAX: config carries the per-descriptor
        //   max byte length at offset 0x08. Without this, a guest
        //   can submit a single 4 GB descriptor and force the
        //   device to allocate a matching `Vec<u8>` for I/O —
        //   host OOM hazard on a hostile guest.
        // VIRTIO_BLK_F_FLUSH: device honours T_FLUSH (fdatasync).
        //   Durability semantics depend on the host filesystem
        //   backing the per-test image: for the default
        //   `tempfile()` path on tmpfs, fdatasync is effectively a
        //   no-op (tmpfs doesn't persist across reboot anyway —
        //   only ordering semantics). For btrfs/ext4-backed run
        //   dirs, fdatasync provides the standard kernel-level
        //   ordering guarantee. Advertising F_FLUSH lets the guest
        //   block layer issue REQ_OP_FLUSH at metadata-commit
        //   boundaries (btrfs in the guest depends on this for
        //   tree consistency).
        // VIRTIO_BLK_F_RO: gated on `read_only`. The kernel block
        //   layer marks the disk read-only via `set_disk_ro` after
        //   F_RO negotiation; the device's defensive T_OUT
        //   rejection guards against uncovered write paths.
        // VIRTIO_RING_F_EVENT_IDX (transport feature, bit 29): the
        //   guest can place an "event index" in the avail ring's
        //   `used_event` field telling the device "do not interrupt
        //   me until used.idx reaches this value." `virtio_queue`
        //   tracks that field internally — when this bit is
        //   negotiated, `Queue::needs_notification` returns false
        //   while the guest's threshold is not reached and the
        //   device can suppress the irqfd write. Notification
        //   suppression on bursty IO (multi-chain queue drains)
        //   reduces vCPU exits proportional to the burst size and
        //   is required for high-throughput virtio-blk under
        //   blk-mq. Wire-up: this advertises the bit, set_status
        //   enables event-idx tracking on the queue when FEATURES_OK
        //   negotiates it, and process_requests consults
        //   `Queue::needs_notification` after each drain to decide
        //   whether to fire the irqfd. The V8 split: process_requests
        //   sets VIRTIO_MMIO_INT_VRING unconditionally on any chain
        //   publish, then consults needs_notification to decide
        //   whether to also fire the irqfd.
        let mut feats = (1u64 << VIRTIO_F_VERSION_1)
            | (1u64 << VIRTIO_BLK_F_BLK_SIZE)
            | (1u64 << VIRTIO_BLK_F_SEG_MAX)
            | (1u64 << VIRTIO_BLK_F_SIZE_MAX)
            | (1u64 << VIRTIO_BLK_F_FLUSH)
            | (1u64 << VIRTIO_RING_F_EVENT_IDX);
        if self.worker.read_only {
            feats |= 1u64 << VIRTIO_BLK_F_RO;
        }
        feats
    }

    pub(crate) fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    pub(crate) fn queue_config_allowed(&self) -> bool {
        let status = self.device_status.load(Ordering::Acquire);
        status & S_FEAT == S_FEAT && status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    pub(crate) fn features_write_allowed(&self) -> bool {
        let status = self.device_status.load(Ordering::Acquire);
        status & S_DRV == S_DRV && status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    /// Service `VIRTIO_BLK_T_IN` (read) using a single `preadv(2)`
    /// syscall over a vectored iov chain built from the data
    /// segments. Functionally equivalent to
    /// `Self::handle_read_impl` but coalesces N `pread64` syscalls
    /// (one per segment) plus N memcpy passes (kernel→scratch then
    /// scratch→guest) into one syscall reading directly into guest
    /// memory.
    ///
    /// Mirrors cloud-hypervisor's `block::Request::execute_async`
    /// vectored read path: one iovec per `VolatileSlice` produced by
    /// `mem.get_slices(addr, len)`, then a single vectored read.
    /// `get_slices` handles fragmentation when a descriptor's
    /// `[addr, addr+len)` range spans a guest memory region boundary —
    /// each contiguous host range becomes its own iovec entry.
    ///
    /// `data_len` and `sector` are pre-validated by the caller
    /// (`drain_bracket_impl`): SIZE_MAX, SEG_MAX, sub-sector,
    /// direction, and out-of-range checks all run upstream. The
    /// per-segment direction check is repeated here as
    /// defense-in-depth — matching `Self::handle_read_impl` —
    /// so a future caller that bypasses `drain_bracket_impl` and
    /// calls this helper directly cannot smuggle a device-readable
    /// segment into a T_IN chain (which would have `preadv` write
    /// into a buffer the spec marked read-only from the device's
    /// perspective).
    ///
    /// Short-read handling: `preadv` returns `n` bytes filled
    /// (`n <= data_len`). When `n < data_len` (only reachable on a
    /// short read against a backing whose effective length is below
    /// `capacity_bytes` — pre-validated against `capacity_bytes` by
    /// the caller, so this path is rare in production) the unfilled
    /// `[n..data_len)` byte range is zero-padded by walking the
    /// segments forward from byte `n` and writing zero bytes via
    /// `mem.write_slice`. This mirrors the existing per-segment
    /// short-read pad in `Self::handle_read_impl` and matches the
    /// sparse-file semantic the original implementation relied on.
    ///
    /// Counter taxonomy is preserved exactly:
    /// - `record_read(bytes_from_backing)`: bytes ACTUALLY returned
    ///   by `preadv` (`n`), excluding the zero-pad tail.
    /// - `used.elem.len = bytes_to_guest + 1`: full `data_len` (data
    ///   + zero-pad tail) + 1 status byte (virtio-v1.2 §2.7.7.2 —
    ///     bytes the device wrote into device-writable buffers).
    ///
    /// `too_many_arguments` allow: same disjoint-borrow shape as
    /// `Self::handle_read_impl` — every parameter is a separate
    /// `&self` field that must be passed by reference so the caller
    /// can hold a concurrent mutable borrow of the queues vec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_read_vectored_impl(
        backing: &dyn Backing,
        capacity_bytes: u64,
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        data_len: u64,
    ) -> (u8, u32) {
        let Some(base_offset) = sector.checked_mul(VIRTIO_BLK_SECTOR_SIZE as u64) else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        if base_offset
            .checked_add(data_len)
            .is_none_or(|end| end > capacity_bytes)
        {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }

        // Build the iovec chain: one entry per VolatileSlice produced
        // by `mem.get_slices(addr, len)`. `get_slices` iterates over
        // the contiguous host ranges that together cover the guest
        // address span, so a descriptor whose `[addr, addr+len)`
        // straddles a `GuestMemoryMmap` region boundary contributes
        // multiple iovec entries — preserving correctness without
        // requiring the descriptor to fit in a single region.
        //
        // The `PtrGuardMut`s returned by `slice.ptr_guard_mut()` are
        // collected into `_guards` so they remain alive for the
        // duration of the syscall: with the `xen` feature enabled
        // they wrap an `MmapXenSlice` whose Drop unmaps the host
        // mapping; without `xen` the guard is a thin wrapper around
        // the raw pointer. Either way, holding the guards across
        // the syscall guarantees the `iov_base` pointers stay valid.
        //
        // Local Vec rather than a reusable scratch on `BlkWorkerState`:
        // `libc::iovec` contains a raw pointer, which is `!Send`, so
        // storing a `Vec<libc::iovec>` on `BlkWorkerState` would make
        // the whole struct `!Send` and break the `JoinHandle<…>`
        // payload. The initial capacity is a `VIRTIO_BLK_SEG_MAX + 2`
        // hint (one entry per data segment plus header+status); the
        // Vec grows via reallocation if multi-region descriptors
        // fragment into more iovec entries. The per-call allocation
        // is amortized against the single `preadv` syscall it
        // replaces — a quantum of overhead vastly smaller than the
        // N kernel-mode syscall transitions the legacy per-segment
        // path performed.
        let mut iovecs: Vec<libc::iovec> = Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2);
        let mut _guards: Vec<vm_memory::volatile_memory::PtrGuardMut> =
            Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2);
        for seg in data_segments {
            if !seg.is_write_only {
                // Spec violation — a T_IN request's data SGs must
                // be device-writable. Defense-in-depth: the outer
                // gate in process_requests already rejected this
                // chain before throttle. Mirrors the same check in
                // `Self::handle_read_impl` so a future caller
                // that bypasses `drain_bracket_impl` cannot reach
                // `preadv` with a device-readable buffer.
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, 1);
            }
            let len = seg.len as usize;
            if len == 0 {
                // Zero-length data descriptor: legal per virtio
                // (qemu/firecracker accept). Skip — preadv with a
                // zero-length iovec entry is a no-op and skipping
                // avoids an unnecessary `get_slices` round-trip.
                continue;
            }
            for slice_result in mem.get_slices(seg.addr, len) {
                let slice = match slice_result {
                    Ok(s) => s,
                    Err(_) => {
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, 1);
                    }
                };
                let guard = slice.ptr_guard_mut();
                iovecs.push(libc::iovec {
                    iov_base: guard.as_ptr() as *mut libc::c_void,
                    iov_len: slice.len(),
                });
                _guards.push(guard);
            }
        }

        // Tripwire for the structural iovec-count bound derived in the
        // SAFETY comment below (SEG_MAX * 2 = 256 <= IOV_MAX). Debug-
        // only (zero release cost); a future change that broke the
        // bound trips the proptest fuzzers here instead of surfacing as
        // a runtime `preadv` EINVAL.
        debug_assert!(
            iovecs.len() <= IOV_MAX,
            "blk preadv iovcnt {} exceeds IOV_MAX {} — structural bound broke (see SAFETY comment)",
            iovecs.len(),
            IOV_MAX,
        );

        // Empty iovec means data_len == 0 — every data descriptor in
        // the chain had len == 0. The upstream zero-data gate in
        // drain.rs gates on `data_segments.is_empty()`, NOT on
        // `data_len == 0`, so a chain with one or more zero-length
        // data descriptors passes the gate and reaches here. Linux
        // `preadv` with iovcnt=0 returns 0 (lib/iov_iter.c
        // `iovec_from_user`: "Linux has traditionally returned zero
        // for zero segments"), so a syscall would be harmless —
        // skipping it just avoids the kernel-mode round-trip on a
        // path that has nothing to do.
        let bytes_from_backing: u64 = if iovecs.is_empty() {
            0
        } else {
            // SAFETY: `iovecs` is a non-empty Vec of `libc::iovec`
            // entries built from valid host pointers (each came from
            // a `VolatileSlice` produced by `mem.get_slices(...)`
            // and the corresponding `PtrGuardMut` is alive in
            // `_guards` for the duration of this call, keeping the
            // backing pointer valid) — this upholds the
            // `Backing::preadv` precondition that each `iov_base` is
            // valid for `iov_len` bytes across the call. The backing
            // (`&dyn Backing`) is borrowed from the `BlkWorkerState`
            // the caller (`drain_bracket_impl`) owns for the drain.
            //
            // `iovecs.len()` is structurally bounded at <= 256 — a
            // quarter of Linux's `IOV_MAX` (UIO_MAXIOV = 1024) — so
            // `preadv` never sees `iovcnt > IOV_MAX`. Derived from
            // three enforced/structural facts:
            //   - at most `VIRTIO_BLK_SEG_MAX` (128) data segments:
            //     drain.rs rejects `chain_len > SEG_MAX + 2` before
            //     this loop runs;
            //   - each segment is <= `VIRTIO_BLK_SIZE_MAX` (1 MiB):
            //     drain.rs rejects `any(len > SIZE_MAX)` before this
            //     loop runs;
            //   - every `GuestMemoryMmap` region is >= 1 MiB and
            //     MiB-granular: numa_mem allocates per-node memory in
            //     whole MiB (a 0-MiB node is omitted) and the MMIO-gap
            //     split is GiB-aligned, so even the split pieces are
            //     MiB-granular (numa_mem.rs `push_node_regions`).
            // `mem.get_slices(addr, len)` returns one slice per
            // `GuestMemoryMmap` region the `[addr, addr+len)` range
            // spans. A <= 1 MiB segment over >= 1 MiB regions crosses
            // at most one region boundary -> at most 2 slices, so the
            // total is at most `SEG_MAX * 2 = 256`. This is a
            // STRUCTURAL invariant (the three facts above), NOT a
            // code-enforced cap: if a future change lowers the minimum
            // region size, raises SEG_MAX/SIZE_MAX, or makes
            // `get_slices` fragment more finely (e.g. per-page),
            // re-derive this bound or add an explicit
            // `iovcnt <= IOV_MAX` chunking loop.
            // `base_offset` is a `u64` validated above to fit in
            // `[0, capacity_bytes]`; `capacity_bytes` is host-trusted
            // (constructed in `with_options`) so it cannot exceed
            // `i64::MAX` for any realistic disk size and fits in
            // `off_t` losslessly.
            // Retry ONLY a signal-interrupted read (`Err(Interrupted)`,
            // 0 bytes transferred). A short POSITIVE return is EOF for a
            // regular file (a thin/truncated backing shorter than the
            // advertised capacity) — NOT a signal — so it exits the loop
            // and the short-read pad below zero-fills the `[n..data_len)`
            // tail. This is asymmetric to the write path: a read returns
            // at EOF, whereas a write must retry a short-positive to
            // completion. (read(2): a catchable-signal interrupt with 0
            // bytes yields `-EINTR` (Err); a partial-then-signal yields
            // the positive count, indistinguishable from EOF and treated
            // as EOF — retrying an at-EOF read would spin.)
            // Divergence: firecracker errors a short read
            // (`read_exact_volatile` → UnexpectedEof → S_IOERR); we
            // zero-pad it as EOF — the more lenient regular-file-EOF
            // choice, matching qemu's short-read `iov_memset` pad.
            let mut eintr_retries: u32 = 0;
            loop {
                // SAFETY: each iovec base points at live guest memory
                // held by `_guards` for this call; `iovecs.len()` is
                // structurally <= IOV_MAX (see the bound derivation
                // above).
                match unsafe { backing.preadv(&iovecs, base_offset) } {
                    Ok(n) => break n as u64,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                        // EINTR here is a pending FATAL signal (the read
                        // waits TASK_KILLABLE; a fatal signal does not
                        // clear), so re-issue is bounded to avoid
                        // spinning a dying IO thread — see
                        // MAX_EINTR_RETRIES.
                        eintr_retries += 1;
                        if eintr_retries > MAX_EINTR_RETRIES {
                            tracing::warn!(
                                sector,
                                "virtio-blk preadv: EINTR retry cap hit (pending fatal signal?); failing"
                            );
                            counters.record_io_error();
                            return (VIRTIO_BLK_S_IOERR as u8, 1);
                        }
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(sector, %e, "virtio-blk preadv error");
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, 1);
                    }
                }
            }
        };

        // Short-read pad: zero-fill the unfilled `[n..data_len)`
        // tail across the data segments. Walks segments forward,
        // skipping over already-filled bytes (`< n`) and zeroing
        // any remainder. Matches the per-segment behavior in
        // `handle_read_impl` (the unfilled tail of the segment that
        // straddles `n` plus all subsequent segments are zeroed).
        if bytes_from_backing < data_len {
            let mut filled = bytes_from_backing;
            let mut to_zero = data_len - bytes_from_backing;
            // Stack zero buffer sized to a fixed 64 KiB chunk.
            // Smaller than `VIRTIO_BLK_SIZE_MAX` (1 MiB) so a single
            // segment of the maximum size cannot be zeroed in one
            // pass; the inner `while remaining > 0` loop iterates
            // up to 16 times per segment, each iteration writing
            // one 64 KiB chunk via `mem.write_slice`. 64 KiB keeps
            // the stack footprint small while amortizing the
            // per-write_slice overhead across multiple sectors.
            const ZERO_BUF_LEN: usize = 65536;
            let zeros = [0u8; ZERO_BUF_LEN];
            for seg in data_segments {
                if to_zero == 0 {
                    break;
                }
                let seg_len = seg.len as u64;
                if filled >= seg_len {
                    // Segment already fully filled by preadv; skip.
                    filled -= seg_len;
                    continue;
                }
                // Zero from offset `filled` within this segment to
                // its end (or until `to_zero` runs out, whichever
                // comes first).
                let seg_offset = filled as u32;
                let seg_remaining = (seg_len - filled).min(to_zero) as u32;
                let Some(zero_addr_u64) = seg.addr.0.checked_add(seg_offset as u64) else {
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, 1);
                };
                let mut zero_addr = GuestAddress(zero_addr_u64);
                let mut remaining = seg_remaining;
                while remaining > 0 {
                    let chunk = (remaining as usize).min(ZERO_BUF_LEN);
                    if mem.write_slice(&zeros[..chunk], zero_addr).is_err() {
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, 1);
                    }
                    let Some(next) = zero_addr.0.checked_add(chunk as u64) else {
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, 1);
                    };
                    zero_addr = GuestAddress(next);
                    remaining -= chunk as u32;
                }
                to_zero -= seg_remaining as u64;
                filled = 0;
            }
        }

        counters.record_read(bytes_from_backing);
        // bytes_to_guest = full data_len (data + zero-pad tail). Cap
        // at u32::MAX; SEG_MAX (128) × SIZE_MAX (1 MiB) = 128 MiB ≪
        // u32::MAX, so the cast cannot truncate.
        let bytes_to_guest = data_len as u32;
        (VIRTIO_BLK_S_OK as u8, bytes_to_guest + 1)
    }

    /// Service `VIRTIO_BLK_T_OUT` (write) using a single `pwritev(2)`
    /// syscall over a vectored iov chain built from the data
    /// segments. Functionally equivalent to
    /// `Self::handle_write_impl` but coalesces N `pwrite64`
    /// syscalls plus N memcpy passes into one syscall writing
    /// directly from guest memory.
    ///
    /// Mirrors cloud-hypervisor's vectored write path with the same
    /// `iovecs.push(...)` build step followed by `write_vectored`.
    ///
    /// `data_len` and `sector` are pre-validated by the caller
    /// (`drain_bracket_impl`): SIZE_MAX, SEG_MAX, sub-sector,
    /// direction, RO-mode, and out-of-range checks all run upstream.
    /// The per-segment direction check is repeated here as
    /// defense-in-depth — matching `Self::handle_write_impl` —
    /// so a future caller that bypasses `drain_bracket_impl` cannot
    /// smuggle a device-writable segment into a T_OUT chain (which
    /// would have `pwritev` read from a buffer the spec marked
    /// device-only-writable from the driver's perspective).
    ///
    /// Short-write handling: `pwritev` may return `0 < n < data_len`
    /// (e.g. ENOSPC mid-write or a signal-interrupted partial). A
    /// short POSITIVE write is recoverable forward progress, so the
    /// loop advances the iovecs past `n` and re-issues the remainder
    /// until the full `data_len` lands — matching the kernel's
    /// `generic_perform_write` copy loop. Only `Ok(0)` (zero forward
    /// progress) or a hard `Err` is a genuine failure → S_IOERR + an
    /// `io_errors` bump; `Err(Interrupted)` is retried up to
    /// `MAX_EINTR_RETRIES`. This DIVERGES from the cfg(test)
    /// per-segment `Self::handle_write_impl`, which rejects on the
    /// first short `Ok(n)` — the single-shot reject was a
    /// guest-data-integrity bug (a completable write failed the
    /// guest's bio).
    ///
    /// Counter taxonomy is preserved exactly:
    /// - `record_write(total_written)`: bytes accepted by the
    ///   backing file. On success `total_written == data_len`.
    /// - `used.elem.len = 1` (status byte only — write data is
    ///   not written back into guest memory).
    ///
    /// `too_many_arguments` allow: same disjoint-borrow shape as
    /// [`Self::handle_read_vectored_impl`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_write_vectored_impl(
        backing: &dyn Backing,
        capacity_bytes: u64,
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        data_len: u64,
    ) -> (u8, u32) {
        let Some(base_offset) = sector.checked_mul(VIRTIO_BLK_SECTOR_SIZE as u64) else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        if base_offset
            .checked_add(data_len)
            .is_none_or(|end| end > capacity_bytes)
        {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }

        // Build the iovec chain. `get_slices` (not `get_slice`) so
        // a descriptor whose `[addr, addr+len)` straddles a region
        // boundary contributes multiple iovec entries. See the
        // `handle_read_vectored_impl` doc for the
        // `_guards`-keep-alive and per-call-allocation rationale.
        //
        // For T_OUT, the iovec entries are read-only with respect to
        // the syscall (pwritev READS the iov_base pointers), so we
        // hold `PtrGuard`s rather than `PtrGuardMut`s — no dirty
        // tracking is needed because we are not modifying guest
        // memory here.
        let mut iovecs: Vec<libc::iovec> = Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2);
        let mut _guards: Vec<vm_memory::volatile_memory::PtrGuard> =
            Vec::with_capacity(VIRTIO_BLK_SEG_MAX as usize + 2);
        for seg in data_segments {
            if seg.is_write_only {
                // Spec violation — a T_OUT request's data SGs must
                // be device-readable. Defense-in-depth: the outer
                // gate in process_requests already rejected this
                // chain before throttle. Mirrors the same check in
                // `Self::handle_write_impl` so a future caller
                // that bypasses `drain_bracket_impl` cannot reach
                // `pwritev` against a device-writable buffer.
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, 1);
            }
            let len = seg.len as usize;
            if len == 0 {
                continue;
            }
            for slice_result in mem.get_slices(seg.addr, len) {
                let slice = match slice_result {
                    Ok(s) => s,
                    Err(_) => {
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, 1);
                    }
                };
                let guard = slice.ptr_guard();
                iovecs.push(libc::iovec {
                    // `iovec.iov_base` is `*mut c_void` regardless of
                    // direction; the kernel reads from it for
                    // pwritev and writes to it for preadv. Casting
                    // the read-only pointer to `*mut` is fine because
                    // pwritev does not mutate the buffer; the
                    // mut-ness in the type is a libc convention, not
                    // a behavior contract.
                    iov_base: guard.as_ptr() as *mut libc::c_void,
                    iov_len: slice.len(),
                });
                _guards.push(guard);
            }
        }

        // Tripwire for the structural iovec-count bound (SEG_MAX * 2 =
        // 256 <= IOV_MAX); see the pwritev SAFETY comment below.
        // Debug-only; a future change that broke the bound trips the
        // proptest fuzzers here instead of a runtime `pwritev` EINVAL.
        debug_assert!(
            iovecs.len() <= IOV_MAX,
            "blk pwritev iovcnt {} exceeds IOV_MAX {} — structural bound broke (see SAFETY comment)",
            iovecs.len(),
            IOV_MAX,
        );

        if iovecs.is_empty() {
            // data_len == 0 — every data descriptor in the chain had
            // len == 0. The upstream zero-data gate in drain.rs
            // gates on `data_segments.is_empty()`, NOT on
            // `data_len == 0`, so a chain with one or more
            // zero-length data descriptors passes the gate and
            // reaches here. Linux `pwritev` with iovcnt=0 returns 0
            // for the same reason as `preadv` (see the read path
            // doc), so a syscall would be harmless — skipping it
            // just avoids the kernel-mode round-trip on a path
            // that has nothing to do.
            counters.record_write(0);
            return (VIRTIO_BLK_S_OK as u8, 1);
        }

        // Retry-to-completion. A SHORT positive write (`0 < n <
        // remaining`) is recoverable forward progress — `pwritev` may
        // transfer fewer bytes than requested on ENOSPC mid-write or a
        // signal-interrupted partial — so advance the iovecs past `n`
        // and re-issue the remainder until the full `data_len` lands.
        // Only `Ok(0)` (zero forward progress, e.g. immediate ENOSPC)
        // or a hard `Err` is a genuine failure → S_IOERR. The prior
        // single-pwritev returned S_IOERR on ANY short write, failing
        // the guest's bio for a write the device could have completed —
        // a guest-data-integrity bug. Asymmetric to the read path
        // (short positive read = EOF).
        //
        // Reference behavior (matches + intentional divergences):
        //   - kernel `generic_perform_write` (mm/filemap.c): MATCH —
        //     the page-copy loop advances the iter and re-copies until
        //     the byte count drains; a short write is recoverable
        //     progress returned as a positive partial.
        //   - firecracker `write_all_volatile` (vm-memory): MATCH —
        //     loops `offset(n)` past each short write until the buffer
        //     is empty; `Ok(0)` → WriteZero.
        //   - qemu file-posix: also retries to completion, but via a
        //     DIFFERENT mechanism — it abandons the vectored syscall on
        //     a short return and linearizes into a bounce buffer
        //     (`handle_aiocb_rw_linear` loops; returns EINVAL if a
        //     write is still short). We advance the iovecs and re-issue
        //     `pwritev` directly — same goal, no bounce-buffer copy.
        //   - cloud-hypervisor and libkrun: a silent-truncation bug we
        //     do NOT mirror — both accept a short positive write as
        //     success (cloud-hypervisor's block completion handler
        //     treats any `result >= 0` as S_OK with the partial byte
        //     count as `used.len`; libkrun's `consume` runs the write
        //     callback once, no retry), truncating the guest's write.
        let mut remaining: &mut [libc::iovec] = &mut iovecs;
        let mut off = base_offset;
        let mut total_written: u64 = 0;
        let mut eintr_retries: u32 = 0;
        loop {
            // SAFETY: each iovec base points at live guest memory held
            // by `_guards` for this call (`_guards` outlives the loop);
            // `remaining.len()` only shrinks from the
            // structurally-bounded initial `iovecs.len()` (<= IOV_MAX),
            // so iovcnt stays in range.
            let n = match unsafe { backing.pwritev(remaining, off) } {
                Ok(0) => {
                    tracing::warn!(
                        sector,
                        total_written,
                        data_len,
                        "virtio-blk pwritev made zero forward progress; failing"
                    );
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, 1);
                }
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    // EINTR here is a pending FATAL signal (the write
                    // path returns -EINTR under fatal_signal_pending; a
                    // fatal signal does not clear), so bound the
                    // re-issue to avoid spinning a dying IO thread and
                    // delaying the freeze dump — see MAX_EINTR_RETRIES.
                    eintr_retries += 1;
                    if eintr_retries > MAX_EINTR_RETRIES {
                        tracing::warn!(
                            sector,
                            "virtio-blk pwritev: EINTR retry cap hit (pending fatal signal?); failing"
                        );
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, 1);
                    }
                    continue;
                }
                Err(e) => {
                    tracing::warn!(sector, %e, "virtio-blk pwritev error");
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, 1);
                }
            };
            total_written += n as u64;
            off += n as u64;
            if total_written == data_len {
                break;
            }
            // total_written < data_len ⇒ n < the remaining iovec bytes,
            // so `advance_iovecs` returns a non-empty slice.
            remaining = advance_iovecs(remaining, n);
            debug_assert!(
                !remaining.is_empty(),
                "virtio-blk pwritev: iovecs exhausted at {total_written} of {data_len} bytes"
            );
        }
        counters.record_write(total_written);
        // used_len: 1 (status byte only — write data is not
        // written back into guest memory).
        (VIRTIO_BLK_S_OK as u8, 1)
    }

    /// Classify a request type into a "pre-throttle terminal status"
    /// when applicable. Returns `Some((status_byte, used_len))` for
    /// requests the device decides without ever touching the backing
    /// file or the throttle (RO-mode writes, RO-mode flushes, unknown
    /// request types). Returns `None` for the normal
    /// IN/OUT/FLUSH/GET_ID paths that need the backend handlers.
    ///
    /// Side effects per branch:
    ///
    /// - **RO-mode `T_OUT` → `S_IOERR`**: increments `io_errors`.
    ///   The guest negotiated `F_RO` (or, defensively, ignored it
    ///   and tried to write anyway). Either way, the device rejected
    ///   real work — that's an IO error from a host-observability
    ///   standpoint, surfaced in the failure-dump counters.
    /// - **RO-mode `T_FLUSH` → `S_OK`**: increments
    ///   `flushes_completed`. The guest issued a real flush; the
    ///   device serviced it (trivially, because nothing's dirty on
    ///   a read-only disk). The counter records the delivery, not
    ///   the work — symmetric with how the writable-disk path
    ///   bumps `flushes_completed` after `fdatasync` returns Ok.
    /// - **`T_GET_ID` → None (regardless of `read_only`)**: T_GET_ID
    ///   is a metadata read that never touches the backing file, so
    ///   the RO disk accepts it the same as a writable disk. Per
    ///   virtio-v1.2 §5.2.6.4 GET_ID is not gated on any feature
    ///   bit and is always accepted.
    /// - **Unknown type → `S_UNSUPP`**: NO counter bump. UNSUPP is
    ///   a graceful decline ("the device doesn't speak this
    ///   request"), not a service failure — the device never tried
    ///   anything that could fail.
    ///
    /// Counter writes belong with the classification because the
    /// dispatch decision IS the moment that bookkeeping happens —
    /// hoisting them out would force the caller to re-derive the
    /// request type.
    ///
    /// Extracted into a free associated function (no `&self`) so it
    /// can be tested directly without constructing a fully-wired
    /// `VirtioBlk` and so `process_requests` can call it while
    /// holding `&mut self.queues[..]`.
    pub(crate) fn classify_pre_throttle(
        req_type: u32,
        read_only: bool,
        counters: &VirtioBlkCounters,
    ) -> Option<(u8, u32)> {
        match req_type {
            VIRTIO_BLK_T_OUT if read_only => {
                counters.record_io_error();
                Some((VIRTIO_BLK_S_IOERR as u8, 1))
            }
            VIRTIO_BLK_T_FLUSH if read_only => {
                // No-op flush on a read-only disk: nothing dirty to
                // flush, but count it as a completed flush for
                // visibility in the failure-dump counters — the
                // guest issued a real flush and the device serviced
                // it.
                counters.record_flush();
                Some((VIRTIO_BLK_S_OK as u8, 1))
            }
            VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_FLUSH | VIRTIO_BLK_T_GET_ID => None,
            _ => Some((VIRTIO_BLK_S_UNSUPP as u8, 1)),
        }
    }

    // ------------------------------------------------------------------
    // Request queue processing
    // ------------------------------------------------------------------

    /// Drive the request queue. In `cfg(test)` the drain runs
    /// inline on the caller thread (preserving the synchronous
    /// test surface). In production this is a non-blocking
    /// kick of the worker thread's eventfd — `mmio_write` of
    /// `QUEUE_NOTIFY` returns immediately so the vCPU thread
    /// doesn't block on backing-file IO.
    pub(crate) fn process_requests(&mut self) {
        #[cfg(test)]
        {
            self.drain_inline();
        }
        #[cfg(not(test))]
        {
            // Non-blocking kick. The worker thread's epoll_wait
            // resumes and runs one drain iteration per kick. EAGAIN
            // (counter saturated at u64::MAX-1) is implausible under
            // any realistic workload — the worker would have to be
            // ~2^64 kicks behind — and on encountering it we drop
            // the spurious kick because counter-mode coalesces all
            // pending kicks into a single read by the worker on the
            // next wakeup, so no QUEUE_NOTIFY is permanently lost.
            let WorkerEngine::Spawned(eng) = &self.worker.engine;
            let _ = eng.kick_fd.write(1);
        }
    }

    /// Inline drain (test-mode only). Resolves the Inline engine,
    /// fetches a `&mem` reference from the shared `Arc<OnceLock<…>>`
    /// via a lock-free `OnceLock::get`, and calls `drain_bracket_impl`
    /// directly with the worker state + queue + irq + interrupt_status
    /// borrows. No clone is needed — `drain_bracket_impl` accepts
    /// `&GuestMemoryMmap` and the lifetime ends inside this fn.
    #[cfg(test)]
    pub(crate) fn drain_inline(&mut self) {
        let Some(mem) = self.mem.get() else {
            // Caller (kvm wiring in src/vmm/mod.rs) is supposed to
            // call `set_mem` before any vCPU runs. A queue-notify
            // before that is a wiring bug; surface it once per
            // device so the log isn't flooded if the guest spams
            // notifies on the broken setup.
            if !self.mem_unset_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "virtio-blk: queue notify before set_mem; \
                     dropping requests until guest memory is wired"
                );
            }
            return;
        };
        let WorkerEngine::Inline(engine) = &mut self.worker.engine;
        // The cfg(test) inline path discards the DrainOutcome —
        // tests step the throttle bucket forward via
        // `set_last_refill_for_test` and re-issue QUEUE_NOTIFY to
        // exercise the post-stall retry path. There is no timer
        // arming because there is no worker thread to wake.
        let _ = super::drain_bracket_impl(
            &mut engine.state,
            &mut self.worker.queues,
            mem,
            &self.irq_evt,
            &self.interrupt_status,
            &self.device_status,
        );
    }
}
