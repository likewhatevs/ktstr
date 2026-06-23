//! Virtio-net cumulative event counters and their serializable
//! snapshot, split from device.rs for module locality. Self-contained:
//! the counter fields are AtomicU64 and the snapshot derives serde; the
//! MAX_FRAME_SIZE / VIRTIO_NET_HDR_LEN names below appear only in doc
//! comments referring to device.rs constants.
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Counters (host-side observability)
// ---------------------------------------------------------------------------

/// Per-device counters surfaced to the host monitor. All atomic so
/// the monitor can read them without locking the device struct.
///
/// Mirrors the [`super::super::virtio_blk::VirtioBlkCounters`] pattern:
/// `record_*` helper methods enforce field-pairing invariants, and
/// per-field `pub fn` accessors perform `Relaxed` loads. Counters are
/// cumulative for the device's lifetime — `VirtioNet::reset()` does
/// NOT zero them, so an operator monitoring `tx_packets` etc. observes
/// a monotonically non-decreasing series across guest re-binds.
///
/// # Counter taxonomy
///
/// All counters here are **per-event cumulative**. There are no
/// per-request live gauges in v0 — the loopback path is synchronous
/// (no deferred RX, no throttle) so there is no "currently waiting"
/// state to gauge. A future async backend (TAP, AF_PACKET) would add
/// a `currently_deferred_rx_gauge` mirroring virtio-blk's
/// `currently_throttled_gauge`.
#[derive(Debug, Default)]
pub struct VirtioNetCounters {
    /// Cumulative count of TX chains the device accepted from the
    /// guest, parsed cleanly, AND successfully marked used (TX-side
    /// `add_used` returned Ok). A TX chain rejected for malformed
    /// shape (short header, wrong direction) bumps `tx_chain_invalid`
    /// only. A parsed TX chain whose `add_used` then fails bumps
    /// `tx_add_used_failures` only. So `tx_packets` reflects chains
    /// the guest can actually observe as completed.
    ///
    /// Each TX chain the device accepts lands in exactly one
    /// observable outcome: successful loopback delivery (bumps
    /// rx_packets); dropped because the RX queue had no buffer
    /// (bumps tx_dropped_no_rx_buffer); RX chain-shape rejection
    /// during loopback (bumps rx_chain_invalid); RX guest-memory
    /// `write_slice` failure during loopback (bumps
    /// rx_write_failed — chain shape was fine but the
    /// descriptor's GPA was unmapped); RX `add_used` failure
    /// (bumps rx_add_used_failures); TX-side chain-shape
    /// rejection at parse time (bumps tx_chain_invalid, no TX
    /// add_used attempted); or TX `add_used` failure (bumps
    /// tx_add_used_failures). `tx_packets` reflects only the
    /// chains where the TX-side add_used actually succeeded;
    /// `tx_packets - rx_packets` is NOT a generic shortfall
    /// formula because chains lost on the TX side
    /// (tx_chain_invalid, tx_add_used_failures) never bumped
    /// tx_packets in the first place.
    pub(crate) tx_packets: AtomicU64,
    /// Cumulative bytes of L2 frame data accepted from successfully
    /// completed TX chains (i.e. those that bumped `tx_packets`).
    /// Excludes the 12-byte virtio header. Paired with `tx_packets`
    /// via [`Self::record_tx_completed`].
    pub(crate) tx_bytes: AtomicU64,
    /// Cumulative count of RX chains the device successfully wrote
    /// (header + frame) AND successfully marked used (`add_used`
    /// returned Ok AND the used-ring index advanced). RX chains
    /// where `add_used` failed bump `rx_add_used_failures` only —
    /// the guest never observes the publish, so it would be wrong
    /// to count it as a delivery.
    /// Paired with `rx_bytes` via [`Self::record_rx_delivered`].
    pub(crate) rx_packets: AtomicU64,
    /// Cumulative bytes of L2 frame data successfully delivered to
    /// the guest's RX chains (i.e. paired with `rx_packets`).
    /// Excludes the 12-byte virtio header. On a chain whose RX
    /// buffer was smaller than `header + frame`, this counter
    /// reflects the actual bytes written into the descriptor minus
    /// the header — NOT the source `frame_len`. An operator sees
    /// the real bytes the guest can read, not the bytes the device
    /// intended to deliver.
    pub(crate) rx_bytes: AtomicU64,
    /// Cumulative count of successfully-captured TX frames the
    /// device could not deliver to RX because the RX queue was
    /// empty. Per-event counter; a guest that never posts RX buffers
    /// and floods TX produces one bump per dropped TX frame. The TX
    /// chain is still marked used (the guest sees TX completion via
    /// `tx_packets`); the frame never arrives on RX (no `rx_packets`
    /// bump). Distinct from `tx_chain_invalid` (TX chain shape
    /// rejected before any RX delivery was attempted).
    pub(crate) tx_dropped_no_rx_buffer: AtomicU64,
    /// Cumulative count of successfully-captured TX frames the device
    /// dropped because the RX queue was POISONED — a prior structural
    /// `Error::InvalidAvailRingIndex` left `queue_poisoned[RXQ]` set.
    /// Per-event counter. Like `tx_dropped_no_rx_buffer`, the TX chain
    /// is still marked used (the guest sees TX completion via
    /// `tx_packets`); the frame never reaches RX (no `rx_packets`
    /// bump). Distinct from `tx_dropped_no_rx_buffer` so an operator
    /// can tell "RX queue was simply empty (transient back-pressure)"
    /// from "RX queue is wedged on a guest avail-ring violation" —
    /// the latter does not clear until the guest issues a virtio
    /// reset. Bumped on BOTH RX-poison loopback outcomes: the queue
    /// was just poisoned this drain (`JustRxPoisoned`) or was already
    /// poisoned from a prior kick (`RxAlreadyPoisoned`).
    pub(crate) tx_dropped_rx_poisoned: AtomicU64,
    /// Cumulative count of TX chains rejected for malformed shape:
    /// missing header, write-only descriptor in TX (TX descriptors
    /// must be device-readable), header-read failure. The TX chain
    /// is still marked used so the guest doesn't hang on the
    /// request, but the frame is dropped without an RX delivery and
    /// neither `tx_packets` nor `rx_packets` is bumped. Per-event
    /// counter.
    pub(crate) tx_chain_invalid: AtomicU64,
    /// Cumulative count of TX chains DROPPED because the captured
    /// post-header frame data exceeded `MAX_FRAME_SIZE` (the largest
    /// L2 frame the guest's `max_mtu` permits). The TX chain is still
    /// marked used so the guest doesn't hang, but the frame is dropped
    /// — NOT truncated — and neither `tx_packets` nor `rx_packets` is
    /// bumped (mutually exclusive with `tx_packets` per chain, like
    /// `tx_chain_invalid`). Per-event counter. Distinct from
    /// `tx_chain_invalid`: the chain shape was well-formed (readable
    /// descriptors, full 12-byte header) — it was simply too large.
    /// Silently truncating an over-size frame would corrupt traffic
    /// the guest believes it transmitted intact, so the device drops
    /// the whole frame and surfaces the event here. A non-zero value
    /// means the guest emitted a frame larger than its advertised
    /// `max_mtu` allows — a guest bug or a hostile descriptor chain.
    pub(crate) tx_oversize_dropped: AtomicU64,
    /// Cumulative count of RX chains rejected for malformed shape on
    /// the loopback delivery side: read-only descriptor in RX (RX
    /// descriptors must be device-writable) or attacker-controlled
    /// `desc.addr() + take` overflow (the descriptor's address itself
    /// is malformed). The RX chain is still marked used (with
    /// `len = 0`) so the guest's network-stack equivalent of a
    /// hung-task watchdog doesn't fire on a stuck request.
    /// Per-event counter; bumped exactly once per chain rejected for
    /// shape (the `tx_dropped_no_rx_buffer` counter is NOT also
    /// bumped — they are mutually exclusive failure modes, see
    /// [`Self::record_rx_chain_invalid`]).
    ///
    /// **Distinct from [`Self::rx_write_failed`]**: a guest-memory
    /// `write_slice` failure (header or frame bytes) means the
    /// chain's SHAPE was acceptable but the GPA targeted by a
    /// device-writable descriptor isn't mapped — that bumps
    /// `rx_write_failed`, NOT this counter. Operators
    /// distinguishing "guest sent malformed RX chain" from "guest's
    /// posted RX buffer points at unmapped memory" need the two
    /// counters separated.
    pub(crate) rx_chain_invalid: AtomicU64,
    /// Cumulative count of RX chains where the chain shape was valid
    /// (every descriptor was device-writable, addresses didn't
    /// overflow) but a guest-memory `write_slice` to one of the
    /// descriptors failed — typically because the descriptor's GPA
    /// is unmapped. Either the 12-byte header `write_slice` or the
    /// frame-data `write_slice` can fail; both bump this counter.
    /// The RX chain is still marked used (with `len = 0`) so the
    /// guest doesn't hang on the request. Per-event counter;
    /// bumped exactly once per chain whose write actually failed
    /// (chain-shape rejections route to `rx_chain_invalid`
    /// instead — the two counters are mutually exclusive per
    /// chain).
    ///
    /// Distinct from `rx_chain_invalid` so an operator's failure
    /// dump can separate "guest violated the RX descriptor-direction
    /// rule" from "guest posted a buffer at an unmapped GPA". A
    /// non-zero `rx_write_failed` with `rx_chain_invalid == 0`
    /// points at GPA / page-table breakage rather than driver-side
    /// malformation; the inverse points at driver-side direction
    /// violations or address-overflow attacks.
    pub(crate) rx_write_failed: AtomicU64,
    /// Cumulative count of `add_used` failures on the TX queue. A
    /// non-zero value means the queue's used-ring address is
    /// unmapped or otherwise inaccessible — distinct from a chain-
    /// shape rejection (which uses `tx_chain_invalid`). Per-event
    /// counter. Operators monitoring `tx_add_used_failures > 0`
    /// know the queue itself is broken and the guest has not seen
    /// any TX completion since the failure started; the typical
    /// recovery path is a virtio reset (write `STATUS=0`). Distinct
    /// from `tx_chain_invalid` so an operator can tell "guest sent
    /// malformed frame" from "queue itself is broken".
    pub(crate) tx_add_used_failures: AtomicU64,
    /// Cumulative count of `add_used` failures on the RX queue. As
    /// with `tx_add_used_failures`, indicates a queue-state failure
    /// (used-ring unmapped) distinct from chain-shape rejection.
    /// Bumped on the RX side from both the malformed-chain branch
    /// and the successful-frame-write branch when the trailing
    /// `add_used` fails — both branches mean the device tried to
    /// publish a used-ring entry and the publish itself failed.
    pub(crate) rx_add_used_failures: AtomicU64,
    /// Cumulative count of `Error::InvalidAvailRingIndex` events
    /// observed across both queues. Bumped each time the
    /// virtio-queue iter() rejects an avail.idx whose distance from
    /// `next_avail` exceeds the queue size — a hostile or buggy
    /// guest condition.
    ///
    /// Per-event counter (NOT per-request): the per-queue poison
    /// flag short-circuits further attempts on the same queue, so
    /// the false→true transition produces exactly one bump per
    /// poison event. Without the flag, every QUEUE_NOTIFY kick
    /// would re-enter `iter()`, observe the same error, log via
    /// `error!()`, return None from the swallowing default impl,
    /// and re-bump this counter — three concrete problems:
    /// (a) the per-event counter taxonomy is violated (counter
    /// reflects kick rate rather than poison event rate),
    /// (b) the operator has no signal that the device is wedged
    /// (no NEEDS_RESET, no STATUS bit change), and (c) every kick
    /// floods the host log with the same error line. The poison
    /// flag fixes all three. Note: this is NOT a "livelock" —
    /// virtio-net has no enable_notification/disable_notification
    /// bracket, so each kick re-trips the error ONCE per MMIO
    /// exit, then returns. The harm is observability + log spam,
    /// not unbounded CPU consumption.
    ///
    /// Successive QUEUE_NOTIFY kicks against an unresetted
    /// poisoned queue take the entry-gate short-circuit and
    /// produce zero additional bumps until the guest performs a
    /// virtio reset.
    pub(crate) invalid_avail_idx_count: AtomicU64,
}

impl VirtioNetCounters {
    /// Record TX-side completion: a parsed TX chain whose
    /// `add_used` returned Ok. Bumps `tx_packets` + `tx_bytes`.
    /// MUST be called AFTER the TX `add_used` succeeds — calling
    /// it before would let the counter lie if the publish fails
    /// (the guest would never observe the completion).
    pub(crate) fn record_tx_completed(&self, frame_bytes: u64) {
        self.tx_packets.fetch_add(1, Ordering::Relaxed);
        self.tx_bytes.fetch_add(frame_bytes, Ordering::Relaxed);
    }

    /// Record successful RX delivery (frame written to a guest
    /// descriptor chain, `add_used` returned Ok). Bumps
    /// `rx_packets` + `rx_bytes`. MUST be called AFTER the RX
    /// `add_used` succeeds — if the publish fails, the guest never
    /// observes the frame and the counter would lie. The byte count
    /// is the actual L2 bytes written into the descriptor (i.e.
    /// `bytes_written - VIRTIO_NET_HDR_LEN`), which differs from
    /// the source `frame_len` when the guest's RX buffer was
    /// smaller than `header + frame`.
    pub(crate) fn record_rx_delivered(&self, frame_bytes: u64) {
        self.rx_packets.fetch_add(1, Ordering::Relaxed);
        self.rx_bytes.fetch_add(frame_bytes, Ordering::Relaxed);
    }

    /// Record one TX chain dropped because the RX queue is empty
    /// (the TX-side already completed via [`Self::record_tx_completed`];
    /// this counter records the RX-delivery failure).
    pub(crate) fn record_tx_dropped_no_rx_buffer(&self) {
        self.tx_dropped_no_rx_buffer.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one successfully-captured TX frame dropped because the
    /// RX queue is poisoned (the TX-side still completes via
    /// [`Self::record_tx_completed`] when its `add_used` succeeds;
    /// this counter records the RX-delivery failure). Distinct from
    /// [`Self::record_tx_dropped_no_rx_buffer`] (empty RX queue,
    /// transient) — a poisoned RX queue stays wedged until the guest
    /// resets the device.
    pub(crate) fn record_tx_dropped_rx_poisoned(&self) {
        self.tx_dropped_rx_poisoned.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one TX chain rejected for malformed shape (short
    /// header, wrong direction, header-read failure). The TX chain
    /// is marked used but neither `tx_packets` nor `rx_packets` is
    /// bumped — this is the protocol-violation path.
    pub(crate) fn record_tx_chain_invalid(&self) {
        self.tx_chain_invalid.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one TX chain dropped because its post-header frame data
    /// exceeded `MAX_FRAME_SIZE`. The TX chain is marked used but
    /// neither `tx_packets` nor `rx_packets` is bumped — the frame is
    /// dropped, not truncated. Distinct from
    /// [`Self::record_tx_chain_invalid`]: the chain shape was valid,
    /// it was simply over the maximum frame size the guest's `max_mtu`
    /// permits.
    pub(crate) fn record_tx_oversize_dropped(&self) {
        self.tx_oversize_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one RX chain rejected for malformed shape on the
    /// loopback delivery side (read-only descriptor or
    /// address-overflow on the descriptor's GPA). Mutually exclusive
    /// with [`Self::record_tx_dropped_no_rx_buffer`]: a chain is
    /// either missing entirely (queue empty →
    /// `tx_dropped_no_rx_buffer`) or present but shape-malformed
    /// (this counter). Mutually exclusive PER CHAIN with
    /// [`Self::record_rx_write_failed`]: a chain is either
    /// shape-rejected (this counter) or write-rejected
    /// (`rx_write_failed`); the caller routes each malformed RX
    /// chain to exactly one of the two so the per-event counter
    /// taxonomy stays 1:1 with chains.
    pub(crate) fn record_rx_chain_invalid(&self) {
        self.rx_chain_invalid.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one RX chain whose shape was valid (every descriptor
    /// was device-writable, no address overflow) but whose guest-
    /// memory `write_slice` failed mid-walk — header or frame
    /// bytes hit an unmapped GPA. Mutually exclusive PER CHAIN with
    /// [`Self::record_rx_chain_invalid`]: a chain rejected for
    /// shape NEVER also bumps this counter, and vice versa. The
    /// caller routes via the module-scope `InvalidReason` enum (set
    /// in `write_rx_chain`, routed in `finalize_rx`).
    pub(crate) fn record_rx_write_failed(&self) {
        self.rx_write_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one `add_used` failure on the TX queue. Distinct from
    /// `record_tx_chain_invalid` so operators can tell queue-state
    /// breakage from chain-shape rejection.
    pub(crate) fn record_tx_add_used_failure(&self) {
        self.tx_add_used_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one `add_used` failure on the RX queue. Distinct from
    /// `record_rx_chain_invalid` so operators can tell queue-state
    /// breakage from chain-shape rejection.
    pub(crate) fn record_rx_add_used_failure(&self) {
        self.rx_add_used_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one observed `Error::InvalidAvailRingIndex` event
    /// from `Queue::iter`. Called by `process_tx_loopback` /
    /// `pop_rx_chain` (the RX-pull phase of `try_loopback_to_rx`)
    /// when the avail ring's `idx` is more than `queue.size` ahead
    /// of `next_avail` — a virtio-spec
    /// violation by the guest. The caller also sets
    /// `VirtioNet::queue_poisoned` so a single hostile-guest event
    /// produces exactly one bump regardless of how many subsequent
    /// kicks land before the next reset (subsequent drains
    /// short-circuit on the poison flag and never re-call `iter`).
    pub(crate) fn record_invalid_avail_idx(&self) {
        self.invalid_avail_idx_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Read the cumulative count of TX chains successfully looped to
    /// RX. Per-event counter: bumped exactly once per TX chain that
    /// completed both halves of the loopback.
    pub fn tx_packets(&self) -> u64 {
        self.tx_packets.load(Ordering::Relaxed)
    }

    /// Read the cumulative bytes of L2 frame data successfully looped
    /// to RX. Excludes the 12-byte virtio header.
    pub fn tx_bytes(&self) -> u64 {
        self.tx_bytes.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX chains delivered to the guest.
    /// Equal to `tx_packets()` in v0's pure-loopback mode.
    pub fn rx_packets(&self) -> u64 {
        self.rx_packets.load(Ordering::Relaxed)
    }

    /// Read the cumulative bytes of L2 frame data delivered to the
    /// guest's RX chains. Excludes the 12-byte virtio header.
    pub fn rx_bytes(&self) -> u64 {
        self.rx_bytes.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX chains dropped because the RX
    /// queue had no buffer.
    pub fn tx_dropped_no_rx_buffer(&self) -> u64 {
        self.tx_dropped_no_rx_buffer.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of successfully-captured TX frames
    /// dropped because the RX queue was poisoned. Distinct from
    /// [`Self::tx_dropped_no_rx_buffer`] (empty RX queue): a poisoned
    /// RX queue is wedged until a guest virtio reset.
    pub fn tx_dropped_rx_poisoned(&self) -> u64 {
        self.tx_dropped_rx_poisoned.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX chains rejected for malformed
    /// shape (missing/short header, wrong direction, header read
    /// failure).
    pub fn tx_chain_invalid(&self) -> u64 {
        self.tx_chain_invalid.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX chains dropped for exceeding
    /// `MAX_FRAME_SIZE`. Distinct from [`Self::tx_chain_invalid`]
    /// (malformed shape): an over-size chain was well-formed but too
    /// large, and is dropped rather than truncated.
    pub fn tx_oversize_dropped(&self) -> u64 {
        self.tx_oversize_dropped.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX chains rejected for malformed
    /// shape (read-only descriptor on the receive side, or
    /// attacker-controlled address overflow on the descriptor's
    /// GPA). Distinct from [`Self::rx_write_failed`] (chain shape
    /// was fine but a guest-memory `write_slice` hit an unmapped
    /// GPA mid-walk).
    pub fn rx_chain_invalid(&self) -> u64 {
        self.rx_chain_invalid.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX chains whose shape was valid
    /// but whose guest-memory `write_slice` failed mid-walk
    /// (header or frame bytes hit an unmapped GPA). Distinct from
    /// [`Self::rx_chain_invalid`] (chain-shape rejection); the two
    /// are mutually exclusive per chain so an operator's failure
    /// dump can separate "guest violated the RX descriptor-direction
    /// rule" from "guest posted a buffer at an unmapped GPA".
    pub fn rx_write_failed(&self) -> u64 {
        self.rx_write_failed.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of TX `add_used` failures (queue's
    /// used-ring address unmapped or otherwise inaccessible).
    /// Non-zero means the TX queue itself is structurally broken;
    /// distinct from `tx_chain_invalid` (chain-shape rejection).
    pub fn tx_add_used_failures(&self) -> u64 {
        self.tx_add_used_failures.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of RX `add_used` failures.
    /// Non-zero means the RX queue itself is structurally broken;
    /// distinct from `rx_chain_invalid` (chain-shape rejection).
    pub fn rx_add_used_failures(&self) -> u64 {
        self.rx_add_used_failures.load(Ordering::Relaxed)
    }

    /// Read the cumulative count of `Error::InvalidAvailRingIndex`
    /// events the device has observed. Per-event counter (NOT
    /// per-request): the queue-poison flag short-circuits subsequent
    /// kicks against the same hostile state, so one guest fault
    /// produces exactly one bump regardless of how many notifications
    /// follow before reset. A non-zero value means the guest violated
    /// virtio-v1.2 §2.7.13.3 — the device is in the "structurally
    /// broken queue" state and will not service IO until the guest
    /// issues a virtio reset.
    pub fn invalid_avail_idx_count(&self) -> u64 {
        self.invalid_avail_idx_count.load(Ordering::Relaxed)
    }

    /// Freeze every atomic into a plain-u64 snapshot for the
    /// host-side post-mortem path in [`crate::vmm::VmResult`].
    /// virtio-net is single-threaded — `process_tx_loopback` runs
    /// inline on the kicking vCPU thread, so the sole writers to
    /// these counters are the vCPUs themselves. By the time
    /// `collect_results` reaches the snapshot site every vCPU
    /// thread has joined, so no writer remains and the relaxed
    /// loads observe the final cumulative state.
    pub fn snapshot(&self) -> VirtioNetCountersSnapshot {
        VirtioNetCountersSnapshot {
            tx_packets: self.tx_packets(),
            tx_bytes: self.tx_bytes(),
            rx_packets: self.rx_packets(),
            rx_bytes: self.rx_bytes(),
            tx_dropped_no_rx_buffer: self.tx_dropped_no_rx_buffer(),
            tx_dropped_rx_poisoned: self.tx_dropped_rx_poisoned(),
            tx_chain_invalid: self.tx_chain_invalid(),
            tx_oversize_dropped: self.tx_oversize_dropped(),
            rx_chain_invalid: self.rx_chain_invalid(),
            rx_write_failed: self.rx_write_failed(),
            tx_add_used_failures: self.tx_add_used_failures(),
            rx_add_used_failures: self.rx_add_used_failures(),
            invalid_avail_idx_count: self.invalid_avail_idx_count(),
        }
    }
}

/// Plain-u64 snapshot of `VirtioNetCounters` taken at VM-result
/// construction time. Mirrors every atomic field by name.
///
/// Decouples [`crate::vmm::VmResult`] from the internal
/// atomic-shared writer state — consumers see immutable owned
/// data they can `Clone`, compare, and round-trip through serde
/// without the `Arc<AtomicU64>` ceremony. virtio-net is
/// single-threaded — the vCPU thread continues to bump the
/// atomics inline from `process_tx_loopback` via the
/// `VirtioNetCounters` `record_*` mutators; only the
/// result-construction path moves to the snapshot.
///
/// Field semantics match the atomic source one-for-one — see
/// `VirtioNetCounters` for the per-counter taxonomy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VirtioNetCountersSnapshot {
    pub tx_packets: u64,
    pub tx_bytes: u64,
    pub rx_packets: u64,
    pub rx_bytes: u64,
    pub tx_dropped_no_rx_buffer: u64,
    pub tx_dropped_rx_poisoned: u64,
    pub tx_chain_invalid: u64,
    pub tx_oversize_dropped: u64,
    pub rx_chain_invalid: u64,
    pub rx_write_failed: u64,
    pub tx_add_used_failures: u64,
    pub rx_add_used_failures: u64,
    pub invalid_avail_idx_count: u64,
}
