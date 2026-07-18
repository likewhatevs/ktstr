//! Unit coverage for the TOKEN_TX dispatch's CRC-gated promotion
//! and decode paths.
//!
//! Three production gates inside [`dispatch_bulk_message`] inspect each
//! `BulkMessage` the streaming [`crate::vmm::bulk::HostAssembler`]
//! yields from the virtio-console port-1 TX byte stream:
//!
//!   * `MSG_TYPE_SCHED_EXIT && crc_ok` — flips the run-wide kill flag
//!     and writes the kill eventfd so the BSP loop and the watchdog
//!     exit promptly. CRC failures must NOT promote — a torn frame
//!     would otherwise let a hostile guest force a false early exit.
//!   * `MSG_TYPE_SNAPSHOT_REQUEST && crc_ok &&
//!     decode_snapshot_request(payload).is_some()` — pushes the
//!     decoded request onto the per-iteration pending list for
//!     dispatch to `freeze_and_dispatch` / `arm_user_watchpoint`.
//!     CRC failures must NOT decode — a torn snapshot request would
//!     otherwise let a hostile guest force a spurious capture or
//!     watchpoint arm.
//!   * `MSG_TYPE_SYS_RDY && crc_ok && payload.is_empty()` — fires the
//!     boot-complete eventfd exactly once (`Option::take`). CRC
//!     failures must NOT fire — a torn frame would race ahead of
//!     percpu/KASLR setup.
//!
//! These gates are exercised through the production
//! [`dispatch_bulk_message`] (a `pub(super)` fn reachable from this
//! freeze-coord child module) driving a real [`BulkDispatchSinks`]:
//! build a CRC-mangled TLV byte stream, run it through the same
//! `HostAssembler::feed` the closure uses, dispatch each resulting
//! `BulkMessage`, then assert on the sinks (kill flag / kill_evt
//! counter / sys_rdy take + counter / snapshot-pending len) and the
//! returned `Option<ShmEntry>`. A regression that drops a `crc_ok`
//! clause in the real dispatch fails here.
use super::dispatch::{BulkDispatchSinks, dispatch_bulk_message};
use super::state::SnapshotRequest;
use crate::vmm::bulk::HostAssembler;
use crate::vmm::wire::{
    FRAME_HEADER_SIZE, MSG_TYPE_SCHED_EXIT, MSG_TYPE_SNAPSHOT_REQUEST, MSG_TYPE_SYS_RDY,
    SNAPSHOT_KIND_CAPTURE, SNAPSHOT_TAG_MAX, ShmEntry, ShmMessage, SnapshotRequestPayload,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64};
use std::time::Instant;
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
use zerocopy::IntoBytes;

/// Owns every backing value a [`BulkDispatchSinks`] borrows so a test
/// can drive the production [`dispatch_bulk_message`] and then read
/// the post-dispatch state out of the same owned values. Mirrors the
/// freeze-coordinator closure scope: the kill flag + eventfd, the
/// SysRdy one-shot handle, the snapshot/kernel-op pending vecs, the
/// KASLR slots, and the watchdog/scenario atoms (all zero/`None`
/// initialised — the CRC gates under test never touch them).
struct SinkState {
    kill: Arc<AtomicBool>,
    kill_evt: Arc<EventFd>,
    run_is_wprof: bool,
    sys_rdy_evt: Option<Arc<EventFd>>,
    snapshot_requests_pending: Vec<SnapshotRequest>,
    kernel_op_requests_pending: Vec<crate::vmm::wire::KernelOpRequestPayload>,
    kern_phys_base: Arc<AtomicU64>,
    kern_phys_base_evt: EventFd,
    kern_virt_kaslr: Arc<AtomicU64>,
    kern_virt_kaslr_evt: EventFd,
    kern_addrs_frames: AtomicU64,
    kern_addrs_crc_bad: AtomicU64,
    unknown_type_frames: AtomicU64,
    watchdog_pause_ns: AtomicU64,
    scenario_start_ns: AtomicU64,
    sched_swap_notify: AtomicBool,
    scenario_pause_cumulative_ns: AtomicU64,
    run_start: Instant,
    current_step: Arc<AtomicU16>,
    periodic_guest_elapsed_ns: AtomicU64,
}

impl SinkState {
    fn new() -> Self {
        Self {
            kill: Arc::new(AtomicBool::new(false)),
            kill_evt: Arc::new(EventFd::new(EFD_NONBLOCK).expect("kill eventfd")),
            run_is_wprof: false,
            sys_rdy_evt: Some(Arc::new(
                EventFd::new(EFD_NONBLOCK).expect("sys_rdy eventfd"),
            )),
            snapshot_requests_pending: Vec::new(),
            kernel_op_requests_pending: Vec::new(),
            kern_phys_base: Arc::new(AtomicU64::new(0)),
            kern_phys_base_evt: EventFd::new(EFD_NONBLOCK).expect("phys_base eventfd"),
            kern_virt_kaslr: Arc::new(AtomicU64::new(0)),
            kern_virt_kaslr_evt: EventFd::new(EFD_NONBLOCK).expect("virt_kaslr eventfd"),
            kern_addrs_frames: AtomicU64::new(0),
            kern_addrs_crc_bad: AtomicU64::new(0),
            unknown_type_frames: AtomicU64::new(0),
            watchdog_pause_ns: AtomicU64::new(0),
            scenario_start_ns: AtomicU64::new(0),
            sched_swap_notify: AtomicBool::new(false),
            scenario_pause_cumulative_ns: AtomicU64::new(0),
            run_start: Instant::now(),
            current_step: Arc::new(AtomicU16::new(0)),
            periodic_guest_elapsed_ns: AtomicU64::new(0),
        }
    }

    fn sinks(&mut self) -> BulkDispatchSinks<'_> {
        BulkDispatchSinks {
            kill: &self.kill,
            kill_evt: &self.kill_evt,
            run_is_wprof: self.run_is_wprof,
            sys_rdy_evt: &mut self.sys_rdy_evt,
            bpf_map_write_ready_evt: None,
            snapshot_requests_pending: &mut self.snapshot_requests_pending,
            kernel_op_requests_pending: &mut self.kernel_op_requests_pending,
            kern_phys_base: &self.kern_phys_base,
            kern_phys_base_evt: &self.kern_phys_base_evt,
            kern_virt_kaslr: &self.kern_virt_kaslr,
            kern_virt_kaslr_evt: &self.kern_virt_kaslr_evt,
            kernel_text_link_kva: 0,
            kern_addrs_frames: &self.kern_addrs_frames,
            kern_addrs_crc_bad: &self.kern_addrs_crc_bad,
            unknown_type_frames: &self.unknown_type_frames,
            watchdog_reset: None,
            watchdog_pause_ns: &self.watchdog_pause_ns,
            scenario_start_ns: &self.scenario_start_ns,
            sched_swap_notify: &self.sched_swap_notify,
            scenario_pause_cumulative_ns: &self.scenario_pause_cumulative_ns,
            run_start: self.run_start,
            current_step: &self.current_step,
            periodic_guest_elapsed_ns: &self.periodic_guest_elapsed_ns,
            // No ledger wired: these CRC-gate fixtures do not assert
            // lifecycle-stage side effects, so every advance site no-ops.
            progress_ledger: None,
            contention_recorder: None,
            attach_attempts: None,
            attach_control_console: None,
            // No expected build-id: the KERN_BUILD_ID arm's check is
            // disabled (a `None` expectation never flags a mismatch).
            expected_kernel_build_id: None,
        }
    }
}

/// Dispatch every message through the production
/// [`dispatch_bulk_message`], collecting the verdict-bearing entries
/// into a bucket — exactly the freeze-coordinator's per-iteration
/// loop. Returns `(state, bucket)` so a test can read the sink
/// post-state and inspect what bucketed.
fn run_dispatch(messages: &[crate::vmm::bulk::BulkMessage]) -> (SinkState, Vec<ShmEntry>) {
    let mut state = SinkState::new();
    let mut bucket = Vec::new();
    {
        let mut sinks = state.sinks();
        for msg in messages {
            if let Some(entry) = dispatch_bulk_message(msg, &mut sinks) {
                bucket.push(entry);
            }
        }
    }
    (state, bucket)
}

/// Read an EFD_NONBLOCK eventfd's accumulated counter (0 if EAGAIN).
fn read_counter(evt: &EventFd) -> u64 {
    evt.read().unwrap_or(0)
}

/// Build a TLV frame whose header CRC matches the supplied payload
/// — `HostAssembler::feed` will produce a `BulkMessage` with
/// `crc_ok=true`. Mirrors `bulk.rs`'s test helper so the
/// assertions below pin the production behaviour the closure
/// observes, not a synthetic in-test path.
fn frame_with_crc(msg_type: u32, payload: &[u8]) -> Vec<u8> {
    let header = ShmMessage {
        msg_type,
        length: payload.len() as u32,
        crc32: crc32fast::hash(payload),
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Build a TLV frame whose header CRC deliberately does NOT match
/// the payload — `HostAssembler::feed` produces a `BulkMessage`
/// with `crc_ok=false`, exactly as it would on a torn or
/// hostile-guest publish. The bogus CRC is the recomputed CRC
/// XOR'd with `0xFFFF_FFFF` so the mismatch is total (every
/// bit flipped) rather than a near-miss that could match if the
/// payload were hashed differently.
fn frame_with_torn_crc(msg_type: u32, payload: &[u8]) -> Vec<u8> {
    let real_crc = crc32fast::hash(payload);
    let header = ShmMessage {
        msg_type,
        length: payload.len() as u32,
        crc32: real_crc ^ 0xFFFF_FFFF,
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Build a SnapshotRequest payload using the wire-layout type so
/// the bytes match what a guest publisher emits. Identical to
/// `make_request_bytes` in `snapshot_tlv_tests` — duplicated here
/// so this module remains self-contained.
fn snapshot_request_bytes(request_id: u32, kind: u32, tag: &str) -> Vec<u8> {
    let tag_bytes = tag.as_bytes();
    let mut tag_buf = [0u8; SNAPSHOT_TAG_MAX];
    let n = tag_bytes.len().min(SNAPSHOT_TAG_MAX);
    tag_buf[..n].copy_from_slice(&tag_bytes[..n]);
    SnapshotRequestPayload {
        request_id,
        kind,
        tag: tag_buf,
    }
    .as_bytes()
    .to_vec()
}

/// CRC-failed SCHED_EXIT MUST NOT promote the run-wide kill flag,
/// write the kill eventfd, or bucket a verdict entry. A torn or
/// hostile-guest frame would otherwise let an attacker force the BSP
/// loop and the watchdog to exit early, ending a test before its
/// scheduler under test had a chance to misbehave.
#[test]
fn sched_exit_with_torn_crc_does_not_promote_kill() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1, "assembler emits one message");
    assert!(
        !drained.messages[0].crc_ok,
        "torn CRC must surface as crc_ok=false"
    );
    assert_eq!(
        drained.messages[0].msg_type, MSG_TYPE_SCHED_EXIT,
        "msg_type unaffected by CRC mismatch — gate dispatch is by type"
    );
    let (state, bucket) = run_dispatch(&drained.messages);
    assert!(
        !state.kill.load(std::sync::atomic::Ordering::Acquire),
        "kill flag must NOT flip on CRC-failed SCHED_EXIT — \
         hostile guest must not force early exit"
    );
    assert_eq!(
        read_counter(&state.kill_evt),
        0,
        "kill eventfd must NOT be written on CRC-failed SCHED_EXIT — \
         the BSP loop and watchdog must not be woken"
    );
    assert!(
        bucket.is_empty(),
        "CRC-failed SCHED_EXIT must NOT surface as a verdict entry"
    );
}

/// Positive control: a CRC-valid SCHED_EXIT DOES promote and bucket.
/// Pins the test against a degenerate case where the gate is broken
/// and the negative test passes for the wrong reason (kill never
/// promotes regardless of input). Without this control, a fix that
/// accidentally inverts the predicate (`!msg.crc_ok` instead of
/// `msg.crc_ok`) would still pass the torn-CRC test but break
/// production.
#[test]
fn sched_exit_with_valid_crc_does_promote_kill() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        drained.messages[0].crc_ok,
        "matching CRC must surface as crc_ok=true"
    );
    let (state, bucket) = run_dispatch(&drained.messages);
    assert!(
        state.kill.load(std::sync::atomic::Ordering::Acquire),
        "kill flag MUST flip on CRC-valid SCHED_EXIT — promotion is \
         the load-bearing path that ends a test promptly"
    );
    assert_eq!(
        read_counter(&state.kill_evt),
        1,
        "kill eventfd MUST be written once on CRC-valid SCHED_EXIT — \
         the BSP loop and watchdog need an epoll wake to exit"
    );
    assert_eq!(
        bucket.len(),
        1,
        "CRC-valid SCHED_EXIT must bucket exactly once"
    );
    assert_eq!(bucket[0].msg_type, MSG_TYPE_SCHED_EXIT);
    assert_eq!(bucket[0].payload, b"exit-payload"[..]);
    assert!(bucket[0].crc_ok);
}

/// On a wprof run (`run_is_wprof=true`) a CRC-valid SCHED_EXIT must NOT
/// promote the run-wide kill flag or write the kill eventfd — on a
/// self-crash the still-live guest sched-exit monitor sends SCHED_EXIT
/// during Phase 5, and promoting the kill here would tear the coordinator
/// down before the guest ships its Phase-5 wprof trace. Gating on the RUN
/// being wprof (not on a grace-deadline snapshot) closes the arming-order
/// race. The verdict entry is still bucketed; the coord arms the ship
/// grace on this SCHED_EXIT and terminates on the WprofTrace frame or the
/// `WPROF_SHIP_GRACE` deadline backstop.
#[test]
fn sched_exit_on_wprof_run_defers_kill_but_still_buckets() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(drained.messages[0].crc_ok);

    // Drive dispatch on a wprof run (run_is_wprof=true).
    let mut state = SinkState::new();
    state.run_is_wprof = true;
    let mut bucket = Vec::new();
    {
        let mut sinks = state.sinks();
        for msg in &drained.messages {
            if let Some(entry) = dispatch_bulk_message(msg, &mut sinks) {
                bucket.push(entry);
            }
        }
    }

    assert!(
        !state.kill.load(std::sync::atomic::Ordering::Acquire),
        "kill flag must NOT flip on SCHED_EXIT on a wprof run — the \
         coordinator holds the VM open for the guest's Phase-5 wprof ship \
         instead of killing"
    );
    assert_eq!(
        read_counter(&state.kill_evt),
        0,
        "kill eventfd must NOT be written on SCHED_EXIT on a wprof run"
    );
    assert_eq!(
        bucket.len(),
        1,
        "SCHED_EXIT is still verdict data — bucket it on a wprof run"
    );
    assert_eq!(bucket[0].msg_type, MSG_TYPE_SCHED_EXIT);
}

/// Mixed batch: a CRC-failed SCHED_EXIT alongside other CRC-valid
/// frames must not promote. The gate is per-message, not per-batch —
/// every CRC failure must short-circuit independently regardless of
/// what arrived alongside it. This catches a regression where the
/// gate erroneously walks the batch and trusts the first valid frame
/// to authorise the rest.
#[test]
fn sched_exit_torn_crc_does_not_promote_when_other_valid_frames_present() {
    let mut a = HostAssembler::new();
    // Build a batch: torn SCHED_EXIT first, then a valid STIMULUS
    // frame (not a SCHED_EXIT — must not promote on its own), then a
    // torn SCHED_EXIT-typed frame.
    let mut buf = Vec::new();
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"first"));
    buf.extend(frame_with_crc(
        crate::vmm::wire::MSG_TYPE_STIMULUS,
        b"valid",
    ));
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"second"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 3);
    assert!(!drained.messages[0].crc_ok);
    assert!(drained.messages[1].crc_ok);
    assert!(!drained.messages[2].crc_ok);
    let (state, bucket) = run_dispatch(&drained.messages);
    assert!(
        !state.kill.load(std::sync::atomic::Ordering::Acquire),
        "neither torn SCHED_EXIT may promote even though a CRC-valid \
         non-SCHED_EXIT frame arrived alongside them"
    );
    assert_eq!(
        read_counter(&state.kill_evt),
        0,
        "kill eventfd must remain undisturbed"
    );
    // The STIMULUS frame still buckets (it is verdict-bearing); the
    // two torn SCHED_EXITs do not.
    assert_eq!(
        bucket.len(),
        1,
        "only the CRC-valid STIMULUS buckets; both torn SCHED_EXITs drop"
    );
    assert_eq!(bucket[0].msg_type, crate::vmm::wire::MSG_TYPE_STIMULUS);
}

/// CRC-failed SNAPSHOT_REQUEST MUST be dropped before
/// `decode_snapshot_request` runs. A torn or hostile-guest snapshot
/// request would otherwise let an attacker force a spurious
/// `freeze_and_dispatch` (host-side stall, dump allocation) or
/// `arm_user_watchpoint` (DR slot consumption, `KVM_SET_GUEST_DEBUG`
/// reprogram) without ever generating a matching CRC.
#[test]
fn snapshot_request_with_torn_crc_is_dropped() {
    let mut a = HostAssembler::new();
    let payload = snapshot_request_bytes(7, SNAPSHOT_KIND_CAPTURE, "snap_dump");
    let bytes = frame_with_torn_crc(MSG_TYPE_SNAPSHOT_REQUEST, &payload);
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1, "assembler emits one message");
    assert!(
        !drained.messages[0].crc_ok,
        "torn CRC must surface as crc_ok=false"
    );
    assert_eq!(
        drained.messages[0].msg_type, MSG_TYPE_SNAPSHOT_REQUEST,
        "msg_type unaffected by CRC mismatch"
    );
    let (state, bucket) = run_dispatch(&drained.messages);
    assert_eq!(
        state.snapshot_requests_pending.len(),
        0,
        "CRC-failed SNAPSHOT_REQUEST must NOT decode — \
         hostile guest must not force a capture or watchpoint arm"
    );
    assert!(
        bucket.is_empty(),
        "SNAPSHOT_REQUEST is coordinator-internal — never buckets"
    );
}

/// Positive control: a CRC-valid SNAPSHOT_REQUEST with a well-formed
/// payload IS pushed onto the pending list with the decoded fields.
/// Same degenerate-pass guard rationale as the SCHED_EXIT positive
/// control above.
#[test]
fn snapshot_request_with_valid_crc_is_pushed() {
    let mut a = HostAssembler::new();
    let payload = snapshot_request_bytes(42, SNAPSHOT_KIND_CAPTURE, "valid_tag");
    let bytes = frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &payload);
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        drained.messages[0].crc_ok,
        "matching CRC must surface as crc_ok=true"
    );
    let (state, bucket) = run_dispatch(&drained.messages);
    assert_eq!(
        state.snapshot_requests_pending.len(),
        1,
        "CRC-valid well-formed SNAPSHOT_REQUEST MUST decode and push"
    );
    assert_eq!(
        state.snapshot_requests_pending[0].request_id, 42,
        "decoded request_id must round-trip through the gate"
    );
    assert!(
        bucket.is_empty(),
        "SNAPSHOT_REQUEST is coordinator-internal — never buckets"
    );
}

/// Mixed batch: CRC-failed SNAPSHOT_REQUEST sandwiched between
/// CRC-valid SNAPSHOT_REQUESTs. Only the valid ones must push; the
/// torn frame must drop independently. Pins the per-message gate
/// behaviour against a regression that decodes the whole batch when
/// any CRC matches.
#[test]
fn snapshot_request_torn_crc_dropped_in_mixed_batch() {
    let mut a = HostAssembler::new();
    let p_first = snapshot_request_bytes(1, SNAPSHOT_KIND_CAPTURE, "first");
    let p_torn = snapshot_request_bytes(2, SNAPSHOT_KIND_CAPTURE, "torn");
    let p_third = snapshot_request_bytes(3, SNAPSHOT_KIND_CAPTURE, "third");
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &p_first));
    buf.extend(frame_with_torn_crc(MSG_TYPE_SNAPSHOT_REQUEST, &p_torn));
    buf.extend(frame_with_crc(MSG_TYPE_SNAPSHOT_REQUEST, &p_third));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 3);
    assert!(drained.messages[0].crc_ok);
    assert!(!drained.messages[1].crc_ok);
    assert!(drained.messages[2].crc_ok);
    let (state, _bucket) = run_dispatch(&drained.messages);
    assert_eq!(
        state.snapshot_requests_pending.len(),
        2,
        "exactly the two CRC-valid SNAPSHOT_REQUESTs must push; \
         the torn middle frame must drop independently"
    );
    let ids: Vec<u32> = state
        .snapshot_requests_pending
        .iter()
        .map(|r| r.request_id)
        .collect();
    assert_eq!(ids, vec![1, 3], "torn id=2 must be filtered out");
}

/// CRC-failed SCHED_EXIT followed by CRC-failed SNAPSHOT_REQUEST in a
/// single drain: BOTH gates must short-circuit. A regression where
/// the SCHED_EXIT gate's `crc_ok` check is correct but the
/// SNAPSHOT_REQUEST gate's check is dropped would still pass the
/// SCHED_EXIT-only test; this multi-gate test catches that.
#[test]
fn both_gates_drop_torn_frames_in_same_drain() {
    let mut a = HostAssembler::new();
    let snap_payload = snapshot_request_bytes(99, SNAPSHOT_KIND_CAPTURE, "tag");
    let mut buf = Vec::new();
    buf.extend(frame_with_torn_crc(MSG_TYPE_SCHED_EXIT, b"sched-exit"));
    buf.extend(frame_with_torn_crc(
        MSG_TYPE_SNAPSHOT_REQUEST,
        &snap_payload,
    ));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 2);
    assert!(!drained.messages[0].crc_ok);
    assert!(!drained.messages[1].crc_ok);
    let (state, bucket) = run_dispatch(&drained.messages);
    assert!(
        !state.kill.load(std::sync::atomic::Ordering::Acquire),
        "torn SCHED_EXIT must not promote kill"
    );
    assert_eq!(
        read_counter(&state.kill_evt),
        0,
        "torn SCHED_EXIT must not write kill eventfd"
    );
    assert_eq!(
        state.snapshot_requests_pending.len(),
        0,
        "torn SNAPSHOT_REQUEST must not decode"
    );
    assert!(
        bucket.is_empty(),
        "both torn frames must drop from the bucket"
    );
}

/// CRC-failed SYS_RDY MUST NOT fire the boot-complete eventfd, and
/// MUST leave the one-shot handle intact for a later valid frame. A
/// torn or hostile-guest frame would otherwise let an attacker race
/// ahead of `setup_per_cpu_areas` / KASLR randomization, causing the
/// monitor's first sample iteration to read against pre-boot zeros.
#[test]
fn sys_rdy_with_torn_crc_does_not_fire_eventfd() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_torn_crc(MSG_TYPE_SYS_RDY, b"");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1, "assembler emits one message");
    assert!(
        !drained.messages[0].crc_ok,
        "torn CRC must surface as crc_ok=false"
    );
    assert_eq!(
        drained.messages[0].msg_type, MSG_TYPE_SYS_RDY,
        "msg_type unaffected by CRC mismatch"
    );
    // Capture the handle before dispatch so the counter is readable
    // post-dispatch even though the gate would have `take`n it.
    let evt_handle = run_dispatch_sys_rdy(&drained.messages);
    assert_eq!(
        evt_handle.counter, 0,
        "boot-complete eventfd must NOT be written on CRC-failed \
         SYS_RDY — hostile guest must not race ahead of percpu/KASLR"
    );
    assert!(
        evt_handle.handle_remaining,
        "Option::take must NOT consume the handle on a dropped frame — \
         a later CRC-valid SYS_RDY must still be able to promote"
    );
}

/// Positive control: a CRC-valid SYS_RDY DOES fire the eventfd and
/// consumes the Option (fire-once semantics).
#[test]
fn sys_rdy_with_valid_crc_fires_eventfd_once() {
    let mut a = HostAssembler::new();
    let bytes = frame_with_crc(MSG_TYPE_SYS_RDY, b"");
    let drained = a.feed(&bytes);
    assert_eq!(drained.messages.len(), 1);
    assert!(
        drained.messages[0].crc_ok,
        "matching CRC must surface as crc_ok=true"
    );
    let out = run_dispatch_sys_rdy(&drained.messages);
    assert_eq!(
        out.counter, 1,
        "boot-complete eventfd MUST receive a single write on \
         CRC-valid SYS_RDY"
    );
    assert!(
        !out.handle_remaining,
        "Option::take must consume the handle so subsequent \
         SYS_RDY frames do not pump the counter"
    );
}

/// Two CRC-valid SYS_RDY frames in sequence: the first promotes,
/// every subsequent frame drops. Pins `Option::take` semantics so a
/// hostile or buggy guest resending SYS_RDY cannot pump the eventfd
/// counter into EAGAIN territory or wedge a later boot signal.
#[test]
fn sys_rdy_with_valid_crc_fires_once_then_subsequent_drops() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 2);
    assert!(drained.messages[0].crc_ok);
    assert!(drained.messages[1].crc_ok);
    let out = run_dispatch_sys_rdy(&drained.messages);
    assert_eq!(
        out.counter, 1,
        "second SYS_RDY must NOT pump the eventfd — \
         Option::take consumed the handle on the first promotion"
    );
    assert!(!out.handle_remaining);
}

/// CRC-valid SYS_RDY alongside CRC-valid SCHED_EXIT in the same
/// drain: both gates fire independently. Pins per-message gate
/// dispatch — a regression that aliased the two type checks would let
/// one gate's failure mask the other.
#[test]
fn sys_rdy_and_sched_exit_fire_independently() {
    let mut a = HostAssembler::new();
    let mut buf = Vec::new();
    buf.extend(frame_with_crc(MSG_TYPE_SYS_RDY, b""));
    buf.extend(frame_with_crc(MSG_TYPE_SCHED_EXIT, b"exit-payload"));
    let drained = a.feed(&buf);
    assert_eq!(drained.messages.len(), 2);
    assert!(drained.messages[0].crc_ok);
    assert!(drained.messages[1].crc_ok);
    let out = run_dispatch_sys_rdy(&drained.messages);
    assert_eq!(out.counter, 1, "SYS_RDY must promote");
    assert!(!out.handle_remaining, "SYS_RDY handle must be consumed");
    assert!(out.kill, "SCHED_EXIT must promote kill");
    assert_eq!(
        out.kill_evt_counter, 1,
        "SCHED_EXIT must write kill eventfd"
    );
}

/// Post-dispatch SYS_RDY observations: the boot-complete eventfd
/// counter, whether the one-shot handle survived, and (for the
/// joint-fire test) the kill flag + kill eventfd counter. Reading the
/// SYS_RDY counter requires holding a clone of the handle the gate
/// `take`s, so this drives the dispatch with an out-of-band clone.
struct SysRdyOutcome {
    counter: u64,
    handle_remaining: bool,
    kill: bool,
    kill_evt_counter: u64,
}

fn run_dispatch_sys_rdy(messages: &[crate::vmm::bulk::BulkMessage]) -> SysRdyOutcome {
    let mut state = SinkState::new();
    // Clone the SYS_RDY eventfd Arc so the counter is readable even
    // after the gate `take`s the sink's handle.
    let evt_clone = state
        .sys_rdy_evt
        .as_ref()
        .expect("sys_rdy handle present at start")
        .clone();
    {
        let mut sinks = state.sinks();
        for msg in messages {
            let _ = dispatch_bulk_message(msg, &mut sinks);
        }
    }
    SysRdyOutcome {
        counter: read_counter(&evt_clone),
        handle_remaining: state.sys_rdy_evt.is_some(),
        kill: state.kill.load(std::sync::atomic::Ordering::Acquire),
        kill_evt_counter: read_counter(&state.kill_evt),
    }
}

/// Chain-level roundtrip for the wprof chunk transport, through the REAL
/// [`crate::vmm::bulk::HostAssembler`] — closes the gap the reassemble-only
/// unit tests (which build `ShmEntry` by hand) leave open. Systematically
/// mutates each frame's CRC (firecracker `parse_random_requests` pattern) and
/// splits the byte stream across two feeds, exercising the full path: guest
/// split (`wprof_trace_frames`) → real TLV bytes → `HostAssembler::feed`
/// (partial-frame residual reassembly) → `BulkMessage` → `ShmEntry` →
/// `reassemble_wprof_trace`.
#[cfg(feature = "wprof")]
mod wprof_chunk_chain_proptest {
    use super::{frame_with_crc, frame_with_torn_crc};
    use crate::vmm::bulk::HostAssembler;
    use crate::vmm::guest_comms::wprof_trace_frames;
    use crate::vmm::wire::{MsgType, ShmEntry};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn wprof_reassembly_survives_transport_and_mutation(
            buf in proptest::collection::vec(any::<u8>(), 0..64usize),
            cap in 1usize..=8,
            torn_flags in proptest::collection::vec(any::<bool>(), 0..70usize),
            split_frac in 0.0f64..=1.0,
            interleave_frac in 0.0f64..=1.0,
        ) {
            // Guest side: split the trace into (msg_type, chunk) frames exactly
            // as production does. wprof_trace_frames always puts the terminal
            // LAST, so any torn wprof frame is reached by reassembly before it
            // breaks at the terminal.
            let frames = wprof_trace_frames(&buf, cap);
            let interleave_at = (interleave_frac * frames.len() as f64) as usize;
            let mut any_wprof_torn = false;
            let mut stream: Vec<u8> = Vec::new();
            for (i, (msg_type, chunk)) in frames.iter().enumerate() {
                // A crc-GOOD unrelated frame injected BETWEEN wprof frames must
                // be skipped by reassembly without breaking chunk order.
                if i == interleave_at {
                    stream.extend_from_slice(&frame_with_crc(
                        MsgType::Stdout.wire_value(),
                        b"noise",
                    ));
                }
                if torn_flags.get(i).copied().unwrap_or(false) {
                    any_wprof_torn = true;
                    stream.extend_from_slice(&frame_with_torn_crc(*msg_type, chunk));
                } else {
                    stream.extend_from_slice(&frame_with_crc(*msg_type, chunk));
                }
            }
            // Host side: feed the byte stream through the real assembler, split
            // across two feeds so the split can land mid-header or mid-payload
            // (exercises HostAssembler's partial-frame residual retention).
            let split = (split_frac * stream.len() as f64) as usize;
            let mut a = HostAssembler::new();
            let mut entries: Vec<ShmEntry> = Vec::new();
            for part in [&stream[..split], &stream[split..]] {
                for m in a.feed(part).messages {
                    entries.push(ShmEntry {
                        msg_type: m.msg_type,
                        payload: m.payload.to_vec(),
                        crc_ok: m.crc_ok,
                    });
                }
            }
            let got = crate::test_support::wprof::reassemble_wprof_trace(&entries);
            // Expected: None when any wprof frame is torn (reassembly bails to
            // None on the first torn wprof frame) OR the trace is empty (empty
            // terminal → skip-on-empty); otherwise the reassembled bytes equal
            // the original trace exactly, proving guest-split → transport →
            // host-reassemble is lossless and order-preserving.
            let expected = if any_wprof_torn || buf.is_empty() {
                None
            } else {
                Some(buf.clone())
            };
            prop_assert_eq!(got, expected);
        }
    }
}
