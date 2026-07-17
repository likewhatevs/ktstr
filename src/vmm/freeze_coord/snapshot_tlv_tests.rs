//! Unit coverage for the TLV-based snapshot req/reply wiring.
//!
//! `decode_snapshot_request` and `frame_snapshot_reply` are
//! testable in isolation — every assertion in the freeze
//! coordinator's TOKEN_TX dispatch flows through these two
//! helpers, so verifying their wire-format contract pins the
//! load-bearing behaviour without booting a VM. Chain-level
//! integration coverage of `queue_input_port1` lives in
//! `virtio_console`'s own test module; here we cover only the
//! payload encode / decode boundary.
use super::*;
use crate::vmm::wire::{
    FRAME_HEADER_SIZE, MSG_TYPE_SNAPSHOT_REPLY, MSG_TYPE_TEARDOWN_BARRIER_ACK,
    SNAPSHOT_KIND_CAPTURE, SNAPSHOT_KIND_NONE, SNAPSHOT_KIND_WATCH, SNAPSHOT_REASON_MAX,
    SNAPSHOT_STATUS_ERR, SNAPSHOT_STATUS_OK, SNAPSHOT_TAG_MAX, ShmMessage, SnapshotReplyPayload,
    SnapshotRequestPayload,
};
use zerocopy::{FromBytes, IntoBytes};

fn make_request_bytes(request_id: u32, kind: u32, tag: &str) -> Vec<u8> {
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

/// Happy-path CAPTURE request decodes to the matching typed
/// fields and trims the tag at the first NUL.
#[test]
fn decode_capture_request_round_trip() {
    let bytes = make_request_bytes(7, SNAPSHOT_KIND_CAPTURE, "snap_1");
    let req = decode_snapshot_request(&bytes).expect("valid request decodes");
    assert_eq!(req.request_id, 7);
    assert_eq!(req.kind, SNAPSHOT_KIND_CAPTURE);
    assert_eq!(req.tag, "snap_1");
}

/// WATCH request decodes the same way as CAPTURE — the kind
/// dispatch happens at the call site, not inside the decoder.
#[test]
fn decode_watch_request_round_trip() {
    let bytes = make_request_bytes(99, SNAPSHOT_KIND_WATCH, "scx_root");
    let req = decode_snapshot_request(&bytes).expect("valid request decodes");
    assert_eq!(req.kind, SNAPSHOT_KIND_WATCH);
    assert_eq!(req.tag, "scx_root");
}

/// Wrong-sized payload (1 byte short of the typed payload) is
/// rejected — protects against a malformed guest stamping a
/// partial request that would otherwise zerocopy into stack
/// garbage.
#[test]
fn decode_rejects_undersized_payload() {
    let mut bytes = make_request_bytes(1, SNAPSHOT_KIND_CAPTURE, "x");
    bytes.pop();
    assert!(decode_snapshot_request(&bytes).is_none());
}

/// Wrong-sized payload (1 byte longer than typed payload) is
/// rejected.
#[test]
fn decode_rejects_oversized_payload() {
    let mut bytes = make_request_bytes(1, SNAPSHOT_KIND_CAPTURE, "x");
    bytes.push(0xAA);
    assert!(decode_snapshot_request(&bytes).is_none());
}

/// `request_id == 0` is rejected — the wire-format contract
/// reserves zero so a zero-initialised reply payload from a
/// prior protocol version cannot accidentally match.
#[test]
fn decode_rejects_zero_request_id() {
    let bytes = make_request_bytes(0, SNAPSHOT_KIND_CAPTURE, "x");
    assert!(decode_snapshot_request(&bytes).is_none());
}

/// `kind == NONE` is rejected — the sentinel value must not
/// appear on the wire.
#[test]
fn decode_rejects_kind_none() {
    let bytes = make_request_bytes(1, SNAPSHOT_KIND_NONE, "x");
    assert!(decode_snapshot_request(&bytes).is_none());
}

/// Unknown kind values decode to `Some` — the dispatch in the
/// freeze coord matches on `kind` and frames an ERR reply for
/// anything outside the CAPTURE/WATCH set, so the decoder must
/// not pre-filter on kind.
#[test]
fn decode_accepts_unknown_kind_for_dispatch_handling() {
    let bytes = make_request_bytes(42, 0xDEAD_BEEF, "tag");
    let req = decode_snapshot_request(&bytes).expect("decode succeeds");
    assert_eq!(req.kind, 0xDEAD_BEEF);
    assert_eq!(req.tag, "tag");
}

/// Tag without an internal NUL fills the whole buffer; the
/// decoder takes the full `SNAPSHOT_TAG_MAX` bytes.
#[test]
fn decode_full_buffer_tag_uses_full_length() {
    let long = "a".repeat(SNAPSHOT_TAG_MAX);
    let bytes = make_request_bytes(1, SNAPSHOT_KIND_CAPTURE, &long);
    let req = decode_snapshot_request(&bytes).expect("decode succeeds");
    assert_eq!(req.tag.len(), SNAPSHOT_TAG_MAX);
    assert!(req.tag.chars().all(|c| c == 'a'));
}

/// Reply frame is exactly header + `size_of::<SnapshotReplyPayload>()`;
/// CRC32 over payload bytes matches the wire-format contract the
/// guest's `request_snapshot` reply reader enforces (guest_comms.rs)
/// when validating the host->guest reply frame.
#[test]
fn frame_reply_size_and_crc() {
    let bytes = frame_snapshot_reply(123, SNAPSHOT_STATUS_OK, "");
    assert_eq!(
        bytes.len(),
        FRAME_HEADER_SIZE + std::mem::size_of::<SnapshotReplyPayload>()
    );
    let header = ShmMessage::read_from_bytes(&bytes[..FRAME_HEADER_SIZE]).expect("header decodes");
    assert_eq!(header.msg_type, MSG_TYPE_SNAPSHOT_REPLY);
    assert_eq!(
        header.length as usize,
        std::mem::size_of::<SnapshotReplyPayload>()
    );
    let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
    assert_eq!(header.crc32, crc32fast::hash(payload_bytes));
}

/// The teardown-barrier acknowledgement is an empty, CRC-valid TLV.
/// The guest blocks on exactly this frame before entering the
/// intentional infinite wedge.
#[test]
fn teardown_barrier_ack_is_empty_crc_valid_tlv() {
    let bytes = frame_teardown_barrier_ack();
    assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
    let header = ShmMessage::read_from_bytes(&bytes).expect("header decodes");
    assert_eq!(header.msg_type, MSG_TYPE_TEARDOWN_BARRIER_ACK);
    assert_eq!(header.length, 0);
    assert_eq!(header.crc32, crc32fast::hash(&[]));
}

/// Reply payload round-trips through bytes — the request_id
/// echo, the status, and the reason text are preserved
/// exactly.
#[test]
fn frame_reply_payload_round_trip() {
    let bytes = frame_snapshot_reply(0xCAFE_BABE, SNAPSHOT_STATUS_ERR, "rendezvous timeout");
    let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
    let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes).expect("payload decodes");
    assert_eq!(reply.request_id, 0xCAFE_BABE);
    assert_eq!(reply.status, SNAPSHOT_STATUS_ERR);
    let len = reply
        .reason
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(SNAPSHOT_REASON_MAX);
    assert_eq!(&reply.reason[..len], b"rendezvous timeout");
}

/// Reasons longer than `SNAPSHOT_REASON_MAX` are truncated to
/// the buffer; the trailing byte may be a partial UTF-8
/// sequence but never overflows.
#[test]
fn frame_reply_truncates_long_reason() {
    let long = "x".repeat(SNAPSHOT_REASON_MAX + 16);
    let bytes = frame_snapshot_reply(1, SNAPSHOT_STATUS_ERR, &long);
    let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
    let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes).expect("payload decodes");
    assert_eq!(reply.reason.len(), SNAPSHOT_REASON_MAX);
    assert!(reply.reason.iter().all(|&b| b == b'x'));
}

/// Empty reason yields a fully-zeroed reason buffer — the
/// guest side renders this as the empty string.
#[test]
fn frame_reply_empty_reason_zero_pads() {
    let bytes = frame_snapshot_reply(1, SNAPSHOT_STATUS_OK, "");
    let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
    let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes).expect("payload decodes");
    assert!(reply.reason.iter().all(|&b| b == 0));
}

/// Pin three load-bearing substrings of the cold-path "kaslr_offset
/// == 0 but link_kva is high-half" diagnostic (Fix C, the
/// `kaslr_offset == 0` high-half silent-misfire gate in
/// `arm_user_watchpoint`) survive the wire round-trip
/// without truncation:
///   - `kaslr_offset` and `kern_virt_kaslr` — the WHY the e2e test
///     `kaslr_off_watch_snapshot_jiffies_64_errs` greps for to verify
///     Fix C's typed-Err path fired (not the silent-arm regression).
///   - `#[ktstr_test(kaslr = false)]` — the actionable remediation
///     tip the test author needs to see when they trip this rejection.
///
/// At the original 64-byte `SNAPSHOT_REASON_MAX` all three substrings
/// landed past the cut and the e2e failed on truncation rather than
/// on Fix C semantics. An intermediate 256-byte size fit the WHY
/// substrings but still truncated the remediation tail. 512 fits
/// the full rendered diagnostic with headroom. This test guards the
/// diagnostic capacity from sliding back.
#[test]
fn frame_reply_preserves_fix_c_diagnostic_substrings() {
    let fix_c = "symbol 'jiffies_64' link_kva 0xffffffff812d2000 is in \
                 x86_64 kernel high-half but kern_virt_kaslr Arc has not \
                 published a non-zero slide (coord_kaslr_offset() == 0); \
                 refusing to arm DR at link-time KVA — would never match \
                 guest writes under KASLR-on. If nokaslr semantics intended, \
                 set `#[ktstr_test(kaslr = false)]`.";
    let bytes = frame_snapshot_reply(1, SNAPSHOT_STATUS_ERR, fix_c);
    let payload_bytes = &bytes[FRAME_HEADER_SIZE..];
    let reply = SnapshotReplyPayload::read_from_bytes(payload_bytes).expect("payload decodes");
    let len = reply
        .reason
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(SNAPSHOT_REASON_MAX);
    let s = std::str::from_utf8(&reply.reason[..len]).expect("Fix C diagnostic is valid UTF-8");
    assert!(
        s.contains("kaslr_offset"),
        "diagnostic lost 'kaslr_offset' substring; \
         tests/kaslr_axis_e2e.rs greps for it (recorded reason: {s:?})"
    );
    assert!(
        s.contains("kern_virt_kaslr"),
        "diagnostic lost 'kern_virt_kaslr' substring; \
         tests/kaslr_axis_e2e.rs greps for it (recorded reason: {s:?})"
    );
    assert!(
        s.contains("#[ktstr_test(kaslr = false)]"),
        "diagnostic lost the actionable remediation tip; \
         test author would not know how to unblock (recorded reason: {s:?})"
    );
}
