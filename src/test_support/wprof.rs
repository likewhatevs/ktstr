//! Public wire-format constants + assertion helpers for the wprof
//! Perfetto-trace artifacts produced by `#[ktstr_test(wprof, ...)]`
//! tests.
//!
//! The wprof trace is generated inside the guest VM and shipped to
//! the host via the `MsgType::WprofTrace` virtq message; the
//! host-side dispatch arm in
//! `test_support::eval::run_ktstr_test_inner`'s wprof handler writes
//! a `.wprof.pb` file per test run under
//! `{sidecar_dir()}/{test_name}-{variant_hash:016x}.wprof.pb` — the
//! variant hash keys the artifact to the resolved variant so sibling
//! gauntlet presets of the same test do not clobber each other (same
//! convention as the `.repro.failure-dump.json` sidecar).
//!
//! Assertions on the `.pb` MUST run host-side via the
//! `#[ktstr_test(post_vm = ...)]` callback, NOT inside the guest
//! test body — the test body runs INSIDE the guest VM and the
//! guest mount table does NOT include a virtio-fs mount of the
//! host sidecar directory. A guest-side `std::fs::read(...)` on
//! the host sidecar path resolves to a host path the guest cannot
//! open and returns ENOENT regardless of whether the host-side
//! write succeeded.
//!
//! ## Drift-safe test naming
//!
//! Test authors writing `post_vm` callbacks should derive
//! `.wprof.pb` paths through the
//! [`VmResult::wprof_pb_path`](crate::vmm::VmResult::wprof_pb_path)
//! method on the `&VmResult` arg rather than recomputing the path
//! with a hardcoded fn-name literal. The method derives the path
//! from the entry name the macro stamped at compile time — a future
//! rename of the test fn surfaces the drift as a `None` bail rather
//! than a runtime ENOENT against a stale literal.
//!
//! The high-level
//! [`VmResult::assert_wprof_pb_landed`](crate::vmm::VmResult::assert_wprof_pb_landed)
//! sugar collapses the recurring `post_vm`-callback boilerplate into
//! a single method call. Use it as the default; reach for
//! [`assert_wprof_pb_shape`] directly only when the callback owns
//! path resolution (e.g. checking a specific
//! `.repro.wprof.pb` artifact via
//! [`VmResult::repro_wprof_pb_path`](crate::vmm::VmResult::repro_wprof_pb_path)).

use anyhow::{Context, anyhow, ensure};
use std::path::Path;

/// Minimum wprof `.pb` file size in bytes. wprof's `init_pb_trace`
/// emits a ~4 KB interned-string table (CAT + NAME + ANNK + ANNV
/// ranges, ~216 entries × ~20 bytes wire cost) on every capture
/// regardless of trace activity. A smaller file means wprof either
/// aborted before `init_pb_trace` OR the .pb write/transport
/// truncated.
pub const WPROF_PB_MIN_BYTES: usize = 4096;

/// Perfetto wire-format leading byte: `(1 << 3) | 2 == 0x0a` for
/// `message Trace { repeated TracePacket packets = 1; }` (field=1,
/// wire_type=2 length-delimited). Stable across Perfetto's
/// published schema history.
pub const PERFETTO_TRACE_PACKETS_TAG: u8 = 0x0a;

/// Reassemble a wprof `.pb` from a bulk-port drain: concatenate every
/// [`WprofTraceChunk`](crate::vmm::wire::MsgType::WprofTraceChunk) payload in
/// arrival order, then append the terminal
/// [`WprofTrace`](crate::vmm::wire::MsgType::WprofTrace) payload. The guest
/// (`guest_comms::send_wprof_trace`) ships a trace that fits in one frame as
/// a lone terminal `WprofTrace` (no chunks), so this reassembles to exactly
/// that payload for the common case.
///
/// Returns `None` — the caller then writes no `.pb` — when:
/// - no terminal `WprofTrace` frame is present (an incomplete or absent
///   trace), or the reassembled trace is empty, matching the pre-chunking
///   skip-on-empty/absent behavior; or
/// - any wprof frame (chunk or terminal) failed its transport CRC. A torn
///   frame means the trace is corrupt; concatenating the surviving frames
///   would produce a hole-y `.pb` that still passes the leading-tag/size
///   shape check but decodes to garbage. Bailing to `None` fails loudly (the
///   post_vm `.pb`-landed assert reports a missing file) rather than shipping
///   a plausible-but-corrupt artifact. Non-wprof crc-bad frames are ignored
///   — only the trace's own frames gate it.
pub(crate) fn reassemble_wprof_trace(entries: &[crate::vmm::wire::ShmEntry]) -> Option<Vec<u8>> {
    use crate::vmm::wire::MsgType;
    let mut out: Vec<u8> = Vec::new();
    let mut saw_terminal = false;
    for e in entries {
        match MsgType::from_wire(e.msg_type) {
            Some(MsgType::WprofTraceChunk) => {
                if !e.crc_ok {
                    return None;
                }
                out.extend_from_slice(&e.payload);
            }
            Some(MsgType::WprofTrace) => {
                if !e.crc_ok {
                    return None;
                }
                out.extend_from_slice(&e.payload);
                saw_terminal = true;
                break;
            }
            _ => {}
        }
    }
    (saw_terminal && !out.is_empty()).then_some(out)
}

/// Verify the wprof `.pb` at `path` exists, is at least
/// [`WPROF_PB_MIN_BYTES`] bytes, and leads with
/// [`PERFETTO_TRACE_PACKETS_TAG`].
///
/// Returns `Err(_)` with a diagnostic naming the specific
/// regression hop (missing file, truncated, wrong format) so a
/// debugging operator can trace the failure back to the transport
/// site that broke. The error message references the host-side
/// write site at `test_support::eval` for missing-file diagnoses.
///
/// Intended use: a `#[ktstr_test(post_vm = my_check)]` callback
/// resolves the per-test `.wprof.pb` path via
/// [`VmResult::wprof_pb_path`](crate::vmm::VmResult::wprof_pb_path)
/// and forwards the `Result` from this helper. The
/// [`VmResult::assert_wprof_pb_landed`](crate::vmm::VmResult::assert_wprof_pb_landed)
/// method packages the common path-derive + shape-check into a
/// single call. Do NOT call from inside the guest test body —
/// the guest cannot read the host sidecar directory (see the
/// module-level doc).
pub fn assert_wprof_pb_shape(path: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "wprof .pb missing at {}. Chain: #[ktstr_test(wprof)] → \
             KtstrTestEntry::wprof → primary VM builder.wprof(Some(config)) \
             at src/test_support/eval/mod.rs → KTSTR_WPROF_ARGS cmdline → \
             guest spawn_wprof_if_configured → send_wprof_trace → host \
             MsgType::WprofTrace arm → \
             sidecar_dir.join(\"<name>-<variant_hash:016x>.wprof.pb\")",
            path.display(),
        )
    })?;
    ensure!(
        bytes.len() >= WPROF_PB_MIN_BYTES,
        "wprof .pb at {} is only {} bytes — expected >= {WPROF_PB_MIN_BYTES}. \
         wprof's init_pb_trace emits a ~4 KB interned-string table on every \
         capture; a smaller file means wprof either aborted before \
         init_pb_trace or the .pb write/transport truncated.",
        path.display(),
        bytes.len(),
    );
    // Size check above guarantees bytes is non-empty; bytes[0] is
    // a direct index rather than `bytes.first()` to make the
    // size-check → indexability dependency explicit.
    let first = bytes[0];
    if first != PERFETTO_TRACE_PACKETS_TAG {
        return Err(anyhow!(
            "wprof .pb at {} first byte {first:#04x} — expected \
             {PERFETTO_TRACE_PACKETS_TAG:#04x} (field=1, wire_type=2, the \
             Perfetto `Trace.packets` repeated TracePacket tag). File may \
             be truncated, in a different format, or corrupted.",
            path.display(),
        ));
    }
    Ok(())
}

#[cfg(all(test, feature = "wprof"))]
mod tests {
    use super::reassemble_wprof_trace;
    use crate::vmm::guest_comms::wprof_trace_frames;
    use crate::vmm::wire::{MsgType, ShmEntry};

    /// Build the `ShmEntry` drain the host parses from the frames the guest
    /// planner emits for `buf` at `cap` (all crc-good), then reassemble it —
    /// the full guest-split → host-concat roundtrip against the exact
    /// production splitting code.
    fn roundtrip(buf: &[u8], cap: usize) -> Option<Vec<u8>> {
        let entries: Vec<ShmEntry> = wprof_trace_frames(buf, cap)
            .into_iter()
            .map(|(msg_type, chunk)| ShmEntry {
                msg_type,
                payload: chunk.to_vec(),
                crc_ok: true,
            })
            .collect();
        reassemble_wprof_trace(&entries)
    }

    #[test]
    fn frames_tag_all_but_last_as_chunk() {
        let buf: Vec<u8> = (0..10u8).collect();
        // 10 bytes / cap 4 => [4, 4, 2] => chunk, chunk, terminal.
        let frames = wprof_trace_frames(&buf, 4);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].0, MsgType::WprofTraceChunk.wire_value());
        assert_eq!(frames[1].0, MsgType::WprofTraceChunk.wire_value());
        assert_eq!(frames[2].0, MsgType::WprofTrace.wire_value());
        assert_eq!(frames[0].1, &[0, 1, 2, 3]);
        assert_eq!(frames[1].1, &[4, 5, 6, 7]);
        assert_eq!(frames[2].1, &[8, 9]);
    }

    #[test]
    fn single_frame_is_lone_terminal() {
        // A buf that fits in one cap-byte frame ships as one terminal
        // WprofTrace, no chunks (host reassembly is a no-op concat).
        let buf: Vec<u8> = (0..4u8).collect();
        let frames = wprof_trace_frames(&buf, 4);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, MsgType::WprofTrace.wire_value());
    }

    #[test]
    fn roundtrip_small_cap_byte_identical() {
        for len in [0usize, 1, 3, 4, 5, 8, 9, 100] {
            let buf: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let out = roundtrip(&buf, 4);
            if buf.is_empty() {
                // Empty trace: one empty terminal frame → reassembly returns
                // None (skip-on-empty), matching pre-chunking behavior.
                assert_eq!(out, None, "empty trace reassembles to None");
            } else {
                assert_eq!(out.as_deref(), Some(buf.as_slice()), "len={len}");
            }
        }
    }

    #[test]
    fn boundary_at_real_cap() {
        let cap = crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD as usize;
        // Exactly cap bytes => single terminal frame, no chunks.
        let exact = vec![0xABu8; cap];
        let frames = wprof_trace_frames(&exact, cap);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, MsgType::WprofTrace.wire_value());
        assert_eq!(roundtrip(&exact, cap).as_deref(), Some(exact.as_slice()));

        // cap + 1 bytes => one cap-byte chunk + a 1-byte terminal.
        let mut over = vec![0xCDu8; cap];
        over.push(0xEE);
        let frames = wprof_trace_frames(&over, cap);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].0, MsgType::WprofTraceChunk.wire_value());
        assert_eq!(frames[0].1.len(), cap);
        assert_eq!(frames[1].0, MsgType::WprofTrace.wire_value());
        assert_eq!(frames[1].1, &[0xEE]);
        assert_eq!(roundtrip(&over, cap).as_deref(), Some(over.as_slice()));
    }

    #[test]
    fn torn_chunk_bails_to_none() {
        // A crc-bad chunk means the trace is corrupt; reassembly returns None
        // (write no .pb) rather than concatenating a hole-y .pb.
        let entries = vec![
            ShmEntry {
                msg_type: MsgType::WprofTraceChunk.wire_value(),
                payload: vec![1, 2, 3],
                crc_ok: false,
            },
            ShmEntry {
                msg_type: MsgType::WprofTrace.wire_value(),
                payload: vec![4, 5],
                crc_ok: true,
            },
        ];
        assert_eq!(reassemble_wprof_trace(&entries), None);
    }

    #[test]
    fn torn_terminal_bails_to_none() {
        let entries = vec![
            ShmEntry {
                msg_type: MsgType::WprofTraceChunk.wire_value(),
                payload: vec![1, 2, 3],
                crc_ok: true,
            },
            ShmEntry {
                msg_type: MsgType::WprofTrace.wire_value(),
                payload: vec![4, 5],
                crc_ok: false,
            },
        ];
        assert_eq!(reassemble_wprof_trace(&entries), None);
    }

    #[test]
    fn no_terminal_returns_none() {
        // Chunks present but no terminal WprofTrace => incomplete => None.
        let entries = vec![ShmEntry {
            msg_type: MsgType::WprofTraceChunk.wire_value(),
            payload: vec![1, 2, 3],
            crc_ok: true,
        }];
        assert_eq!(reassemble_wprof_trace(&entries), None);
    }

    #[test]
    fn unrelated_crc_bad_frame_ignored() {
        // A crc-bad frame of an unrelated msg_type does not gate the trace.
        let entries = vec![
            ShmEntry {
                msg_type: MsgType::Stdout.wire_value(),
                payload: vec![9, 9, 9],
                crc_ok: false,
            },
            ShmEntry {
                msg_type: MsgType::WprofTrace.wire_value(),
                payload: vec![4, 5],
                crc_ok: true,
            },
        ];
        assert_eq!(reassemble_wprof_trace(&entries).as_deref(), Some(&[4u8, 5][..]));
    }

    fn chunk(payload: &[u8], crc_ok: bool) -> ShmEntry {
        ShmEntry {
            msg_type: MsgType::WprofTraceChunk.wire_value(),
            payload: payload.to_vec(),
            crc_ok,
        }
    }

    fn terminal(payload: &[u8], crc_ok: bool) -> ShmEntry {
        ShmEntry {
            msg_type: MsgType::WprofTrace.wire_value(),
            payload: payload.to_vec(),
            crc_ok,
        }
    }

    #[test]
    fn torn_middle_chunk_in_n_gt_2_bails() {
        // A crc-bad chunk at an INTERIOR position (not just the first) torns
        // the whole trace → None. Pins the "any torn wprof frame → None"
        // invariant at an arbitrary position (firecracker per-element rigor).
        let entries = vec![
            chunk(&[1, 2], true),
            chunk(&[3, 4], false), // torn middle chunk
            chunk(&[5, 6], true),
            terminal(&[7, 8], true),
        ];
        assert_eq!(reassemble_wprof_trace(&entries), None);
    }

    #[test]
    fn interleaved_crc_good_nonwprof_frame_skipped_preserves_order() {
        // A crc-GOOD unrelated frame BETWEEN chunks is skipped by msg_type and
        // does NOT break chunk concatenation order (only unrelated crc-BAD was
        // covered before).
        let entries = vec![
            chunk(&[1, 2], true),
            ShmEntry {
                msg_type: MsgType::Stdout.wire_value(),
                payload: vec![9, 9],
                crc_ok: true,
            },
            chunk(&[3, 4], true),
            terminal(&[5], true),
        ];
        assert_eq!(
            reassemble_wprof_trace(&entries).as_deref(),
            Some(&[1, 2, 3, 4, 5][..])
        );
    }

    #[test]
    fn duplicate_terminal_stops_at_first() {
        // Reassembly breaks at the FIRST crc-good terminal; a second terminal
        // and its bytes are ignored. A well-formed guest sends exactly one
        // terminal (the last frame), so a duplicate is malformed/hostile —
        // pinning break-at-first (not append-second) is the defensive choice.
        let entries = vec![chunk(&[1], true), terminal(&[2], true), terminal(&[3], true)];
        assert_eq!(reassemble_wprof_trace(&entries).as_deref(), Some(&[1u8, 2][..]));
    }

    #[test]
    fn chunk_after_terminal_ignored() {
        // A WprofTraceChunk AFTER the terminal is swallowed by the
        // break-at-first-terminal path (a well-formed guest never emits one).
        let entries = vec![chunk(&[1], true), terminal(&[2], true), chunk(&[3], true)];
        assert_eq!(reassemble_wprof_trace(&entries).as_deref(), Some(&[1u8, 2][..]));
    }
}
