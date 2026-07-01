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
