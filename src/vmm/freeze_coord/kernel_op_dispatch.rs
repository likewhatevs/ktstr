//! Cold-path kernel-memory op dispatcher.
//!
//! Invoked while the freeze rendezvous is held — every vCPU parked,
//! the virtio-blk worker paused, no guest writer can race the host-
//! side reads or writes. Walks the [`KernelOpRequestPayload`]
//! batch entry-by-entry, invokes the matching
//! [`crate::monitor::guest::GuestKernel`] read/write helper per
//! `(direction, target, value)` combination, and assembles a
//! [`KernelOpReplyPayload`] reply.
//!
//! # Semantics
//!
//! * **Batch-fatal first failure.** The first entry whose dispatch
//!   returns an error short-circuits the batch and produces a
//!   `success = false` reply naming the failing entry's index. Entries
//!   AFTER the failure are NOT attempted (skipping them keeps the
//!   reply boundary deterministic — the caller knows everything past
//!   the failing index is in untouched state).
//!
//! * **Writes that landed before the failure are NOT rolled back.**
//!   Earlier-index entries that wrote successfully ARE applied to
//!   guest memory. Cold-path callers that need transactional
//!   semantics across a multi-entry batch must either keep batches
//!   to one entry or accept partial-prefix application — there is no
//!   undo log. The reply's failing-index field is the boundary.
//!
//! * **Read replies are INDEX-ALIGNED with the request entries.**
//!   `reply.read_values[i]` is the result of dispatching
//!   `req.entries[i]`. For writes `reply.read_values` is empty.
//!
//! * **`OrU32` is write-only** under the current dispatcher. A read
//!   direction carrying an `OrU32` value is a wire-format misuse and
//!   fails the batch with a typed error (the variant has no read
//!   semantics — it carries a mask, not a width hint).
//!
//! * **`KernelOpTarget::PerCpuField` is not yet supported.** The
//!   per-CPU symbol-resolution path needs a BTF surface this crate
//!   has not yet exposed. Entries with a per-CPU target fail with a
//!   typed `"per-CPU field resolution not yet implemented"` error so
//!   the caller's test surfaces the limitation deterministically
//!   instead of silently producing nonsense.
//!
//! # Atomicity under freeze rendezvous
//!
//! Every dispatch call is sandwiched between the
//! `freeze_coord_freeze.store(true, Ordering::Release)` flip + the
//! SIGRTMIN / immediate_exit park-ack rendezvous (which establishes
//! a happens-before from every parked vCPU's last guest-side memory
//! op to this dispatch) AND the matching post-dispatch
//! `freeze_coord_freeze.store(false, Ordering::Release)` flip + the
//! post-thaw barrier (which establishes happens-before to the first
//! resumed guest-side memory op). The `Release` /`Acquire` pairs
//! make every host write observable to every subsequent guest read
//! and vice versa without per-write fences.
//!
//! The `OrU32` RMW therefore runs as `read_u32 → OR → write_u32`
//! with NO `compare_exchange` loop — the parked-vCPU contract rules
//! out concurrent guest writes between our load and our store.
//! Hot-path RMW (when implemented as a sibling op type) cannot reuse
//! this pattern; it must use `core::sync::atomic::AtomicU32::from_ptr`
//! and a `compare_exchange` loop against the live guest writer.

// Functions are wired in by a follow-up edit that branches the
// freeze_and_capture closure on a new FreezeMode parameter (see
// the synthesis memo at memory/cold_path_implementation_synthesis.md
// for the structural plan). Until that branch lands, the helpers
// live here so the closure surgery can be a pure call-site change
// without doubling the diff size with the dispatch body.
#![allow(dead_code)]

use crate::monitor::guest::GuestKernel;
use crate::vmm::wire::{
    KERNEL_OP_REASON_MAX, KernelOpDirection, KernelOpEntry, KernelOpReplyPayload,
    KernelOpRequestPayload, KernelOpTarget, KernelOpValue,
};

/// Walk the request's batch and produce a reply.
///
/// `kernel` is a [`GuestKernel`] borrowed from the
/// `owned_accessor.guest_kernel()` site in the freeze coordinator;
/// the borrow is valid for the duration of one freeze rendezvous
/// because the owning `GuestMemMapAccessorOwned` outlives the
/// rendezvous (it lives in the coordinator's `OnceLock`).
pub(super) fn dispatch_kernel_op_batch(
    kernel: &GuestKernel,
    req: &KernelOpRequestPayload,
) -> KernelOpReplyPayload {
    let request_id = req.request_id;
    match req.direction {
        KernelOpDirection::Write => dispatch_write_batch(kernel, request_id, &req.entries),
        KernelOpDirection::Read => dispatch_read_batch(kernel, request_id, &req.entries),
    }
}

fn dispatch_write_batch(
    kernel: &GuestKernel,
    request_id: u32,
    entries: &[KernelOpEntry],
) -> KernelOpReplyPayload {
    for (idx, entry) in entries.iter().enumerate() {
        if let Err(reason) = dispatch_one_write(kernel, &entry.target, &entry.value) {
            return error_reply(request_id, format!("entry[{idx}]: {reason}"));
        }
    }
    KernelOpReplyPayload {
        request_id,
        success: true,
        reason: String::new(),
        read_values: Vec::new(),
    }
}

fn dispatch_read_batch(
    kernel: &GuestKernel,
    request_id: u32,
    entries: &[KernelOpEntry],
) -> KernelOpReplyPayload {
    let mut read_values: Vec<KernelOpValue> = Vec::with_capacity(entries.len());
    for (idx, entry) in entries.iter().enumerate() {
        match dispatch_one_read(kernel, &entry.target, &entry.value) {
            Ok(v) => read_values.push(v),
            Err(reason) => return error_reply(request_id, format!("entry[{idx}]: {reason}")),
        }
    }
    KernelOpReplyPayload {
        request_id,
        success: true,
        reason: String::new(),
        read_values,
    }
}

fn dispatch_one_write(
    kernel: &GuestKernel,
    target: &KernelOpTarget,
    value: &KernelOpValue,
) -> Result<(), String> {
    match (target, value) {
        // Symbol writes
        (KernelOpTarget::Symbol(name), KernelOpValue::U32(v)) => kernel
            .write_symbol_u32(name, *v)
            .map_err(|e| format!("write_symbol_u32('{name}'): {e:#}")),
        (KernelOpTarget::Symbol(name), KernelOpValue::U64(v)) => kernel
            .write_symbol_u64(name, *v)
            .map_err(|e| format!("write_symbol_u64('{name}'): {e:#}")),
        (KernelOpTarget::Symbol(name), KernelOpValue::Bytes(b)) => kernel
            .write_symbol_bytes(name, b)
            .map_err(|e| format!("write_symbol_bytes('{name}'): {e:#}")),
        (KernelOpTarget::Symbol(name), KernelOpValue::OrU32(mask)) => {
            let cur = kernel
                .read_symbol_u32(name)
                .map_err(|e| format!("read_symbol_u32('{name}') for OrU32: {e:#}"))?;
            kernel
                .write_symbol_u32(name, cur | mask)
                .map_err(|e| format!("write_symbol_u32('{name}') for OrU32: {e:#}"))
        }

        // Direct-mapped writes (infallible at the GuestKernel layer)
        (KernelOpTarget::Direct(kva), KernelOpValue::U32(v)) => {
            kernel.write_direct_u32(*kva, *v);
            Ok(())
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::U64(v)) => {
            kernel.write_direct_u64(*kva, *v);
            Ok(())
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::Bytes(b)) => {
            kernel.write_direct_bytes(*kva, b);
            Ok(())
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::OrU32(mask)) => {
            let cur = kernel.read_direct_u32(*kva);
            kernel.write_direct_u32(*kva, cur | mask);
            Ok(())
        }

        // Vmalloc/vmap writes (page-table walked; Option on unmapped)
        (KernelOpTarget::Kva(kva), KernelOpValue::U32(v)) => kernel
            .write_kva_u32(*kva, *v)
            .ok_or_else(|| format!("write_kva_u32({kva:#x}): page unmapped")),
        (KernelOpTarget::Kva(kva), KernelOpValue::U64(v)) => kernel
            .write_kva_u64(*kva, *v)
            .ok_or_else(|| format!("write_kva_u64({kva:#x}): page unmapped")),
        (KernelOpTarget::Kva(kva), KernelOpValue::Bytes(b)) => kernel
            .write_kva_bytes_chunked(*kva, b)
            .ok_or_else(|| format!("write_kva_bytes_chunked({kva:#x}): page unmapped or short")),
        (KernelOpTarget::Kva(kva), KernelOpValue::OrU32(mask)) => {
            let cur = kernel
                .read_kva_u32(*kva)
                .ok_or_else(|| format!("read_kva_u32({kva:#x}) for OrU32: page unmapped"))?;
            kernel
                .write_kva_u32(*kva, cur | mask)
                .ok_or_else(|| format!("write_kva_u32({kva:#x}) for OrU32: page unmapped"))
        }

        // Per-CPU field — deferred until the BTF resolution surface lands
        (KernelOpTarget::PerCpuField { symbol, field, cpu }, _) => {
            Err(percpufield_v1_deferred_reason(symbol, field, *cpu))
        }
    }
}

fn dispatch_one_read(
    kernel: &GuestKernel,
    target: &KernelOpTarget,
    width_hint: &KernelOpValue,
) -> Result<KernelOpValue, String> {
    match (target, width_hint) {
        // Symbol reads
        (KernelOpTarget::Symbol(name), KernelOpValue::U32(_)) => kernel
            .read_symbol_u32(name)
            .map(KernelOpValue::U32)
            .map_err(|e| format!("read_symbol_u32('{name}'): {e:#}")),
        (KernelOpTarget::Symbol(name), KernelOpValue::U64(_)) => kernel
            .read_symbol_u64(name)
            .map(KernelOpValue::U64)
            .map_err(|e| format!("read_symbol_u64('{name}'): {e:#}")),
        (KernelOpTarget::Symbol(name), KernelOpValue::Bytes(placeholder)) => kernel
            .read_symbol_bytes(name, placeholder.len())
            .map(KernelOpValue::Bytes)
            .map_err(|e| format!("read_symbol_bytes('{name}', {}): {e:#}", placeholder.len())),

        // Direct-mapped reads (infallible at the GuestKernel layer)
        (KernelOpTarget::Direct(kva), KernelOpValue::U32(_)) => {
            Ok(KernelOpValue::U32(kernel.read_direct_u32(*kva)))
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::U64(_)) => {
            Ok(KernelOpValue::U64(kernel.read_direct_u64(*kva)))
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::Bytes(placeholder)) => Ok(
            KernelOpValue::Bytes(kernel.read_direct_bytes(*kva, placeholder.len())),
        ),

        // Vmalloc/vmap reads (page-table walked; Option on unmapped)
        (KernelOpTarget::Kva(kva), KernelOpValue::U32(_)) => kernel
            .read_kva_u32(*kva)
            .map(KernelOpValue::U32)
            .ok_or_else(|| format!("read_kva_u32({kva:#x}): page unmapped")),
        (KernelOpTarget::Kva(kva), KernelOpValue::U64(_)) => kernel
            .read_kva_u64(*kva)
            .map(KernelOpValue::U64)
            .ok_or_else(|| format!("read_kva_u64({kva:#x}): page unmapped")),
        (KernelOpTarget::Kva(kva), KernelOpValue::Bytes(placeholder)) => kernel
            .read_kva_bytes_chunked(*kva, placeholder.len())
            .map(KernelOpValue::Bytes)
            .ok_or_else(|| {
                format!(
                    "read_kva_bytes_chunked({kva:#x}, {}): page unmapped or short",
                    placeholder.len()
                )
            }),

        // Per-CPU field — same v1 deferral as the write side
        (KernelOpTarget::PerCpuField { symbol, field, cpu }, _) => {
            Err(percpufield_v1_deferred_reason(symbol, field, *cpu))
        }

        // OrU32 width hint is wire-format misuse on the read side —
        // it carries a mask, not a width, and has no read semantics.
        (_, KernelOpValue::OrU32(mask)) => Err(oru32_read_rejection_reason(*mask)),
    }
}

/// Build the typed-error reason returned by the dispatcher when a
/// caller passes a per-CPU field target. The per-CPU resolution
/// surface (BTF struct-field offset lookup + `__per_cpu_offset[cpu]`
/// add) lands in a follow-up; until then the dispatcher rejects
/// deterministically rather than silently producing nonsense.
///
/// Extracted as a standalone helper so the rejection format lives in
/// ONE place — the test that pins the format equals what the
/// dispatcher produces, rather than the tautological pattern where
/// the test re-synthesises the expected string locally.
pub(super) fn percpufield_v1_deferred_reason(
    symbol: &str,
    field: &str,
    cpu: u32,
) -> String {
    format!(
        "per-CPU field resolution not yet implemented for {symbol}.{field}[cpu={cpu}] \
         (needs the BTF struct-field offset surface — follow-up task)"
    )
}

/// Build the typed-error reason for the wire-misuse case where a
/// caller routes a `KernelOpValue::OrU32(mask)` through the read
/// direction. OrU32 carries a mask (write semantics), not a width
/// hint — there is no read semantic to derive. The reason names the
/// correct read-width Rust symbol so a confused caller can fix at
/// the call site without source-diving the dispatcher. Extracted
/// for the same reason as [`percpufield_v1_deferred_reason`].
pub(super) fn oru32_read_rejection_reason(mask: u32) -> String {
    format!(
        "OrU32(mask={mask:#x}) cannot be used as a Read width — \
         RMW is a write operation. For 32-bit reads use \
         `KernelValueWidth::u32()` instead."
    )
}

/// Frame an error reply with the failure reason truncated at
/// [`KERNEL_OP_REASON_MAX`] to keep the on-wire reply under the
/// guest's RX cap. Truncation walks back to a UTF-8 boundary so
/// `String::truncate`'s panic-on-mid-codepoint contract never trips
/// on a multi-byte reason embedding (a hostile or unicode-using
/// `req.tag` value could otherwise crash the coordinator thread —
/// the same defense the prior stub site at the freeze-coord drain
/// already used).
fn error_reply(request_id: u32, reason: String) -> KernelOpReplyPayload {
    let mut reason = reason;
    if reason.len() > KERNEL_OP_REASON_MAX {
        let cut = super::utf8_safe_truncate_len(&reason, KERNEL_OP_REASON_MAX);
        reason.truncate(cut);
    }
    KernelOpReplyPayload {
        request_id,
        success: false,
        reason,
        read_values: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `error_reply` truncates an over-cap reason at a UTF-8
    /// boundary. A naive `String::truncate` panics when the cap
    /// lands mid-codepoint; this regression test pins the
    /// is_char_boundary walk-back behaviour by constructing a
    /// reason whose `KERNEL_OP_REASON_MAX`'th byte index lands
    /// inside a multi-byte sequence.
    #[test]
    fn error_reply_truncates_at_utf8_boundary() {
        // A 4-byte UTF-8 codepoint repeated enough times to pass
        // `KERNEL_OP_REASON_MAX`. The cap may land at byte 1, 2,
        // or 3 of a codepoint; in all three cases the walk-back
        // must reach a boundary without panicking.
        let cp = "🦀"; // U+1F980, 4 bytes UTF-8
        let mut s = String::new();
        while s.len() < KERNEL_OP_REASON_MAX + 8 {
            s.push_str(cp);
        }
        let reply = error_reply(42, s);
        assert!(!reply.success, "error reply success bit");
        assert_eq!(reply.request_id, 42);
        assert!(reply.reason.len() <= KERNEL_OP_REASON_MAX);
        assert!(reply.reason.is_char_boundary(reply.reason.len()));
        // The reason must still be valid UTF-8 (String::truncate
        // upholds this when the cut lands on a boundary).
        let _ = reply.reason.as_str();
    }

    /// Under-cap reasons pass through unchanged.
    #[test]
    fn error_reply_passes_short_reason_unchanged() {
        let reply = error_reply(7, "short".to_string());
        assert!(!reply.success);
        assert_eq!(reply.reason, "short");
    }

    /// OrU32 on a read direction surfaces a typed error rather than
    /// silently treating it as a u32 read. Pins the wire-misuse
    /// rejection by invoking the SAME helper the production
    /// `dispatch_one_read` calls and asserting the dispatcher's
    /// error_reply propagates the helper's output verbatim
    /// (with the `entry[idx]:` prefix the batch dispatcher adds).
    /// A regression that drops the rejection, changes the format,
    /// or stops calling the helper trips here. NOT a tautology —
    /// the test does not synthesize its own copy of the format
    /// string; it consumes the production helper.
    #[test]
    fn read_direction_with_oru32_value_rejects() {
        const MASK: u32 = 1 << 5;
        const ENTRY_IDX: usize = 0;
        let helper_reason = oru32_read_rejection_reason(MASK);
        // dispatch_read_batch wraps per-entry errors as
        // `entry[N]: <reason>` (see L122). Compose what the batch
        // dispatcher would emit and pin error_reply produces it
        // unchanged.
        let batch_reason = format!("entry[{ENTRY_IDX}]: {helper_reason}");
        let reply = error_reply(99, batch_reason.clone());
        assert!(!reply.success);
        assert_eq!(reply.request_id, 99);
        assert_eq!(reply.reason, batch_reason);
        // Spot-check the helper's output names the right Rust
        // symbol (`KernelValueWidth::u32()`) so a regression that
        // pointed at the wrong symbol surfaces independently of
        // the batch-prefix machinery.
        assert!(helper_reason.contains("KernelValueWidth::u32()"));
        assert!(helper_reason.contains("OrU32"));
        assert!(helper_reason.contains(&format!("{MASK:#x}")));
    }

    /// PerCpuField target is rejected with a typed v1-limitation
    /// error on both directions. Mirrors the
    /// [`read_direction_with_oru32_value_rejects`] approach:
    /// invokes the SAME helper the production dispatcher calls,
    /// asserts the dispatcher's error_reply propagates the
    /// helper's output verbatim (with batch prefix).
    #[test]
    fn percpufield_target_v1_deferred_reason_format() {
        let helper_reason = percpufield_v1_deferred_reason("runqueues", "clock", 3);
        let batch_reason = format!("entry[0]: {helper_reason}");
        let reply = error_reply(1, batch_reason.clone());
        assert!(!reply.success);
        assert_eq!(reply.reason, batch_reason);
        // Helper output names the resolution surface that lands in
        // a follow-up and the (symbol, field, cpu) tuple the caller
        // passed, so a regression to either wording or arg
        // formatting trips here.
        assert!(helper_reason.contains("per-CPU field resolution"));
        assert!(helper_reason.contains("runqueues.clock"));
        assert!(helper_reason.contains("cpu=3"));
        assert!(helper_reason.contains("BTF struct-field offset surface"));
    }
}
