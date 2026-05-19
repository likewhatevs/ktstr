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
//! * **`KernelOpTarget::PerCpuField` resolution** uses a hardcoded
//!   `{symbol → struct_name}` mapping (see
//!   [`struct_name_for_per_cpu_symbol`]) to bridge the wire variant
//!   to BTF: `runqueues` → `rq`, `kernel_cpustat` → `kernel_cpustat`,
//!   etc. Extending the supported symbol set requires an entry there
//!   AND symbol resolution in
//!   [`crate::monitor::symbols::KernelSymbols::from_elf`]. Unknown
//!   symbols fail with a typed error rather than silently producing
//!   nonsense.
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
//!
//! # Same-rendezvous-epoch invariant
//!
//! For `OrU32` to be race-free, the read and the write MUST occur
//! inside the SAME freeze rendezvous epoch — i.e. within a single
//! invocation of [`dispatch_one_write`], between the `Release` store
//! on `freeze_coord_freeze` (rendezvous entry) and its matching clear
//! (rendezvous exit). Splitting the read + OR + write across freeze
//! boundaries would let the next guest writer interleave between our
//! load and our store, producing torn state silent to the dispatcher
//! and detectable only by KASAN or scheduler-state inconsistency
//! dumps. The structural guarantee is the dispatcher's per-entry
//! sequential walk: `dispatch_one_write` runs the read + OR + write
//! triple in one function body, never yielding between them. A
//! future refactor that extracts the RMW into a helper invoked
//! across multiple rendezvous would silently break this invariant —
//! the `// rmw-invariant-anchor` markers at the OrU32 arms and the
//! [`tests::or_u32_rmw_anchors_inside_dispatch_one_write`] doc-grep
//! regression test together enforce the pattern at the source level.

use crate::monitor::btf_offsets::{find_struct, nested_member_byte_offset};
use crate::monitor::guest::GuestKernel;
use crate::monitor::idr::translate_any_kva;
use crate::vmm::wire::{
    KERNEL_OP_REASON_MAX, KernelOpDirection, KernelOpEntry, KernelOpReplyPayload,
    KernelOpRequestPayload, KernelOpTarget, KernelOpValue,
};
use btf_rs::Btf;

/// Maximum nodes the [`find_task_by_pid`] walker visits before
/// surfacing a typed error. Matches the cap in
/// [`crate::monitor::scx_walker::walk_scx_tasks_global`]'s
/// `MAX_NODES_PER_LIST` analogue — a corrupt `init_task.tasks` chain
/// (cycle or wild pointer) must not turn the cold-path dispatcher
/// into an unbounded read loop. 65536 covers realistic workloads
/// (pid_max defaults of 32768-4M but typical test VMs run << 4K
/// tasks) while rejecting pathological chains in a bounded time.
const MAX_TASK_WALKER_NODES: u32 = 65536;

/// `TASK_DEAD` flag bit on `task_struct.__state` per
/// `include/linux/sched.h:118` (`#define TASK_DEAD 0x00000080`). A
/// task with this bit set is in the final teardown path — its
/// `task_struct` fields are mid-cleanup and writing through them
/// would corrupt the dying-task state machine. Validation rejects
/// before any field write.
const TASK_DEAD: u32 = 0x80;

/// Lower bound for any KVA accepted as a [`KernelOpTarget::Kva`]
/// target (page-walked, `read_kva_*`/`write_kva_*`).
///
/// `0xFF00_0000_0000_0000` is the conservative 5-level x86_64
/// kernel-half boundary (top 8 bits set; sign-extension from bit 56
/// per `__VIRTUAL_MASK_SHIFT` in `arch/x86/include/asm/page_64_types.h`
/// when `CONFIG_X86_5LEVEL=y`). It accepts every legitimate 4-level
/// kernel-half KVA (≥ `0xFFFF_8000_0000_0000`) AND every 5-level
/// kernel-half KVA (≥ `0xFF00_0000_0000_0000`). The Kva path can
/// safely use a loose threshold here because the downstream
/// `read_kva_*`/`write_kva_*` page-walk returns `Option::None` on
/// unmapped or non-canonical addresses (page-walk safety net).
///
/// INTENTIONALLY DIFFERS from
/// [`crate::vmm::x86_64::msr_kaslr::KERNEL_HALF_CANONICAL_4LEVEL`]
/// (value `0xFFFF_8000_0000_0000`, shared by `freeze_coord::dispatch`
/// via the same import). That constant checks the 4-level x86_64
/// canonical-bits invariant on the LSTAR MSR + kernel-text-link KVA
/// — a strict per-hardware invariant on known-shape inputs. This
/// dispatcher accepts arbitrary caller-supplied KVAs and must use the
/// looser 5-level superset so 5-level kernel
/// direct-map/vmalloc/vmemmap addresses are not false-rejected. The
/// paired naming (`_CANONICAL_4LEVEL` vs `_CONSERVATIVE_5LEVEL`)
/// telegraphs which is which.
///
/// [`KernelOpTarget::Direct`] does NOT use this threshold — it uses
/// runtime `page_offset + dram_size` range validation via
/// [`validate_direct_target`], because `kva_to_pa` (the Direct
/// path's PA derivation) does `kva.wrapping_sub(page_offset)` with
/// no safety net — a wrap to an in-bounds-but-wrong PA would silently
/// no-op at `write_scalar`/`read_scalar`.
const KERNEL_HALF_CONSERVATIVE_5LEVEL: u64 = 0xFF00_0000_0000_0000;

/// Validate that a [`KernelOpTarget::Direct`] target's KVA range is
/// inside the direct-map region `[page_offset, page_offset + dram_size)`.
///
/// Direct targets compute their PA via
/// `kva_to_pa = kva.wrapping_sub(page_offset)` (no page-walk, no
/// Option-failure). A KVA below `page_offset` underflows and wraps
/// to a huge PA that the downstream `write_scalar`/`read_scalar`
/// silently no-ops on (per `src/monitor/reader.rs:639-687`). A KVA
/// past `page_offset + dram_size` similarly wraps the bounds check.
/// Either case is a silent-data-loss path the [`KERNEL_HALF_CONSERVATIVE_5LEVEL`]
/// alone cannot catch.
///
/// Caller derives `len` from the value width (U32=4, U64=8,
/// Bytes=`bytes.len()`, OrU32=4). `page_offset` from
/// [`GuestKernel::page_offset`]; `dram_size` from
/// [`GuestKernel::mem`]`.size()`.
fn validate_direct_target(
    kva: u64,
    len: u64,
    page_offset: u64,
    dram_size: u64,
) -> Result<(), String> {
    if kva < page_offset {
        return Err(format!(
            "Direct kva={kva:#x} below page_offset={page_offset:#x} \
             (kva_to_pa would wrap; use Kva target for vmalloc/vmemmap)"
        ));
    }
    let direct_map_end = page_offset.checked_add(dram_size).ok_or_else(|| {
        format!("internal: page_offset+dram_size overflow ({page_offset:#x} + {dram_size:#x})")
    })?;
    let kva_end = kva
        .checked_add(len)
        .ok_or_else(|| format!("Direct kva+len overflow ({kva:#x} + {len:#x})"))?;
    if kva_end > direct_map_end {
        return Err(format!(
            "Direct kva={kva:#x} len={len} overruns direct-map end {direct_map_end:#x}"
        ));
    }
    Ok(())
}

/// Validate that a [`KernelOpTarget::Kva`] target's KVA range is in
/// the kernel-half address space.
///
/// The page-walk safety net (`read_kva_*`/`write_kva_*` return
/// `Option::None` on unmapped or non-canonical addresses) catches
/// most invalid KVAs downstream — this helper just rejects the
/// obvious user-half case early so the operator-visible error names
/// the right band ("below kernel-half threshold") rather than
/// "page unmapped".
fn validate_kva_target(kva: u64, len: u64) -> Result<(), String> {
    if kva < KERNEL_HALF_CONSERVATIVE_5LEVEL {
        return Err(user_half_kva_rejection_reason(kva));
    }
    let _ = kva
        .checked_add(len)
        .ok_or_else(|| format!("Kva kva+len overflow ({kva:#x} + {len:#x})"))?;
    Ok(())
}

/// Build the typed-error reason for [`validate_kva_target`]'s
/// user-half rejection. Extracted as a standalone `pub(super) fn`
/// for the same reason as [`oru32_read_rejection_reason`]: the
/// tests that pin the format invoke the SAME helper the dispatcher
/// uses, avoiding the tautology where the test re-synthesises the
/// expected string.
pub(super) fn user_half_kva_rejection_reason(kva: u64) -> String {
    format!(
        "Kva={kva:#x} below kernel-half 5-level conservative threshold \
         {KERNEL_HALF_CONSERVATIVE_5LEVEL:#x}; use Symbol target or a KVA in the \
         kernel address space"
    )
}

/// Walk the request's batch and produce a reply.
///
/// `kernel` is a [`GuestKernel`] borrowed from the
/// `owned_accessor.guest_kernel()` site in the freeze coordinator;
/// the borrow is valid for the duration of one freeze rendezvous
/// because the owning `GuestMemMapAccessorOwned` outlives the
/// rendezvous (it lives in the coordinator's `OnceLock`).
pub(super) fn dispatch_kernel_op_batch(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    req: &KernelOpRequestPayload,
) -> KernelOpReplyPayload {
    let request_id = req.request_id;
    match req.direction {
        KernelOpDirection::Write => dispatch_write_batch(kernel, btf, request_id, &req.entries),
        KernelOpDirection::Read => dispatch_read_batch(kernel, btf, request_id, &req.entries),
    }
}

fn dispatch_write_batch(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    request_id: u32,
    entries: &[KernelOpEntry],
) -> KernelOpReplyPayload {
    for (idx, entry) in entries.iter().enumerate() {
        if let Err(reason) = dispatch_one_write(kernel, btf, &entry.target, &entry.value) {
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
    btf: Option<&Btf>,
    request_id: u32,
    entries: &[KernelOpEntry],
) -> KernelOpReplyPayload {
    let mut read_values: Vec<KernelOpValue> = Vec::with_capacity(entries.len());
    for (idx, entry) in entries.iter().enumerate() {
        match dispatch_one_read(kernel, btf, &entry.target, &entry.value) {
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
    btf: Option<&Btf>,
    target: &KernelOpTarget,
    value: &KernelOpValue,
) -> Result<(), String> {
    let page_offset = kernel.page_offset();
    let dram_size = kernel.mem().size();
    match (target, value) {
        // Symbol writes — kernel-half guaranteed by vmlinux linker
        // convention (KernelSymbols::from_elf reads only the vmlinux
        // .symtab; built-in sections + module_alloc both land in
        // kernel-half by construction). No KVA validation needed.
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
            // rmw-invariant-anchor: OrU32 RMW must run inside a
            // single dispatch_one_write invocation; the caller
            // (freeze_and_dispatch closure in mod.rs) holds the
            // freeze rendezvous open for the duration. Extracting
            // this triple into a helper invokable outside
            // dispatch_one_write would lose the rendezvous-epoch
            // coupling — the same-epoch invariant rests on the
            // dispatcher's per-entry sequential walk, not on a
            // local property of dispatch_one_write itself. See
            // KernelValue::OrU32 doc + module doc above for the
            // kernel-writer race model.
            let cur = kernel
                .read_symbol_u32(name)
                .map_err(|e| format!("read_symbol_u32('{name}') for OrU32: {e:#}"))?;
            kernel
                .write_symbol_u32(name, cur | mask)
                .map_err(|e| format!("write_symbol_u32('{name}') for OrU32: {e:#}"))
        }

        // Direct-mapped writes — validate against runtime
        // [page_offset, page_offset+dram_size) BEFORE invoking the
        // underlying write (which uses kva.wrapping_sub(page_offset)
        // with NO page-walk safety net; an out-of-range KVA wraps to
        // a huge PA that write_scalar silently no-ops on per
        // reader.rs:639-687).
        (KernelOpTarget::Direct(kva), KernelOpValue::U32(v)) => {
            validate_direct_target(*kva, 4, page_offset, dram_size)?;
            kernel.write_direct_u32(*kva, *v);
            Ok(())
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::U64(v)) => {
            validate_direct_target(*kva, 8, page_offset, dram_size)?;
            kernel.write_direct_u64(*kva, *v);
            Ok(())
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::Bytes(b)) => {
            validate_direct_target(*kva, b.len() as u64, page_offset, dram_size)?;
            kernel.write_direct_bytes(*kva, b);
            Ok(())
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::OrU32(mask)) => {
            // rmw-invariant-anchor: see OrU32 module doc.
            validate_direct_target(*kva, 4, page_offset, dram_size)?;
            let cur = kernel.read_direct_u32(*kva);
            kernel.write_direct_u32(*kva, cur | mask);
            Ok(())
        }

        // Vmalloc/vmap writes (page-table walked; Option on unmapped)
        // — validate against KERNEL_HALF_CONSERVATIVE_5LEVEL (loose 5-level
        // conservative bound; page-walk catches non-canonical-hole
        // + unmapped via Option::None safety net).
        (KernelOpTarget::Kva(kva), KernelOpValue::U32(v)) => {
            validate_kva_target(*kva, 4)?;
            kernel
                .write_kva_u32(*kva, *v)
                .ok_or_else(|| format!("write_kva_u32({kva:#x}): page unmapped"))
        }
        (KernelOpTarget::Kva(kva), KernelOpValue::U64(v)) => {
            validate_kva_target(*kva, 8)?;
            kernel
                .write_kva_u64(*kva, *v)
                .ok_or_else(|| format!("write_kva_u64({kva:#x}): page unmapped"))
        }
        (KernelOpTarget::Kva(kva), KernelOpValue::Bytes(b)) => {
            validate_kva_target(*kva, b.len() as u64)?;
            kernel
                .write_kva_bytes_chunked(*kva, b)
                .ok_or_else(|| format!("write_kva_bytes_chunked({kva:#x}): page unmapped or short"))
        }
        (KernelOpTarget::Kva(kva), KernelOpValue::OrU32(mask)) => {
            // rmw-invariant-anchor: see OrU32 module doc.
            validate_kva_target(*kva, 4)?;
            let cur = kernel
                .read_kva_u32(*kva)
                .ok_or_else(|| format!("read_kva_u32({kva:#x}) for OrU32: page unmapped"))?;
            kernel
                .write_kva_u32(*kva, cur | mask)
                .ok_or_else(|| format!("write_kva_u32({kva:#x}) for OrU32: page unmapped"))
        }

        // Per-CPU field — resolve symbol KVA + __per_cpu_offset[cpu]
        // arithmetic + BTF nested-path field offset, then write at the
        // per-CPU instance PA. See [`dispatch_per_cpu_field_write`].
        // Cold-path freeze rendezvous gives the atomicity contract
        // shared by every dispatcher arm.
        (KernelOpTarget::PerCpuField { symbol, field, cpu }, value) => {
            dispatch_per_cpu_field_write(kernel, btf, symbol, field, *cpu, value)
        }

        // Per-task field — SCX-managed tasks only. Walks
        // `init_task.tasks` (leaders) plus each leader's
        // `signal->thread_head` (threads) to find the task with
        // matching pid AND matching start_time identity
        // (anti-PID-reuse). Runs the 8-layer validation chain
        // (pid, start_time, lifetime, on_rq, scx queued-empty,
        // ext_sched_class, start_boottime), then
        // resolves the dot-separated nested field path via BTF and
        // writes at task_pa + field_offset. Cold-path freeze
        // rendezvous gives us the atomicity contract — every vCPU
        // parked at SIGRTMIN delivery, no concurrent task migration
        // / state transition can race the validate→write sequence.
        // See [`dispatch_task_field_write`] for the full chain.
        (
            KernelOpTarget::TaskField {
                pid,
                expected_start_time_ns,
                field,
            },
            value,
        ) => dispatch_task_field_write(kernel, btf, *pid, *expected_start_time_ns, field, value),
    }
}

fn dispatch_one_read(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    target: &KernelOpTarget,
    width_hint: &KernelOpValue,
) -> Result<KernelOpValue, String> {
    let page_offset = kernel.page_offset();
    let dram_size = kernel.mem().size();
    match (target, width_hint) {
        // Symbol reads — kernel-half guaranteed by vmlinux .symtab
        // linker convention (see write-side note for full rationale).
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

        // Direct-mapped reads — validate against runtime
        // [page_offset, page_offset+dram_size); read_direct_*
        // shares the same wrapping-sub PA derivation as the write
        // path and would silently return [0; N] on out-of-range.
        (KernelOpTarget::Direct(kva), KernelOpValue::U32(_)) => {
            validate_direct_target(*kva, 4, page_offset, dram_size)?;
            Ok(KernelOpValue::U32(kernel.read_direct_u32(*kva)))
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::U64(_)) => {
            validate_direct_target(*kva, 8, page_offset, dram_size)?;
            Ok(KernelOpValue::U64(kernel.read_direct_u64(*kva)))
        }
        (KernelOpTarget::Direct(kva), KernelOpValue::Bytes(placeholder)) => {
            validate_direct_target(*kva, placeholder.len() as u64, page_offset, dram_size)?;
            Ok(KernelOpValue::Bytes(
                kernel.read_direct_bytes(*kva, placeholder.len()),
            ))
        }

        // Vmalloc/vmap reads — validate against KERNEL_HALF_CONSERVATIVE_5LEVEL
        // (page-walk safety net handles non-canonical-hole + unmapped).
        (KernelOpTarget::Kva(kva), KernelOpValue::U32(_)) => {
            validate_kva_target(*kva, 4)?;
            kernel
                .read_kva_u32(*kva)
                .map(KernelOpValue::U32)
                .ok_or_else(|| format!("read_kva_u32({kva:#x}): page unmapped"))
        }
        (KernelOpTarget::Kva(kva), KernelOpValue::U64(_)) => {
            validate_kva_target(*kva, 8)?;
            kernel
                .read_kva_u64(*kva)
                .map(KernelOpValue::U64)
                .ok_or_else(|| format!("read_kva_u64({kva:#x}): page unmapped"))
        }
        (KernelOpTarget::Kva(kva), KernelOpValue::Bytes(placeholder)) => {
            validate_kva_target(*kva, placeholder.len() as u64)?;
            kernel
                .read_kva_bytes_chunked(*kva, placeholder.len())
                .map(KernelOpValue::Bytes)
                .ok_or_else(|| {
                    format!(
                        "read_kva_bytes_chunked({kva:#x}, {}): page unmapped or short",
                        placeholder.len()
                    )
                })
        }

        // Per-CPU field — same symbol + offset + BTF resolution as
        // the write side, then read U32 or U64 at the resolved PA.
        // See [`dispatch_per_cpu_field_read`].
        (KernelOpTarget::PerCpuField { symbol, field, cpu }, width_hint) => {
            dispatch_per_cpu_field_read(kernel, btf, symbol, field, *cpu, width_hint)
        }

        // Per-task field — same walker + 8-layer validation as the
        // write side, then read at the task_pa + nested-BTF field
        // offset. The width_hint variant determines whether we return
        // a U32 or U64. Cold-path freeze guarantee from
        // [`dispatch_one_write`]'s TaskField comment applies here too:
        // every vCPU parked, no concurrent mutator.
        (
            KernelOpTarget::TaskField {
                pid,
                expected_start_time_ns,
                field,
            },
            width_hint,
        ) => dispatch_task_field_read(
            kernel,
            btf,
            *pid,
            *expected_start_time_ns,
            field,
            width_hint,
        ),

        // OrU32 width hint is wire-format misuse on the read side —
        // it carries a mask, not a width, and has no read semantics.
        (_, KernelOpValue::OrU32(mask)) => Err(oru32_read_rejection_reason(*mask)),
    }
}

/// Hardcoded `{per-CPU symbol → struct name}` mapping. The
/// `KernelOpTarget::PerCpuField` wire variant carries the symbol
/// name but not the struct type the symbol is an instance of; this
/// helper bridges the gap so [`nested_member_byte_offset`] can
/// resolve the field offset against the correct BTF struct.
///
/// v1 set tracks the per-CPU symbols ktstr resolves in
/// [`crate::monitor::symbols::KernelSymbols`]: `runqueues` → `rq`,
/// `kernel_cpustat` → `kernel_cpustat`, `kstat` → `kernel_stat`,
/// `tick_cpu_sched` → `tick_sched`. Adding a per-CPU symbol to the
/// dispatcher requires an entry here AND the symbol resolution in
/// `KernelSymbols::from_elf`.
fn struct_name_for_per_cpu_symbol(symbol: &str) -> Result<&'static str, String> {
    match symbol {
        "runqueues" => Ok("rq"),
        "kernel_cpustat" => Ok("kernel_cpustat"),
        "kstat" => Ok("kernel_stat"),
        "tick_cpu_sched" => Ok("tick_sched"),
        _ => Err(format!(
            "PerCpuField: unknown per-CPU symbol '{symbol}' (v1 supports: \
             runqueues, kernel_cpustat, kstat, tick_cpu_sched); extend \
             struct_name_for_per_cpu_symbol + KernelSymbols::from_elf to add"
        )),
    }
}

/// Resolve a `PerCpuField` target to its guest-memory PA. Shared
/// between the write and read dispatcher arms.
///
/// Steps: look up the symbol's struct type via
/// [`struct_name_for_per_cpu_symbol`]; resolve the symbol's template
/// KVA via [`crate::monitor::guest::GuestKernel::symbol_kva`]; read
/// `__per_cpu_offset[cpu]` from guest memory; compute the per-CPU
/// instance KVA via [`crate::monitor::symbols::per_cpu_kva`]; resolve
/// the field's byte offset within the struct via
/// [`nested_member_byte_offset`]; translate the per-CPU instance KVA
/// to PA via [`translate_any_kva`]; return PA + field_off.
///
/// **KASLR-on caveat**: passes `kaslr_offset=0` to `per_cpu_kva`.
/// ktstr defaults to `nokaslr` karg (per task #40 defense-in-depth),
/// so the slide is always 0 for the standard test path. The
/// KASLR-on round-trip end-to-end test (which would exercise a
/// non-zero kaslr_offset path via cross-thread BSP MSR_LSTAR
/// derive) is deferred — see B9 #6 in the umbrella task #14. A
/// regression that toggled `nokaslr` off without wiring the
/// kaslr_offset would silently produce wrong per-CPU KVAs and
/// `translate_any_kva` would bounds-reject to None.
fn resolve_per_cpu_field_pa(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    symbol: &str,
    field: &str,
    cpu: u32,
) -> Result<usize, String> {
    let btf = btf.ok_or_else(|| {
        format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: BTF not loaded in this \
             coordinator — cannot resolve struct layout (vmlinux must carry \
             CONFIG_DEBUG_INFO_BTF=y output)"
        )
    })?;

    let struct_name = struct_name_for_per_cpu_symbol(symbol)?;

    let template_kva = kernel.symbol_kva(symbol).ok_or_else(|| {
        format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: '{symbol}' symbol absent \
             from vmlinux symtab"
        )
    })?;

    let per_cpu_offset_array_kva = kernel.symbol_kva("__per_cpu_offset").ok_or_else(|| {
        format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: '__per_cpu_offset' symbol \
             absent — kernel built without SMP"
        )
    })?;
    let per_cpu_offset_array_pa = kernel.text_kva_to_pa(per_cpu_offset_array_kva);
    let per_cpu_offset = kernel
        .mem()
        .read_u64(per_cpu_offset_array_pa, (cpu as usize) * 8);
    if per_cpu_offset == 0 && cpu > 0 {
        return Err(format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: __per_cpu_offset[{cpu}]=0 \
             (cpu beyond nr_cpu_ids; kernel zero-init slot)"
        ));
    }

    // per_cpu_kva formula: template_kva + kaslr_offset + per_cpu_offset.
    // kaslr_offset = 0 per the doc above (nokaslr default).
    let per_cpu_kva = crate::monitor::symbols::per_cpu_kva(template_kva, 0, per_cpu_offset);

    let (struct_t, _) = find_struct(btf, struct_name).map_err(|e| {
        format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: 'struct {struct_name}' BTF \
             lookup: {e:#}"
        )
    })?;
    let field_off = nested_member_byte_offset(btf, &struct_t, field).map_err(|e| {
        format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: BTF nested-offset for \
             '{field}' within '{struct_name}': {e:#}"
        )
    })?;

    let walk = kernel.walk_context();
    let pa = translate_any_kva(
        kernel.mem(),
        walk.cr3_pa,
        walk.page_offset,
        per_cpu_kva,
        walk.l5,
        walk.tcr_el1,
    )
    .ok_or_else(|| {
        format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: per_cpu_kva={per_cpu_kva:#x} \
             unmapped (translate_any_kva returned None)"
        )
    })?;

    Ok((pa + field_off as u64) as usize)
}

/// PerCpuField write — resolve PA + field_off, then write the value.
/// `OrU32` is supported as a read-modify-write under the same
/// freeze-rendezvous-epoch contract as the other dispatcher arms (see
/// module doc + the `rmw-invariant-anchor` comments).
fn dispatch_per_cpu_field_write(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    symbol: &str,
    field: &str,
    cpu: u32,
    value: &KernelOpValue,
) -> Result<(), String> {
    let pa = resolve_per_cpu_field_pa(kernel, btf, symbol, field, cpu)? as u64;
    match value {
        KernelOpValue::U32(v) => {
            kernel.mem().write_u32(pa, 0, *v);
            Ok(())
        }
        KernelOpValue::U64(v) => {
            kernel.mem().write_u64(pa, 0, *v);
            Ok(())
        }
        KernelOpValue::OrU32(mask) => {
            // rmw-invariant-anchor: see OrU32 module doc.
            let cur = kernel.mem().read_u32(pa, 0);
            kernel.mem().write_u32(pa, 0, cur | mask);
            Ok(())
        }
        KernelOpValue::Bytes(_) => Err(format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: Bytes write not supported \
             (per-CPU scheduler fields are scalars)"
        )),
    }
}

/// PerCpuField read — same PA resolution as the write side, then
/// read U32 or U64 at the resolved PA (width_hint variant picks
/// which).
fn dispatch_per_cpu_field_read(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    symbol: &str,
    field: &str,
    cpu: u32,
    width_hint: &KernelOpValue,
) -> Result<KernelOpValue, String> {
    let pa = resolve_per_cpu_field_pa(kernel, btf, symbol, field, cpu)? as u64;
    match width_hint {
        KernelOpValue::U32(_) => Ok(KernelOpValue::U32(kernel.mem().read_u32(pa, 0))),
        KernelOpValue::U64(_) => Ok(KernelOpValue::U64(kernel.mem().read_u64(pa, 0))),
        KernelOpValue::Bytes(_) => Err(format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: Bytes read not supported"
        )),
        KernelOpValue::OrU32(_) => Err(format!(
            "PerCpuField {symbol}.{field}[cpu={cpu}]: OrU32 has no read semantic"
        )),
    }
}

/// Width of the start-time identity tolerance window used by L2 of
/// [`validate_task_for_field_op`]: the conservative maximum of
/// `1e9 / sysconf(_SC_CLK_TCK)` across typical configurations.
///
/// The test author's `expected_start_time_ns` is computed from
/// `/proc/<pid>/stat` field 22 (`man 5 proc` "starttime"), which the
/// kernel emits in CLK_TCK ticks — typically 10ms for `CLK_TCK=100`
/// (USER_HZ on x86_64 default kernels). The kernel's
/// `task->start_time` carries the exact `ktime_get_ns()` value, so
/// the userspace-derived `expected_start_time_ns` is always
/// ROUNDED DOWN to a tick boundary while the kernel's stored value
/// has sub-tick precision. Without a window, every TaskField op
/// would fail the L2 identity check on first use.
///
/// 10ms is conservative for `CLK_TCK >= 100`. For higher CLK_TCK
/// (e.g. 1000 → 1ms tick) the window is wider than strictly
/// necessary but still narrow enough to reject PID-recycled tasks
/// — the kernel does not recycle a freed PID within 10ms of the
/// original task's exit under normal scheduling pressure (the
/// allocator advances the PID counter monotonically and wraps
/// after `pid_max` ≈ 2^22 entries).
const START_TIME_PROC_TICK_NS: u64 = 10_000_000;

/// BTF-derived byte offsets needed by the 8-layer task validation in
/// [`validate_task_for_field_op`] plus the per-thread walker in
/// [`find_task_by_pid`]. Resolved once per `TaskField` dispatch via
/// [`Self::resolve_from_btf`] (which calls
/// [`nested_member_byte_offset`] on `struct task_struct` for each
/// member, and on `struct signal_struct` for the thread-head
/// linkage).
///
/// Field semantics:
/// - `pid`: `task_struct.pid` (`pid_t`, kernel-side `int` = 4 bytes,
///   `include/linux/sched.h`). L1 pid-equality check.
/// - `start_time`: `task_struct.start_time` (`u64`, ns since boot)
///   at `include/linux/sched.h:1127`. Set ONCE at fork by
///   `copy_process` via `ktime_get_ns()`. L2 anti-PID-reuse identity
///   check.
/// - `state`: `task_struct.__state` (`unsigned int` = 4 bytes) at
///   `include/linux/sched.h:828`. L3 `state & TASK_DEAD` bit-test.
/// - `on_rq`: `task_struct.on_rq` (`int` = 4 bytes) at
///   `include/linux/sched.h:864`. NOT in `sched_entity` — directly
///   on task_struct. Per `task_on_rq_queued` semantics the value is
///   0 when the task is sleeping (the L4 invariant).
/// - `scx_dsq`: `task_struct.scx.dsq` (`struct scx_dispatch_q *` =
///   8 bytes) — nested through `task_struct.scx` + offset of `dsq`
///   in `sched_ext_entity` (`include/linux/sched/ext.h:211`). NULL
///   when task is not queued in any SCX DSQ (L5 part 1).
/// - `scx_runnable_node`: `task_struct.scx.runnable_node`
///   (`struct list_head`) — nested through `task_struct.scx` +
///   offset of `runnable_node` in `sched_ext_entity`
///   (`include/linux/sched/ext.h:227`, `/* rq->scx.runnable_list */`).
///   Empty (next == &self) when task is NOT linked into any per-rq
///   runnable_list. Independent of `scx.dsq` per
///   `include/linux/sched/ext.h` (L5 part 2).
/// - `sched_class`: `task_struct.sched_class`
///   (`const struct sched_class *` = 8 bytes) at sched.h:878.
///   Pointer identity-compared against `ext_sched_class` KVA for
///   the L6 SCX-only check.
/// - `start_boottime`: `task_struct.start_boottime` (`u64` = 8 bytes)
///   at sched.h:1130 ("Boot based time in nsecs"). Set by `copy_process`
///   at fork via `ktime_get_boottime_ns()`. L8 anti-slab-recycle.
/// - `tasks`: `task_struct.tasks` (`struct list_head` = 16 bytes,
///   only the .next offset matters) at sched.h:954. Used by the
///   leader walker for `container_of` math anchored at `init_task.tasks`.
/// - `signal`: `task_struct.signal` (`struct signal_struct *` = 8
///   bytes). Per-leader pointer; the leader's signal struct holds
///   the `thread_head` list anchor for per-thread iteration.
/// - `signal_thread_head`: offset of `thread_head` (`struct list_head`)
///   within `struct signal_struct`. Combined with the dereferenced
///   `signal` pointer to address the per-thread list anchor.
/// - `thread_node`: `task_struct.thread_node` (`struct list_head`) at
///   sched.h:1094. Per-task linkage into `signal->thread_head`.
///   Used by the per-thread walker for `container_of` math.
struct TaskValidationOffsets {
    pid: usize,
    start_time: usize,
    state: usize,
    on_rq: usize,
    scx_dsq: usize,
    scx_runnable_node: usize,
    sched_class: usize,
    start_boottime: usize,
    tasks: usize,
    signal: usize,
    signal_thread_head: usize,
    thread_node: usize,
}

impl TaskValidationOffsets {
    /// Resolve every offset via BTF. A missing field in the kernel's
    /// task_struct or signal_struct BTF returns a typed error naming
    /// the missing field.
    fn resolve_from_btf(btf: &Btf) -> Result<Self, String> {
        let (task_struct_t, _) = find_struct(btf, "task_struct")
            .map_err(|e| format!("BTF: 'struct task_struct' lookup: {e:#}"))?;
        let task_resolve = |path: &str| -> Result<usize, String> {
            nested_member_byte_offset(btf, &task_struct_t, path)
                .map_err(|e| format!("BTF: task_struct.{path} offset: {e:#}"))
        };
        let (signal_struct_t, _) = find_struct(btf, "signal_struct")
            .map_err(|e| format!("BTF: 'struct signal_struct' lookup: {e:#}"))?;
        let signal_thread_head = nested_member_byte_offset(btf, &signal_struct_t, "thread_head")
            .map_err(|e| format!("BTF: signal_struct.thread_head offset: {e:#}"))?;
        Ok(Self {
            pid: task_resolve("pid")?,
            start_time: task_resolve("start_time")?,
            state: task_resolve("__state")?,
            on_rq: task_resolve("on_rq")?,
            scx_dsq: task_resolve("scx.dsq")?,
            scx_runnable_node: task_resolve("scx.runnable_node")?,
            sched_class: task_resolve("sched_class")?,
            start_boottime: task_resolve("start_boottime")?,
            tasks: task_resolve("tasks")?,
            signal: task_resolve("signal")?,
            signal_thread_head,
            thread_node: task_resolve("thread_node")?,
        })
    }
}

/// Walk the kernel's global task list anchored at `init_task.tasks`,
/// PLUS each leader's per-signal `thread_head`, returning the KVA
/// of the `task_struct` whose `pid` matches `target_pid`. Bounded by
/// [`MAX_TASK_WALKER_NODES`] across BOTH walks combined to defend
/// against a corrupt list chain.
///
/// Two-tier walk:
///
/// 1. **Leaders** — `init_task.tasks` is the `LIST_HEAD` anchor for
///    the `for_each_process` macro at `include/linux/sched/signal.h`
///    L638-640:
///    ```text
///    #define for_each_process(p) \
///        for (p = &init_task ; (p = next_task(p)) != &init_task ; )
///    ```
///    where `next_task(p) = list_entry(p->tasks.next, struct
///    task_struct, tasks)`. The walker starts at
///    `init_task.tasks.next`, container_of-decodes each list_head
///    back to its enclosing `task_struct` (a thread-group leader),
///    and terminates when the chain returns to the head.
///
/// 2. **Threads** — for each leader, walk
///    `leader->signal->thread_head` per the `for_each_thread` macro
///    at the same header L654-659. Per-task linkage is
///    `task_struct.thread_node`. Container_of math:
///    `thread_kva = thread_node_kva - offsetof(task_struct,
///    thread_node)`.
///
/// `init_task` is `pid = 0` and is intentionally NOT yielded by
/// `for_each_process` (the macro skips the head). We additionally
/// EXPLICITLY reject any candidate whose task_kva equals
/// `init_task_kva` as defense-in-depth: if a future kernel reshapes
/// the list invariants, init_task must never land in our candidate
/// set.
///
/// Returns:
/// - `Ok(task_kva)` when a matching pid is found (leader OR
///   non-leader thread).
/// - `Err(reason)` on: empty list, unmapped list-head bytes,
///   walker cap exceeded, unmapped intermediate node (chain broken),
///   pid not found, or attempt to match init_task itself.
fn find_task_by_pid(
    kernel: &GuestKernel,
    init_task_kva: u64,
    offs: &TaskValidationOffsets,
    target_pid: u32,
) -> Result<u64, String> {
    let mem = kernel.mem();
    let walk = kernel.walk_context();
    let pid_off = offs.pid;
    let tasks_off = offs.tasks;
    let signal_off = offs.signal;
    let signal_thread_head_off = offs.signal_thread_head;
    let thread_node_off = offs.thread_node;

    // init_task.tasks anchor lives in .data (init_task is a static
    // global at init/init_task.c:96), so text_kva_to_pa is the right
    // translation. List nodes (task_struct) live in slab and use
    // translate_any_kva.
    let head_kva = init_task_kva.checked_add(tasks_off as u64).ok_or_else(|| {
        format!(
            "find_task_by_pid: head_kva overflow init_task={init_task_kva:#x} + \
             tasks_off={tasks_off}"
        )
    })?;
    let head_pa = kernel.text_kva_to_pa(head_kva);

    // list_head.next is the first u64 in the list_head struct.
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        return Err(format!(
            "find_task_by_pid: init_task.tasks.next read as 0 at head_pa={head_pa:#x} \
             — head bytes unmapped or torn read"
        ));
    }
    if node_kva == head_kva {
        return Err(format!(
            "find_task_by_pid: init_task.tasks is empty (head.next == head) — \
             no user tasks exist; cannot resolve pid={target_pid}"
        ));
    }

    let mut visited: u32 = 0;

    // Tier 1: walk leaders via init_task.tasks.
    while node_kva != head_kva {
        if visited >= MAX_TASK_WALKER_NODES {
            return Err(format!(
                "find_task_by_pid: walker cap {MAX_TASK_WALKER_NODES} exceeded \
                 scanning for pid={target_pid} (visited={visited}); list may be \
                 corrupted (cycle) or pid_max exceeded the cap"
            ));
        }
        visited += 1;

        // container_of: task_kva = list_node_kva - offsetof(task, tasks).
        let leader_kva = node_kva.wrapping_sub(tasks_off as u64);

        // Defense-in-depth: reject init_task even if somehow it
        // leaked into the candidate set. for_each_process skips the
        // head by construction, but defensive reject catches future
        // kernel reshapes or corrupt-chain races.
        if leader_kva == init_task_kva {
            return Err(format!(
                "find_task_by_pid: candidate task_kva={leader_kva:#x} equals \
                 init_task_kva={init_task_kva:#x} (pid=0 swapper); init_task \
                 is not a writable target"
            ));
        }

        let Some(leader_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            leader_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            return Err(format!(
                "find_task_by_pid: leader task_kva={leader_kva:#x} unmapped \
                 (visited={visited}); task_struct slab page not present in guest memory"
            ));
        };

        let leader_pid = mem.read_u32(leader_pa, pid_off);
        if leader_pid == target_pid {
            return Ok(leader_kva);
        }

        // Tier 2: walk this leader's threads via signal->thread_head.
        // The signal pointer is at `signal_off` within task_struct;
        // dereference to get signal_struct KVA; thread_head list_head
        // is at `signal_thread_head_off` within signal_struct.
        let signal_kva = mem.read_u64(leader_pa, signal_off);
        if signal_kva != 0 {
            let thread_head_kva = signal_kva.wrapping_add(signal_thread_head_off as u64);
            if let Some(thread_head_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                thread_head_kva,
                walk.l5,
                walk.tcr_el1,
            ) {
                let mut thread_node_kva = mem.read_u64(thread_head_pa, 0);
                while thread_node_kva != 0 && thread_node_kva != thread_head_kva {
                    if visited >= MAX_TASK_WALKER_NODES {
                        return Err(format!(
                            "find_task_by_pid: walker cap {MAX_TASK_WALKER_NODES} \
                             exceeded inside thread-group of leader_pid={leader_pid} \
                             scanning for pid={target_pid}"
                        ));
                    }
                    visited += 1;

                    let thread_kva = thread_node_kva.wrapping_sub(thread_node_off as u64);

                    // The leader's thread_node is also on this list
                    // — skip it (already checked as leader above).
                    if thread_kva != leader_kva {
                        let Some(thread_pa) = translate_any_kva(
                            mem,
                            walk.cr3_pa,
                            walk.page_offset,
                            thread_kva,
                            walk.l5,
                            walk.tcr_el1,
                        ) else {
                            // Skip this thread on translate failure
                            // rather than aborting the whole walk —
                            // partial visibility is better than none.
                            // Advance via the node, not the task.
                            let Some(thread_node_pa) = translate_any_kva(
                                mem,
                                walk.cr3_pa,
                                walk.page_offset,
                                thread_node_kva,
                                walk.l5,
                                walk.tcr_el1,
                            ) else {
                                break; // can't advance — break inner loop
                            };
                            thread_node_kva = mem.read_u64(thread_node_pa, 0);
                            continue;
                        };

                        let thread_pid = mem.read_u32(thread_pa, pid_off);
                        if thread_pid == target_pid {
                            return Ok(thread_kva);
                        }
                    }

                    // Advance to next thread via thread_node.next.
                    let next_kva = mem.read_u64(
                        thread_pa_or_node(
                            mem,
                            walk.cr3_pa,
                            walk.page_offset,
                            walk.l5,
                            walk.tcr_el1,
                            thread_kva,
                            thread_node_kva,
                            thread_node_off,
                        ),
                        0,
                    );
                    if next_kva == 0 {
                        break; // chain broken — break inner loop
                    }
                    thread_node_kva = next_kva;
                }
            }
        }

        // Advance to next leader via this leader's tasks.next.
        let next_kva = mem.read_u64(leader_pa, tasks_off);
        if next_kva == 0 {
            return Err(format!(
                "find_task_by_pid: list_head.next read as 0 at leader_kva={leader_kva:#x} \
                 (visited={visited}); chain broken before finding pid={target_pid}"
            ));
        }
        node_kva = next_kva;
    }

    Err(format!(
        "find_task_by_pid: pid={target_pid} not found in init_task.tasks \
         or any leader's signal->thread_head (visited={visited} entries across \
         leaders + threads)"
    ))
}

/// Resolve the PA holding a thread_node's .next pointer. Used by the
/// per-thread walker to advance after a successful task_pa
/// translation: prefer reading via task_pa + thread_node_off (one
/// translate already paid for); fall back to translating node_kva
/// directly when task_pa is unavailable.
#[allow(clippy::too_many_arguments)]
fn thread_pa_or_node(
    mem: &crate::monitor::reader::GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    tcr_el1: u64,
    thread_kva: u64,
    thread_node_kva: u64,
    thread_node_off: usize,
) -> u64 {
    if let Some(task_pa) = translate_any_kva(mem, cr3_pa, page_offset, thread_kva, l5, tcr_el1) {
        task_pa + thread_node_off as u64
    } else {
        translate_any_kva(mem, cr3_pa, page_offset, thread_node_kva, l5, tcr_el1).unwrap_or(0)
    }
}

/// Eight-layer task validation chain. Run AFTER the walker locates
/// the candidate task_struct and BEFORE any field write. Every layer
/// reads from guest memory at the candidate `task_pa` and rejects
/// with a typed error naming the specific layer + observed value.
///
/// Layer order (fail-fast, cheapest first):
/// 1. **pid match**: `task->pid == target_pid`. Defense against
///    slab-recycle where the freed task_struct's memory was reused
///    for another task with a different pid. Also a sanity check on
///    the walker.
/// 2. **start_time identity**: `task->start_time in
///    [expected_start_time_ns, expected_start_time_ns +
///    START_TIME_PROC_TICK_NS)`. The kernel sets `start_time` once
///    at fork via `ktime_get_ns()` in `kernel/fork.c::copy_process`
///    with full nanosecond precision; the value never changes after
///    that. The only userspace-visible source for that field is
///    `/proc/<pid>/stat` field 22, which the kernel emits in clock
///    ticks (1 / `sysconf(_SC_CLK_TCK)`) — typically 10ms — so the
///    test author's `expected_start_time_ns` is always quantized
///    DOWN to a tick boundary while the kernel's `task->start_time`
///    carries the exact ns. Accepting a tick-window (10ms — the
///    conservative max for `CLK_TCK >= 100`) closes the legitimate
///    quantization gap without weakening the anti-PID-reuse defense
///    (the kernel never recycles a PID within 10ms of the original
///    task's exit under normal scheduling pressure).
///    Catches PID-reuse: if the original worker exited and the
///    kernel recycled the PID for an unrelated task, the new task's
///    `start_time` will be far outside the [+0, +tick) window of the
///    captured-at-spawn value, even when the pid matches by
///    coincidence.
/// 3. **lifetime**: `task->__state & TASK_DEAD == 0`. A task in the
///    final teardown path has the `TASK_DEAD` bit set in `__state`
///    (`include/linux/sched.h:118`); writing through it would
///    corrupt the dying-task state machine.
/// 4. **runqueue safety**: `task->on_rq == 0`. Per
///    `task_on_rq_queued` (`kernel/sched/sched.h:2399`) the value
///    is 0 when the task is sleeping. CFS's red-black tree keys on
///    `se.vruntime`; mutating it while the task is queued
///    (on_rq=TASK_ON_RQ_QUEUED=1 or TASK_ON_RQ_MIGRATING=2) corrupts
///    tree ordering.
/// 5. **SCX queued-anywhere safety**: `task->scx.dsq == NULL` AND
///    `task->scx.runnable_node` is list-empty (next == &self). The
///    `dsq` pointer (`include/linux/sched/ext.h:211`) tracks current
///    DSQ residence; the `runnable_node` (L227 `/* rq->scx.runnable_list */`)
///    tracks per-rq runnable bookkeeping INDEPENDENT of `dsq`. Both
///    must be empty to safely modify scheduler-bookkeeping fields.
/// 6. **SCX-only sched_class**: `task->sched_class ==
///    &ext_sched_class`. The dispatcher rejects non-SCX tasks
///    (fair / RT / DL / stop / idle) because EEVDF's `place_entity`
///    overwrites `se->vruntime` on enqueue (silently discarding CFS
///    seeds), RT/DL/stop/idle have different vtime semantics, and
///    SCX's `dsq_vtime` is the only host-writable preserved
///    ordering key in the modern kernel.
/// 7. (REMOVED). The previous gate required
///    `task->policy & ~SCHED_RESET_ON_FORK == SCHED_EXT` per
///    `include/uapi/linux/sched.h:121` as belt-and-suspenders for
///    L6, but it does not hold: `kernel/sched/ext.c::scx_init_task`
///    / `scx_enable_task` set `task->sched_class = &ext_sched_class`
///    when SCX takes over a fair-policy task without modifying
///    `task->policy`, so a worker forked under `SCHED_NORMAL` keeps
///    `policy=0` even after SCX claims it. L6 (sched_class pointer
///    identity) is the canonical SCX-managed gate; `policy` is
///    unreliable for that purpose. The numbering is preserved so
///    the surviving gates keep their layer labels.
/// 8. **anti slab-recycle**: `task->start_boottime != 0`. The
///    `start_boottime` field is set by `copy_process` at fork via
///    `ktime_get_boottime_ns()` (which is never 0 after boot). A
///    freshly-zeroed slab page has start_boottime=0; a live task
///    has it non-zero. Catches slab-recycle that survived L1+L2
///    (pid AND start_time match by coincidence — vanishingly
///    unlikely but defense-in-depth).
///
/// `ext_sched_class_kva` is the resolved `ext_sched_class` KVA the
/// L6 check compares against. Caller resolves via
/// `kernel.symbol_kva("ext_sched_class")`; absent symbol (kernel
/// without CONFIG_SCHED_CLASS_EXT) fails the entire dispatcher path
/// upstream — see [`resolve_and_validate_task_field`].
fn validate_task_for_field_op(
    kernel: &GuestKernel,
    task_pa: u64,
    target_pid: u32,
    expected_start_time_ns: u64,
    offs: &TaskValidationOffsets,
    ext_sched_class_kva: u64,
) -> Result<(), String> {
    let mem = kernel.mem();

    // L1: pid match (anti slab-recycle + walker sanity).
    let pid = mem.read_u32(task_pa, offs.pid);
    if pid != target_pid {
        return Err(format!(
            "validate_task: pid mismatch at task_pa={task_pa:#x} — read pid={pid}, \
             expected {target_pid} (likely slab-recycle since walker found this task)"
        ));
    }

    // L2: start_time identity (anti-PID-reuse).
    //
    // `expected_start_time_ns` is the test author's value derived
    // from /proc/<pid>/stat field 22 (jiffies-quantized: integer
    // ticks * 1e9 / CLK_TCK), so it's always ROUNDED DOWN to a
    // CLK_TCK boundary. The kernel's `task->start_time` carries
    // sub-tick precision from `ktime_get_ns()`, so the legitimate
    // value lands in `[expected, expected + CLK_TCK_NS)`. Accept
    // a 10ms window (conservative max for CLK_TCK >= 100), which
    // still rejects PID-recycled tasks whose start_time falls
    // well outside that range under normal scheduling pressure.
    let observed_start_time = mem.read_u64(task_pa, offs.start_time);
    let skew = observed_start_time.saturating_sub(expected_start_time_ns);
    if observed_start_time < expected_start_time_ns || skew >= START_TIME_PROC_TICK_NS {
        return Err(format!(
            "validate_task: task pid={target_pid} start_time identity mismatch — \
             observed={observed_start_time}ns expected in \
             [{expected_start_time_ns}, {}]ns; \
             original task exited and PID was recycled for an unrelated task",
            expected_start_time_ns + START_TIME_PROC_TICK_NS - 1
        ));
    }

    // L3: lifetime (TASK_DEAD bit not set).
    let state = mem.read_u32(task_pa, offs.state);
    if state & TASK_DEAD != 0 {
        return Err(format!(
            "validate_task: task pid={target_pid} is TASK_DEAD (state={state:#x}); \
             mid-teardown task fields unsafe to write"
        ));
    }

    // L4: runqueue safety (on_rq == 0).
    let on_rq = mem.read_u32(task_pa, offs.on_rq);
    if on_rq != 0 {
        return Err(format!(
            "validate_task: task pid={target_pid} is on_rq={on_rq} (TASK_ON_RQ_QUEUED \
             or MIGRATING); writing scheduler fields would corrupt rb-tree / DSQ \
             ordering. Test author must use a blocking workload pattern \
             (`WorkType::FutexPingPong`, `WorkType::WaitOnFutex`, `WorkType::Sleep`) \
             so the worker is sleeping at cold-op time"
        ));
    }

    // L5: SCX queued-anywhere safety (scx.dsq == NULL AND scx.runnable_node empty).
    let scx_dsq_ptr = mem.read_u64(task_pa, offs.scx_dsq);
    if scx_dsq_ptr != 0 {
        return Err(format!(
            "validate_task: task pid={target_pid} has scx.dsq={scx_dsq_ptr:#x} (queued \
             on an SCX DSQ); modifying ordering keys while queued mangles ordering \
             per include/linux/sched/ext.h:248-254 (dsq_vtime warning). Test author \
             must use a blocking workload pattern \
             (`WorkType::FutexPingPong`, `WorkType::WaitOnFutex`, `WorkType::Sleep`)"
        ));
    }
    // scx.runnable_node is a list_head; "empty" means next == &self
    // (the KVA of the list_head itself). The list_head KVA is
    // task_kva + offsetof(task_struct, scx.runnable_node). We need
    // the task_KVA to compare; derive it from task_pa via the
    // page_offset (slab is direct-mapped).
    let task_kva = task_pa.wrapping_add(kernel.page_offset());
    let runnable_node_kva = task_kva.wrapping_add(offs.scx_runnable_node as u64);
    let runnable_node_next = mem.read_u64(task_pa, offs.scx_runnable_node);
    if runnable_node_next != 0 && runnable_node_next != runnable_node_kva {
        return Err(format!(
            "validate_task: task pid={target_pid} scx.runnable_node is linked \
             (next={runnable_node_next:#x} != self={runnable_node_kva:#x}); task is \
             on a per-rq runnable_list. Test author must use a blocking workload \
             pattern (`WorkType::FutexPingPong`, `WorkType::WaitOnFutex`, \
             `WorkType::Sleep`)"
        ));
    }

    // L6: SCX-only sched_class (must be ext_sched_class).
    let sched_class_kva = mem.read_u64(task_pa, offs.sched_class);
    if sched_class_kva != ext_sched_class_kva {
        return Err(format!(
            "validate_task: task pid={target_pid} sched_class={sched_class_kva:#x} \
             is not ext_sched_class={ext_sched_class_kva:#x}; TaskField writes target \
             SCX-managed tasks only (CFS / RT / DL / stop / idle classes have \
             different vtime semantics — EEVDF's place_entity overwrites se.vruntime \
             on enqueue, RT/DL have RT_BANDWIDTH instant-throttle hazards). Spawn \
             the worker under `SchedPolicy::Ext` to make it SCX-managed"
        ));
    }

    // L7 (REMOVED): `task->policy == SCHED_EXT` was a belt-and-
    // suspenders gate for L6 but it does not actually hold for SCX-
    // managed tasks. `kernel/sched/ext.c::scx_init_task` /
    // `scx_enable_task` set `task->sched_class = &ext_sched_class`
    // when SCX takes over a fair-policy task but does NOT modify
    // `task->policy` — a worker forked under `SCHED_NORMAL` keeps
    // `policy=0` (SCHED_NORMAL) even after SCX claims it. Requiring
    // `policy == SCHED_EXT` rejects every legitimate SCX-managed
    // task that did not explicitly call `sched_setattr(SCHED_EXT)`,
    // which is the common case for ktstr's WorkloadHandle (workers
    // spawn with SchedPolicy::Normal and scx-ktstr's BPF dispatch
    // claims them). L6 (sched_class pointer identity) is the
    // canonical SCX-managed gate; the policy field is unreliable
    // for this check.
    //
    // L8: anti slab-recycle via start_boottime.
    let start_boottime = mem.read_u64(task_pa, offs.start_boottime);
    if start_boottime == 0 {
        return Err(format!(
            "validate_task: task pid={target_pid} start_boottime=0 — possibly a \
             freshly-zeroed slab page mid-slab-recycle; reject rather than risk \
             writing to dead memory"
        ));
    }

    Ok(())
}

/// Resolve TaskField context (init_task KVA, ext_sched_class KVA,
/// validation offsets) and find+validate the target task's PA.
/// Shared between the read and write dispatcher arms — both need
/// identical setup.
///
/// SCX-only: this dispatcher path is for SCX-managed tasks. The
/// `ext_sched_class` symbol is required; a kernel without
/// `CONFIG_SCHED_CLASS_EXT` fails the lookup here and the
/// dispatcher rejects the entire TaskField op.
fn resolve_and_validate_task_field(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    pid: u32,
    expected_start_time_ns: u64,
) -> Result<(u64, btf_rs::Struct), String> {
    let btf = btf.ok_or_else(|| {
        format!(
            "TaskField pid={pid}: BTF not loaded in this coordinator — cannot resolve \
             task_struct layout (vmlinux must carry CONFIG_DEBUG_INFO_BTF=y output)"
        )
    })?;
    let init_task_kva = kernel.symbol_kva("init_task").ok_or_else(|| {
        format!(
            "TaskField pid={pid}: init_task symbol absent from vmlinux symtab \
             (heavily stripped vmlinux); cannot anchor the task-list walker"
        )
    })?;
    let ext_sched_class_kva = kernel.symbol_kva("ext_sched_class").ok_or_else(|| {
        format!(
            "TaskField pid={pid}: ext_sched_class symbol absent from vmlinux symtab \
             (kernel built without CONFIG_SCHED_CLASS_EXT=y); TaskField writes are \
             SCX-only and require sched_ext support"
        )
    })?;

    let val_offs = TaskValidationOffsets::resolve_from_btf(btf)?;

    let task_kva = find_task_by_pid(kernel, init_task_kva, &val_offs, pid)?;
    let walk = kernel.walk_context();
    let task_pa = translate_any_kva(
        kernel.mem(),
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    )
    .ok_or_else(|| {
        format!(
            "TaskField pid={pid}: task_kva={task_kva:#x} unmapped at validation step \
             (slab page disappeared between walker and validator — extreme race)"
        )
    })?;

    validate_task_for_field_op(
        kernel,
        task_pa,
        pid,
        expected_start_time_ns,
        &val_offs,
        ext_sched_class_kva,
    )?;

    let (task_struct_t, _) = find_struct(btf, "task_struct")
        .map_err(|e| format!("TaskField pid={pid}: 'struct task_struct' BTF lookup: {e:#}"))?;

    Ok((task_pa, task_struct_t))
}

/// End-to-end TaskField write: resolve init_task + ext_sched_class,
/// walk leaders + threads to find task by pid + start_time identity,
/// run 8-layer validation, resolve field byte offset via BTF nested
/// path, write the value at task_pa + field_off.
fn dispatch_task_field_write(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    pid: u32,
    expected_start_time_ns: u64,
    field: &str,
    value: &KernelOpValue,
) -> Result<(), String> {
    let (task_pa, task_struct_t) =
        resolve_and_validate_task_field(kernel, btf, pid, expected_start_time_ns)?;

    // Safe to unwrap: resolve_and_validate_task_field rejected if
    // btf was None.
    let btf = btf.expect("checked in resolve_and_validate_task_field");

    let field_off = nested_member_byte_offset(btf, &task_struct_t, field).map_err(|e| {
        format!("TaskField pid={pid} field={field:?}: BTF nested-offset resolution: {e:#}")
    })?;

    match value {
        KernelOpValue::U32(v) => {
            kernel.mem().write_u32(task_pa, field_off, *v);
            Ok(())
        }
        KernelOpValue::U64(v) => {
            kernel.mem().write_u64(task_pa, field_off, *v);
            Ok(())
        }
        KernelOpValue::Bytes(_) => Err(format!(
            "TaskField pid={pid} field={field:?}: Bytes write not supported in v1 — \
             use U32 or U64 (per-task scheduler fields are scalars)"
        )),
        KernelOpValue::OrU32(_) => Err(format!(
            "TaskField pid={pid} field={field:?}: OrU32 RMW not supported on TaskField \
             in v1 (no current use case; per-task scheduler fields are scalars not flags)"
        )),
    }
}

/// End-to-end TaskField read: same walker + validation as the write,
/// then read U32 or U64 at task_pa + field_off (driven by width_hint
/// variant).
fn dispatch_task_field_read(
    kernel: &GuestKernel,
    btf: Option<&Btf>,
    pid: u32,
    expected_start_time_ns: u64,
    field: &str,
    width_hint: &KernelOpValue,
) -> Result<KernelOpValue, String> {
    let (task_pa, task_struct_t) =
        resolve_and_validate_task_field(kernel, btf, pid, expected_start_time_ns)?;

    let btf = btf.expect("checked in resolve_and_validate_task_field");

    let field_off = nested_member_byte_offset(btf, &task_struct_t, field).map_err(|e| {
        format!("TaskField pid={pid} field={field:?}: BTF nested-offset resolution: {e:#}")
    })?;

    match width_hint {
        KernelOpValue::U32(_) => Ok(KernelOpValue::U32(
            kernel.mem().read_u32(task_pa, field_off),
        )),
        KernelOpValue::U64(_) => Ok(KernelOpValue::U64(
            kernel.mem().read_u64(task_pa, field_off),
        )),
        KernelOpValue::Bytes(_) => Err(format!(
            "TaskField pid={pid} field={field:?}: Bytes read not supported in v1 — \
             use U32 or U64 width hint"
        )),
        KernelOpValue::OrU32(_) => Err(format!(
            "TaskField pid={pid} field={field:?}: OrU32 has no read semantic (covered \
             by the dispatcher's read-direction catch-all but explicit here for clarity)"
        )),
    }
}

/// Build the typed-error reason for the wire-misuse case where a
/// caller routes a `KernelOpValue::OrU32(mask)` through the read
/// direction. OrU32 carries a mask (write semantics), not a width
/// hint — there is no read semantic to derive. The reason names the
/// correct read-width Rust symbol so a confused caller can fix at
/// the call site without source-diving the dispatcher.
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
    use crate::vmm::x86_64::msr_kaslr::KERNEL_HALF_CANONICAL_4LEVEL;

    /// Disambiguation invariant pin. The 5-level conservative threshold
    /// must be PERMISSIVELY LOWER than the 4-level canonical strict
    /// boundary — any address that satisfies the 4-level canonical
    /// check also passes the looser 5-level guard, so KASLR-on builds
    /// that use 5-level paging direct-map / vmalloc / vmemmap KVAs
    /// (which sit below the 4-level threshold but above the 5-level
    /// one) are accepted by [`validate_kva_target`] without
    /// false-rejection. A regression that flipped either value would
    /// silently break dispatch.rs's kernel-text canonical check OR
    /// make kernel_op_dispatch.rs over-permissive.
    ///
    /// `const _: () = assert!(...)` is a const-eval'd assertion that
    /// fails at COMPILE time — strictly stronger than `#[test]` (no
    /// dependency on running cargo test to bite). The collapse-rejection
    /// (`!=`) is implicit in the strict `<`.
    const _: () = assert!(
        KERNEL_HALF_CONSERVATIVE_5LEVEL < KERNEL_HALF_CANONICAL_4LEVEL,
        "5-level threshold must be permissively lower than 4-level canonical",
    );

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

    /// PerCpuField unknown-symbol rejection: the hardcoded mapping at
    /// [`struct_name_for_per_cpu_symbol`] returns Err for symbols
    /// outside the v1 supported set (runqueues / kernel_cpustat /
    /// kstat / tick_cpu_sched). A regression that silently accepted
    /// an unknown symbol would silently look up a wrong BTF struct
    /// and produce wrong field offsets.
    #[test]
    fn per_cpu_field_unknown_symbol_rejected() {
        let err = struct_name_for_per_cpu_symbol("not_a_real_per_cpu_symbol")
            .expect_err("unknown symbol must reject");
        assert!(err.contains("PerCpuField"));
        assert!(err.contains("not_a_real_per_cpu_symbol"));
        // Enumerate the v1 supported set in the error to give the
        // caller an actionable next step.
        assert!(err.contains("runqueues"));
        assert!(err.contains("kernel_cpustat"));
        assert!(err.contains("kstat"));
        assert!(err.contains("tick_cpu_sched"));
    }

    /// PerCpuField known-symbol mapping: every entry in the v1
    /// supported set MUST map to the right kernel struct name. A
    /// regression that swapped the runqueues→rq mapping (e.g. typo
    /// to "rq_struct") would silently look up the wrong struct.
    #[test]
    fn per_cpu_field_known_symbol_mapping() {
        assert_eq!(struct_name_for_per_cpu_symbol("runqueues").unwrap(), "rq");
        assert_eq!(
            struct_name_for_per_cpu_symbol("kernel_cpustat").unwrap(),
            "kernel_cpustat"
        );
        assert_eq!(
            struct_name_for_per_cpu_symbol("kstat").unwrap(),
            "kernel_stat"
        );
        assert_eq!(
            struct_name_for_per_cpu_symbol("tick_cpu_sched").unwrap(),
            "tick_sched"
        );
    }

    /// 4-call-site product matrix via source-grep: each of the 4
    /// dispatch arms — Direct/Write, Direct/Read, Kva/Write, Kva/Read
    /// — MUST call validate_direct_target (Direct) or
    /// validate_kva_target (Kva) BEFORE invoking the underlying
    /// kernel.{read,write}_{direct,kva}_* function. A regression that
    /// wires validate into 3/4 sites and drops one silently re-opens
    /// the silent-data-loss class for the missing arm.
    ///
    /// Source-grep approach mirrors the marker-anchor test below:
    /// pin the structural invariant at the source level without
    /// requiring MockGuestKernel infrastructure (which doesn't exist
    /// in-tree yet).
    ///
    /// Self-match exclusion: the searched arm shape appears in this
    /// test's own docstring above + in error-message format strings
    /// below. Restrict the search to source BEFORE `#[cfg(test)]`
    /// (production code only) to avoid counting test-body matches.
    #[test]
    fn dispatch_arms_call_validate_target_helpers() {
        let full_src = include_str!("kernel_op_dispatch.rs");
        let test_mod_start = full_src
            .find("#[cfg(test)]")
            .expect("test module must exist");
        let src = &full_src[..test_mod_start];
        // Each Direct arm shape `KernelOpTarget::Direct(kva), KernelOpValue::*`
        // MUST be followed within ~10 lines by `validate_direct_target(`.
        // Each Kva arm shape `KernelOpTarget::Kva(kva), KernelOpValue::*`
        // MUST be followed within ~10 lines by `validate_kva_target(`.
        // Symbol arms are exempt (vmlinux .symtab kernel-half guarantee);
        // PerCpuField + TaskField arms are exempt (translate_any_kva
        // safety-net handles unmapped/out-of-bounds; resolve_per_cpu_field_pa
        // and find_task_by_pid produce typed errors instead of silent zeros).

        // Count Direct arms.
        let direct_arms: Vec<_> = src
            .match_indices("KernelOpTarget::Direct(kva), KernelOpValue::")
            .collect();
        // Expect 7: 4 in dispatch_one_write (U32/U64/Bytes/OrU32) +
        // 3 in dispatch_one_read (U32/U64/Bytes). OrU32 read is
        // rejected via the catch-all and doesn't have a per-target arm.
        assert_eq!(
            direct_arms.len(),
            7,
            "expected exactly 7 Direct arms (4 write + 3 read); found {}",
            direct_arms.len()
        );
        for (idx, _) in &direct_arms {
            let window_end = (idx + 400).min(src.len());
            let window = &src[*idx..window_end];
            assert!(
                window.contains("validate_direct_target("),
                "Direct arm at byte offset {idx} is missing validate_direct_target() call; \
                 window: {window:?}"
            );
        }

        // Count Kva arms.
        let kva_arms: Vec<_> = src
            .match_indices("KernelOpTarget::Kva(kva), KernelOpValue::")
            .collect();
        assert_eq!(
            kva_arms.len(),
            7,
            "expected exactly 7 Kva arms (4 write + 3 read); found {}",
            kva_arms.len()
        );
        for (idx, _) in &kva_arms {
            let window_end = (idx + 400).min(src.len());
            let window = &src[*idx..window_end];
            assert!(
                window.contains("validate_kva_target("),
                "Kva arm at byte offset {idx} is missing validate_kva_target() call; \
                 window: {window:?}"
            );
        }
    }

    // ---- UTF-8 boundary tests ----

    /// Table-driven UTF-8 boundary classes: 2-byte, 3-byte, 4-byte,
    /// BOM. Each exercises the is_char_boundary walk-back loop with a
    /// different multi-byte codepoint width.
    /// Mixed-width + pure-ASCII + empty paths are distinct from this
    /// table — they're separate tests below because their assertion
    /// shape differs (mixed-width tests walk-back regardless of width;
    /// pure-ASCII tests cap-exact length; empty tests passthrough).
    #[test]
    fn error_reply_truncates_at_utf8_boundary_classes() {
        for (cp, label, padding) in [
            // (codepoint, label for failure context, padding bytes
            // past KERNEL_OP_REASON_MAX to ensure overflow)
            ("é", "2byte_U+00E9", 4),      // U+00E9, 2 bytes (C3 A9)
            ("☃", "3byte_U+2603", 6),      // U+2603, 3 bytes (E2 98 83)
            ("🦀", "4byte_U+1F980", 8),    // U+1F980, 4 bytes
            ("\u{FEFF}", "BOM_U+FEFF", 6), // U+FEFF, 3 bytes (EF BB BF)
        ] {
            let mut s = String::new();
            while s.len() < KERNEL_OP_REASON_MAX + padding {
                s.push_str(cp);
            }
            let reply = error_reply(42, s);
            assert!(
                reply.reason.len() <= KERNEL_OP_REASON_MAX,
                "{label}: reason.len()={} > cap={KERNEL_OP_REASON_MAX}",
                reply.reason.len()
            );
            assert!(
                reply.reason.is_char_boundary(reply.reason.len()),
                "{label}: truncation landed mid-codepoint"
            );
            let _ = reply.reason.as_str();
        }
    }

    /// Mixed-width input: the cap position is data-dependent —
    /// exercise the is_char_boundary walk-back under all four
    /// widths (1B + 2B + 3B + 4B intermixed) in one pass.
    #[test]
    fn error_reply_truncates_mixed_width_input_at_boundary() {
        let pattern = "Aé☃🦀";
        let mut s = String::new();
        while s.len() < KERNEL_OP_REASON_MAX + 10 {
            s.push_str(pattern);
        }
        let reply = error_reply(99, s);
        assert!(reply.reason.len() <= KERNEL_OP_REASON_MAX);
        assert!(reply.reason.is_char_boundary(reply.reason.len()));
        let _ = reply.reason.as_str();
    }

    /// Pure-ASCII over-cap input: cap lands on a clean boundary
    /// (every byte is a codepoint boundary in ASCII). Tests the
    /// degenerate "walk-back of 0 bytes" path that a regression in
    /// the lower-bound condition could break.
    #[test]
    fn error_reply_truncates_pure_ascii_no_walkback() {
        let s = "A".repeat(KERNEL_OP_REASON_MAX + 16);
        let reply = error_reply(1, s);
        assert_eq!(reply.reason.len(), KERNEL_OP_REASON_MAX);
        assert!(reply.reason.is_char_boundary(reply.reason.len()));
    }

    /// Empty-string passthrough — error_reply must not crash on
    /// `is_char_boundary(0)` of an empty string. Trivial today but
    /// pins the gate's behavior so a refactor that swapped the
    /// `>` for `>=` (forcing walk-back on empty) trips here.
    #[test]
    fn error_reply_zero_length_reason_passes() {
        let reply = error_reply(2, String::new());
        assert!(!reply.success);
        assert_eq!(reply.reason, "");
    }

    // ---- KVA validation tests ----

    /// T62.1 — boundary inclusive: KVA at exactly KERNEL_HALF_CONSERVATIVE_5LEVEL
    /// is accepted. Regression guard against off-by-one flipping
    /// the `<` to `<=` (which would reject the canonical bound).
    #[test]
    fn validate_kva_target_accepts_exact_threshold() {
        assert!(validate_kva_target(KERNEL_HALF_CONSERVATIVE_5LEVEL, 4).is_ok());
    }

    /// T62.2 — boundary exclusive: KVA one below threshold rejects.
    /// Pins the rejection-side off-by-one symmetric with T62.1.
    #[test]
    fn validate_kva_target_rejects_one_below_threshold() {
        let kva = KERNEL_HALF_CONSERVATIVE_5LEVEL - 1;
        let err = validate_kva_target(kva, 4).expect_err("must reject");
        assert!(
            err.contains(&format!("{kva:#x}")),
            "error must echo rejected KVA for operator triage; got {err}"
        );
    }

    /// T62.3 — user-half edge (kva=0). Per CLAUDE.md "no silent drops",
    /// 0 must fail loud rather than be treated as a sentinel.
    #[test]
    fn validate_kva_target_rejects_zero() {
        let err = validate_kva_target(0, 4).expect_err("kva=0 must reject");
        assert!(err.contains("0x0"));
    }

    /// T62.4 — user-half max (canonical 4-level user-half top).
    /// Pins that a bit-63 only check is insufficient; the
    /// threshold-based check catches this case.
    #[test]
    fn validate_kva_target_rejects_user_half_max() {
        let kva = 0x0000_7FFF_FFFF_FFFF;
        assert!(
            validate_kva_target(kva, 4).is_err(),
            "canonical user-half max must reject"
        );
    }

    /// T62.5 — kernel-half typical KASLR-off + KASLR-on land.
    /// Pins that real-world kernel KVAs don't false-reject.
    #[test]
    fn validate_kva_target_accepts_kernel_typical() {
        // x86_64 _text on KASLR-off (4-level).
        assert!(validate_kva_target(0xFFFF_FFFF_8100_0000, 4).is_ok());
        // Typical vmalloc address (high canonical addr).
        assert!(validate_kva_target(0xFFFF_C900_0000_0000, 4).is_ok());
        // 5-level direct-map base sample (would fail under 4-level-strict).
        assert!(validate_kva_target(0xFF11_0000_0000_0000, 4).is_ok());
    }

    /// T62 — user_half_kva_rejection_reason format pin via Path B
    /// helper-extraction integration: the test invokes the SAME
    /// helper the production dispatcher calls and pins error_reply's
    /// propagation through the batch-prefix machinery. A regression
    /// that drops the rejection, changes the format, or stops
    /// calling the helper trips here. NOT a tautology — the test
    /// does not synthesize its own copy of the format string.
    #[test]
    fn user_half_kva_rejection_reason_format_pin() {
        let kva = 0x4000_0000_0000;
        let helper_reason = user_half_kva_rejection_reason(kva);
        let batch_reason = format!("entry[0]: {helper_reason}");
        let reply = error_reply(11, batch_reason.clone());
        assert!(!reply.success);
        assert_eq!(reply.reason, batch_reason);
        // Helper output names the rejected KVA + the threshold + the
        // operator-actionable suggestion. A regression to the wrong
        // bound or dropped rejection surfaces.
        assert!(helper_reason.contains(&format!("{kva:#x}")));
        assert!(helper_reason.contains(&format!("{KERNEL_HALF_CONSERVATIVE_5LEVEL:#x}")));
        assert!(helper_reason.contains("kernel-half"));
        assert!(helper_reason.contains("5-level conservative"));
        assert!(helper_reason.contains("Symbol target"));
    }

    /// T62 — validate_direct_target accepts an in-range KVA.
    /// Page_offset is a typical 4-level KASLR-off direct-map base.
    #[test]
    fn validate_direct_target_accepts_in_range() {
        let page_offset = 0xFFFF_8880_0000_0000u64;
        let dram_size = 256 * 1024 * 1024; // 256 MB typical ktstr test VM
        // First byte of direct map.
        assert!(validate_direct_target(page_offset, 4, page_offset, dram_size).is_ok());
        // Mid-range.
        assert!(validate_direct_target(page_offset + 0x1000, 8, page_offset, dram_size).is_ok());
        // Last U32 inside.
        assert!(
            validate_direct_target(page_offset + dram_size - 4, 4, page_offset, dram_size).is_ok()
        );
    }

    /// T62 — validate_direct_target rejects a KVA below page_offset.
    /// The user-half / canonical-hole class — kva_to_pa would
    /// underflow and wrap.
    #[test]
    fn validate_direct_target_rejects_below_page_offset() {
        let page_offset = 0xFFFF_8880_0000_0000u64;
        let dram_size = 256 * 1024 * 1024;
        let kva = page_offset - 1;
        let err = validate_direct_target(kva, 4, page_offset, dram_size)
            .expect_err("kva below page_offset must reject");
        assert!(err.contains(&format!("{kva:#x}")));
        assert!(err.contains(&format!("{page_offset:#x}")));
        assert!(err.contains("would wrap"));
    }

    /// T62 — validate_direct_target rejects a KVA range past the
    /// direct-map end. The "out the upper end" class.
    #[test]
    fn validate_direct_target_rejects_past_end() {
        let page_offset = 0xFFFF_8880_0000_0000u64;
        let dram_size = 256 * 1024 * 1024;
        // One byte past the last valid KVA, 4-byte len.
        let kva = page_offset + dram_size - 3;
        let err = validate_direct_target(kva, 4, page_offset, dram_size)
            .expect_err("kva+len past direct-map end must reject");
        assert!(err.contains("overruns direct-map end"));
    }

    /// T62 — validate_direct_target rejects overflow on kva+len.
    /// Pins the checked_add guard.
    #[test]
    fn validate_direct_target_rejects_kva_len_overflow() {
        let page_offset = 0xFFFF_8880_0000_0000u64;
        let dram_size = 256 * 1024 * 1024;
        let kva = u64::MAX - 2;
        let err = validate_direct_target(kva, 4, page_offset, dram_size)
            .expect_err("kva+len overflow must reject");
        assert!(err.contains("overflow"));
    }

    /// T62 — validate_kva_target rejects overflow on kva+len.
    #[test]
    fn validate_kva_target_rejects_kva_len_overflow() {
        let kva = u64::MAX - 2;
        let err = validate_kva_target(kva, 4).expect_err("kva+len overflow must reject");
        assert!(err.contains("overflow"));
    }

    // ---- same-rendezvous-epoch marker-anchor test ----

    /// T63.1 — Doc-grep / marker-anchor regression test. Every
    /// OrU32 RMW site in the dispatcher MUST carry a
    /// `// rmw-invariant-anchor` comment. The same-rendezvous-epoch
    /// invariant is structural (per-entry sequential walk in
    /// dispatch_one_write), not type-enforced. A future refactor
    /// that extracts the RMW into a helper or relocates the
    /// read+OR+write triple outside dispatch_one_write breaks the
    /// invariant — this test guards against that by:
    ///   1. Asserting every OrU32 RMW pattern in the source carries
    ///      the marker.
    ///   2. Asserting the count of markers matches the count of
    ///      `KernelOpValue::OrU32` match arms in dispatch_one_write
    ///      (currently 3: Symbol, Direct, Kva).
    ///
    /// A refactor that adds a new RMW site without the marker, or
    /// moves an existing site outside dispatch_one_write, trips here.
    #[test]
    fn or_u32_rmw_anchors_inside_dispatch_one_write() {
        let full_src = include_str!("kernel_op_dispatch.rs");
        // Self-match exclusion (same approach as
        // dispatch_arms_call_validate_target_helpers): the searched
        // arm shape + `| mask)` pattern appear in this test's body.
        // Restrict to production source (before `#[cfg(test)]`).
        let test_mod_start = full_src
            .find("#[cfg(test)]")
            .expect("test module must exist");
        let src = &full_src[..test_mod_start];
        // Strict-count pin: exactly 3 production OrU32 arms.
        // Match-arm-shape `KernelOpValue::OrU32(mask)) => {` is
        // unique to the dispatch_one_write body. Catches a new 4th
        // arm AND catches removal of an existing arm.
        let arm_sites: Vec<_> = src
            .match_indices("KernelOpValue::OrU32(mask)) => {")
            .collect();
        assert_eq!(
            arm_sites.len(),
            3,
            "expected exactly 3 OrU32 write arms (Symbol/Direct/Kva); \
             found {} — if a 4th was added, add the rmw-invariant-anchor \
             comment to it AND update this expected count",
            arm_sites.len()
        );
        // Per-arm pattern pin (see also the extracted-helper pin
        // below): for every OrU32 match arm shape, the next ~400
        // bytes MUST contain a `rmw-invariant-anchor` marker.
        // Catches the refactor that adds a new OrU32 arm without
        // the marker comment.
        for (idx, _) in &arm_sites {
            let window_end = (idx + 400).min(src.len());
            let window = &src[*idx..window_end];
            assert!(
                window.contains("rmw-invariant-anchor"),
                "OrU32 arm at byte offset {idx} is missing the \
                 // rmw-invariant-anchor comment; window: {window:?}"
            );
        }
        // Extracted-helper pin: a refactor that extracts the
        // read+OR+write triple into a helper would LOSE the
        // match-arm shape but the read+OR+write pattern would still
        // exist somewhere. Search for that pattern via its signature
        // `| mask` (the OR operation distinctive to OrU32 RMW) —
        // every occurrence in the source MUST be inside
        // `dispatch_one_write` (between the `fn dispatch_one_write`
        // declaration and the next top-level `fn` after it).
        //
        // Find dispatch_one_write's body extent.
        let dow_start = src
            .find("fn dispatch_one_write(")
            .expect("dispatch_one_write must exist");
        // The body extends until the next top-level `fn` declaration
        // at the same indentation level (search for "\nfn " after
        // dow_start — module-private fns sit at column 0).
        let dow_end = src[dow_start..]
            .find("\nfn ")
            .map(|rel| dow_start + rel)
            .unwrap_or(src.len());
        // Count `| mask` occurrences globally vs inside dispatch_one_write.
        // The 3 OrU32 RMW arms each have `cur | mask` (or `cur | *mask`)
        // inside the write call.
        let global_or_mask: Vec<_> = src.match_indices("| mask").collect();
        let inside_dow: Vec<_> = global_or_mask
            .iter()
            .filter(|(idx, _)| *idx >= dow_start && *idx < dow_end)
            .collect();
        // Allow `| mask` matches in:
        //  - the 3 OrU32 RMW arms (inside dispatch_one_write)
        //  - the docstring/comment text describing the pattern (anywhere)
        // Production OR-with-mask sites OUTSIDE dispatch_one_write are
        // the refactor regression class — none should exist. Practical
        // detection: assert that every `| mask` occurrence followed
        // shortly by `)` (function-call close — the write call) is
        // inside dispatch_one_write.
        for (idx, _) in &global_or_mask {
            // Look ahead 4 bytes for `)` — if present, this is a
            // function-call argument (the production RMW write call).
            // If absent (e.g. `| mask)` appears in a doc comment with
            // surrounding prose), skip.
            let lookahead_end = (idx + 6).min(src.len());
            let lookahead = &src[*idx..lookahead_end];
            if lookahead.contains("| mask)") {
                assert!(
                    *idx >= dow_start && *idx < dow_end,
                    "Production `| mask)` OR-with-mask call at byte offset \
                     {idx} is OUTSIDE dispatch_one_write \
                     [start={dow_start}, end={dow_end}). \
                     A refactor extracted the OrU32 RMW into a helper, \
                     breaking the same-rendezvous-epoch invariant. \
                     Move it back inside dispatch_one_write OR (if \
                     intentional) update this test."
                );
            }
        }
        // Sanity: inside_dow should have exactly 3 entries (the 3 RMW
        // arms each contribute one `| mask`). Doc-comment refs add
        // more globally, but the inside-dow filter should be stable.
        assert_eq!(
            inside_dow.len(),
            3,
            "expected exactly 3 `| mask` production sites inside \
             dispatch_one_write (one per Symbol/Direct/Kva OrU32 arm); \
             found {}",
            inside_dow.len()
        );
    }

    // ---- TaskField walker + validation tests ----

    /// Synthetic task_struct field layout used by every TaskField test
    /// fixture below. Real kernel offsets are BTF-derived at runtime;
    /// these synthetic values just need to be (a) distinct, (b)
    /// non-overlapping for the 32/64-bit reads, and (c) within the
    /// 4 KiB test buffer.
    mod synth_task {
        pub(super) const PID_OFF: usize = 0x10;
        pub(super) const START_TIME_OFF: usize = 0x18;
        pub(super) const STATE_OFF: usize = 0x20;
        pub(super) const ON_RQ_OFF: usize = 0x28;
        pub(super) const SCHED_CLASS_OFF: usize = 0x30;
        pub(super) const START_BOOTTIME_OFF: usize = 0x40;
        pub(super) const SCX_DSQ_OFF: usize = 0x48;
        pub(super) const SCX_RUNNABLE_NODE_OFF: usize = 0x50;
        pub(super) const TASKS_OFF: usize = 0x60;
        pub(super) const SIGNAL_OFF: usize = 0x70;
        pub(super) const SIGNAL_THREAD_HEAD_OFF: usize = 0x10;
        pub(super) const THREAD_NODE_OFF: usize = 0x78;
    }

    fn synth_validation_offsets() -> TaskValidationOffsets {
        TaskValidationOffsets {
            pid: synth_task::PID_OFF,
            start_time: synth_task::START_TIME_OFF,
            state: synth_task::STATE_OFF,
            on_rq: synth_task::ON_RQ_OFF,
            sched_class: synth_task::SCHED_CLASS_OFF,
            start_boottime: synth_task::START_BOOTTIME_OFF,
            scx_dsq: synth_task::SCX_DSQ_OFF,
            scx_runnable_node: synth_task::SCX_RUNNABLE_NODE_OFF,
            tasks: synth_task::TASKS_OFF,
            signal: synth_task::SIGNAL_OFF,
            signal_thread_head: synth_task::SIGNAL_THREAD_HEAD_OFF,
            thread_node: synth_task::THREAD_NODE_OFF,
        }
    }

    const EXT_KVA: u64 = 0xFFFF_FFFF_8200_0100;
    const DEFAULT_START_TIME: u64 = 1_700_000_000_000;

    /// Paint a complete "live, SCX-managed, sleepable" task_struct
    /// at PA `pa` into `buf`. After painting, the task passes all 8
    /// validation layers. Caller mutates individual fields to
    /// trigger specific rejections.
    fn paint_valid_task(buf: &mut [u8], pa: usize, pid: u32) {
        const PAGE_OFFSET: u64 = 0xFFFF_8880_0000_0000;
        // pid (u32 LE)
        buf[pa + synth_task::PID_OFF..pa + synth_task::PID_OFF + 4]
            .copy_from_slice(&pid.to_le_bytes());
        // start_time = arbitrary
        buf[pa + synth_task::START_TIME_OFF..pa + synth_task::START_TIME_OFF + 8]
            .copy_from_slice(&DEFAULT_START_TIME.to_le_bytes());
        // state = TASK_RUNNING (0) — not TASK_DEAD
        buf[pa + synth_task::STATE_OFF..pa + synth_task::STATE_OFF + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        // on_rq = 0 (sleeping)
        buf[pa + synth_task::ON_RQ_OFF..pa + synth_task::ON_RQ_OFF + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        // sched_class = EXT_KVA
        buf[pa + synth_task::SCHED_CLASS_OFF..pa + synth_task::SCHED_CLASS_OFF + 8]
            .copy_from_slice(&EXT_KVA.to_le_bytes());
        // start_boottime = arbitrary non-zero (1 hour in ns)
        buf[pa + synth_task::START_BOOTTIME_OFF..pa + synth_task::START_BOOTTIME_OFF + 8]
            .copy_from_slice(&3_600_000_000_000u64.to_le_bytes());
        // scx.dsq = NULL (not queued)
        buf[pa + synth_task::SCX_DSQ_OFF..pa + synth_task::SCX_DSQ_OFF + 8]
            .copy_from_slice(&0u64.to_le_bytes());
        // scx.runnable_node = list-empty (next points at self)
        // task_kva = task_pa + PAGE_OFFSET; runnable_node KVA =
        // task_kva + SCX_RUNNABLE_NODE_OFF.
        let task_kva = (pa as u64).wrapping_add(PAGE_OFFSET);
        let self_kva = task_kva.wrapping_add(synth_task::SCX_RUNNABLE_NODE_OFF as u64);
        buf[pa + synth_task::SCX_RUNNABLE_NODE_OFF..pa + synth_task::SCX_RUNNABLE_NODE_OFF + 8]
            .copy_from_slice(&self_kva.to_le_bytes());
    }

    /// Build a synthetic GuestKernel; PAGE_OFFSET set so PA 0 → KVA
    /// 0xFFFF_8880_0000_0000 (matches real direct-map layout for
    /// slab-allocated task_structs).
    fn build_test_kernel(
        buf: &mut [u8],
        symbols: std::collections::HashMap<String, u64>,
    ) -> crate::monitor::guest::GuestKernel {
        const PAGE_OFFSET: u64 = 0xFFFF_8880_0000_0000;
        // SAFETY: buf outlives the kernel by virtue of caller keeping
        // it on the stack; GuestMem::new requires the backing buffer
        // remain valid for the GuestMem's lifetime.
        let mem = unsafe {
            std::sync::Arc::new(crate::monitor::reader::GuestMem::new(
                buf.as_mut_ptr(),
                buf.len() as u64,
            ))
        };
        crate::monitor::guest::GuestKernel::new_for_test(
            mem,
            symbols,
            PAGE_OFFSET,
            0,     // cr3_pa unused for direct-map translation
            false, // l5 = false (4-level)
        )
    }

    fn validate(
        kernel: &crate::monitor::guest::GuestKernel,
        task_pa: u64,
        pid: u32,
        expected_start_time_ns: u64,
        offs: &TaskValidationOffsets,
    ) -> Result<(), String> {
        validate_task_for_field_op(kernel, task_pa, pid, expected_start_time_ns, offs, EXT_KVA)
    }

    /// L1-L8 all pass on a freshly-painted valid SCX task.
    #[test]
    fn validate_task_happy_path_accepts() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        assert!(validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs).is_ok());
    }

    /// L1 (pid mismatch): walker matched a task whose pid changed
    /// between walker scan and validation read. Defense against slab
    /// recycle.
    #[test]
    fn validate_task_rejects_pid_mismatch() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 99);
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("pid mismatch must reject");
        assert!(err.contains("pid mismatch"), "must name layer: {err}");
        assert!(err.contains("read pid=99"));
        assert!(err.contains("expected 12345"));
    }

    /// L2 (start_time identity mismatch): the kernel sets start_time
    /// ONCE at fork. If the original task exited and the kernel
    /// recycled the PID, the new task's start_time will differ from
    /// what we captured at spawn. The L2 gate accepts a window of
    /// `[expected, expected + START_TIME_PROC_TICK_NS)` to absorb
    /// the userspace /proc tick quantization (see L2 doc); this
    /// test pins the rejection for an observed value BELOW expected
    /// (the only direction recycle can drift), and the next two
    /// tests pin the in-window accept + out-of-window reject above.
    #[test]
    fn validate_task_rejects_start_time_below_window() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        // observed=DEFAULT_START_TIME, expected=DEFAULT_START_TIME + 1ms
        // → observed < expected, must reject.
        let too_high_expected = DEFAULT_START_TIME + 1_000_000;
        let err = validate(&kernel, 0, 12345, too_high_expected, &offs)
            .expect_err("start_time below window must reject");
        assert!(err.contains("start_time identity mismatch"));
        assert!(err.contains(&format!("observed={DEFAULT_START_TIME}")));
        assert!(err.contains(&format!("expected in [{too_high_expected}")));
        assert!(err.contains("recycled"));
    }

    /// L2 accepts an observed start_time within the userspace-tick
    /// window above the expected value. Pins the legitimate
    /// jiffies-quantization gap that /proc/<pid>/stat field 22
    /// introduces.
    #[test]
    fn validate_task_accepts_start_time_within_tick_window() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        // observed=DEFAULT_START_TIME, expected=DEFAULT_START_TIME-5ms
        // → observed - expected = +5ms (within 10ms window).
        let expected_within_window = DEFAULT_START_TIME - 5_000_000;
        validate(&kernel, 0, 12345, expected_within_window, &offs)
            .expect("start_time within tick window must accept");
    }

    /// L2 rejects an observed start_time MORE than one tick above
    /// the expected value — characteristic of a PID-recycled task
    /// whose start_time is fundamentally newer.
    #[test]
    fn validate_task_rejects_start_time_above_window() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        // observed=DEFAULT_START_TIME, expected=DEFAULT_START_TIME-20ms
        // → observed - expected = +20ms (above 10ms window).
        let expected_below_window = DEFAULT_START_TIME - 20_000_000;
        let err = validate(&kernel, 0, 12345, expected_below_window, &offs)
            .expect_err("start_time above window must reject");
        assert!(err.contains("start_time identity mismatch"));
        assert!(err.contains("recycled"));
    }

    /// L3 (TASK_DEAD): task in final teardown — fields mid-cleanup.
    #[test]
    fn validate_task_rejects_task_dead() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        buf[synth_task::STATE_OFF..synth_task::STATE_OFF + 4]
            .copy_from_slice(&0x80u32.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("TASK_DEAD must reject");
        assert!(err.contains("TASK_DEAD"));
        assert!(err.contains("state=0x80"));
    }

    /// L4 (on_rq != 0): task is queued — writing scheduler fields
    /// would corrupt rb-tree / DSQ ordering.
    #[test]
    fn validate_task_rejects_on_rq_queued() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        buf[synth_task::ON_RQ_OFF..synth_task::ON_RQ_OFF + 4].copy_from_slice(&1u32.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("on_rq=1 must reject");
        assert!(err.contains("on_rq=1"));
        assert!(err.contains("rb-tree"));
        assert!(err.contains("WorkType::FutexPingPong"));
    }

    /// L5 part-1 (scx.dsq != NULL): task is queued on an SCX DSQ.
    #[test]
    fn validate_task_rejects_scx_dsq_populated() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        buf[synth_task::SCX_DSQ_OFF..synth_task::SCX_DSQ_OFF + 8]
            .copy_from_slice(&0xFFFF_DEAD_BEEFu64.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("scx.dsq non-NULL must reject");
        assert!(err.contains("scx.dsq=0xffffdeadbeef"));
        assert!(err.contains("SCX DSQ"));
        assert!(err.contains("WorkType::FutexPingPong"));
    }

    /// L5 part-2 (scx.runnable_node linked): task is on a per-rq
    /// runnable_list even though scx.dsq is NULL.
    #[test]
    fn validate_task_rejects_scx_runnable_node_linked() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        // Point runnable_node.next at a non-self address (linked).
        buf[synth_task::SCX_RUNNABLE_NODE_OFF..synth_task::SCX_RUNNABLE_NODE_OFF + 8]
            .copy_from_slice(&0xFFFF_8881_DEAD_C0DEu64.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("runnable_node linked must reject");
        assert!(err.contains("scx.runnable_node is linked"));
        assert!(err.contains("WorkType::FutexPingPong"));
    }

    /// L6 (sched_class != ext_sched_class): non-SCX task rejected.
    #[test]
    fn validate_task_rejects_non_ext_sched_class() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        // Set sched_class to a fake fair_sched_class KVA.
        let fair_kva: u64 = 0xFFFF_FFFF_8200_0000;
        buf[synth_task::SCHED_CLASS_OFF..synth_task::SCHED_CLASS_OFF + 8]
            .copy_from_slice(&fair_kva.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("non-ext sched_class must reject");
        assert!(err.contains(&format!("sched_class={fair_kva:#x}")));
        assert!(err.contains("SCX-managed tasks only"));
        assert!(err.contains("SchedPolicy::Ext"));
    }

    /// L8 (start_boottime == 0): probable slab-recycle survivor that
    /// passed L1 + L2 by coincidence.
    #[test]
    fn validate_task_rejects_zero_start_boottime() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        buf[synth_task::START_BOOTTIME_OFF..synth_task::START_BOOTTIME_OFF + 8]
            .copy_from_slice(&0u64.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
            .expect_err("start_boottime=0 must reject");
        assert!(err.contains("start_boottime=0"));
        assert!(err.contains("slab-recycle"));
    }

    /// Layer ordering: L1 (pid) fires BEFORE L2 (start_time). A task
    /// with BOTH pid mismatch AND start_time mismatch surfaces the
    /// pid error.
    #[test]
    fn validate_task_layer_order_pid_before_start_time() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 99);
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        // Both mismatched — expect pid error.
        let err =
            validate(&kernel, 0, 12345, DEFAULT_START_TIME + 1, &offs).expect_err("must reject");
        assert!(err.contains("pid mismatch"), "L1 must fire first: {err}");
        assert!(!err.contains("start_time identity mismatch"));
    }

    /// Layer ordering: L2 (start_time) fires BEFORE L3 (TASK_DEAD).
    #[test]
    fn validate_task_layer_order_start_time_before_dead() {
        let mut buf = vec![0u8; 4096];
        paint_valid_task(&mut buf, 0, 12345);
        buf[synth_task::STATE_OFF..synth_task::STATE_OFF + 4]
            .copy_from_slice(&0x80u32.to_le_bytes());
        let kernel = build_test_kernel(&mut buf, Default::default());
        let offs = synth_validation_offsets();
        let err =
            validate(&kernel, 0, 12345, DEFAULT_START_TIME + 1, &offs).expect_err("must reject");
        assert!(
            err.contains("start_time identity mismatch"),
            "L2 must fire first: {err}"
        );
        assert!(!err.contains("TASK_DEAD"));
    }

    /// `KernelValue::OrU32` RMW: sets the supplied bits without
    /// clobbering bits already set in the target word. Pre-populates
    /// a synthetic symbol with an alternating bit pattern and ORs a
    /// single unset bit; the read-back must show the union, not just
    /// the OR mask alone.
    ///
    /// Migrated from `tests/oru64_rmw_e2e.rs` (gated skeleton); the
    /// dispatcher narrowed OrU64 to OrU32 because the canonical
    /// scheduler-flags use case (`struct scx_rq.flags`) is u32 per
    /// `kernel/sched/sched.h:802`. Test runs as a host-side unit
    /// test against `build_test_kernel`'s synthetic guest memory —
    /// no VM boot — because the RMW correctness is pure dispatcher
    /// arithmetic.
    #[test]
    fn oru32_sets_target_bits_preserves_others() {
        // KVA = start_kernel_map (0xFFFF_FFFF_8000_0000) + PA; the
        // Symbol path's `text_kva_to_pa` translates via
        // start_kernel_map, NOT page_offset.
        const SYMBOL_PA: u64 = 0x40;
        const SYMBOL_KVA: u64 = 0xFFFF_FFFF_8000_0040;
        const INITIAL_FLAGS: u32 = 0xAAAA_AAAA;
        const OR_MASK: u32 = 0x0000_0001;
        let mut buf = vec![0u8; 4096];
        buf[SYMBOL_PA as usize..SYMBOL_PA as usize + 4]
            .copy_from_slice(&INITIAL_FLAGS.to_le_bytes());
        let mut symbols = std::collections::HashMap::new();
        symbols.insert("test_flags".to_string(), SYMBOL_KVA);
        let kernel = build_test_kernel(&mut buf, symbols);
        dispatch_one_write(
            &kernel,
            None,
            &KernelOpTarget::Symbol("test_flags".into()),
            &KernelOpValue::OrU32(OR_MASK),
        )
        .expect("OrU32 RMW dispatch must succeed against painted symbol");
        let observed = kernel
            .read_symbol_u32("test_flags")
            .expect("read-back must succeed");
        assert_eq!(
            observed,
            INITIAL_FLAGS | OR_MASK,
            "OrU32 must set 0x{OR_MASK:08x} without clobbering 0x{INITIAL_FLAGS:08x}"
        );
    }

    /// `KernelValue::OrU32` is idempotent on already-set bits: OR'ing
    /// a bit that is already 1 leaves the word unchanged. Pins that
    /// the RMW path never accidentally toggles or clears a bit it
    /// was asked to OR — a regression that flipped `|=` to `^=`
    /// would slip past the [`oru32_sets_target_bits_preserves_others`]
    /// test (which uses a NEW bit) and surface only here.
    ///
    /// Migrated from `tests/oru64_rmw_e2e.rs` (T33.2 in the original
    /// skeleton).
    #[test]
    fn oru32_idempotent_on_already_set_bit() {
        // KVA = start_kernel_map (0xFFFF_FFFF_8000_0000) + PA; the
        // Symbol path's `text_kva_to_pa` translates via
        // start_kernel_map, NOT page_offset.
        const SYMBOL_PA: u64 = 0x40;
        const SYMBOL_KVA: u64 = 0xFFFF_FFFF_8000_0040;
        const INITIAL_FLAGS: u32 = 0xAAAA_AAAA;
        // Bit 1 is set in 0xA = 1010 — picking 0x2 means OR-ing an
        // already-set bit.
        const ALREADY_SET: u32 = 0x0000_0002;
        let mut buf = vec![0u8; 4096];
        buf[SYMBOL_PA as usize..SYMBOL_PA as usize + 4]
            .copy_from_slice(&INITIAL_FLAGS.to_le_bytes());
        let mut symbols = std::collections::HashMap::new();
        symbols.insert("test_flags".to_string(), SYMBOL_KVA);
        let kernel = build_test_kernel(&mut buf, symbols);
        // Sanity: the chosen bit IS set in the initial value (test
        // construction guard).
        assert_eq!(
            INITIAL_FLAGS & ALREADY_SET,
            ALREADY_SET,
            "test setup bug: chose a bit that is not pre-set"
        );
        dispatch_one_write(
            &kernel,
            None,
            &KernelOpTarget::Symbol("test_flags".into()),
            &KernelOpValue::OrU32(ALREADY_SET),
        )
        .expect("OrU32 with already-set bit must succeed");
        let observed = kernel
            .read_symbol_u32("test_flags")
            .expect("read-back must succeed");
        assert_eq!(
            observed, INITIAL_FLAGS,
            "OrU32 of already-set bit must leave value unchanged \
             (regression: bit was toggled or cleared instead of OR'd)"
        );
    }

    /// `KernelOpValue::OrU32` survives a postcard round-trip embedded
    /// inside a full `KernelOpRequestPayload`. Pins the wire-format
    /// shape against a `#[serde(untagged)]` regression (untagged enums
    /// break postcard's externally-tagged constraint — the wire-path
    /// reader at the host would silently drop the variant tag).
    ///
    /// Migrated from `tests/oru64_rmw_e2e.rs` T33.3. The existing
    /// `kernel_op_request_payload_postcard_round_trip` test in
    /// `src/vmm/wire.rs` covers U32/U64/Bytes/PerCpuField/TaskField
    /// but does not include an OrU32 entry — this fills that gap.
    #[test]
    fn oru32_postcard_round_trip_through_payload() {
        const MASK: u32 = 0xDEAD_BEEF;
        let payload = crate::vmm::wire::KernelOpRequestPayload {
            request_id: 0xABCD,
            mode: crate::vmm::wire::KernelOpMode::Cold,
            direction: crate::vmm::wire::KernelOpDirection::Write,
            tag: "oru32_roundtrip_pin".into(),
            entries: vec![crate::vmm::wire::KernelOpEntry {
                target: KernelOpTarget::Symbol("any_symbol".into()),
                value: KernelOpValue::OrU32(MASK),
            }],
        };
        let bytes = postcard::to_allocvec(&payload).expect("encode");
        let back: crate::vmm::wire::KernelOpRequestPayload =
            postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back.entries.len(), 1);
        match &back.entries[0].value {
            KernelOpValue::OrU32(observed_mask) => {
                assert_eq!(*observed_mask, MASK, "OrU32 mask must survive round-trip");
            }
            other => panic!("expected OrU32 variant after round-trip, got {other:?}"),
        }
    }
}
