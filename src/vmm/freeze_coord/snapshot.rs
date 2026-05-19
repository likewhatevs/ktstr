//! Stateless helpers for the snapshot / watchpoint paths.
//!
//! Every function here is a pure transformation between guest-wire
//! bytes and host-side typed values, plus the small symbol-cache
//! abstraction over the vmlinux ELF. Behaviour requiring the
//! coordinator's captured Arcs (`freeze_coord_*` Vecs, `WatchpointArm`
//! lifetime, vCPU pthread tids) lives in the higher-level call
//! sites; the helpers here only need their explicit arguments.
//!
//! Two reasons to keep these out of the run-loop closure:
//!
//! 1. They are testable in isolation — unit tests for
//!    [`frame_snapshot_reply`] and [`decode_snapshot_request`]
//!    cover the wire-format contract without booting a VM.
//! 2. They are reused across paths inside the closure body
//!    (CAPTURE replies use `frame_snapshot_reply`; the late-trigger
//!    path uses `snapshot_tagged_path`; the symbol cache backs
//!    every WATCH request through `arm_user_watchpoint`).
//!
//! No state, no statics — only functions and the
//! [`VmlinuxSymbolCache`] which is built once per coordinator at
//! `run_vm` scope.

use crate::sync::MutexExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::EventFd;

use super::super::vcpu::{ImmediateExitHandle, WatchpointArm, vcpu_signal};
use super::state::SnapshotRequest;
use crate::vmm::x86_64::msr_kaslr::KERNEL_HALF_CANONICAL_4LEVEL;

/// Frame a `MSG_TYPE_SNAPSHOT_REPLY` TLV — header (16 bytes) plus
/// [`crate::vmm::wire::SnapshotReplyPayload`] (72 bytes) — into a
/// single buffer the coordinator pushes through
/// [`crate::vmm::virtio_console::VirtioConsole::queue_input_port1`].
/// The reply is delivered atomically as one TLV: the buffer is
/// concatenated before the call so a partial push that splits header
/// and payload across multiple `queue_input_port1` invocations cannot
/// arise. CRC32 is computed over the payload bytes only — matches
/// the wire-format contract `parse_tlv_stream` enforces on the
/// guest's `read_bulk_port_frame`.
/// Frame a `MSG_TYPE_KERNEL_OP_REPLY` TLV — header (16 bytes) plus
/// the postcard-encoded [`crate::vmm::wire::KernelOpReplyPayload`]
/// — into a single buffer the coordinator pushes through
/// [`crate::vmm::virtio_console::VirtioConsole::queue_input_port1`].
/// Mirrors [`frame_snapshot_reply`]'s atomic-TLV discipline: the
/// header + payload buffer is built before the queue call so a
/// partial push that splits header from payload cannot arise. CRC32
/// is computed over the postcard bytes only.
///
/// `Err(postcard::Error)` propagates when the payload would not
/// serialize (impossible in practice for the typed
/// `KernelOpReplyPayload` shape, but the typed `Result` keeps the
/// caller honest about the encode invariant). Caller maps the error
/// to a tracing event + drops the reply (the guest's blocking
/// reader times out at its 30 s per-op deadline — same shape as
/// any other transport failure).
pub(super) fn frame_kernel_op_reply(
    reply: &crate::vmm::wire::KernelOpReplyPayload,
) -> Result<Vec<u8>, postcard::Error> {
    use crate::vmm::wire::{FRAME_HEADER_SIZE, MSG_TYPE_KERNEL_OP_REPLY, ShmMessage};
    use zerocopy::IntoBytes;
    let payload_bytes = postcard::to_allocvec(reply)?;
    let header = ShmMessage {
        msg_type: MSG_TYPE_KERNEL_OP_REPLY,
        length: payload_bytes.len() as u32,
        crc32: crc32fast::hash(&payload_bytes),
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload_bytes.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&payload_bytes);
    Ok(buf)
}

pub(super) fn frame_snapshot_reply(request_id: u32, status: u32, reason: &str) -> Vec<u8> {
    use crate::vmm::wire::{
        FRAME_HEADER_SIZE, MSG_TYPE_SNAPSHOT_REPLY, SNAPSHOT_REASON_MAX, ShmMessage,
        SnapshotReplyPayload,
    };
    use zerocopy::IntoBytes;
    // Reason buffer: NUL-terminated UTF-8, truncated to the buffer
    // size. Trailing zeros remain from the array initializer so a
    // shorter reason terminates cleanly on the guest side.
    let reason_bytes = reason.as_bytes();
    let reason_len = reason_bytes.len().min(SNAPSHOT_REASON_MAX);
    let mut reason_buf = [0u8; SNAPSHOT_REASON_MAX];
    reason_buf[..reason_len].copy_from_slice(&reason_bytes[..reason_len]);
    let payload = SnapshotReplyPayload {
        request_id,
        status,
        reason: reason_buf,
    };
    let payload_bytes = payload.as_bytes();
    let header = ShmMessage {
        msg_type: MSG_TYPE_SNAPSHOT_REPLY,
        length: payload_bytes.len() as u32,
        crc32: crc32fast::hash(payload_bytes),
        _pad: 0,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload_bytes.len());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(payload_bytes);
    buf
}

/// Decode a guest-side `MSG_TYPE_SNAPSHOT_REQUEST` TLV payload into
/// the typed [`SnapshotRequest`]. `payload` must be exactly
/// `size_of::<SnapshotRequestPayload>()` bytes — the bulk parser
/// already enforces the per-frame cap, but a malformed guest may
/// publish a frame whose announced length doesn't match the typed
/// payload size. Returns `None` for any size or layout mismatch so
/// the TOKEN_TX handler can drop the frame without touching dispatch.
pub(super) fn decode_snapshot_request(payload: &[u8]) -> Option<SnapshotRequest> {
    use crate::vmm::wire::{SNAPSHOT_KIND_NONE, SNAPSHOT_TAG_MAX, SnapshotRequestPayload};
    use zerocopy::FromBytes;
    if payload.len() != std::mem::size_of::<SnapshotRequestPayload>() {
        return None;
    }
    let req = SnapshotRequestPayload::read_from_bytes(payload).ok()?;
    if req.request_id == 0 || req.kind == SNAPSHOT_KIND_NONE {
        return None;
    }
    let len = req
        .tag
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(SNAPSHOT_TAG_MAX);
    let tag = String::from_utf8_lossy(&req.tag[..len]).to_string();
    Some(SnapshotRequest {
        request_id: req.request_id,
        kind: req.kind,
        tag,
    })
}

/// Cached `name -> KVA` map built once at coordinator init from the
/// vmlinux ELF symbol table. Lets [`arm_user_watchpoint`] look up
/// `Op::WatchSnapshot` symbols without re-reading and re-parsing
/// the 50MB+ vmlinux per request.
pub(super) struct VmlinuxSymbolCache {
    symbols: std::collections::HashMap<String, u64>,
}

impl VmlinuxSymbolCache {
    /// Read and parse `path` once, extracting every symbol whose
    /// `st_shndx` is not `SHN_UNDEF` into the cache. Errors propagate
    /// as caller-side diagnostics so arming surfaces the same reason
    /// strings the per-call parse used.
    ///
    /// SHN_UNDEF (== 0 per ELF spec) marks linker placeholders and
    /// imports; those have no defining section and must be filtered.
    /// Filtering on `st_value == 0` instead would also drop legitimate
    /// symbols at section offset 0 (the percpu
    /// case, and the same filter masks any defined symbol whose KVA
    /// happens to be 0 — `arm_user_watchpoint` rejects `kva == 0`
    /// downstream so a 0-valued defined symbol still surfaces a
    /// diagnostic instead of being silently absent).
    pub(super) fn from_path(path: &std::path::Path) -> std::result::Result<Self, String> {
        const SHN_UNDEF: usize = 0;
        let data_arc = super::super::vmlinux::cached_vmlinux_bytes(path)
            .ok_or_else(|| format!("read vmlinux at {}", path.display()))?;
        let data = &*data_arc;
        let elf = goblin::elf::Elf::parse(data).map_err(|e| format!("parse vmlinux ELF: {e}"))?;
        let mut symbols = std::collections::HashMap::new();
        for s in elf.syms.iter() {
            if s.st_shndx == SHN_UNDEF {
                continue;
            }
            if let Some(name) = elf.strtab.get_at(s.st_name) {
                symbols.insert(name.to_string(), s.st_value);
            }
        }
        Ok(Self { symbols })
    }

    pub(super) fn lookup(&self, symbol: &str) -> Option<u64> {
        self.symbols.get(symbol).copied()
    }

    /// Test-only constructor that bypasses the vmlinux ELF read /
    /// parse path of [`Self::from_path`] and seeds the cache from a
    /// pre-built `name -> KVA` map. Lets unit tests exercise
    /// [`Self::lookup`] and [`arm_user_watchpoint`]'s symbol /
    /// alignment / slot dispatch without needing a real 50MB+
    /// vmlinux blob on disk. Mirrors the production invariant that
    /// `lookup` returns whatever was last inserted under `name` —
    /// duplicate keys in the source map collapse to last-write-wins
    /// semantics from `HashMap::insert`, exactly as `from_path`
    /// produces when two ELF symbols share a name.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn from_symbols_for_test(symbols: std::collections::HashMap<String, u64>) -> Self {
        Self { symbols }
    }
}

/// Resolve a kernel symbol by name from the cached vmlinux symbol
/// table and arm a user watchpoint slot (slots 1..=3) on it.
/// Returns the slot index (0..=2 mapping to slots 1..=3) on
/// success, or a host-side diagnostic on failure.
///
/// The vCPU thread's `self_arm_watchpoint` notices the change on
/// the next loop iteration (Acquire load on the slot's
/// `request_kva`) and reprograms `KVM_SET_GUEST_DEBUG` with the
/// new DR layout.
/// Arm a user watchpoint slot on `symbol`'s resolved KVA.
///
/// On success, the slot's `request_kva` is published with `Release`,
/// `WatchpointArm::mark_armed()` flips the fast-path gate, and every
/// vCPU thread (BSP + APs) is kicked out of `KVM_RUN` so its next
/// loop iteration runs `self_arm_watchpoint` and reprograms
/// `KVM_SET_GUEST_DEBUG`.
///
/// Without the gate flip, the per-vCPU `self_arm_watchpoint` short-
/// circuits at the `any_armed.load(Relaxed) == 0` check and never
/// observes the published `request_kva`. Without the kick, vCPU
/// threads sitting in `KVM_RUN` only re-check the slot on their next
/// natural exit (HLT, IO, IRQ) — for compute-bound guests that can
/// be many seconds, missing the very write the user requested to
/// observe. Mirrors the freeze-rendezvous kick pattern (pass 1: set
/// every immediate_exit byte; pass 2: deliver SIGRTMIN to every
/// vCPU TID), differing only in that arming does NOT request a
/// freeze — vCPUs immediately re-enter `KVM_RUN` after the arm.
///
/// `bsp_alive` is the same Acquire-bool the freeze_and_capture
/// closure consults: a `false` reading means the BSP `VcpuFd` is
/// gone and writing through `bsp_ie_handle` would touch unmapped
/// memory. The Arc is borrowed (not a snapshot) so each gated site
/// performs its own fresh `load(Acquire)` immediately before the
/// BSP-touching syscall — both `ie.set(1)` (which writes through
/// the kvm_run mmap) and `pthread_kill` (which dereferences the
/// BSP `pthread_t`) re-check liveness at the moment of the call.
/// A snapshot taken at the start of the kick pass would be stale
/// by tens of microseconds — long enough for the BSP run-loop's
/// post-loop `bsp_alive.store(false, Release)` plus the BSP
/// `VcpuFd` drop to land between snapshot and use, leaving the
/// fast-path writes targeting freed mmap pages.
#[allow(clippy::too_many_arguments)]
pub(super) fn arm_user_watchpoint(
    watchpoint: &Arc<WatchpointArm>,
    symbol_cache: &VmlinuxSymbolCache,
    symbol: &str,
    kaslr_offset: u64,
    ap_pthreads: &[libc::pthread_t],
    ap_ies: &[Option<ImmediateExitHandle>],
    ap_alive: &[Arc<AtomicBool>],
    bsp_tid: libc::pthread_t,
    bsp_ie: Option<&ImmediateExitHandle>,
    bsp_alive: &Arc<AtomicBool>,
) -> std::result::Result<usize, String> {
    // Check cap and find a free slot.
    let mut free_slot: Option<usize> = None;
    for (i, slot) in watchpoint.user.iter().enumerate() {
        if slot.request_kva.load(Ordering::Acquire) == 0 {
            free_slot = Some(i);
            break;
        }
    }
    let Some(idx) = free_slot else {
        return Err(format!(
            "no free user watchpoint slot — slots 1..=3 all occupied by prior \
             Op::WatchSnapshot registrations (cap = {})",
            watchpoint.user.len()
        ));
    };
    // Resolve the symbol via the cached vmlinux symbol table.
    // The cache is built once at coord init; per-call lookups are
    // O(1) HashMap reads instead of 50MB+ file reads + ELF parses.
    let link_kva = symbol_cache
        .lookup(symbol)
        .ok_or_else(|| format!("symbol '{symbol}' not found in vmlinux symtab"))?;
    let kva = link_kva.wrapping_add(kaslr_offset);
    // Silent-misfire detection: when `kaslr_offset == 0` AND the
    // resolved `link_kva` is in the x86_64 kernel high-half
    // (canonical-space addresses start at [`KERNEL_HALF_CANONICAL_4LEVEL`]
    // = `0xffff_8000_0000_0000`
    // per Documentation/arch/x86/x86_64/mm.rst, where the guard
    // hole / hypervisor mapping ends and kernel mappings begin),
    // arming at the link-time address under `CONFIG_RANDOMIZE_BASE=y`
    // (the default in `ktstr.kconfig`) would program the DR at a
    // KVA the guest never writes — the runtime symbol lives at
    // `link_kva + runtime_kaslr_slide`, and the link-time DR never
    // fires. The operator would see a valid-looking arm with zero
    // hits and no diagnostic.
    //
    // The arm REFUSES with a typed `Err` rather than completing.
    // This is a deliberate fail-loud: callers get an actionable
    // diagnostic naming the symbol, the link_kva, and the operator
    // workaround (`#[ktstr_test(kaslr = false)]`). The warn-once
    // tracker keeps CI logs readable when many arms target the
    // same symbol; the `Err` itself fires on every failing arm so
    // every caller observes the failure.
    //
    // Low-address synthetic fixtures (link_kva below the high-half
    // threshold) bypass the check entirely — those are tests, not
    // production kernel symbols, and aren't subject to KASLR
    // slide. Callers that legitimately want to arm against a
    // post-slide KVA pass `kaslr_offset != 0` (e.g. the coord's
    // `kern_virt_kaslr` Arc, populated by the MSR_LSTAR readback
    // path); both branches of `kaslr_offset != 0 OR link_kva <
    // high-half` skip the gate.
    //
    // Arch caveats: the [`KERNEL_HALF_CANONICAL_4LEVEL`] threshold is the
    // x86_64 canonical-space lower bound (Documentation/arch/
    // x86/x86_64/mm.rst). arm64 layouts differ — VA_BITS=48 puts
    // PAGE_OFFSET at `0xffff_0000_0000_0000` (memory.rst L52), so
    // some arm64 kernel-text addresses fall below the x86_64
    // threshold and won't trip the warn. That conservative miss
    // is acceptable today (arm64 watchpoint paths are gated
    // separately and the host coordinator's KASLR-derive plumbing
    // is x86_64-first); revisit when arm64 grows full watchpoint
    // support.
    //
    // Log cadence: once per unique `(symbol, link_kva)` per
    // process. The first arm fires the warn, every subsequent
    // arm against the same symbol/KVA stays silent — operators
    // get the signal without log spam dominating a CI run.
    if kaslr_offset == 0 && link_kva >= KERNEL_HALF_CANONICAL_4LEVEL {
        // Process-wide one-shot tracker keyed on `(symbol,
        // link_kva)` so the warn line in the Err message is
        // emitted at most once per unique (symbol, link_kva) per
        // process — keeps CI logs readable when a test triggers
        // many WatchSnapshot arms against the same symbol. The
        // typed Err itself is always returned (no once-per-process
        // suppression on the error path — every failing arm sees
        // the same typed reply).
        static WARN_ONCE: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashSet<(String, u64)>>,
        > = std::sync::OnceLock::new();
        let seen =
            WARN_ONCE.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let key = (symbol.to_string(), link_kva);
        let first_fire = {
            let mut guard = seen.lock().unwrap_or_else(|e| e.into_inner());
            guard.insert(key)
        };
        if first_fire {
            tracing::warn!(
                symbol = symbol,
                link_kva = format_args!("{link_kva:#x}"),
                "arm_user_watchpoint: kaslr_offset = 0 but link_kva is in \
                 x86_64 kernel high-half — the kern_virt_kaslr Arc has not \
                 yet published a non-zero slide. Returning Err to fail loud \
                 rather than arming the DR at the link-time KVA (which would \
                 never match guest writes under CONFIG_RANDOMIZE_BASE=y, \
                 silently no-op'ing the watch). If the test legitimately \
                 needs nokaslr semantics, set `#[ktstr_test(kaslr = false)]` \
                 or `Scheduler::kargs(&[\"nokaslr\"])` so the cmdline pins \
                 kaslr off and the link-time KVA IS the runtime KVA. \
                 (Warn logged once per unique symbol/KVA pair per process; \
                 the Err return fires on every failing arm.)"
            );
        }
        return Err(format!(
            "symbol '{symbol}' link_kva {link_kva:#x} is in x86_64 kernel \
             high-half but kern_virt_kaslr Arc has not published a non-zero \
             slide (coord_kaslr_offset() == 0); refusing to arm DR at \
             link-time KVA — would never match guest writes under KASLR-on. \
             If nokaslr semantics intended, set `#[ktstr_test(kaslr = false)]`."
        ));
    }
    // `request_kva == 0` is the slot's "free" sentinel — the per-vCPU
    // `self_arm_watchpoint` short-circuits on a zero request_kva, and
    // the free-slot scan above treats kva == 0 as available. Arming
    // a slot with kva == 0 would publish a no-op store the vCPU would
    // ignore, leaving the slot wedged in a half-armed state (tag
    // populated but no DR programmed). Reject explicitly so the caller
    // surfaces a diagnostic instead of a silent success.
    if kva == 0 {
        return Err(format!(
            "symbol '{symbol}' resolved to KVA 0 — defined symbol at \
             address 0 is not arm-able (slot's `request_kva == 0` is \
             the free-slot sentinel; arming would be a silent no-op)"
        ));
    }
    if kva & 0x3 != 0 {
        return Err(format!(
            "symbol '{symbol}' KVA {kva:#x} is not 4-byte aligned. \
             x86_64 DR_LEN_4 watchpoints (Intel SDM Vol. 3B Ch. 17) \
             and aarch64 DBGWVR (ARM ARM D7.3.10, requires VA[1:0] = \
             00) both require 4-byte aligned targets for the 4-byte \
             write-watch the failure-dump trigger uses"
        ));
    }
    // Publish tag first, then KVA last (the vCPU's Acquire load on
    // request_kva synchronises-with this Release; the tag must be
    // visible by the time the vCPU latches a hit on this slot).
    {
        let mut tag_guard = watchpoint.user[idx].tag.lock_unpoisoned();
        *tag_guard = symbol.to_string();
    }
    watchpoint.user[idx]
        .request_kva
        .store(kva, Ordering::Release);
    // Flip the fast-path gate so per-vCPU `self_arm_watchpoint` calls
    // stop short-circuiting on `any_armed == 0`. Idempotent — repeated
    // calls keep the gate at 1. Must happen AFTER the Release on
    // `request_kva`: the `mark_armed` store is `Relaxed`, so the
    // synchronizes-with edge that publishes the new KVA value comes
    // from `request_kva`'s Release / per-vCPU Acquire pair, not the
    // gate. Once a vCPU sees `any_armed == 1` it falls through to the
    // Acquire load on `request_kva` which carries the edge.
    watchpoint.mark_armed();
    // Two-pass kick (pass 1: every immediate_exit byte; pass 2:
    // SIGRTMIN to every vCPU TID), separated by a Release fence so
    // the immediate_exit writes are observable before any vCPU's
    // signal handler returns and re-enters KVM_RUN. Mirrors the
    // freeze rendezvous kick path so a future refactor of either
    // changes them in lock-step.
    //
    // The eventfd write that USED to live here (commented as "the
    // cleanest available wake fd") was load-bearing-shaped but
    // semantically a no-op: vCPU threads do not block on `hit_evt`,
    // so writing to it does NOT wake them out of `KVM_RUN`. The
    // actual wake mechanism is immediate_exit + SIGRTMIN — the same
    // pair the freeze rendezvous uses for parking.
    //
    // Per-AP `alive` gate mirrors the freeze rendezvous pass-1 kick:
    // an AP that panicked under `panic = "unwind"` (test profile)
    // has its hook flip `alive` to `false` BEFORE the stack drop
    // unmaps `kvm_run`. Loading Acquire here pairs with the panic
    // hook's Release store; a `true` reading observed at this site
    // happens-before any subsequent unwind drop, so the
    // `ie.set(1)` writes through a still-live mmap. The
    // `iter().enumerate()` walk keeps `ap_alive[i]` index-aligned
    // with `ap_ies[i]`.
    for (i, ie) in ap_ies.iter().enumerate() {
        if let Some(ie) = ie
            && ap_alive[i].load(Ordering::Acquire)
        {
            ie.set(1);
        }
    }
    // Fresh Acquire-load of bsp_alive immediately before `ie.set(1)`.
    // A snapshot taken earlier would race with a BSP that finishes
    // its run loop, stores `false` (Release), and starts dropping
    // its `VcpuFd` — `ie.set(1)` writes through `kvm_run.immediate_exit`
    // which lives in the BSP VcpuFd's mmap, so a stale `true` here
    // would dereference freed pages. The Acquire load synchronises
    // with the BSP run-loop's Release store of `false`: a `true`
    // observed here happens-before any subsequent `false` the BSP
    // could publish, which means the BSP VcpuFd is still alive AT
    // the moment of `ie.set()` and cannot be dropped until the next
    // load reads false (the pthread_kill below issues its own fresh
    // load for the same TOCTOU reason).
    if bsp_alive.load(Ordering::Acquire)
        && let Some(ie) = bsp_ie
    {
        ie.set(1);
    }
    std::sync::atomic::fence(Ordering::Release);
    for &tid in ap_pthreads {
        // SAFETY: pthread_kill against a tid whose thread has
        // already exited returns ESRCH. The AP threads are joined
        // by `collect_results` AFTER this coordinator joins (see
        // `run_vm`); during arm_user_watchpoint the coord is alive
        // and every AP `pthread_t` it captured at spawn is still
        // valid. ESRCH is harmless here — a kicked-but-already-gone
        // AP simply means the kick is unnecessary.
        unsafe {
            libc::pthread_kill(tid, vcpu_signal());
        }
    }
    // Second fresh Acquire-load right before BSP pthread_kill. The
    // window between the `ie.set` load and this one is microseconds
    // (one Release fence + N AP signal deliveries), but a BSP exit
    // can land in that window. A stale `true` reading here would
    // signal a pthread_t whose thread has already returned — ESRCH
    // is harmless on its own, but the surrounding contract relies
    // on the bsp_alive bool being authoritative for all BSP-touching
    // operations in this function. Loading fresh keeps the contract
    // honest and matches the freeze_and_capture pattern at the
    // mod.rs pass-2 site.
    if bsp_alive.load(Ordering::Acquire) {
        // SAFETY: bsp_alive is Acquire-loaded immediately above;
        // while true the BSP `VcpuFd` and its kvm_run mmap are
        // live. The BSP TID was captured at coord spawn from the
        // BSP thread's `pthread_self()` and remains valid until
        // the BSP thread joins, which `run_vm` only allows AFTER
        // this coordinator joins.
        unsafe {
            libc::pthread_kill(bsp_tid, vcpu_signal());
        }
    }
    Ok(idx)
}

/// Build a name-tagged sibling path for a CAPTURE-class on-demand
/// snapshot. Given `{base}/{stem}.failure-dump.json` and tag
/// `mid_run`, returns `{base}/{stem}.snapshot.mid_run.json`. Used by
/// the freeze coordinator's CAPTURE handler so the test's
/// post-scenario reader can find the file by snapshot tag without
/// guessing the on-demand counter.
///
/// The tag is sanitised: any byte that is not `[A-Za-z0-9._-]` is
/// replaced with `_` to keep the resulting filename safe across
/// filesystems regardless of what UTF-8 the guest passed.
pub(super) fn snapshot_tagged_path(base: &std::path::Path, tag: &str) -> std::path::PathBuf {
    let mut tagged = base.to_path_buf();
    let raw_stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty() && !s.starts_with('.'))
        .unwrap_or("dump");
    let stem = raw_stem.strip_suffix(".failure-dump").unwrap_or(raw_stem);
    let ext = base.extension().and_then(|e| e.to_str());
    let safe_tag: String = tag
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let new_name = match ext {
        Some(ext) => format!("{stem}.snapshot.{safe_tag}.{ext}"),
        None => format!("{stem}.snapshot.{safe_tag}"),
    };
    tagged.set_file_name(new_name);
    tagged
}

/// Wait up to `timeout_ms` for `evt` to become readable, returning
/// when the eventfd's counter is non-zero OR the timeout elapses.
/// Does not consume the counter — `poll(POLLIN)` is level-triggered,
/// so a single `evt.write(1)` from any cloned writer fans out to
/// every reader: each reader's poll returns immediately (level held
/// high) and re-checks its own readiness condition. This is the
/// broadcast wake primitive for the probes-ready eventfd shared
/// across the monitor and bpf-map-write threads — the first thread
/// to detect its readiness writes 1, and every other waiter
/// observes the level transition without racing on a consuming
/// `read()`.
///
/// Treats every poll() return path (timeout, ready, EINTR, error)
/// as "wake-up time" — the caller re-checks its own deadline and
/// wake-byte / kernel-state condition each iteration regardless.
/// EINTR from a signal during the wait is therefore harmless.
pub(super) fn poll_eventfd_until_ready_or_timeout(evt: &EventFd, timeout_ms: i32) {
    use std::os::fd::AsRawFd;
    let mut pfd = libc::pollfd {
        fd: evt.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: pfd is a valid &mut pointing to a single pollfd; nfds
    // is 1 matching the slice length; timeout_ms is forwarded
    // directly to the kernel which interprets it per poll(2). The
    // return value is intentionally discarded — every outcome
    // (ready, timeout, EINTR, error) drives the caller back into
    // its own condition check loop, which re-evaluates kill /
    // deadline / wake-byte each iteration.
    unsafe {
        libc::poll(&mut pfd, 1, timeout_ms);
    }
}

#[cfg(test)]
mod snapshot_tagged_path_tests {
    //! Unit coverage for [`snapshot_tagged_path`].
    //!
    //! The CAPTURE handler in the freeze coordinator's TLV dispatch
    //! builds a name-tagged sibling path next to the failure-dump
    //! base so the test harness can locate on-demand snapshots by
    //! tag without guessing the on-demand counter. The function is
    //! a pure path-string transform, but its three contracts are
    //! load-bearing:
    //!
    //!   * The `.failure-dump` suffix on the base stem is stripped
    //!     so two-pass tagging
    //!     (`base.failure-dump.json` → `base.snapshot.t.json`) does
    //!     not double-prefix.
    //!   * The tag is sanitised — any byte outside
    //!     `[A-Za-z0-9._-]` is replaced with `_` so a hostile guest
    //!     cannot smuggle a path traversal (`/`, `..`, NUL) past
    //!     the filename boundary.
    //!   * The result is idempotent under repeated calls with the
    //!     same input (same input bytes → same output path).
    use super::snapshot_tagged_path;
    use std::path::{Path, PathBuf};

    /// Healthy CAPTURE path: `{base}.failure-dump.json` with a
    /// sanitised tag yields `{base}.snapshot.{tag}.{ext}`. The
    /// `.failure-dump` suffix on the stem is stripped so the
    /// resulting path does not double-tag (`...failure-dump.snapshot...`).
    #[test]
    fn strips_failure_dump_suffix_from_stem() {
        let base = Path::new("/tmp/run/coord.failure-dump.json");
        let out = snapshot_tagged_path(base, "mid_run");
        assert_eq!(out, PathBuf::from("/tmp/run/coord.snapshot.mid_run.json"));
    }

    /// Tag with a path-traversal character (`/`) is sanitised to
    /// `_`. A hostile guest publishing a tag like `../etc/passwd`
    /// must NOT escape the directory the base path lives in.
    #[test]
    fn sanitises_path_traversal_in_tag() {
        let base = Path::new("/tmp/run/coord.failure-dump.json");
        let out = snapshot_tagged_path(base, "../etc/passwd");
        assert_eq!(
            out,
            PathBuf::from("/tmp/run/coord.snapshot..._etc_passwd.json")
        );
    }

    /// Tag with NUL and shell metacharacters is sanitised. The
    /// per-char filter rejects every byte that is not
    /// `[A-Za-z0-9._-]`, so a NUL or shell metacharacter cannot
    /// terminate the filename early or change shell semantics if
    /// the path is ever expanded.
    #[test]
    fn sanitises_nul_and_shell_metachars_in_tag() {
        let base = Path::new("/tmp/run/coord.failure-dump.json");
        let out = snapshot_tagged_path(base, "x\0y;rm -rf");
        // \0, ;, space → _ ; alphanumeric and `-` survive.
        assert_eq!(
            out,
            PathBuf::from("/tmp/run/coord.snapshot.x_y_rm_-rf.json")
        );
    }

    /// Allowed character set survives sanitisation verbatim.
    /// Period, underscore, dash plus alphanumerics map to
    /// themselves — confirms the inverse of the rejection check
    /// for the bytes the production path explicitly white-lists.
    #[test]
    fn preserves_allowed_chars_in_tag() {
        let base = Path::new("/tmp/run/coord.failure-dump.json");
        let out = snapshot_tagged_path(base, "Tag.1_v-2");
        assert_eq!(out, PathBuf::from("/tmp/run/coord.snapshot.Tag.1_v-2.json"));
    }

    /// Path with no extension yields a result with no extension —
    /// the `match ext` branch falls through to the
    /// `format!("{stem}.snapshot.{safe_tag}")` arm so the original
    /// extension-less shape is preserved.
    #[test]
    fn no_extension_path_omits_extension() {
        let base = Path::new("/tmp/run/coord");
        let out = snapshot_tagged_path(base, "tag1");
        assert_eq!(out, PathBuf::from("/tmp/run/coord.snapshot.tag1"));
    }

    /// Path with only an extension and no recognisable stem
    /// (`/tmp/run/.json` — a hidden-file convention with no name)
    /// falls back to the `"dump"` literal in the `unwrap_or`
    /// clause. `Path::file_stem` on `.json` returns `None` so the
    /// fallback is exercised; the suffix strip is a no-op on
    /// `"dump"` so the resulting filename is
    /// `dump.snapshot.{tag}.json`.
    #[test]
    fn no_stem_path_falls_back_to_dump() {
        let base = Path::new("/tmp/run/.json");
        let out = snapshot_tagged_path(base, "tag1");
        // `.json` is a dotfile (no stem, no extension per Rust Path).
        // The function falls back to stem="dump" and ext=None.
        assert_eq!(out, PathBuf::from("/tmp/run/dump.snapshot.tag1"));
    }

    /// Idempotence under re-stripping: the helper does NOT
    /// recursively strip prior `.snapshot` segments — it only
    /// strips `.failure-dump`. Pins this exact behaviour so a
    /// future caller passing a previously-tagged path produces a
    /// predictable result rather than an accidental three-level
    /// filename. Also confirms calling the helper twice with the
    /// same inputs yields the same path (pure-function
    /// idempotence — no counter, no time-based suffix).
    #[test]
    fn restripping_after_first_tag_is_idempotent_per_call() {
        let base = Path::new("/tmp/run/coord.failure-dump.json");
        // First tag produces `coord.snapshot.t1.json`.
        let once = snapshot_tagged_path(base, "t1");
        assert_eq!(once, PathBuf::from("/tmp/run/coord.snapshot.t1.json"));
        // Re-tagging that result produces
        // `coord.snapshot.t1.snapshot.t2.json` — single-pass
        // strip, not recursive. The caller must always feed the
        // original base path, not a previously-tagged result.
        let twice = snapshot_tagged_path(&once, "t2");
        assert_eq!(
            twice,
            PathBuf::from("/tmp/run/coord.snapshot.t1.snapshot.t2.json")
        );
        // Pure-function idempotence: same inputs → same output
        // across calls. Catches a regression that introduces a
        // counter-based or time-based suffix inside the helper.
        let again = snapshot_tagged_path(base, "t1");
        assert_eq!(again, once);
    }

    /// Plain stem without `.failure-dump` suffix is NOT stripped
    /// — the `strip_suffix` returns `None` and the stem flows
    /// through verbatim. Pins the gate so a caller passing
    /// `{base}.json` (without the failure-dump marker) lands at
    /// `{base}.snapshot.{tag}.json` rather than a stripped
    /// fragment.
    #[test]
    fn stem_without_failure_dump_suffix_is_preserved() {
        let base = Path::new("/tmp/run/coord.json");
        let out = snapshot_tagged_path(base, "tag1");
        assert_eq!(out, PathBuf::from("/tmp/run/coord.snapshot.tag1.json"));
    }

    /// Composition pin: each public SNAPSHOT_TAG_* constant from
    /// `crate::monitor::dump` composes into a path containing the
    /// expected tag substring. Catches a regression where a tag
    /// constant gets silently moved between call-sites (e.g. the
    /// late-Suppressed arm accidentally using the EARLY_DEGRADED
    /// constant) — the test verifies the constant flowing through
    /// `snapshot_tagged_path` produces the expected operator-
    /// readable filename, not just that the constant value is
    /// pinned in isolation.
    #[test]
    fn snapshot_tagged_path_composition_per_dump_tag_constant() {
        use crate::monitor::dump::{
            SNAPSHOT_TAG_EARLY_DEGRADED, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED,
            SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED, SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
        };
        let base = Path::new("/tmp/run/coord.failure-dump.json");
        let cases = [
            (
                SNAPSHOT_TAG_EARLY_DEGRADED,
                "/tmp/run/coord.snapshot.early-degraded.json",
            ),
            (
                SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
                "/tmp/run/coord.snapshot.early-pre-late-degraded.json",
            ),
            (
                SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED,
                "/tmp/run/coord.snapshot.early-only-late-suppressed.json",
            ),
            (
                SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED,
                "/tmp/run/coord.snapshot.early-only-late-never-fired.json",
            ),
        ];
        for (tag_const, expected_path) in cases {
            let out = snapshot_tagged_path(base, tag_const);
            assert_eq!(
                out,
                PathBuf::from(expected_path),
                "snapshot_tagged_path(base, {tag_const:?}) did not match expected"
            );
        }
    }
}

#[cfg(test)]
mod vmlinux_symbol_cache_tests {
    //! Unit coverage for [`VmlinuxSymbolCache`].
    //!
    //! `from_path` reads the vmlinux ELF on disk and parses every
    //! defined symbol into a HashMap; `lookup` returns the cached
    //! KVA. The `from_symbols_for_test` constructor lets tests
    //! seed the map directly without manufacturing a 50MB+ ELF
    //! blob, so the lookup contract can be pinned in isolation.
    //!
    //! The contract these tests guard:
    //!   * `lookup` returns `Some(kva)` for an inserted name.
    //!   * `lookup` returns `None` for an absent name (no panic,
    //!     no default-zero return).
    //!   * Duplicate inserts under the same name resolve to the
    //!     last-write — matches `from_path`'s symbol-table walk
    //!     where two ELF symbols sharing a name (e.g. weak +
    //!     strong) collapse into the last-seen entry.
    use super::VmlinuxSymbolCache;
    use std::collections::HashMap;

    /// `from_symbols_for_test` round-trips: every key inserted is
    /// retrievable via `lookup` with the inserted KVA. Pins the
    /// constructor's contract — the test cache and the production
    /// `from_path` cache both feed the same `lookup` impl, so a
    /// regression in the underlying HashMap shape would surface
    /// here before any test that depends on the cache.
    #[test]
    fn lookup_returns_inserted_kva() {
        let mut symbols = HashMap::new();
        symbols.insert("scx_root".to_string(), 0xffff_8000_0000_4000u64);
        symbols.insert(
            "ktstr_err_exit_detected".to_string(),
            0xffff_8000_0000_8000u64,
        );
        let cache = VmlinuxSymbolCache::from_symbols_for_test(symbols);
        assert_eq!(cache.lookup("scx_root"), Some(0xffff_8000_0000_4000u64));
        assert_eq!(
            cache.lookup("ktstr_err_exit_detected"),
            Some(0xffff_8000_0000_8000u64)
        );
    }

    /// Lookup of an absent symbol returns `None`. Pins the
    /// contract `arm_user_watchpoint` relies on at the
    /// `.ok_or_else` site — without `None`, a hostile WATCH tag
    /// referencing an unknown symbol would silently arm a slot
    /// at a default KVA instead of surfacing a diagnostic.
    #[test]
    fn lookup_returns_none_for_missing_symbol() {
        let mut symbols = HashMap::new();
        symbols.insert("scx_root".to_string(), 0xffff_8000_0000_4000u64);
        let cache = VmlinuxSymbolCache::from_symbols_for_test(symbols);
        assert_eq!(cache.lookup("nonexistent_symbol"), None);
        assert_eq!(cache.lookup(""), None);
    }

    /// Duplicate symbol inserts collapse to the last-written
    /// value. Mirrors `from_path`'s symbol-table walk where two
    /// ELF symbols sharing a name (a weak default and a strong
    /// override, or duplicates produced by inline static
    /// hoisting) flow through `HashMap::insert` and the second
    /// insert wins. The cache exposes a single KVA per name to
    /// `arm_user_watchpoint`, so a regression that switched to
    /// first-write-wins (e.g. `entry().or_insert`) would surface
    /// as a different KVA than the linker's final binding.
    #[test]
    fn duplicate_symbol_resolves_to_last_inserted() {
        // HashMap::insert returns Some(previous) on a duplicate
        // key — the cache uses the second value. Build the map
        // by hand to stage the exact pre/post state the
        // production walk exposes.
        let mut symbols = HashMap::new();
        symbols.insert("dup_sym".to_string(), 0x1000u64);
        let prior = symbols.insert("dup_sym".to_string(), 0x2000u64);
        assert_eq!(prior, Some(0x1000u64));
        let cache = VmlinuxSymbolCache::from_symbols_for_test(symbols);
        assert_eq!(cache.lookup("dup_sym"), Some(0x2000u64));
    }

    /// Empty-cache lookup returns `None` for every query. Pins
    /// the early-coord-init path where `from_path` succeeded but
    /// the symbol table happened to be empty (zero-symbol ELF
    /// blobs are pathological but possible) — every WATCH would
    /// fail-soft via the `None` arm rather than panicking.
    #[test]
    fn empty_cache_lookup_always_none() {
        let cache = VmlinuxSymbolCache::from_symbols_for_test(HashMap::new());
        assert_eq!(cache.lookup("any"), None);
        assert_eq!(cache.lookup("scx_root"), None);
    }
}

#[cfg(test)]
mod arm_user_watchpoint_tests {
    //! Unit coverage for [`arm_user_watchpoint`].
    //!
    //! The function publishes a resolved KVA into one of the three
    //! user watchpoint slots and kicks every vCPU thread out of
    //! `KVM_RUN`. The kick path walks `ap_pthreads`, `ap_ies`,
    //! `ap_alive`, and the BSP triple — and the BSP-touching paths
    //! are ALL gated on `bsp_alive.load(Acquire)`. When `ap_pthreads`
    //! is empty AND `bsp_alive` reads `false`, no `pthread_kill`
    //! and no `ie.set` runs — exactly what these tests need to
    //! exercise the symbol/alignment/slot-allocation logic in
    //! isolation, without spawning real vCPU threads or holding
    //! live `ImmediateExitHandle`s.
    //!
    //! The contracts pinned here:
    //!   * Unaligned KVA (low 2 bits set) is rejected with a
    //!     diagnostic. x86_64 DR_LEN_4 and aarch64 DBGWVR both
    //!     require 4-byte alignment — a non-aligned target would
    //!     silently mis-program the hardware register on the next
    //!     `KVM_SET_GUEST_DEBUG`.
    //!   * Missing symbol is rejected. `VmlinuxSymbolCache::lookup`
    //!     returns `None` and the `.ok_or_else` arm bubbles a
    //!     diagnostic.
    //!   * Zero KVA is rejected explicitly. The slot's
    //!     `request_kva == 0` is the free-slot sentinel; arming
    //!     with 0 would publish a no-op store the vCPU would
    //!     ignore, leaving a half-armed slot. The explicit
    //!     zero-check at the head of the function rejects this
    //!     case so the caller surfaces a diagnostic.
    //!   * A successful arm consumes the lowest free slot,
    //!     publishes the resolved KVA into `request_kva`, sets the
    //!     `tag`, and flips `any_armed`.
    //!   * Slots are allocated in index order — a fresh request
    //!     after a prior arm goes into the next free slot, not
    //!     the first slot.
    //!   * Slot exhaustion (all three slots occupied) returns an
    //!     error rather than silently overwriting an earlier arm.
    //!   * Slot reuse: clearing `request_kva` to 0 (the
    //!     coordinator's reset path on a slot fire) makes the
    //!     slot available again on the next arm.
    use super::{KERNEL_HALF_CANONICAL_4LEVEL, VmlinuxSymbolCache, arm_user_watchpoint};
    use crate::vmm::vcpu::WatchpointArm;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Build a `bsp_alive=false` Arc. The kick path's
    /// `bsp_alive.load(Acquire)` reads `false`, so the BSP
    /// `ie.set(1)` and `pthread_kill` calls are skipped — exactly
    /// what the tests want when no real BSP thread exists.
    fn dead_bsp() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    /// Build a single-symbol cache for the named symbol/KVA pair.
    fn cache_with(name: &str, kva: u64) -> VmlinuxSymbolCache {
        let mut m = HashMap::new();
        m.insert(name.to_string(), kva);
        VmlinuxSymbolCache::from_symbols_for_test(m)
    }

    /// Symbol with a low-bit-set KVA (e.g. 0x..._4001) is rejected.
    /// 4-byte alignment is required by both x86_64 DR_LEN_4 and
    /// aarch64 DBGWVR; without the check the hardware register
    /// load would silently round the address and watch the wrong
    /// 4-byte window.
    ///
    /// `kaslr_offset != 0` so the new fail-loud gate (at the head
    /// of the function, refusing high-half link_kva + zero offset)
    /// doesn't pre-empt the alignment check this test pins.
    #[test]
    fn rejects_unaligned_kva() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let cache = cache_with("misaligned_sym", 0xffff_8000_0000_4001u64);
        let bsp_alive = dead_bsp();
        let err = arm_user_watchpoint(
            &wp,
            &cache,
            "misaligned_sym",
            0x1000_0000,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap_err();
        assert!(
            err.contains("4-byte aligned") || err.contains("4-byte"),
            "unaligned-KVA error must mention 4-byte alignment, got: {err}"
        );
        // The slot must NOT have been written — a future arm
        // with a valid KVA goes into slot 0 because no slot was
        // consumed by the rejected request.
        for slot in &wp.user {
            assert_eq!(
                slot.request_kva.load(Ordering::Acquire),
                0,
                "rejected arm must not write request_kva"
            );
        }
    }

    /// Unknown symbol is rejected by the `.ok_or_else` arm. The
    /// cache contains a different name, so `lookup` returns `None`
    /// and the function bubbles a diagnostic without consuming a
    /// slot.
    #[test]
    fn rejects_missing_symbol() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let cache = cache_with("present_sym", 0xffff_8000_0000_4000u64);
        let bsp_alive = dead_bsp();
        let err = arm_user_watchpoint(
            &wp,
            &cache,
            "absent_sym",
            0,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap_err();
        assert!(
            err.contains("absent_sym"),
            "missing-symbol error must name the symbol, got: {err}"
        );
        for slot in &wp.user {
            assert_eq!(
                slot.request_kva.load(Ordering::Acquire),
                0,
                "rejected arm must not write request_kva"
            );
        }
    }

    /// KVA of 0 is rejected. The slot's `request_kva == 0` is the
    /// free-slot sentinel for both the per-vCPU
    /// `self_arm_watchpoint` short-circuit AND the free-slot scan
    /// at the top of `arm_user_watchpoint`; arming a slot with 0
    /// would publish a no-op store that the vCPU ignores, leaving
    /// the slot in a half-armed state (tag populated, no DR
    /// programmed, slot still appears free to the next arm).
    #[test]
    fn rejects_zero_kva_explicitly() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let cache = cache_with("zero_sym", 0);
        let bsp_alive = dead_bsp();
        let err = arm_user_watchpoint(
            &wp,
            &cache,
            "zero_sym",
            0,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap_err();
        assert!(
            err.contains("KVA 0") || err.contains("zero_sym"),
            "zero-KVA error must mention the symbol or zero, got: {err}"
        );
        for slot in &wp.user {
            assert_eq!(
                slot.request_kva.load(Ordering::Acquire),
                0,
                "rejected zero-KVA arm must not write request_kva"
            );
        }
    }

    /// Successful arm consumes the lowest free slot, publishes
    /// the KVA, sets the tag, and flips the any_armed gate. A
    /// fresh `WatchpointArm` has every slot's `request_kva == 0`
    /// (the free sentinel), so the first arm lands in slot 0
    /// (DR1 on x86_64 / watchpoint 1 on aarch64).
    ///
    /// `kaslr_offset != 0` so the fail-loud high-half-zero-offset
    /// gate doesn't fire; the stored `request_kva` is
    /// `link_kva + kaslr_offset` (post-slide).
    #[test]
    fn successful_arm_consumes_first_free_slot() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let link_kva = 0xffff_8000_0000_4000u64;
        let kaslr_offset = 0x1000_0000u64;
        let expected_post_slide_kva = link_kva + kaslr_offset;
        let cache = cache_with("scx_root", link_kva);
        let bsp_alive = dead_bsp();
        let idx = arm_user_watchpoint(
            &wp,
            &cache,
            "scx_root",
            kaslr_offset,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .expect("aligned valid symbol must arm");
        assert_eq!(idx, 0, "first free slot is index 0");
        assert_eq!(
            wp.user[0].request_kva.load(Ordering::Acquire),
            expected_post_slide_kva,
            "slot 0 must hold the resolved post-slide KVA"
        );
        let tag = wp.user[0].tag.lock().unwrap().clone();
        assert_eq!(tag, "scx_root", "slot 0 tag must match symbol name");
        assert_eq!(
            wp.any_armed.load(Ordering::Relaxed),
            1,
            "any_armed gate must flip to 1 after successful arm"
        );
        // Sibling slots untouched.
        for sibling in 1..3 {
            assert_eq!(
                wp.user[sibling].request_kva.load(Ordering::Acquire),
                0,
                "sibling slot {sibling} must remain unarmed"
            );
        }
    }

    /// Two successive arms land in slots 0 then 1 — the free-slot
    /// scan walks `user[..]` in index order. Pins the allocation
    /// strategy so a regression to a non-deterministic order
    /// (e.g. searching from the back, picking the largest gap)
    /// surfaces here.
    ///
    /// `kaslr_offset != 0` so the fail-loud high-half-zero-offset
    /// gate doesn't fire; the stored `request_kva` is
    /// `link_kva + kaslr_offset` (post-slide).
    #[test]
    fn arms_consume_slots_in_index_order() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let bsp_alive = dead_bsp();
        let kaslr_offset = 0x1000_0000u64;
        let mut symbols = HashMap::new();
        symbols.insert("sym_a".to_string(), 0xffff_8000_0000_4000u64);
        symbols.insert("sym_b".to_string(), 0xffff_8000_0000_5000u64);
        let cache = VmlinuxSymbolCache::from_symbols_for_test(symbols);
        let i0 = arm_user_watchpoint(
            &wp,
            &cache,
            "sym_a",
            kaslr_offset,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap();
        let i1 = arm_user_watchpoint(
            &wp,
            &cache,
            "sym_b",
            kaslr_offset,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap();
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(
            wp.user[0].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_4000u64 + kaslr_offset
        );
        assert_eq!(
            wp.user[1].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_5000u64 + kaslr_offset
        );
        assert_eq!(
            wp.user[2].request_kva.load(Ordering::Acquire),
            0,
            "third slot must remain unarmed when only two arms ran"
        );
    }

    /// All three user slots filled — the next arm returns an
    /// error rather than silently overwriting one of the prior
    /// arms. Pins the cap-check at the top of the function: the
    /// production code MUST refuse to enroll more than the
    /// hardware register count (3 user watchpoints across both
    /// x86_64 and aarch64 in this codebase).
    ///
    /// `kaslr_offset != 0` so the fail-loud high-half-zero-offset
    /// gate doesn't fire; the stored `request_kva` is
    /// `link_kva + kaslr_offset` (post-slide).
    #[test]
    fn arm_returns_error_when_all_slots_occupied() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let bsp_alive = dead_bsp();
        let kaslr_offset = 0x1000_0000u64;
        let mut symbols = HashMap::new();
        symbols.insert("sym_a".to_string(), 0xffff_8000_0000_4000u64);
        symbols.insert("sym_b".to_string(), 0xffff_8000_0000_5000u64);
        symbols.insert("sym_c".to_string(), 0xffff_8000_0000_6000u64);
        symbols.insert("sym_d".to_string(), 0xffff_8000_0000_7000u64);
        let cache = VmlinuxSymbolCache::from_symbols_for_test(symbols);
        for sym in &["sym_a", "sym_b", "sym_c"] {
            arm_user_watchpoint(
                &wp,
                &cache,
                sym,
                kaslr_offset,
                &[],
                &[],
                &[],
                0 as libc::pthread_t,
                None,
                &bsp_alive,
            )
            .expect("first three arms succeed");
        }
        let err = arm_user_watchpoint(
            &wp,
            &cache,
            "sym_d",
            kaslr_offset,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap_err();
        assert!(
            err.contains("no free user watchpoint slot")
                || err.contains("slots 1..=3 all occupied"),
            "exhaustion error must mention slot capacity, got: {err}"
        );
        // Prior arms still in place — the failed arm did NOT
        // overwrite any earlier slot.
        assert_eq!(
            wp.user[0].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_4000u64 + kaslr_offset
        );
        assert_eq!(
            wp.user[1].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_5000u64 + kaslr_offset
        );
        assert_eq!(
            wp.user[2].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_6000u64 + kaslr_offset
        );
    }

    /// Slot reuse: after the coordinator's reset path clears a
    /// slot's `request_kva` to 0 (the free sentinel), the next
    /// arm must land in that newly-free slot. Pins the contract
    /// from the slot-reuse fix — a slot fire that does NOT clear
    /// `request_kva` would permanently strand the slot. This
    /// test stages a 3-slot fill, clears slot 1's KVA in place,
    /// then re-arms and verifies the new arm lands in slot 1
    /// (the freed middle slot, not slot 3 which is still full).
    ///
    /// `kaslr_offset != 0` so the fail-loud high-half-zero-offset
    /// gate doesn't fire; the stored `request_kva` is
    /// `link_kva + kaslr_offset` (post-slide).
    #[test]
    fn slot_becomes_reusable_after_request_kva_cleared() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let bsp_alive = dead_bsp();
        let kaslr_offset = 0x1000_0000u64;
        let mut symbols = HashMap::new();
        symbols.insert("sym_a".to_string(), 0xffff_8000_0000_4000u64);
        symbols.insert("sym_b".to_string(), 0xffff_8000_0000_5000u64);
        symbols.insert("sym_c".to_string(), 0xffff_8000_0000_6000u64);
        symbols.insert("sym_d".to_string(), 0xffff_8000_0000_8000u64);
        let cache = VmlinuxSymbolCache::from_symbols_for_test(symbols);
        for sym in &["sym_a", "sym_b", "sym_c"] {
            arm_user_watchpoint(
                &wp,
                &cache,
                sym,
                kaslr_offset,
                &[],
                &[],
                &[],
                0 as libc::pthread_t,
                None,
                &bsp_alive,
            )
            .unwrap();
        }
        // Simulate the coordinator's slot-fire reset for slot 1.
        // Production performs this `Release` store inside the
        // freeze coordinator's `freeze_and_capture` path after a
        // user-slot dump completes; the test models the
        // post-reset state directly so the arm path is exercised
        // in isolation.
        wp.user[1].request_kva.store(0, Ordering::Release);
        let idx = arm_user_watchpoint(
            &wp,
            &cache,
            "sym_d",
            kaslr_offset,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap();
        assert_eq!(idx, 1, "freed slot 1 must be reused before slot 2");
        assert_eq!(
            wp.user[1].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_8000u64 + kaslr_offset,
            "freed slot now holds the new post-slide KVA"
        );
        // Slot 0 and slot 2 still hold their original arms.
        assert_eq!(
            wp.user[0].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_4000u64 + kaslr_offset
        );
        assert_eq!(
            wp.user[2].request_kva.load(Ordering::Acquire),
            0xffff_8000_0000_6000u64 + kaslr_offset
        );
    }

    /// arm_user_watchpoint returns a typed `Err` AND logs a
    /// `tracing::warn!` when `kaslr_offset == 0` AND `link_kva`
    /// is in the kernel high-half (`>= KERNEL_HALF_CANONICAL_4LEVEL`)
    /// — the silent-misfire fail-loud path. Without the gate,
    /// `Op::WatchSnapshot` against a kernel-text symbol under
    /// `CONFIG_RANDOMIZE_BASE=y` would arm at the link-time KVA
    /// (wrong under runtime KASLR slide) and never fire, with no
    /// operator-visible signal that anything was wrong. Pin the
    /// Err+warn so a refactor that drops either (or downgrades
    /// the Err back to a silent arm) trips here.
    ///
    /// Each test uses a UNIQUE symbol name + link_kva so the
    /// process-wide warn-once tracker fires per test, regardless
    /// of test execution order.
    #[tracing_test::traced_test]
    #[test]
    fn errs_when_kaslr_zero_against_kernel_high_half() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let cache = cache_with("test_sym_warn_high_half", 0xffff_8000_0000_4040u64);
        let bsp_alive = dead_bsp();
        let err = arm_user_watchpoint(
            &wp,
            &cache,
            "test_sym_warn_high_half",
            0, // kaslr_offset == 0 — triggers the Err+warn
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap_err();
        // Err must name the symbol AND the workaround so the
        // caller has an actionable next step.
        assert!(
            err.contains("test_sym_warn_high_half"),
            "Err must name the symbol, got: {err}"
        );
        assert!(
            err.contains("nokaslr") || err.contains("kaslr = false"),
            "Err must cite the operator workaround, got: {err}"
        );
        // No arm completed — the slot remains the free sentinel.
        for slot in &wp.user {
            assert_eq!(
                slot.request_kva.load(Ordering::Acquire),
                0,
                "fail-loud path must not write request_kva"
            );
        }
        // The warn line also fires (once per unique sym/KVA),
        // naming the symbol and the operator workaround.
        assert!(
            logs_contain("arm_user_watchpoint"),
            "expected arm_user_watchpoint warn marker in log output",
        );
        assert!(
            logs_contain("test_sym_warn_high_half"),
            "warn must name the symbol",
        );
        assert!(
            logs_contain("nokaslr"),
            "warn must cite the nokaslr operator workaround so users have an action",
        );
    }

    /// arm_user_watchpoint returns Err + logs the warn at the
    /// EXACT high-half boundary (`link_kva ==
    /// KERNEL_HALF_CANONICAL_4LEVEL`) — the gate check uses `>=`
    /// (inclusive). A regression that flipped to `>` (exclusive)
    /// would silently allow arming at the boundary. Pin the
    /// inclusive-bound semantic via the Err return.
    #[tracing_test::traced_test]
    #[test]
    fn errs_when_link_kva_exactly_at_kernel_high_half_boundary() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        // Boundary KVA must still be 4-byte aligned (the
        // alignment check at the function tail enforces this);
        // KERNEL_HALF_CANONICAL_4LEVEL (= 0xffff_8000_0000_0000) is
        // 16-byte aligned.
        let cache = cache_with("test_sym_boundary", KERNEL_HALF_CANONICAL_4LEVEL);
        let bsp_alive = dead_bsp();
        // Arm at exactly the boundary. With kaslr_offset == 0,
        // the new gate fires and the call returns Err — the
        // `>=` check IS inclusive of the boundary value.
        let err = arm_user_watchpoint(
            &wp,
            &cache,
            "test_sym_boundary",
            0,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap_err();
        assert!(
            err.contains("test_sym_boundary"),
            "boundary-fail Err must name the symbol, got: {err}"
        );
        assert!(
            logs_contain("arm_user_watchpoint"),
            "link_kva exactly at high-half boundary (>= check is inclusive) must warn",
        );
    }

    /// Inverse: arm_user_watchpoint does NOT log the warn when
    /// `kaslr_offset != 0` (a runtime KASLR slide IS plumbed —
    /// the warn path is for the structural-zero / nokaslr
    /// scenarios). Pinned so a future regression that drops the
    /// conditional (and warns on every arm) trips here.
    #[tracing_test::traced_test]
    #[test]
    fn no_warn_when_kaslr_nonzero() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let cache = cache_with("test_sym_warn_nonzero", 0xffff_8000_0000_5040u64);
        let bsp_alive = dead_bsp();
        let _idx = arm_user_watchpoint(
            &wp,
            &cache,
            "test_sym_warn_nonzero",
            0x1000, // kaslr_offset != 0 — warn must NOT fire
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap();
        assert!(
            !logs_contain("arm_user_watchpoint"),
            "non-zero kaslr_offset must NOT log the silent-misfire warn",
        );
    }

    /// Inverse: arm_user_watchpoint does NOT log the warn when
    /// `kaslr_offset == 0` BUT `link_kva` falls below the kernel
    /// high-half (e.g. a test fixture using low addresses). The
    /// silent-misfire concern only applies to kernel-text symbols
    /// (high-half KVA under `CONFIG_RANDOMIZE_BASE`); low-address
    /// arms are tests / synthetic and not subject to KASLR slide.
    #[tracing_test::traced_test]
    #[test]
    fn no_warn_when_link_kva_below_kernel_high_half() {
        let wp = Arc::new(WatchpointArm::new().unwrap());
        let cache = cache_with("test_sym_warn_low", 0x4000u64); // low address
        let bsp_alive = dead_bsp();
        let _idx = arm_user_watchpoint(
            &wp,
            &cache,
            "test_sym_warn_low",
            0,
            &[],
            &[],
            &[],
            0 as libc::pthread_t,
            None,
            &bsp_alive,
        )
        .unwrap();
        assert!(
            !logs_contain("arm_user_watchpoint"),
            "low-address link_kva must NOT trigger the kernel-text silent-misfire warn",
        );
    }
}
