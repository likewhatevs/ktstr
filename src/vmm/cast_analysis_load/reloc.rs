//! Host-side relocation, kfunc/subprog patching, alloc-size recovery,
//! and `.BTF.ext` func-info parsing for the BPF cast-analysis loader.
//!
//! Extracted from `mod.rs` for module locality; reaches the loader's
//! shared helpers (`find_section`, `btf_str_at`, the `BTF_*` / BPF
//! constants, `BpfInsn`, and the cast-analysis result types) via the
//! `use super::*` glob. Items the staying loader code and the test
//! module consume are `pub(crate)`; purely-internal helpers
//! (`iter_text_relocs`, `ALLOC_SUBPROG_NAMES`, `insn_writes_r1`) stay
//! private.
use super::*;

/// Walk every ELF relocation section in `elf` whose target section
/// is indexed by `section_bases`, validate each relocation's
/// `r_offset`, and yield surviving entries paired with the
/// translated instruction index in the concatenated text stream.
///
/// Used by [`patch_kfunc_calls`], [`build_subprog_returns`], and
/// [`build_datasec_pointers`] — every consumer needs the same
/// "rel section → target program text section → per-reloc
/// `insn_idx`" pipeline. Centralising it here removes the
/// rel-section / bounds / alignment preamble from each consumer
/// and guarantees identical gating across all three.
///
/// # Filtering rules (shared with all consumers)
///
/// 1. The rel section's `sh_info` must point at a real section
///    header. Out-of-range `sh_info` is silently skipped.
/// 2. The target section must appear in `section_bases` — only
///    program text sections we concatenated into `text_concat` are
///    eligible. A rel section targeting `.maps`, `.BTF.ext`, or any
///    non-text section yields no items.
/// 3. The target section header must resolve so we can read its
///    byte size (`sh_size`); a missing header rejects the section.
/// 4. Per-reloc gates: `r_offset` must be a multiple of
///    [`BPF_INSN_SIZE`] (BPF instructions are 8-byte aligned) and
///    strictly less than the target section's byte size. Failures
///    drop the individual relocation.
///
/// Each surviving item is `(insn_idx, reloc)` where `insn_idx`
/// equals `base + r_offset / BPF_INSN_SIZE` (saturating-add against
/// the unlikely-but-possible overflow of a corrupted ELF). The
/// caller then fetches the instruction from `text_concat` (mutably
/// or immutably as needed) and applies its own consumer-specific
/// gates (call opcode, src_reg, datasec lookup, …).
fn iter_text_relocs<'a, 'elf: 'a>(
    elf: &'a goblin::elf::Elf<'elf>,
    section_bases: &'a HashMap<u32, usize>,
) -> impl Iterator<Item = (usize, goblin::elf::Reloc)> + 'a {
    elf.shdr_relocs
        .iter()
        .flat_map(move |(rel_section_idx, reloc_section)| {
            // Resolve which section the relocations target.
            let target_section_idx = elf.section_headers.get(*rel_section_idx).map(|h| h.sh_info);
            // Only program text sections appear in `section_bases`.
            let scope = target_section_idx.and_then(|idx| {
                let base = *section_bases.get(&idx)?;
                let sh = elf.section_headers.get(idx as usize)?;
                Some((base, sh.sh_size as usize))
            });
            // `into_iter` collapses `Option<I>` to an iterator (one
            // pass when `Some`, empty when `None`), so the outer
            // `flat_map` sees the correct shape regardless of
            // whether the rel section was in scope.
            scope
                .into_iter()
                .flat_map(move |(base, section_byte_size)| {
                    reloc_section.iter().filter_map(move |reloc| {
                        let off = reloc.r_offset as usize;
                        if !off.is_multiple_of(BPF_INSN_SIZE) {
                            return None;
                        }
                        if off >= section_byte_size {
                            return None;
                        }
                        let insn_idx = base.saturating_add(off / BPF_INSN_SIZE);
                        Some((insn_idx, reloc))
                    })
                })
        })
}

/// Names of in-tree BPF subprograms whose return values are arena
/// virtual addresses stored in `u64` slots. The cast analyzer's
/// STX-flow path tags any slot the returned value is stored into as
/// an Arena cast finding (resolved via the renderer's
/// [`crate::monitor::btf_render::MemReader::resolve_arena_type`]
/// bridge at chase time).
///
/// Order is alphabetical for readability — the allowlist is
/// consulted by linear scan in [`build_subprog_returns`] (small N,
/// no perf concern). Each entry must be `__always_inline`-d in the
/// scheduler source for the analyzer to see the call site at the
/// stash location; non-inlined helpers move the `STX` of the
/// returned R0 into the helper's own frame (R0 is clobbered at the
/// caller's call site), so the analyzer never sees the tag flow
/// across the call boundary. The non-inlined-allocator warn surfaces at
/// finalize when arena STX evidence is present but no LDX→cast
/// chain landed for any slot, prompting operators to mark missing
/// helpers `__always_inline`.
const ALLOC_SUBPROG_NAMES: &[&str] = &[
    // sdt_alloc lib allocator for per-task / per-cgroup contexts
    // (lib/sdt_alloc.bpf.c). Distinct from `scx_static_alloc_internal`
    // — sdt_alloc adds a per-allocation header (`union sdt_id`)
    // before the payload, but the returned u64 is still an arena
    // VA suitable for STX-flow tagging.
    "scx_alloc_internal",
    // scx-shared static allocator that returns a u64 carrying an
    // arena VA with NO per-allocation header (the slot just holds
    // the start of an arbitrary-typed payload, e.g. `struct
    // scx_cgroup_ctx`). Drives the deferred-resolve arena cast
    // path: the renderer's `resolve_arena_type` bridge resolves
    // the payload type at chase time.
    "scx_static_alloc_internal",
    // The kernel kfunc `bpf_arena_alloc_pages` is intentionally
    // NOT in this allowlist — it is a kfunc (`SHN_UNDEF` /
    // `STT_NOTYPE`), not a subprog, so every gate in
    // [`build_subprog_returns`] (`STT_FUNC`, non-`SHN_UNDEF`,
    // `BPF_PSEUDO_CALL`) rejects it. Arena allocator kfuncs are
    // tagged on the kfunc-side allowlist
    // [`crate::monitor::cast_analysis::ARENA_ALLOC_KFUNC_NAMES`]
    // consulted by [`crate::monitor::cast_analysis::Analyzer::handle_kfunc_call`].
    // Putting `bpf_arena_alloc_pages` here would have been dead code
    // — it failed every gate silently — but kept the wrong impression
    // that subprog detection covered kernel arena allocation.
];

/// Walk every ELF relocation section in `elf` and emit one
/// [`SubprogReturn`] per `BPF_PSEUDO_CALL` site whose resolved
/// subprog name matches the arena-allocator allowlist (see
/// [`ALLOC_SUBPROG_NAMES`]).
///
/// Pre-relocation `.bpf.o` (the form embedded inside an scx-built
/// scheduler binary's `.bpf.objs` section) emits BPF-to-BPF calls
/// to in-tree library subprograms as:
///
/// ```text
///     code = BPF_JMP|BPF_CALL = 0x85
///     dst_reg = 0, src_reg = BPF_PSEUDO_CALL = 1
///     off = 0
///     imm = pc-relative offset to the subprog's first insn
/// ```
///
/// paired with an ELF relocation entry at the call's byte offset
/// pointing to the subprog's `STT_FUNC` symbol. Unlike kfunc
/// calls (`SHN_UNDEF`), library subprogs are linked into the same
/// program text section (or a sibling section with `SHF_EXECINSTR`)
/// — the reloc's symbol's `st_shndx` is non-`SHN_UNDEF` and
/// `st_type == STT_FUNC`. The symbol's name is the subprog's name
/// in the program BTF (clang preserves the C identifier).
///
/// The function does NOT patch any instruction; it only records the
/// call PC for the analyzer to consume. Distinct from
/// [`patch_kfunc_calls`] which rewrites kfunc call sites in place.
///
/// # Errors
///
/// Never fails. Symbol resolve failures, relocations on non-call
/// instructions, missing subprog names — all silent no-ops. The
/// analyzer falls through to the existing shape-inference path.
pub(crate) fn build_subprog_returns(
    text_concat: &[BpfInsn],
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) -> Vec<SubprogReturn> {
    let mut out: Vec<SubprogReturn> = Vec::new();
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item is
    // a relocation that targets a known program text section at an
    // 8-byte-aligned, in-bounds offset; the call-site / symbol /
    // allowlist gates below are subprog-specific.
    for (insn_idx, reloc) in iter_text_relocs(elf, section_bases) {
        let Some(insn) = text_concat.get(insn_idx) else {
            continue;
        };
        // Gate 1: the instruction must be a BPF call site.
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            continue;
        }
        // Gate 2: the call must be a `BPF_PSEUDO_CALL`. Kfunc
        // calls (`BPF_PSEUDO_KFUNC_CALL`) and helper calls
        // (`src_reg == 0`) are not subprog calls.
        if insn.src_reg() != BPF_PSEUDO_CALL {
            continue;
        }
        // Resolve the symbol → name. The symbol must be `STT_FUNC`
        // with a defined section (`st_shndx != SHN_UNDEF`) — that's
        // the in-tree-subprog shape. Extern (kfunc) callsites have
        // `st_shndx == SHN_UNDEF` and are handled by
        // [`patch_kfunc_calls`] separately.
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        const STT_FUNC: u8 = goblin::elf::sym::STT_FUNC;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx == SHN_UNDEF {
            continue;
        }
        if sym.st_type() != STT_FUNC {
            continue;
        }
        let name = match elf.strtab.get_at(sym.st_name) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        // Allowlist match: linear scan over the small list. The
        // names are exact (no prefix / glob); a future change to
        // allow prefix matching would require a dedicated test for
        // cross-allocator name collisions (e.g.
        // `scx_static_alloc_internal_v2`).
        if !ALLOC_SUBPROG_NAMES.contains(&name) {
            continue;
        }
        // For `scx_static_alloc_internal` callers, recover the
        // `size` argument from R1 by scanning backward from the call
        // PC for the most recent `BPF_MOV64_IMM r1, <imm>`. The bump
        // allocator emits no per-slot header, so the renderer's
        // [`crate::monitor::btf_render::MemReader::resolve_arena_type`]
        // bridge has no entry to resolve the payload type id from —
        // size-based BTF matching via
        // [`crate::monitor::sdt_alloc::discover_payload_btf_id`] is
        // the only resolution path. The captured size threads from
        // [`SubprogReturn::alloc_size`] all the way to
        // [`crate::monitor::cast_analysis::CastHit::alloc_size`].
        //
        // For other allocators (e.g. `scx_alloc_internal`) the
        // bridge handles resolution via the per-slot header, so
        // the captured size is not needed and we leave
        // `alloc_size: None` to keep the chase on the bridge path.
        //
        // The lookback is bounded at [`ALLOC_SIZE_LOOKBACK`]
        // instructions: clang's allocator inlining emits the
        // `mov r1, <imm>` immediately before the call in real
        // schedulers, but a small budget tolerates conservative
        // codegen (a constant rematerialized into a different
        // register, then moved into r1) without opening the door
        // to spurious matches from unrelated MOVs many
        // instructions back. A failed lookback yields
        // `alloc_size: None`, falling back to the bridge or the
        // skip-with-reason chase outcome.
        let alloc_size = if name == "scx_static_alloc_internal" {
            recover_alloc_size_from_r1(text_concat, insn_idx)
        } else {
            None
        };
        out.push(SubprogReturn {
            insn_offset: insn_idx,
            alloc_size,
            // Size-only allocators (`scx_static_alloc_internal`,
            // `scx_alloc_internal`): the analyzer tags R0 as
            // `ArenaU64FromAlloc` and the payload struct is resolved at
            // chase time (bridge or size-match). The typed-return
            // upgrade is emitted separately by
            // [`typed_alloc_returns`], which owns the BTF-side
            // resolution.
            return_struct_id: None,
        });
    }
    out
}

/// Name of the single per-task sdt_alloc allocator that
/// `scx_task_alloc` / `scx_task_init` operate on (a global
/// `struct scx_allocator scx_task_allocator` in `lib/sdt_task.bpf.c`).
/// Passed to [`crate::monitor::sdt_alloc::discover_payload_btf_id`] as
/// the `allocator_name` disambiguator so the analyzer resolves the
/// payload struct EXACTLY as the renderer does at chase time (the
/// renderer keys the same allocator var name — see
/// `dump/mod.rs::append_arena_slot_index_for_allocator`).
const SCX_TASK_ALLOCATOR_NAME: &str = "scx_task_allocator";

/// Subprog that returns a per-task sdt_alloc payload as a bare
/// `void __arena *` (`lib/sdt_task.bpf.c::scx_task_alloc`). The
/// analyzer cannot type this return from the callee FuncProto (the
/// declared return is `void *`), so a chase THROUGH the returned
/// pointer would key its STX against an untyped base and drop the
/// finding. [`typed_alloc_returns`] recovers the payload struct id and
/// upgrades the return to `Pointer{struct}` when unambiguous.
const SCX_TASK_ALLOC_NAME: &str = "scx_task_alloc";

/// Subprog that DECLARES the per-task allocator's element size
/// (`lib/sdt_task.bpf.c::scx_task_init(__u64 data_size)`, called once
/// with `sizeof(payload)`). `scx_task_alloc` carries no size argument
/// at its own call site, so the size is recovered from this call's R1
/// immediate. Both operate on the singleton `scx_task_allocator`, so
/// one `scx_task_init(size)` fixes the size for every `scx_task_alloc`
/// return in the object.
const SCX_TASK_INIT_NAME: &str = "scx_task_init";

/// Recover typed allocator returns for `scx_task_alloc`-class calls,
/// where the payload struct id is resolvable but the callee FuncProto
/// return type (`void __arena *`) is not.
///
/// Two-step, evidence-based:
///
/// 1. Recover the per-task allocator element size from the
///    `scx_task_init(<size>)` call's R1 immediate (the ONLY static
///    signal for a size that `scx_task_alloc` sets up at init, not at
///    each alloc). Multiple `scx_task_init` calls with disagreeing
///    sizes, or a size the lookback cannot recover, yield `None` and
///    the whole pass emits nothing — no size, no typed return.
/// 2. Map that size to a payload struct via
///    [`crate::monitor::sdt_alloc::discover_payload_btf_id`] — the SAME
///    size+name match the renderer runs at chase time, so analyzer and
///    renderer never disagree on the id. It returns `0` on ANY
///    ambiguity (zero or multiple size candidates that the name
///    heuristic cannot break); the `!= 0` gate is the correctness
///    boundary. A `0` result emits nothing (the return stays untyped —
///    false negative over false positive).
///
/// Emits one [`SubprogReturn`] per `scx_task_alloc` call site with
/// `return_struct_id: Some(sid)`; the analyzer types R0 = `Pointer{sid}`.
///
/// Determinism: the reloc walk is section/offset ordered and
/// `discover_payload_btf_id` scans BTF ids in order with ordered
/// pattern arms — no HashMap iteration influences the result.
pub(crate) fn typed_alloc_returns(
    text_concat: &[BpfInsn],
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
    btf: &btf_rs::Btf,
) -> Vec<SubprogReturn> {
    // Step 1: recover the per-task allocator element size, requiring a
    // unique agreed value across every `scx_task_init` call site.
    let mut per_task_size: Option<u64> = None;
    for (insn_idx, name) in iter_named_pseudo_calls(text_concat, elf, section_bases) {
        if name != SCX_TASK_INIT_NAME {
            continue;
        }
        let Some(size) = recover_alloc_size_from_r1(text_concat, insn_idx) else {
            // A call whose size we cannot recover makes the per-task
            // size unknown — bail closed rather than guess.
            return Vec::new();
        };
        match per_task_size {
            None => per_task_size = Some(size),
            Some(prev) if prev == size => {}
            // Disagreeing sizes across init calls: ambiguous, bail.
            Some(_) => return Vec::new(),
        }
    }
    let Some(size) = per_task_size else {
        return Vec::new();
    };

    // Step 2: resolve the payload struct. `!= 0` is the len==1
    // correctness boundary enforced inside `discover_payload_btf_id`.
    let choice = crate::monitor::sdt_alloc::discover_payload_btf_id(
        btf,
        size as usize,
        SCX_TASK_ALLOCATOR_NAME,
    );
    if choice.target_type_id == 0 {
        return Vec::new();
    }

    // Emit a typed return for every scx_task_alloc call site.
    let mut out = Vec::new();
    for (insn_idx, name) in iter_named_pseudo_calls(text_concat, elf, section_bases) {
        if name != SCX_TASK_ALLOC_NAME {
            continue;
        }
        out.push(SubprogReturn {
            insn_offset: insn_idx,
            alloc_size: Some(size),
            return_struct_id: Some(choice.target_type_id),
        });
    }
    out
}

/// Yield `(insn_index, subprog_name)` for every `BPF_PSEUDO_CALL` to a
/// defined in-tree `STT_FUNC` subprog — the same call-shape
/// [`build_subprog_returns`] gates on, factored out so
/// [`typed_alloc_returns`] can match callee names without the
/// allocator allowlist. Extern (kfunc) callsites (`SHN_UNDEF`) and
/// non-call relocations are skipped.
fn iter_named_pseudo_calls<'a>(
    text_concat: &'a [BpfInsn],
    elf: &'a goblin::elf::Elf<'a>,
    section_bases: &'a HashMap<u32, usize>,
) -> impl Iterator<Item = (usize, &'a str)> + 'a {
    iter_text_relocs(elf, section_bases).filter_map(move |(insn_idx, reloc)| {
        let insn = text_concat.get(insn_idx)?;
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            return None;
        }
        if insn.src_reg() != BPF_PSEUDO_CALL {
            return None;
        }
        let sym = elf.syms.get(reloc.r_sym)?;
        const STT_FUNC: u8 = goblin::elf::sym::STT_FUNC;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx == SHN_UNDEF || sym.st_type() != STT_FUNC {
            return None;
        }
        let name = elf.strtab.get_at(sym.st_name).filter(|s| !s.is_empty())?;
        Some((insn_idx, name))
    })
}

/// Maximum instructions [`recover_alloc_size_from_r1`] scans backward
/// from a `scx_static_alloc_internal` call site looking for the
/// `BPF_MOV64_IMM r1, <imm>` that materialised R1. Real schedulers
/// emit the MOV adjacent to the call (clang inlines the helper, so
/// the call is preceded by `r1 = sizeof(...)`); 20 instructions is a
/// generous budget that covers a few intervening setup ops without
/// reaching back into unrelated control flow.
pub(crate) const ALLOC_SIZE_LOOKBACK: usize = 20;

/// `BPF_ALU64 | BPF_MOV | BPF_K` opcode byte (`= 0xb7`). Sets
/// `dst_reg = imm` (sign-extended to 64 bits). See linux uapi
/// `bpf.h` and `kernel/bpf/verifier.c` `check_alu_op`. The
/// host-side loader uses this to recognize the
/// `mov rN, <imm>` instructions that clang emits for
/// argument-setup before a BPF-to-BPF subprog call.
pub(crate) const BPF_MOV64_IMM_CODE: u8 = (libbpf_rs::libbpf_sys::BPF_ALU64
    | libbpf_rs::libbpf_sys::BPF_MOV
    | libbpf_rs::libbpf_sys::BPF_K) as u8;

/// Scan backward from `call_pc` in `text` looking for the most recent
/// `BPF_MOV64_IMM r1, <imm>` and return the immediate as a `u64`.
/// Returns `None` when no matching instruction is found within
/// [`ALLOC_SIZE_LOOKBACK`] instructions, when `call_pc` is `0`
/// (no predecessors to scan), or when `call_pc` is out of bounds.
///
/// The scan stops at the MOST RECENT write to R1. If that write is
/// `MOV r1, imm`, its immediate is the value that survived to the
/// call site. If it is any OTHER write to R1 — an ALU/ALU64/LDX/LD
/// into R1 (including MOV-from-register and `LD_IMM64`), or a
/// `BPF_CALL` clobbering caller-saved r1-r5 — the scan returns `None`:
/// an earlier `MOV r1, imm` did not survive that write, so returning
/// its immediate would capture a STALE value. Conservative misses
/// surface as `alloc_size: None` and the chase falls back to the
/// bridge (no static-alloc match), the safe direction. (See
/// [`insn_writes_r1`] for the write classification.)
///
/// `imm` is sign-extended to `u64` via the `i32 -> i64 -> u64`
/// chain so a negative `i32` would surface as a very large `u64`.
/// Real `sizeof` arguments are non-negative; the analyzer's
/// downstream chase (`discover_payload_btf_id`) returns
/// `target_type_id == 0` for impossible payload sizes, so a
/// pathological negative `imm` cannot misrender — it falls back
/// to the bridge or skips.
pub(crate) fn recover_alloc_size_from_r1(text: &[BpfInsn], call_pc: usize) -> Option<u64> {
    if call_pc == 0 {
        return None;
    }
    let start = call_pc.saturating_sub(ALLOC_SIZE_LOOKBACK);
    // Walk from `call_pc - 1` down to `start` (inclusive). The most
    // recent write to R1 decides the result: a `MOV r1, imm` yields
    // the surviving sizeof; any other write to R1 invalidates an
    // earlier immediate, so stop and miss conservatively (bridge
    // fallback) rather than returning a stale value.
    let mut idx = call_pc;
    while idx > start {
        idx -= 1;
        let insn = text.get(idx)?;
        if insn.code == BPF_MOV64_IMM_CODE && insn.dst_reg() == 1 {
            return Some(insn.imm as i64 as u64);
        }
        if insn_writes_r1(insn) {
            return None;
        }
    }
    None
}

/// True if `insn` writes register R1. Used by
/// [`recover_alloc_size_from_r1`] to stop its backward scan at any R1
/// write that is NOT the `MOV r1, imm` it searches for (the caller
/// checks the MOV-imm case first). Covers:
/// - `ALU`/`ALU64`/`LDX`/`LD` with `dst == r1` — MOV-from-register,
///   arithmetic, `LDX`, and `LD_IMM64`;
/// - any `BPF_CALL`, which clobbers caller-saved r1-r5 regardless of
///   its dst field;
/// - `BPF_STX | BPF_ATOMIC` fetch ops (`XCHG` / fetch-arithmetic) that
///   write the pre-op memory value into `src_reg == r1` (kernel
///   `check_atomic_rmw`, include/uapi/linux/bpf.h; the cast analyzer
///   models the same atomic clobber in
///   [`crate::monitor::cast_analysis`]'s `step`). `CMPXCHG` writes r0,
///   and plain (non-FETCH) atomics write only memory, so neither
///   touches r1.
///
/// `BPF_ST` and non-fetch `BPF_STX` write memory, and non-call
/// `BPF_JMP`/`BPF_JMP32` write no GPR, so none stop the scan.
fn insn_writes_r1(insn: &BpfInsn) -> bool {
    use libbpf_rs::libbpf_sys as bs;
    // A call clobbers caller-saved r1-r5 regardless of its dst field.
    if insn.code == cast_analysis_load_consts::BPF_JMP_CALL_CODE {
        return true;
    }
    // BPF instruction class is the low 3 bits of the opcode byte.
    let class = insn.code & 0x07;
    // Atomic read-modify-write with the FETCH bit writes the pre-op
    // value into a register: XCHG / fetch-arithmetic write `src_reg`,
    // CMPXCHG writes r0. So a fetch atomic that is not CMPXCHG with
    // `src == r1` writes r1 — and `src_reg`, not `dst_reg`, carries
    // the written register, so this is checked before the dst gate.
    // Kernel values: BPF_STX=0x03 (class), BPF_ATOMIC=0xc0 (mode
    // bits), BPF_FETCH=0x01 (imm bit), BPF_CMPXCHG imm=0xf1.
    const BPF_ATOMIC_MODE: u8 = 0xc0;
    const BPF_FETCH_BIT: i32 = 0x01;
    const BPF_CMPXCHG_IMM: i32 = 0xf1;
    if class == bs::BPF_STX as u8
        && (insn.code & 0xe0) == BPF_ATOMIC_MODE
        && (insn.imm & BPF_FETCH_BIT) != 0
        && insn.imm != BPF_CMPXCHG_IMM
        && insn.src_reg() == 1
    {
        return true;
    }
    // Register-writing classes with dst == r1.
    if insn.dst_reg() != 1 {
        return false;
    }
    class == bs::BPF_ALU as u8
        || class == bs::BPF_ALU64 as u8
        || class == bs::BPF_LDX as u8
        || class == bs::BPF_LD as u8
}

/// Walk every ELF relocation section in `elf` and emit a
/// [`DatasecPointer`] for each `R_BPF_64_64` reloc that targets a
/// section the program BTF exposes as a `BTF_KIND_DATASEC`
/// (`.bss`, `.data`, `.rodata`, `.data.<name>`, …).
///
/// Pre-relocation `.bpf.o` (the form embedded inside an scx-built
/// scheduler binary's `.bpf.objs` section) emits `BPF_LD_IMM64`
/// references to global variables in `.bss` / `.data` / `.rodata`
/// with `src_reg = 0`; the relocation entry is the only host-side
/// evidence that the LD_IMM64 targets a specific section. Each
/// reloc's `r_offset` (byte offset within the targeted text
/// section) divided by [`BPF_INSN_SIZE`] gives the instruction PC
/// in `text_concat`. The reloc's symbol resolves either to the
/// section symbol itself (`STT_SECTION`, `st_value == 0`) or to a
/// regular `STT_OBJECT` data symbol whose `st_shndx` points at
/// the section. Either way, the section's name keys the BTF
/// lookup that finds the matching `BTF_KIND_DATASEC` id.
///
/// `base_offset` resolution mirrors libbpf's relocation logic.
/// For SHT_REL (the BPF convention — clang emits SHT_REL, not
/// SHT_RELA, for BPF object files), `r_addend` is absent; the
/// offset comes from `LD_IMM64 insn.imm + sym.st_value`. The
/// LD_IMM64's pre-relocation `imm` field carries the per-variable
/// byte offset within the section (clang emits this for
/// `STT_SECTION` symbols). For `STT_OBJECT` symbols clang emits
/// `imm == 0` and the offset comes from `sym.st_value` (the
/// object symbol's address within its section). The function
/// adds both contributions so both clang patterns produce
/// identical annotations.
///
/// # What gets emitted
///
/// - `R_BPF_64_64` (numeric `r_type == 1`): the LD_IMM64-on-text
///   relocation libbpf rewrites to `BPF_PSEUDO_MAP_VALUE`. Other
///   reloc types are not LD_IMM64-on-text and produce no
///   annotation.
/// - The instruction at the resolved PC must be `BPF_LD_IMM64`
///   (`code == BPF_LD | BPF_DW | BPF_IMM = 0x18`). A reloc on a
///   non-LD_IMM64 instruction is malformed input — drop silently.
/// - The target section must resolve to a `BTF_KIND_DATASEC` in
///   the program BTF. `.text` (executable), `.maps` (BPF map
///   definitions, exposed as a different BTF shape), and `.BTF`
///   itself are not datasecs and produce no annotation.
///
/// # Errors
///
/// Never fails. A relocation we cannot parse, a symbol we cannot
/// resolve, a section name absent from BTF, an out-of-range PC —
/// every failure path produces a silent no-op. False negatives
/// are safe; the analyzer leaves the corresponding LD_IMM64
/// destination as Unknown, which falls through to the original
/// pre-integration u64 counter rendering.
pub(crate) fn build_datasec_pointers(
    text_concat: &[BpfInsn],
    btf: &Btf,
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) -> Vec<DatasecPointer> {
    // R_BPF_64_64 = 1 per linux `tools/lib/bpf/libbpf_internal.h`.
    // goblin's reloc constants table does not expose BPF reloc
    // types, so the numeric value is inlined here. Same gating
    // libbpf applies in `bpf_program__record_reloc` (classifies
    // `RELO_DATA`) and `bpf_object__relocate_data`'s `RELO_DATA` arm.
    const R_BPF_64_64: u32 = 1;
    // BPF_LD | BPF_DW | BPF_IMM opcode byte (= 0x18 per linux
    // uapi `bpf.h`). Used to gate the relocation: a reloc against
    // an instruction whose opcode is not LD_IMM64 must not
    // produce a datasec annotation, since the analyzer's BPF_LD
    // arm only applies datasec annotations on this exact opcode.
    let bpf_ld_imm64_code: u8 = (libbpf_rs::libbpf_sys::BPF_LD
        | libbpf_rs::libbpf_sys::BPF_DW
        | libbpf_rs::libbpf_sys::BPF_IMM) as u8;

    let mut out: Vec<DatasecPointer> = Vec::new();
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item
    // is a relocation that targets a known program text section
    // at an 8-byte-aligned, in-bounds offset; the reloc-type /
    // opcode / symbol / BTF-lookup gates below are datasec-specific.
    for (insn_pc, reloc) in iter_text_relocs(elf, section_bases) {
        // Gate 1: only `R_BPF_64_64` produces a datasec annotation.
        // Other reloc types touch different instruction kinds
        // (call sites, ABS32/64 data references) that are not
        // LD_IMM64.
        if reloc.r_type != R_BPF_64_64 {
            continue;
        }
        // Gate 2: the reloc must target a `BPF_LD_IMM64`
        // instruction.
        let Some(insn) = text_concat.get(insn_pc) else {
            continue;
        };
        if insn.code != bpf_ld_imm64_code {
            continue;
        }
        // Resolve the symbol. `r_sym` indexes the ELF symbol
        // table; the symbol's section (`st_shndx`) identifies
        // the target section, and `st_value` contributes to
        // the base offset for `STT_OBJECT` symbols.
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        // SHN_UNDEF / SHN_ABS / SHN_COMMON: symbols not bound to a
        // real section index. None can refer to a datasec section;
        // drop.
        const SHN_UNDEF: usize = 0;
        const SHN_ABS: usize = 0xFFF1;
        const SHN_COMMON: usize = 0xFFF2;
        if sym.st_shndx == SHN_UNDEF || sym.st_shndx == SHN_ABS || sym.st_shndx == SHN_COMMON {
            continue;
        }
        let target_sec_idx = sym.st_shndx;
        // Resolve the target section's name via the ELF section
        // header strtab.
        let target_sh_for_name = match elf.section_headers.get(target_sec_idx) {
            Some(s) => s,
            None => continue,
        };
        let sec_name = match elf.shdr_strtab.get_at(target_sh_for_name.sh_name) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        // Resolve the section name to a `BTF_KIND_DATASEC` id.
        // `Btf::resolve_ids_by_name` returns every id sharing the
        // name; the helper filters for the Datasec kind.
        let Some(datasec_id) = find_datasec_btf_id(btf, sec_name) else {
            continue;
        };
        // Compute base_offset: pre-relocation LD_IMM64 imm (per-
        // variable offset for `STT_SECTION` syms) plus
        // `sym.st_value` (per-object offset for `STT_OBJECT` syms).
        // Both contributions are non-negative in well-formed input;
        // checked_add guards against overflow that could only arise
        // from a corrupt ELF.
        let imm_off = if insn.imm < 0 { 0 } else { insn.imm as u32 };
        if sym.st_value > u32::MAX as u64 {
            continue;
        }
        let sym_off = sym.st_value as u32;
        let Some(base_offset) = imm_off.checked_add(sym_off) else {
            continue;
        };
        out.push(DatasecPointer {
            insn_offset: insn_pc,
            datasec_type_id: datasec_id,
            base_offset,
        });
    }
    out
}

/// Find the `BTF_KIND_DATASEC` id whose name matches `name`. Returns
/// the first matching id; `None` if no Datasec by that name is
/// indexed in the program BTF.
///
/// Section names are unique per BTF (every `.bss` / `.data` /
/// `.rodata` / `.data.<name>` produces exactly one DATASEC), so
/// the first hit is the only hit in well-formed input. Mirrors
/// the name-keyed lookup style of [`find_extern_func_btf_id`].
pub(crate) fn find_datasec_btf_id(btf: &Btf, name: &str) -> Option<u32> {
    let ids = btf.resolve_ids_by_name(name).ok()?;
    for id in ids {
        let Ok(ty) = btf.resolve_type_by_id(id) else {
            continue;
        };
        if let Type::Datasec(_) = ty {
            return Some(id);
        }
    }
    None
}

/// Mirror libbpf's `RELO_EXTERN_CALL` handler on the host side.
///
/// In a pre-relocation `.bpf.o` (the form embedded inside an scx-
/// built scheduler binary's `.bpf.objs` section), every kfunc call
/// site is emitted by clang as:
///
/// ```text
///     code = BPF_JMP|BPF_CALL = 0x85
///     dst_reg = 0, src_reg = BPF_PSEUDO_CALL = 1
///     off = 0
///     imm = -1                ; placeholder filled in by libbpf
/// ```
///
/// paired with an ELF relocation entry at the call's byte offset
/// pointing to an extern symbol (`STT_NOTYPE`, `STB_GLOBAL` or
/// `STB_WEAK`, `st_shndx == SHN_UNDEF`). At kernel-load time, libbpf
/// resolves the symbol's BTF id (the program's own
/// `BTF_KIND_FUNC` whose name matches the symbol) to the kernel's
/// kfunc BTF id, then rewrites `src_reg` to `BPF_PSEUDO_KFUNC_CALL =
/// 2` and `imm` to the resolved id (libbpf
/// `bpf_object__relocate_data`'s `RELO_EXTERN_CALL` arm).
///
/// The cast analyzer never runs at kernel-load time — it operates
/// purely on the on-disk binary. So this function performs the same
/// rewrite host-side, except that the BTF id we patch in is the
/// program-BTF id of the extern `BTF_KIND_FUNC`, not the running
/// kernel's id. That suffices for cast analysis: the analyzer's
/// `crate::monitor::cast_analysis::Analyzer::handle_kfunc_call`
/// resolves `imm` against the same program BTF (it has no kernel
/// BTF here), peels `Func -> FuncProto -> return type` through
/// `Ptr -> Struct/Union`, and types R0 accordingly. The kfunc's
/// program-BTF Func entry shares the same FuncProto a kernel-BTF
/// Func entry would, so the return type is the same.
///
/// # Symbol → BTF FUNC id mapping
///
/// libbpf walks the `.ksyms` `BTF_KIND_DATASEC`, whose
/// [`btf_rs::VarSecinfo`] entries point to the per-kfunc
/// `BTF_KIND_FUNC` types (with `BTF_FUNC_EXTERN` linkage). We don't
/// need to descend the DATASEC explicitly: every FUNC referenced by
/// `.ksyms` is also indexed in the program BTF's name → id map (see
/// `btf_rs::Btf::resolve_ids_by_name`), so a name-keyed lookup is
/// enough. We still filter the result to FUNCs with extern linkage
/// to avoid colliding with a same-named static helper that happens
/// to share the symbol name.
///
/// # What gets patched
///
/// - The instruction must be a `BPF_JMP|BPF_CALL` (code byte
///   `0x85`).
/// - The current `src_reg` must be `BPF_PSEUDO_CALL` (the clang-
///   emitted form). If it is already `BPF_PSEUDO_KFUNC_CALL` (post-
///   relocation form, observed when the scheduler binary embeds a
///   pre-loaded BPF object) we leave it alone — the imm already
///   carries the kernel-BTF id, which means nothing in the program
///   BTF.
/// - The current `imm` must be `-1` (the placeholder libbpf fills
///   in). A non-`-1` imm would mean clang resolved this call to a
///   subprog (BPF-to-BPF call), and we must not steal those.
///
/// All three conditions plus the name-resolves-to-extern-FUNC check
/// must hold before any byte is patched. Anything else is a no-op,
/// preserving the cast analyzer's "false negative is safe; false
/// positive is not" stance.
///
/// # Errors
///
/// This function never fails. An ELF without relocation sections, a
/// relocation pointing into a section we did not concatenate, a
/// symbol we cannot resolve, a name that does not map to an extern
/// FUNC, or a bounds-violating reloc offset all produce silent
/// no-ops. The cast map ends up identical to the pre-patching world
/// for those instructions.
pub(crate) fn patch_kfunc_calls(
    text_concat: &mut [BpfInsn],
    btf: &Btf,
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) {
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item
    // is a relocation that targets a known program text section
    // at an 8-byte-aligned, in-bounds offset; the kfunc-specific
    // gates (call opcode, imm == -1, BPF_PSEUDO_CALL src_reg,
    // extern NOTYPE symbol, BTF Func/extern resolve) are applied
    // here. The iterator borrows `elf` and `section_bases`
    // immutably while we take a disjoint mutable borrow on
    // `text_concat`.
    for (insn_idx, reloc) in iter_text_relocs(elf, section_bases) {
        let Some(insn) = text_concat.get_mut(insn_idx) else {
            continue;
        };
        // Gate 1: the instruction must be a BPF call site.
        // `BPF_JMP|BPF_CALL` = `0x05 | 0x80 = 0x85`. Anything
        // else (LD_IMM64 referencing a typeless ksym, BTF data
        // reloc, …) leaves the slot alone.
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            continue;
        }
        // Gate 2: `imm` must be the libbpf placeholder. A non-`-1`
        // imm means clang already resolved this call to a same-
        // section subprog (BPF_PSEUDO_CALL with a pc-relative imm),
        // and patching it as a kfunc would corrupt subprog dispatch
        // in the analyzer's eyes.
        if insn.imm != -1 {
            continue;
        }
        // Gate 3: src_reg must be the clang-emitted
        // `BPF_PSEUDO_CALL` (1). If the embedded object has
        // already been through libbpf's relocation pass (rare;
        // observed only when a scheduler binary captures a
        // post-load object), `src_reg` is already
        // `BPF_PSEUDO_KFUNC_CALL` and `imm` is the kernel BTF id —
        // we must not overwrite the kernel id with the program's
        // id, because the analyzer would then resolve the call
        // against the wrong BTF universe.
        if insn.src_reg() != BPF_PSEUDO_CALL {
            continue;
        }
        // Resolve the symbol → name. goblin parses the symbol
        // table referenced by the rel section's sh_link via
        // `elf.syms`. The symbol's `st_name` indexes the
        // associated string table (`elf.strtab`).
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        // Match libbpf's `sym_is_extern`: the symbol must be an
        // undefined NOTYPE with global or weak binding. Anything
        // else is a subprog, a static helper, or a data symbol;
        // not a kfunc.
        const STT_NOTYPE: u8 = goblin::elf::sym::STT_NOTYPE;
        const STB_GLOBAL: u8 = goblin::elf::sym::STB_GLOBAL;
        const STB_WEAK: u8 = goblin::elf::sym::STB_WEAK;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx != SHN_UNDEF {
            continue;
        }
        if sym.st_type() != STT_NOTYPE {
            continue;
        }
        let bind = sym.st_bind();
        if bind != STB_GLOBAL && bind != STB_WEAK {
            continue;
        }
        // The string-table interning goblin builds gives us a
        // borrow of the symbol's name without copying.
        let name = match elf.strtab.get_at(sym.st_name) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        // Look up the symbol name in the program BTF. We want a
        // `BTF_KIND_FUNC` with extern linkage (mirroring libbpf's
        // `find_extern_btf_id`). The helper returns every id
        // sharing this name; we accept only Func/extern. A name
        // that resolves to multiple distinct Func ids (impossible
        // in well-formed BPF BTF since extern names are unique)
        // yields the first match — same as libbpf.
        let Some(func_btf_id) = find_extern_func_btf_id(btf, name) else {
            continue;
        };
        // Patch in place. The two changes mirror libbpf's
        // RELO_EXTERN_CALL handler exactly. Note we mutate the
        // packed `regs` byte directly: src_reg occupies the
        // high 4 bits, dst_reg the low 4, and the analyzer's
        // `BpfInsn::src_reg()` accessor reads them back as
        // expected after the rewrite.
        insn.set_src_reg(BPF_PSEUDO_KFUNC_CALL);
        insn.imm = func_btf_id as i32;
    }
}

/// Mirror libbpf's BPF-to-BPF subprog call patching on the host side.
///
/// libbpf-rs's `Linker` leaves every global BPF-to-BPF subprog call
/// as a `BPF_PSEUDO_CALL` with `imm = -1`, paired with an ELF
/// relocation against an `STT_FUNC` symbol whose containing section
/// is one of the program text sections we concatenated into
/// `text_concat`. Without patching, the cast analyzer's
/// `crate::monitor::cast_analysis::Analyzer::run` computes
/// `callee_pc = pc + 1 + insn.imm = pc + 1 + (-1) = pc` and
/// inserts the caller's R1..R5 snapshot into `caller_arg_types`
/// at the call site itself instead of at the callee's entry PC.
/// The downstream lookup at `entries_by_pc` then reseeds R1..R5
/// at every callee entry with `RegState::Unknown`, dropping all
/// inter-procedural typed-pointer flow.
///
/// At kernel-load time libbpf computes
/// `sub_insn_idx = sym.st_value/8 + insn.imm + 1` per
/// `bpf_object__reloc_code` (tools/lib/bpf/libbpf.c). For a
/// global subprog: `sym.st_value` = byte offset of the callee's
/// first instruction within its section; `insn.imm = -1` is the
/// libbpf placeholder; `+1` accounts for the BPF call ABI
/// (next-instruction-relative). The result `sub_insn_idx` is the
/// callee's entry PC in libbpf's appended-into-main-prog
/// instruction stream — the same shape as our `text_concat`,
/// modulo per-section base offsets we tracked in `section_bases`.
///
/// We patch in place so the analyzer's computation lands on the
/// correct callee entry: target `imm` = `callee_pc - call_pc - 1`
/// where both PCs are absolute indices in `text_concat`. After
/// patching, the analyzer's
/// `callee_pc = pc + 1 + insn.imm = call_pc + 1 + (callee_pc - call_pc - 1)`
/// resolves to the actual callee entry PC.
///
/// # What gets patched
///
/// - Instruction must be `BPF_JMP|BPF_CALL` (code byte `0x85`).
/// - Current `src_reg` must be `BPF_PSEUDO_CALL` (1). After
///   [`patch_kfunc_calls`] runs, kfunc call sites have
///   `src_reg == BPF_PSEUDO_KFUNC_CALL` (2) and skip this gate.
/// - Current `imm` must be `-1` (the libbpf placeholder). Static
///   (file-local) subprog calls have `imm` already pointing at the
///   target byte offset and skip this gate — clang's pre-relocation
///   encoding for static subprogs is correct as-is.
/// - Symbol must be `STT_FUNC` and not `SHN_UNDEF`. Extern calls
///   (`STT_NOTYPE`, `SHN_UNDEF`) were already handled by
///   [`patch_kfunc_calls`]; non-FUNC symbols (data, section,
///   notype) cannot be subprog targets.
/// - Symbol's section must appear in `section_bases` — only
///   sections we concatenated are eligible callee containers.
/// - `sym.st_value` must be a multiple of [`BPF_INSN_SIZE`]; a
///   non-aligned offset is malformed input (no real subprog
///   starts on a non-8-byte-aligned boundary).
///
/// All gates plus the section-base lookup must hold before any
/// byte is patched. Anything else is a no-op.
///
/// # Errors
///
/// This function never fails. An ELF without relocation sections,
/// a relocation pointing into a section we did not concatenate, a
/// symbol we cannot resolve, an out-of-range PC, an unaligned
/// `st_value`, an arithmetic overflow on the imm computation —
/// every failure path produces a silent no-op. The cast map ends
/// up identical to the pre-patching world for those instructions.
/// False negatives are safe per the analyzer's "false negative is
/// safe; false positive is not" stance.
pub(crate) fn patch_subprog_calls(
    text_concat: &mut [BpfInsn],
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) {
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item
    // is a relocation that targets a known program text section
    // at an 8-byte-aligned, in-bounds offset; the subprog-specific
    // gates (call opcode, BPF_PSEUDO_CALL src_reg, STT_FUNC defined
    // symbol, callee section in `section_bases`, st_value alignment)
    // are applied here.
    //
    // Every reloc'd `BPF_PSEUDO_CALL` to a defined subprog is rebased
    // — NOT only the `imm == -1` libbpf placeholder. A single embedded
    // object produced by `bpftool gen object` (the shape inside an
    // scx scheduler's `.bpf.objs`) is already partially linked: a
    // cross-section call (e.g. from a `struct_ops.s/…` program section
    // into a `.text` library subprog) carries a linker-resolved `imm`
    // that is PC-relative to the callee's own section, not to our
    // section-header-order concatenation. Skipping those (the old
    // `imm != -1` gate did) left the analyzer's `pc + 1 + imm` landing
    // far past the callee's real entry, so cross-subprog
    // `caller_arg_types` propagation never reached the callee body.
    // Recomputing `imm` from `section_base + st_value/8` is safe for
    // every reloc'd call: it targets the callee's function ENTRY (no
    // addend — you cannot call into the middle of a BPF subprog), and
    // for a same-section call the base cancels so the result matches
    // the already-correct offset. It also puts the callee PC in the
    // SAME basis `parse_btf_ext_func_entries` uses for `FuncEntry`
    // offsets, which is exactly what `caller_arg_types` keys on.
    //
    // Capture `text_concat.len()` once up front so the callee-PC
    // bound check inside the loop body does not collide with the
    // mutable borrow from `text_concat.get_mut(call_pc)`.
    let text_len = text_concat.len();
    for (call_pc, reloc) in iter_text_relocs(elf, section_bases) {
        let Some(insn) = text_concat.get_mut(call_pc) else {
            continue;
        };
        // Gate 1: the instruction must be a BPF call site.
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            continue;
        }
        // Gate 2: src_reg must be the clang-emitted
        // `BPF_PSEUDO_CALL` (1). After [`patch_kfunc_calls`] runs
        // first, kfunc call sites have `src_reg ==
        // BPF_PSEUDO_KFUNC_CALL` (2) and naturally skip this gate.
        if insn.src_reg() != BPF_PSEUDO_CALL {
            continue;
        }
        // Resolve the symbol. Two defined-subprog reloc shapes reach
        // here, distinguished by symbol type:
        //
        // - `STT_FUNC`: the reloc names the callee function directly;
        //   `st_value` is the callee's byte offset within its section.
        //   This is the classic single-.text / libbpf-placeholder
        //   (`imm == -1`) shape.
        // - `STT_SECTION`: the reloc names the callee's *section*
        //   (name empty, `st_value == 0`) and the callee's in-section
        //   instruction index is carried in the call's own `imm` as
        //   `imm + 1`. `bpftool gen object` emits this for a partially
        //   linked cross-section call — e.g. a `struct_ops.s/…` program
        //   calling a `.text` library subprog — because it cannot fold
        //   the two sections into one final PC space. The pre-existing
        //   `imm` is PC-relative to the CALLEE section start, not to our
        //   concatenation, so it must be rebased.
        //
        // Extern kfunc calls (`SHN_UNDEF`) were handled upstream; data
        // / NOTYPE symbols are not subprog targets.
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        const STT_FUNC: u8 = goblin::elf::sym::STT_FUNC;
        const STT_SECTION: u8 = goblin::elf::sym::STT_SECTION;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx == SHN_UNDEF {
            continue;
        }
        let sym_type = sym.st_type();
        if sym_type != STT_FUNC && sym_type != STT_SECTION {
            continue;
        }
        // The callee's section must appear in `section_bases` — only
        // sections we concatenated are valid callee containers. A
        // subprog defined in a section we did not collect (e.g.
        // SHF_EXECINSTR-less PROGBITS, or one whose size is not a
        // multiple of [`BPF_INSN_SIZE`]) cannot be resolved to a callee
        // PC and is skipped silently.
        let callee_sec_idx = sym.st_shndx as u32;
        let Some(&callee_section_base) = section_bases.get(&callee_sec_idx) else {
            continue;
        };
        // In-section instruction index of the callee's entry.
        let callee_insn_in_section = if sym_type == STT_FUNC {
            // Only the libbpf `imm == -1` global-subprog placeholder is
            // rebased on the FUNC-symbol path. A static (file-local)
            // subprog call already carries the correct PC-relative
            // offset in `imm`; recomputing from `st_value` would be a
            // no-op at best and must not clobber a non-placeholder
            // value, so leave it alone.
            if insn.imm != -1 {
                continue;
            }
            // `sym.st_value` is the callee's byte offset within its
            // section (relative to sh_addr, which is 0 for the
            // non-allocated BPF text sections; subtract it defensively
            // for any future allocated-section shape).
            let Some(callee_section) = elf.section_headers.get(callee_sec_idx as usize) else {
                continue;
            };
            let Some(sym_offset_bytes) = sym.st_value.checked_sub(callee_section.sh_addr) else {
                continue;
            };
            let sym_offset_bytes = sym_offset_bytes as usize;
            if !sym_offset_bytes.is_multiple_of(BPF_INSN_SIZE) {
                continue;
            }
            sym_offset_bytes / BPF_INSN_SIZE
        } else {
            // STT_SECTION: the callee's in-section instruction index is
            // `imm + 1` (the linker encodes the target as if the call
            // sat at instruction -1 of the callee section). Skip any
            // `imm < 0` — that includes the `-1` placeholder, which on
            // the SECTION path is ambiguous with a genuine call to the
            // section's first instruction; a false negative (leaving
            // the call unresolved) is the safe direction.
            if insn.imm < 0 {
                continue;
            }
            (insn.imm as usize) + 1
        };
        let callee_pc = match callee_section_base.checked_add(callee_insn_in_section) {
            Some(p) => p,
            None => continue,
        };
        // Bound-check the callee PC against text_concat — a
        // st_value past the end of the concatenated stream is a
        // corrupt ELF and would produce a meaningless caller_arg
        // entry; drop silently.
        if callee_pc >= text_len {
            continue;
        }
        // Compute the new `imm` so the analyzer's
        // `pc + 1 + imm` lands on `callee_pc`. The signed-
        // arithmetic conversion handles call sites that point
        // backward (callee earlier in the stream than caller).
        let call_pc_i64 = call_pc as i64;
        let callee_pc_i64 = callee_pc as i64;
        let new_imm = callee_pc_i64 - call_pc_i64 - 1;
        // i32 range guard: a single BPF program text plus its
        // siblings cannot exceed 2^31 instructions in any realistic
        // build, but the source ELF is attacker-influenced so we
        // bound-check rather than silently truncate.
        if new_imm < i32::MIN as i64 || new_imm > i32::MAX as i64 {
            continue;
        }
        insn.imm = new_imm as i32;
    }
}

/// Find the `BTF_KIND_FUNC` whose name matches `name` and whose
/// linkage is extern. Returns `None` if the name does not resolve
/// in the BTF or if the only matching id is not a Func / not extern.
///
/// Mirrors libbpf's `find_extern_btf_id` restricted to FUNC kinds
/// — the cast analyzer only consumes FUNCs (it does not type-
/// recover ksym data variables, just kfunc returns).
pub(crate) fn find_extern_func_btf_id(btf: &Btf, name: &str) -> Option<u32> {
    let ids = btf.resolve_ids_by_name(name).ok()?;
    for id in ids {
        let Ok(ty) = btf.resolve_type_by_id(id) else {
            continue;
        };
        if let Type::Func(f) = ty
            && f.is_extern()
        {
            return Some(id);
        }
    }
    None
}

/// Constants this module needs to talk about BPF instruction wire
/// encoding without pulling the full `cast_analysis` constants set
/// into module scope. Kept private so the loader's surface stays
/// minimal.
pub(crate) mod cast_analysis_load_consts {
    use libbpf_rs::libbpf_sys as bs;
    /// `BPF_JMP | BPF_CALL` opcode byte = `0x85`. The single value
    /// every BPF call instruction (helper, subprog, kfunc) carries
    /// in its `code` field. Used by the kfunc-relocation patcher
    /// to confirm the relocated slot is in fact a call site before
    /// rewriting `src_reg` / `imm`.
    pub(crate) const BPF_JMP_CALL_CODE: u8 = (bs::BPF_JMP | bs::BPF_CALL) as u8;
}

/// Parse `.BTF.ext` and emit one [`FuncEntry`] per `bpf_func_info`
/// record in every section.
///
/// Returns an empty Vec on any malformed input. The format matches
/// `struct btf_ext_header` + per-info-section blobs from
/// `tools/lib/bpf/libbpf_internal.h`:
///
/// ```text
/// btf_ext_header { u16 magic; u8 version; u8 flags; u32 hdr_len;
///                  u32 func_info_off; u32 func_info_len;
///                  u32 line_info_off; u32 line_info_len;
///                  // optional: u32 core_relo_off; u32 core_relo_len; }
/// // After header (at offset hdr_len):
/// // func_info section starts at hdr_len + func_info_off:
/// //   u32 record_size
/// //   repeated for each program section that has func_info:
/// //     btf_ext_info_sec { u32 sec_name_off; u32 num_info; }
/// //     bpf_func_info_min[num_info] { u32 insn_off; u32 type_id; }
/// // ...
/// ```
///
/// `insn_off` is in BYTES; we divide by [`BPF_INSN_SIZE`] (8) to
/// translate to an instruction index. Records are scoped to the
/// section named by `sec_name_off` in the `.BTF` strtab; the
/// instruction index gets offset by that section's base in the
/// concatenated text stream. A section whose name we cannot resolve,
/// or that we did not collect into the concatenated stream (e.g. it
/// lacked SHF_EXECINSTR), is silently skipped — its records produce
/// no [`FuncEntry`].
pub(crate) fn parse_btf_ext_func_entries(
    data: &[u8],
    btf_bytes: &[u8],
    inner_elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) -> Vec<FuncEntry> {
    if data.len() < BTF_EXT_HEADER_MIN_LEN as usize {
        return Vec::new();
    }
    let magic = u16::from_le_bytes([data[0], data[1]]);
    if magic != BTF_MAGIC {
        // Wrong-endian or corrupted; we don't try to byteswap. Cast
        // analysis is best-effort.
        return Vec::new();
    }
    // data[2] = version, data[3] = flags — not consulted; the
    // wire layout is documented in the BTF_EXT_HEADER_MIN_LEN comment.
    let hdr_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let func_info_off = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let func_info_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    if hdr_len < BTF_EXT_HEADER_MIN_LEN || (hdr_len as usize) > data.len() {
        return Vec::new();
    }
    if func_info_len == 0 {
        return Vec::new();
    }
    // The func_info data starts at `hdr_len + func_info_off` and runs
    // for `func_info_len` bytes. Bound-check that whole window.
    let info_start = (hdr_len as usize).checked_add(func_info_off as usize);
    let info_end = info_start.and_then(|s| s.checked_add(func_info_len as usize));
    let (info_start, info_end) = match (info_start, info_end) {
        (Some(s), Some(e)) => (s, e),
        _ => return Vec::new(),
    };
    if info_end > data.len() {
        return Vec::new();
    }
    let info = &data[info_start..info_end];
    if info.len() < 4 {
        return Vec::new();
    }
    let record_size = u32::from_le_bytes([info[0], info[1], info[2], info[3]]) as usize;
    // Minimum bpf_func_info layout is { u32 insn_off; u32 type_id; }
    // — 8 bytes. Newer kernels may pad to a larger record_size; we
    // only consume the first 8 bytes of each record (`insn_off` and
    // `type_id`) and skip the rest, mirroring `bpf_func_info_min` in
    // libbpf_internal.h.
    if record_size < 8 {
        return Vec::new();
    }
    let mut cursor = 4usize;
    let mut out: Vec<FuncEntry> = Vec::new();
    while cursor + 8 <= info.len() {
        let sec_name_off = u32::from_le_bytes([
            info[cursor],
            info[cursor + 1],
            info[cursor + 2],
            info[cursor + 3],
        ]);
        let num_info = u32::from_le_bytes([
            info[cursor + 4],
            info[cursor + 5],
            info[cursor + 6],
            info[cursor + 7],
        ]) as usize;
        cursor += 8;
        let records_bytes = num_info.saturating_mul(record_size);
        match cursor.checked_add(records_bytes) {
            Some(end) if end <= info.len() => {}
            _ => break,
        }
        // Resolve section name via the BTF string table — per kernel
        // libbpf `bpf_object__init_btf` (`btf__name_by_offset`), `.BTF.ext`
        // `sec_name_off` indexes the BTF strtab, NOT the ELF
        // section-header strtab. The BTF strtab starts at
        // `hdr_len + str_off` within the `.BTF` blob.
        let sec_name = match btf_str_at(btf_bytes, sec_name_off) {
            Some(s) => s,
            None => {
                cursor += records_bytes;
                continue;
            }
        };
        let sec_idx = match find_section(inner_elf, sec_name) {
            Some(i) => i as u32,
            None => {
                cursor += records_bytes;
                continue;
            }
        };
        let base = match section_bases.get(&sec_idx) {
            Some(b) => *b,
            None => {
                cursor += records_bytes;
                continue;
            }
        };
        for i in 0..num_info {
            let rec_off = cursor + i * record_size;
            // Read the first 8 bytes (`bpf_func_info_min`); ignore
            // any trailing padding in newer record layouts.
            let insn_off = u32::from_le_bytes([
                info[rec_off],
                info[rec_off + 1],
                info[rec_off + 2],
                info[rec_off + 3],
            ]) as usize;
            let type_id = u32::from_le_bytes([
                info[rec_off + 4],
                info[rec_off + 5],
                info[rec_off + 6],
                info[rec_off + 7],
            ]);
            // insn_off is in BYTES per libbpf docs; translate to an
            // instruction index. A non-multiple-of-8 byte offset is
            // malformed (no real BPF function starts on a non-aligned
            // boundary); skip silently — false negative is safe.
            if !insn_off.is_multiple_of(BPF_INSN_SIZE) {
                continue;
            }
            let entry_idx = base.saturating_add(insn_off / BPF_INSN_SIZE);
            out.push(FuncEntry {
                insn_offset: entry_idx,
                func_proto_id: type_id,
            });
        }
        cursor += records_bytes;
    }
    out
}
