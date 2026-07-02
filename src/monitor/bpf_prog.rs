//! Host-side BPF program enumeration via guest physical memory.
//!
//! Walks the kernel's `prog_idr` xarray from the host to discover
//! loaded BPF programs and read verifier stats from `bpf_prog_aux`.
//! No guest cooperation is needed — all reads go through the guest
//! physical memory mapping.

use super::btf_offsets::{BpfMapOffsets, BpfProgOffsets};
use super::idr::{translate_any_kva, xa_load};
use super::reader::{GuestMem, WalkContext};
use super::symbols::text_kva_to_pa_with_base;

/// BPF_PROG_TYPE_STRUCT_OPS from include/uapi/linux/bpf.h.
const BPF_PROG_TYPE_STRUCT_OPS: u32 = 27;

/// Maximum `used_map_cnt` the walker will iterate. The kernel
/// enforces a per-prog limit of 64 used_maps in `kernel/bpf/verifier.c`
/// (`MAX_USED_MAPS = 64`), so a higher value here means the read
/// raced against `bpf_prog_bind_map`'s "increment cnt, then swap
/// pointer" sequence and got a stale-pointer + new-cnt observation.
/// Capping at the kernel's own limit bounds the walk past the old
/// allocation and matches the upper bound a healthy prog can ever
/// reach.
pub const MAX_USED_MAPS: u32 = 64;

/// BPF_OBJ_NAME_LEN from include/linux/bpf.h.
const BPF_OBJ_NAME_LEN: usize = 16;

/// Iterate every alive `BPF_PROG_TYPE_STRUCT_OPS` prog in the
/// kernel's `prog_idr`, invoking `payload` with each prog's
/// `(prog_pa, aux_pa, aux_kva)`. The closure returns `Option<T>`;
/// `Some(value)` is appended to the result vector, `None` skips.
///
/// Encapsulates the `prog_idr` walk shared by every per-struct-ops-
/// prog reader in this module: translate `prog_idr_kva` → `idr_pa`,
/// read the xarray head, iterate ids 0..idr_next (capped at 65536
/// for safety against corrupted reads — a real kernel never
/// approaches that limit), `xa_load` each entry, translate to
/// `prog_pa`, filter on `prog_type == BPF_PROG_TYPE_STRUCT_OPS`,
/// then translate `aux_kva` → `aux_pa`. Translation failures or
/// zero pointers cause the entry to be skipped silently — matches
/// the prior per-walker behavior and is the right policy under
/// race conditions (torn reads from slab recycling) where the
/// alternative is to publish garbage.
fn for_each_struct_ops_prog<T, F>(
    mem: &GuestMem,
    walk: WalkContext,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    start_kernel_map: u64,
    phys_base: u64,
    mut payload: F,
) -> Vec<T>
where
    F: FnMut(u64, u64, u64) -> Option<T>,
{
    let idr_pa = text_kva_to_pa_with_base(prog_idr_kva, start_kernel_map, phys_base);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    // Cap at 64K entries. A real kernel never has millions of BPF
    // programs; a larger `idr_next` means the PA is wrong or the
    // IDR is corrupt. Bounds runaway loops on garbage reads.
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next).min(65536);

    let mut out = Vec::new();
    for id in 0..idr_next {
        let Some(entry) = xa_load(
            mem,
            walk.page_offset,
            xa_head,
            id as u64,
            offsets.xa_node_slots,
            offsets.xa_node_shift,
        ) else {
            continue;
        };
        if entry == 0 {
            continue;
        }
        let Some(prog_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            entry,
            walk.l5,
            walk.tcr_el1,
        ) else {
            continue;
        };
        let prog_type = mem.read_u32(prog_pa, offsets.prog_type);
        if prog_type != BPF_PROG_TYPE_STRUCT_OPS {
            continue;
        }
        let aux_kva = mem.read_u64(prog_pa, offsets.prog_aux);
        if aux_kva == 0 {
            continue;
        }
        let Some(aux_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            aux_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            continue;
        };
        if let Some(value) = payload(prog_pa, aux_pa, aux_kva) {
            out.push(value);
        }
    }
    out
}

/// Per-program BPF verifier statistics collected from the host.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgVerifierStats {
    /// Program name as registered with the kernel.
    pub name: String,
    /// Instructions processed by the verifier (path-exploration count,
    /// not static program size), from `bpf_prog_aux->verified_insns`.
    pub verified_insns: u32,
}

/// Enumerate struct_ops BPF programs from the kernel's `prog_idr`.
///
/// Reads `prog_idr` from guest memory, walks the xarray, and for
/// each `bpf_prog` with `type == BPF_PROG_TYPE_STRUCT_OPS`, reads
/// `aux->verified_insns` and `aux->name`. `start_kernel_map` is the
/// runtime kernel image base used to translate `prog_idr_kva` to a
/// guest physical address.
pub(crate) fn find_struct_ops_progs(
    mem: &GuestMem,
    walk: WalkContext,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    start_kernel_map: u64,
    phys_base: u64,
) -> Vec<ProgVerifierStats> {
    for_each_struct_ops_prog(
        mem,
        walk,
        prog_idr_kva,
        offsets,
        start_kernel_map,
        phys_base,
        |_prog_pa, aux_pa, _aux_kva| {
            let verified_insns = mem.read_u32(aux_pa, offsets.aux_verified_insns);
            let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
            mem.read_bytes(aux_pa + offsets.aux_name as u64, &mut name_buf);
            let name_len = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(BPF_OBJ_NAME_LEN);
            let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();
            Some(ProgVerifierStats {
                name,
                verified_insns,
            })
        },
    )
}

/// Target-free active-scheduler walker. See trait method
/// [`BpfProgAccessor::find_active_struct_ops_obj_no_target`] for the
/// motivation (Phase 0 sched_kva==value_kva equality is broken on
/// kernels where `struct scx_sched` allocates fresh and copies
/// `sched_ext_ops` into its embedded `ops` field).
///
/// Walks `prog_idr` for the FIRST `BPF_PROG_TYPE_STRUCT_OPS` prog
/// whose `aux->used_maps` carries a sibling `<obj>.bss/.data/.rodata`
/// global-section map. Returns that prog's obj prefix + full
/// used_map_kvas snapshot. Returns `None` when no such prog exists
/// (no scheduler attached, or only non-libbpf STRUCT_OPS subsystems
/// active).
///
/// **Threat model: ktstr guest VM is single-tenant.** This walker
/// returns the FIRST match in `prog_idr` iteration order — it does
/// not assert uniqueness. ktstr-loaded guests are minimal (only
/// scx-ktstr runs), so in practice only the live sched_ext
/// scheduler's prog satisfies the filter. Two reinforcing reasons:
///
/// 1. Sched_ext is the only struct_ops subsystem ktstr loads.
/// 2. The kernel enforces single-ENABLE for sched_ext: the enable
///    path rejects a second scheduler while one is already enabled
///    (pre-6.16: `scx_ops_enable_state() != SCX_OPS_DISABLED` ->
///    `-EBUSY`; the accessor and states were renamed on later
///    kernels -- 6.16's `scx_enable_state()`/`SCX_DISABLED`), so at
///    most one sched_ext STRUCT_OPS prog is ENABLED at a time.
///    This does NOT by itself guarantee the OLD prog has left
///    `prog_idr` before the NEW prog is added: a detached struct_ops
///    prog leaves `prog_idr` only when its owning struct_ops MAP's
///    last userspace fd closes AND an RCU grace elapses (map free is
///    RCU-deferred, see `kernel/bpf/bpf_struct_ops.c`), and the
///    kernel does not serialize old-removal before new-add. The
///    single-alive-prog property this walker relies on is therefore
///    ktstr's swap sequencing -- `Op::ReplaceScheduler` kills the
///    outgoing scheduler and waits for its process to exit (closing
///    the outgoing map's fds) before loading the next -- not a kernel
///    ordering invariant.
///
/// If a future setup loads non-sched_ext libbpf-named STRUCT_OPS
/// progs (e.g. `tcp_congestion_ops`), this filter would need to also
/// gate on `aux->btf` matching the sched_ext_ops btf type id.
///
/// Standard `prog_idr` walk: read xa_head → iterate ids 0..idr_next
/// → translate each prog kva → filter to STRUCT_OPS → read aux's
/// used_maps → derive obj prefix from any global-section sibling
/// map.
pub(crate) fn find_active_struct_ops_obj_no_target(
    mem: &GuestMem,
    walk: WalkContext,
    prog_idr_kva: u64,
    prog_offsets: &BpfProgOffsets,
    map_offsets: &BpfMapOffsets,
    start_kernel_map: u64,
    phys_base: u64,
) -> Option<ActiveObjMatch> {
    // Returns the FIRST matching prog's ActiveObjMatch via Some,
    // skips non-matching progs with None. `into_iter().next()`
    // extracts that single match below.
    for_each_struct_ops_prog(
        mem,
        walk,
        prog_idr_kva,
        prog_offsets,
        start_kernel_map,
        phys_base,
        |_prog_pa, aux_pa, _aux_kva| {
            // Read used_maps pointer FIRST, then cnt — pairs with
            // bpf_prog_bind_map's cnt-then-pointer mutation order
            // (kernel/bpf/syscall.c): the kernel bumps cnt before
            // swapping the pointer, so cnt-then-pointer reads
            // would index past the old allocation on a
            // mid-mutation read. pointer-then-cnt observes cnt ≤
            // pointer's slot count. Safe under freeze-rendezvous
            // (vCPUs paused) but the protocol is defense-in-depth
            // for any out-of-freeze caller.
            let used_maps_kva = mem.read_u64(aux_pa, prog_offsets.aux_used_maps);
            if used_maps_kva == 0 {
                return None;
            }
            let used_map_cnt = mem
                .read_u32(aux_pa, prog_offsets.aux_used_map_cnt)
                .min(MAX_USED_MAPS);
            if used_map_cnt == 0 {
                return None;
            }
            let used_maps_pa = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                used_maps_kva,
                walk.l5,
                walk.tcr_el1,
            )?;

            // Snapshot every non-zero used_maps entry (downstream
            // disambiguation needs the full set as the KVA
            // whitelist).
            let mut entries: Vec<u64> = Vec::with_capacity(used_map_cnt as usize);
            for i in 0..used_map_cnt {
                let entry_kva = mem.read_u64(used_maps_pa, (i as usize) * 8);
                if entry_kva != 0 {
                    entries.push(entry_kva);
                }
            }
            // Find a global-section map in the snapshot and derive
            // the obj prefix. If none, this isn't a libbpf-loaded
            // scheduler prog (could be a different struct_ops
            // subsystem like tcp_congestion_ops without libbpf-
            // named global maps) — return None to skip.
            for &map_kva in &entries {
                let Some(map_pa) = translate_any_kva(
                    mem,
                    walk.cr3_pa,
                    walk.page_offset,
                    map_kva,
                    walk.l5,
                    walk.tcr_el1,
                ) else {
                    continue;
                };
                let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
                mem.read_bytes(map_pa + map_offsets.map_name as u64, &mut name_buf);
                let name_len = name_buf
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(BPF_OBJ_NAME_LEN);
                let Ok(name) = std::str::from_utf8(&name_buf[..name_len]) else {
                    continue;
                };
                if let Some(obj) = extract_global_section_obj_prefix(name) {
                    return Some(ActiveObjMatch {
                        obj_name: obj.to_string(),
                        used_map_kvas: entries,
                    });
                }
            }
            None
        },
    )
    .into_iter()
    .next()
}

/// Result of [`find_active_struct_ops_obj_no_target`]: the matched
/// scheduler's obj prefix plus the full set of used_maps KVAs from
/// the matched prog's aux table. The KVA set lets the consumer
/// distinguish two scheduler instances loaded from the SAME binary
/// (whose maps share an obj prefix but live at distinct kernel
/// addresses) — see
/// [`crate::scenario::snapshot::Snapshot::active`] for the
/// downstream filter that combines (obj-name match AND KVA-in-set)
/// to defend against KVA aliasing across captures.
#[derive(Debug, Clone)]
pub(crate) struct ActiveObjMatch {
    pub obj_name: String,
    pub used_map_kvas: Vec<u64>,
}

/// If `map_name` matches `<obj>.bss` / `<obj>.data` / `<obj>.rodata`
/// (libbpf naming for global-section maps), return `<obj>` (the
/// prefix before the section suffix). Returns None for any other
/// map name (struct_ops `ktstr_ops`, libbpf-named kfunc helpers,
/// hashtables, etc.). Used by the active-obj walker to derive a
/// scheduler obj prefix from a prog's `used_maps` entries.
///
/// The obj prefix returned by this helper is already truncated by
/// the kernel to fit within `BPF_OBJ_NAME_LEN - section_suffix - 1`
/// (libbpf's internal_map_name in tools/lib/bpf/libbpf.c). Callers
/// must match against the same truncated obj prefix when
/// cross-referencing the captured global-section maps.
fn extract_global_section_obj_prefix(map_name: &str) -> Option<&str> {
    for suffix in [".bss", ".data", ".rodata"] {
        if let Some(prefix) = map_name.strip_suffix(suffix)
            && !prefix.is_empty()
        {
            return Some(prefix);
        }
    }
    None
}

#[cfg(test)]
mod extract_global_section_obj_prefix_tests {
    use super::*;

    #[test]
    fn extracts_bss_prefix() {
        assert_eq!(
            extract_global_section_obj_prefix("ktstr.bss"),
            Some("ktstr")
        );
    }

    #[test]
    fn extracts_data_prefix() {
        assert_eq!(
            extract_global_section_obj_prefix("scx_layered.data"),
            Some("scx_layered"),
        );
    }

    #[test]
    fn extracts_rodata_prefix() {
        assert_eq!(
            extract_global_section_obj_prefix("mitosis.rodata"),
            Some("mitosis"),
        );
    }

    #[test]
    fn rejects_struct_ops_map_name() {
        assert_eq!(extract_global_section_obj_prefix("ktstr_ops"), None);
        assert_eq!(extract_global_section_obj_prefix("mitosis_ops"), None);
    }

    #[test]
    fn rejects_unrelated_map_name() {
        assert_eq!(extract_global_section_obj_prefix("scx_per_task"), None);
        assert_eq!(extract_global_section_obj_prefix("bpf_runq"), None);
    }

    #[test]
    fn rejects_empty_prefix_before_suffix() {
        // ".bss" with no obj — degenerate map name; skip.
        assert_eq!(extract_global_section_obj_prefix(".bss"), None);
    }
}

/// Per-program runtime stats summed across all CPUs.
///
/// Mirrors the kernel's `struct bpf_prog_stats` (include/linux/filter.h):
/// `cnt` (invocations), `nsecs` (cumulative runtime), `misses` (recursion
/// re-entries skipped via `bpf_prog_inc_misses_counter`,
/// kernel/bpf/syscall.c). All three counters are u64 monotonics summed
/// across the program's per-CPU `bpf_prog_stats` slots.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProgRuntimeStats {
    /// Program name as registered with the kernel.
    pub name: String,
    /// Total invocation count across all CPUs.
    pub cnt: u64,
    /// Total CPU time in nanoseconds across all CPUs.
    pub nsecs: u64,
    /// Total recursion misses across all CPUs. A miss is a re-entry
    /// attempt blocked by the program's per-CPU recursion guard.
    pub misses: u64,
}

impl ProgRuntimeStats {
    /// Mean nanoseconds per invocation: `nsecs / cnt`. Returns
    /// `0.0` when `cnt == 0` (program never ran or counter not
    /// running) so the result never propagates `NaN` / `Infinity`
    /// into downstream `finite_or_zero` filters. Method-only access
    /// (no stored shadow) — recomputed every call from the raw
    /// fields, matching the [`super::super::assert::CgroupStats::wake_latency_tail_ratio`]
    /// derived-ratio convention.
    ///
    /// Unitless-from-bpftop's perspective: bpftop-style triage
    /// reads "ns/call" as the primary cost-per-invocation metric;
    /// surfacing it here lets a failure-dump consumer compare two
    /// programs' per-call cost without dividing the wire counters
    /// manually.
    pub fn ns_per_call(&self) -> f64 {
        if self.cnt > 0 {
            self.nsecs as f64 / self.cnt as f64
        } else {
            0.0
        }
    }

    /// Fraction of invocation attempts blocked by the per-CPU
    /// recursion guard: `misses / (cnt + misses)`. Returns `0.0`
    /// when both counters are zero (no signal); never produces
    /// `NaN` / `Infinity` even on a saturated `cnt + misses`
    /// overflow because `saturating_add` floors at `u64::MAX` and
    /// the resulting denominator is non-zero.
    ///
    /// A non-trivial miss rate signals lock contention or a
    /// misconfigured recursion guard — bpftop-style triage flags
    /// any program with `miss_rate > 0.01` as a hot recursion
    /// path. Method-only access (no stored shadow); the wire
    /// format carries `cnt` and `misses` separately so consumers
    /// who want the raw counts can recover them.
    pub fn miss_rate(&self) -> f64 {
        let total = self.cnt.saturating_add(self.misses);
        if total > 0 {
            self.misses as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for ProgRuntimeStats {
    /// One-line summary used by [`super::dump::FailureDumpReport`]'s
    /// human-readable rendering: name + the three counter sums plus
    /// the bpftop-style derived metrics (ns/call, miss-rate fraction).
    /// Derived metrics elide when their guards fire (cnt==0 or
    /// cnt+misses==0) so a program that never ran renders without
    /// misleading "0.000 ns/call" noise.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: cnt={} nsecs={} misses={}",
            self.name, self.cnt, self.nsecs, self.misses
        )?;
        if self.cnt > 0 {
            // Three decimals on ns/call: bpftop uses two; we add
            // one for sub-microsecond precision since scheduler
            // BPF ops typically run in tens of nanoseconds.
            write!(f, " ns/call={:.3}", self.ns_per_call())?;
        }
        if self.cnt.saturating_add(self.misses) > 0 && self.misses > 0 {
            // Render miss_rate only when there were actual misses
            // — `0.000` would just be noise on healthy programs.
            // Four decimals: a 0.0001 (= 1 in 10K) miss rate is
            // already actionable for a hot scheduler op.
            write!(f, " miss_rate={:.4}", self.miss_rate())?;
        }
        Ok(())
    }
}

/// Walk `prog_idr` and produce per-program runtime stats in a single
/// IDR pass.
///
/// Folds the previous discover-then-read split into one visitor: for
/// each struct_ops program reached via `xa_load`, read
/// `bpf_prog->stats` (per-CPU base) and `bpf_prog_aux->name` and then
/// sum `cnt`/`nsecs`/`misses` across `per_cpu_offsets`. Halves the
/// per-prog kernel-memory reads relative to the prior split (one
/// `prog_idr` walk and one `bpf_prog`/`aux` translate per program
/// instead of two of each).
///
/// `cnt`/`nsecs`/`misses` are u64 monotonic counters per the kernel's
/// `struct bpf_prog_stats` (include/linux/filter.h) — see
/// [`ProgRuntimeStats`] for provenance and the saturation contract.
/// Address translation uses [`translate_any_kva`] so per-CPU pages
/// served from vmalloc'd memory (`pcpu_get_vm_areas`) translate
/// correctly alongside direct-mapping percpu allocations.
pub(crate) fn walk_struct_ops_runtime_stats(
    mem: &GuestMem,
    walk: WalkContext,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    per_cpu_offsets: &[u64],
    start_kernel_map: u64,
    phys_base: u64,
) -> Vec<ProgRuntimeStats> {
    for_each_struct_ops_prog(
        mem,
        walk,
        prog_idr_kva,
        offsets,
        start_kernel_map,
        phys_base,
        |prog_pa, aux_pa, _aux_kva| {
            let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
            mem.read_bytes(aux_pa + offsets.aux_name as u64, &mut name_buf);
            let name_len = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(BPF_OBJ_NAME_LEN);
            let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

            let stats_percpu_kva = mem.read_u64(prog_pa, offsets.prog_stats);
            if stats_percpu_kva == 0 {
                return None;
            }

            // Per-CPU sum. saturating_add prevents the
            // `attempt to add with overflow` panic that's been
            // observed when uninitialized / scrambled per-CPU pages
            // yield near-u64::MAX values; see `ProgRuntimeStats`.
            let mut cnt: u64 = 0;
            let mut nsecs: u64 = 0;
            let mut misses: u64 = 0;
            for (cpu_index, &cpu_off) in per_cpu_offsets.iter().enumerate() {
                // Out-of-range CPU detection: kernel `setup_per_cpu_areas`
                // only writes `__per_cpu_offset[cpu]` for CPUs in
                // `for_each_possible_cpu`, leaving slots beyond
                // `nr_cpu_ids` at the BSS-initialized 0. Real SMP
                // kernels assign each possible CPU a strictly-positive
                // offset for `cpu > 0`; only the BSP (cpu_index == 0)
                // can legitimately observe a zero offset. Skip
                // `cpu_off == 0 && cpu_index > 0` to avoid double-
                // counting CPU 0's stats for every BSS-zero tail slot.
                // Mirrors the guard in
                // [`super::bpf_map::read_percpu_array_value`].
                if cpu_off == 0 && cpu_index > 0 {
                    continue;
                }
                let stats_kva = stats_percpu_kva.wrapping_add(cpu_off);
                if let Some(stats_pa) = translate_any_kva(
                    mem,
                    walk.cr3_pa,
                    walk.page_offset,
                    stats_kva,
                    walk.l5,
                    walk.tcr_el1,
                ) && stats_pa < mem.size()
                {
                    // Batch the three u64 stat reads into one bulk
                    // `read_bytes` covering the contiguous span from
                    // `min(cnt, nsecs, misses)` to `max(...) + 8`. The
                    // kernel's `struct bpf_prog_stats` packs `cnt`,
                    // `nsecs`, and `misses` as adjacent u64_stats_t
                    // (8 bytes each) and the BTF resolver accepts only
                    // layouts where the three fields land in 24
                    // contiguous bytes. The bulk read pays one bounds
                    // check + region resolve instead of three per CPU,
                    // and parses the values from the local buffer
                    // without further volatile loads.
                    let lo = offsets
                        .stats_cnt
                        .min(offsets.stats_nsecs)
                        .min(offsets.stats_misses);
                    let hi = offsets
                        .stats_cnt
                        .max(offsets.stats_nsecs)
                        .max(offsets.stats_misses)
                        + 8;
                    let span = hi - lo;
                    if span <= 64 {
                        let mut buf = [0u8; 64];
                        let n = mem.read_bytes(stats_pa + lo as u64, &mut buf[..span]);
                        if n == span {
                            let parse = |off: usize| -> u64 {
                                let i = off - lo;
                                u64::from_ne_bytes(buf[i..i + 8].try_into().unwrap())
                            };
                            cnt = cnt.saturating_add(parse(offsets.stats_cnt));
                            nsecs = nsecs.saturating_add(parse(offsets.stats_nsecs));
                            misses = misses.saturating_add(parse(offsets.stats_misses));
                        } else {
                            // Partial copy (page straddle / end-of-DRAM)
                            // — fall back to scalar reads to retain the
                            // original semantics.
                            cnt = cnt.saturating_add(mem.read_u64(stats_pa, offsets.stats_cnt));
                            nsecs =
                                nsecs.saturating_add(mem.read_u64(stats_pa, offsets.stats_nsecs));
                            misses =
                                misses.saturating_add(mem.read_u64(stats_pa, offsets.stats_misses));
                        }
                    } else {
                        // Span exceeds the inline buffer. Should be
                        // unreachable for the production
                        // `bpf_prog_stats` layout (24 bytes), but
                        // tolerate exotic layouts via the scalar path
                        // rather than panicking.
                        cnt = cnt.saturating_add(mem.read_u64(stats_pa, offsets.stats_cnt));
                        nsecs = nsecs.saturating_add(mem.read_u64(stats_pa, offsets.stats_nsecs));
                        misses =
                            misses.saturating_add(mem.read_u64(stats_pa, offsets.stats_misses));
                    }
                }
            }

            Some(ProgRuntimeStats {
                name,
                cnt,
                nsecs,
                misses,
            })
        },
    )
}

/// Read-only abstraction over BPF program enumeration and per-program
/// stats reads across data sources. Mirror of
/// [`super::bpf_map::BpfMapAccessor`] for the program side.
///
/// Currently one implementation: [`GuestMemProgAccessor`] (PTE-walks a
/// frozen guest's `prog_idr`). The planned live-host backend
/// will walk loaded programs via `BPF_PROG_GET_NEXT_ID` /
/// `BPF_OBJ_GET_INFO_BY_FD` and produce the same
/// `Vec<ProgVerifierStats>` / `Vec<ProgRuntimeStats>` shapes, so the
/// failure-dump renderer stays data-source-agnostic.
pub trait BpfProgAccessor {
    /// Enumerate struct_ops BPF programs and collect verifier stats.
    fn struct_ops_progs(&self) -> Vec<ProgVerifierStats>;

    /// Snapshot per-program runtime stats (`cnt`, `nsecs`, `misses`)
    /// for every struct_ops BPF program, summed across all CPUs.
    ///
    /// `per_cpu_offsets` is the kernel's `__per_cpu_offset[]` array,
    /// typically obtained via [`super::symbols::read_per_cpu_offsets`].
    /// The live-host backend will ignore this argument (the kernel
    /// provides per-CPU sums via `BPF_OBJ_GET_INFO_BY_FD`).
    fn struct_ops_runtime_stats(&self, per_cpu_offsets: &[u64]) -> Vec<ProgRuntimeStats>;

    /// Target-free active-scheduler walker: find the FIRST alive
    /// `BPF_PROG_TYPE_STRUCT_OPS` prog whose `aux->used_maps` carries
    /// a sibling `<obj>.bss/.data/.rodata` global-section map, and
    /// return that prog's obj prefix + full used_map_kvas set.
    ///
    /// **Why this exists.** The prior `value_kva == *scx_root`
    /// equality approach in `identify_active_obj_from_struct_ops`
    /// required identifying the active struct_ops map first. That
    /// equality breaks on kernels where `struct scx_sched` allocates
    /// a fresh kernel-side struct and COPIES the user's
    /// `sched_ext_ops` into its embedded `ops` field (offset 0) —
    /// `*scx_root` then points at the kernel-allocated `scx_sched`
    /// (whose address equals `&scx_sched.ops`), NOT at the struct_ops
    /// map's `kvalue.data` buffer (the user's source ops table at a
    /// separate address). Without a target, this walker iterates
    /// `prog_idr` and uses the "prog has a global-section map" signal
    /// directly.
    ///
    /// **Uniqueness via ktstr threat model.** This walker returns
    /// the FIRST match -- it does not assert uniqueness. ktstr's guest
    /// VM is single-tenant (only scx-ktstr runs), and the kernel
    /// enforces single-ENABLE for sched_ext: the enable path rejects a
    /// second scheduler while one is already enabled (pre-6.16:
    /// `scx_ops_enable_state() != SCX_OPS_DISABLED` -> `-EBUSY`;
    /// renamed to `scx_enable_state()`/`SCX_DISABLED` on 6.16+), so at
    /// most one sched_ext prog is ENABLED at a time. That does NOT
    /// guarantee one prog alive in `prog_idr`: a detached prog lingers
    /// until its owning struct_ops map's last fd closes and an RCU
    /// grace elapses, so the single-alive-in-`prog_idr` property this
    /// walker's FIRST-match relies on rests on ktstr's swap sequencing
    /// (kill the outgoing scheduler, wait for its process to exit
    /// before loading the next), not a kernel invariant. No other
    /// struct_ops subsystem (e.g. `tcp_congestion_ops`) ever loads in
    /// ktstr-managed guests. If a future setup loads non-sched_ext
    /// libbpf STRUCT_OPS progs, this filter would need to also gate on
    /// `aux->btf` matching the sched_ext_ops btf type id.
    ///
    /// Returns `None` when no live STRUCT_OPS prog has global-section
    /// maps (no scheduler attached, or only non-sched_ext struct_ops
    /// subsystems are running). The caller's prefix-grouping fallback
    /// handles the no-match case.
    fn find_active_struct_ops_obj_no_target(
        &self,
        map_offsets: &BpfMapOffsets,
    ) -> Option<ActiveObjMatch>;
}

/// Host-side BPF program accessor backed by direct guest physical-memory
/// reads. PTE-walks a frozen guest's `prog_idr` to enumerate loaded
/// programs and reads `bpf_prog_stats` per-CPU slots inline.
pub struct GuestMemProgAccessor<'a> {
    kernel: &'a super::guest::GuestKernel,
    prog_idr_kva: u64,
    /// Borrowed from the caller. Mirrors the
    /// [`super::bpf_map::GuestMemMapAccessor`] pattern:
    /// `BpfProgOffsets` is a ~112-byte POD built once from the
    /// vmlinux BTF, and every hot-path method reads it by reference,
    /// so owning it in the accessor would charge a clone that serves
    /// no purpose.
    offsets: &'a BpfProgOffsets,
}

impl<'a> GuestMemProgAccessor<'a> {
    /// Create from an existing [`GuestKernel`](super::guest::GuestKernel)
    /// and a caller-owned [`BpfProgOffsets`]. The accessor borrows both
    /// for its lifetime — build `offsets` once via
    /// [`BpfProgOffsets::from_vmlinux`] and reuse across calls.
    pub fn from_guest_kernel(
        kernel: &'a super::guest::GuestKernel,
        offsets: &'a BpfProgOffsets,
    ) -> anyhow::Result<Self> {
        let prog_idr_kva = kernel
            .symbol_kva("prog_idr")
            .ok_or_else(|| anyhow::anyhow!("prog_idr symbol not found in vmlinux"))?;

        Ok(Self {
            kernel,
            prog_idr_kva,
            offsets,
        })
    }
}

impl BpfProgAccessor for GuestMemProgAccessor<'_> {
    fn struct_ops_progs(&self) -> Vec<ProgVerifierStats> {
        find_struct_ops_progs(
            self.kernel.mem(),
            self.kernel.walk_context(),
            self.prog_idr_kva,
            self.offsets,
            self.kernel.start_kernel_map(),
            self.kernel.phys_base(),
        )
    }

    /// Mirrors the kernel-side per-CPU accumulation: `cnt` is
    /// bumped via `u64_stats_inc` and `nsecs` is bumped via
    /// `u64_stats_add(&stats->nsecs, duration)` inside
    /// `__bpf_prog_run` (include/linux/filter.h), invoked through
    /// the JIT-emitted entry path on every program invocation.
    /// `misses` is bumped by `bpf_prog_inc_misses_counter`
    /// (defined in `kernel/bpf/syscall.c`) called from
    /// `kernel/bpf/trampoline.c::__bpf_prog_enter_recur` when a
    /// program re-enters and the recursion guard rejects it.
    fn struct_ops_runtime_stats(&self, per_cpu_offsets: &[u64]) -> Vec<ProgRuntimeStats> {
        walk_struct_ops_runtime_stats(
            self.kernel.mem(),
            self.kernel.walk_context(),
            self.prog_idr_kva,
            self.offsets,
            per_cpu_offsets,
            self.kernel.start_kernel_map(),
            self.kernel.phys_base(),
        )
    }

    fn find_active_struct_ops_obj_no_target(
        &self,
        map_offsets: &BpfMapOffsets,
    ) -> Option<ActiveObjMatch> {
        find_active_struct_ops_obj_no_target(
            self.kernel.mem(),
            self.kernel.walk_context(),
            self.prog_idr_kva,
            self.offsets,
            map_offsets,
            self.kernel.start_kernel_map(),
            self.kernel.phys_base(),
        )
    }
}

/// Owns a [`super::guest::GuestKernel`] and a [`BpfProgOffsets`],
/// providing BPF program access through a borrowed
/// [`GuestMemProgAccessor`].
///
/// Mirrors [`super::bpf_map::GuestMemMapAccessorOwned`] for the
/// program-side surface: callers that don't already hold a
/// `GuestKernel` + `BpfProgOffsets` pair (e.g. the freeze
/// coordinator) construct one of these once at start, retain it
/// across the run, and borrow [`Self::as_accessor`] for each
/// read. Owning the offsets here keeps the BTF parse to once per
/// VM run rather than once per dump.
pub struct GuestMemProgAccessorOwned {
    kernel: super::guest::GuestKernel,
    prog_idr_kva: u64,
    offsets: BpfProgOffsets,
}

impl GuestMemProgAccessorOwned {
    pub fn finish(
        kernel: super::guest::GuestKernel,
        elf: &goblin::elf::Elf<'_>,
        data: &[u8],
        vmlinux: &std::path::Path,
    ) -> anyhow::Result<Self> {
        let offsets = BpfProgOffsets::from_elf(elf, data, vmlinux)?;
        let prog_idr_kva = kernel
            .symbol_kva("prog_idr")
            .ok_or_else(|| anyhow::anyhow!("prog_idr symbol not found in vmlinux"))?;
        Ok(Self {
            kernel,
            prog_idr_kva,
            offsets,
        })
    }

    /// Borrow as a [`GuestMemProgAccessor`] for program operations.
    ///
    /// Infallible — `finish` already resolved `prog_idr_kva` and the
    /// borrow returns the cached KVA directly. Mirrors
    /// [`super::bpf_map::GuestMemMapAccessorOwned::as_accessor`].
    pub fn as_accessor(&self) -> GuestMemProgAccessor<'_> {
        GuestMemProgAccessor {
            kernel: &self.kernel,
            prog_idr_kva: self.prog_idr_kva,
            offsets: &self.offsets,
        }
    }

    /// Access the underlying [`super::guest::GuestKernel`] for
    /// callers that need symbol resolution / page-walk primitives
    /// outside the prog-discovery surface (e.g. resolving
    /// `__per_cpu_offset` for `struct_ops_runtime_stats`).
    #[allow(dead_code)]
    pub fn guest_kernel(&self) -> &super::guest::GuestKernel {
        &self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::START_KERNEL_MAP;

    #[test]
    fn prog_verifier_stats_serde_roundtrip() {
        let info = ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 42000,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "dispatch");
        assert_eq!(loaded.verified_insns, 42000);
    }

    #[test]
    fn prog_verifier_stats_vec_serde_roundtrip() {
        let stats = vec![
            ProgVerifierStats {
                name: "dispatch".to_string(),
                verified_insns: 100000,
            },
            ProgVerifierStats {
                name: "enqueue".to_string(),
                verified_insns: 50000,
            },
        ];
        let json = serde_json::to_vec(&stats).unwrap();
        let loaded: Vec<ProgVerifierStats> = serde_json::from_slice(&json).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "dispatch");
        assert_eq!(loaded[0].verified_insns, 100000);
        assert_eq!(loaded[1].name, "enqueue");
        assert_eq!(loaded[1].verified_insns, 50000);
    }

    #[test]
    fn prog_verifier_stats_empty_name() {
        let info = ProgVerifierStats {
            name: String::new(),
            verified_insns: 0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "");
        assert_eq!(loaded.verified_insns, 0);
    }

    #[test]
    fn prog_verifier_stats_max_values() {
        let info = ProgVerifierStats {
            name: "x".repeat(16),
            verified_insns: u32::MAX,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verified_insns, u32::MAX);
        assert_eq!(loaded.name.len(), 16);
    }

    #[test]
    fn prog_runtime_stats_serde_roundtrip() {
        let info = ProgRuntimeStats {
            name: "ktstr_dispatch".to_string(),
            cnt: 12345,
            nsecs: 9_876_543,
            misses: 7,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgRuntimeStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "ktstr_dispatch");
        assert_eq!(loaded.cnt, 12345);
        assert_eq!(loaded.nsecs, 9_876_543);
        assert_eq!(loaded.misses, 7);
    }

    /// All three counters use `saturating_add` in
    /// [`walk_struct_ops_runtime_stats`] when summing per-CPU slots, so a
    /// long-running guest with a hot BPF program (or scrambled
    /// per-CPU pages from an unmapped slot) can produce a `u64::MAX`
    /// sum instead of wrapping. Pinning the wire shape here proves
    /// the serde codec preserves the saturated value end-to-end —
    /// any future migration that swaps the field type would surface
    /// here before bleeding into the failure-dump consumers.
    #[test]
    fn prog_runtime_stats_max_u64_saturation_roundtrip() {
        let info = ProgRuntimeStats {
            name: "saturated".to_string(),
            cnt: u64::MAX,
            nsecs: u64::MAX,
            misses: u64::MAX,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgRuntimeStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.cnt, u64::MAX);
        assert_eq!(loaded.nsecs, u64::MAX);
        assert_eq!(loaded.misses, u64::MAX);
    }

    #[test]
    fn prog_runtime_stats_default_zero() {
        let info = ProgRuntimeStats::default();
        assert_eq!(info.name, "");
        assert_eq!(info.cnt, 0);
        assert_eq!(info.nsecs, 0);
        assert_eq!(info.misses, 0);
    }

    /// The Display impl is the entry point used by
    /// [`super::dump::FailureDumpReport`]'s human-readable rendering;
    /// pin the format so a downstream change to the impl is caught
    /// before the failure-dump output silently changes shape.
    ///
    /// Two derived metrics surface on the line when their guards
    /// pass: `ns/call` whenever `cnt > 0`, and `miss_rate`
    /// whenever there are any misses. A program that never ran
    /// (cnt=0) elides both — `prog_runtime_stats_display_zero_counters_elides_derived`
    /// covers that branch.
    #[test]
    fn prog_runtime_stats_display_format() {
        let info = ProgRuntimeStats {
            name: "ktstr_enqueue".to_string(),
            cnt: 100,
            nsecs: 200,
            misses: 3,
        };
        // cnt=100, nsecs=200 → ns/call = 2.000.
        // misses=3, cnt+misses=103 → miss_rate = 3/103 ≈ 0.0291.
        assert_eq!(
            format!("{info}"),
            "ktstr_enqueue: cnt=100 nsecs=200 misses=3 ns/call=2.000 miss_rate=0.0291",
        );
    }

    /// A program that never ran (cnt=0) renders only the four
    /// raw counters — both derived metrics are guarded out.
    /// Pin the elision so a regression that strips the guard and
    /// emits "ns/call=0.000 miss_rate=0.0000" surfaces here.
    #[test]
    fn prog_runtime_stats_display_zero_counters_elides_derived() {
        let info = ProgRuntimeStats {
            name: "never_ran".to_string(),
            cnt: 0,
            nsecs: 0,
            misses: 0,
        };
        let s = format!("{info}");
        assert_eq!(s, "never_ran: cnt=0 nsecs=0 misses=0");
        assert!(!s.contains("ns/call"), "ns/call must elide when cnt=0: {s}");
        assert!(
            !s.contains("miss_rate"),
            "miss_rate must elide when total=0: {s}"
        );
    }

    /// Healthy program with no recursion misses — `ns/call`
    /// surfaces but `miss_rate` elides (since misses=0).
    /// A regression that flipped the gate and rendered a
    /// "miss_rate=0.0000" line on every healthy program would
    /// trip here.
    #[test]
    fn prog_runtime_stats_display_no_misses_elides_miss_rate() {
        let info = ProgRuntimeStats {
            name: "healthy".to_string(),
            cnt: 1000,
            nsecs: 50_000,
            misses: 0,
        };
        let s = format!("{info}");
        assert!(s.contains("ns/call=50.000"), "ns/call must render: {s}");
        assert!(
            !s.contains("miss_rate"),
            "miss_rate must elide when misses=0: {s}",
        );
    }

    /// `ns_per_call` derived accessor: pin happy-path math + zero-
    /// divisor guard. Mirrors the `CgroupStats::wake_latency_tail_ratio`
    /// test pattern from assert.rs.
    #[test]
    fn prog_runtime_stats_ns_per_call_derived() {
        // Happy path: 1000 cnt + 50000 nsecs = 50 ns/call.
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 1000,
            nsecs: 50_000,
            misses: 0,
        };
        assert_eq!(info.ns_per_call(), 50.0);
        assert!(info.ns_per_call().is_finite());

        // Zero divisor: cnt=0 → 0.0 (not NaN).
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 0,
            nsecs: 999_999,
            misses: 0,
        };
        assert_eq!(info.ns_per_call(), 0.0);
        assert!(info.ns_per_call().is_finite());
    }

    /// `miss_rate` derived accessor: pin happy-path math + zero-
    /// divisor guard + saturating_add edge.
    #[test]
    fn prog_runtime_stats_miss_rate_derived() {
        // Happy path: 9 misses / (1 cnt + 9 misses) = 0.9.
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 1,
            nsecs: 0,
            misses: 9,
        };
        assert!((info.miss_rate() - 0.9).abs() < 1e-12);
        assert!(info.miss_rate().is_finite());

        // Zero divisor: both counters zero → 0.0 (not NaN).
        let info = ProgRuntimeStats::default();
        assert_eq!(info.miss_rate(), 0.0);
        assert!(info.miss_rate().is_finite());

        // Saturating-add edge: cnt at u64::MAX, misses also non-
        // trivial — `saturating_add` floors at u64::MAX, so the
        // denominator stays non-zero and the rate is finite.
        let info = ProgRuntimeStats {
            name: "saturated".to_string(),
            cnt: u64::MAX,
            nsecs: 0,
            misses: 1000,
        };
        assert!(info.miss_rate().is_finite());
        // Result is essentially 0 (1000 / u64::MAX) but the
        // important contract is finiteness — a regression that
        // overflowed and produced inf/NaN trips here.
        assert!(info.miss_rate() >= 0.0);
    }

    /// Wire format must NOT carry the derived ratios — they are
    /// method-only and recomputed on read. Pin so a regression
    /// that re-introduces a stored shadow trips here.
    #[test]
    fn prog_runtime_stats_wire_format_omits_derived_keys() {
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 100,
            nsecs: 200,
            misses: 3,
        };
        let json = serde_json::to_value(&info).unwrap();
        let map = match json {
            serde_json::Value::Object(m) => m,
            other => panic!("expected object, got {other:?}"),
        };
        assert!(
            !map.contains_key("ns_per_call"),
            "derived methods must NOT appear as wire fields: {map:#?}",
        );
        assert!(
            !map.contains_key("miss_rate"),
            "derived methods must NOT appear as wire fields: {map:#?}",
        );
        // Cross-check: methods still compute correctly.
        assert_eq!(info.ns_per_call(), 2.0);
        assert!((info.miss_rate() - 3.0_f64 / 103.0).abs() < 1e-12);
    }

    /// Build a minimal `BpfProgOffsets` keyed for the synthetic
    /// chain test below. The exact field offsets are arbitrary —
    /// they only need to be consistent with how the test buffer
    /// is laid out — but `stats_cnt`/`stats_nsecs`/`stats_misses`
    /// MUST sit within a 24-byte window so the bulk-read path
    /// fires (`span <= 64`). Drift in these three offsets would
    /// silently switch the walker to the scalar fallback and
    /// the bulk-read assertion below would still pass for the
    /// wrong reason.
    fn synthetic_prog_offsets() -> BpfProgOffsets {
        BpfProgOffsets {
            prog_type: 0,
            prog_aux: 8,
            aux_verified_insns: 0,
            aux_name: 8,
            aux_used_maps: 24,
            aux_used_map_cnt: 32,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 0,
            idr_next: 8,
            prog_stats: 16,
            stats_cnt: 0,
            stats_nsecs: 8,
            stats_misses: 16,
        }
    }

    /// Run the bulk-24-byte-read end-to-end chain at a caller-
    /// supplied `page_offset`. Both the x86_64 and aarch64 wrapper
    /// tests call this with their respective `PAGE_OFFSET` baselines
    /// so the bulk-read fast path is exercised on both arches.
    fn walk_struct_ops_runtime_stats_bulk_chain_at_page_offset(page_offset: u64) {
        use crate::monitor::reader::{GuestMem, WalkContext};

        // Layout (all PAs offset by `page_offset` to form KVAs in
        // the direct-mapping range, except `prog_idr_kva` which
        // sits in the kernel-text range and translates via
        // `text_kva_to_pa_with_base`):
        //
        //   0x0000  prog_idr (xa_head + idr_next)
        //   0x1000  bpf_prog (prog_type, prog_aux, prog_stats)
        //   0x2000  bpf_prog_aux (verified_insns, name)
        //   0x3000  per-CPU bpf_prog_stats (cnt, nsecs, misses)
        let total: usize = 0x4000;
        let mut buf = vec![0u8; total];

        let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };

        let idr_pa: u64 = 0x0000;
        let prog_pa: u64 = 0x1000;
        let aux_pa: u64 = 0x2000;
        let stats_pa: u64 = 0x3000;

        // Single-entry xarray: `xa_head` IS the prog KVA with
        // bit 1 clear (leaf marker). `pa_to_kva(prog_pa)` has
        // bit 1 clear because prog_pa is 4 KiB-aligned.
        let prog_kva = pa_to_kva(prog_pa);
        assert_eq!(prog_kva & 2, 0, "prog_kva must be a leaf entry");

        let offsets = synthetic_prog_offsets();
        // Sanity: the bulk-read fast path requires
        // `span = hi - lo <= 64`. With offsets {0, 8, 16}:
        // lo = 0, hi = 16 + 8 = 24, span = 24. Pinning here so
        // a future offset change that pushed `span > 64`
        // (forcing the scalar fallback) trips the assert
        // before the test runs.
        let lo = offsets
            .stats_cnt
            .min(offsets.stats_nsecs)
            .min(offsets.stats_misses);
        let hi = offsets
            .stats_cnt
            .max(offsets.stats_nsecs)
            .max(offsets.stats_misses)
            + 8;
        assert!(
            hi - lo <= 64,
            "test premise: stats span must be small enough for the bulk path"
        );

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // IDR: xa_head = prog_kva, idr_next = 1.
        write_u64(&mut buf, idr_pa + offsets.idr_xa_head as u64, prog_kva);
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 1);

        // bpf_prog: type = STRUCT_OPS, aux = aux_kva, stats = stats_kva.
        write_u32(
            &mut buf,
            prog_pa + offsets.prog_type as u64,
            BPF_PROG_TYPE_STRUCT_OPS,
        );
        write_u64(
            &mut buf,
            prog_pa + offsets.prog_aux as u64,
            pa_to_kva(aux_pa),
        );
        write_u64(
            &mut buf,
            prog_pa + offsets.prog_stats as u64,
            pa_to_kva(stats_pa),
        );

        // bpf_prog_aux: verified_insns + name. Name must NUL-
        // terminate within BPF_OBJ_NAME_LEN so the walker's
        // `position(|&b| b == 0)` finds the end.
        write_u32(&mut buf, aux_pa + offsets.aux_verified_insns as u64, 12_345);
        let name = b"bulk_test";
        let name_pa = (aux_pa + offsets.aux_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        // Stats: write the three u64 counters at the synthetic
        // offsets. These are the bytes the bulk read MUST surface
        // through the parse closure.
        let known_cnt: u64 = 0x1111_1111_1111_1111;
        let known_nsecs: u64 = 0x2222_2222_2222_2222;
        let known_misses: u64 = 0x3333_3333_3333_3333;
        write_u64(&mut buf, stats_pa + offsets.stats_cnt as u64, known_cnt);
        write_u64(&mut buf, stats_pa + offsets.stats_nsecs as u64, known_nsecs);
        write_u64(
            &mut buf,
            stats_pa + offsets.stats_misses as u64,
            known_misses,
        );

        // SAFETY: buf is a live local Vec<u8> whose backing storage
        // outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let walk = WalkContext {
            cr3_pa: 0,
            page_offset,
            l5: false,
            tcr_el1: 0,
        };
        // One CPU. `cpu_off == 0` is allowed at `cpu_index == 0`
        // (BSP). `stats_kva + 0 = stats_kva`, which translates
        // through the direct mapping to `stats_pa`.
        let per_cpu_offsets = vec![0u64];

        let prog_idr_kva = idr_pa + START_KERNEL_MAP;
        let stats = walk_struct_ops_runtime_stats(
            &mem,
            walk,
            prog_idr_kva,
            &offsets,
            &per_cpu_offsets,
            START_KERNEL_MAP,
            0,
        );

        assert_eq!(stats.len(), 1, "single STRUCT_OPS prog must surface");
        assert_eq!(stats[0].name, "bulk_test");
        assert_eq!(
            stats[0].cnt, known_cnt,
            "bulk read must parse cnt at offsets.stats_cnt within the 24-byte window",
        );
        assert_eq!(
            stats[0].nsecs, known_nsecs,
            "bulk read must parse nsecs at offsets.stats_nsecs within the 24-byte window",
        );
        assert_eq!(
            stats[0].misses, known_misses,
            "bulk read must parse misses at offsets.stats_misses within the 24-byte window",
        );
    }

    /// End-to-end chain test for the bulk 24-byte
    /// `bpf_prog_stats` read on x86_64. The walker reads `cnt`,
    /// `nsecs`, and `misses` (three adjacent u64s in the kernel
    /// `struct bpf_prog_stats`) via one `read_bytes` over the
    /// `[lo, hi)` span and parses each value from the local
    /// buffer. The aarch64 wrapper below pins the same chain
    /// against the aarch64 `PAGE_OFFSET` baseline.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_struct_ops_runtime_stats_bulk_24byte_read_parses_three_offsets() {
        // x86_64 PAGE_OFFSET (4-level paging, non-KASLR baseline).
        walk_struct_ops_runtime_stats_bulk_chain_at_page_offset(0xFFFF_8880_0000_0000);
    }

    /// End-to-end chain test for the bulk 24-byte
    /// `bpf_prog_stats` read on aarch64. Mirrors the x86_64
    /// wrapper above against the aarch64 direct-mapping
    /// `PAGE_OFFSET` baseline so the bulk-read fast path is
    /// pinned on both arches.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn walk_struct_ops_runtime_stats_bulk_24byte_read_parses_three_offsets() {
        // aarch64 PAGE_OFFSET baseline (48-bit VA, 4 KiB granule).
        walk_struct_ops_runtime_stats_bulk_chain_at_page_offset(0xFFFF_0000_0000_0000);
    }

    /// Format chain integration: the `ProgRuntimeStats` Display
    /// output must appear verbatim inside `FailureDumpReport`'s
    /// Display output. Pins the chain
    /// `ProgRuntimeStats::fmt` (bpf_prog.rs) →
    /// `FailureDumpReport::fmt::std::fmt::Display::fmt(stats, f)`
    /// (dump/display.rs `prog_runtime_stats:` arm).
    ///
    /// The standalone `prog_runtime_stats_display_format` test pins
    /// the inner Display in isolation; the dump-side
    /// `report_display_renders_prog_runtime_stats` test pins the
    /// outer section header. Neither catches a regression that
    /// SUBSTITUTED the inner Display call (e.g. introducing a
    /// custom rendering branch in the outer formatter that bypasses
    /// `ProgRuntimeStats::fmt`). This test catches that drift by
    /// asserting BOTH layers render identically and the inner
    /// string appears as a substring of the outer — a substitution
    /// would break either equality.
    #[test]
    fn prog_runtime_stats_format_chain_inner_appears_in_outer() {
        use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
        let info = ProgRuntimeStats {
            name: "chain_test".to_string(),
            cnt: 7,
            nsecs: 42,
            misses: 1,
        };
        let inner = format!("{info}");
        // Direct Display on ProgRuntimeStats: pinned shape includes
        // the bpftop-style derived metrics. cnt=7 nsecs=42 →
        // ns/call=6.000; misses=1 → miss_rate=1/8=0.1250.
        assert_eq!(
            inner,
            "chain_test: cnt=7 nsecs=42 misses=1 ns/call=6.000 miss_rate=0.1250",
        );

        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            prog_runtime_stats: vec![info],
            ..Default::default()
        };
        let outer = format!("{report}");
        // The outer's `prog_runtime_stats:` section calls
        // `std::fmt::Display::fmt(stats, f)` on each entry; that
        // call dispatches through THIS module's Display impl. If a
        // future regression replaced the dispatch with a custom
        // formatter, the inner string would no longer appear in
        // the outer output — surfacing as substring failure.
        assert!(
            outer.contains(&inner),
            "FailureDumpReport's Display chain must dispatch through \
             ProgRuntimeStats::fmt — inner {inner:?} must appear \
             verbatim inside outer:\n{outer}",
        );
        // Sanity: the outer also wraps with the expected section
        // header, so the substring match is finding the chain
        // through the correct arm of FailureDumpReport's fmt and
        // not (e.g.) a coincidence in the schema marker.
        assert!(
            outer.contains("prog_runtime_stats:"),
            "outer Display must carry the prog_runtime_stats section \
             header; without it the chain test could pass even when the \
             inner string matched a different format arm:\n{outer}",
        );
    }

    // -- prog_idr chain fixtures: synthetic guest memory that drives
    //    the full for_each_struct_ops_prog walk on the host with no
    //    VM and no live kernel. The layout mirrors the existing
    //    `walk_struct_ops_runtime_stats_bulk_chain_at_page_offset`
    //    helper above (idr@0x0, prog@0x1000, aux@0x2000) and reuses
    //    `synthetic_prog_offsets()`. PAs are offset by `page_offset`
    //    to form direct-map KVAs that `translate_any_kva` resolves
    //    via `kva_to_pa` (the direct-map fast path), and the
    //    `prog_idr_kva` sits in kernel-text range translated via
    //    `text_kva_to_pa_with_base`.

    /// x86_64 PAGE_OFFSET baseline (4-level paging, non-KASLR) used
    /// by the host-only chain fixtures below. The exact value only
    /// has to keep `page_offset + pa` outside `[0, buf.len())` so the
    /// formed KVAs are unambiguously in the direct-map range and
    /// translate back to `pa` via `kva_to_pa`.
    const FIXTURE_PAGE_OFFSET: u64 = 0xFFFF_8880_0000_0000;

    /// Mutable byte buffer wrapped as guest DRAM for the prog_idr
    /// chain fixtures. Owns the backing `Vec` so the unsafe
    /// [`GuestMem::new`] pointer stays valid for the fixture's life.
    struct ProgChainFixture {
        buf: Vec<u8>,
        page_offset: u64,
    }

    impl ProgChainFixture {
        fn new(size: usize) -> Self {
            Self {
                buf: vec![0u8; size],
                page_offset: FIXTURE_PAGE_OFFSET,
            }
        }

        /// Direct-map KVA for a DRAM offset.
        fn pa_to_kva(&self, pa: u64) -> u64 {
            self.page_offset.wrapping_add(pa)
        }

        fn write_u64(&mut self, pa: u64, val: u64) {
            let off = pa as usize;
            self.buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        }

        fn write_u32(&mut self, pa: u64, val: u32) {
            let off = pa as usize;
            self.buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        }

        /// Write a NUL-terminated (within `BPF_OBJ_NAME_LEN`) name
        /// blob at `pa`. Panics if the name needs a terminator but
        /// fills the whole field — every fixture name here is short.
        fn write_name(&mut self, pa: u64, name: &[u8]) {
            assert!(
                name.len() < BPF_OBJ_NAME_LEN,
                "fixture name must leave room for the NUL terminator",
            );
            let off = pa as usize;
            self.buf[off..off + name.len()].copy_from_slice(name);
        }

        fn mem(&self) -> GuestMem {
            // SAFETY: `self.buf` is a live Vec<u8> owned by the
            // fixture; it outlives every GuestMem read in the test
            // because the fixture is dropped after the assertions.
            unsafe { GuestMem::new(self.buf.as_ptr() as *mut u8, self.buf.len() as u64) }
        }

        fn walk(&self) -> WalkContext {
            WalkContext {
                cr3_pa: 0,
                page_offset: self.page_offset,
                l5: false,
                tcr_el1: 0,
            }
        }
    }

    /// PA constants shared by the chain fixtures.
    const FIX_IDR_PA: u64 = 0x0000;
    const FIX_PROG_PA: u64 = 0x1000;
    const FIX_AUX_PA: u64 = 0x2000;

    /// Build a fixture whose prog_idr holds a single STRUCT_OPS prog
    /// (single-entry xarray: `xa_head` IS the prog KVA, `idr_next=1`).
    /// `prog_type` lets a caller override the type to exercise the
    /// non-struct_ops skip arm. The prog's `aux` pointer is wired but
    /// the aux body (name, verified_insns, used_maps, stats) is left
    /// for the caller to populate. Returns the fixture; the
    /// `prog_idr_kva` for the walk is `FIX_IDR_PA + START_KERNEL_MAP`.
    fn single_prog_fixture(
        size: usize,
        prog_type: u32,
        offsets: &BpfProgOffsets,
    ) -> ProgChainFixture {
        let mut fx = ProgChainFixture::new(size);
        let prog_kva = fx.pa_to_kva(FIX_PROG_PA);
        // Single-entry xarray leaf marker: prog_kva must have bits
        // 0-1 clear so `xa_is_node` treats it as a direct entry.
        assert_eq!(prog_kva & 3, 0, "prog_kva must be a leaf entry");

        // IDR: xa_head = prog_kva, idr_next = 1.
        fx.write_u64(FIX_IDR_PA + offsets.idr_xa_head as u64, prog_kva);
        fx.write_u32(FIX_IDR_PA + offsets.idr_next as u64, 1);

        // bpf_prog: type + aux pointer.
        fx.write_u32(FIX_PROG_PA + offsets.prog_type as u64, prog_type);
        fx.write_u64(
            FIX_PROG_PA + offsets.prog_aux as u64,
            fx.pa_to_kva(FIX_AUX_PA),
        );
        fx
    }

    // ---- find_struct_ops_progs ----

    /// Happy path: a single STRUCT_OPS prog whose aux carries a name
    /// and verified_insns count surfaces with both fields read from
    /// the synthetic offsets. Covers the `find_struct_ops_progs`
    /// payload closure (aux_verified_insns read + aux_name NUL-scan +
    /// String build) and the `for_each_struct_ops_prog` happy path.
    #[test]
    fn find_struct_ops_progs_single_prog_reads_name_and_verified_insns() {
        let offsets = synthetic_prog_offsets();
        let mut fx = single_prog_fixture(0x3000, BPF_PROG_TYPE_STRUCT_OPS, &offsets);
        fx.write_u32(FIX_AUX_PA + offsets.aux_verified_insns as u64, 12_345);
        fx.write_name(FIX_AUX_PA + offsets.aux_name as u64, b"dispatch");

        let progs = find_struct_ops_progs(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(progs.len(), 1);
        assert_eq!(progs[0].name, "dispatch");
        assert_eq!(progs[0].verified_insns, 12_345u32);
    }

    /// The `prog_type != BPF_PROG_TYPE_STRUCT_OPS { continue }` filter
    /// at `for_each_struct_ops_prog` line ~94: a prog whose type is
    /// not 27 (here KPROBE=2) is skipped before its aux is read, so
    /// the result is empty.
    #[test]
    fn find_struct_ops_progs_skips_non_struct_ops_type() {
        const BPF_PROG_TYPE_KPROBE: u32 = 2;
        let offsets = synthetic_prog_offsets();
        // Populate the aux body too, to prove the skip happens at the
        // type filter and not because aux was unreadable.
        let mut fx = single_prog_fixture(0x3000, BPF_PROG_TYPE_KPROBE, &offsets);
        fx.write_u32(FIX_AUX_PA + offsets.aux_verified_insns as u64, 999);
        fx.write_name(FIX_AUX_PA + offsets.aux_name as u64, b"kprobe_prog");

        let progs = find_struct_ops_progs(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(progs.len(), 0);
        assert!(progs.is_empty());
    }

    /// An all-zero IDR (xa_head left unwritten) yields an empty result:
    /// `for_each_struct_ops_prog` short-circuits on `xa_head == 0` before
    /// the id loop, but even without that guard an empty xarray surfaces
    /// no progs — so this pins the empty-IDR→empty outcome, not the
    /// short-circuit in isolation. Reached via `find_struct_ops_progs`.
    #[test]
    fn for_each_struct_ops_prog_empty_xa_head_returns_empty() {
        let offsets = synthetic_prog_offsets();
        // Default-zero buffer: do NOT write xa_head. idr_next is
        // irrelevant because the xa_head guard fires first.
        let fx = ProgChainFixture::new(0x3000);

        let progs = find_struct_ops_progs(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(progs.len(), 0);
        assert!(progs.is_empty());
    }

    /// A corrupt `idr_next` of `u32::MAX` still returns the correct
    /// result — the one real prog at id 0 — and terminates, because the
    /// `.min(65536)` clamp on `idr_next` in `for_each_struct_ops_prog`
    /// bounds the loop. The single-entry xarray returns the prog for id 0
    /// and `Some(0)` for every id > 0 (see `idr::xa_load`). This pins the
    /// RESULT under a corrupt count; a clamp regression would surface as
    /// a slow run rather than a failed assertion (pinning the exact 65536
    /// boundary would need a multi-level xarray with an entry past the
    /// cap — out of scope here).
    #[test]
    fn for_each_struct_ops_prog_caps_idr_next_at_65536() {
        let offsets = synthetic_prog_offsets();
        let mut fx = single_prog_fixture(0x3000, BPF_PROG_TYPE_STRUCT_OPS, &offsets);
        // Overwrite idr_next with u32::MAX. Without the .min(65536)
        // clamp this loop would attempt ~4 billion iterations.
        fx.write_u32(FIX_IDR_PA + offsets.idr_next as u64, u32::MAX);
        fx.write_u32(FIX_AUX_PA + offsets.aux_verified_insns as u64, 7);
        fx.write_name(FIX_AUX_PA + offsets.aux_name as u64, b"capped");

        let progs = find_struct_ops_progs(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(progs.len(), 1);
        assert_eq!(progs[0].name, "capped");
        assert_eq!(progs[0].verified_insns, 7u32);
    }

    // ---- walk_struct_ops_runtime_stats ----

    /// The `if stats_percpu_kva == 0 { return None }` skip at
    /// `walk_struct_ops_runtime_stats` line ~533: a prog whose
    /// `prog_stats` per-CPU base is NULL is dropped from the result
    /// (closure returns None -> not pushed). prog_stats is left
    /// unwritten (zero) on the single STRUCT_OPS prog.
    #[test]
    fn walk_runtime_stats_skips_prog_with_null_stats_pointer() {
        let offsets = synthetic_prog_offsets();
        let mut fx = single_prog_fixture(0x3000, BPF_PROG_TYPE_STRUCT_OPS, &offsets);
        fx.write_name(FIX_AUX_PA + offsets.aux_name as u64, b"no_stats");
        // Deliberately leave prog_stats == 0 (do not write it).

        let per_cpu_offsets = vec![0u64];
        let stats = walk_struct_ops_runtime_stats(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            &per_cpu_offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(stats.len(), 0);
        assert!(stats.is_empty());
    }

    /// Multi-CPU accumulation: with two distinct, non-zero per-CPU
    /// offsets the walker translates both per-CPU `bpf_prog_stats`
    /// blocks and `saturating_add`s `cnt`/`nsecs`/`misses` across
    /// them. Covers the per-CPU sum loop at
    /// `walk_struct_ops_runtime_stats` lines ~544-623 for >1 CPU.
    #[test]
    fn walk_runtime_stats_sums_across_two_cpus() {
        let offsets = synthetic_prog_offsets();
        // Buffer must hold both stats blocks. stats0 @0x3000,
        // stats1 @0x3800 — both within a 0x4000 buffer.
        let stats_pa0: u64 = 0x3000;
        let stats_pa1: u64 = 0x3800;
        let mut fx = single_prog_fixture(0x4000, BPF_PROG_TYPE_STRUCT_OPS, &offsets);
        fx.write_name(FIX_AUX_PA + offsets.aux_name as u64, b"two_cpu");
        // prog_stats per-CPU base = stats_pa0's KVA. cpu_off shifts it.
        fx.write_u64(
            FIX_PROG_PA + offsets.prog_stats as u64,
            fx.pa_to_kva(stats_pa0),
        );

        // Distinct small counters per block.
        let cnt0: u64 = 10;
        let nsecs0: u64 = 100;
        let misses0: u64 = 1;
        let cnt1: u64 = 7;
        let nsecs1: u64 = 70;
        let misses1: u64 = 3;
        fx.write_u64(stats_pa0 + offsets.stats_cnt as u64, cnt0);
        fx.write_u64(stats_pa0 + offsets.stats_nsecs as u64, nsecs0);
        fx.write_u64(stats_pa0 + offsets.stats_misses as u64, misses0);
        fx.write_u64(stats_pa1 + offsets.stats_cnt as u64, cnt1);
        fx.write_u64(stats_pa1 + offsets.stats_nsecs as u64, nsecs1);
        fx.write_u64(stats_pa1 + offsets.stats_misses as u64, misses1);

        // cpu0: cpu_off=0 reads stats_pa0 (BSP). cpu1: cpu_off=delta
        // reads stats_pa0+delta = stats_pa1.
        let per_cpu_offsets = vec![0u64, stats_pa1 - stats_pa0];
        let stats = walk_struct_ops_runtime_stats(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            &per_cpu_offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].cnt, cnt0 + cnt1);
        assert_eq!(stats[0].nsecs, nsecs0 + nsecs1);
        assert_eq!(stats[0].misses, misses0 + misses1);
    }

    /// The `if cpu_off == 0 && cpu_index > 0 { continue }` BSS-zero-
    /// tail guard at `walk_struct_ops_runtime_stats` lines ~556-558:
    /// a trailing `__per_cpu_offset[]=0` slot (cpu_index > 0) must be
    /// skipped so CPU 0's stats are NOT double-counted. With
    /// `per_cpu_offsets = [0, 0]` the summed fields equal the single
    /// block's values, not twice them — a regression that dropped the
    /// `cpu_index > 0` guard would double them.
    #[test]
    fn walk_runtime_stats_skips_zero_offset_tail_cpu() {
        let offsets = synthetic_prog_offsets();
        let stats_pa: u64 = 0x3000;
        let mut fx = single_prog_fixture(0x4000, BPF_PROG_TYPE_STRUCT_OPS, &offsets);
        fx.write_name(FIX_AUX_PA + offsets.aux_name as u64, b"bss_tail");
        fx.write_u64(
            FIX_PROG_PA + offsets.prog_stats as u64,
            fx.pa_to_kva(stats_pa),
        );

        let cnt0: u64 = 42;
        let nsecs0: u64 = 4200;
        let misses0: u64 = 5;
        fx.write_u64(stats_pa + offsets.stats_cnt as u64, cnt0);
        fx.write_u64(stats_pa + offsets.stats_nsecs as u64, nsecs0);
        fx.write_u64(stats_pa + offsets.stats_misses as u64, misses0);

        // Two slots: cpu0 (cpu_off=0, BSP, allowed) reads stats_pa;
        // cpu1 (cpu_off=0, cpu_index=1) is skipped by the guard.
        let per_cpu_offsets = vec![0u64, 0u64];
        let stats = walk_struct_ops_runtime_stats(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &offsets,
            &per_cpu_offsets,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].cnt, cnt0);
        assert_eq!(stats[0].nsecs, nsecs0);
        assert_eq!(stats[0].misses, misses0);
    }

    // ---- find_active_struct_ops_obj_no_target ----

    /// PA constants for the active-obj fixture's used_maps array and
    /// map structs.
    const FIX_USED_MAPS_PA: u64 = 0x3000;
    const FIX_MAP0_PA: u64 = 0x4000;
    const FIX_MAP1_PA: u64 = 0x5000;

    /// Happy path: a STRUCT_OPS prog whose aux->used_maps holds two
    /// map pointers — a struct_ops map (no global-section suffix) and
    /// a `<obj>.bss` global-section map — resolves to the obj prefix
    /// and the full used_map_kvas snapshot. Covers
    /// `find_active_struct_ops_obj_no_target`: used_maps!=0,
    /// used_map_cnt!=0, the entries snapshot loop, the per-map name
    /// read + `extract_global_section_obj_prefix` match.
    #[test]
    fn find_active_struct_ops_obj_returns_obj_prefix_from_bss_map() {
        let prog_offsets = synthetic_prog_offsets();
        let map_offsets = BpfMapOffsets {
            map_name: 0,
            ..BpfMapOffsets::EMPTY
        };
        let mut fx = single_prog_fixture(0x6000, BPF_PROG_TYPE_STRUCT_OPS, &prog_offsets);

        // aux->used_maps = used_maps array KVA, used_map_cnt = 2.
        let map0_kva = fx.pa_to_kva(FIX_MAP0_PA);
        let map1_kva = fx.pa_to_kva(FIX_MAP1_PA);
        fx.write_u64(
            FIX_AUX_PA + prog_offsets.aux_used_maps as u64,
            fx.pa_to_kva(FIX_USED_MAPS_PA),
        );
        fx.write_u32(FIX_AUX_PA + prog_offsets.aux_used_map_cnt as u64, 2);
        fx.write_u64(FIX_USED_MAPS_PA, map0_kva);
        fx.write_u64(FIX_USED_MAPS_PA + 8, map1_kva);

        // map0: struct_ops map name (no suffix -> no match).
        fx.write_name(FIX_MAP0_PA + map_offsets.map_name as u64, b"ktstr_ops");
        // map1: global-section .bss map (matches -> obj "bpf_bpf").
        fx.write_name(FIX_MAP1_PA + map_offsets.map_name as u64, b"bpf_bpf.bss");

        let result = find_active_struct_ops_obj_no_target(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &prog_offsets,
            &map_offsets,
            START_KERNEL_MAP,
            0,
        );
        let m = result.unwrap();
        assert_eq!(m.obj_name, "bpf_bpf");
        assert_eq!(m.used_map_kvas, vec![map0_kva, map1_kva]);
    }

    /// The closure-returns-None path: every map in the snapshot is
    /// scanned but none matches a global-section suffix, so the
    /// closure returns None and `.into_iter().next()` yields None.
    /// Distinct from the null-used_maps and zero-cnt early skips.
    #[test]
    fn find_active_struct_ops_obj_none_when_no_global_section_map() {
        let prog_offsets = synthetic_prog_offsets();
        let map_offsets = BpfMapOffsets {
            map_name: 0,
            ..BpfMapOffsets::EMPTY
        };
        let mut fx = single_prog_fixture(0x6000, BPF_PROG_TYPE_STRUCT_OPS, &prog_offsets);

        fx.write_u64(
            FIX_AUX_PA + prog_offsets.aux_used_maps as u64,
            fx.pa_to_kva(FIX_USED_MAPS_PA),
        );
        fx.write_u32(FIX_AUX_PA + prog_offsets.aux_used_map_cnt as u64, 2);
        fx.write_u64(FIX_USED_MAPS_PA, fx.pa_to_kva(FIX_MAP0_PA));
        fx.write_u64(FIX_USED_MAPS_PA + 8, fx.pa_to_kva(FIX_MAP1_PA));

        // Both maps lack any .bss/.data/.rodata suffix.
        fx.write_name(FIX_MAP0_PA + map_offsets.map_name as u64, b"ktstr_ops");
        fx.write_name(FIX_MAP1_PA + map_offsets.map_name as u64, b"bpf_runq");

        let result = find_active_struct_ops_obj_no_target(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &prog_offsets,
            &map_offsets,
            START_KERNEL_MAP,
            0,
        );
        assert!(result.is_none());
    }

    /// A STRUCT_OPS prog whose aux->used_maps pointer is NULL yields no
    /// active obj: `find_active_struct_ops_obj` returns None on the
    /// `used_maps_kva == 0` skip. The all-zero fixture also has
    /// used_map_cnt == 0 (whose guard would likewise return None), so
    /// this pins the NULL-used_maps→None outcome, not that one guard in
    /// isolation. used_maps is left unwritten (zero).
    #[test]
    fn find_active_struct_ops_obj_none_when_used_maps_null() {
        let prog_offsets = synthetic_prog_offsets();
        let map_offsets = BpfMapOffsets {
            map_name: 0,
            ..BpfMapOffsets::EMPTY
        };
        // aux->used_maps left 0; used_map_cnt irrelevant.
        let fx = single_prog_fixture(0x6000, BPF_PROG_TYPE_STRUCT_OPS, &prog_offsets);

        let result = find_active_struct_ops_obj_no_target(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &prog_offsets,
            &map_offsets,
            START_KERNEL_MAP,
            0,
        );
        assert!(result.is_none());
    }

    /// The `.min(MAX_USED_MAPS)` clamp on used_map_cnt: a corrupt
    /// used_map_cnt of 70 (> MAX_USED_MAPS=64) must bound the snapshot
    /// loop to exactly 64 reads. Entry index 1 is a global-section
    /// .bss map so the prefix still resolves, but the captured
    /// used_map_kvas vector is capped at 64. The entries beyond index
    /// 1 are never translated (the match returns at index 1), so only
    /// maps 0 and 1 need a real backing struct; the rest are non-zero
    /// KVAs that only enter the snapshot.
    #[test]
    fn find_active_struct_ops_obj_caps_used_map_cnt_at_max_used_maps() {
        let prog_offsets = synthetic_prog_offsets();
        let map_offsets = BpfMapOffsets {
            map_name: 0,
            ..BpfMapOffsets::EMPTY
        };
        let mut fx = single_prog_fixture(0x6000, BPF_PROG_TYPE_STRUCT_OPS, &prog_offsets);

        // used_maps array of 70 non-zero entries (> MAX_USED_MAPS).
        const CORRUPT_CNT: u32 = 70;
        let map0_kva = fx.pa_to_kva(FIX_MAP0_PA);
        let map1_kva = fx.pa_to_kva(FIX_MAP1_PA);
        fx.write_u64(
            FIX_AUX_PA + prog_offsets.aux_used_maps as u64,
            fx.pa_to_kva(FIX_USED_MAPS_PA),
        );
        fx.write_u32(
            FIX_AUX_PA + prog_offsets.aux_used_map_cnt as u64,
            CORRUPT_CNT,
        );
        fx.write_u64(FIX_USED_MAPS_PA, map0_kva);
        fx.write_u64(FIX_USED_MAPS_PA + 8, map1_kva);
        // Entries 2..70: arbitrary non-zero KVAs (never translated —
        // the match returns at index 1). They only populate the
        // snapshot, so the clamp is what bounds the loop.
        for i in 2..CORRUPT_CNT as u64 {
            fx.write_u64(FIX_USED_MAPS_PA + i * 8, 0xDEAD_0000 + i);
        }

        // map0: struct_ops name (no match). map1: .bss (match).
        fx.write_name(FIX_MAP0_PA + map_offsets.map_name as u64, b"ktstr_ops");
        fx.write_name(FIX_MAP1_PA + map_offsets.map_name as u64, b"bpf_bpf.bss");

        let result = find_active_struct_ops_obj_no_target(
            &fx.mem(),
            fx.walk(),
            FIX_IDR_PA + START_KERNEL_MAP,
            &prog_offsets,
            &map_offsets,
            START_KERNEL_MAP,
            0,
        );
        let m = result.unwrap();
        assert_eq!(m.obj_name, "bpf_bpf");
        assert!(m.used_map_kvas.len() <= MAX_USED_MAPS as usize);
        assert_eq!(m.used_map_kvas.len(), 64);
    }
}
