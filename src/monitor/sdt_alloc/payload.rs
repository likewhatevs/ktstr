//! BTF payload-type discovery heuristic — picks a struct type id
//! whose size matches the sdt_alloc slot stride so the renderer can
//! produce named-field output instead of a raw hex dump.

use btf_rs::{Btf, Type};

use super::MAX_BTF_ID_PROBE;

/// Result of [`discover_payload_btf_id`] — pairs the chosen BTF type
/// id (0 for fallback) with a human-readable reason describing the
/// fallback path when the id is 0.
#[derive(Debug, Clone)]
pub struct PayloadTypeChoice {
    pub target_type_id: u32,
    pub reason: String,
}

/// Heuristic: pick a payload BTF type id matching the slot stride.
///
/// `pool.elem_size = sizeof(sdt_data) + payload_size`, rounded up to
/// 8. So `payload_size = elem_size - sizeof(sdt_data)`. We search the
/// BTF for struct types whose `.size()` equals `payload_size` exactly,
/// then narrow:
///
///   1. Exactly one match → use it.
///   2. Multiple matches → prefer names matching the conventional
///      patterns: `task_ctx` (exact), then `*_arena_ctx`,
///      `*_task_ctx`, `*_ctx` (suffix). scx schedulers consistently
///      use these suffixes; ktstr's own test fixture struct is
///      `ktstr_arena_ctx`. If 2+ structs match the same pattern arm,
///      the heuristic continues to lower-priority arms — a collision
///      at a higher-specificity level does not prevent a lower-
///      specificity unambiguous match from resolving.
///   3. No match or still ambiguous → return 0 to fall back to a hex
///      dump.
///
/// The function is intentionally conservative: a wrong type id renders
/// nonsense field names; falling back to hex always shows the operator
/// raw bytes they can decode by hand. The returned reason string is
/// surfaced to the operator via `SdtAllocatorSnapshot::payload_type_reason`
/// so the fallback paths are distinguishable without re-running the
/// heuristic.
pub fn discover_payload_btf_id(
    btf: &Btf,
    payload_size: usize,
    allocator_name: &str,
) -> PayloadTypeChoice {
    if payload_size == 0 {
        return PayloadTypeChoice {
            target_type_id: 0,
            reason: "payload_size == 0".into(),
        };
    }
    let mut size_matches: Vec<(u32, String)> = Vec::new();

    // For split BTF — the scheduler's program types layered on the
    // vmlinux base — probe ONLY the program section's id range so
    // vmlinux base structs of the same size (kernel `*_ctx` structs,
    // etc.) cannot shadow the scheduler's payload struct. `Btf::split`
    // returns the program section for split BTF (None otherwise) and
    // its `type_id_range` starts at `base_nr_types + 1`; for non-split
    // BTF the whole object is the candidate set.
    //
    // Within that range we keep an explicit id-probe loop (rather than
    // `Btf::type_iter`) so a run of CONSECUTIVE_FAIL_CAP consecutive
    // lookup failures early-bails on a sparse-id table. `probe_scan_end`
    // bounds the WINDOW to MAX_BTF_ID_PROBE ids from `first_id` — for
    // split BTF `first_id` is base-relative (`base_nr_types + 1`), so an
    // ABSOLUTE ceiling would scan zero ids once the vmlinux base exceeds
    // MAX_BTF_ID_PROBE (real DEBUG_INFO_BTF kernels carry 80k-160k+ base
    // types). Real ktstr program BTFs top out in the low thousands, so
    // the window normally covers the whole program range.
    const CONSECUTIVE_FAIL_CAP: u32 = 64;

    let (first_id, last_id) = match btf.split() {
        Some(prog) => prog.type_id_range(),
        None => btf.base().type_id_range(),
    };
    let scan_end = probe_scan_end(first_id, last_id);

    let mut tid: u32 = first_id.max(1);
    let mut consecutive_fail: u32 = 0;
    while tid <= scan_end {
        match btf.resolve_type_by_id(tid) {
            Ok(ty) => {
                consecutive_fail = 0;
                if let Type::Struct(s) = ty
                    && s.size() == payload_size
                    && let Ok(name) = btf.resolve_name(&s)
                    && !name.is_empty()
                {
                    size_matches.push((tid, name));
                }
            }
            Err(_) => {
                consecutive_fail += 1;
                if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                    break;
                }
            }
        }
        tid += 1;
    }

    match size_matches.len() {
        0 => PayloadTypeChoice {
            target_type_id: 0,
            reason: format!("no candidate of size {payload_size}"),
        },
        1 => PayloadTypeChoice {
            target_type_id: size_matches[0].0,
            reason: String::new(),
        },
        n => {
            // Multiple size-match candidates. First try matching the
            // allocator name: `scx_atq_allocator` → strip `_allocator`
            // → `scx_atq` → match struct named `scx_atq`. This is the
            // highest-specificity signal because scx_static_alloc
            // allocators are conventionally named after their payload type.
            let name_stem = allocator_name
                .strip_suffix("_allocator")
                .unwrap_or(allocator_name);
            if !name_stem.is_empty() {
                let stems: &[&str] = &[name_stem, name_stem.strip_prefix("scx_").unwrap_or("")];
                for stem in stems {
                    if stem.is_empty() {
                        continue;
                    }
                    let hits: Vec<u32> = size_matches
                        .iter()
                        .filter(|(_, sn)| sn == stem)
                        .map(|(id, _)| *id)
                        .collect();
                    if hits.len() == 1 {
                        return PayloadTypeChoice {
                            target_type_id: hits[0],
                            reason: String::new(),
                        };
                    }
                }
            }

            // Fall through to conventional suffix patterns. Order is
            // most-specific first so a struct named exactly `task_ctx`
            // wins over `foo_task_ctx`. If 2+ structs share the SAME
            // pattern arm, the loop continues to lower-priority arms.
            type Pat = fn(&str) -> bool;
            let patterns: &[Pat] = &[
                |n: &str| n == "task_ctx",
                |n: &str| n.ends_with("_arena_ctx"),
                |n: &str| n.ends_with("_task_ctx"),
                |n: &str| n.ends_with("_ctx"),
            ];
            for pat in patterns {
                let hits: Vec<u32> = size_matches
                    .iter()
                    .filter(|(_, n)| pat(n))
                    .map(|(id, _)| *id)
                    .collect();
                match hits.len() {
                    0 => continue,
                    1 => {
                        return PayloadTypeChoice {
                            target_type_id: hits[0],
                            reason: String::new(),
                        };
                    }
                    _ => {
                        // 2+ matches in the SAME pattern arm —
                        // ambiguous at this priority level. Continue
                        // to the next (lower-priority) pattern arm:
                        // a higher-specificity collision shouldn't
                        // prevent a lower-specificity unambiguous
                        // match from resolving.
                        continue;
                    }
                }
            }
            // No unambiguous pattern winner — fall back to hex.
            PayloadTypeChoice {
                target_type_id: 0,
                reason: format!("ambiguous: {n} candidates"),
            }
        }
    }
}

/// Inclusive upper id for the probe scan window. Bounds the NUMBER of
/// ids probed to [`MAX_BTF_ID_PROBE`] starting at `first_id`, rather
/// than an absolute id ceiling: for split BTF `first_id` is
/// base-relative (`base_nr_types + 1`), so an absolute ceiling would
/// scan zero ids once the vmlinux base exceeds MAX_BTF_ID_PROBE. Also
/// clamped to `last_id` (the section's real last id).
fn probe_scan_end(first_id: u32, last_id: u32) -> u32 {
    last_id.min(first_id.saturating_add(MAX_BTF_ID_PROBE).saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_scan_end_bounds_window_not_absolute_id() {
        // Small base (non-split or tiny base): the window dwarfs the
        // program range, so the scan end is simply `last_id`.
        assert_eq!(probe_scan_end(1, 50), 50);
        assert_eq!(probe_scan_end(3, 500), 500);

        // Large base pushes first_id past the absolute cap. An absolute
        // `MAX_BTF_ID_PROBE - 1` (= 99_999) ceiling would be < first_id
        // → an empty scan window → every split payload silently
        // unresolved (the empty-scan regression). The window-relative
        // bound must still reach the program range.
        let first = MAX_BTF_ID_PROBE + 1; // 100_001, base ~100k types
        let last = first + 4; // a 5-type program section
        let end = probe_scan_end(first, last);
        assert_eq!(end, last, "window must cover the full program range");
        assert!(
            end >= first,
            "scan window must reach first_id even when the base exceeds \
             MAX_BTF_ID_PROBE (got end={end}, first={first})"
        );

        // A corrupt/huge last_id is bounded to MAX_BTF_ID_PROBE ids
        // from first_id (defensive window cap).
        assert_eq!(probe_scan_end(10, u32::MAX), 10 + MAX_BTF_ID_PROBE - 1);
        // saturating arithmetic guards first_id near u32::MAX.
        assert_eq!(probe_scan_end(u32::MAX, u32::MAX), u32::MAX - 1);
    }
}
