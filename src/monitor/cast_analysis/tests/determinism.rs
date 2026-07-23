use super::*;

// ----- Fixpoint / finalize determinism ------------------------
//
// [`analyze_casts`] must be a pure function of its (insns, btf,
// seeds) inputs: the same program analyzed twice — in the same
// process or across processes — must yield a byte-identical
// [`CastMap`]. The renderer keys cast findings by
// `(parent_type_id, offset)`, and a struct field that resolves to
// a cast pointer on one run but a plain `u64` on another surfaces
// downstream as an intermittent "rendered as Uint instead of Ptr"
// failure (see the `cast_analysis_e2e` scenarios).
//
// The analyzer's carried fixpoint state is `BTreeMap`/`BTreeSet`;
// the only unordered containers in the output path are the
// `HashSet<u32>` candidate sets built by `build_layout_index` and
// intersected in `finalize`'s shape-inference / STX-flow arms.
// `HashSet` iteration order is randomized per process (a fresh
// `RandomState` seed per `HashSet::new`), so these loops re-run the
// analysis many times WITHIN one process — each `analyze_casts`
// call mints fresh internal maps with advancing hasher seeds — and
// assert the result never diverges from the first. A regression
// that lets candidate-set iteration order leak into the emitted
// `target_type_id` (or into whether a slot emits at all) fails here
// deterministically rather than flaking in CI.

/// Iterations per determinism assertion. Each call to
/// [`analyze_casts`] constructs fresh `HashSet`/`HashMap` state, and
/// the default `RandomState` advances a per-thread seed counter on
/// every construction, so a few thousand back-to-back runs exercise
/// a wide spread of iteration orders in a single process.
const DETERMINISM_ITERS: usize = 4000;

fn assert_stable<F>(build: F)
where
    F: Fn() -> CastMap,
{
    let first = build();
    for i in 1..DETERMINISM_ITERS {
        let next = build();
        assert_eq!(
            next, first,
            "analyze_casts diverged on iteration {i}: candidate-set \
             iteration order leaked into the CastMap. first={first:?} \
             next={next:?}"
        );
    }
}

/// Unique shape-inference target: the `(offset, size)` access
/// pattern resolves `candidates` to exactly one BTF id, so the
/// `candidates.len() == 1` pick must return that id on every run.
#[test]
fn shape_inference_unique_target_is_order_stable() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(4, 2, 1),
        ldx(BPF_SIZE_DW, 3, 4, 0),
        exit(),
    ];
    assert_stable(|| {
        analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
            &[],
        )
    });
    // Sanity: the stable value is the resolved target, not a drop.
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(t_id, 8)).map(|h| h.target_type_id),
        Some(q_id),
        "expected unique shape-inference to resolve to Q: {map:?}"
    );
}

/// Ambiguous shape (two same-shape targets Q1/Q2): the candidate
/// intersection holds two ids, so `candidates.len() == 1` fails and
/// the slot takes the deferred-resolve emit (`target_type_id == 0`)
/// via `arena_confirmed`. The choice between "resolve" and "defer"
/// must not depend on which candidate `HashSet` iteration visits
/// first.
#[test]
fn shape_inference_ambiguous_target_defers_order_stable() {
    let (blob, t_id) = btf_source_and_two_targets(8);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r2 = T.f (u64 @ 8); r4 = addr_space_cast(r2) -> arena_confirmed
    // for (T, 8); r3 = *(u64*)(r4 + 0) -> pattern access (0, 8) which
    // matches BOTH Q1 and Q2 in the layout index.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(4, 2, 1),
        ldx(BPF_SIZE_DW, 3, 4, 0),
        exit(),
    ];
    assert_stable(|| {
        analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
            &[],
        )
    });
    // Sanity: ambiguity defers (target 0), never leaks Q1 vs Q2.
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "ambiguous shape must defer with target_type_id=0, never pick \
         Q1 or Q2 by iteration order: {map:?}"
    );
}

/// STX-flow arena finding whose slot ALSO carries an ambiguous
/// shape: the finding is emitted unconditionally (target 0), but the
/// `inferred_target` computation runs the same candidate `HashSet`
/// intersection. Confirm the emitted hit is byte-stable regardless
/// of iteration order.
#[test]
fn stx_flow_with_ambiguous_shape_is_order_stable() {
    let (blob, t_id) = btf_source_and_two_targets(8);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Insn 0: pseudo-call flagged as an allocator return -> R0 tagged
    //         ArenaU64FromAlloc.
    // Insn 1: STX [R6 + 8] = R0 -> records (T, 8) as an Arena STX
    //         finding; R6 seeded Pointer{T}.
    // Insn 2: LDX r3 = *(u64*)(r6 + 8) then deref at 0 would give a
    //         shape access, but the STX-flow emit does not require it.
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let insns = vec![pseudo_call, stx(BPF_SIZE_DW, 6, 0, 8), exit()];
    assert_stable(|| {
        analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: t_id,
            }],
            &[],
            &[],
            &[SubprogReturn {
                alloc_size: None,
                insn_offset: 0,
            }],
        )
    });
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "STX-flow arena finding must be byte-stable: {map:?}"
    );
}
