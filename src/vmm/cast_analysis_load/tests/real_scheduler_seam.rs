//! VM-free seam regression test for the cast analyzer against the
//! REAL `scx-ktstr` scheduler binary.
//!
//! Unlike the synthetic-ELF fixtures in the sibling test modules, this
//! runs the full [`super::super::build_cast_analysis_from_bytes`]
//! pipeline over the workspace-built `scx-ktstr` binary and asserts the
//! recovered `cast_maps[0]` carries the exact cross-subprog findings a
//! failure-dump render depends on:
//!
//! - `(ktstr_arena_ctx=772, 24) -> Arena`: the `stashed_arena_ptr`
//!   field written in `ktstr_cross_btf_chase` THROUGH a `taskc`
//!   returned by the untyped `void __arena *scx_task_alloc(...)`. This
//!   is the finding the typed-allocator-return inference recovers — the
//!   analyzer types `scx_task_alloc`'s return as a typed
//!   `ArenaU64FromAlloc{struct_type_id: ktstr_arena_ctx}` (size 32 →
//!   unique payload via `discover_payload_btf_id`), which as an STX base
//!   keys against parent 772. WITHOUT the inference this key is absent
//!   and the field renders as a plain `Uint` (the FIXPOINT REGRESSION).
//! - `(ktstr_arena_ctx=772, 16) -> Kernel`: `task_kptr`, recovered via
//!   the directly-typed `ktstr_stash_task_kptr` param (control: proves
//!   the pipeline typed the struct at all).
//! - `(ktstr_cross_btf_value=690, 0) -> Arena`: the publish side —
//!   `cached_ptr` stashed from `scx_static_alloc` (control: proves the
//!   arena STX-flow path is intact).
//!
//! The struct ids (772/690) are stable properties of the compiled
//! `scx-ktstr` BTF; the test resolves them by name from the parsed BTF
//! rather than hard-coding, so a BTF-layout shift does not silently
//! void the assertions.
//!
//! Skips (does not fail) when no `scx-ktstr` binary is present — a
//! bare unit-test lane that never built the scheduler has nothing to
//! check. Every lane that builds `scx-ktstr` (the cast e2e / gauntlet
//! lanes) exercises the real proof.

use std::path::PathBuf;

use btf_rs::{Btf, Type};

/// Locate the workspace-built `scx-ktstr` outer scheduler ELF (the one
/// carrying the embedded `.bpf.objs`). Prefers the release profile
/// (what the cast e2e lane builds), then debug. Honors an explicit
/// `CARGO_TARGET_DIR` override before falling back to the crate's
/// `target/`.
fn locate_scx_ktstr() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        roots.push(PathBuf::from(dir));
    }
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    for root in roots {
        for profile in ["release", "debug"] {
            let candidate = root.join(profile).join("scx-ktstr");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Resolve a program-BTF struct id by name from the first embedded
/// object's parsed BTF. Returns the first `Struct` whose resolved name
/// matches — the compiled `scx-ktstr` BTF carries exactly one of each
/// of these names.
fn struct_id_by_name(btf: &Btf, want: &str) -> Option<u32> {
    let (first, last) = match btf.split() {
        Some(prog) => prog.type_id_range(),
        None => btf.base().type_id_range(),
    };
    for id in first..=last {
        if let Ok(Type::Struct(s)) = btf.resolve_type_by_id(id)
            && let Ok(name) = btf.resolve_name(&s)
            && name == want
        {
            return Some(id);
        }
    }
    None
}

#[test]
fn cast_map_recovers_cross_subprog_arena_findings_from_real_scx_ktstr() {
    let Some(path) = locate_scx_ktstr() else {
        eprintln!(
            "real_scheduler_seam: no scx-ktstr binary under target/{{release,debug}} \
             (build it with `cargo build -p scx-ktstr`); skipping the real-binary proof"
        );
        return;
    };
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read scx-ktstr binary {}: {e}", path.display()));

    let out = super::super::build_cast_analysis_from_bytes(&bytes);
    let btf = out
        .btfs
        .first()
        .expect("scx-ktstr produced no embedded program BTF");
    let cast_map = out
        .cast_maps
        .first()
        .expect("scx-ktstr produced no cast map");

    use crate::monitor::cast_analysis::AddrSpace;
    let arena_ctx = struct_id_by_name(btf, "ktstr_arena_ctx")
        .expect("ktstr_arena_ctx not found in scx-ktstr program BTF");
    let cross_btf_value = struct_id_by_name(btf, "ktstr_cross_btf_value")
        .expect("ktstr_cross_btf_value not found in scx-ktstr program BTF");
    let bss_holder = struct_id_by_name(btf, "ktstr_bss_arena_holder")
        .expect("ktstr_bss_arena_holder not found in scx-ktstr program BTF");

    // The FULL expected scheduler-specific finding set — every one of
    // these backs a `cast_analysis_*` e2e assertion, so asserting the
    // whole set here catches a PARTIAL regression (one finding lost
    // while another is recovered) before it ships. `(53,*)` sdt_desc
    // library internals are intentionally NOT asserted: they belong to
    // the sdt_alloc lib, not this fixture, and may shift across lib
    // revisions.
    //
    //   (ktstr_arena_ctx, 16)  Kernel — `task_kptr`, via the directly
    //       typed `ktstr_stash_task_kptr` param.
    //   (ktstr_arena_ctx, 24)  Arena  — `stashed_arena_ptr`, the
    //       typed-allocator-return path (scx_task_alloc → chase).
    //   (ktstr_cross_btf_value, 0) Arena — publish-side `cached_ptr`.
    //   (ktstr_bss_arena_holder, 0) Arena — the bss→arena trainer's
    //       LDX-side detection. This one is the canary for the
    //       allocator-return-typing interaction: if the allocator
    //       return were typed as a plain `Pointer{arena_ctx}` (rather
    //       than a typed `ArenaU64FromAlloc`), the init_task store
    //       `holder->arena_target = taskc` would record a spurious
    //       KPTR finding here and finalize would drop the slot.
    let expected: &[((u32, u32), AddrSpace, &str)] = &[
        (
            (arena_ctx, 16),
            AddrSpace::Kernel,
            "ktstr_arena_ctx.task_kptr",
        ),
        (
            (arena_ctx, 24),
            AddrSpace::Arena,
            "ktstr_arena_ctx.stashed_arena_ptr (typed-allocator-return path)",
        ),
        (
            (cross_btf_value, 0),
            AddrSpace::Arena,
            "ktstr_cross_btf_value.cached_ptr (publish side)",
        ),
        (
            (bss_holder, 0),
            AddrSpace::Arena,
            "ktstr_bss_arena_holder.arena_target (bss→arena trainer)",
        ),
    ];

    let keys: Vec<_> = cast_map.keys().collect();
    let mut missing = Vec::new();
    let mut wrong_space = Vec::new();
    for &(key, space, label) in expected {
        match cast_map.get(&key) {
            None => missing.push(format!("{label} {key:?}")),
            Some(hit) if hit.addr_space != space => wrong_space.push(format!(
                "{label} {key:?}: expected {space:?}, got {:?}",
                hit.addr_space
            )),
            Some(_) => {}
        }
    }
    assert!(
        missing.is_empty() && wrong_space.is_empty(),
        "cast_map regression against {}:\n  missing: {missing:?}\n  wrong addr_space: {wrong_space:?}\n  actual keys: {keys:?}",
        path.display()
    );

    eprintln!(
        "real_scheduler_seam: OK against {} — full expected set present: \
         (arena_ctx={arena_ctx}:16 Kernel, :24 Arena), \
         (cross_btf_value={cross_btf_value}:0 Arena), \
         (bss_holder={bss_holder}:0 Arena)",
        path.display()
    );
}
