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
//!   analyzer types `scx_task_alloc`'s return as `Pointer{ktstr_arena_ctx}`
//!   (size 32 → unique payload via `discover_payload_btf_id`) so the STX
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

    let arena_ctx = struct_id_by_name(btf, "ktstr_arena_ctx")
        .expect("ktstr_arena_ctx not found in scx-ktstr program BTF");
    let cross_btf_value = struct_id_by_name(btf, "ktstr_cross_btf_value")
        .expect("ktstr_cross_btf_value not found in scx-ktstr program BTF");

    // Control findings: prove the pipeline typed the structs and the
    // arena STX-flow path is intact. A regression here means something
    // OTHER than the typed-allocator-return inference broke.
    assert!(
        cast_map.contains_key(&(arena_ctx, 16)),
        "(ktstr_arena_ctx={arena_ctx}, 16) [task_kptr, directly-typed param] \
         missing — the analyzer failed to type the struct at all. \
         cast_map keys: {:?}",
        cast_map.keys().collect::<Vec<_>>()
    );
    assert!(
        cast_map.contains_key(&(cross_btf_value, 0)),
        "(ktstr_cross_btf_value={cross_btf_value}, 0) [publish-side cached_ptr] \
         missing — the arena STX-flow path regressed. cast_map keys: {:?}",
        cast_map.keys().collect::<Vec<_>>()
    );

    // The finding under test: recovered ONLY when the typed-allocator-
    // return inference types scx_task_alloc's `void __arena *` return
    // as Pointer{ktstr_arena_ctx}, so the chase-side STX keys parent 772.
    let hit = cast_map.get(&(arena_ctx, 24));
    assert!(
        hit.is_some(),
        "FIXPOINT REGRESSION: (ktstr_arena_ctx={arena_ctx}, 24) [stashed_arena_ptr] \
         missing — the typed-allocator-return inference did not carry \
         Pointer{{ktstr_arena_ctx}} across the scx_task_alloc -> \
         ktstr_cross_btf_chase boundary. Without it stashed_arena_ptr \
         renders as a plain Uint. cast_map keys: {:?}",
        cast_map.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        hit.unwrap().addr_space,
        crate::monitor::cast_analysis::AddrSpace::Arena,
        "(ktstr_arena_ctx={arena_ctx}, 24) recovered but not tagged Arena",
    );

    eprintln!(
        "real_scheduler_seam: OK against {} — (772={arena_ctx},24) Arena, \
         (772,16) control, (690={cross_btf_value},0) control all present",
        path.display()
    );
}
