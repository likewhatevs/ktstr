// Builds scx-ktstr's BPF object. The scheduler exercises the
// `sdt_alloc` arena allocator on top of the BPF arena so that ktstr's
// failure-dump arena renderer has real allocator-shaped state to
// capture. The two `.bpf.c` files that implement the allocator
// (`lib/sdt_alloc.bpf.c`, `lib/sdt_task.bpf.c`) and the one header
// they pull in via a quoted include (`lib/scxtest/scx_test.h`) are
// fetched from the upstream scx repo at build time rather than
// vendored into this tree.

use std::env;
use std::path::{Path, PathBuf};

use ahash as gix_acquire_ahash;
use fs2 as gix_acquire_fs2;
use gix as gix_acquire_gix;
use jobserver as gix_acquire_jobserver;

#[path = "../build_support/gix_acquire.rs"]
mod gix_acquire;

// Exact scx release that produced the `scx_cargo` / `scx_utils` 1.1.1
// crates and their bundled `scx_utils-bpf_h` headers. Keeping the fetched
// allocator sources on the same release matters beyond API spelling: v1.1.1
// contains an allocator-loop verifier fix and refreshed per-architecture
// kernel headers. Tags ≤ v1.0.11 ship the older `sdt_*` API
// (the rename to `scx_*` first appeared at v1.0.12, verified by
// reading `lib/sdt_task.bpf.c` at both tags); pinning to a pre-rename
// tag would link-fail against scx_cargo 1.1.1 headers.
const SCX_TAG: &str = "v1.1.1";
const SCX_REV: &str = "0eedd05bc233129fd3c884d7045edeb2c2a474a7";
const SCX_URL: &str = "https://github.com/sched-ext/scx.git";
const SCX_TAG_SENTINEL: &str = ".scx-tag";

// Only this source subset is published from the private exact checkout.
const SCX_FETCH_FILES: &[&str] = &[
    "lib/sdt_alloc.bpf.c",
    "lib/sdt_task.bpf.c",
    "lib/scxtest/scx_test.h",
];
const SCX_FETCH_MANIFEST: &str = "lib/sdt_alloc.bpf.c\nlib/sdt_task.bpf.c\nlib/scxtest/scx_test.h";

fn scx_source_stamp() -> String {
    format!("{SCX_TAG}\n{SCX_REV}\n")
}

fn main() {
    println!("cargo:rerun-if-changed=../build_support/gix_acquire.rs");
    for variable in ["KTSTR_CACHE_DIR", "XDG_CACHE_HOME", "HOME"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let scx_lib = out_dir.join("scx-lib");

    fetch_scx_lib(&scx_lib);

    // Compile main.bpf.c plus the fetched allocator sources into a
    // single linked BPF object. `sdt_alloc.bpf.c` provides the
    // `scx_alloc_*` allocator surface and pulls in the arena map
    // definition via `<lib/arena_map.h>` (the actual file that
    // declares `arena __weak SEC(".maps")`); `sdt_task.bpf.c`
    // defines the task-storage map and the `scx_task_*` per-task
    // wrappers that main.bpf.c calls. `compile_link_gen` writes
    // one `bpf.bpf.o` from all three sources via libbpf-rs's
    // bpf_linker.
    //
    // Both source files include `"scxtest/scx_test.h"` with a quoted
    // include, so the selected-source copy preserves their upstream
    // `lib/` layout under OUT_DIR.
    let sdt_alloc_path = scx_lib.join("lib/sdt_alloc.bpf.c");
    let sdt_task_path = scx_lib.join("lib/sdt_task.bpf.c");

    scx_cargo::BpfBuilder::new()
        .expect("BpfBuilder::new")
        .enable_skel("src/bpf/main.bpf.c", "bpf")
        .add_source(
            sdt_alloc_path
                .to_str()
                .expect("OUT_DIR/scx-lib/lib/sdt_alloc.bpf.c path must be UTF-8"),
        )
        .add_source(
            sdt_task_path
                .to_str()
                .expect("OUT_DIR/scx-lib/lib/sdt_task.bpf.c path must be UTF-8"),
        )
        .compile_link_gen()
        .expect("BpfBuilder::compile_link_gen");
}

/// Obtain the selected scx sources once per machine cache. Builder election
/// happens before the network connection, so concurrent Cargo build scripts
/// wait for and reuse one exact checkout rather than all fetching in parallel.
fn fetch_scx_lib(dest: &Path) {
    if scx_lib_complete(dest) {
        return;
    }
    let key_parts = [
        "scx-selected-source-v1",
        SCX_URL,
        SCX_TAG,
        SCX_REV,
        SCX_FETCH_MANIFEST,
    ];
    let cache_root = gix_acquire::cache_root("scx-lib").unwrap_or_else(|| {
        println!(
            "cargo:warning=no absolute cache home; scx source reuse is limited \
             to this Cargo output directory"
        );
        dest.parent()
            .expect("OUT_DIR/scx-lib has a parent")
            .join(".ktstr-content-cache")
            .join("scx-lib")
    });
    let expected_entry = gix_acquire::cache_entry(&cache_root, &key_parts);
    let source_cache_root = gix_acquire::cache_root("source-nodes").unwrap_or_else(|| {
        dest.parent()
            .expect("OUT_DIR/scx-lib has a parent")
            .join(".ktstr-content-cache")
            .join("source-nodes")
    });

    // Preserve the manual-placement escape hatch: the selected files plus
    // `.scx-tag` are sufficient even if an operator populated the computed
    // content path without ktstr's private completion sentinel.
    let shared = if scx_lib_complete(&expected_entry) {
        expected_entry
    } else {
        gix_acquire::ensure_cached(
            &cache_root,
            &key_parts,
            "scx selected-source acquisition",
            scx_lib_complete,
            |stage, progress| populate_scx_stage(stage, &source_cache_root, progress),
        )
        .unwrap_or_else(|err| {
            panic!(
                "failed to obtain scx {SCX_TAG} sources with in-process gix: {err}\n\
                 First build requires HTTPS access to {SCX_URL}.\n\
                 If network acquisition is unavailable, manually place the files at\n\
                 {}/lib/{{sdt_alloc.bpf.c,sdt_task.bpf.c,scxtest/scx_test.h}}\n\
                 and write the tag and pinned commit on separate lines to {}/{}:\n\
                 {SCX_TAG}\n{SCX_REV}",
                expected_entry.display(),
                expected_entry.display(),
                SCX_TAG_SENTINEL,
            )
        })
    };
    copy_scx_lib(&shared, dest);
}

fn populate_scx_stage(
    stage: &Path,
    source_cache_root: &Path,
    progress: &gix_acquire::ProgressReporter,
) -> Result<(), String> {
    let checkout = stage.join(".checkout");
    let source_ref = format!("refs/tags/{SCX_TAG}");
    gix_acquire::assemble_exact_cached(
        source_cache_root,
        SCX_URL,
        &source_ref,
        SCX_REV,
        &checkout,
        progress,
    )?;
    progress.set_phase("publishing selected scx source files");
    for relative in SCX_FETCH_FILES {
        let source = checkout.join(relative);
        if !source.is_file() {
            return Err(format!(
                "exact scx checkout is missing expected file {relative}"
            ));
        }
        let destination = stage.join(relative);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("create {}: {err}", parent.display()))?;
        }
        std::fs::copy(&source, &destination)
            .map_err(|err| format!("copy selected source {relative}: {err}"))?;
    }
    std::fs::write(stage.join(SCX_TAG_SENTINEL), scx_source_stamp())
        .map_err(|err| format!("stamp selected scx source: {err}"))?;
    std::fs::remove_dir_all(&checkout)
        .map_err(|err| format!("remove private scx checkout: {err}"))?;
    Ok(())
}

fn copy_scx_lib(shared: &Path, dest: &Path) {
    if dest.exists() {
        std::fs::remove_dir_all(dest).expect("remove stale OUT_DIR/scx-lib");
    }
    for relative in SCX_FETCH_FILES {
        let source = shared.join(relative);
        let destination = dest.join(relative);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|err| panic!("create {}: {err}", parent.display()));
        }
        std::fs::copy(&source, &destination).unwrap_or_else(|err| {
            panic!(
                "copy {} -> {}: {err}",
                source.display(),
                destination.display()
            )
        });
    }
    std::fs::write(dest.join(SCX_TAG_SENTINEL), scx_source_stamp())
        .expect("stamp OUT_DIR scx source selection");
}

fn scx_lib_complete(path: &Path) -> bool {
    SCX_FETCH_FILES
        .iter()
        .all(|relative| path.join(relative).is_file())
        && std::fs::read_to_string(path.join(SCX_TAG_SENTINEL))
            .is_ok_and(|stamp| stamp == scx_source_stamp())
}
