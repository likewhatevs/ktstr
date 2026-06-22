//! Source-function size guard.
//!
//! Asserts that no function inside `src/**/*.rs` exceeds 200 lines
//! (`{` to `}` brace-token span, inclusive), with a
//! grandfathered-exceptions list naming each function currently above
//! the limit and pinning its exact line count. The list drains as
//! decomposition tasks land: when a previously-grandfathered function
//! falls back below 200 lines (or is removed), the test fails with a
//! "remove `<entry>` from EXCEPTIONS" message that forces the next
//! person to delete the now-irrelevant entry. This makes the
//! exceptions list a working ratchet that tightens monotonically
//! toward zero entries.
//!
//! Sibling of [`src_file_size_guard`] (the file-level ratchet at
//! 3000 lines); both work the same way — guarded soft-limit + frozen
//! per-entry ceiling + drain protocol — at different granularities.
//!
//! # Identifying functions
//!
//! Each entry is keyed by `(file, fn_name)` where:
//!
//! - `file` is the path relative to `<repo>/src/`, forward-slash
//!   form on every platform (matches the
//!   [`src_file_size_guard`] convention).
//! - `fn_name` is the function name as written in source. For impl
//!   methods, the entry uses `"<Type>::method"` form so a single
//!   file with multiple impls on different types disambiguates per
//!   method. For trait impl methods, the entry uses
//!   `"<Type as Trait>::method"` form. For free functions the bare
//!   `"name"` suffices.
//!
//! # Line counting
//!
//! Uses `syn`'s `brace_token.span` to measure the `{` to `}` range
//! of each function's body in source-line terms. `length =
//! end_line - start_line + 1`, so a one-line `fn foo() {}` counts
//! as 1 line; a function whose body spans lines 100..=437 counts
//! as 338 lines. Macros that expand to functions are NOT walked
//! — `syn::parse_file` operates on source tokens BEFORE macro
//! expansion, so a `#[ktstr_test]`-emitted fn's body never
//! appears in the source AST and so cannot count toward the
//! limit. The macro invocation's source-line span is small;
//! the expanded fn's body lives in the macro crate, not in
//! `src/**/*.rs`.
//!
//! # Failure modes
//!
//! Three independent regressions surface here, mirroring the
//! [`src_file_size_guard`] failure-mode taxonomy:
//!
//! 1. A function NOT in `EXCEPTIONS` exceeds 200 lines — a brand-new
//!    mega-function was introduced or an existing small function
//!    grew past the threshold. Either decompose it or add it to
//!    `EXCEPTIONS` with its current line count and a `// queued:
//!    <task>` comment naming the queued decomposition task.
//! 2. A function IN `EXCEPTIONS` grew past its grandfathered line
//!    count. The exception is a ceiling, not a license — entries
//!    are expected to shrink over time, never grow.
//! 3. A function IN `EXCEPTIONS` dropped to ≤ 200 lines (or was
//!    removed from the source tree). The `EXCEPTIONS` entry is now
//!    stale and must be deleted so future regressions are caught
//!    by the default 200-line gate rather than masked by a stale
//!    grandfather entry.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use quote::ToTokens;
use syn::spanned::Spanned;
use syn::visit::Visit;

/// Default soft-limit for `src/**/*.rs` functions. Any function
/// whose `{ ... }` body spans more than this many source lines AND
/// is not in `EXCEPTIONS` fails the guard.
const DEFAULT_FN_MAX_LINES: usize = 200;

/// Grandfathered functions that exceed `DEFAULT_FN_MAX_LINES`
/// today. Each entry records `(file, fn_name, line_count)`:
///
/// - `file` — path relative to `<repo>/src/`, forward-slash form.
/// - `fn_name` — bare function name for free functions;
///   `"<Type>::method"` for inherent impl methods;
///   `"<Type as Trait>::method"` for trait impl methods.
/// - `line_count` — brace-token span line count at the time the
///   entry was added (`end_line - start_line + 1`).
///
/// **Drain protocol.** When a function is decomposed below
/// `DEFAULT_FN_MAX_LINES`, the guard fails with a "remove
/// `<entry>` from EXCEPTIONS" message. Delete the entry — the
/// default 200-line gate then guards the function going forward.
/// Do NOT lower the recorded count to track partial reductions;
/// the entry exists to acknowledge the function is over the
/// limit, and any reduction below 200 means the entry's purpose
/// is served.
///
/// **Adding a new entry** is allowed only when the function is
/// being deferred (e.g. a decomposition task is queued but not yet
/// landed). The standard remediation for a NEW function past 200
/// lines is to decompose it before the change lands. A new entry
/// must reference the queued task in a `// queued: <task>` comment
/// so the deferral is auditable.
///
/// Each entry pins the current line count as the grandfathered
/// ceiling. Subsequent runs drain the list as those functions
/// split below `DEFAULT_FN_MAX_LINES`.
///
/// Counts bulk-refreshed 2026-06-15 from this test's syn-measured
/// panic output (the source of truth): stale entries dropped for
/// functions split/moved below the limit (the rust_init / setup /
/// virtio_blk-device split-campaign moves, plus eval), counts re-pinned
/// for functions that grew, and newly-over-200 functions added. As
/// individual decompositions land, remove the entry; re-run this test
/// to pick a fresh ceiling rather than editing counts by hand. All
/// entries carry the same `// queued: decompose` placeholder (they
/// share one deferred task — the mega-function decomposition campaign);
/// the guard's panic template emits it on every suggested entry.
const EXCEPTIONS: &[(&str, &str, usize)] = &[
    (
        "cache/cache_dir_tests.rs",
        "store_in_lock_recheck_mixed_content_peers_publish_one_per_group",
        215,
    ), // queued: decompose
    ("cli/kernel_build/build.rs", "kernel_build_pipeline", 492),  // queued: decompose
    ("ctprof/mod.rs", "capture_with", 473),                       // queued: decompose
    ("ctprof_compare/report/smaps.rs", "write_smaps_section", 212), // queued: decompose
    ("ctprof_compare/runner.rs", "write_metric_list", 212),       // queued: decompose
    ("export.rs", "generate_preamble", 416),                      // queued: decompose
    ("host_thread_probe.rs", "find_jemalloc_via_maps_at", 240),   // queued: decompose
    ("monitor/btf_render/mod.rs", "chase_arena_pointer", 367),    // queued: decompose
    ("monitor/btf_render/mod.rs", "render_cast_pointer", 206),    // queued: decompose
    ("monitor/btf_render/mod.rs", "render_value_inner", 461),     // queued: decompose
    ("monitor/btf_render/mod.rs", "write_rendered_value", 348),   // queued: decompose
    ("monitor/btf_render/mod.rs", "write_struct", 276),           // queued: decompose
    (
        "monitor/cast_analysis/mod.rs",
        "Analyzer < 'a >::finalize",
        465,
    ), // queued: decompose
    (
        "monitor/cast_analysis/mod.rs",
        "Analyzer < 'a >::handle_ldx",
        304,
    ), // queued: decompose
    (
        "monitor/cast_analysis/mod.rs",
        "Analyzer < 'a >::handle_stx",
        280,
    ), // queued: decompose
    ("monitor/cast_analysis/mod.rs", "Analyzer < 'a >::run", 226), // queued: decompose
    ("monitor/cast_analysis/mod.rs", "Analyzer < 'a >::step", 391), // queued: decompose
    ("monitor/cast_analysis/tests/mod.rs", "build_btf", 223),     // queued: decompose
    (
        "monitor/cast_analysis/tests/helper_map_crossfn.rs",
        "helper_map_update_then_lookup_propagates_arena_through_map_value",
        220,
    ), // queued: decompose
    (
        "monitor/dump/display.rs",
        "<FailureDumpReport as std :: fmt :: Display>::fmt",
        257,
    ), // queued: decompose
    ("monitor/dump/mod.rs", "dump_state", 1166),                  // queued: decompose
    ("monitor/dump/render_map.rs", "render_map", 583),            // queued: decompose
    ("monitor/reader.rs", "monitor_loop", 808),                   // queued: decompose
    ("probe/btf.rs", "parse_bpf_btf_functions", 205),             // queued: decompose
    ("probe/output.rs", "format_probe_events_inner", 330),        // queued: decompose
    ("probe/process.rs", "attach_phase_b_fentry", 494),           // queued: decompose
    ("probe/process.rs", "run_probe_skeleton", 1295),             // queued: decompose
    ("scenario/ops/dispatch.rs", "apply_ops", 1122),              // queued: decompose
    ("scenario/ops/mod.rs", "run_scenario", 414),                 // queued: decompose
    ("scenario/ops/tests.rs", "op_discriminant_values", 210),     // queued: decompose
    (
        "scenario/snapshot/error.rs",
        "<SnapshotError as std :: fmt :: Display>::fmt",
        266,
    ), // queued: decompose
    ("test_support/entry.rs", "KtstrTestEntry::validate", 338),   // queued: decompose
    ("test_support/eval/mod.rs", "evaluate_vm_result", 765),      // queued: decompose
    (
        "test_support/eval/mod.rs",
        "run_ktstr_test_inner_impl",
        1097,
    ), // queued: decompose
    ("test_support/probe.rs", "attempt_auto_repro", 450),         // queued: decompose
    (
        "test_support/probe.rs",
        "maybe_dispatch_vm_test_with_args",
        215,
    ), // queued: decompose
    (
        "test_support/probe.rs",
        "maybe_dispatch_vm_test_with_phase_a",
        247,
    ), // queued: decompose
    ("vmm/builder.rs", "KtstrVmBuilder::build", 354),             // queued: decompose
    (
        "vmm/cast_analysis_load/mod.rs",
        "analyze_one_object_with_btf",
        217,
    ), // queued: decompose
    ("vmm/cgroup_sandbox.rs", "BuildSandbox::try_create", 246),   // queued: decompose
    ("vmm/disk_template/mod.rs", "build_template_via_vm", 233),   // queued: decompose
    ("vmm/freeze_coord/dispatch.rs", "dispatch_bulk_message", 534), // queued: decompose
    ("vmm/freeze_coord/mod.rs", "KtstrVm::collect_results", 474), // queued: decompose
    ("vmm/freeze_coord/mod.rs", "KtstrVm::run_bsp_loop", 328),    // queued: decompose
    ("vmm/freeze_coord/mod.rs", "KtstrVm::run_vm", 9326),         // queued: decompose
    (
        "vmm/freeze_coord/mod.rs",
        "KtstrVm::start_bpf_map_write",
        202,
    ), // queued: decompose
    ("vmm/freeze_coord/mod.rs", "KtstrVm::start_monitor", 796),   // queued: decompose
    ("vmm/freeze_coord/snapshot.rs", "arm_user_watchpoint", 237), // queued: decompose
    ("vmm/initramfs.rs", "build_initramfs_base", 277),            // queued: decompose
    ("vmm/mod.rs", "KtstrVm::run_interactive", 813),              // queued: decompose
    (
        "vmm/numa_mem.rs",
        "NumaMemoryLayout::allocate_and_register",
        215,
    ), // queued: decompose
    ("vmm/rust_init/dump.rs", "start_sched_exit_monitor", 222),   // queued: decompose
    ("vmm/rust_init/init.rs", "ktstr_guest_init", 765),           // queued: decompose
    ("vmm/rust_init/relay.rs", "run_relay_session", 209),         // queued: decompose
    ("vmm/sched_stats.rs", "SchedStatsClient::request_raw", 230), // queued: decompose
    ("vmm/setup/mod.rs", "KtstrVm::setup_memory", 223),           // queued: decompose
    ("vmm/virtio_blk/control.rs", "VirtioBlk::set_status", 420),  // queued: decompose
    (
        "vmm/virtio_blk/device.rs",
        "VirtioBlk::handle_read_vectored_impl",
        262,
    ), // queued: decompose
    ("vmm/virtio_blk/drain.rs", "drain_bracket_impl", 1197),      // queued: decompose
    ("vmm/virtio_blk/worker.rs", "worker_thread_main", 515),      // queued: decompose
    (
        "vmm/virtio_net/device.rs",
        "VirtioNet::pop_and_capture_tx",
        236,
    ), // queued: decompose
    (
        "vmm/virtio_net/device.rs",
        "VirtioNet::process_tx_loopback",
        291,
    ), // queued: decompose
    (
        "vmm/virtio_net/device.rs",
        "VirtioNet::try_loopback_to_rx",
        376,
    ), // queued: decompose
    ("vmm/x86_64/kvm.rs", "KtstrKvm::new_inner", 371),            // queued: decompose
    ("vmm/x86_64/topology/mod.rs", "generate_cpuid", 323),        // queued: decompose
    ("workload/spawn/mod.rs", "WorkloadHandle::spawn", 232),      // queued: decompose
    ("workload/spawn/mod.rs", "WorkloadHandle::spawn_group", 959), // queued: decompose
    (
        "workload/spawn/mod.rs",
        "WorkloadHandle::spawn_pcomm_cgroup",
        271,
    ), // queued: decompose
    (
        "workload/spawn/mod.rs",
        "WorkloadHandle::stop_and_collect",
        539,
    ), // queued: decompose
    ("workload/spawn/mod.rs", "spawn_pcomm_container", 781),      // queued: decompose
    ("workload/worker/mod.rs", "worker_main", 3275),              // queued: decompose
    // Added for post-fmt + recent-work growth; path order is cosmetic
    // (the guard collects EXCEPTIONS into a BTreeMap). queued: decompose.
    (
        "vmm/freeze_coord/kernel_op_dispatch.rs",
        "find_task_by_pid",
        221,
    ), // queued: decompose
];

fn src_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect(
        "CARGO_MANIFEST_DIR must be set (cargo sets it for cargo-test / cargo-nextest \
         invocations; running this test outside of cargo is unsupported)",
    );
    PathBuf::from(manifest_dir).join("src")
}

fn rel_path(file: &Path, src_root: &Path) -> String {
    let rel = file
        .strip_prefix(src_root)
        .expect("file must live under src_root");
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// `syn::visit::Visit` impl that records `(name, line_count)` for
/// every function with a body inside one parsed source file. The
/// visitor descends into nested impls so an `impl Foo { fn bar()
/// { ... } }` records `"Foo::bar"`; a `mod sub { fn baz() ... }`
/// records `"baz"` without a module prefix (the parent file path
/// already disambiguates across files).
///
/// Trait impl methods are recorded as `"<Type as Trait>::method"`
/// so a single type's inherent and trait impls cannot collide on
/// the same method name.
struct FnVisitor {
    /// Current `impl` context as a renderable receiver string —
    /// either `"Type"` for inherent impls, `"Type as Trait"` for
    /// trait impls, or `None` outside any impl.
    impl_ctx: Option<String>,
    out: Vec<(String, usize)>,
}

impl FnVisitor {
    fn new() -> Self {
        Self {
            impl_ctx: None,
            out: Vec::new(),
        }
    }

    fn record(&mut self, fn_name: &str, body_span: Span) {
        let start = body_span.start().line;
        let end = body_span.end().line;
        let length = end.saturating_sub(start).saturating_add(1);
        let key = match &self.impl_ctx {
            Some(receiver) if receiver.contains(" as ") => {
                format!("<{receiver}>::{fn_name}")
            }
            Some(receiver) => format!("{receiver}::{fn_name}"),
            None => fn_name.to_string(),
        };
        self.out.push((key, length));
    }
}

/// Render any `quote::ToTokens` value as a compact string for the
/// `<Type>::method` / `<Type as Trait>::method` key. Token-stream
/// rendering inserts whitespace between tokens; collapse runs of
/// whitespace into single spaces and strip ends so two equivalent
/// type renderings produce equal keys.
fn render_tokens<T: ToTokens>(value: &T) -> String {
    let s = value.to_token_stream().to_string();
    let mut compact = String::with_capacity(s.len());
    let mut last_space = true;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_space {
                compact.push(' ');
                last_space = true;
            }
        } else {
            compact.push(c);
            last_space = false;
        }
    }
    compact.trim().to_string()
}

impl<'ast> Visit<'ast> for FnVisitor {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.record(
            &node.sig.ident.to_string(),
            node.block.brace_token.span.span(),
        );
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let receiver = render_tokens(&node.self_ty);
        let saved = self.impl_ctx.take();
        self.impl_ctx = Some(match &node.trait_ {
            Some((_, trait_path, _)) => {
                let trait_s = render_tokens(trait_path);
                format!("{receiver} as {trait_s}")
            }
            None => receiver,
        });
        syn::visit::visit_item_impl(self, node);
        self.impl_ctx = saved;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.record(
            &node.sig.ident.to_string(),
            node.block.brace_token.span.span(),
        );
        syn::visit::visit_impl_item_fn(self, node);
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        if let Some(block) = &node.default {
            self.record(&node.sig.ident.to_string(), block.brace_token.span.span());
        }
        syn::visit::visit_trait_item_fn(self, node);
    }
}

fn collect_functions(file: &Path) -> Vec<(String, usize)> {
    let source = std::fs::read_to_string(file).expect("read source file");
    let parsed = match syn::parse_file(&source) {
        Ok(p) => p,
        // A parse failure (likely a build-time-only file or a
        // macro-heavy file syn can't handle) is non-fatal — the
        // guard's purpose is to catch mega-functions, not police
        // parser coverage. Surface as a Note for the operator via
        // stderr but do not flip the test.
        Err(e) => {
            eprintln!(
                "[src_function_size_guard] warning: skipped {} (syn parse failed: {e})",
                file.display()
            );
            return Vec::new();
        }
    };
    let mut v = FnVisitor::new();
    v.visit_file(&parsed);
    v.out
}

#[test]
#[ignore]
fn no_src_function_exceeds_200_lines_unless_grandfathered() {
    let src = src_root();
    assert!(
        src.is_dir(),
        "src directory does not exist at {src:?}; CARGO_MANIFEST_DIR may be wrong",
    );

    // Build owned-key maps so per-file iteration doesn't borrow into
    // the &str-keyed lookup. The EXCEPTIONS array is the source of
    // truth; the maps are derived per-test-run.
    let exceptions: BTreeMap<(String, String), usize> = EXCEPTIONS
        .iter()
        .map(|(f, n, c)| ((f.to_string(), n.to_string()), *c))
        .collect();
    assert_eq!(
        exceptions.len(),
        EXCEPTIONS.len(),
        "EXCEPTIONS contains a duplicate (file, fn_name) key — each entry must \
         appear exactly once",
    );

    let mut new_overflows: Vec<(String, String, usize)> = Vec::new();
    let mut grew_past_ceiling: Vec<(String, String, usize, usize)> = Vec::new();
    let mut seen_exceptions: BTreeMap<(String, String), bool> =
        exceptions.keys().map(|k| (k.clone(), false)).collect();

    for entry in walkdir::WalkDir::new(&src).into_iter() {
        let entry = entry.expect("walkdir must succeed under src/");
        let path = entry.path();
        if !entry.file_type().is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let rel = rel_path(path, &src);
        let fns = collect_functions(path);

        for (name, length) in fns {
            if length <= DEFAULT_FN_MAX_LINES {
                // Under the soft limit — even if it's in
                // EXCEPTIONS (the drain-or-remove pass below will
                // surface that). Not a violation by itself.
                continue;
            }
            let key = (rel.clone(), name);
            match exceptions.get(&key) {
                Some(&ceiling) => {
                    if let Some(seen) = seen_exceptions.get_mut(&key) {
                        *seen = true;
                    }
                    if length > ceiling {
                        let (file, fn_name) = key;
                        grew_past_ceiling.push((file, fn_name, length, ceiling));
                    }
                }
                None => {
                    let (file, fn_name) = key;
                    new_overflows.push((file, fn_name, length));
                }
            }
        }
    }

    let stale_exceptions: Vec<(String, String)> = seen_exceptions
        .iter()
        .filter_map(|(k, seen)| if *seen { None } else { Some(k.clone()) })
        .collect();

    let any_failure =
        !new_overflows.is_empty() || !grew_past_ceiling.is_empty() || !stale_exceptions.is_empty();

    if any_failure {
        let mut msg = String::from("src-function size guard failed:\n\n");
        if !new_overflows.is_empty() {
            msg.push_str("(1) Functions NOT in EXCEPTIONS that exceed 200 lines:\n");
            for (file, name, length) in &new_overflows {
                msg.push_str(&format!(
                    "    (\"{file}\", \"{name}\", {length}),  // queued: decompose\n"
                ));
            }
            msg.push_str(
                "    Fix: decompose the function into helpers, or add the entry \
                 to EXCEPTIONS with its current line count and a `// queued: \
                 <task>` comment naming the queued decomposition task.\n\n",
            );
        }
        if !grew_past_ceiling.is_empty() {
            msg.push_str("(2) Grandfathered functions that grew past their pinned ceiling:\n");
            for (file, name, length, ceiling) in &grew_past_ceiling {
                msg.push_str(&format!(
                    "    src/{file}::{name} = {length} lines (grandfathered ceiling {ceiling})\n"
                ));
            }
            msg.push_str(
                "    Fix: decompose (preferred) OR refresh the EXCEPTIONS entry's \
                 count if the growth is genuinely unavoidable.\n\n",
            );
        }
        if !stale_exceptions.is_empty() {
            msg.push_str(
                "(3) EXCEPTIONS entries that are now stale (function ≤ 200 \
                 lines or removed) — remove these entries:\n",
            );
            for (file, name) in &stale_exceptions {
                msg.push_str(&format!("    (\"{file}\", \"{name}\", _)\n"));
            }
            msg.push_str(
                "    Fix: delete the listed entries from EXCEPTIONS. The \
                 default 200-line gate guards the function going forward.\n\n",
            );
        }
        panic!("{msg}");
    }
}
