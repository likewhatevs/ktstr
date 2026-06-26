//! Codegen for `#[ktstr_test]`: `emit_entry_static` builds the
//! linkme-registered static entry + `#[test]` wrapper from the parsed
//! `AttrValues`. Split out of the parent module to keep both files
//! under the size ceiling; reaches the parent's parsers and helpers via
//! `use super::*`.

use quote::{format_ident, quote};
use syn::ItemFn;

use super::*;

/// Emit the linkme-registered static entry + #[test] wrapper for a
/// `#[ktstr_test]`-annotated function. Takes the parsed function and
/// the parsed-and-validated [`AttrValues`]; returns the token stream
/// the macro expands to. The destructure of `attrs` lives inside this
/// function because the codegen below interpolates each field via
/// `quote!{ #foo }`, which requires a single binding in scope — see
/// the docstring on [`AttrValues`] for the full rationale.
pub(super) fn emit_entry_static(input: ItemFn, attrs: AttrValues) -> proc_macro2::TokenStream {
    let orig_name = &input.sig.ident;
    let inner_name = format_ident!("__ktstr_inner_{}", orig_name);
    let entry_name = format_ident!("__KTSTR_ENTRY_{}", orig_name.to_string().to_uppercase());
    let name_str = orig_name.to_string();

    // Destructure attrs into per-field bare locals. The codegen
    // region below interpolates each field via `quote!{ #foo }`,
    // which requires `foo` to be a single binding in scope. The
    // destructure is the cleanest way to expose every field at once
    // without scattering `let foo = attrs.foo;` lines across the
    // codegen — see the docstring on [`AttrValues`] for the full
    // rationale.
    let AttrValues {
        llcs,
        cores,
        threads,
        numa_nodes,
        llcs_set,
        cores_set,
        threads_set,
        numa_nodes_set,
        memory_mib,
        memory_mib_set,
        duration_s,
        duration_s_set,
        cleanup_budget_ms,
        watchdog_timeout_s,
        watchdog_timeout_s_set,
        scheduler,
        payload,
        workloads,
        staged_schedulers,
        bpf_map_write,
        post_vm,
        post_vm_unconditional,
        disk,
        network,
        not_starved,
        isolation,
        max_gap_ms,
        max_spread_pct,
        max_imbalance_ratio,
        max_local_dsq_depth,
        fail_on_stall,
        sustained_samples,
        max_throughput_cv,
        min_work_rate,
        max_fallback_rate,
        max_keep_last_rate,
        max_p99_wake_latency_ns,
        max_wake_latency_cv,
        min_iteration_rate,
        max_migration_ratio,
        min_page_locality,
        max_cross_node_migration_ratio,
        max_slow_tier_ratio,
        min_numa_nodes,
        min_numa_nodes_set,
        min_llcs,
        min_llcs_set,
        requires_smt,
        requires_smt_set,
        min_cpus,
        min_cpus_set,
        max_llcs,
        max_llcs_set,
        max_numa_nodes,
        max_numa_nodes_set,
        max_cpus,
        max_cpus_set,
        cpu_budget,
        auto_repro,
        auto_repro_set,
        expect_auto_repro,
        expect_auto_repro_set,
        performance_mode,
        performance_mode_set,
        no_perf_mode,
        no_perf_mode_set,
        expect_err,
        expect_err_set,
        survives_storm,
        survives_storm_set,
        allow_inconclusive,
        allow_inconclusive_set,
        host_only,
        host_only_set,
        ignore_test,
        kaslr,
        kaslr_set,
        wprof,
        wprof_set,
        num_snapshots,
        num_snapshots_set,
        extra_sched_args,
        extra_include_files,
        workload_root_cgroup,
        wprof_args,
        expect_scx_bpf_error_contains_tokens,
        expect_scx_bpf_error_matches_tokens,
        config_expr,
        config_set,
    } = attrs;

    // Build the scheduler reference token. The `scheduler` slot on
    // `KtstrTestEntry` is `&'static Scheduler`; callers pass either a
    // `NAME` const emitted by `declare_scheduler!` or `Scheduler::EEVDF`
    // directly. The default is the kernel-default EEVDF placeholder.
    let scheduler_tokens = match &scheduler {
        Some(p) => {
            quote! { &#p }
        }
        None => {
            quote! { &::ktstr::test_support::Scheduler::EEVDF }
        }
    };

    // Build topology tokens. Each dimension independently inherits from
    // the scheduler's topology when not explicitly set. `Scheduler.topology`
    // is a direct field (a `Topology` struct), so the field-of-field
    // access below remains valid inside a `const` initializer.
    let llcs_tokens = inherited_constraint_tokens(
        llcs_set,
        &scheduler,
        quote! { topology.llcs },
        || quote! { #llcs },
    );
    let cores_tokens = inherited_constraint_tokens(
        cores_set,
        &scheduler,
        quote! { topology.cores_per_llc },
        || quote! { #cores },
    );
    let threads_tokens = inherited_constraint_tokens(
        threads_set,
        &scheduler,
        quote! { topology.threads_per_core },
        || quote! { #threads },
    );
    let numa_nodes_tokens = inherited_constraint_tokens(
        numa_nodes_set,
        &scheduler,
        quote! { topology.numa_nodes },
        || quote! { #numa_nodes },
    );
    let topology_tokens = quote! {
        ::ktstr::test_support::Topology {
            llcs: #llcs_tokens,
            cores_per_llc: #cores_tokens,
            threads_per_core: #threads_tokens,
            numa_nodes: #numa_nodes_tokens,
            nodes: None,
            distances: None,
        }
    };

    // Build the renamed inner function
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;
    let attrs = &input.attrs;
    let inner_sig = syn::Signature {
        ident: inner_name.clone(),
        ..sig.clone()
    };

    // Build Assert field tokens.
    let not_starved_tokens = option_tokens(&not_starved);
    let isolation_tokens = option_tokens(&isolation);
    let gap_tokens = option_tokens(&max_gap_ms);
    let spread_tokens = option_tokens(&max_spread_pct);
    let imbalance_tokens = option_tokens(&max_imbalance_ratio);
    let dsq_tokens = option_tokens(&max_local_dsq_depth);
    let stall_tokens = option_tokens(&fail_on_stall);
    let sustained_tokens = option_tokens(&sustained_samples);
    let throughput_cv_tokens = option_tokens(&max_throughput_cv);
    let work_rate_tokens = option_tokens(&min_work_rate);
    let fallback_rate_tokens = option_tokens(&max_fallback_rate);
    let keep_last_rate_tokens = option_tokens(&max_keep_last_rate);
    let p99_wake_tokens = option_tokens(&max_p99_wake_latency_ns);
    let wake_cv_tokens = option_tokens(&max_wake_latency_cv);
    let iter_rate_tokens = option_tokens(&min_iteration_rate);
    let mig_ratio_tokens = option_tokens(&max_migration_ratio);
    let page_locality_tokens = option_tokens(&min_page_locality);
    let cross_node_mig_tokens = option_tokens(&max_cross_node_migration_ratio);
    let slow_tier_tokens = option_tokens(&max_slow_tier_ratio);

    // `cleanup_budget_ms` lives on the macro side as `Option<u64>` of
    // milliseconds; the entry field is `Option<Duration>`, so wrap
    // the literal in `Duration::from_millis(...)` at emission time.
    let cleanup_budget_tokens = match cleanup_budget_ms {
        Some(ms) => {
            quote! { ::core::option::Option::Some(::std::time::Duration::from_millis(#ms)) }
        }
        None => quote! { ::core::option::Option::None },
    };

    let bpf_map_write_tokens = match &bpf_map_write {
        Some(p) => quote! { &[&#p] },
        None => quote! { &[] },
    };

    // Emit `Option<&'static Payload>` for the primary payload. The
    // user supplies a path (`&FIO` equivalent in source), so we
    // wrap it in `Some(&#p)` at emission time to preserve the
    // `entry.payload: Option<&'static Payload>` field type.
    let payload_tokens = match &payload {
        Some(p) => quote! { Some(&#p) },
        None => quote! { None },
    };
    // Emit `&'static [&'static Payload]` for workloads. Each path
    // the user supplied is a `const Payload`; we take `&` on each
    // to match the stored type.
    let workloads_slice: &[syn::Path] = workloads.as_deref().unwrap_or(&[]);
    let workload_refs: Vec<proc_macro2::TokenStream> =
        workloads_slice.iter().map(|p| quote! { &#p }).collect();
    let workloads_tokens = quote! { &[#(#workload_refs),*] };

    // Emit `&'static [&'static Scheduler]` for staged_schedulers.
    // Each user-supplied path is a `const Scheduler`; take `&` on
    // each to match the stored type. Empty slice when the attribute
    // is absent (the common no-staging case) keeps existing tests
    // working without any per-test change.
    let staged_schedulers_slice: &[syn::Path] = staged_schedulers.as_deref().unwrap_or(&[]);
    let staged_sched_refs: Vec<proc_macro2::TokenStream> = staged_schedulers_slice
        .iter()
        .map(|p| quote! { &#p })
        .collect();
    let staged_schedulers_tokens = if staged_schedulers.is_some() {
        quote! { staged_schedulers: &[#(#staged_sched_refs),*], }
    } else {
        quote! {}
    };

    // Conditionally-emitted KtstrTestEntry fields. Each block is
    // either an empty TokenStream (so the field is left to
    // `..KtstrTestEntry::DEFAULT` in the spread) or a `field: VAL,`
    // pair when the macro must override the default. This pattern
    // means new struct fields with sane defaults need no macro
    // change — adding to KtstrTestEntry::DEFAULT alone is enough.
    let memory_mib_field = entry_field(
        memory_mib_set,
        quote! { memory_mib },
        quote! { #memory_mib },
    );
    // cpu_budget is Option<u32>: emit `cpu_budget: Some(N),` only when the
    // attr was supplied, else fall through to KtstrTestEntry::DEFAULT (None).
    // entry_field can't express this — it wraps the value verbatim, not in
    // Some(_) — so use the wprof_args Some/None match pattern.
    let cpu_budget_field = match cpu_budget {
        Some(n) => quote! { cpu_budget: Some(#n), },
        None => quote! {},
    };
    let payload_field = entry_field(
        payload.is_some(),
        quote! { payload },
        quote! { #payload_tokens },
    );
    let workloads_field = entry_field(
        workloads.is_some(),
        quote! { workloads },
        quote! { #workloads_tokens },
    );
    let auto_repro_field = entry_field(
        auto_repro_set,
        quote! { auto_repro },
        quote! { #auto_repro },
    );
    let expect_auto_repro_field = entry_field(
        expect_auto_repro_set,
        quote! { expect_auto_repro },
        quote! { #expect_auto_repro },
    );
    let kaslr_field = entry_field(kaslr_set, quote! { kaslr }, quote! { #kaslr });
    let wprof_field = entry_field(wprof_set, quote! { wprof }, quote! { #wprof });
    let wprof_args_field = match wprof_args {
        Some(ref tokens) => quote! { wprof_args: Some(#tokens), },
        None => quote! {},
    };
    // Any of the per-check assert fields supplied by the attribute
    // forces emission of the full `assert: Assert { .. }` block. When
    // none are set the spread inherits `Assert::NO_OVERRIDES` from
    // `KtstrTestEntry::DEFAULT`, which is bit-for-bit identical to
    // the all-`None` Assert the prior unconditional emission produced.
    let any_assert_set = not_starved.is_some()
        || isolation.is_some()
        || max_gap_ms.is_some()
        || max_spread_pct.is_some()
        || max_throughput_cv.is_some()
        || min_work_rate.is_some()
        || max_p99_wake_latency_ns.is_some()
        || max_wake_latency_cv.is_some()
        || min_iteration_rate.is_some()
        || max_migration_ratio.is_some()
        || max_imbalance_ratio.is_some()
        || max_local_dsq_depth.is_some()
        || fail_on_stall.is_some()
        || sustained_samples.is_some()
        || max_fallback_rate.is_some()
        || max_keep_last_rate.is_some()
        || min_page_locality.is_some()
        || max_cross_node_migration_ratio.is_some()
        || max_slow_tier_ratio.is_some()
        || expect_scx_bpf_error_contains_tokens.is_some()
        || expect_scx_bpf_error_matches_tokens.is_some();
    let expect_scx_bpf_error_contains_field = option_tokens(&expect_scx_bpf_error_contains_tokens);
    let expect_scx_bpf_error_matches_field = option_tokens(&expect_scx_bpf_error_matches_tokens);
    let assert_field = if any_assert_set {
        quote! {
            assert: ::ktstr::assert::Assert {
                not_starved: #not_starved_tokens,
                isolation: #isolation_tokens,
                max_gap_ms: #gap_tokens,
                max_spread_pct: #spread_tokens,
                max_throughput_cv: #throughput_cv_tokens,
                min_work_rate: #work_rate_tokens,
                max_p99_wake_latency_ns: #p99_wake_tokens,
                max_wake_latency_cv: #wake_cv_tokens,
                min_iteration_rate: #iter_rate_tokens,
                max_migration_ratio: #mig_ratio_tokens,
                max_imbalance_ratio: #imbalance_tokens,
                max_local_dsq_depth: #dsq_tokens,
                fail_on_stall: #stall_tokens,
                sustained_samples: #sustained_tokens,
                max_fallback_rate: #fallback_rate_tokens,
                max_keep_last_rate: #keep_last_rate_tokens,
                enforce_monitor_thresholds: false,
                min_page_locality: #page_locality_tokens,
                max_cross_node_migration_ratio: #cross_node_mig_tokens,
                max_slow_tier_ratio: #slow_tier_tokens,
                expect_scx_bpf_error_contains: #expect_scx_bpf_error_contains_field,
                expect_scx_bpf_error_matches: #expect_scx_bpf_error_matches_field,
            },
        }
    } else {
        quote! {}
    };
    let extra_sched_args_field = entry_field(
        !extra_sched_args.is_empty(),
        quote! { extra_sched_args },
        quote! { &[#(#extra_sched_args),*] },
    );
    let watchdog_timeout_field = entry_field(
        watchdog_timeout_s_set,
        quote! { watchdog_timeout },
        quote! { ::std::time::Duration::from_secs(#watchdog_timeout_s) },
    );
    let bpf_map_write_field = entry_field(
        bpf_map_write.is_some(),
        quote! { bpf_map_write },
        quote! { #bpf_map_write_tokens },
    );
    let performance_mode_field = entry_field(
        performance_mode_set,
        quote! { performance_mode },
        quote! { #performance_mode },
    );
    let no_perf_mode_field = entry_field(
        no_perf_mode_set,
        quote! { no_perf_mode },
        quote! { #no_perf_mode },
    );
    let duration_field = entry_field(
        duration_s_set,
        quote! { duration },
        quote! { ::std::time::Duration::from_secs(#duration_s) },
    );
    let num_snapshots_field = entry_field(
        num_snapshots_set,
        quote! { num_snapshots },
        quote! { #num_snapshots },
    );
    let expect_err_field = entry_field(
        expect_err_set,
        quote! { expect_err },
        quote! { #expect_err },
    );
    let survives_storm_field = entry_field(
        survives_storm_set,
        quote! { survives_storm },
        quote! { #survives_storm },
    );
    let allow_inconclusive_field = entry_field(
        allow_inconclusive_set,
        quote! { allow_inconclusive },
        quote! { #allow_inconclusive },
    );
    let host_only_field = entry_field(host_only_set, quote! { host_only }, quote! { #host_only });
    let ignore_attr = if ignore_test {
        quote! { #[ignore] }
    } else {
        quote! {}
    };
    let extra_include_files_field = entry_field(
        !extra_include_files.is_empty(),
        quote! { extra_include_files },
        quote! { &[#(#extra_include_files),*] },
    );
    let cleanup_budget_field = entry_field(
        cleanup_budget_ms.is_some(),
        quote! { cleanup_budget },
        quote! { #cleanup_budget_tokens },
    );
    // The user-supplied path resolves to a `fn(&VmResult) ->
    // Result<()>`. Wrap in `Some(...)` so the entry's
    // `Option<fn(&VmResult) -> Result<()>>` field accepts it.
    //
    // When the attribute omits `post_vm = ...`, default to
    // `ktstr::test_support::default_post_vm_periodic_fired` — the
    // smoke-floor assertion that at least one periodic snapshot
    // fired when periodic was configured (no-op when periodic was
    // disabled). Saves every periodic-configured test from
    // hand-rolling the "did the scheduler attach + snapshot fire"
    // boilerplate.
    let post_vm_field = if let Some(ref p) = post_vm {
        quote! { post_vm: Some(#p), }
    } else {
        quote! { post_vm: Some(::ktstr::test_support::default_post_vm_periodic_fired), }
    };
    // `post_vm_unconditional` has no default — when the attribute
    // omits it, the field stays `None` (matching `KtstrTestEntry::DEFAULT`)
    // and the unconditional dispatch arm at
    // `src/test_support/eval.rs` is a no-op for that entry.
    let post_vm_unconditional_field =
        some_wrapped_entry_field(&post_vm_unconditional, quote! { post_vm_unconditional });
    // `config = EXPR` lands in `KtstrTestEntry::config_content`, which
    // is `Option<&'static str>`. Wrap the user-supplied expression in
    // `Some(...)` at emission so the spread site sees a typed Option.
    let config_content_field = some_wrapped_entry_field(&config_expr, quote! { config_content });
    // `disk = PATH` resolves to a `const DiskConfig`. The entry's
    // `disk` field is `Option<DiskConfig>` (owned, but const-
    // constructible — see DiskConfig::DEFAULT). Wrap in `Some(...)`
    // at emission. The struct is `Clone` so spreading a const ref
    // into a `static` initializer works.
    let disk_field = some_wrapped_entry_field(&disk, quote! { disk });
    let network_field = some_wrapped_entry_field(&network, quote! { network });

    // `workload_root_cgroup = "/path"` lands in
    // `KtstrTestEntry::workload_root_cgroup` as
    // `Some(CgroupPath::new("/path"))`. `CgroupPath::new` is `const fn`
    // with a compile-time gate (rejects missing leading `/`, bare `/`,
    // `..` components), so a malformed path panics at const-eval
    // rather than at boot.
    let workload_root_cgroup_field = if let Some(ref path) = workload_root_cgroup {
        quote! {
            workload_root_cgroup: ::core::option::Option::Some(
                ::ktstr::test_support::CgroupPath::new(#path),
            ),
        }
    } else {
        quote! {}
    };

    // Compile-time assert: `config = ...` must be paired with a
    // scheduler that declares `config_file_def`, and vice versa. The
    // macro can't read the scheduler const's value (it sees only a
    // path), but both `Payload::config_file_def` and `Option::is_some`
    // are `const fn`, so a `const _: () = assert!(...)` block can
    // verify the pairing at compile time. The `KtstrTestEntry::validate`
    // method enforces the same gate at runtime so direct programmatic
    // construction doesn't bypass the macro path.
    let config_set_lit = config_set;
    let pairing_assert_const_name = format_ident!(
        "__KTSTR_CONFIG_PAIRING_{}",
        orig_name.to_string().to_uppercase()
    );
    let pairing_assert = quote! {
        const #pairing_assert_const_name: () = {
            let has_def = (#scheduler_tokens).config_file_def.is_some();
            let has_content: bool = #config_set_lit;
            if has_def && !has_content {
                panic!(
                    "scheduler declares `config_file_def` but the test \
                     does not supply `config = ...`; provide an inline \
                     scheduler config or remove `config_file_def` from \
                     the scheduler definition"
                );
            }
            if !has_def && has_content {
                panic!(
                    "test supplies `config = ...` but the scheduler does \
                     not declare `config_file_def`; remove `config = ...` \
                     or add `config_file_def(arg_template, guest_path)` \
                     to the scheduler definition"
                );
            }
        };
    };

    // Host-error handling shared by the expect_err and expect_ok test
    // bodies. All SIX host-insufficiency types — KernelUnavailable plus the
    // five host_topology types — route through the shared
    // `test_support::classify_host_error`, the single source of truth for
    // the guard order + per-class skip/fail policy, shared with
    // `err_to_exit_code` so the two sites cannot drift. (KernelUnavailable
    // is "harness not configured" — no kernel resolved; classify_host_error
    // makes it a SKIP by default, a FAIL under KTSTR_NO_SKIP_MODE, same as
    // the dispatch path.) `classify_host_error` returns a verdict; the
    // libtest control flow (eprintln + return for a skip, panic for a fail)
    // lives HERE, in the generated test fn body, so a fail panics in the
    // test frame. The bare `reason` gets its `ktstr: SKIP:` / `ktstr: FAIL:`
    // prefix at the emit site, matching the dispatch channels. The trailing
    // NotHostClass behavior differs per direction (expect_err swallows a
    // non-host failure; expect_ok panics with it), so it is interpolated.
    let host_arms = |not_host_class: proc_macro2::TokenStream| {
        quote! {
            Err(e) => match ::ktstr::test_support::classify_host_error(
                &e,
                ::std::env::var_os("KTSTR_NO_SKIP_MODE").is_some(),
            ) {
                ::ktstr::test_support::HostClass::Skip { reason } => {
                    eprintln!("ktstr: SKIP: {reason}");
                    return;
                }
                ::ktstr::test_support::HostClass::Fail { reason } => {
                    panic!("ktstr: FAIL: {reason}");
                }
                ::ktstr::test_support::HostClass::NotHostClass => #not_host_class,
            },
        }
    };
    let test_body = if expect_err {
        // NotHostClass under expect_err: a non-host failure IS the
        // expected failure, so swallow it. The host-class arms above win
        // over this swallow (a host-insufficiency skip/fail is never the
        // test's expected logical failure).
        let host_arms = host_arms(quote! { {} });
        quote! {
            match ::ktstr::test_support::run_ktstr_test(&#entry_name) {
                // A skip returns Ok(AssertResult::skip), NOT an Err —
                // e.g. the overcommit auto-skip, performance_mode /
                // perf_only skips, or an in-VM scenario topology-floor
                // skip (all via test_support::eval). A skipped run did
                // not actually execute, so it is a non-failure in BOTH
                // directions: an expect_err test that never ran did not
                // "pass". Route is_skip first (mirroring ok_to_exit_code's
                // is_skip -> EXIT_PASS precedence) so the skip is not
                // mistaken for the expected failure. The SKIP banner +
                // sidecar are already emitted inside run_ktstr_test_inner.
                Ok(r) if r.is_skip() => {}
                Ok(_) => panic!("expected test to fail but it passed"),
                #host_arms
            }
        }
    } else {
        // NotHostClass under expect_ok: a non-host failure is a real
        // test failure — panic with the full error chain.
        let host_arms = host_arms(quote! { panic!("{e:#}") });
        quote! {
            match ::ktstr::test_support::run_ktstr_test(&#entry_name) {
                Ok(_) => {}
                #host_arms
            }
        }
    };

    // Build constraint tokens. Each field independently inherits from
    // the scheduler's constraints when not explicitly set, following
    // the same pattern as topology inheritance.
    let min_numa_nodes_tokens = inherited_constraint_tokens(
        min_numa_nodes_set,
        &scheduler,
        quote! { constraints.min_numa_nodes },
        || quote! { #min_numa_nodes },
    );
    let max_numa_nodes_tokens = inherited_constraint_tokens(
        max_numa_nodes_set,
        &scheduler,
        quote! { constraints.max_numa_nodes },
        || option_tokens(&max_numa_nodes),
    );
    let min_llcs_tokens = inherited_constraint_tokens(
        min_llcs_set,
        &scheduler,
        quote! { constraints.min_llcs },
        || quote! { #min_llcs },
    );
    let max_llcs_tokens = inherited_constraint_tokens(
        max_llcs_set,
        &scheduler,
        quote! { constraints.max_llcs },
        || option_tokens(&max_llcs),
    );
    let requires_smt_tokens = inherited_constraint_tokens(
        requires_smt_set,
        &scheduler,
        quote! { constraints.requires_smt },
        || quote! { #requires_smt },
    );
    let min_cpus_tokens = inherited_constraint_tokens(
        min_cpus_set,
        &scheduler,
        quote! { constraints.min_cpus },
        || quote! { #min_cpus },
    );
    let max_cpus_tokens = inherited_constraint_tokens(
        max_cpus_set,
        &scheduler,
        quote! { constraints.max_cpus },
        || option_tokens(&max_cpus),
    );

    let expanded = quote! {
        #(#attrs)*
        #vis #inner_sig #block

        #[::ktstr::distributed_slice(::ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ::ktstr::linkme)]
        static #entry_name: ::ktstr::test_support::KtstrTestEntry = ::ktstr::test_support::KtstrTestEntry {
            // Always-emit fields. `name`/`func` are macro-generated;
            // `topology`/`constraints` inherit from the scheduler
            // via field access that the spread cannot
            // recover; `scheduler` substitutes
            // `Scheduler::EEVDF` when no `scheduler = ...`
            // attribute was supplied. Every remaining field below
            // (memory_mib, payload, workloads, auto_repro, assert,
            // extra_sched_args, ..., disk, and any future addition)
            // falls through to `..KtstrTestEntry::DEFAULT` when the
            // attribute did not specify a value, so future fields
            // with sane defaults need no macro change at all.
            name: #name_str,
            func: #inner_name,
            topology: #topology_tokens,
            constraints: ::ktstr::test_support::TopologyConstraints {
                min_numa_nodes: #min_numa_nodes_tokens,
                max_numa_nodes: #max_numa_nodes_tokens,
                min_llcs: #min_llcs_tokens,
                max_llcs: #max_llcs_tokens,
                requires_smt: #requires_smt_tokens,
                min_cpus: #min_cpus_tokens,
                max_cpus: #max_cpus_tokens,
            },
            scheduler: #scheduler_tokens,
            #memory_mib_field
            #cpu_budget_field
            #payload_field
            #workloads_field
            #staged_schedulers_tokens
            #auto_repro_field
            #expect_auto_repro_field
            #kaslr_field
            #wprof_field
            #wprof_args_field
            #assert_field
            #extra_sched_args_field
            #watchdog_timeout_field
            #bpf_map_write_field
            #performance_mode_field
            #no_perf_mode_field
            #duration_field
            #num_snapshots_field
            #expect_err_field
            #survives_storm_field
            #allow_inconclusive_field
            #host_only_field
            #extra_include_files_field
            #cleanup_budget_field
            #post_vm_field
            #post_vm_unconditional_field
            #config_content_field
            #disk_field
            #network_field
            #workload_root_cgroup_field
            ..::ktstr::test_support::KtstrTestEntry::DEFAULT
        };

        #pairing_assert

        #[test]
        #ignore_attr
        fn #orig_name() {
            #test_body
        }
    };

    expanded
}
