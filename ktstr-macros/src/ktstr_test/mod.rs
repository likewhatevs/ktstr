//! `#[ktstr_test]` attribute-macro implementation: parses the test
//! attribute grammar into `AttrValues`, validates cross-attribute
//! constraints, and emits the test's distributed-slice entry. The
//! `#[proc_macro_attribute] ktstr_test` entry point lives in `lib.rs`
//! (Rust requires proc-macro fns at the crate root); this module holds
//! the parse + validation + codegen it calls.

use quote::{ToTokens, quote};
use syn::{ItemFn, Meta, MetaNameValue, parse::Parser};

mod codegen;

/// Emit `Some(value)` or `None` as token streams.
pub(crate) fn option_tokens<T: ToTokens>(opt: &Option<T>) -> proc_macro2::TokenStream {
    match opt {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    }
}

/// Extract a [`syn::Path`] from an attribute value, returning a focused
/// error spanned to the offending expression when the user supplied
/// something other than a path (an int, a string, an array, …).
/// Collapses the five `"scheduler" | "payload" | "post_vm" |
/// "post_vm_unconditional" | "disk"` parse arms onto a single shape.
fn expect_path_value(value: &syn::Expr, error_hint: &str) -> Result<syn::Path, syn::Error> {
    match value {
        syn::Expr::Path(ep) => Ok(ep.path.clone()),
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Extract `Vec<syn::Path>` from an attribute value shaped like
/// `[A, B, C]`. The two diagnostic hints distinguish the array-shape
/// failure (the whole value is not an array) from the per-element
/// failure (one entry is not a path). Collapses the `"workloads" |
/// "staged_schedulers" | "networks"` parse arms.
fn expect_array_of_paths(
    value: &syn::Expr,
    array_error_hint: &str,
    elem_error_hint: &str,
) -> Result<Vec<syn::Path>, syn::Error> {
    let arr = match value {
        syn::Expr::Array(ea) => ea,
        _ => return Err(syn::Error::new_spanned(value, array_error_hint)),
    };
    let mut entries = Vec::with_capacity(arr.elems.len());
    for elem in &arr.elems {
        match elem {
            syn::Expr::Path(ep) => entries.push(ep.path.clone()),
            _ => return Err(syn::Error::new_spanned(elem, elem_error_hint)),
        }
    }
    Ok(entries)
}

/// Accept EITHER a single path (`A`) OR a `[A, B, ...]` array of paths,
/// returning `Vec<syn::Path>` — a single path yields a one-element vec.
/// Used by `bpf_map_write` and `watch_bpf_maps`, each of which may
/// declare one const (`= A`) or several (`= [A, B]`); codegen borrows
/// each entry into `&[&…]`, so the one-element vec reproduces the former
/// single-const behaviour exactly. `error_hint` covers both the wrong
/// top-level shape and a non-path array element.
fn expect_path_or_array_of_paths(
    value: &syn::Expr,
    error_hint: &str,
) -> Result<Vec<syn::Path>, syn::Error> {
    match value {
        syn::Expr::Path(ep) => Ok(vec![ep.path.clone()]),
        syn::Expr::Array(ea) => {
            let mut entries = Vec::with_capacity(ea.elems.len());
            for elem in &ea.elems {
                match elem {
                    syn::Expr::Path(ep) => entries.push(ep.path.clone()),
                    _ => return Err(syn::Error::new_spanned(elem, error_hint)),
                }
            }
            Ok(entries)
        }
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Extract the [`String`] value of a `Lit::Str` attribute and
/// return a spanned diagnostic for any other expression shape.
/// Used by attributes whose target is `Option<String>` rather than
/// the tokens-passthrough shape `expect_string_or_path_tokens`
/// handles.
fn expect_string_literal_value(value: &syn::Expr, error_hint: &str) -> Result<String, syn::Error> {
    match value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(ls),
            ..
        }) => Ok(ls.value()),
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Extract a [`struct@syn::LitBool`] reference from a `Lit::Bool` value,
/// returning a spanned diagnostic for any other expression shape.
fn expect_bool_literal<'a>(
    value: &'a syn::Expr,
    error_hint: &str,
) -> Result<&'a syn::LitBool, syn::Error> {
    match value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Bool(lb),
            ..
        }) => Ok(lb),
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Accept a string literal (`"..."`) OR a path to a
/// `const &'static str` (`MY_CONST`) and emit the token stream that
/// resolves to the value at expansion. Used by every attribute whose
/// field is `Option<&'static str>` (`expect_scx_bpf_error_contains`,
/// `expect_scx_bpf_error_matches`, `config`). Any other expression
/// shape would fail to borrow as `'static` or coerce — reject early
/// with a targeted error rather than letting rustc surface a confusing
/// borrow / type-mismatch at the spread site.
fn expect_string_or_path_tokens(
    value: &syn::Expr,
    error_hint: &str,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    match value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(_),
            ..
        }) => Ok(quote! { #value }),
        syn::Expr::Path(_) => Ok(quote! { #value }),
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Extract a [`struct@syn::LitInt`] from an attribute value, returning a
/// spanned diagnostic when the user supplied something other than
/// an integer literal. Hint is reused as the operator-facing message.
fn expect_int_literal<'a>(
    value: &'a syn::Expr,
    error_hint: &str,
) -> Result<&'a syn::LitInt, syn::Error> {
    match value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(li),
            ..
        }) => Ok(li),
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Extract a [`struct@syn::LitFloat`] from an attribute value, returning a
/// spanned diagnostic when the user supplied something other than
/// a float literal.
fn expect_float_literal<'a>(
    value: &'a syn::Expr,
    error_hint: &str,
) -> Result<&'a syn::LitFloat, syn::Error> {
    match value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Float(lf),
            ..
        }) => Ok(lf),
        _ => Err(syn::Error::new_spanned(value, error_hint)),
    }
}

/// Extract `Vec<String>` from an attribute value shaped like
/// `["a", "b", "c"]`. Collapses the `"extra_sched_args" |
/// "extra_include_files"` parse arms (and any future
/// `["literal", "literal"]` attribute).
fn expect_array_of_string_literals(
    value: &syn::Expr,
    array_error_hint: &str,
    elem_error_hint: &str,
) -> Result<Vec<String>, syn::Error> {
    let arr = match value {
        syn::Expr::Array(ea) => ea,
        _ => return Err(syn::Error::new_spanned(value, array_error_hint)),
    };
    let mut entries = Vec::with_capacity(arr.elems.len());
    for elem in &arr.elems {
        match elem {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(ls),
                ..
            }) => entries.push(ls.value()),
            _ => return Err(syn::Error::new_spanned(elem, elem_error_hint)),
        }
    }
    Ok(entries)
}

/// Emit `name: ::core::option::Option::Some(value),` when `value` is
/// `Some`, otherwise an empty TokenStream so the spread of
/// `KtstrTestEntry::DEFAULT` supplies the `None` instead. Used by
/// the optional-Some-wrapped attribute fields whose value comes from
/// the attribute parser as an `Option<T: ToTokens>` (`post_vm_unconditional`,
/// `config_content`, `disk`). Distinct from [`entry_field`], which
/// emits the value verbatim and so cannot wrap `None` into `Some()`
/// without producing invalid Rust.
fn some_wrapped_entry_field<T: ToTokens>(
    value: &Option<T>,
    name: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match value {
        Some(v) => quote! { #name: ::core::option::Option::Some(#v), },
        None => quote! {},
    }
}

/// Emit `name: value,` when `present`, otherwise an empty TokenStream so
/// the surrounding struct literal's `..::ktstr::test_support::KtstrTestEntry::DEFAULT`
/// spread supplies the field instead. Collapses the dozen-plus
/// `if X_set { quote!{ X: #X, } } else { quote!{} }` triples that the
/// per-attribute codegen below assembles.
fn entry_field(
    present: bool,
    name: proc_macro2::TokenStream,
    value: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if present {
        quote! { #name: #value, }
    } else {
        quote! {}
    }
}

/// Collapse the constraint/topology inheritance triple
/// (`if X_set { fallback } else if scheduler { scheduler.X } else { fallback }`)
/// emitted seven times for `min_*` / `max_*` / `requires_smt` constraints.
///
/// `scheduler_path` is appended to the scheduler reference when the macro
/// inherits — e.g. `quote! { constraints.min_llcs }` produces
/// `#scheduler.constraints.min_llcs`. `fallback` runs at most once and
/// supplies the explicit-set / no-scheduler branch's tokens.
fn inherited_constraint_tokens(
    is_set: bool,
    scheduler: &Option<syn::Path>,
    scheduler_path: proc_macro2::TokenStream,
    fallback: impl FnOnce() -> proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if is_set {
        fallback()
    } else if let Some(p) = scheduler {
        quote! { #p.#scheduler_path }
    } else {
        fallback()
    }
}

/// Render the error for a multi-segment attribute key
/// (e.g. `#[ktstr_test(crate::host_only)]` or
/// `#[ktstr_test(crate::host_only = true)]`). syn parses the path
/// fine, but `#[ktstr_test]` dispatches on bare single-segment idents
/// against the known attribute list, so multi-segment paths can never
/// resolve. The diagnostic names both correct forms with three
/// concrete value-attr examples (a scheduler path, a payload path, an
/// integer) plus the full enumeration of the bool attrs (BOOL_ATTR_NAMES)
/// that accept the bare form, so the operator can identify their fix
/// directly from the error text.
fn multi_segment_attr_error(path: &syn::Path) -> syn::Error {
    let path_repr = path.to_token_stream().to_string();
    let msg = format!(
        "unexpected multi-segment path `{path_repr}` — `#[ktstr_test]` \
         accepts either `key = value` for value attributes (e.g. \
         `scheduler = MITOSIS`, `payload = FIO`, `llcs = 4`) or the \
         bare single-segment form for bool attributes ({})",
        BOOL_ATTR_NAMES.join(", "),
    );
    syn::Error::new_spanned(path, msg)
}

/// Render a duplicate-attribute error spanning `span`. Specific
/// attributes get a tailored message; everything else falls back to a
/// uniform "appears more than once" diagnostic. Centralised so the
/// Meta::NameValue (`key = value`) and Meta::Path (bare) arms both
/// emit identical messaging for the same ident, and so future per-attr
/// wording lives in one place rather than scattered through the
/// parse loop.
fn duplicate_attr_error(ident: &str, span: &dyn ToTokens) -> syn::Error {
    let msg = match ident {
        "payload" => String::from(
            "duplicate `payload = ...` — each test declares at most one \
             primary payload; extras belong in `workloads = [..]`",
        ),
        "workloads" => String::from(
            "duplicate `workloads = [...]` — combine all entries into a \
             single array",
        ),
        "config" => String::from(
            "duplicate `config = ...` — each test declares at most one \
             inline scheduler config",
        ),
        "expect_scx_bpf_error_contains" => String::from(
            "duplicate `expect_scx_bpf_error_contains = ...` — each test \
             declares at most one literal matcher",
        ),
        "expect_scx_bpf_error_matches" => String::from(
            "duplicate `expect_scx_bpf_error_matches = ...` — each test \
             declares at most one regex matcher",
        ),
        _ => format!(
            "duplicate attribute `{ident}` — each attribute may appear at \
             most once on a single `#[ktstr_test]` invocation"
        ),
    };
    syn::Error::new_spanned(span, msg)
}

/// Default topology and memory for ktstr_test-annotated functions.
pub(crate) const DEFAULT_LLCS: u32 = 1;
pub(crate) const DEFAULT_CORES: u32 = 2;
pub(crate) const DEFAULT_THREADS: u32 = 1;
pub(crate) const DEFAULT_MEMORY_MIB: u32 = 256;

/// Canonical list of bool attributes the `#[ktstr_test]` parser
/// accepts. Used by the NameValue arm's group guard, the bare-form
/// Path arm's error message, and the user-facing docstring + doc
/// page; the dispatch table in [`AttrValues::assign_bool`] is the
/// matching per-attr behavior. Adding a new bool attribute touches:
/// (1) this slice; (2) [`AttrValues::assign_bool`]; (3) the matching
/// `<name>` + `<name>_set` field(s) on [`AttrValues`] (and the
/// matching default in [`AttrValues::default`]); (4) the codegen
/// gate that conditionally emits the new field; (5) `KtstrTestEntry`
/// + its `DEFAULT` in `src/test_support/entry.rs` (cross-crate).
pub(crate) const BOOL_ATTR_NAMES: &[&str] = &[
    "auto_repro",
    "expect_auto_repro",
    "not_stuck",
    "isolation",
    "performance_mode",
    "pci",
    "no_perf_mode",
    "requires_smt",
    "expect_err",
    "survives_storm",
    "allow_inconclusive",
    "fail_on_rq_clock_stuck",
    "host_only",
    "ignore",
    "kaslr",
    "wprof",
];

/// Canonical list of value-taking attributes the `#[ktstr_test]` parser
/// accepts (the `key = value` forms — paths, strings, integers, floats,
/// arrays). Companion to [`BOOL_ATTR_NAMES`]; the two together are the full
/// accepted set the unknown-attribute diagnostic suggests. The NameValue
/// match arms in [`ktstr_test_impl`] are the matching per-attr dispatch; this
/// slice mirrors their idents. Adding a new value attribute touches: (1) this
/// slice; (2) the matching arm (an own arm, or the integer/float/array group
/// pattern); (3) the matching field(s) on [`AttrValues`] (and the matching
/// default in [`AttrValues::default`]); (4) the codegen that emits the field.
/// The unknown-attribute catch-all `assert!`s that no name in either slice
/// reaches it (a name here with no handling arm is a const-vs-dispatch
/// divergence), and a unit test pins disjointness from [`BOOL_ATTR_NAMES`]
/// plus the total cardinality.
pub(crate) const VALUE_ATTR_NAMES: &[&str] = &[
    // Path / string / token (own arms):
    "scheduler",
    "payload",
    "workloads",
    "staged_schedulers",
    "bpf_map_write",
    "watch_bpf_maps",
    "perf_delta_assertions",
    "post_vm",
    "post_vm_unconditional",
    "disk",
    "networks",
    "config",
    "expect_scx_bpf_error_contains",
    "expect_scx_bpf_error_matches",
    "wprof_args",
    "workload_root_cgroup",
    // Array-of-path/string (own arms):
    "extra_sched_args",
    "extra_include_files",
    // Integer (group pattern):
    "llcs",
    "cores",
    "threads",
    "numa_nodes",
    "memory_mib",
    "sustained_samples",
    "max_gap_ms",
    "watchdog_timeout_s",
    "duration_s",
    "max_local_dsq_depth",
    "min_numa_nodes",
    "min_llcs",
    "min_cpus",
    "max_llcs",
    "max_numa_nodes",
    "max_cpus",
    "cpu_budget",
    "max_p99_wake_latency_ns",
    "cleanup_budget_ms",
    "num_snapshots",
    // Float (group pattern):
    "max_imbalance_ratio",
    "max_fallback_rate",
    "max_keep_last_rate",
    "max_spread_pct",
    "max_throughput_cv",
    "min_work_rate",
    "max_wake_latency_cv",
    "min_iteration_rate",
    "max_migration_ratio",
    "min_page_locality",
    "max_cross_node_migration_ratio",
    "max_slow_tier_ratio",
];

/// Owned bundle of every `#[ktstr_test]` attribute slot. The parse
/// loop in [`ktstr_test_impl`] writes `attrs.foo = ...` (and routes
/// bool dispatch through [`AttrValues::assign_bool`]) instead of the
/// 60+ bare mut locals the function used to declare. After validation
/// runs against `&attrs`, the codegen-input destructure
/// (`let AttrValues { llcs, cores, ... } = attrs;`) consumes the
/// bundle and exposes every field as a bare local for the `quote!`
/// codegen sites — `quote!{ #foo }` requires a single binding in
/// scope, so the destructure is the cleanest way to make every field
/// available to interpolation without scattering `let foo = attrs.foo;`
/// lines across the codegen region.
///
/// Each field's [`Default`] value matches the pre-bundle local's
/// initializer one-for-one. The struct is structured to mirror the
/// parse-loop assignment order: topology + memory + scheduler refs +
/// assert overrides + topology constraints + bool slots + strings +
/// token-streams.
pub(crate) struct AttrValues {
    // -- Topology --
    pub(crate) llcs: u32,
    pub(crate) cores: u32,
    pub(crate) threads: u32,
    pub(crate) numa_nodes: u32,
    pub(crate) llcs_set: bool,
    pub(crate) cores_set: bool,
    pub(crate) threads_set: bool,
    pub(crate) numa_nodes_set: bool,
    // -- Memory + duration --
    pub(crate) memory_mib: u32,
    pub(crate) memory_mib_set: bool,
    pub(crate) duration_s: u64,
    pub(crate) duration_s_set: bool,
    pub(crate) cleanup_budget_ms: Option<u64>,
    pub(crate) watchdog_timeout_s: u64,
    pub(crate) watchdog_timeout_s_set: bool,
    // -- Scheduler refs --
    pub(crate) scheduler: Option<syn::Path>,
    pub(crate) payload: Option<syn::Path>,
    pub(crate) workloads: Option<Vec<syn::Path>>,
    pub(crate) staged_schedulers: Option<Vec<syn::Path>>,
    pub(crate) bpf_map_write: Option<Vec<syn::Path>>,
    pub(crate) watch_bpf_maps: Option<Vec<syn::Path>>,
    pub(crate) perf_delta_assertions: Option<Vec<syn::Path>>,
    pub(crate) post_vm: Option<syn::Path>,
    pub(crate) post_vm_unconditional: Option<syn::Path>,
    pub(crate) disk: Option<syn::Path>,
    pub(crate) networks: Option<Vec<syn::Path>>,
    // -- Assert overrides (Option<T>) --
    pub(crate) not_stuck: Option<bool>,
    pub(crate) isolation: Option<bool>,
    pub(crate) max_gap_ms: Option<u64>,
    pub(crate) max_spread_pct: Option<f64>,
    pub(crate) max_imbalance_ratio: Option<f64>,
    pub(crate) max_local_dsq_depth: Option<u32>,
    pub(crate) fail_on_rq_clock_stuck: Option<bool>,
    pub(crate) sustained_samples: Option<usize>,
    pub(crate) max_throughput_cv: Option<f64>,
    pub(crate) min_work_rate: Option<f64>,
    pub(crate) max_fallback_rate: Option<f64>,
    pub(crate) max_keep_last_rate: Option<f64>,
    pub(crate) max_p99_wake_latency_ns: Option<u64>,
    pub(crate) max_wake_latency_cv: Option<f64>,
    pub(crate) min_iteration_rate: Option<f64>,
    pub(crate) max_migration_ratio: Option<f64>,
    pub(crate) min_page_locality: Option<f64>,
    pub(crate) max_cross_node_migration_ratio: Option<f64>,
    pub(crate) max_slow_tier_ratio: Option<f64>,
    // -- TopologyConstraints --
    pub(crate) min_numa_nodes: u32,
    pub(crate) min_numa_nodes_set: bool,
    pub(crate) min_llcs: u32,
    pub(crate) min_llcs_set: bool,
    pub(crate) requires_smt: bool,
    pub(crate) requires_smt_set: bool,
    pub(crate) min_cpus: u32,
    pub(crate) min_cpus_set: bool,
    pub(crate) max_llcs: Option<u32>,
    pub(crate) max_llcs_set: bool,
    pub(crate) max_numa_nodes: Option<u32>,
    pub(crate) max_numa_nodes_set: bool,
    pub(crate) max_cpus: Option<u32>,
    pub(crate) max_cpus_set: bool,
    // -- Resource budget: explicit no-perf host-CPU mask size override --
    pub(crate) cpu_budget: Option<u32>,
    // -- Bool attrs (per BOOL_ATTR_NAMES) --
    pub(crate) auto_repro: bool,
    pub(crate) auto_repro_set: bool,
    pub(crate) expect_auto_repro: bool,
    pub(crate) expect_auto_repro_set: bool,
    pub(crate) performance_mode: bool,
    pub(crate) performance_mode_set: bool,
    pub(crate) pci: bool,
    pub(crate) pci_set: bool,
    pub(crate) no_perf_mode: bool,
    pub(crate) no_perf_mode_set: bool,
    pub(crate) expect_err: bool,
    pub(crate) expect_err_set: bool,
    pub(crate) survives_storm: bool,
    pub(crate) survives_storm_set: bool,
    pub(crate) allow_inconclusive: bool,
    pub(crate) allow_inconclusive_set: bool,
    pub(crate) host_only: bool,
    pub(crate) host_only_set: bool,
    pub(crate) ignore_test: bool,
    pub(crate) kaslr: bool,
    pub(crate) kaslr_set: bool,
    pub(crate) wprof: bool,
    pub(crate) wprof_set: bool,
    pub(crate) num_snapshots: u32,
    pub(crate) num_snapshots_set: bool,
    // -- Strings + tokens --
    pub(crate) extra_sched_args: Vec<String>,
    pub(crate) extra_include_files: Vec<String>,
    pub(crate) workload_root_cgroup: Option<String>,
    pub(crate) wprof_args: Option<proc_macro2::TokenStream>,
    pub(crate) expect_scx_bpf_error_contains_tokens: Option<proc_macro2::TokenStream>,
    pub(crate) expect_scx_bpf_error_matches_tokens: Option<proc_macro2::TokenStream>,
    pub(crate) config_expr: Option<proc_macro2::TokenStream>,
    pub(crate) config_set: bool,
}

impl Default for AttrValues {
    fn default() -> Self {
        Self {
            // Topology
            llcs: DEFAULT_LLCS,
            cores: DEFAULT_CORES,
            threads: DEFAULT_THREADS,
            numa_nodes: 1,
            llcs_set: false,
            cores_set: false,
            threads_set: false,
            numa_nodes_set: false,
            // Memory + duration
            memory_mib: DEFAULT_MEMORY_MIB,
            memory_mib_set: false,
            duration_s: 2,
            duration_s_set: false,
            cleanup_budget_ms: None,
            watchdog_timeout_s: 4,
            watchdog_timeout_s_set: false,
            // Scheduler refs
            scheduler: None,
            payload: None,
            workloads: None,
            staged_schedulers: None,
            bpf_map_write: None,
            watch_bpf_maps: None,
            perf_delta_assertions: None,
            post_vm: None,
            post_vm_unconditional: None,
            disk: None,
            networks: None,
            // Assert overrides
            not_stuck: None,
            isolation: None,
            max_gap_ms: None,
            max_spread_pct: None,
            max_imbalance_ratio: None,
            max_local_dsq_depth: None,
            fail_on_rq_clock_stuck: None,
            sustained_samples: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_fallback_rate: None,
            max_keep_last_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: None,
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
            // TopologyConstraints
            min_numa_nodes: 1,
            min_numa_nodes_set: false,
            min_llcs: 1,
            min_llcs_set: false,
            requires_smt: false,
            requires_smt_set: false,
            min_cpus: 1,
            min_cpus_set: false,
            max_llcs: Some(12),
            max_llcs_set: false,
            max_numa_nodes: Some(1),
            max_numa_nodes_set: false,
            max_cpus: Some(192),
            max_cpus_set: false,
            cpu_budget: None,
            // Bool attrs — auto_repro + kaslr default TRUE; others false
            auto_repro: true,
            auto_repro_set: false,
            expect_auto_repro: false,
            expect_auto_repro_set: false,
            performance_mode: false,
            performance_mode_set: false,
            pci: false,
            pci_set: false,
            no_perf_mode: false,
            no_perf_mode_set: false,
            expect_err: false,
            expect_err_set: false,
            survives_storm: false,
            survives_storm_set: false,
            allow_inconclusive: false,
            allow_inconclusive_set: false,
            host_only: false,
            host_only_set: false,
            ignore_test: false,
            kaslr: true,
            kaslr_set: false,
            wprof: false,
            wprof_set: false,
            num_snapshots: 0,
            num_snapshots_set: false,
            // Strings + tokens
            extra_sched_args: Vec::new(),
            extra_include_files: Vec::new(),
            workload_root_cgroup: None,
            wprof_args: None,
            expect_scx_bpf_error_contains_tokens: None,
            expect_scx_bpf_error_matches_tokens: None,
            config_expr: None,
            config_set: false,
        }
    }
}

impl AttrValues {
    /// Assign `value` to the bool slot named `ident`. Returns `true`
    /// on a known bool ident, `false` otherwise (so the caller can
    /// route the unknown case to a targeted error). Bare-form
    /// callers (`Meta::Path` arm) pass `true`; explicit-form callers
    /// (`Meta::NameValue` arm) pass the parsed `lit_bool.value()`.
    /// The list of accepted idents is mirrored by [`BOOL_ATTR_NAMES`].
    fn assign_bool(&mut self, ident: &str, value: bool) -> bool {
        match ident {
            "auto_repro" => {
                self.auto_repro = value;
                self.auto_repro_set = true;
            }
            "expect_auto_repro" => {
                self.expect_auto_repro = value;
                self.expect_auto_repro_set = true;
            }
            "not_stuck" => self.not_stuck = Some(value),
            "isolation" => self.isolation = Some(value),
            "performance_mode" => {
                self.performance_mode = value;
                self.performance_mode_set = true;
            }
            "pci" => {
                self.pci = value;
                self.pci_set = true;
            }
            "no_perf_mode" => {
                self.no_perf_mode = value;
                self.no_perf_mode_set = true;
            }
            "requires_smt" => {
                self.requires_smt = value;
                self.requires_smt_set = true;
            }
            "expect_err" => {
                self.expect_err = value;
                self.expect_err_set = true;
            }
            "survives_storm" => {
                self.survives_storm = value;
                self.survives_storm_set = true;
            }
            "allow_inconclusive" => {
                self.allow_inconclusive = value;
                self.allow_inconclusive_set = true;
            }
            "fail_on_rq_clock_stuck" => self.fail_on_rq_clock_stuck = Some(value),
            "host_only" => {
                self.host_only = value;
                self.host_only_set = true;
            }
            "ignore" => self.ignore_test = value,
            "kaslr" => {
                self.kaslr = value;
                self.kaslr_set = true;
            }
            "wprof" => {
                self.wprof = value;
                self.wprof_set = true;
            }
            _ => return false,
        }
        true
    }
}

/// Reject the three mutex pairs that fire only when
/// `host_only = true`. Each branch describes which VM-only feature
/// the host_only short-circuit would silently no-op — extracted out
/// of [`validate_cross_attr`] so the outer fn's flat rule list stays
/// readable.
fn validate_host_only_mutex(attrs: &AttrValues) -> syn::Result<()> {
    if attrs.scheduler.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "host_only = true and a `scheduler = ...` attribute are mutually \
             exclusive — host_only skips the VM boot that owns the scheduler \
             lifecycle, so the declared scheduler would never attach. Drop \
             one of host_only or scheduler; the host's currently-active \
             scheduler (default EEVDF when none is loaded) runs the test \
             under host_only.",
        ));
    }
    if attrs.num_snapshots_set && attrs.num_snapshots > 0 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "host_only = true and num_snapshots > 0 are mutually exclusive — \
             host_only skips the VM boot that owns the qcow2 snapshot \
             lifecycle, so the declared snapshots would never be taken. \
             Drop one of host_only or num_snapshots; host_only tests run \
             once and produce no snapshots.",
        ));
    }
    if attrs.auto_repro_set && attrs.auto_repro {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "host_only = true and auto_repro = true are mutually exclusive — \
             host_only skips the VM boot that owns the auto-repro \
             machinery (probe re-launch under the failing scheduler in a \
             second VM), so auto_repro would silently no-op. Drop one of \
             host_only or auto_repro = true; host_only tests cannot trigger \
             an automatic reproducer because there is no VM to relaunch.",
        ));
    }
    Ok(())
}

/// Reject the five mutex pairs that fire only when
/// `expect_auto_repro = true`. Each branch describes which attribute
/// pair is incompatible and why — extracted out of
/// [`validate_cross_attr`] so the outer fn's flat rule list stays
/// readable without scrolling past five nested error messages.
fn validate_expect_auto_repro_mutex(attrs: &AttrValues) -> syn::Result<()> {
    if attrs.auto_repro_set && !attrs.auto_repro {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expect_auto_repro = true and auto_repro = false are \
             mutually exclusive — expect_auto_repro asserts the \
             auto-repro path FIRED, but auto_repro = false guarantees \
             it cannot fire. Drop one: either set auto_repro = true \
             (default) so the path is enabled, or drop \
             expect_auto_repro = true if the assertion isn't wanted.",
        ));
    }
    if attrs.expect_err {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expect_auto_repro = true and expect_err = true are \
             mutually exclusive — the effective auto-repro gate \
             (entry.auto_repro && scheduler.is_some() && !entry.expect_err) \
             already disables auto-repro when expect_err is set, so the \
             expect_auto_repro assertion could never be satisfied. Drop \
             one: pick expect_err for `deliberate-fail, don't probe` \
             OR pick expect_auto_repro for `deliberate-fail, DO probe`.",
        ));
    }
    if attrs.host_only_set && attrs.host_only {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expect_auto_repro = true and host_only = true are \
             mutually exclusive — host_only skips the VM boot that \
             owns the auto-repro machinery (no second VM to relaunch \
             the failing scheduler under), so the auto-repro path \
             cannot fire and the assertion could never be satisfied. \
             Drop one of host_only or expect_auto_repro = true.",
        ));
    }
    if attrs.wprof_set && !attrs.wprof {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expect_auto_repro = true and wprof = false are mutually \
             exclusive — the .repro.wprof.pb artifact whose presence \
             satisfies the expect_auto_repro assertion is written by \
             the wprof binary attached in the auto-repro VM. Without \
             wprof, no artifact lands and the assertion could never \
             be satisfied. Drop one: set wprof = true (requires the \
             `wprof` cargo feature) or drop expect_auto_repro.",
        ));
    }
    if attrs.scheduler.is_none() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expect_auto_repro = true requires a scheduler — the \
             auto-repro path re-launches the failing scheduler in a \
             second VM. Without a scheduler attribute, there is \
             nothing to relaunch and the assertion could never be \
             satisfied. Add a scheduler = ... attribute or drop \
             expect_auto_repro = true.",
        ));
    }
    Ok(())
}

/// Reject `survives_storm` combinations the assertion cannot honor:
/// `survives_storm = true` with `expect_err = true` (contradictory — one
/// demands the run pass with the scheduler alive, the other demands it
/// fail), and `survives_storm = true` with no scheduler (the kernel default
/// has no scx scheduler to die/eject, so survival is vacuous). Mirrors
/// [`validate_expect_auto_repro_mutex`]; called only when `survives_storm`
/// is set. The runtime `KtstrTestEntry::validate` re-checks both for
/// programmatically-built entries that bypass the macro.
fn validate_survives_storm_mutex(attrs: &AttrValues) -> syn::Result<()> {
    if attrs.expect_err {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "survives_storm = true and expect_err = true are mutually \
             exclusive — survives_storm asserts the scheduler SURVIVES (the \
             run passes), expect_err asserts the run FAILS. Pick one.",
        ));
    }
    if attrs.expect_auto_repro {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "survives_storm = true and expect_auto_repro = true are mutually \
             exclusive — survives_storm forces a scheduler-death failure to \
             EXIT_FAIL, while expect_auto_repro inverts a crash-with-repro \
             failure to PASS. Pick one.",
        ));
    }
    if attrs.scheduler.is_none() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "survives_storm = true requires a scheduler — the kernel default \
             (EEVDF) has no scx scheduler to die or be ejected, so survival \
             is vacuous. Add a scheduler = ... attribute or drop \
             survives_storm.",
        ));
    }
    Ok(())
}

/// Reject attribute combinations that the codegen below cannot honor.
///
/// Organised as a flat list of rules — single-attribute zero-rejects
/// via [`check_set_nonzero`] / [`check_max_nonzero`], min/max symmetry
/// via [`check_min_le_max`], and the cross-attribute mutual-exclusion
/// families via [`validate_host_only_mutex`] /
/// [`validate_expect_auto_repro_mutex`]. The few residual inline rules
/// (`llcs` divisibility, `memory_mib`/`duration_s`/`cleanup_budget_ms`
/// zero-rejects, `performance_mode` ↔ `no_perf_mode` mutex,
/// `expect_scx_bpf_error_*` ↔ `expect_err` gate) are one-off shapes
/// that don't benefit from a sibling extract. All bail on the first
/// violation, so the diagnostic the operator sees names the highest-
/// priority conflict in scroll order rather than batching every
/// reachable error.
///
/// Inputs arrive in the owned [`AttrValues`] bundle the parse loop in
/// [`ktstr_test_impl`] populates. Reading directly from `&AttrValues`
/// keeps this signature flat regardless of how many attributes feed
/// validation.
fn validate_cross_attr(attrs: &AttrValues) -> syn::Result<()> {
    check_set_nonzero(
        attrs.llcs_set,
        attrs.llcs,
        "llcs must be > 0 (a topology with zero LLCs has zero CPUs — \
         `total_cpus = llcs * cores * threads` — so the VM would boot \
         with no addressable processors)",
    )?;
    check_set_nonzero(
        attrs.cores_set,
        attrs.cores,
        "cores must be > 0 (a topology with zero cores per LLC has \
         zero CPUs — `total_cpus = llcs * cores * threads` — so the \
         VM would boot with no addressable processors)",
    )?;
    check_set_nonzero(
        attrs.threads_set,
        attrs.threads,
        "threads must be > 0 (a topology with zero threads per core \
         has zero CPUs — `total_cpus = llcs * cores * threads` — so \
         the VM would boot with no addressable processors)",
    )?;
    check_set_nonzero(
        attrs.numa_nodes_set,
        attrs.numa_nodes,
        "numa_nodes must be > 0 (a topology with zero NUMA nodes has \
         nothing to attach LLCs or memory to; every downstream \
         accessor would observe an empty node set)",
    )?;
    if attrs.llcs_set && attrs.numa_nodes_set && !attrs.llcs.is_multiple_of(attrs.numa_nodes) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "llcs ({}) must be divisible by numa_nodes ({})",
                attrs.llcs, attrs.numa_nodes
            ),
        ));
    }
    if attrs.memory_mib == 0 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "memory_mib must be > 0 (a VM with zero memory cannot boot)",
        ));
    }
    if attrs.duration_s == 0 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "duration_s must be > 0 (a zero-duration run never exercises the \
             scheduler and produces no data for assertions)",
        ));
    }
    if attrs.cleanup_budget_ms == Some(0) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cleanup_budget_ms must be > 0 — a zero budget would \
             reject every successful run (any measurable cleanup \
             duration overshoots zero). Omit the attribute to \
             disable the check.",
        ));
    }
    if attrs.cpu_budget == Some(0) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cpu_budget must be > 0 (a zero host-CPU budget cannot run a \
             VM; omit the attribute to auto-size the no-perf mask to the \
             vCPU count)",
        ));
    }
    // `performance_mode = true` and `no_perf_mode = true` are
    // mutually exclusive — the macro docstring (Mutually exclusive
    // with `performance_mode = true`) is the only previous guard and
    // a user setting both gets silent precedence behavior at runtime.
    // Reject explicitly so the conflict surfaces at compile time.
    if attrs.performance_mode_set
        && attrs.no_perf_mode_set
        && attrs.performance_mode
        && attrs.no_perf_mode
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "performance_mode = true and no_perf_mode = true are mutually \
             exclusive — one disables what the other enables. Set at most \
             one of the two; remove the other or set it to `false`.",
        ));
    }
    // `cpu_budget` only sizes the no-perf vCPU-thread mask; under
    // `performance_mode` (vCPUs pinned 1:1) or the default LLC mode the
    // budget is never read, so a `cpu_budget` set without `no_perf_mode`
    // would be a silent no-op (a contention test that quietly runs
    // un-contended). Require `no_perf_mode` so the knob always takes
    // effect — same compile-time cross-attr shape as the
    // `performance_mode` ↔ `no_perf_mode` mutex above. The
    // programmatic-construction path (bypassing the macro) is guarded at
    // runtime in `src/test_support/entry_validate.rs::validate`.
    if attrs.cpu_budget.is_some() && !(attrs.no_perf_mode_set && attrs.no_perf_mode) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cpu_budget requires no_perf_mode — the budget sizes the \
             no-perf vCPU-thread mask; under performance_mode vCPUs are \
             pinned 1:1 and cpu_budget would be silently ignored. Add \
             no_perf_mode (or drop cpu_budget).",
        ));
    }
    // `perf_delta_assertions` tighten the perf-delta noise threshold on a
    // metric, which is only meaningful on a pinned run. Under no-perf mode
    // (or the default LLC mode) the metric carries host-CPU-oversubscription
    // noise the gate would misread — false regressions, or a real one masked
    // by the narrowed band. Require `performance_mode` so a declared gate
    // always sees pinned data. Same compile-time cross-attr shape as the
    // `cpu_budget ⇒ no_perf_mode` require above; the programmatic-
    // construction path is guarded at runtime in
    // `src/test_support/entry_validate.rs::validate_perf_delta_assertions`.
    if attrs
        .perf_delta_assertions
        .as_ref()
        .is_some_and(|paths| !paths.is_empty())
        && !(attrs.performance_mode_set && attrs.performance_mode)
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "perf_delta_assertions requires performance_mode — a declared \
             regression gate tightens the noise threshold on a metric, which \
             is only meaningful on a pinned (performance_mode) run. Under \
             no-perf mode the metric carries host-CPU-oversubscription noise \
             the gate would misread. Add performance_mode = true (or drop \
             the assertions).",
        ));
    }
    // `host_only = true` short-circuits the VM-boot pipeline before
    // any of the VM-only features fire. Catch attribute combinations
    // where the user pairs `host_only = true` with attributes that
    // require the VM to exist — the runtime would silently no-op the
    // VM-only side, leaving the user wondering why the scheduler/
    // snapshot/auto-repro behavior they declared never runs. Promote
    // to compile-time so the conflict surfaces at edit-compile time
    // rather than at test-run time.
    //
    // Defense-in-depth coverage matrix (compile-time here +
    // programmatic-construction runtime check in
    // `src/test_support/entry_validate.rs::validate`):
    //   - host_only + disk         : runtime-only (validate_host_only_mutex
    //                                 below gates only scheduler/num_snapshots/
    //                                 auto_repro, not disk; the conflict is
    //                                 caught in `entry::validate`)
    //   - host_only + networks     : runtime-only (same as disk — caught in
    //                                 `entry::validate`, not compile-time)
    //   - host_only + scheduler    : BOTH (runtime matches against
    //                                 `SchedulerSpec::Eevdf` variant —
    //                                 spec-safe value comparison vs
    //                                 fragile pointer-identity on
    //                                 `&CONST_EXPR`)
    //   - host_only + num_snapshots: BOTH (runtime checks
    //                                 `num_snapshots > 0`)
    //   - host_only + auto_repro   : compile-time-only (auto_repro
    //                                 defaults to `true`, and the
    //                                 entry struct lacks provenance
    //                                 for "user explicitly set" vs
    //                                 "default applied"; runtime
    //                                 cannot tell whether the
    //                                 conflict is intentional)
    //   - matcher + expect_err=false: BOTH (runtime check fires on
    //                                 omitted-default expect_err too)
    // Ordering note: the three host_only mutex checks below (the
    // fourth check on matcher + !expect_err is a separate conflict
    // class on different attributes, included in the same block for
    // proximity) are ordered by likely-user-confusion descending.
    // scheduler-first because a declared-but-never-attached
    // scheduler is the most disorienting silent no-op. The trybuild
    // fixture `ktstr_test_multi_mutex_first_wins` pins this
    // precedence so a refactor that reorders the blocks doesn't
    // silently change which diagnostic the operator sees first.
    if attrs.host_only_set && attrs.host_only {
        validate_host_only_mutex(attrs)?;
    }
    if attrs.expect_auto_repro_set && attrs.expect_auto_repro {
        validate_expect_auto_repro_mutex(attrs)?;
    }
    if attrs.survives_storm_set && attrs.survives_storm {
        validate_survives_storm_mutex(attrs)?;
    }
    if (attrs.expect_scx_bpf_error_contains_tokens.is_some()
        || attrs.expect_scx_bpf_error_matches_tokens.is_some())
        && !attrs.expect_err
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expect_scx_bpf_error_contains/matches without expect_err = true \
             are mutually exclusive — an scx_bpf_error matcher narrows \
             which failure counts as the expected bug and only applies to \
             expected-error tests. Drop the matcher (if you want any failure \
             to count) or set expect_err = true (if you want this specific \
             error to be the expected bug). The runtime check at \
             entry_validate.rs::validate enforces the same invariant for \
             programmatic-construction paths that bypass the macro.",
        ));
    }
    // Validate explicitly set constraint values. When a field is
    // inherited from the scheduler, the proc macro doesn't know the
    // value so cross-field validation is deferred to runtime.
    check_max_nonzero(
        attrs.max_llcs_set,
        attrs.max_llcs,
        "max_llcs must be > 0 (a zero cap excludes every host from \
         the gauntlet — use a non-zero cap, or omit the field to \
         use the default)",
    )?;
    check_max_nonzero(
        attrs.max_numa_nodes_set,
        attrs.max_numa_nodes,
        "max_numa_nodes must be > 0 (a zero cap excludes every host \
         from the gauntlet — use a non-zero cap, or omit the field \
         to inherit the scheduler-level default)",
    )?;
    check_max_nonzero(
        attrs.max_cpus_set,
        attrs.max_cpus,
        "max_cpus must be > 0 (a zero cap excludes every host from \
         the gauntlet — use a non-zero cap, or omit the field to \
         use the default)",
    )?;
    check_min_le_max(
        "llcs",
        attrs.min_llcs_set,
        attrs.min_llcs,
        attrs.max_llcs_set,
        attrs.max_llcs,
    )?;
    check_min_le_max(
        "numa_nodes",
        attrs.min_numa_nodes_set,
        attrs.min_numa_nodes,
        attrs.max_numa_nodes_set,
        attrs.max_numa_nodes,
    )?;
    check_min_le_max(
        "cpus",
        attrs.min_cpus_set,
        attrs.min_cpus,
        attrs.max_cpus_set,
        attrs.max_cpus,
    )?;
    Ok(())
}

/// Reject `FIELD = 0` when the field is explicitly set. Used by
/// the topology zero-rejects (`llcs`, `cores`, `threads`,
/// `numa_nodes`) — each had the same `if FIELD_set && FIELD == 0
/// { return Err... }` triple inline.
fn check_set_nonzero(set: bool, value: u32, error_hint: &str) -> syn::Result<()> {
    if set && value == 0 {
        return Err(syn::Error::new(proc_macro2::Span::call_site(), error_hint));
    }
    Ok(())
}

/// Reject `max_FIELD = Some(0)` when the field is explicitly set.
/// Each of the three (`max_llcs`, `max_numa_nodes`, `max_cpus`)
/// repeats the same set-and-zero check inline with a slightly
/// different diagnostic; folded into one helper that takes the
/// diagnostic message verbatim so the per-field text stays intact.
fn check_max_nonzero(set: bool, value: Option<u32>, error_hint: &str) -> syn::Result<()> {
    if set && value == Some(0) {
        return Err(syn::Error::new(proc_macro2::Span::call_site(), error_hint));
    }
    Ok(())
}

/// Reject `min_FIELD > max_FIELD` when both are explicitly set.
/// Each of the three (`llcs`, `numa_nodes`, `cpus`) had the same
/// `min_set && max_set && matches!(max, Some(m) if m < min)` triple
/// inline; folded into one helper keyed on the user-facing field
/// stem.
fn check_min_le_max(
    field: &str,
    min_set: bool,
    min_value: u32,
    max_set: bool,
    max_value: Option<u32>,
) -> syn::Result<()> {
    if min_set
        && max_set
        && let Some(m) = max_value
        && m < min_value
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "min_{field} ({min_value}) exceeds max_{field} ({m}). Set max_{field} explicitly."
            ),
        ));
    }
    Ok(())
}

/// Reject payload-list redundancy: the primary `payload = ...` ident
/// must not also appear in `workloads = [..]`, and the `workloads`
/// array itself must have no pairwise repeats. Both rules compare
/// via `ToTokens` string form, so in-file aliases collide
/// (`FIO` == `FIO`) but resolved-path identity does not
/// (`FIO` vs `crate::FIO`) — macro expansion has no name-resolution
/// context. Runtime validation can extend the check once the entry
/// is constructed.
pub(crate) fn validate_payload_workloads_dedup(
    payload: &Option<syn::Path>,
    workloads_slice: &[syn::Path],
) -> syn::Result<()> {
    if let Some(primary) = payload.as_ref() {
        let primary_repr = primary.to_token_stream().to_string();
        for w in workloads_slice {
            if w.to_token_stream().to_string() == primary_repr {
                return Err(syn::Error::new_spanned(
                    w,
                    format!(
                        "`{primary_repr}` appears in both `payload = ...` and \
                         `workloads = [..]` — pick one. The primary payload \
                         runs as the test's main workload; entries in \
                         `workloads` are composed alongside it."
                    ),
                ));
            }
        }
    }
    for (i, wi_path) in workloads_slice.iter().enumerate() {
        let wi = wi_path.to_token_stream().to_string();
        for wj_path in workloads_slice.iter().skip(i + 1) {
            let wj = wj_path.to_token_stream().to_string();
            if wi == wj {
                return Err(syn::Error::new_spanned(
                    wj_path,
                    format!(
                        "`{wi}` appears twice in `workloads = [..]` — each \
                         workload entry must be distinct. Remove the \
                         duplicate or compose the payload once and rely \
                         on runtime scheduling to spread it across cgroups."
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Inner implementation of `ktstr_test` operating on `proc_macro2::TokenStream`
/// so unit tests in this crate can synthesize input + assert output without
/// the `proc_macro` runtime context (`proc_macro` types panic outside a
/// procedural-macro invocation). The proc-macro entry above is a thin wrapper
/// that converts `proc_macro::TokenStream` ↔ `proc_macro2::TokenStream` and
/// projects `Err` to the compile-error token stream.
pub(crate) fn ktstr_test_impl(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let input: ItemFn = syn::parse2(item)?;

    // Parse attributes into a single owned bundle. Per-field defaults
    // live on [`AttrValues::default`]; the parse loop below writes
    // `attrs.foo = ...` and the bool dispatch routes through
    // [`AttrValues::assign_bool`]. The bare locals previously declared
    // here are re-introduced as a destructure bridge immediately after
    // the parse loop so validators and codegen continue to read them
    // unchanged (eliminated in steps 3 + 4 of the refactor).
    let mut attrs = AttrValues::default();

    let attr_parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    let parsed_attrs = attr_parser.parse2(attr)?;

    // Track every attribute key seen on this `#[ktstr_test]` invocation so
    // accidental duplicates (`host_only = true, host_only`,
    // `llcs = 4, llcs = 8`) are caught at expansion rather than silently
    // letting the later value win. Checked at the top of each Meta arm
    // once the ident has been extracted. The set is keyed by the bare
    // identifier string so the bare-form (`host_only`) and explicit-form
    // (`host_only = true`) of the same attribute collide as expected.
    let mut seen_attrs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for meta in &parsed_attrs {
        match meta {
            Meta::NameValue(MetaNameValue { path, value, .. }) => {
                let ident = match path.get_ident() {
                    Some(id) => id.to_string(),
                    None => return Err(multi_segment_attr_error(path)),
                };
                if !seen_attrs.insert(ident.clone()) {
                    return Err(duplicate_attr_error(&ident, path));
                }
                match ident.as_str() {
                    "scheduler" => {
                        attrs.scheduler = Some(expect_path_value(
                            value,
                            "expected path for scheduler (e.g. MITOSIS or crate::MITOSIS)",
                        )?);
                    }
                    "payload" => {
                        attrs.payload = Some(expect_path_value(
                            value,
                            "expected path for payload (e.g. FIO or crate::FIO)",
                        )?);
                    }
                    "workloads" => {
                        attrs.workloads = Some(expect_array_of_paths(
                            value,
                            "expected array of Payload paths for workloads \
                             (e.g. [FIO, STRESS_NG])",
                            "expected Payload path in workloads array",
                        )?);
                    }
                    "staged_schedulers" => {
                        attrs.staged_schedulers = Some(expect_array_of_paths(
                            value,
                            "expected array of Scheduler paths for \
                             staged_schedulers (e.g. [SCX_VARIANT_A, \
                             SCX_VARIANT_B])",
                            "expected Scheduler path in \
                             staged_schedulers array",
                        )?);
                    }
                    "bpf_map_write" => {
                        attrs.bpf_map_write = Some(expect_path_or_array_of_paths(
                            value,
                            "expected a BpfMapWrite path or [A, B] array for \
                             bpf_map_write (e.g. BPF_CRASH or [WRITE_A, WRITE_B])",
                        )?);
                    }
                    "watch_bpf_maps" => {
                        attrs.watch_bpf_maps = Some(expect_path_or_array_of_paths(
                            value,
                            "expected a WatchBpfMap path or [A, B] array for \
                             watch_bpf_maps (e.g. WATCH or [WATCH_A, WATCH_B])",
                        )?);
                    }
                    "perf_delta_assertions" => {
                        attrs.perf_delta_assertions = Some(expect_path_or_array_of_paths(
                            value,
                            "expected a PerfDeltaAssertion path or [A, B] array for \
                             perf_delta_assertions (e.g. RPS_GATE or [GATE_A, GATE_B])",
                        )?);
                    }
                    "post_vm" => {
                        attrs.post_vm = Some(expect_path_value(
                            value,
                            "expected path for post_vm (e.g. my_post_vm_check)",
                        )?);
                    }
                    "post_vm_unconditional" => {
                        attrs.post_vm_unconditional = Some(expect_path_value(
                            value,
                            "expected path for post_vm_unconditional \
                             (e.g. my_callback)",
                        )?);
                    }
                    "disk" => {
                        attrs.disk = Some(expect_path_value(
                            value,
                            "expected path for disk (e.g. MY_DISK \
                             where MY_DISK is a `const DiskConfig`); \
                             construct via `DiskConfig::DEFAULT.with_name(...)` \
                             or similar const-fn chain",
                        )?);
                    }
                    "networks" => {
                        attrs.networks = Some(expect_array_of_paths(
                            value,
                            "expected array of NetConfig paths for networks \
                             (e.g. [NET_A, NET_B]); each is a `const NetConfig` \
                             constructed via `NetConfig::DEFAULT.mac(...)` or a \
                             similar const-fn chain",
                            "expected NetConfig path in networks array",
                        )?);
                    }
                    "config" => {
                        let tokens = expect_string_or_path_tokens(
                            value,
                            "expected string literal or path to a \
                             `const &'static str` for `config` (e.g. \
                             `config = \"{...}\"` or `config = MY_CONFIG`)",
                        )?;
                        attrs.config_expr = Some(tokens);
                        attrs.config_set = true;
                    }
                    "expect_scx_bpf_error_contains" | "expect_scx_bpf_error_matches" => {
                        let tokens = expect_string_or_path_tokens(
                            value,
                            &format!(
                                "expected string literal or path to a \
                                 `const &'static str` for `{ident}` (e.g. \
                                 `{ident} = \"apply_cell_config returned \
                                 -EINVAL\"` or `{ident} = MY_PATTERN`)",
                            ),
                        )?;
                        match ident.as_str() {
                            "expect_scx_bpf_error_contains" => {
                                attrs.expect_scx_bpf_error_contains_tokens = Some(tokens);
                            }
                            "expect_scx_bpf_error_matches" => {
                                attrs.expect_scx_bpf_error_matches_tokens = Some(tokens);
                            }
                            _ => unreachable!(),
                        }
                    }
                    "wprof_args" => {
                        // Accept either a string literal (with
                        // macro-time empty-check) or a path to a
                        // `const &'static str` (deferred to runtime —
                        // the const value is opaque at macro time).
                        // Mirrors the `expect_scx_bpf_error_contains`
                        // / `expect_scx_bpf_error_matches` pattern so
                        // tests can dedupe their override args via a
                        // shared `const OVERRIDE_ARGS: &str = "..."`
                        // instead of duplicating the literal at every
                        // call site (which drifts under any future
                        // refactor that renames or shortens the
                        // option list).
                        let tokens = match value {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(ls),
                                ..
                            }) => {
                                // Empty / whitespace-only literals
                                // tokenize to a zero-length Vec via
                                // `split_whitespace`, then render as
                                // `KTSTR_WPROF_ARGS=` on the kernel
                                // cmdline -- wprof launches with no
                                // args and the operator's "override"
                                // intent becomes a silent no-op. Bail
                                // at macro-parse time with an
                                // actionable error that names both
                                // recovery paths. Only runs for
                                // literal values; const-path values
                                // skip this gate (the const body is
                                // unknown at macro time).
                                let lit_str = ls.value();
                                if lit_str.trim().is_empty() {
                                    return Err(syn::Error::new_spanned(
                                        value,
                                        "wprof_args literal is empty or \
                                         whitespace-only -- drop the attribute \
                                         to use the wprof default args, or pass \
                                         a non-empty token list (e.g. \
                                         `wprof_args = \"-d 2000 -e \
                                         sched,irq\"`). An empty override \
                                         silently launches wprof with zero \
                                         args, which is not a meaningful \
                                         configuration.",
                                    ));
                                }
                                quote! { #value }
                            }
                            syn::Expr::Path(_) => quote! { #value },
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    value,
                                    "expected string literal or path to a \
                                     `const &'static str` for `wprof_args` \
                                     (e.g. `wprof_args = \"-d 2000 -e \
                                     sched,irq\"` or `wprof_args = \
                                     MY_OVERRIDE_ARGS`)",
                                ));
                            }
                        };
                        attrs.wprof_args = Some(tokens);
                    }
                    "workload_root_cgroup" => {
                        // `workload_root_cgroup = "/path"` lands in
                        // KtstrTestEntry::workload_root_cgroup as
                        // `Some(CgroupPath::new(path))`. The CgroupPath
                        // const constructor const-panics on malformed
                        // input (missing leading `/`, bare `/`, `..`
                        // components), so the validation surface stays
                        // identical to the runtime
                        // `Scheduler::cgroup_parent` builder path —
                        // see `CgroupPath::new` for the gate.
                        attrs.workload_root_cgroup = Some(expect_string_literal_value(
                            value,
                            "expected string literal for \
                             workload_root_cgroup (e.g. \
                             `workload_root_cgroup = \"/my_workloads\"`)",
                        )?);
                    }
                    _ if BOOL_ATTR_NAMES.contains(&ident.as_str()) => {
                        let lit_bool = expect_bool_literal(
                            value,
                            &format!("expected bool literal for {ident}"),
                        )?;
                        // Guard guarantees `ident` is in [`BOOL_ATTR_NAMES`], so
                        // [`AttrValues::assign_bool`] always hits a slot. The
                        // `assert!` catches a soft invariant: a future bool
                        // attribute added to [`BOOL_ATTR_NAMES`] but missing
                        // from [`AttrValues::assign_bool`] would silently drop
                        // here without this gate.
                        assert!(
                            attrs.assign_bool(&ident, lit_bool.value()),
                            "internal: `{ident}` is in BOOL_ATTR_NAMES but \
                             AttrValues::assign_bool has no arm for it",
                        );
                    }
                    "llcs"
                    | "cores"
                    | "threads"
                    | "numa_nodes"
                    | "memory_mib"
                    | "sustained_samples"
                    | "max_gap_ms"
                    | "watchdog_timeout_s"
                    | "duration_s"
                    | "max_local_dsq_depth"
                    | "min_numa_nodes"
                    | "min_llcs"
                    | "min_cpus"
                    | "max_llcs"
                    | "max_numa_nodes"
                    | "max_cpus"
                    | "cpu_budget"
                    | "max_p99_wake_latency_ns"
                    | "cleanup_budget_ms"
                    | "num_snapshots" => {
                        let lit_int = expect_int_literal(value, "expected integer literal")?;
                        match ident.as_str() {
                            "llcs" => {
                                attrs.llcs = lit_int.base10_parse()?;
                                attrs.llcs_set = true;
                            }
                            "numa_nodes" => {
                                attrs.numa_nodes = lit_int.base10_parse()?;
                                attrs.numa_nodes_set = true;
                            }
                            "cores" => {
                                attrs.cores = lit_int.base10_parse()?;
                                attrs.cores_set = true;
                            }
                            "threads" => {
                                attrs.threads = lit_int.base10_parse()?;
                                attrs.threads_set = true;
                            }
                            "memory_mib" => {
                                attrs.memory_mib = lit_int.base10_parse()?;
                                attrs.memory_mib_set = true;
                            }
                            "sustained_samples" => {
                                attrs.sustained_samples = Some(lit_int.base10_parse()?);
                            }
                            "cpu_budget" => {
                                attrs.cpu_budget = Some(lit_int.base10_parse()?);
                            }
                            "max_gap_ms" => {
                                attrs.max_gap_ms = Some(lit_int.base10_parse()?);
                            }
                            "cleanup_budget_ms" => {
                                attrs.cleanup_budget_ms = Some(lit_int.base10_parse()?);
                            }
                            "watchdog_timeout_s" => {
                                attrs.watchdog_timeout_s = lit_int.base10_parse()?;
                                attrs.watchdog_timeout_s_set = true;
                            }
                            "duration_s" => {
                                attrs.duration_s = lit_int.base10_parse()?;
                                attrs.duration_s_set = true;
                            }
                            "num_snapshots" => {
                                attrs.num_snapshots = lit_int.base10_parse()?;
                                attrs.num_snapshots_set = true;
                            }
                            "max_local_dsq_depth" => {
                                attrs.max_local_dsq_depth = Some(lit_int.base10_parse()?);
                            }
                            "min_numa_nodes" => {
                                attrs.min_numa_nodes = lit_int.base10_parse()?;
                                attrs.min_numa_nodes_set = true;
                            }
                            "min_llcs" => {
                                attrs.min_llcs = lit_int.base10_parse()?;
                                attrs.min_llcs_set = true;
                            }
                            "min_cpus" => {
                                attrs.min_cpus = lit_int.base10_parse()?;
                                attrs.min_cpus_set = true;
                            }
                            "max_llcs" => {
                                attrs.max_llcs = Some(lit_int.base10_parse()?);
                                attrs.max_llcs_set = true;
                            }
                            "max_numa_nodes" => {
                                attrs.max_numa_nodes = Some(lit_int.base10_parse()?);
                                attrs.max_numa_nodes_set = true;
                            }
                            "max_cpus" => {
                                attrs.max_cpus = Some(lit_int.base10_parse()?);
                                attrs.max_cpus_set = true;
                            }
                            "max_p99_wake_latency_ns" => {
                                attrs.max_p99_wake_latency_ns = Some(lit_int.base10_parse()?);
                            }
                            _ => unreachable!(),
                        }
                    }
                    "max_imbalance_ratio"
                    | "max_fallback_rate"
                    | "max_keep_last_rate"
                    | "max_spread_pct"
                    | "max_throughput_cv"
                    | "min_work_rate"
                    | "max_wake_latency_cv"
                    | "min_iteration_rate"
                    | "max_migration_ratio"
                    | "min_page_locality"
                    | "max_cross_node_migration_ratio"
                    | "max_slow_tier_ratio" => {
                        let lit_float = expect_float_literal(
                            value,
                            &format!("expected float literal for {ident}"),
                        )?;
                        let v: f64 = lit_float.base10_parse()?;
                        match ident.as_str() {
                            "max_imbalance_ratio" => attrs.max_imbalance_ratio = Some(v),
                            "max_fallback_rate" => attrs.max_fallback_rate = Some(v),
                            "max_keep_last_rate" => attrs.max_keep_last_rate = Some(v),
                            "max_spread_pct" => attrs.max_spread_pct = Some(v),
                            "max_throughput_cv" => attrs.max_throughput_cv = Some(v),
                            "min_work_rate" => attrs.min_work_rate = Some(v),
                            "max_wake_latency_cv" => attrs.max_wake_latency_cv = Some(v),
                            "min_iteration_rate" => attrs.min_iteration_rate = Some(v),
                            "max_migration_ratio" => attrs.max_migration_ratio = Some(v),
                            "min_page_locality" => attrs.min_page_locality = Some(v),
                            "max_cross_node_migration_ratio" => {
                                attrs.max_cross_node_migration_ratio = Some(v)
                            }
                            "max_slow_tier_ratio" => attrs.max_slow_tier_ratio = Some(v),
                            _ => unreachable!(),
                        }
                    }
                    "extra_sched_args" => {
                        let entries = expect_array_of_string_literals(
                            value,
                            "expected array of string literals for extra_sched_args",
                            "expected string literal in extra_sched_args",
                        )?;
                        // `/sched_args` frames one argument per line; a line
                        // break inside an element would re-split it into
                        // separate scheduler argv entries in the guest.
                        if let Some(bad) = entries.iter().find(|e| e.contains(['\n', '\r'])) {
                            return Err(syn::Error::new_spanned(
                                value,
                                format!(
                                    "extra_sched_args element {bad:?} must not contain a line break"
                                ),
                            ));
                        }
                        attrs.extra_sched_args.extend(entries);
                    }
                    "extra_include_files" => {
                        attrs
                            .extra_include_files
                            .extend(expect_array_of_string_literals(
                                value,
                                "expected array of string literals for extra_include_files",
                                "expected string literal in extra_include_files",
                            )?);
                    }
                    "workers_per_cgroup" => {
                        return Err(syn::Error::new_spanned(
                            path,
                            "`workers_per_cgroup` is no longer a `#[ktstr_test]` \
                             attribute. Set workers per cgroup directly in the \
                             scenario body via `CgroupDef::named(\"X\").workers(N)`.",
                        ));
                    }
                    _ => {
                        // The unknown-attribute "expected:" list is DERIVED from
                        // the two name registries (sorted for a scannable
                        // diagnostic), not hand-maintained. A name in either
                        // registry must have matched an arm above; reaching here
                        // with such a name is a const-vs-dispatch divergence — the
                        // same soft-invariant the bool path guards via the
                        // BOOL_ATTR_NAMES `assert!`.
                        let s = ident.as_str();
                        assert!(
                            !BOOL_ATTR_NAMES.contains(&s) && !VALUE_ATTR_NAMES.contains(&s),
                            "internal: `{ident}` is a known attribute (in \
                             BOOL_ATTR_NAMES/VALUE_ATTR_NAMES) but reached the \
                             unknown-attribute arm — const-vs-dispatch divergence",
                        );
                        let mut expected: Vec<&str> = BOOL_ATTR_NAMES
                            .iter()
                            .chain(VALUE_ATTR_NAMES)
                            .copied()
                            .collect();
                        expected.sort_unstable();
                        return Err(syn::Error::new_spanned(
                            path,
                            format!(
                                "unknown attribute `{ident}`, expected: {}",
                                expected.join(", "),
                            ),
                        ));
                    }
                }
            }
            Meta::Path(p) => {
                // Sugar: a bare bool attr (e.g. `#[ktstr_test(host_only)]`)
                // is equivalent to `key = true`. Only the bool
                // attributes (BOOL_ATTR_NAMES) accept this form; bare ints/floats/paths
                // still error so a typo on a non-bool attr ("threads"
                // instead of "threads = 4") routes to a targeted
                // diagnostic rather than the generic Meta catch-all.
                let ident = match p.get_ident() {
                    Some(id) => id.to_string(),
                    None => return Err(multi_segment_attr_error(p)),
                };
                if !seen_attrs.insert(ident.clone()) {
                    return Err(duplicate_attr_error(&ident, p));
                }
                if !attrs.assign_bool(&ident, true) {
                    return Err(syn::Error::new_spanned(
                        p,
                        format!(
                            "bare attribute `{ident}` is not a bool flag; only \
                             bool attributes accept the bare form. Other \
                             attributes require `key = value`. Bool attrs: {}.",
                            BOOL_ATTR_NAMES.join(", "),
                        ),
                    ));
                }
            }
            Meta::List(ml) => {
                // `key(args)` syntax (e.g. `#[ktstr_test(host_only(false))]`)
                // — accepted by `syn::Meta` parsing but never valid for
                // `#[ktstr_test]`. Emit a targeted message naming the
                // attribute so the fix (`= value` or bare) is obvious,
                // rather than the generic Meta dispatch above. Multi-segment
                // paths (`crate::host_only(false)`) cannot resolve against
                // the attribute table either way, so route those to the
                // shared multi-segment diagnostic instead of producing a
                // nonsense `crate :: host_only = value` suggestion.
                let path_ident = match ml.path.get_ident() {
                    Some(id) => id.to_string(),
                    None => return Err(multi_segment_attr_error(&ml.path)),
                };
                return Err(syn::Error::new_spanned(
                    &ml.path,
                    format!(
                        "unexpected parenthesised arguments on `{path_ident}`; \
                         use `{path_ident} = value` for value attributes or bare \
                         `{path_ident}` for bool attributes ({})",
                        BOOL_ATTR_NAMES.join(", "),
                    ),
                ));
            }
        }
    }

    // Validate BEFORE destructure — both validators borrow `attrs`,
    // and the destructure below consumes it.
    validate_payload_workloads_dedup(&attrs.payload, attrs.workloads.as_deref().unwrap_or(&[]))?;
    validate_cross_attr(&attrs)?;

    #[cfg(not(feature = "wprof"))]
    {
        if attrs.wprof {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "wprof requires the `wprof` cargo feature — add \
                 `features = [\"wprof\"]` to your ktstr dependency \
                 in Cargo.toml and rebuild cargo-ktstr with the same \
                 feature. wprof is opt-in (default off) because it \
                 pulls a multi-minute build (git clone + BPF compile).",
            ));
        }
        if attrs.wprof_args.is_some() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "wprof_args requires the `wprof` cargo feature — add \
                 `features = [\"wprof\"]` to your ktstr dependency \
                 in Cargo.toml and rebuild cargo-ktstr with the same \
                 feature.",
            ));
        }
        if attrs.expect_auto_repro {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "expect_auto_repro requires the `wprof` cargo feature \
                 — the .repro.wprof.pb artifact it asserts on is \
                 produced by the wprof binary, which is only embedded \
                 when the feature is enabled. Add `features = \
                 [\"wprof\"]` to your ktstr dependency in Cargo.toml.",
            ));
        }
    }

    Ok(codegen::emit_entry_static(input, attrs))
}
