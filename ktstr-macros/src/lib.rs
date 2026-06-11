use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, ItemFn, Meta, MetaNameValue, parse::Parser, parse_macro_input,
};

#[allow(dead_code)]
mod kernel_path;

/// Emit `Some(value)` or `None` as token streams.
fn option_tokens<T: ToTokens>(opt: &Option<T>) -> proc_macro2::TokenStream {
    match opt {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    }
}

/// Extract a [`syn::Path`] from an attribute value, returning a focused
/// error spanned to the offending expression when the user supplied
/// something other than a path (an int, a string, an array, …).
/// Collapses the six `"scheduler" | "payload" | "bpf_map_write" | "post_vm" |
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
/// "staged_schedulers"` parse arms.
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
/// integer) plus the full enumeration of the 10 bool attrs that
/// accept the bare form, so the operator can identify their fix
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
const DEFAULT_LLCS: u32 = 1;
const DEFAULT_CORES: u32 = 2;
const DEFAULT_THREADS: u32 = 1;
const DEFAULT_MEMORY_MIB: u32 = 2048;

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
const BOOL_ATTR_NAMES: &[&str] = &[
    "auto_repro",
    "expect_auto_repro",
    "not_starved",
    "isolation",
    "performance_mode",
    "no_perf_mode",
    "requires_smt",
    "expect_err",
    "allow_inconclusive",
    "fail_on_stall",
    "host_only",
    "ignore",
    "kaslr",
    "wprof",
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
struct AttrValues {
    // -- Topology --
    llcs: u32,
    cores: u32,
    threads: u32,
    numa_nodes: u32,
    llcs_set: bool,
    cores_set: bool,
    threads_set: bool,
    numa_nodes_set: bool,
    // -- Memory + duration --
    memory_mib: u32,
    memory_mib_set: bool,
    duration_s: u64,
    duration_s_set: bool,
    cleanup_budget_ms: Option<u64>,
    watchdog_timeout_s: u64,
    watchdog_timeout_s_set: bool,
    // -- Scheduler refs --
    scheduler: Option<syn::Path>,
    payload: Option<syn::Path>,
    workloads: Option<Vec<syn::Path>>,
    staged_schedulers: Option<Vec<syn::Path>>,
    bpf_map_write: Option<syn::Path>,
    post_vm: Option<syn::Path>,
    post_vm_unconditional: Option<syn::Path>,
    disk: Option<syn::Path>,
    // -- Assert overrides (Option<T>) --
    not_starved: Option<bool>,
    isolation: Option<bool>,
    max_gap_ms: Option<u64>,
    max_spread_pct: Option<f64>,
    max_imbalance_ratio: Option<f64>,
    max_local_dsq_depth: Option<u32>,
    fail_on_stall: Option<bool>,
    sustained_samples: Option<usize>,
    max_throughput_cv: Option<f64>,
    min_work_rate: Option<f64>,
    max_fallback_rate: Option<f64>,
    max_keep_last_rate: Option<f64>,
    max_p99_wake_latency_ns: Option<u64>,
    max_wake_latency_cv: Option<f64>,
    min_iteration_rate: Option<f64>,
    max_migration_ratio: Option<f64>,
    min_page_locality: Option<f64>,
    max_cross_node_migration_ratio: Option<f64>,
    max_slow_tier_ratio: Option<f64>,
    // -- TopologyConstraints --
    min_numa_nodes: u32,
    min_numa_nodes_set: bool,
    min_llcs: u32,
    min_llcs_set: bool,
    requires_smt: bool,
    requires_smt_set: bool,
    min_cpus: u32,
    min_cpus_set: bool,
    max_llcs: Option<u32>,
    max_llcs_set: bool,
    max_numa_nodes: Option<u32>,
    max_numa_nodes_set: bool,
    max_cpus: Option<u32>,
    max_cpus_set: bool,
    // -- Resource budget: explicit no-perf host-CPU mask size override --
    cpu_budget: Option<u32>,
    // -- Bool attrs (per BOOL_ATTR_NAMES) --
    auto_repro: bool,
    auto_repro_set: bool,
    expect_auto_repro: bool,
    expect_auto_repro_set: bool,
    performance_mode: bool,
    performance_mode_set: bool,
    no_perf_mode: bool,
    no_perf_mode_set: bool,
    expect_err: bool,
    expect_err_set: bool,
    allow_inconclusive: bool,
    allow_inconclusive_set: bool,
    host_only: bool,
    host_only_set: bool,
    ignore_test: bool,
    kaslr: bool,
    kaslr_set: bool,
    wprof: bool,
    wprof_set: bool,
    num_snapshots: u32,
    num_snapshots_set: bool,
    // -- Strings + tokens --
    extra_sched_args: Vec<String>,
    extra_include_files: Vec<String>,
    workload_root_cgroup: Option<String>,
    wprof_args: Option<proc_macro2::TokenStream>,
    expect_scx_bpf_error_contains_tokens: Option<proc_macro2::TokenStream>,
    expect_scx_bpf_error_matches_tokens: Option<proc_macro2::TokenStream>,
    config_expr: Option<proc_macro2::TokenStream>,
    config_set: bool,
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
            post_vm: None,
            post_vm_unconditional: None,
            disk: None,
            // Assert overrides
            not_starved: None,
            isolation: None,
            max_gap_ms: None,
            max_spread_pct: None,
            max_imbalance_ratio: None,
            max_local_dsq_depth: None,
            fail_on_stall: None,
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
            no_perf_mode: false,
            no_perf_mode_set: false,
            expect_err: false,
            expect_err_set: false,
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
            "not_starved" => self.not_starved = Some(value),
            "isolation" => self.isolation = Some(value),
            "performance_mode" => {
                self.performance_mode = value;
                self.performance_mode_set = true;
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
            "allow_inconclusive" => {
                self.allow_inconclusive = value;
                self.allow_inconclusive_set = true;
            }
            "fail_on_stall" => self.fail_on_stall = Some(value),
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

/// Attribute macro that registers a function as a ktstr integration test.
///
/// The annotated function must have signature `fn(&ktstr::scenario::Ctx) ->
/// anyhow::Result<ktstr::assert::AssertResult>`. The macro:
///
/// 1. Renames the original function to `__ktstr_inner_{name}`.
/// 2. Registers it in the `KTSTR_TESTS` distributed slice via linkme.
/// 3. Emits a `#[test]` wrapper that boots a VM and runs the function
///    inside it.
///
/// Every attribute is optional. Most take a `key = value` form; the
/// thirteen boolean attributes (`auto_repro`, `not_starved`, `isolation`,
/// `performance_mode`, `no_perf_mode`, `requires_smt`, `expect_err`,
/// `allow_inconclusive`, `fail_on_stall`, `host_only`, `ignore`, `kaslr`,
/// `wprof`) also accept a bare form as shorthand for `= true` — e.g.
/// `#[ktstr_test(host_only)]` is equivalent to
/// `#[ktstr_test(host_only = true)]`. Of the thirteen, `auto_repro`
/// and `kaslr` are the two whose default is `true`, so the bare form
/// is a no-op; `auto_repro = false` / `kaslr = false` are the only
/// way to disable each. The other eleven default to `false`, so the
/// bare form is the meaningful shorthand.
///
/// The accepted attributes and their defaults are the fields of
/// `ktstr::test_support::KtstrTestEntry` (runtime metadata) and
/// `ktstr::assert::Assert` (checking thresholds). A few are
/// worth calling out because their names differ from the underlying
/// field or because they have nontrivial defaults:
///
///   - `llcs = N` — number of LLCs (default: inherited from
///     scheduler, or 1).
///   - `cores = N` (default: inherited from scheduler, or 2)
///   - `threads = N` (default: inherited from scheduler, or 1)
///   - `numa_nodes = N` (default: inherited from scheduler, or 1)
///   - `memory_mib = N` — per-test minimum memory in MiB (default:
///     2048). The framework picks `max(total_cpus * 64, 256,
///     memory_mib)` MiB at VM-launch time, so for tests with more
///     than 32 vCPUs the cpu-based floor dominates the macro
///     default. Below ~4 vCPUs the absolute 256-MiB floor wins if
///     `memory_mib` is also below it. Setting `memory_mib` above
///     the cpu-based floor is only meaningful when the test needs
///     more headroom than the per-cpu budget. The unit is binary
///     mebibytes; the conversion at VM-launch is `value << 20`
///     bytes, not decimal megabytes.
///   - `duration_s = N` — scenario run duration in seconds; maps
///     onto `KtstrTestEntry::duration`
///   - `watchdog_timeout_s = N` — watchdog fire threshold in
///     seconds; maps onto `KtstrTestEntry::watchdog_timeout`
///   - `cleanup_budget_ms = N` — sub-watchdog cap on host-side VM
///     teardown wall time; maps onto `KtstrTestEntry::cleanup_budget`
///     as `Duration::from_millis(N)`. Default: `None` (unenforced).
///   - `num_snapshots = N` — fire `N` periodic
///     `freeze_and_capture(false)` boundaries inside the workload's
///     10 %–90 % window, stored on the host
///     `SnapshotBridge` under `periodic_NNN`. `0` (default)
///     disables periodic capture entirely. Maps onto
///     `KtstrTestEntry::num_snapshots`; runtime
///     `KtstrTestEntry::validate` rejects values past the bridge
///     cap (`MAX_STORED_SNAPSHOTS`), `host_only = true`, and
///     duration / `N` settings that would land boundaries closer
///     than 100 ms apart.
///   - `scheduler = PATH` — path to a `const Scheduler` (typically
///     produced by `declare_scheduler!(...)`). Maps onto
///     `KtstrTestEntry::scheduler`, which is typed
///     `&'static Scheduler`. Default: `&Scheduler::EEVDF`, the
///     no-scx placeholder that runs under the kernel's default
///     scheduler.
///   - `payload = PATH` — path to a `const Payload` used as the
///     primary binary workload (must be `PayloadKind::Binary`;
///     runtime-enforced). Default: `None` (scheduler-only test).
///     Coexists with `scheduler = PATH` — the payload runs *under*
///     the selected scheduler.
///   - `workloads = [PATH, PATH, ...]` — additional `const Payload`
///     references composed with the primary via `Ctx::payload` in
///     the test body. Default: `&[]`. Must not contain the same
///     path as `payload` — reject at expansion time to catch the
///     common "fio as primary AND workload" slip.
///   - `auto_repro = bool` (default: `true`)
///   - `wprof = bool` (default: `false`; requires the `wprof`
///     cargo feature on ktstr) — attach `/bin/wprof` to the
///     test's VM(s) and ship the Perfetto trace to the host.
///   - `wprof_args = "..."` (requires the `wprof` cargo feature)
///     — override `WprofConfig::default_args`. Only meaningful
///     with `wprof = true`. Parsed as space-separated tokens.
///   - `host_only = bool` (default: `false`) — run the test function
///     on the host instead of inside a VM
///   - `no_perf_mode = bool` (default: `false`) — decouple the
///     virtual topology from host hardware. The VM is built with
///     the declared `numa_nodes` / `llcs` / `cores` / `threads`
///     even on smaller hosts; vCPU pinning, hugepages, NUMA mbind,
///     RT scheduling, and KVM exit suppression are skipped, and
///     gauntlet preset filtering relaxes host-topology checks
///     to the single "host has enough total CPUs" inequality.
///     Mutually exclusive with `performance_mode = true`. Maps onto
///     `KtstrTestEntry::no_perf_mode`.
///   - `post_vm = PATH` — host-side callback invoked after
///     `vm.run()` returns, with access to the full `VmResult`.
///     Use for assertions that need host-side state — e.g.
///     draining `VmResult.snapshot_bridge` after a snapshot
///     capture pipeline fires inside the guest. The function
///     must have signature
///     `fn(&ktstr::vmm::VmResult) -> anyhow::Result<()>`. PATH
///     accepts any Rust path-expression that resolves to a value
///     of that fn type — both free-function refs
///     (`my_post_vm_check`) AND UFCS method refs
///     (`VmResult::assert_wprof_pb_landed`) work via Rust's
///     function-item-to-fn-pointer coercion, so a method that
///     already has the right `&self -> Result<()>` shape can be
///     pointed at directly without wrapping in a one-line
///     delegating free fn.
///     SUPPRESSED on guest-reported fail — see
///     `post_vm_unconditional` for the always-runs sibling.
///     Default: `None` (no callback).
///   - `post_vm_unconditional = PATH` — host-side callback that
///     always runs after `vm.run()` returns, bypassing the
///     guest-fail suppression that gates `post_vm`. Same
///     signature as `post_vm` and PATH accepts the same form:
///     any Rust path-expression resolving to a value of that fn
///     type — both free-function refs AND UFCS method refs
///     (`VmResult::assert_wprof_pb_landed`-style) work via
///     function-item-to-fn-pointer coercion. Use when the
///     callback must observe host-side state regardless of
///     guest-side outcome (e.g. verifying a sidecar artifact
///     landed even when the guest reported a deliberate fail).
///     The callback is responsible for guarding against missing
///     state when the scheduler crashed before producing it —
///     the canonical guard is
///     `if !result.success { return Ok(()); }` at the top of the
///     callback body. Setting `post_vm_unconditional` does NOT
///     invert the test verdict — a guest-reported fail still
///     fails the test even if the unconditional callback returns
///     Ok. Both attributes may be set on the same entry (both
///     errors surface via `combine_post_vm_errs` when both
///     fire). Default: `None` (no callback).
///   - `disk = PATH` — path to a `const DiskConfig` attached to the
///     VM as a virtio-blk device at `/dev/vda`. Construct via
///     `DiskConfig::DEFAULT.with_name("data")` or similar const-fn
///     chain (the `with_name` builder takes `&'static str` so the
///     full expression is const-evaluable). Maps onto
///     `KtstrTestEntry::disk`. Default: `None` (no disk).
///     Mutually exclusive with `host_only = true` —
///     `host_only` skips the VM boot that owns the device lifecycle,
///     so a `disk` attached under `host_only` would never bind;
///     `KtstrTestEntry::validate` rejects the pairing at runtime.
///   - `config = EXPR` — inline scheduler config content, written
///     into the guest at the path declared by the scheduler's
///     `config_file_def`. `EXPR` is either a string literal or a
///     path to a `const &'static str` (e.g. `LAYERED_CONFIG`).
///     Maps onto `KtstrTestEntry::config_content`. Required when
///     the scheduler declares `config_file_def`; rejected when the
///     scheduler does not. The pairing is enforced at compile time
///     via a `const` assertion against `Payload::config_file_def`,
///     and again at runtime by `KtstrTestEntry::validate` so direct
///     programmatic-entry construction sees the same gate.
///   - `expect_scx_bpf_error_contains = EXPR` — literal-substring
///     matcher applied to the captured `scx_bpf_error` text in
///     reproducer mode. `EXPR` is either a string literal or a path
///     to a `const &'static str`. Maps onto
///     `Assert::expect_scx_bpf_error_contains`. Requires
///     `expect_err = true` (rejected at construction otherwise by
///     `KtstrTestEntry::validate`). Empty strings panic at
///     construction. When both `_contains` and `_matches` are set,
///     the evaluator ANDs them — every set matcher must hit.
///   - `expect_scx_bpf_error_matches = EXPR` — regex matcher with
///     the same accepted forms and gating as `_contains`. Maps onto
///     `Assert::expect_scx_bpf_error_matches`. Validated at
///     construction: empty patterns, invalid regex syntax, and any
///     pattern satisfying `is_match("")` all panic immediately. The
///     `is_match("")` predicate catches two no-op classes with one
///     check: patterns that match every position (e.g. `a?`, `.*`,
///     `(?:)`) trivially pass against any corpus, and patterns that
///     match only the empty string (e.g. `^$`) trivially fail
///     against any non-empty corpus — both are equally useless pins.
///     Bare `\b` slips the gate (no word characters in `""`); see
///     `Assert::expect_scx_bpf_error_matches` for the operator
///     direction.
///   - `extra_include_files = ["PATH", "PATH", ...]` — host-side
///     file paths to bundle into the guest initramfs beyond what
///     the entry's `scheduler` / `payload` / `workloads` already
///     declare via their own `include_files`. Use this for
///     test-level dependencies that don't belong on a specific
///     Payload: auxiliary data files, per-test helper scripts,
///     fixtures. Each element must be a string literal (no
///     expressions). Maps onto
///     `KtstrTestEntry::extra_include_files` and is unioned with
///     the per-payload specs at `run_ktstr_test` time via
///     `KtstrTestEntry::all_include_files`. Default: `[]`.
///     Path resolution: bare names (no `/`) search `PATH`; paths
///     containing `/` are absolute or relative to the test process
///     current directory; directories are walked recursively at
///     test-run time (rejected by `cargo ktstr export` since the
///     `.run` packager handles regular files only — recursive
///     directory packaging is a v2 enhancement); a missing file
///     fails loudly at setup with an actionable error naming the
///     missing path.
///
/// Duplicate keys: each attribute KEY may appear at most once per
/// `#[ktstr_test]` invocation; duplicate keys (whether the values
/// match or differ) fail at expansion rather than silently letting
/// the later value win. `#[ktstr_test(host_only = false,
/// host_only)]` and `#[ktstr_test(llcs = 4, llcs = 8)]` both fail.
/// The bare form (`host_only`) and explicit form (`host_only =
/// true`) of the same attribute collide as well — they refer to
/// the same slot. List values like `workloads = [FIO, FIO]` are
/// NOT affected by this rule; the duplicate check is on attribute
/// keys, not on values within an array. `payload = ...` and
/// `workloads = [..]` keep their tailored messages directing the
/// author to the right home for extras; `config = ...` and
/// `expect_scx_bpf_error_{contains,matches} = ...` likewise have
/// tailored wording; every other attribute uses a uniform
/// "duplicate attribute" diagnostic.
///
/// Path / list forms: `#[ktstr_test(crate::host_only)]` (a
/// multi-segment path, whether bare or as a key in
/// `crate::host_only = true`) is rejected with a targeted message
/// naming both valid forms with concrete examples — the macro only
/// accepts bare single-segment idents because routing dispatches on
/// the ident string against `BOOL_ATTR_NAMES` or the value-attr
/// table. `#[ktstr_test(host_only(false))]` (parenthesised
/// arguments) is rejected with a separate targeted message naming
/// the attribute and the two valid forms (`= value` or bare); the
/// same diagnostic fires for `crate::host_only(false)` so the
/// operator sees one combined error rather than chasing two.
#[proc_macro_attribute]
pub fn ktstr_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    match ktstr_test_impl(attr.into(), item.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
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
    // runtime in `src/test_support/entry.rs::validate`.
    if attrs.cpu_budget.is_some() && !(attrs.no_perf_mode_set && attrs.no_perf_mode) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cpu_budget requires no_perf_mode — the budget sizes the \
             no-perf vCPU-thread mask; under performance_mode vCPUs are \
             pinned 1:1 and cpu_budget would be silently ignored. Add \
             no_perf_mode (or drop cpu_budget).",
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
    // `src/test_support/entry.rs::validate`):
    //   - host_only + disk         : runtime-only (macro doesn't
    //                                 expose `disk` attribute)
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
             entry.rs::validate enforces the same invariant for \
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
fn validate_payload_workloads_dedup(
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

/// Inner implementation of [`ktstr_test`] operating on `proc_macro2::TokenStream`
/// so unit tests in this crate can synthesize input + assert output without
/// the `proc_macro` runtime context (`proc_macro` types panic outside a
/// procedural-macro invocation). The proc-macro entry above is a thin wrapper
/// that converts `proc_macro::TokenStream` ↔ `proc_macro2::TokenStream` and
/// projects `Err` to the compile-error token stream.
fn ktstr_test_impl(
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
                        attrs.bpf_map_write = Some(expect_path_value(
                            value,
                            "expected path for bpf_map_write (e.g. BPF_CRASH)",
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
                        attrs
                            .extra_sched_args
                            .extend(expect_array_of_string_literals(
                                value,
                                "expected array of string literals for extra_sched_args",
                                "expected string literal in extra_sched_args",
                            )?);
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
                        return Err(syn::Error::new_spanned(
                            path,
                            format!(
                                "unknown attribute `{ident}`, expected: llcs, cores, threads, numa_nodes, memory_mib, scheduler, staged_schedulers, payload, workloads, auto_repro, not_starved, isolation, max_gap_ms, max_spread_pct, max_throughput_cv, min_work_rate, max_p99_wake_latency_ns, max_wake_latency_cv, min_iteration_rate, max_migration_ratio, max_imbalance_ratio, max_local_dsq_depth, fail_on_stall, sustained_samples, max_fallback_rate, max_keep_last_rate, min_page_locality, max_cross_node_migration_ratio, max_slow_tier_ratio, expect_scx_bpf_error_contains, expect_scx_bpf_error_matches, extra_sched_args, extra_include_files, min_numa_nodes, min_llcs, requires_smt, min_cpus, max_llcs, max_numa_nodes, max_cpus, cpu_budget, watchdog_timeout_s, performance_mode, no_perf_mode, duration_s, bpf_map_write, expect_err, allow_inconclusive, host_only, ignore, cleanup_budget_ms, post_vm, post_vm_unconditional, config, disk, num_snapshots, wprof, wprof_args"
                            ),
                        ));
                    }
                }
            }
            Meta::Path(p) => {
                // Sugar: a bare bool attr (e.g. `#[ktstr_test(host_only)]`)
                // is equivalent to `key = true`. Only the ten bool
                // attributes accept this form; bare ints/floats/paths
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

    Ok(emit_entry_static(input, attrs))
}

/// Emit the linkme-registered static entry + #[test] wrapper for a
/// `#[ktstr_test]`-annotated function. Takes the parsed function and
/// the parsed-and-validated [`AttrValues`]; returns the token stream
/// the macro expands to. The destructure of `attrs` lives inside this
/// function because the codegen below interpolates each field via
/// `quote!{ #foo }`, which requires a single binding in scope — see
/// the docstring on [`AttrValues`] for the full rationale.
fn emit_entry_static(input: ItemFn, attrs: AttrValues) -> proc_macro2::TokenStream {
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

    // Both expect_err and expect_ok test bodies share the same
    // skip-handling arms (harness-not-configured and resource-
    // contention bailouts) — they only differ on the success arm
    // and the unconditional-Err arm. Factor the shared arms into
    // one TokenStream so a future change to skip semantics lands
    // in one place and both branches inherit it.
    let skip_arms = quote! {
        Err(e) if ::ktstr::test_support::is_kernel_unavailable(&e) => {
            // Harness not configured (no kernel resolved): the
            // binary was likely invoked outside `cargo ktstr test`,
            // which builds and injects a kernel automatically.
            // Skip cleanly so a developer running `cargo nextest
            // run` directly sees a SKIP banner rather than a
            // confusing "no kernel found" panic. Applies in both
            // the expect_err and expect_ok directions — an
            // expect_err test that never ran is also a non-failure.
            eprintln!("ktstr: SKIP: harness not configured: {e:#}");
            return;
        }
        Err(e) if ::ktstr::test_support::is_resource_contention(&e) => {
            // Resource contention is host-infra, not a test
            // outcome: emit the canonical SKIP banner and early-
            // return so libtest sees pass. The skip sidecar is
            // recorded inside `run_ktstr_test_inner` at every
            // contention site, so stats tooling still sees the
            // skip without a panic-driven retry. For expect_err
            // tests this also prevents host contention from
            // masquerading as the expected failure (a false-
            // positive pass for the wrong reason).
            //
            // KTSTR_NO_SKIP_MODE inverts the policy: CI runs that
            // demand every test execute against the available
            // hardware promote contention to a hard failure so a
            // misconfigured host surfaces instead of silently
            // passing.
            if ::std::env::var_os("KTSTR_NO_SKIP_MODE").is_some() {
                panic!(
                    "ktstr: FAIL: resource contention under --no-skip-mode: {e:#}. \
                     Either provision hardware that satisfies the test's topology \
                     requirement, or drop --no-skip-mode / KTSTR_NO_SKIP_MODE to \
                     accept the skip."
                );
            }
            eprintln!("ktstr: SKIP: resource contention: {e:#}");
            return;
        }
    };
    let test_body = if expect_err {
        quote! {
            match ::ktstr::test_support::run_ktstr_test(&#entry_name) {
                Ok(_) => panic!("expected test to fail but it passed"),
                #skip_arms
                Err(_) => {}
            }
        }
    } else {
        quote! {
            match ::ktstr::test_support::run_ktstr_test(&#entry_name) {
                Ok(_) => {}
                #skip_arms
                Err(e) => panic!("{e:#}"),
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
            #allow_inconclusive_field
            #host_only_field
            #extra_include_files_field
            #cleanup_budget_field
            #post_vm_field
            #post_vm_unconditional_field
            #config_content_field
            #disk_field
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

/// Convert a CamelCase identifier to SCREAMING_SNAKE_CASE.
///
/// Handles acronyms (consecutive uppercase): a separator is inserted
/// before the last letter of a run when followed by lowercase.
///
/// `Llc` -> `"LLC"`, `RejectPin` -> `"REJECT_PIN"`, `NoCtrl` -> `"NO_CTRL"`,
/// `LLC` -> `"LLC"`, `HTTPServer` -> `"HTTP_SERVER"`.
fn camel_to_screaming_snake(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        if ch.is_uppercase() && i > 0 {
            let prev_upper = chars[i - 1].is_uppercase();
            let next_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());
            if !prev_upper || next_lower {
                out.push('_');
            }
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

/// Function-style macro that registers a `Scheduler` const.
///
/// # Syntax
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// declare_scheduler!(MITOSIS, {
///     name = "mitosis",
///     binary = "scx_mitosis",
///     cgroup_parent = "/ktstr",
///     sched_args = ["--exit-dump-len", "1048576"],
///     kernels = ["6.14", "7.0..=7.2"],
///     constraints = TopologyConstraints {
///         min_llcs: 1, max_llcs: Some(8), max_cpus: Some(64),
///         ..TopologyConstraints::DEFAULT
///     },
/// });
/// ```
///
/// # Generated items
///
/// Given `declare_scheduler!(MITOSIS, { ... })`:
///
/// - `pub static MITOSIS: ::ktstr::test_support::Scheduler` — the declared
///   scheduler value. No `_PAYLOAD` suffix; the const IS the
///   `Scheduler`.
/// - A hidden `static __KTSTR_SCHED_REG_MITOSIS: &'static Scheduler`
///   registered in `KTSTR_SCHEDULERS` (`ktstr::test_support::KTSTR_SCHEDULERS`)
///   via linkme so the verifier can discover the declaration by
///   spawning the test binary with `--ktstr-list-schedulers`.
///
/// # Visibility prefix
///
/// An optional Rust visibility prefix may precede the const name:
///
/// ```rust,ignore
/// declare_scheduler!(MY_SCHED, { ... });             // defaults to `pub`
/// declare_scheduler!(pub MY_SCHED, { ... });          // explicit `pub`
/// declare_scheduler!(pub(crate) MY_SCHED, { ... });   // crate-local
/// declare_scheduler!(pub(super) MY_SCHED, { ... });   // parent-module
/// declare_scheduler!(pub(in crate::test_support) MY_SCHED, { ... });
/// ```
///
/// Omitting the prefix defaults to `pub` — schedulers are normally
/// public so the verifier and other crates can reference them; an
/// explicit prefix is needed only when the declaration sits inside
/// a module that wants to narrow the exposed name. (Field content
/// shown above as `{ ... }` is elided; consult the Syntax example
/// for the required fields.) The hidden registry static (see
/// Generated items above) is always `static` (private) regardless
/// of the user-facing const's visibility — `linkme` gathers it via
/// link-section walking, not Rust name resolution, so the slice
/// mechanism works at every visibility level.
///
/// # Accepted fields
///
/// Exactly one scheduler-source must be declared: `binary`,
/// `binary_path`, or the `kernel_builtin_enable` + `kernel_builtin_disable`
/// pair. The three options select between the matching
/// `SchedulerSpec` variants. To run under the kernel default
/// instead, reference `ktstr::test_support::Scheduler::EEVDF`
/// directly rather than declaring a new scheduler.
///
/// | Field | Required | Description |
/// |---|---|---|
/// | `name = "..."` | yes | Scheduler name (sidecar / logs). |
/// | `binary = "..."` | one source | Binary name → `SchedulerSpec::Discover(...)`. Matched against `[[bin]]` names in `target/{debug,release}/`, the test binary's directory, or `KTSTR_SCHEDULER` env var. Often equal to the cargo package name but not required to be. |
/// | `binary_path = "/abs/path"` | one source | Absolute filesystem path → `SchedulerSpec::Path(...)`. The runtime does not auto-build this variant: the file must already exist at the path when the test runs. Use for prebuilt binaries that live outside the cargo discovery cascade. Macro-time validation rejects empty strings, relative paths, and `~`-prefixed paths (no compile-time tilde expansion); existence is the runtime's job. |
/// | `kernel_builtin_enable = [..]` + `kernel_builtin_disable = [..]` | one source | Two string-array literals that together select `SchedulerSpec::KernelBuiltin { enable: &[..], disable: &[..] }`. The framework writes the enable commands to the guest's `/sched_enable` and the disable commands to `/sched_disable` (see `src/vmm/initramfs.rs`), and the guest interpreter runs each entry once at scenario start / teardown. Both fields must be set together — setting only one is rejected. The interpreter (`src/vmm/rust_init.rs`) accepts EXACTLY ONE shell-line shape: `echo VALUE > /path` (plus blank lines and `#` comments). Pipes, `>>`, `;`, variable expansion, and any other syntax silently no-ops at runtime, so the macro rejects entries that don't match `echo … > /…` at expand time. At least one of the two arrays must be non-empty: a pair that supplies neither enable nor disable commands is equivalent to the EEVDF baseline — reference `Scheduler::EEVDF` for that. Note: `cargo ktstr export` currently bails on KernelBuiltin schedulers (`src/export.rs`); declarations using this variant cannot be reproduced via the export-to-shar workflow until that limitation is lifted. |
/// | `topology = (numa, llcs, cores, threads)` | no | Default VM topology. Default: `(1, 1, 2, 1)` (from `Scheduler::named`). Validated at compile time: each value must be non-zero, and `llcs` must be a multiple of `numa`. |
/// | `cgroup_parent = "..."` | no | Cgroup parent path (must begin with `/`). |
/// | `sched_args = [..]` | no | Scheduler CLI args prepended before per-test `extra_sched_args`. |
/// | `sysctls = [Sysctl::new("k", "v"), ..]` | no | Guest sysctls. |
/// | `kargs = [..]` | no | Extra guest kernel cmdline args. |
/// | `kernels = ["6.14", "7.0..=7.2", ..]` | no | Kernel specs the verifier sweeps. Same parser as the `--kernel` CLI flag — accepts exact versions, ranges (`..` or `..=`, both inclusive), git refs (`git+URL#REF`), paths, and cache keys. Each entry is validated at macro-expand time via the same `KernelId::parse` + `validate` the verifier uses at runtime; empty entries, inverted ranges, and `..`-containing strings whose endpoints aren't version-shaped (e.g. `"abc..def"`) are rejected. |
/// | `constraints = TopologyConstraints { .. }` | no | Gauntlet preset constraints — maps directly onto `Scheduler::constraints`. Filters which gauntlet topology presets exercise this scheduler. When given as a struct literal, the macro additionally cross-checks each literal field against the effective topology (explicit `topology` field if present, otherwise the `(1, 1, 2, 1)` default from `Scheduler::named`) and rejects infeasible pairings; non-struct-literal forms (e.g. `OTHER::CONST_CONSTRAINTS`) skip that check. |
/// | `assert = Assert::NO_OVERRIDES.method().chain()` | no | Scheduler-wide assertion overrides — maps directly onto `Scheduler::assert`. Merged with `Assert::default_checks()` and the per-test `assert` at runtime (`default ← scheduler ← per-test`). Accepts any const-evaluable expression: a const path like `Assert::NO_OVERRIDES`, a const-fn call like `Assert::default_checks()`, or a chain of const-fn setters like `Assert::NO_OVERRIDES.check_not_starved().max_gap_ms(50)`. The macro accepts MethodCall chains and Path-rooted (type/module-prefixed) Calls — only bare single-segment lowercase Calls like `helper()` are rejected as non-const free-fn patterns; non-const methods on a Path receiver slip through and surface as a deep const-eval failure at the spread site. |
/// | `config_file = "..."` | no | Host-side config file path. |
/// | `config_file_def = ("--config {file}", "/include-files/cfg.json")` | no | Inline-config plumbing — maps directly onto `Scheduler::config_file_def`. 2-tuple of string literals: arg_template (CLI arg with `{file}` placeholder substituted at run time) and guest_path (absolute path where the framework writes the JSON inside the guest). Distinct from `config_file` (which references a pre-existing host file). The macro validates: tuple-arity = 2, both elements non-empty string literals, `{file}` placeholder present in arg_template, guest_path absolute. |
///
/// # Const naming rules
///
/// The first argument must be a SCREAMING_SNAKE_CASE identifier and
/// must NOT be one of the reserved built-in names (`EEVDF`,
/// `KERNEL_DEFAULT`). The macro emits a `compile_error!` if either rule
/// is violated.
#[proc_macro]
pub fn declare_scheduler(input: TokenStream) -> TokenStream {
    match declare_scheduler_inner(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn declare_scheduler_inner(
    input: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    struct DeclareSchedulerInput {
        visibility: syn::Visibility,
        const_name: syn::Ident,
        fields: Vec<(syn::Ident, syn::Expr)>,
    }

    impl syn::parse::Parse for DeclareSchedulerInput {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            // Optional visibility prefix: `pub`, `pub(crate)`,
            // `pub(super)`, `pub(in path)`, or none. `syn::Visibility`
            // returns `Visibility::Inherited` when no prefix is given;
            // the emit treats this as the default-pub case so call
            // sites without a visibility prefix produce `pub static`.
            let visibility: syn::Visibility = input.parse()?;
            let const_name: syn::Ident = input.parse()?;
            // Detect the trailing-visibility shape (`MY_SCHED pub,`)
            // before the standard `,` parse emits the generic
            // "expected `,`" diagnostic. A `pub` token in this
            // position is always a misplaced visibility prefix —
            // surface that explicitly so the operator knows where
            // it should go.
            if input.peek(syn::Token![pub]) {
                return Err(syn::Error::new(
                    input.span(),
                    "declare_scheduler!: visibility prefix must precede \
                     the const name (e.g. `pub MY_SCHED`, `pub(crate) MY_SCHED`), \
                     not follow it. Move the `pub` token to before \
                     the const name.",
                ));
            }
            let _: syn::Token![,] = input.parse()?;
            let body;
            syn::braced!(body in input);
            let mut fields = Vec::new();
            while !body.is_empty() {
                let key: syn::Ident = body.parse()?;
                let _: syn::Token![=] = body.parse()?;
                let value: syn::Expr = body.parse()?;
                fields.push((key, value));
                if body.peek(syn::Token![,]) {
                    let _: syn::Token![,] = body.parse()?;
                }
            }
            Ok(DeclareSchedulerInput {
                visibility,
                const_name,
                fields,
            })
        }
    }

    let DeclareSchedulerInput {
        visibility,
        const_name,
        fields,
    } = syn::parse2(input)?;

    // Validate const name: SCREAMING_SNAKE_CASE + not reserved.
    let const_name_str = const_name.to_string();
    if const_name_str != const_name_str.to_uppercase() {
        return Err(syn::Error::new(
            const_name.span(),
            format!(
                "declare_scheduler!: const name `{const_name_str}` must be SCREAMING_SNAKE_CASE"
            ),
        ));
    }
    // Reserve the const names that match the built-in `Scheduler::EEVDF`
    // and `Payload::KERNEL_DEFAULT` baselines so user code cannot shadow
    // either symbol. Match by exact identifier — the spelling is
    // case-sensitive in Rust so the lowercase form (e.g. `eevdf`) is
    // already rejected by the SCREAMING_SNAKE_CASE check above. The
    // companion string-name reservation (handled on the `name = "..."`
    // arm below) is case-insensitive because wire names typically
    // lowercase.
    match const_name_str.as_str() {
        "EEVDF" => {
            return Err(syn::Error::new(
                const_name.span(),
                format!(
                    "declare_scheduler!: const name `{const_name_str}` is reserved \
                     for the built-in Scheduler::EEVDF baseline; pick a different identifier"
                ),
            ));
        }
        "KERNEL_DEFAULT" => {
            return Err(syn::Error::new(
                const_name.span(),
                format!(
                    "declare_scheduler!: const name `{const_name_str}` is reserved \
                     for the built-in Payload::KERNEL_DEFAULT baseline; pick a different identifier"
                ),
            ));
        }
        _ => {}
    }

    // Parse fields.
    let mut sched_name: Option<String> = None;
    let mut sched_binary: Option<String> = None;
    let mut sched_binary_path: Option<String> = None;
    let mut sched_kernel_builtin_enable: Option<Vec<String>> = None;
    let mut sched_kernel_builtin_disable: Option<Vec<String>> = None;
    let mut sched_kernel_builtin_enable_span: Option<proc_macro2::Span> = None;
    let mut sched_kernel_builtin_disable_span: Option<proc_macro2::Span> = None;
    let mut sched_topology: Option<(u32, u32, u32, u32)> = None;
    let mut sched_cgroup_parent: Option<String> = None;
    let mut sched_args: Vec<String> = Vec::new();
    let mut sched_args_set = false;
    let mut sched_sysctls: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut sched_sysctls_set = false;
    let mut sched_kargs: Vec<String> = Vec::new();
    let mut sched_kargs_set = false;
    let mut sched_kernels: Vec<String> = Vec::new();
    let mut sched_kernels_set = false;
    let mut sched_constraints: Option<syn::Expr> = None;
    let mut sched_config_file: Option<String> = None;
    let mut sched_assert: Option<syn::Expr> = None;
    let mut sched_config_file_def: Option<(String, String)> = None;
    // Carry the `name = "..."` expression out of the loop so the
    // post-loop reservation checks can point the caret at the
    // literal via `new_spanned` (mirroring the inline `name` arm's
    // span style) instead of the const-name fallback.
    let mut sched_name_expr: Option<syn::Expr> = None;

    let mut seen_fields = std::collections::HashSet::<String>::new();
    for (key, value) in fields {
        let key_str = key.to_string();
        if !seen_fields.insert(key_str.clone()) {
            return Err(syn::Error::new(
                key.span(),
                format!("declare_scheduler!: duplicate field `{key_str}`"),
            ));
        }
        match key_str.as_str() {
            "name" => {
                let lit = expect_str_lit(&value, &key, "name")?;
                // Reject empty, whitespace-only, and invisible-only names
                // inline so the caret lands on the offending literal via
                // new_spanned. `str::trim` strips Unicode White_Space but
                // not Cf-category invisibles (ZWSP, BOM, etc.), so an
                // operator copy-pasting a string with stray invisible
                // chars could slip past `trim().is_empty()` and surface
                // confusingly at runtime sidecar lookup. See
                // [`check_visible_lit`] / [`is_visibly_empty`].
                check_visible_lit(&lit, &value, "name")?;
                // Mirror the const-name reservation above: the string
                // names of the built-in `Scheduler::EEVDF` (`"eevdf"`)
                // and `Payload::KERNEL_DEFAULT` (`"kernel_default"`)
                // cannot be reused. A `declare_scheduler!` whose
                // `name = "eevdf"` would silently shadow the baseline
                // in `find_scheduler` lookups and sidecar comparisons.
                // Trim+lowercase before the match so both whitespace
                // padding (`"  eevdf  "`) and case (`"EEVDF"`, `"Eevdf"`)
                // resolve to the same canonical reserved keyword —
                // the reservation's intent is "don't shadow the baselines"
                // regardless of how the user formatted the literal.
                let lit_lower = lit.trim().to_lowercase();
                if matches!(lit_lower.as_str(), "eevdf" | "kernel_default") {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "declare_scheduler!: `name = \"{lit}\"` is reserved \
                             for the built-in {} baseline; pick a different name",
                            if lit_lower == "eevdf" {
                                "Scheduler::EEVDF"
                            } else {
                                "Payload::KERNEL_DEFAULT"
                            }
                        ),
                    ));
                }
                sched_name = Some(lit);
                sched_name_expr = Some(value);
            }
            "binary" => {
                let lit = expect_str_lit(&value, &key, "binary")?;
                // Reject empty, whitespace-only, and invisible-only
                // binary names — all flow into `SchedulerSpec::Discover(...)`
                // and fail confusingly at runtime inside
                // `build_and_find_binary`. See [`check_visible_lit`] /
                // [`is_visibly_empty`] for the invisible-char (ZWSP,
                // BOM, etc.) rejection rationale.
                check_visible_lit(&lit, &value, "binary")?;
                sched_binary = Some(lit);
            }
            "binary_path" => {
                let lit = expect_str_lit(&value, &key, "binary_path")?;
                // Reject empty, whitespace-only, and invisible-only
                // paths via the same [`check_visible_lit`] /
                // [`is_visibly_empty`] predicate the `name` and `binary`
                // arms use. Anything else falls through to the `~` /
                // relative-path checks below; an invisible-only string
                // would otherwise surface as "must be absolute" instead
                // of naming the actual root cause.
                check_visible_lit(&lit, &value, "binary_path")?;
                // Tilde expansion does not happen at compile time and
                // the runtime does not expand it either — `path.exists()`
                // checks the literal `~/foo` against the filesystem, which
                // never matches. Reject up-front with the actionable fix.
                if lit.starts_with('~') {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "declare_scheduler!: `binary_path = \"{lit}\"` \
                             starts with `~` — tilde paths are not expanded \
                             at compile time or by the runtime. Use an \
                             absolute path (e.g. `\"/home/user/bin/scx_foo\"`)."
                        ),
                    ));
                }
                // Relative paths are ambiguous between "sibling file" and
                // "discover-by-name" intent. Force the operator to commit:
                // if they want discovery, use `binary = "name"`; if they
                // want a specific file, write the absolute path.
                if !lit.starts_with('/') {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "declare_scheduler!: `binary_path = \"{lit}\"` \
                             must be absolute (start with `/`). For \
                             discovery-by-name, use `binary = \"...\"` \
                             instead; for a specific file, write the \
                             absolute path."
                        ),
                    ));
                }
                sched_binary_path = Some(lit);
            }
            "kernel_builtin_enable" => {
                sched_kernel_builtin_enable_span = Some(key.span());
                sched_kernel_builtin_enable = Some(parse_kernel_builtin_cmds(
                    &value,
                    &key,
                    "kernel_builtin_enable",
                    "enable",
                )?);
            }
            "kernel_builtin_disable" => {
                sched_kernel_builtin_disable_span = Some(key.span());
                sched_kernel_builtin_disable = Some(parse_kernel_builtin_cmds(
                    &value,
                    &key,
                    "kernel_builtin_disable",
                    "disable",
                )?);
            }
            "topology" => {
                if let syn::Expr::Tuple(t) = &value {
                    let mut parts = [0u32; 4];
                    if t.elems.len() != 4 {
                        return Err(syn::Error::new_spanned(
                            t,
                            "topology must be a 4-tuple (numa_nodes, llcs, cores, threads)",
                        ));
                    }
                    for (i, e) in t.elems.iter().enumerate() {
                        parts[i] = expect_u32_lit(e, &key, "topology")?;
                    }
                    const DIM_NAMES: [&str; 4] = ["numa_nodes", "llcs", "cores", "threads"];
                    for (i, &v) in parts.iter().enumerate() {
                        if v == 0 {
                            return Err(syn::Error::new_spanned(
                                t,
                                format!(
                                    "topology ({}, {}, {}, {}): {} must be > 0",
                                    parts[0], parts[1], parts[2], parts[3], DIM_NAMES[i],
                                ),
                            ));
                        }
                    }
                    if parts[1] % parts[0] != 0 {
                        return Err(syn::Error::new_spanned(
                            t,
                            format!(
                                "topology (numa_nodes={}, llcs={}, cores={}, threads={}): \
                                 llcs must be a multiple of numa_nodes — likely a numa/llcs swap",
                                parts[0], parts[1], parts[2], parts[3],
                            ),
                        ));
                    }
                    sched_topology = Some((parts[0], parts[1], parts[2], parts[3]));
                } else {
                    return Err(syn::Error::new_spanned(
                        &value,
                        "topology must be a tuple expression: topology = (numa_nodes, llcs, cores, threads)",
                    ));
                }
            }
            "cgroup_parent" => {
                let lit = expect_str_lit(&value, &key, "cgroup_parent")?;
                if !lit.starts_with('/') {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "cgroup_parent must begin with `/` (e.g. \"/ktstr\"); \
                             got {lit:?} — relative paths const-panic at CgroupPath::new \
                             during macro expansion",
                        ),
                    ));
                }
                if lit.split('/').any(|seg| seg == "..") {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "cgroup_parent must not contain `..` segments; \
                             got {lit:?} — path traversal would escape the cgroup hierarchy \
                             and the const-panic at CgroupPath::new would point at macro \
                             expansion instead of the literal",
                        ),
                    ));
                }
                sched_cgroup_parent = Some(lit);
            }
            "sched_args" => {
                sched_args_set = true;
                append_str_array_into(&value, &key, "sched_args", &mut sched_args)?;
            }
            "sysctls" => {
                sched_sysctls_set = true;
                let arr = expect_array(&value, &key, "sysctls")?;
                for elem in &arr.elems {
                    sched_sysctls.push(elem.to_token_stream());
                }
            }
            "kargs" => {
                sched_kargs_set = true;
                append_str_array_into(&value, &key, "kargs", &mut sched_kargs)?;
            }
            "kernels" => {
                sched_kernels_set = true;
                let arr = expect_array(&value, &key, "kernels")?;
                for elem in &arr.elems {
                    let s = expect_str_lit_element(elem, "kernels")?;
                    // Empty kernel strings parse as `CacheKey("")` and
                    // fail confusingly at verifier runtime with "cache
                    // key not found". Reject up-front so the diagnostic
                    // lands on the literal in the source.
                    if s.is_empty() {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!(
                                "declare_scheduler!: `kernels` entry must \
                                 be a non-empty string. Accepted forms: {}.",
                                kernel_path::KERNEL_ID_GRAMMAR,
                            ),
                        ));
                    }
                    // Run the same `KernelId::parse` + `validate` the
                    // verifier uses so any malformed entry — inverted
                    // range, suspicious `..` substring that fails the
                    // range grammar — is caught at the call site
                    // rather than as a confusing runtime "cache key
                    // not found" error.
                    let parsed = kernel_path::KernelId::parse(&s);
                    if let Err(msg) = parsed.validate() {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!(
                                "declare_scheduler!: invalid kernel \
                                     spec `{s}`: {msg}"
                            ),
                        ));
                    }
                    // A literal containing `..` that did not classify
                    // as a Range almost always indicates a typo'd
                    // range spec (e.g. `"abc..def"` where neither
                    // endpoint is version-shaped, or `"6.14..xyz"`).
                    // Per-variant disambiguation: `CacheKey("a..b")`
                    // is wrong because cache keys are content-addressed
                    // identifiers that don't carry `..` separators;
                    // `Path("foo/..bar")` is fine (file paths legally
                    // contain `..`) and already matched the Path arm.
                    if s.contains("..") && matches!(parsed, kernel_path::KernelId::CacheKey(_)) {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!(
                                "declare_scheduler!: `kernels` entry `{s}` \
                                 contains `..` but the endpoints aren't both \
                                 version-shaped (`MAJOR.MINOR[.PATCH][-rcN]`). \
                                 If this was meant as a version range, \
                                 fix the endpoints (e.g. `6.14..7.0`). \
                                 If this is a literal cache key, remove \
                                 the `..` — cache keys do not use \
                                 range syntax.",
                            ),
                        ));
                    }
                    sched_kernels.push(s);
                }
            }
            "constraints" => {
                // `constraints` lands in a `pub static`, so the
                // expression must be const-evaluable. Reject the
                // common typo of passing a non-const helper call
                // (`build_constraints()`, `default_topology().min_llcs(4)`,
                // `Foo::derive(...).constraints`) up-front so the
                // diagnostic explains the constraint instead of
                // letting rustc surface a deep, confusing
                // const-eval-failure chain at the spread site.
                //
                // Accepted shapes:
                //   - `TopologyConstraints { ..TopologyConstraints::DEFAULT }`
                //     (struct literal — the canonical form, used by
                //     every in-tree call site)
                //   - `TopologyConstraints { ..TopologyConstraints::new() }`
                //     or `..TopologyConstraints::default()` (struct
                //     literal spreading a const-fn constructor — modern
                //     Rust promotes the temporary when the return type
                //     is trivially-Drop; if the called fn isn't `const
                //     fn`, rustc surfaces a clean E0015 at the spread
                //     site; if the returned type has non-trivial Drop
                //     fields, rustc surfaces a clean E0493 instead)
                //   - `TopologyConstraints::DEFAULT` (path expression
                //     — bare DEFAULT or any other `const` path)
                //   - `( … )` (parenthesized const-eligible expression
                //     — pass-through to the underlying form so a user
                //     who wraps for clarity is not punished)
                //   - reference / unary on top of any accepted form
                //
                // Free-helper calls (`build_helper()`) and unrecognized
                // lowercase methods are rejected with a hint describing
                // the const-eligible alternatives. Method chains are
                // rejected outright in this mode (the canonical pattern
                // is struct literal or const path, not chained setters).
                validate_const_eligible(
                    &value,
                    "constraints",
                    CONSTRAINTS_ACCEPTED_SHAPES,
                    ConstEligibility::StructLiteralOnly,
                )?;
                sched_constraints = Some(value);
            }
            "config_file" => {
                let lit = expect_str_lit(&value, &key, "config_file")?;
                sched_config_file = Some(lit);
            }
            "assert" => {
                // `assert` lands in a `pub static`, so the expression
                // must be const-evaluable. Unlike `constraints`, the
                // canonical Assert pattern is METHOD-CHAINING on const
                // fns (`Assert::NO_OVERRIDES.check_not_starved()...`),
                // so the assert validator accepts MethodCall chains
                // and Path-rooted Calls (`Assert::default_checks()`,
                // `Some(x)`). Only bare single-segment lowercase
                // Calls (`helper()`) are rejected as the free-fn
                // pattern; non-const methods on a Path receiver
                // slip through and surface as a deep const-eval
                // failure at the spread site.
                // See `validate_const_eligible` with
                // `ConstEligibility::AllowConstMethodChains`.
                validate_const_eligible(
                    &value,
                    "assert",
                    ASSERT_ACCEPTED_SHAPES,
                    ConstEligibility::AllowConstMethodChains,
                )?;
                sched_assert = Some(value);
            }
            "config_file_def" => {
                // `config_file_def` is `Option<(arg_template,
                // guest_path)>`. The macro accepts a 2-tuple of string
                // literals and auto-wraps in `Some` via the existing
                // `.config_file_def(arg, path)` builder. Validate at
                // expand time: tuple-arity = 2, each element is a
                // non-empty string literal, arg_template contains the
                // `{file}` placeholder (the runtime substitutes the
                // guest path at that position; a template without it
                // silently fails at dispatch), and guest_path is
                // absolute (the runtime writes the config there, and
                // a relative path breaks the `mkdir -p` invariant).
                let tup = if let syn::Expr::Tuple(t) = &value {
                    t
                } else {
                    return Err(syn::Error::new_spanned(
                        &value,
                        "declare_scheduler!: `config_file_def` must be a \
                         2-tuple of string literals: `(arg_template, guest_path)`. \
                         Example: `(\"--config {file}\", \"/include-files/cfg.json\")`.",
                    ));
                };
                if tup.elems.len() != 2 {
                    return Err(syn::Error::new_spanned(
                        tup,
                        format!(
                            "declare_scheduler!: `config_file_def` must be a \
                             2-tuple of string literals (`(arg_template, guest_path)`), \
                             got {}-tuple.",
                            tup.elems.len()
                        ),
                    ));
                }
                let arg_template = expect_str_lit_element(&tup.elems[0], "config_file_def")?;
                let guest_path = expect_str_lit_element(&tup.elems[1], "config_file_def")?;
                if arg_template.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[0],
                        "declare_scheduler!: `config_file_def` arg_template \
                         (element 0) must be a non-empty string. Example: \
                         `\"--config {file}\"`.",
                    ));
                }
                if guest_path.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[1],
                        "declare_scheduler!: `config_file_def` guest_path \
                         (element 1) must be a non-empty string. Example: \
                         `\"/include-files/cfg.json\"`.",
                    ));
                }
                if !arg_template.contains("{file}") {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[0],
                        format!(
                            "declare_scheduler!: `config_file_def` arg_template \
                             `{arg_template}` is missing the `{{file}}` placeholder \
                             — the framework substitutes the guest path at \
                             that position when invoking the scheduler. \
                             Add `{{file}}` (e.g. `\"--config {{file}}\"`)."
                        ),
                    ));
                }
                if !guest_path.starts_with('/') {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[1],
                        format!(
                            "declare_scheduler!: `config_file_def` guest_path \
                             `{guest_path}` must be absolute (start with `/`). \
                             The framework writes the config file at this path \
                             inside the guest, and a relative path breaks the \
                             `mkdir -p` invariant."
                        ),
                    ));
                }
                sched_config_file_def = Some((arg_template, guest_path));
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("declare_scheduler!: unknown field `{other}`"),
                ));
            }
        }
    }

    let sched_name = sched_name.ok_or_else(|| {
        syn::Error::new(
            const_name.span(),
            "declare_scheduler!: missing required field `name`",
        )
    })?;
    validate_kernel_builtin_pair(
        sched_kernel_builtin_enable_span,
        sched_kernel_builtin_disable_span,
    )?;
    let kernel_builtin_set =
        sched_kernel_builtin_enable.is_some() || sched_kernel_builtin_disable.is_some();
    // Both arrays empty would register a KernelBuiltin scheduler that
    // does nothing — functionally identical to the EEVDF baseline.
    if let (Some(en), Some(di)) = (
        sched_kernel_builtin_enable.as_ref(),
        sched_kernel_builtin_disable.as_ref(),
    ) && en.is_empty()
        && di.is_empty()
    {
        return Err(syn::Error::new(
            const_name.span(),
            "declare_scheduler!: `kernel_builtin_enable = []` paired \
             with `kernel_builtin_disable = []` has no commands on \
             either side — that is functionally identical to the \
             kernel-default baseline. Reference \
             `ktstr::test_support::Scheduler::EEVDF` directly instead \
             of declaring a KernelBuiltin scheduler with no commands.",
        ));
    }
    validate_exactly_one_source(
        sched_binary.is_some(),
        sched_binary_path.is_some(),
        kernel_builtin_set,
        const_name.span(),
    )?;
    validate_kernel_name_collision(kernel_builtin_set, &sched_name, sched_name_expr.as_ref())?;

    // Sanity-check the effective topology vs explicit
    // struct-literal constraints. Without this, both
    // `topology = (1, 2, 4, 1)` AND an omitted topology paired
    // with `constraints = TopologyConstraints { min_llcs: 100, .. }`
    // are silently accepted: every gauntlet preset rejects the
    // test at runtime because the effective topology violates
    // the declared minimum (100 LLCs), and the test never runs.
    //
    // When `topology` is omitted the runtime falls back to
    // `Scheduler::named`'s default (numa_nodes=1, llcs=1,
    // cores_per_llc=2, threads_per_core=1, total_cpus=2) — see
    // `Scheduler::named` in `src/test_support/entry.rs`. The macro
    // checks against the same default so infeasible constraints
    // are caught regardless of whether the caller pinned a
    // topology.
    //
    // The macro can only walk the constraint fields when the
    // expression is a struct literal — non-struct-literal forms
    // (`TopologyConstraints::DEFAULT`, a const path) carry values
    // the macro cannot inspect at expand time, so the check no-ops
    // for those shapes.
    if let Some(constraints_expr) = sched_constraints.as_ref()
        && let syn::Expr::Struct(es) = constraints_expr
    {
        let topology_is_default = sched_topology.is_none();
        let (n, l, c, t) = sched_topology.unwrap_or((1, 1, 2, 1));
        let total = (l as u64) * (c as u64) * (t as u64);
        check_constraint_field_against_topology(es, n, l, total, t, topology_is_default)?;
    }

    // Build the Scheduler const expression via the builder chain.
    let sched_name_str = sched_name;
    let mut builder_chain = quote! {
        ::ktstr::test_support::Scheduler::named(#sched_name_str)
    };

    let binary_spec = if let Some(name) = &sched_binary {
        quote! { ::ktstr::test_support::SchedulerSpec::Discover(#name) }
    } else if let Some(path) = &sched_binary_path {
        quote! { ::ktstr::test_support::SchedulerSpec::Path(#path) }
    } else if kernel_builtin_set {
        // Both fields are guaranteed set together by the pair check above.
        let enable = sched_kernel_builtin_enable
            .as_ref()
            .expect("kernel_builtin pair check requires both fields set");
        let disable = sched_kernel_builtin_disable
            .as_ref()
            .expect("kernel_builtin pair check requires both fields set");
        quote! {
            ::ktstr::test_support::SchedulerSpec::KernelBuiltin {
                enable: &[#(#enable),*],
                disable: &[#(#disable),*],
            }
        }
    } else {
        unreachable!("source_count check above proves at least one source set")
    };
    builder_chain = quote! {
        #builder_chain.binary(#binary_spec)
    };
    if let Some((n, l, c, t)) = sched_topology {
        builder_chain = quote! { #builder_chain.topology(#n, #l, #c, #t) };
    }
    if let Some(parent) = &sched_cgroup_parent {
        builder_chain = quote! { #builder_chain.cgroup_parent(#parent) };
    }
    if sched_args_set {
        builder_chain = quote! { #builder_chain.sched_args(&[#(#sched_args),*]) };
    }
    if sched_sysctls_set {
        let entries = &sched_sysctls;
        builder_chain = quote! { #builder_chain.sysctls(&[#(#entries),*]) };
    }
    if sched_kargs_set {
        builder_chain = quote! { #builder_chain.kargs(&[#(#sched_kargs),*]) };
    }
    if sched_kernels_set {
        builder_chain = quote! { #builder_chain.kernels(&[#(#sched_kernels),*]) };
    }
    if let Some(tc) = &sched_constraints {
        builder_chain = quote! { #builder_chain.constraints(#tc) };
    }
    if let Some(cf) = &sched_config_file {
        builder_chain = quote! { #builder_chain.config_file(#cf) };
    }
    if let Some(a) = &sched_assert {
        builder_chain = quote! { #builder_chain.assert(#a) };
    }
    if let Some((arg, path)) = &sched_config_file_def {
        builder_chain = quote! { #builder_chain.config_file_def(#arg, #path) };
    }

    let registry_ident = format_ident!("__KTSTR_SCHED_REG_{}", const_name);

    // Default the emitted const's visibility to `pub` when the user
    // omits a prefix. Explicit prefixes (`pub`, `pub(crate)`,
    // `pub(super)`, `pub(in path)`) flow through verbatim via
    // `quote!`'s `ToTokens` impl for `syn::Visibility`.
    let effective_visibility = match &visibility {
        syn::Visibility::Inherited => quote! { pub },
        v => quote! { #v },
    };

    let expanded = quote! {
        // Suppress `missing_docs` on the emitted static so consumer
        // crates that set `#![deny(missing_docs)]` can still invoke
        // `declare_scheduler!`. The const name is the user-supplied
        // identifier and the macro itself is the documented entry
        // point — requiring a doc comment per declaration would force
        // boilerplate at every call site.
        #[allow(missing_docs)]
        #effective_visibility static #const_name: ::ktstr::test_support::Scheduler = #builder_chain;

        // The registry static stays plain `static` regardless of the
        // user-facing const's visibility — linkme gathers it via
        // link-section walking (not Rust name resolution), so its
        // visibility is irrelevant to the slice mechanism. Keeping it
        // private keeps the registry symbol opaque even when the
        // user-facing const is `pub`.
        #[::ktstr::distributed_slice(::ktstr::test_support::KTSTR_SCHEDULERS)]
        #[linkme(crate = ::ktstr::linkme)]
        static #registry_ident: &'static ::ktstr::test_support::Scheduler = &#const_name;
    };

    Ok(expanded)
}

fn expect_str_lit(expr: &syn::Expr, key: &syn::Ident, field: &str) -> syn::Result<String> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(ls),
            ..
        }) => Ok(ls.value()),
        _ => Err(syn::Error::new(
            key.span(),
            format!("declare_scheduler!: `{field}` must be a string literal"),
        )),
    }
}

fn expect_str_lit_element(expr: &syn::Expr, field: &str) -> syn::Result<String> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(ls),
            ..
        }) => Ok(ls.value()),
        _ => Err(syn::Error::new_spanned(
            expr,
            format!("declare_scheduler!: element of `{field}` must be a string literal"),
        )),
    }
}

fn expect_u32_lit(expr: &syn::Expr, key: &syn::Ident, field: &str) -> syn::Result<u32> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(li),
            ..
        }) => li.base10_parse(),
        _ => Err(syn::Error::new(
            key.span(),
            format!("declare_scheduler!: `{field}` element must be an integer literal"),
        )),
    }
}

fn expect_array<'a>(
    expr: &'a syn::Expr,
    key: &syn::Ident,
    field: &str,
) -> syn::Result<&'a syn::ExprArray> {
    if let syn::Expr::Array(arr) = expr {
        Ok(arr)
    } else {
        Err(syn::Error::new(
            key.span(),
            format!("declare_scheduler!: `{field}` must be an array literal `[..]`"),
        ))
    }
}

/// Returns true when `s` contains no visible characters — i.e. it is
/// empty, contains only Unicode `White_Space` (stripped by `str::trim`),
/// or contains only Cf-category invisibles that `trim` leaves behind.
///
/// The Cf carve-out enumerates the realistic copy-paste hazards rather
/// than relying on a Unicode-category lookup (which would pull in a UCD
/// crate just for a macro-time check). Covered:
/// - U+00AD SOFT HYPHEN (browser-wrapped text)
/// - U+200B ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER,
///   U+200D ZERO WIDTH JOINER, U+200E LEFT-TO-RIGHT MARK,
///   U+200F RIGHT-TO-LEFT MARK
/// - U+202A-U+202E bidi formatting (LRE, RLE, PDF, LRO, RLO)
/// - U+2060 WORD JOINER, U+2061-U+2064 math invisibles
///   (FUNCTION APPLICATION, INVISIBLE TIMES, INVISIBLE SEPARATOR,
///   INVISIBLE PLUS)
/// - U+2066-U+2069 bidi isolates (LRI, RLI, FSI, PDI)
/// - U+FEFF BYTE ORDER MARK / ZWNBSP
///
/// Without this carve-out, an operator who pastes a string containing
/// any of these characters past the bare `trim().is_empty()` check
/// ships a literal that surfaces as a confusing runtime failure
/// (sidecar lookup miss / binary not found) instead of an explicit
/// compile-time rejection.
fn is_visibly_empty(s: &str) -> bool {
    s.chars().all(|c| {
        c.is_whitespace()
            || matches!(
                c,
                '\u{00AD}'
                    | '\u{200B}'..='\u{200F}'
                    | '\u{202A}'..='\u{202E}'
                    | '\u{2060}'..='\u{2064}'
                    | '\u{2066}'..='\u{2069}'
                    | '\u{FEFF}'
            )
    })
}

/// Reject an empty / whitespace-only / invisible-only string literal
/// for a `declare_scheduler!` field. The diagnostic names the
/// offending field so the call sites (`name`, `binary`,
/// `binary_path`) can share one helper instead of repeating the
/// `is_visibly_empty` check + tailored error block at each arm.
fn check_visible_lit(lit: &str, value: &syn::Expr, field: &str) -> syn::Result<()> {
    if is_visibly_empty(lit) {
        return Err(syn::Error::new_spanned(
            value,
            format!(
                "declare_scheduler!: `{field}` must contain at least \
                 one visible character (empty, whitespace-only, \
                 and invisible-only literals are rejected)"
            ),
        ));
    }
    Ok(())
}

/// Reject the asymmetric `kernel_builtin_enable` / `kernel_builtin_disable`
/// pair: setting one without the other is always a typo because the
/// `SchedulerSpec::KernelBuiltin` variant carries both lists in the
/// same struct, so requiring both at macro-time mirrors the type-level
/// invariant. The per-direction diagnostics name the missing side and
/// the canonical recovery action (start-side vs teardown-side). The
/// caret lands on the field-name token of whichever side IS set —
/// not the const-name fallback the older shape used — so the
/// operator sees the offending key, not just the scheduler name.
fn validate_kernel_builtin_pair(
    enable_span: Option<proc_macro2::Span>,
    disable_span: Option<proc_macro2::Span>,
) -> syn::Result<()> {
    match (enable_span, disable_span) {
        (Some(span), None) => Err(syn::Error::new(
            span,
            "declare_scheduler!: `kernel_builtin_enable` set without \
             `kernel_builtin_disable`. Both fields must be set \
             together (or both omitted). Add a `kernel_builtin_disable \
             = [\"echo ...\"]` line that restores the kernel default \
             at teardown.",
        )),
        (None, Some(span)) => Err(syn::Error::new(
            span,
            "declare_scheduler!: `kernel_builtin_disable` set without \
             `kernel_builtin_enable`. Both fields must be set \
             together (or both omitted). Add a `kernel_builtin_enable \
             = [\"echo ...\"]` line that switches the kernel into the \
             chosen policy at scenario start.",
        )),
        _ => Ok(()),
    }
}

/// Reject `name = "kernel"` (case- and whitespace-insensitive) when
/// the scheduler is a `KernelBuiltin` variant. The variant's
/// `SchedulerSpec::display_name` is the literal `"kernel"`, so a
/// user scheduler whose name resolves to `"kernel"` would collide
/// with the variant label in failure-dump headers and sidecar
/// comparisons. No-ops outside the KernelBuiltin case (the `"eevdf"`
/// / `"kernel_default"` reservations for non-KernelBuiltin paths
/// fire at parse time in the `name` arm).
fn validate_kernel_name_collision(
    kernel_builtin_set: bool,
    sched_name: &str,
    sched_name_expr: Option<&syn::Expr>,
) -> syn::Result<()> {
    if !kernel_builtin_set || sched_name.trim().to_lowercase() != "kernel" {
        return Ok(());
    }
    let expr = sched_name_expr.expect("sched_name_expr is populated whenever sched_name is Some");
    Err(syn::Error::new_spanned(
        expr,
        format!(
            "declare_scheduler!: `name = \"{sched_name}\"` collides with \
             the KernelBuiltin variant's display_name (`\"kernel\"`). \
             Pick a different name so failure dumps and sidecar \
             entries can distinguish this scheduler from the \
             variant label."
        ),
    ))
}

/// Require exactly one of the three `declare_scheduler!` source
/// fields — `binary` (Discover), `binary_path` (Path), or the paired
/// `kernel_builtin_{enable,disable}` (KernelBuiltin). Setting more
/// than one is ambiguous because each maps to a different
/// `SchedulerSpec` variant and they cannot stack; setting none is
/// rejected so any user wanting the kernel-default baseline
/// references `Scheduler::EEVDF` directly rather than declaring a
/// scheduler with no source.
fn validate_exactly_one_source(
    binary_set: bool,
    binary_path_set: bool,
    kernel_builtin_set: bool,
    span: proc_macro2::Span,
) -> syn::Result<()> {
    let count = [binary_set, binary_path_set, kernel_builtin_set]
        .iter()
        .filter(|b| **b)
        .count();
    if count == 0 {
        return Err(syn::Error::new(
            span,
            "declare_scheduler!: no scheduler source declared. Pick one of:\n  \
             - `binary = \"scx_my_sched\"` (discover the binary by name)\n  \
             - `binary_path = \"/abs/path/to/scx_custom\"` (absolute filesystem path)\n  \
             - `kernel_builtin_enable = [\"echo 1 > /sys/...\"]` + \
             `kernel_builtin_disable = [\"echo 0 > /sys/...\"]` \
             (in-kernel scheduling policy toggled via shell commands)\n\
             To test under the kernel-default EEVDF baseline, reference \
             `ktstr::test_support::Scheduler::EEVDF` directly instead \
             of declaring a new scheduler.",
        ));
    }
    if count > 1 {
        return Err(syn::Error::new(
            span,
            "declare_scheduler!: more than one scheduler source declared. \
             Pick exactly one of `binary`, `binary_path`, or the \
             `kernel_builtin_enable` + `kernel_builtin_disable` pair. \
             Each maps to a different `SchedulerSpec` variant \
             (`Discover`, `Path`, `KernelBuiltin`) and they cannot stack.",
        ));
    }
    Ok(())
}

/// Validate a single `kernel_builtin_enable` or `kernel_builtin_disable`
/// command string against the grammar accepted by the guest
/// interpreter at `src/vmm/rust_init.rs`'s `exec_shell_line`. Anything
/// else (`>>`, pipes, `;`, variable expansion, sysctl -w, etc.) silently
/// no-ops at runtime, so the macro rejects up-front. Accepted shapes:
///
/// - `echo VALUE > /path` — writes VALUE+newline to /path
/// - blank line (skipped)
/// - `#`-prefixed comment (skipped)
fn validate_kernel_builtin_cmd(elem: &syn::Expr, cmd: &str, slot: &str) -> syn::Result<()> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(());
    }
    if !trimmed.starts_with("echo ") {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` does not start with `echo ` — the guest \
                 interpreter accepts only `echo VALUE > /path` (plus \
                 blank lines and `#` comments). Other shell syntax \
                 (`>>`, pipes, `;`, variable expansion, sysctl) \
                 silently no-ops at runtime."
            ),
        ));
    }
    // Reject append `>>` explicitly. The guest's split_once(\" > \") would
    // miss the substring on `echo X >> /path` (no space between `>>`)
    // and fall through to the unsupported-command no-op; surface the
    // intent at expand time with the append-specific diagnostic.
    if trimmed.contains(">>") {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` uses `>>` (append) — the guest interpreter \
                 only handles single-`>` truncating writes. Use `>` \
                 instead."
            ),
        ));
    }
    let rest = &trimmed["echo ".len()..];
    let (value, path) = match rest.split_once(" > ") {
        Some((v, p)) => (v.trim(), p.trim()),
        None => {
            return Err(syn::Error::new_spanned(
                elem,
                format!(
                    "declare_scheduler!: `kernel_builtin_{slot}` command \
                     `{cmd}` is missing the ` > ` (space-greater-space) \
                     redirect — the guest interpreter requires \
                     `echo VALUE > /path` with literal spaces around \
                     `>` (`exec_shell_line` in `src/vmm/rust_init.rs` \
                     uses `split_once(\" > \")`)."
                ),
            ));
        }
    };
    if value.is_empty() {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` writes an empty value — `echo > /path` is \
                 valid shell but useless. Provide the value to write \
                 (e.g. `echo 1 > /sys/...`)."
            ),
        ));
    }
    if !path.starts_with('/') {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` writes to relative path `{path}` — the guest \
                 interpreter writes via `std::fs::write`, which resolves \
                 relative to the guest init's cwd (`/`). Use an absolute \
                 path to be explicit."
            ),
        ));
    }
    Ok(())
}

/// Append every string-literal element of an array-literal-typed
/// `declare_scheduler!` field into `out`. The `sched_args` and
/// `kargs` arms both use this exact loop with no per-element
/// validation beyond `expect_str_lit_element`; collapsing them into
/// one helper keeps the two arms one line apiece. Fields that need
/// per-element gates (`kernels`, `kernel_builtin_{enable,disable}`)
/// keep their own loop with the additional checks.
fn append_str_array_into(
    value: &syn::Expr,
    key: &syn::Ident,
    field: &str,
    out: &mut Vec<String>,
) -> syn::Result<()> {
    let arr = expect_array(value, key, field)?;
    for elem in &arr.elems {
        out.push(expect_str_lit_element(elem, field)?);
    }
    Ok(())
}

/// Parse the array literal under `kernel_builtin_enable` or
/// `kernel_builtin_disable` into the cmds vec consumed by
/// `SchedulerSpec::KernelBuiltin`. Each element must be a string
/// literal and each command must match the guest interpreter's
/// `echo VALUE > /path` grammar (validated via
/// [`validate_kernel_builtin_cmd`]). The two call sites differ only
/// in field name and slot, so folding the loop avoids a copy-paste
/// pair that drifts under any future change to the per-element gate.
fn parse_kernel_builtin_cmds(
    value: &syn::Expr,
    key: &syn::Ident,
    field: &str,
    slot: &str,
) -> syn::Result<Vec<String>> {
    let arr = expect_array(value, key, field)?;
    let mut cmds = Vec::with_capacity(arr.elems.len());
    for elem in &arr.elems {
        let s = expect_str_lit_element(elem, field)?;
        validate_kernel_builtin_cmd(elem, &s, slot)?;
        cmds.push(s);
    }
    Ok(cmds)
}

/// Policy axis for `validate_const_eligible`. Different
/// `declare_scheduler!` fields have different canonical
/// const-construction patterns and need different MethodCall + Call
/// tolerances.
#[derive(Clone, Copy)]
enum ConstEligibility {
    /// Used for `constraints`. Rejects `MethodCall(...)` and rejects
    /// `Call(...)` whose function path tail is not PascalCase. The
    /// canonical pattern is a struct literal (`TopologyConstraints
    /// { .. }`) or a const path (`TopologyConstraints::DEFAULT`) —
    /// method chains in that position are always wrong because
    /// `TopologyConstraints` has no const-fn builder.
    StructLiteralOnly,
    /// Used for `assert`. Accepts `MethodCall(...)` and recurses
    /// into receiver + args; accepts `Call(...)` with multi-segment
    /// or PascalCase function path and recurses into args; rejects
    /// `Call(...)` with single-segment lowercase path (bare local
    /// helper). Required because `Assert`'s canonical const
    /// constructors are snake_case: `Assert::NO_OVERRIDES`,
    /// `Assert::default_checks()`, and the
    /// `Assert::NO_OVERRIDES.check_not_starved()` chain pattern.
    AllowConstMethodChains,
}

/// Reject a `declare_scheduler!` field whose value cannot be
/// const-evaluated. The field lands in a `pub static`, so non-const
/// helper calls yield deep const-eval failures at the spread site
/// that are hard to map back to the original mistake. This validator
/// catches them at expand time with a per-field tailored diagnostic.
///
/// Recurses into struct-literal field values and the `..rest`
/// spread; both PascalCase Call args and MethodCall args (when
/// allowed by `mode`) are recursed too.
fn validate_const_eligible(
    expr: &syn::Expr,
    field_name: &str,
    accepted_shapes: &str,
    mode: ConstEligibility,
) -> syn::Result<()> {
    let recurse = |e: &syn::Expr| validate_const_eligible(e, field_name, accepted_shapes, mode);
    match expr {
        syn::Expr::Struct(es) => {
            for fv in &es.fields {
                recurse(&fv.expr)?;
            }
            if let Some(rest) = &es.rest {
                recurse(rest)?;
            }
            Ok(())
        }
        syn::Expr::Path(_) => Ok(()),
        syn::Expr::Paren(p) => recurse(&p.expr),
        syn::Expr::Reference(r) => recurse(&r.expr),
        syn::Expr::Unary(u) => recurse(&u.expr),
        syn::Expr::Binary(b) => {
            recurse(&b.left)?;
            recurse(&b.right)?;
            Ok(())
        }
        syn::Expr::Lit(_) => Ok(()),
        syn::Expr::MethodCall(mc) => match mode {
            ConstEligibility::StructLiteralOnly => Err(field_not_const_error(
                field_name,
                accepted_shapes,
                expr,
                true,
            )),
            ConstEligibility::AllowConstMethodChains => {
                recurse(&mc.receiver)?;
                for arg in &mc.args {
                    recurse(arg)?;
                }
                Ok(())
            }
        },
        syn::Expr::Call(call) => match mode {
            ConstEligibility::StructLiteralOnly => {
                if call_func_looks_const_eligible_constructor(&call.func) {
                    for arg in &call.args {
                        recurse(arg)?;
                    }
                    Ok(())
                } else {
                    Err(field_not_const_error(
                        field_name,
                        accepted_shapes,
                        expr,
                        true,
                    ))
                }
            }
            ConstEligibility::AllowConstMethodChains => {
                if call_func_is_single_segment_lowercase(&call.func) {
                    Err(field_not_const_error(
                        field_name,
                        accepted_shapes,
                        expr,
                        true,
                    ))
                } else {
                    for arg in &call.args {
                        recurse(arg)?;
                    }
                    Ok(())
                }
            }
        },
        syn::Expr::Block(_) => Err(field_block_not_const_error(field_name, expr)),
        _ => Err(field_not_const_error(
            field_name,
            accepted_shapes,
            expr,
            false,
        )),
    }
}

/// Heuristic: does this call expression look like a const-eligible
/// constructor? Two shapes qualify:
///
/// 1. PascalCase last segment (`Some(x)`, `MyVariant(x)`, `Foo::Bar(x)`):
///    Rust naming convention reserves PascalCase for types and variants,
///    so a path tail starting with an uppercase ASCII letter is very
///    likely a const-eligible tuple-struct or enum-variant constructor.
/// 2. The conventional `new` / `default` last segments (`Type::new()`,
///    `Type::default()`, `Default::default()`): these are the universally
///    recognized const-fn constructor names.
///
/// Modern Rust accepts `..Type::new()` spreads in `static` / `const`
/// when the called fn is `const fn` AND the returned type is
/// trivially-Drop (no Drop impl, no fields with destructors). For
/// trivially-Drop types like TopologyConstraints (all-primitive Copy
/// struct), the temporary returned by the const-fn constructor is
/// promoted to static lifetime via rustc's MIR promotion machinery
/// (`rustc_mir_transform::promote_consts::validate_call` accepts
/// arbitrary const-fn calls in static contexts via the
/// `promote_all_fn = Static(_)` arm).
///
/// Two rustc errors surface at the spread site when those conditions
/// don't hold; both are CLEANER than the deep const-eval diagnostic
/// chain we'd produce by pre-rejecting at the macro layer:
/// - E0015 ("cannot call non-const associated function in statics") —
///   the called fn isn't `const fn`. Surfaced when a user writes
///   `..Type::new()` and `new` is a regular fn rather than `const fn`.
/// - E0493 ("destructor of T cannot be evaluated at compile-time") —
///   the called fn IS `const fn` but the returned type has a
///   non-trivial Drop (any field with a destructor: String, Vec,
///   Box, Arc, or any of those transitively via Option, Box, etc.).
///   Hit by types like KtstrTestEntry (which carries `Option<String>`
///   via `Option<DiskConfig>`); their DEFAULT must remain a
///   struct-literal const, NOT a const-fn-returning-Self.
///
/// Lowercase free-fn patterns (`build_helper()`) and unrecognized
/// lowercase methods (`Type::custom()`) are not accepted — the macro
/// has no way to tell whether those are const-eligible, and getting
/// it wrong produces the bad deep-const-eval diagnostic at the spread
/// site.
fn call_func_looks_const_eligible_constructor(func: &syn::Expr) -> bool {
    let syn::Expr::Path(ep) = unwrap_parens(func) else {
        return false;
    };
    path_last_segment_ident(&ep.path).is_some_and(|ident| {
        let s = ident.to_string();
        s.chars().next().is_some_and(|c| c.is_ascii_uppercase()) || s == "new" || s == "default"
    })
}

/// Return the last segment's identifier of a path, or `None` for an
/// empty path. Used by helpers that match on a path's tail name
/// (`Some`, `BTreeSet`, etc.) without forcing the caller to import
/// the full path.
fn path_last_segment_ident(path: &syn::Path) -> Option<&syn::Ident> {
    path.segments.last().map(|s| &s.ident)
}

/// Heuristic: is this call expression a single-segment lowercase
/// function path (`build_helper()`, `default()`, snake_case-style)?
/// Used by `validate_const_eligible` under
/// `ConstEligibility::AllowConstMethodChains` to reject bare local
/// helpers while accepting type/module-prefixed const-fn calls
/// (`Assert::default_checks()`, `Some(x)`, `path::to::helper()`).
fn call_func_is_single_segment_lowercase(func: &syn::Expr) -> bool {
    let syn::Expr::Path(ep) = unwrap_parens(func) else {
        return false;
    };
    if ep.path.segments.len() != 1 {
        return false;
    }
    path_last_segment_ident(&ep.path).is_some_and(|ident| {
        ident
            .to_string()
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase())
    })
}

/// Strip wrapping `Expr::Paren` layers from an expression so
/// heuristics that match on shape (`Expr::Path` for constructor or
/// snake-case detection) see through `(build_helper)()` style
/// parenthesization. Without this, a deliberately parenthesized
/// bare ident would bypass the lowercase-bare-call rejection.
fn unwrap_parens(expr: &syn::Expr) -> &syn::Expr {
    let mut cur = expr;
    while let syn::Expr::Paren(p) = cur {
        cur = &p.expr;
    }
    cur
}

/// Emit the shared `declare_scheduler!`: `<field>` not-const-eligible
/// diagnostic. `accepted_shapes` is the field-specific sentence
/// listing the accepted shapes (e.g. struct literal vs const path).
/// `append_call_hint` adds the trailing sentence about helper calls
/// and method chains failing at the spread site — used for the
/// `Call`/`MethodCall` arm where that hint is load-bearing.
fn field_not_const_error(
    field_name: &str,
    accepted_shapes: &str,
    expr: &syn::Expr,
    append_call_hint: bool,
) -> syn::Error {
    let header = format!(
        "declare_scheduler!: `{field_name}` must be a const-evaluable \
         expression (emitted into a `pub static`). {accepted_shapes}"
    );
    let msg = if append_call_hint {
        format!(
            "{header} Non-const helper calls and method chains are not \
             const-eligible and would fail with a deep const-eval \
             diagnostic at the spread site."
        )
    } else {
        header
    };
    syn::Error::new_spanned(expr, msg)
}

/// Emit the shared `declare_scheduler!`: `<field>` block-expression
/// rejection diagnostic. Block expressions need tailored guidance
/// (drop the braces / use a const binding) that the generic
/// non-const-eligible message doesn't carry.
fn field_block_not_const_error(field_name: &str, expr: &syn::Expr) -> syn::Error {
    syn::Error::new_spanned(
        expr,
        format!(
            "declare_scheduler!: `{field_name}` must be a const-evaluable \
             expression (emitted into a `pub static`). Block expressions \
             like `{{ ... }}` are not const-eligible here — for a single \
             literal value, drop the braces. For shared values, use a \
             const binding (`const MY_VAL: T = ...;` then reference \
             `MY_VAL`)."
        ),
    )
}

/// Field-specific accepted-shapes sentence for `constraints`.
const CONSTRAINTS_ACCEPTED_SHAPES: &str = "Use a struct literal `TopologyConstraints { ..TopologyConstraints::DEFAULT }`, \
     a struct literal spreading a const-fn constructor like `TopologyConstraints { ..TopologyConstraints::new() }`, \
     or a const path like `TopologyConstraints::DEFAULT`.";

/// Field-specific accepted-shapes sentence for `assert`.
const ASSERT_ACCEPTED_SHAPES: &str = "Use a const path like `Assert::NO_OVERRIDES`, a const-fn call like \
     `Assert::default_checks()`, or a chain of const-fn setters like \
     `Assert::NO_OVERRIDES.check_not_starved().max_gap_ms(50)`.";

/// Walk a `TopologyConstraints { .. }` struct literal and reject
/// fields whose literal values make the declared scheduler topology
/// `(numa, llcs, _, threads)` infeasible (total CPUs = `total`,
/// threads_per_core = `threads`).
///
/// Only literal-valued fields are checked — non-literal expressions
/// (paths, calls) carry values the macro cannot evaluate. Fields
/// dropped via `..TopologyConstraints::DEFAULT` are also not
/// validated against the DEFAULT values; doing so would silently
/// reject test authors who pair an explicit non-default topology
/// with the default constraint set on the assumption that those
/// defaults match. Limiting the check to fields the user explicitly
/// wrote keeps the diagnostic targeted: it fires only when an
/// explicit constraint contradicts an explicit topology.
fn check_constraint_field_against_topology(
    es: &syn::ExprStruct,
    numa: u32,
    llcs: u32,
    total_cpus: u64,
    threads_per_core: u32,
    topology_is_default: bool,
) -> syn::Result<()> {
    // When `topology` was omitted, the macro inferred Scheduler::named
    // defaults. Reading "effective topology llcs (1)" without
    // context makes a user wonder where the 1 came from — they
    // didn't write a topology field. Append a tail that names the
    // fallback source + the override syntax.
    let topology_origin_tail = if topology_is_default {
        " (`topology` field omitted; macro fell back to \
         Scheduler::named's default `(numa=1, llcs=1, \
         cores=2, threads=1)`. Add an explicit \
         `topology = (numa, llcs, cores, threads)` to \
         override.)"
    } else {
        ""
    };
    for fv in &es.fields {
        let syn::Member::Named(ident) = &fv.member else {
            continue;
        };
        let name = ident.to_string();
        match name.as_str() {
            "min_llcs" => {
                if let Some(v) = u64_from_lit_expr(&fv.expr)
                    && v > llcs as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.min_llcs \
                             ({v}) exceeds effective topology llcs \
                             ({llcs}); every gauntlet preset would \
                             reject this test at runtime and the test \
                             would never execute. Lower min_llcs to \
                             {llcs} or fewer, or raise topology llcs.\
                             {topology_origin_tail}",
                        ),
                    ));
                }
            }
            "max_llcs" => {
                if let Some(v) = u64_from_option_some_lit(&fv.expr)
                    && v < llcs as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.max_llcs \
                             (Some({v})) is below effective topology \
                             llcs ({llcs}); every gauntlet preset \
                             would reject this test at runtime and \
                             the test would never execute. Raise \
                             max_llcs to {llcs} or higher, or lower \
                             topology llcs.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "min_numa_nodes" => {
                if let Some(v) = u64_from_lit_expr(&fv.expr)
                    && v > numa as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.min_numa_nodes \
                             ({v}) exceeds effective topology numa_nodes \
                             ({numa}); every gauntlet preset would reject \
                             this test at runtime and the test would \
                             never execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "max_numa_nodes" => {
                if let Some(v) = u64_from_option_some_lit(&fv.expr)
                    && v < numa as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.max_numa_nodes \
                             (Some({v})) is below effective topology \
                             numa_nodes ({numa}); every gauntlet preset \
                             would reject this test at runtime and the \
                             test would never execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "min_cpus" => {
                if let Some(v) = u64_from_lit_expr(&fv.expr)
                    && v > total_cpus
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.min_cpus \
                             ({v}) exceeds effective topology total_cpus \
                             ({total_cpus} = llcs * cores * threads); \
                             every gauntlet preset would reject this \
                             test at runtime and the test would never \
                             execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "max_cpus" => {
                if let Some(v) = u64_from_option_some_lit(&fv.expr)
                    && v < total_cpus
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.max_cpus \
                             (Some({v})) is below effective topology \
                             total_cpus ({total_cpus} = llcs * cores * \
                             threads); every gauntlet preset would \
                             reject this test at runtime and the test \
                             would never execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "requires_smt" => {
                if let Some(true) = bool_from_lit_expr(&fv.expr)
                    && threads_per_core < 2
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.requires_smt \
                             = true but effective topology \
                             threads_per_core = {threads_per_core}; SMT \
                             requires threads_per_core >= 2. Set topology \
                             threads_per_core to 2 (or higher) or drop \
                             the requires_smt constraint.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract a `u64` from a literal integer expression. Returns `None`
/// for any other shape so non-literal field values pass through the
/// macro-time check.
fn u64_from_lit_expr(expr: &syn::Expr) -> Option<u64> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(li),
        ..
    }) = expr
    else {
        return None;
    };
    li.base10_parse().ok()
}

/// Extract a `u64` from a `Some(<int literal>)` expression. Returns
/// `None` for `None`, paths, or anything else — non-literal forms
/// pass through unchecked.
fn u64_from_option_some_lit(expr: &syn::Expr) -> Option<u64> {
    let expr = unwrap_parens(expr);
    let syn::Expr::Call(call) = expr else {
        return None;
    };
    let syn::Expr::Path(ep) = unwrap_parens(&call.func) else {
        return None;
    };
    if path_last_segment_ident(&ep.path)? != "Some" {
        return None;
    }
    if call.args.len() != 1 {
        return None;
    }
    u64_from_lit_expr(unwrap_parens(&call.args[0]))
}

/// Extract a `bool` from a literal boolean expression. Returns `None`
/// for any other shape.
fn bool_from_lit_expr(expr: &syn::Expr) -> Option<bool> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Bool(lb),
        ..
    }) = expr
    else {
        return None;
    };
    Some(lb.value())
}

/// Derive macro that generates a `Payload` const from an annotated
/// struct for a userspace binary workload (stress-ng, fio, and
/// similar tools test authors compose under a scheduler).
///
/// # Required struct-level attributes (`#[payload(...)]`)
///
/// - `binary = "..."` — the binary name resolved by the guest's
///   include-files infrastructure (required). Becomes
///   `PayloadKind::Binary(name)` (`ktstr::test_support::PayloadKind::Binary`),
///   and is also auto-prepended to the emitted `include_files` slice
///   so the binary is packaged into the initramfs without needing a
///   separate `#[include_files("...")]` entry. Extra auxiliary files
///   (helpers, configs, fixtures) still go on `#[include_files(...)]`.
///
/// # Optional struct-level attributes
///
/// - `name = "..."` — short name used in logs and sidecar records.
///   Defaults to the binary name.
/// - `output = Json | ExitCode | LlmExtract("hint")` — how the
///   framework extracts metrics from the payload's stdout. The
///   variant names match the `OutputFormat` enum and the `Polarity`
///   kwarg grammar. Defaults to `ExitCode`. The `LlmExtract` form
///   accepts an optional string literal focus hint appended to the
///   default LLM prompt; bare `LlmExtract` with no parenthesized
///   argument is a shorthand for `LlmExtract()` (no hint).
///
/// # Optional outer attributes
///
/// - `#[default_args("--a", "--b", ...)]` — variadic string
///   literals appended to the binary's argv when the payload runs.
///   May repeat across multiple `#[default_args(...)]` attrs; entries
///   accumulate in source order.
/// - `#[default_check(...)]` — one `MetricCheck` (`ktstr::test_support::MetricCheck`)
///   construction expression (e.g. `min("iops", 1000.0)`,
///   `exit_code_eq(0)`). May repeat; entries accumulate in source
///   order. Both `min(...)` and `MetricCheck::min(...)` are accepted: the
///   macro prepends `::ktstr::test_support::MetricCheck::` when the
///   expression doesn't already spell `MetricCheck::` on its callee path,
///   so bare constructors work without an import and qualified
///   constructors read naturally in modules that already have
///   `MetricCheck` in scope.
/// - `#[metric(name = "...", polarity = ..., unit = "...")]` —
///   kwarg form. `polarity` is one of `HigherBetter`, `LowerBetter`,
///   `TargetValue(f64)`, `Unknown`. May repeat; entries accumulate.
/// - `#[include_files("helper", "config.json", ...)]` — variadic
///   string literals appended to the emitted `include_files` slice
///   after the auto-injected binary entry. Each entry passes through
///   the same resolver used by the CLI `-i` flag (bare names search
///   host `PATH`; explicit paths must exist; directories are walked).
///   The primary binary is already packaged automatically, so this
///   attribute is only needed for auxiliary files the payload
///   depends on.
///
/// # Const name derivation
///
/// Strip trailing `"Payload"` suffix (if present), then convert to
/// `SCREAMING_SNAKE_CASE`. `FioPayload` → `FIO`,
/// `StressNgPayload` → `STRESS_NG`, `Fio` (no suffix) → `FIO`.
///
/// # Example
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// #[derive(Payload)]
/// #[payload(binary = "fio", output = Json)]
/// #[default_args("--output-format=json", "--minimal")]
/// #[default_check(exit_code_eq(0))]
/// #[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
/// struct FioPayload;
/// ```
#[proc_macro_derive(Payload, attributes(payload, default_args, default_check, metric))]
pub fn derive_payload(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_payload_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn derive_payload_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;
    // Inherit the input struct's visibility so the emitted `const`
    // matches: `pub struct FooPayload` → `pub const FOO: Payload`.
    // Private structs produce private consts, preserving the
    // previous behavior for in-crate tests that rely on it.
    let struct_vis = &input.vis;

    // Reject non-struct inputs; the payload attribute grammar is
    // struct-only, keeping the attribute space unambiguous.
    if !matches!(&input.data, Data::Struct(_)) {
        return Err(syn::Error::new_spanned(
            struct_name,
            "Payload can only be derived for structs",
        ));
    }

    let mut binary: Option<String> = None;
    let mut name_override: Option<String> = None;
    // `None` means "not specified" → default ExitCode at emit time.
    // `Some(tokens)` holds the fully-qualified OutputFormat variant
    // the user selected (possibly with an LlmExtract hint expression).
    let mut output_tokens: Option<proc_macro2::TokenStream> = None;

    for attr in &input.attrs {
        if !attr.path().is_ident("payload") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("binary") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                binary = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("name") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                name_override = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("output") {
                let value = meta.value()?;
                let expr: syn::Expr = value.parse()?;
                output_tokens = Some(output_from_expr(&expr)?);
                Ok(())
            } else {
                Err(meta.error(format!(
                    "unknown payload attribute `{}`",
                    meta.path
                        .get_ident()
                        .map(|i| i.to_string())
                        .unwrap_or_default()
                )))
            }
        })?;
    }

    let binary = binary.ok_or_else(|| {
        syn::Error::new_spanned(struct_name, "missing `binary = \"...\"` in #[payload(...)]")
    })?;

    // Default output = ExitCode. Resolve once here to a canonical
    // TokenStream so the emitter only has one path.
    let output_tokens = output_tokens.unwrap_or_else(|| {
        quote! { ::ktstr::test_support::OutputFormat::ExitCode }
    });

    // `name` falls back to `binary` when omitted.
    let payload_name = name_override.unwrap_or_else(|| binary.clone());

    // Walk outer `#[default_args(...)]` / `#[default_check(...)]` /
    // `#[metric(...)]` / `#[include_files(...)]` attrs in source
    // order so the emitted slices match the declaration.
    let mut default_args: Vec<String> = Vec::new();
    let mut default_checks: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut metrics: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut seen_metric_names: Vec<String> = Vec::new();
    let mut include_files: Vec<String> = Vec::new();

    for attr in &input.attrs {
        if attr.path().is_ident("default_args") {
            // Variadic string literals: `#[default_args("--a", "--b")]`.
            let parser =
                syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated;
            let parsed = attr.parse_args_with(parser).map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "default_args must be one or more string literals separated by `,`",
                )
            })?;
            for lit in parsed {
                default_args.push(lit.value());
            }
        } else if attr.path().is_ident("default_check") {
            // Single MetricCheck-constructing expression. Two forms accepted:
            //   - bare: `min("iops", 1000.0)` — the macro prepends
            //     `::ktstr::test_support::MetricCheck::` so users don't have
            //     to import `MetricCheck` in every module that derives.
            //   - qualified: `MetricCheck::min("iops", 1000.0)` — the user
            //     wrote `MetricCheck::` themselves; emit the expression
            //     unchanged so the user's own path resolution wins
            //     (and a double `MetricCheck::MetricCheck::` prefix can't happen).
            let expr: syn::Expr = attr.parse_args().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "default_check must be a MetricCheck constructor expression (e.g. min(\"iops\", 1000.0))",
                )
            })?;
            if expr_has_check_prefix(&expr) {
                default_checks.push(quote! { #expr });
            } else {
                default_checks.push(quote! { ::ktstr::test_support::MetricCheck::#expr });
            }
        } else if attr.path().is_ident("metric") {
            // Kwarg form: name = "...", polarity = ..., unit = "...".
            let (metric_name, tokens) = parse_metric_attr(attr)?;
            // Reject duplicate metric names — two `#[metric(name = "x", ...)]`
            // lines with the same name are almost certainly a copy-paste
            // typo; the runtime pipeline's `resolve_polarities` uses
            // last-wins semantics, so a duplicate silently shadows the
            // first hint without any signal to the test author.
            if let Some(existing) = seen_metric_names.iter().find(|n| *n == &metric_name) {
                return Err(syn::Error::new_spanned(
                    attr,
                    format!(
                        "duplicate metric name `{existing}` — each \
                         `#[metric(name = \"...\")]` declaration must name a \
                         distinct metric. Remove the duplicate or rename one \
                         of them."
                    ),
                ));
            }
            seen_metric_names.push(metric_name);
            metrics.push(tokens);
        } else if attr.path().is_ident("include_files") {
            // Variadic string literals: `#[include_files("helper",
            // "config.json")]`. Each entry is passed through to
            // `Payload::include_files` verbatim; the runtime
            // resolver (`resolve_include_files`) interprets bare
            // names vs explicit paths vs directories the same way
            // the CLI `-i` flag does. Order is preserved so the
            // user's declaration order is visible in the emitted
            // slice — useful when the resolver's dedup policy
            // reports a conflict, as the first-declared entry
            // wins.
            let parser =
                syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated;
            let parsed = attr.parse_args_with(parser).map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "include_files must be one or more string literals separated by `,`",
                )
            })?;
            for lit in parsed {
                include_files.push(lit.value());
            }
        }
    }

    // Derive the const name: strip "Payload" suffix and uppercase.
    let struct_str = struct_name.to_string();
    let base = struct_str.strip_suffix("Payload").unwrap_or(&struct_str);
    if base.is_empty() {
        return Err(syn::Error::new(
            struct_name.span(),
            "struct name cannot be just \"Payload\"",
        ));
    }
    let const_name = format_ident!("{}", camel_to_screaming_snake(base));

    // Auto-inject the `binary` spec as the first entry in the emitted
    // `include_files` slice so `#[payload(binary = "X")]` alone is
    // enough to package `X` into the initramfs — no separate
    // `#[include_files("X")]` required. The runtime's
    // `dedupe_include_files` canonicalizes host paths, so a user who
    // also writes `#[include_files("X")]` (or lists the same binary
    // on `#[ktstr_test(extra_include_files = [..])]`) still works:
    // the duplicate collapses silently. User-declared entries follow
    // in source order — preserving the existing first-declared-wins
    // behavior within the user's own list.
    include_files.insert(0, binary.clone());

    let expanded = quote! {
        #struct_vis const #const_name: ::ktstr::test_support::Payload =
            ::ktstr::test_support::Payload::new(
                #payload_name,
                ::ktstr::test_support::PayloadKind::Binary(#binary),
                #output_tokens,
                &[#(#default_args),*],
                &[#(#default_checks),*],
                &[#(#metrics),*],
                &[#(#include_files),*],
                false,
                None,
                None,
            );
    };

    Ok(expanded)
}

/// Translate the user-facing `output = ...` expression into a
/// fully-qualified `OutputFormat` variant token stream. Accepts
/// the variant names as they appear on the `OutputFormat` enum,
/// so the attribute reads identically to `Polarity` below:
///
/// - `Json` / `ExitCode` — bare idents.
/// - `LlmExtract` — bare ident (no hint).
/// - `LlmExtract("hint")` — call with a single string literal.
/// - `LlmExtract()` — call with no args (no hint).
fn output_from_expr(expr: &syn::Expr) -> syn::Result<proc_macro2::TokenStream> {
    match expr {
        syn::Expr::Path(ep) => {
            let ident = ep.path.get_ident().ok_or_else(|| {
                syn::Error::new_spanned(expr, "expected `Json`, `ExitCode`, or `LlmExtract`")
            })?;
            match ident.to_string().as_str() {
                "Json" => Ok(quote! { ::ktstr::test_support::OutputFormat::Json }),
                "ExitCode" => Ok(quote! { ::ktstr::test_support::OutputFormat::ExitCode }),
                "LlmExtract" => {
                    Ok(quote! { ::ktstr::test_support::OutputFormat::LlmExtract(None) })
                }
                other => Err(syn::Error::new_spanned(
                    expr,
                    format!(
                        "unknown output format `{other}` (expected `Json`, `ExitCode`, or `LlmExtract`)"
                    ),
                )),
            }
        }
        syn::Expr::Call(call) => {
            // Only `LlmExtract(...)` is callable.
            let ident = match &*call.func {
                syn::Expr::Path(ep) => ep.path.get_ident().ok_or_else(|| {
                    syn::Error::new_spanned(expr, "expected `LlmExtract(...)` call form")
                })?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "expected `LlmExtract(...)` call form",
                    ));
                }
            };
            if ident != "LlmExtract" {
                return Err(syn::Error::new_spanned(
                    expr,
                    format!(
                        "unknown output format `{ident}(...)` (only `LlmExtract(...)` takes arguments)"
                    ),
                ));
            }
            match call.args.len() {
                0 => Ok(quote! { ::ktstr::test_support::OutputFormat::LlmExtract(None) }),
                1 => {
                    let arg = &call.args[0];
                    match arg {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => {
                            let hint = ls.value();
                            Ok(quote! {
                                ::ktstr::test_support::OutputFormat::LlmExtract(Some(#hint))
                            })
                        }
                        _ => Err(syn::Error::new_spanned(
                            arg,
                            "LlmExtract argument must be a string literal hint",
                        )),
                    }
                }
                _ => Err(syn::Error::new_spanned(
                    expr,
                    "LlmExtract takes at most one string literal argument",
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "output must be `Json`, `ExitCode`, `LlmExtract`, or `LlmExtract(\"hint\")`",
        )),
    }
}

/// Parse one `#[metric(name = "...", polarity = ..., unit = "...")]`
/// attribute into a `(name, MetricHint { ... } token stream)` pair.
/// The name is returned separately so the caller can check for
/// duplicate `#[metric(name = ...)]` declarations across the struct.
///
/// `polarity` accepts bare idents `HigherBetter`, `LowerBetter`,
/// `Unknown`, and the call form `TargetValue(<float literal>)`. The
/// float literal is stamped into a `Polarity::TargetValue(lit)` so
/// the generated const is const-evaluable.
fn parse_metric_attr(attr: &syn::Attribute) -> syn::Result<(String, proc_macro2::TokenStream)> {
    let mut name: Option<String> = None;
    let mut polarity: Option<proc_macro2::TokenStream> = None;
    let mut unit: String = String::new();
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            name = Some(lit.value());
            Ok(())
        } else if meta.path.is_ident("polarity") {
            let value = meta.value()?;
            let expr: syn::Expr = value.parse()?;
            polarity = Some(polarity_from_expr(&expr)?);
            Ok(())
        } else if meta.path.is_ident("unit") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            unit = lit.value();
            Ok(())
        } else {
            Err(meta.error(format!(
                "unknown metric attribute `{}` (expected name, polarity, unit)",
                meta.path
                    .get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_default()
            )))
        }
    })?;
    let name = name.ok_or_else(|| {
        syn::Error::new_spanned(attr, "metric attribute is missing `name = \"...\"`")
    })?;
    let polarity = polarity.unwrap_or_else(|| {
        quote! { ::ktstr::test_support::Polarity::Unknown }
    });
    let tokens = quote! {
        ::ktstr::test_support::MetricHint {
            name: #name,
            polarity: #polarity,
            unit: #unit,
        }
    };
    Ok((name, tokens))
}

/// Does this `#[default_check(...)]` expression already spell
/// `MetricCheck::` somewhere in its function path? Returns true for
/// `MetricCheck::min(...)` and `::ktstr::test_support::MetricCheck::min(...)`;
/// false for bare `min(...)`. Used to skip the macro's implicit
/// `::ktstr::test_support::MetricCheck::` prepend when the user has
/// already written the prefix, so `MetricCheck::MetricCheck::min(...)` can't
/// happen.
///
/// Only inspects the callee path of an `Expr::Call`; non-call
/// expressions (rare but legal: a free function returning `MetricCheck`,
/// or a `const` value) fall back to the prepend path, matching the
/// pre-bugfix behavior for anything that isn't a plain constructor
/// call. A future refactor could lift this to also handle
/// `MethodCall` / `Path`, but the MetricCheck API today is constructor
/// calls only — adding more shapes is a no-op until a new constructor
/// form lands.
fn expr_has_check_prefix(expr: &syn::Expr) -> bool {
    let syn::Expr::Call(call) = expr else {
        return false;
    };
    let syn::Expr::Path(expr_path) = &*call.func else {
        return false;
    };
    expr_path
        .path
        .segments
        .iter()
        .any(|seg| seg.ident == "MetricCheck")
}

/// Translate the user-facing `polarity = ...` expression to a
/// fully-qualified `Polarity` variant. Accepts the four enum
/// variants in bare-ident form (`HigherBetter`, `LowerBetter`,
/// `Unknown`) or as `TargetValue(<float>)`.
fn polarity_from_expr(expr: &syn::Expr) -> syn::Result<proc_macro2::TokenStream> {
    match expr {
        syn::Expr::Path(ep) => {
            let ident = ep.path.get_ident().ok_or_else(|| {
                syn::Error::new_spanned(
                    expr,
                    "expected `HigherBetter`, `LowerBetter`, `TargetValue(..)`, or `Unknown`",
                )
            })?;
            match ident.to_string().as_str() {
                "HigherBetter" => Ok(quote! { ::ktstr::test_support::Polarity::HigherBetter }),
                "LowerBetter" => Ok(quote! { ::ktstr::test_support::Polarity::LowerBetter }),
                "Unknown" => Ok(quote! { ::ktstr::test_support::Polarity::Unknown }),
                "TargetValue" => Err(syn::Error::new_spanned(
                    expr,
                    "TargetValue requires a float argument: `TargetValue(42.0)`",
                )),
                other => Err(syn::Error::new_spanned(
                    expr,
                    format!("unknown polarity `{other}`"),
                )),
            }
        }
        syn::Expr::Call(call) => {
            let ident = match &*call.func {
                syn::Expr::Path(ep) => ep.path.get_ident().ok_or_else(|| {
                    syn::Error::new_spanned(expr, "expected `TargetValue(<float>)`")
                })?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "expected `TargetValue(<float>)`",
                    ));
                }
            };
            if ident != "TargetValue" {
                return Err(syn::Error::new_spanned(
                    expr,
                    format!(
                        "unknown polarity `{ident}(...)` (only `TargetValue` takes an argument)"
                    ),
                ));
            }
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    expr,
                    "TargetValue takes exactly one float literal argument",
                ));
            }
            let arg = &call.args[0];
            let lit = match arg {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Float(lf),
                    ..
                }) => lf,
                _ => {
                    return Err(syn::Error::new_spanned(
                        arg,
                        "TargetValue argument must be a float literal (e.g. 42.0)",
                    ));
                }
            };
            Ok(quote! { ::ktstr::test_support::Polarity::TargetValue(#lit) })
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "polarity must be HigherBetter, LowerBetter, TargetValue(<float>), or Unknown",
        )),
    }
}

// ============================================================================
// #[derive(Claim)] — pointwise-claim accessor generator
// ----------------------------------------------------------------------------
//
// Emits an extension trait `<StructName>Claim` with one
// `claim_<field_name>` method per public scalar field (returning
// `ClaimBuilder<'_, FieldType>`), plus container-typed accessors:
//
//   * `BTreeSet<T>` fields → `claim_<field_name>(&mut self, s: &Struct)
//     -> SetClaim<'_, T>`, dispatched through `Verdict::claim_set`.
//   * `Vec<T>` fields → `claim_<field_name>(&mut self, s: &Struct)
//     -> SeqClaim<'_, T>`, dispatched through `Verdict::claim_seq`.
//   * `BTreeMap<K, V>` / `HashMap<K, V>` fields → SKIPPED (no claim
//     surface for maps in v1; users reach for `claim!(verdict,
//     map.len())` or `claim!(verdict, map.contains_key(&k))` against
//     the explicit expression). Maps that need first-class support
//     can be added in a follow-up — the derive emits no method, so
//     the user's call site fails to compile rather than silently
//     dispatching to a wrong type.
//
// Per-field opt-out via `#[claim(skip)]`. Non-pub fields are skipped
// automatically (the claim surface is for the test author's view of
// the struct; private fields aren't part of that view).
//
// The generated trait is named `<StructName>Claim` with the same
// visibility as the input struct. The single
// `impl <StructName>Claim for ::ktstr::assert::Verdict` lives in the
// same expansion so callers only need `use ::ktstr::prelude::*` (which
// re-exports the trait) to bring the methods into scope.
//
// Label source: every method body is `verdict.claim(stringify!(field),
// ...)` — the label is the field's source-text identifier. Renaming
// the field updates both the method name AND the rendered failure
// label in lock-step; a stale call site that referenced the old
// method name fails to compile. This is the compile-mechanical
// drift-free axis that the design rejected manual-string labels in
// favor of.

/// Detect whether a `syn::Type` is a path whose last segment matches
/// `name` (e.g. `BTreeSet<T>` → `is_path_named(ty, "BTreeSet") == true`).
/// Used to dispatch container fields onto `claim_set` / `claim_seq`
/// without forcing the caller to import `BTreeSet` / `Vec` from a
/// specific module path. Misses `std::collections::BTreeSet<T>` if the
/// caller uses an alias — acceptable in v1 because the project's
/// stats structs all use the canonical `BTreeSet` / `Vec` names.
fn is_path_named(ty: &syn::Type, name: &str) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(ident) = path_last_segment_ident(&tp.path)
    {
        return ident == name;
    }
    false
}

/// Inner element type from a `Container<T>` syn::Type. Returns `None`
/// when the path has no angle-bracketed argument or the argument is
/// not a type (e.g. lifetime-only). Used by [`derive_claim_inner`] to
/// thread the element type through the emitted accessor's return
/// type.
fn first_type_arg(ty: &syn::Type) -> Option<&syn::Type> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
        && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
    {
        for arg in &args.args {
            if let syn::GenericArgument::Type(inner) = arg {
                return Some(inner);
            }
        }
    }
    None
}

/// Detect a `#[claim(skip)]` attribute on a field. Returns true when
/// any `#[claim(skip)]` is present — the field gets no claim accessor
/// in the emitted trait.
fn field_has_claim_skip(field: &syn::Field) -> bool {
    for attr in &field.attrs {
        if !attr.path().is_ident("claim") {
            continue;
        }
        let mut skip = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
            }
            Ok(())
        });
        if skip {
            return true;
        }
    }
    false
}

/// Generate per-field claim accessors on a stats struct.
///
/// See the section comment above this fn for the dispatch rules and
/// label invariant. Reject non-struct inputs and tuple-struct inputs
/// — the claim API is keyed on field names, which tuple structs do
/// not have.
#[proc_macro_derive(Claim, attributes(claim))]
pub fn derive_claim(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_claim_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn derive_claim_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;
    let struct_vis = &input.vis;
    let trait_name = format_ident!("{}Claim", struct_name);

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    struct_name,
                    "Claim cannot be derived for tuple structs (claim labels need field names)",
                ));
            }
            Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    struct_name,
                    "Claim cannot be derived for unit structs (no fields to claim against)",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                struct_name,
                "Claim can only be derived for structs",
            ));
        }
    };

    let mut trait_methods: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut impl_methods: Vec<proc_macro2::TokenStream> = Vec::new();

    for field in fields {
        // Private fields are not part of the claim surface — the
        // generated accessors must be callable from outside the
        // defining crate. A private field would force the impl
        // body to dereference an inaccessible field path. Skip
        // silently rather than erroring; the field is invisible
        // to the API surface either way.
        if !matches!(field.vis, syn::Visibility::Public(_)) {
            continue;
        }
        if field_has_claim_skip(field) {
            continue;
        }
        let Some(field_name) = field.ident.as_ref() else {
            continue;
        };
        let method_name = format_ident!("claim_{}", field_name);
        let field_ty = &field.ty;

        // Container dispatch: BTreeSet<T> / Vec<T> route through
        // claim_set / claim_seq with element type T. Map types are
        // skipped (no method emitted).
        if is_path_named(field_ty, "BTreeSet") {
            let Some(elem) = first_type_arg(field_ty) else {
                continue;
            };
            trait_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SetClaim<'a, #elem>;
            });
            impl_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SetClaim<'a, #elem> {
                    verdict.claim_set(stringify!(#field_name), &self.#field_name)
                }
            });
            continue;
        }
        if is_path_named(field_ty, "Vec") {
            let Some(elem) = first_type_arg(field_ty) else {
                continue;
            };
            trait_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SeqClaim<'a, #elem>;
            });
            impl_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SeqClaim<'a, #elem> {
                    verdict.claim_seq(stringify!(#field_name), &self.#field_name)
                }
            });
            continue;
        }
        if is_path_named(field_ty, "BTreeMap") || is_path_named(field_ty, "HashMap") {
            // Skip map fields — no claim surface in v1.
            continue;
        }

        // Scalar field. Emit a method returning `ClaimBuilder<'_,
        // FieldType>` that copies/clones the value. Cloning compiles
        // for both `Copy` and `Clone` types; primitive fields lower
        // to a single move at -O.
        trait_methods.push(quote! {
            fn #method_name<'a>(
                &'a self,
                verdict: &'a mut ::ktstr::assert::Verdict,
            ) -> ::ktstr::assert::ClaimBuilder<'a, #field_ty>;
        });
        impl_methods.push(quote! {
            fn #method_name<'a>(
                &'a self,
                verdict: &'a mut ::ktstr::assert::Verdict,
            ) -> ::ktstr::assert::ClaimBuilder<'a, #field_ty> {
                verdict.claim(stringify!(#field_name), ::core::clone::Clone::clone(&self.#field_name))
            }
        });
    }

    let doc = format!(
        "Pointwise-claim accessors generated by `#[derive(Claim)]` on \
         [`{name}`]. One `claim_<field>` method per public field, taking \
         `&mut Verdict` as the accumulator; container fields (`BTreeSet`/`Vec`) \
         route through `SetClaim`/`SeqClaim`. Method dispatch keys on the \
         stats struct's type, so identical field names across distinct stats \
         structs do not collide. Brought into scope via `use ktstr::prelude::*`.",
        name = struct_name,
    );

    Ok(quote! {
        #[doc = #doc]
        #struct_vis trait #trait_name {
            #(#trait_methods)*
        }

        impl #trait_name for #struct_name {
            #(#impl_methods)*
        }
    })
}

/// Convert JSON-like Rust tokens into a `&'static str` at compile time.
///
/// Accepts a superset of JSON syntax using Rust token trees:
/// - Objects: `{ "key": value, ... }`
/// - Arrays: `[value, ...]`
/// - Strings: `"hello"`
/// - Numbers: `42`, `3.14`, `-1`
/// - Booleans: `true`, `false`
/// - Null: `null`
/// - Trailing commas are stripped
///
/// ```rust,ignore
/// const CFG: &str = ktstr::json!({
///     "layers": [{
///         "name": "batch",
///         "kind": { "Grouped": { "cpus_range": [0, 4] } },
///     }],
/// });
/// ```
#[proc_macro]
pub fn json(input: TokenStream) -> TokenStream {
    let mut out = String::new();
    tokens_to_json(&mut out, proc_macro2::TokenStream::from(input));
    let lit = syn::LitStr::new(&out, proc_macro2::Span::call_site());
    TokenStream::from(quote! { #lit })
}

fn tokens_to_json(out: &mut String, tokens: proc_macro2::TokenStream) {
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Group(g) => match g.delimiter() {
                proc_macro2::Delimiter::Brace => {
                    out.push('{');
                    emit_comma_separated(out, g.stream(), true);
                    out.push('}');
                }
                proc_macro2::Delimiter::Bracket => {
                    out.push('[');
                    emit_comma_separated(out, g.stream(), false);
                    out.push(']');
                }
                proc_macro2::Delimiter::Parenthesis => {
                    out.push('(');
                    tokens_to_json(out, g.stream());
                    out.push(')');
                }
                proc_macro2::Delimiter::None => {
                    tokens_to_json(out, g.stream());
                }
            },
            proc_macro2::TokenTree::Literal(lit) => {
                out.push_str(&lit.to_string());
            }
            proc_macro2::TokenTree::Ident(id) => {
                let s = id.to_string();
                match s.as_str() {
                    "true" | "false" | "null" => out.push_str(&s),
                    _ => {
                        out.push('"');
                        out.push_str(&s);
                        out.push('"');
                    }
                }
            }
            proc_macro2::TokenTree::Punct(p) => {
                let ch = p.as_char();
                if ch == '-' {
                    out.push('-');
                } else if ch == ':' {
                    out.push(':');
                } else if ch == ',' {
                    out.push(',');
                } else {
                    out.push(ch);
                }
            }
        }
    }
}

fn emit_comma_separated(out: &mut String, tokens: proc_macro2::TokenStream, _is_object: bool) {
    let items = split_on_commas(tokens);
    let mut first = true;
    for item in &items {
        if item.is_empty() {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        tokens_to_json(out, item.clone());
    }
}

fn split_on_commas(tokens: proc_macro2::TokenStream) -> Vec<proc_macro2::TokenStream> {
    let mut result = Vec::new();
    let mut current = proc_macro2::TokenStream::new();
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => {
                result.push(current);
                current = proc_macro2::TokenStream::new();
            }
            _ => {
                current.extend(std::iter::once(tt));
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn camel_to_screaming_snake_acronym_run() {
        assert_eq!(camel_to_screaming_snake("HTTPServer"), "HTTP_SERVER");
    }

    #[test]
    fn camel_to_screaming_snake_single_word() {
        assert_eq!(camel_to_screaming_snake("Llc"), "LLC");
    }

    #[test]
    fn camel_to_screaming_snake_all_caps_passthrough() {
        assert_eq!(camel_to_screaming_snake("LLC"), "LLC");
    }

    #[test]
    fn option_tokens_some_int() {
        let opt: Option<u32> = Some(42);
        let ts = option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { Some(42u32) }.to_string());
    }

    #[test]
    fn option_tokens_none_int() {
        let opt: Option<u32> = None;
        let ts = option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { None }.to_string());
    }

    #[test]
    fn option_tokens_some_bool() {
        let opt: Option<bool> = Some(true);
        let ts = option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { Some(true) }.to_string());
    }

    /// Contract pin: `AttrValues::default()` is the single source of
    /// truth for every `#[ktstr_test]` macro default since step 2/4 of
    /// the parse-loop refactor. Without a field-by-field positive
    /// assertion a maintainer editing the [`Default`] impl can shift
    /// any user-visible default (auto_repro, kaslr, memory_mib, the
    /// gauntlet caps, etc.) with zero test feedback. Same precedent
    /// as `host_mode_default_cgroup_parent_resolves` in
    /// tests/host_mode_e2e.rs pinning a runtime const against
    /// production source.
    #[test]
    fn attr_values_default_matches_documented_macro_defaults() {
        let d = AttrValues::default();

        // -- Topology --
        assert_eq!(d.llcs, DEFAULT_LLCS);
        assert_eq!(d.cores, DEFAULT_CORES);
        assert_eq!(d.threads, DEFAULT_THREADS);
        assert_eq!(d.numa_nodes, 1);
        assert!(!d.llcs_set);
        assert!(!d.cores_set);
        assert!(!d.threads_set);
        assert!(!d.numa_nodes_set);

        // -- Memory + duration --
        assert_eq!(d.memory_mib, DEFAULT_MEMORY_MIB);
        assert!(!d.memory_mib_set);
        assert_eq!(d.duration_s, 2);
        assert!(!d.duration_s_set);
        assert_eq!(d.cleanup_budget_ms, None);
        assert_eq!(d.watchdog_timeout_s, 4);
        assert!(!d.watchdog_timeout_s_set);

        // -- Scheduler refs --
        assert!(d.scheduler.is_none());
        assert!(d.payload.is_none());
        assert!(d.workloads.is_none());
        assert!(d.staged_schedulers.is_none());
        assert!(d.bpf_map_write.is_none());
        assert!(d.post_vm.is_none());
        assert!(d.post_vm_unconditional.is_none());
        assert!(d.disk.is_none());

        // -- Assert overrides --
        assert_eq!(d.not_starved, None);
        assert_eq!(d.isolation, None);
        assert_eq!(d.max_gap_ms, None);
        assert_eq!(d.max_spread_pct, None);
        assert_eq!(d.max_imbalance_ratio, None);
        assert_eq!(d.max_local_dsq_depth, None);
        assert_eq!(d.fail_on_stall, None);
        assert_eq!(d.sustained_samples, None);
        assert_eq!(d.max_throughput_cv, None);
        assert_eq!(d.min_work_rate, None);
        assert_eq!(d.max_fallback_rate, None);
        assert_eq!(d.max_keep_last_rate, None);
        assert_eq!(d.max_p99_wake_latency_ns, None);
        assert_eq!(d.max_wake_latency_cv, None);
        assert_eq!(d.min_iteration_rate, None);
        assert_eq!(d.max_migration_ratio, None);
        assert_eq!(d.min_page_locality, None);
        assert_eq!(d.max_cross_node_migration_ratio, None);
        assert_eq!(d.max_slow_tier_ratio, None);

        // -- TopologyConstraints --
        assert_eq!(d.min_numa_nodes, 1);
        assert!(!d.min_numa_nodes_set);
        assert_eq!(d.min_llcs, 1);
        assert!(!d.min_llcs_set);
        assert!(!d.requires_smt);
        assert!(!d.requires_smt_set);
        assert_eq!(d.min_cpus, 1);
        assert!(!d.min_cpus_set);
        assert_eq!(d.max_llcs, Some(12));
        assert!(!d.max_llcs_set);
        assert_eq!(d.max_numa_nodes, Some(1));
        assert!(!d.max_numa_nodes_set);
        assert_eq!(d.max_cpus, Some(192));
        assert!(!d.max_cpus_set);
        assert_eq!(d.cpu_budget, None);

        // -- Bool attrs (auto_repro + kaslr default TRUE; others false) --
        assert!(d.auto_repro);
        assert!(!d.auto_repro_set);
        assert!(!d.expect_auto_repro);
        assert!(!d.expect_auto_repro_set);
        assert!(!d.performance_mode);
        assert!(!d.performance_mode_set);
        assert!(!d.no_perf_mode);
        assert!(!d.no_perf_mode_set);
        assert!(!d.expect_err);
        assert!(!d.expect_err_set);
        assert!(!d.allow_inconclusive);
        assert!(!d.allow_inconclusive_set);
        assert!(!d.host_only);
        assert!(!d.host_only_set);
        assert!(!d.ignore_test);
        assert!(d.kaslr);
        assert!(!d.kaslr_set);
        assert!(!d.wprof);
        assert!(!d.wprof_set);
        assert_eq!(d.num_snapshots, 0);
        assert!(!d.num_snapshots_set);

        // -- Strings + tokens --
        assert!(d.extra_sched_args.is_empty());
        assert!(d.extra_include_files.is_empty());
        assert_eq!(d.workload_root_cgroup, None);
        assert!(d.wprof_args.is_none());
        assert!(d.expect_scx_bpf_error_contains_tokens.is_none());
        assert!(d.expect_scx_bpf_error_matches_tokens.is_none());
        assert!(d.config_expr.is_none());
        assert!(!d.config_set);
    }

    // -- expect_auto_repro macro-parse positive tests --
    //
    // Synthesize each attribute spelling, invoke ktstr_test_impl
    // directly (bypassing proc_macro::TokenStream which panics outside
    // a procedural-macro invocation), parse the output back into a
    // syn AST, locate the `static __KTSTR_ENTRY_*: KtstrTestEntry =
    // KtstrTestEntry { ... };` registration emitted by the macro, and
    // assert the `expect_auto_repro` field is either:
    //   - absent (omitted spelling — DEFAULT spread carries false), or
    //   - present with a `Lit::Bool { value: true/false }` value.
    //
    // The AST round-trip (rather than substring matching on the
    // output's `.to_string()`) guards against two failure modes:
    //   1. proc_macro2 version drift in colon/whitespace formatting —
    //      a future proc_macro2 release that emits `field: true` (no
    //      space) vs `field : true` (current) would silently flip a
    //      substring-matching test from PASS to FAIL or vice versa.
    //   2. structural defects that a substring check cannot detect —
    //      wrong outer struct name, wrong field value type (e.g. a
    //      String literal where a bool is expected), or a phantom
    //      sub-literal elsewhere in the output that happens to
    //      contain the same substring.
    //
    // For spellings that set expect_auto_repro = true, the cross-
    // attribute validation pass (added at the same time as the field)
    // requires a scheduler attribute + wprof attribute to be present.
    // The fixture inputs satisfy those preconditions so the parser
    // reaches codegen without rejection.

    /// Type-erased extraction from a `syn::Expr` for the
    /// [`field_value_in_static_entry`] helper. Each impl panics with
    /// a descriptive error if the expression shape doesn't match the
    /// expected literal kind — same wrong-type-rejection contract as
    /// the single-purpose helper this generalization replaces.
    trait ExtractFromExpr: Sized {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self;
    }

    impl ExtractFromExpr for bool {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Bool(b),
                    ..
                }) => b.value,
                other => panic!("{field_name} field value must be a Lit::Bool; got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for u32 {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(i),
                    ..
                }) => i
                    .base10_parse::<u32>()
                    .unwrap_or_else(|e| panic!("{field_name} field value parse as u32: {e}")),
                other => panic!("{field_name} field value must be a Lit::Int (u32); got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for u64 {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(i),
                    ..
                }) => i
                    .base10_parse::<u64>()
                    .unwrap_or_else(|e| panic!("{field_name} field value parse as u64: {e}")),
                other => panic!("{field_name} field value must be a Lit::Int (u64); got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for String {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => s.value(),
                other => panic!("{field_name} field value must be a Lit::Str; got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for syn::Path {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Path(p) => p.path.clone(),
                other => panic!("{field_name} field value must be an Expr::Path; got {other:?}"),
            }
        }
    }

    /// Locate the `__KTSTR_ENTRY_*` static the macro emits and return
    /// the value of its `field_name` field if explicitly set, or
    /// `None` if the field is absent from the struct literal
    /// (omitted spellings fall through to the
    /// `..KtstrTestEntry::DEFAULT` spread).
    ///
    /// Generalized from the single-field `expect_auto_repro` helper
    /// to read any field whose value-type implements
    /// [`ExtractFromExpr`] (bool, u32, u64, String, syn::Path).
    /// The wrong-value-type rejection moves into the per-type
    /// `extract_or_panic` impl — a field whose expression doesn't
    /// match the requested type panics with a message naming both
    /// the field and the expected type.
    ///
    /// Verifies along the way:
    /// - exactly one `__KTSTR_ENTRY_*` static is emitted (panics on
    ///   zero or multiple — a future codegen change that emitted
    ///   two prefixed statics would silently key off the first
    ///   without the count assertion),
    /// - its declared type's last path segment is `KtstrTestEntry`,
    /// - its initializer is a struct literal whose path's last
    ///   segment is `KtstrTestEntry` (catches a regression that
    ///   wrapped the literal in a different outer struct),
    /// - any present `field_name` field's value matches the
    ///   requested type (panics via [`ExtractFromExpr::extract_or_panic`]
    ///   otherwise).
    fn field_value_in_static_entry<T: ExtractFromExpr>(
        output: &proc_macro2::TokenStream,
        field_name: &str,
    ) -> Option<T> {
        let file: syn::File =
            syn::parse2(output.clone()).expect("macro output must parse as a syn::File");
        let static_candidates: Vec<&syn::ItemStatic> = file
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Static(s) if s.ident.to_string().starts_with("__KTSTR_ENTRY_") => {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            static_candidates.len(),
            1,
            "macro must emit exactly one __KTSTR_ENTRY_* static; found {}",
            static_candidates.len()
        );
        let static_item = static_candidates[0];
        let static_type_last = match static_item.ty.as_ref() {
            syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
            _ => None,
        };
        assert_eq!(
            static_type_last.as_deref(),
            Some("KtstrTestEntry"),
            "static type's last path segment must be KtstrTestEntry"
        );
        let expr_struct = match static_item.expr.as_ref() {
            syn::Expr::Struct(s) => s,
            other => panic!("static initializer must be a struct literal; got {other:?}"),
        };
        let struct_last = expr_struct
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string());
        assert_eq!(
            struct_last.as_deref(),
            Some("KtstrTestEntry"),
            "struct-literal path's last segment must be KtstrTestEntry"
        );
        for field in &expr_struct.fields {
            let ident_matches = matches!(
                &field.member,
                syn::Member::Named(ident) if ident == field_name
            );
            if !ident_matches {
                continue;
            }
            return Some(T::extract_or_panic(field_name, &field.expr));
        }
        None
    }

    /// `#[ktstr_test(expect_auto_repro)]` (bare form) emits
    /// `expect_auto_repro: true` as a field on the
    /// `KtstrTestEntry` struct literal. Pins the bare-flag arm of
    /// the macro's bool-slot parser. Gated on the `wprof` feature: the
    /// attr pairs `wprof` with `expect_auto_repro` (the latter requires the
    /// former), and the macro only accepts `wprof` when the feature is on —
    /// so this case can only parse successfully under `--features wprof`.
    #[cfg(feature = "wprof")]
    #[test]
    fn macro_parses_expect_auto_repro_bare_to_true() {
        let attr = quote! { scheduler = SCHED, wprof, expect_auto_repro };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test_impl(attr, item).expect("bare attribute must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            Some(true),
            "bare `expect_auto_repro` must emit a `expect_auto_repro: true` field"
        );
    }

    /// `#[ktstr_test(expect_auto_repro = true)]` emits
    /// `expect_auto_repro: true`. Pins the explicit-true arm. Gated on the
    /// `wprof` feature (the attr requires `wprof`, accepted only with the
    /// feature on).
    #[cfg(feature = "wprof")]
    #[test]
    fn macro_parses_expect_auto_repro_explicit_true() {
        let attr = quote! { scheduler = SCHED, wprof, expect_auto_repro = true };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out =
            ktstr_test_impl(attr, item).expect("explicit-true attribute must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            Some(true),
            "explicit `expect_auto_repro = true` must emit a `expect_auto_repro: true` field"
        );
    }

    /// `#[ktstr_test(expect_auto_repro = false)]` emits
    /// `expect_auto_repro: false`. Pins the explicit-false arm
    /// against a regression that conflated explicit-false with
    /// omission (which would silently leave DEFAULT untouched and
    /// lose the user's negative declaration). No cross-attribute
    /// gates apply when expect_auto_repro is false — the only
    /// rejection arms trigger on the true value.
    #[test]
    fn macro_parses_expect_auto_repro_explicit_false() {
        let attr = quote! { expect_auto_repro = false };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out =
            ktstr_test_impl(attr, item).expect("explicit-false attribute must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            Some(false),
            "explicit `expect_auto_repro = false` must emit a `expect_auto_repro: false` field"
        );
    }

    /// Omitting the attribute entirely emits NO
    /// `expect_auto_repro` field — the generated struct literal
    /// uses the `..KtstrTestEntry::DEFAULT` spread to inherit the
    /// false default. Pins backward-compat: an existing
    /// `#[ktstr_test(...)]` with no expect_auto_repro must not
    /// gain a phantom field that flips the entry's behavior.
    #[test]
    fn macro_parses_omitted_expect_auto_repro_leaves_field_unemitted() {
        let attr = quote! {};
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out =
            ktstr_test_impl(attr, item).expect("attribute-less invocation must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            None,
            "omitted attribute must NOT emit any `expect_auto_repro` field — DEFAULT spread carries the false"
        );
    }

    // -- check_visible_lit (commit f4018278) --------------------------

    #[test]
    fn check_visible_lit_visible_string_passes() {
        let expr: syn::Expr = syn::parse_quote!("hello");
        check_visible_lit("hello", &expr, "name").expect("non-empty visible string must pass");
    }

    #[test]
    fn check_visible_lit_empty_string_rejected() {
        let expr: syn::Expr = syn::parse_quote!("");
        let err = check_visible_lit("", &expr, "name").unwrap_err();
        assert!(
            err.to_string()
                .contains("`name` must contain at least one visible character"),
            "expected `name` visible-empty diagnostic, got: {err}"
        );
    }

    #[test]
    fn check_visible_lit_whitespace_only_rejected() {
        let expr: syn::Expr = syn::parse_quote!("   ");
        let err = check_visible_lit("   ", &expr, "binary").unwrap_err();
        assert!(
            err.to_string()
                .contains("`binary` must contain at least one visible character"),
            "expected `binary` visible-empty diagnostic, got: {err}"
        );
    }

    #[test]
    fn check_visible_lit_invisible_only_rejected() {
        let expr: syn::Expr = syn::parse_quote!("zwsp");
        let err = check_visible_lit("\u{200B}", &expr, "binary_path").unwrap_err();
        assert!(
            err.to_string()
                .contains("`binary_path` must contain at least one visible character"),
            "expected `binary_path` visible-empty diagnostic, got: {err}"
        );
    }

    // -- validate_kernel_builtin_pair (commit 753ecf9e) ---------------

    #[test]
    fn validate_kernel_builtin_pair_both_set_passes() {
        let span = proc_macro2::Span::call_site();
        validate_kernel_builtin_pair(Some(span), Some(span))
            .expect("both set is valid (KernelBuiltin)");
    }

    #[test]
    fn validate_kernel_builtin_pair_neither_set_passes() {
        validate_kernel_builtin_pair(None, None).expect("neither set is valid (not KernelBuiltin)");
    }

    #[test]
    fn validate_kernel_builtin_pair_enable_only_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = validate_kernel_builtin_pair(Some(span), None).unwrap_err();
        assert!(
            err.to_string()
                .contains("`kernel_builtin_enable` set without `kernel_builtin_disable`"),
            "expected enable-without-disable diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_kernel_builtin_pair_disable_only_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = validate_kernel_builtin_pair(None, Some(span)).unwrap_err();
        assert!(
            err.to_string()
                .contains("`kernel_builtin_disable` set without `kernel_builtin_enable`"),
            "expected disable-without-enable diagnostic, got: {err}"
        );
    }

    // -- validate_exactly_one_source (commit 7c796939) ----------------

    #[test]
    fn validate_exactly_one_source_none_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = validate_exactly_one_source(false, false, false, span).unwrap_err();
        assert!(
            err.to_string().contains("no scheduler source declared"),
            "expected no-source diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_exactly_one_source_only_binary_passes() {
        let span = proc_macro2::Span::call_site();
        validate_exactly_one_source(true, false, false, span).expect("binary-only is valid");
    }

    #[test]
    fn validate_exactly_one_source_only_binary_path_passes() {
        let span = proc_macro2::Span::call_site();
        validate_exactly_one_source(false, true, false, span).expect("binary_path-only is valid");
    }

    #[test]
    fn validate_exactly_one_source_only_kernel_builtin_passes() {
        let span = proc_macro2::Span::call_site();
        validate_exactly_one_source(false, false, true, span)
            .expect("kernel_builtin-only is valid");
    }

    #[test]
    fn validate_exactly_one_source_binary_and_path_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = validate_exactly_one_source(true, true, false, span).unwrap_err();
        assert!(
            err.to_string()
                .contains("more than one scheduler source declared"),
            "expected multi-source diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_exactly_one_source_all_three_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = validate_exactly_one_source(true, true, true, span).unwrap_err();
        assert!(
            err.to_string()
                .contains("more than one scheduler source declared"),
            "expected multi-source diagnostic, got: {err}"
        );
    }

    // -- validate_kernel_name_collision (commit 7480df1c) -------------

    #[test]
    fn validate_kernel_name_collision_non_kernel_passes() {
        let expr: syn::Expr = syn::parse_quote!("scx_mitosis");
        validate_kernel_name_collision(true, "scx_mitosis", Some(&expr))
            .expect("non-`kernel` name is valid");
    }

    #[test]
    fn validate_kernel_name_collision_not_kernel_builtin_passes() {
        let expr: syn::Expr = syn::parse_quote!("kernel");
        validate_kernel_name_collision(false, "kernel", Some(&expr))
            .expect("`kernel` name is valid when not KernelBuiltin variant");
    }

    #[test]
    fn validate_kernel_name_collision_exact_kernel_rejected() {
        let expr: syn::Expr = syn::parse_quote!("kernel");
        let err = validate_kernel_name_collision(true, "kernel", Some(&expr)).unwrap_err();
        assert!(
            err.to_string()
                .contains("collides with the KernelBuiltin variant's display_name"),
            "expected collision diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_kernel_name_collision_case_insensitive_rejected() {
        let expr: syn::Expr = syn::parse_quote!("Kernel");
        let err = validate_kernel_name_collision(true, "Kernel", Some(&expr)).unwrap_err();
        assert!(
            err.to_string()
                .contains("collides with the KernelBuiltin variant's display_name"),
            "expected case-insensitive collision diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_kernel_name_collision_whitespace_padded_rejected() {
        let expr: syn::Expr = syn::parse_quote!("  Kernel  ");
        let err = validate_kernel_name_collision(true, "  Kernel  ", Some(&expr)).unwrap_err();
        assert!(
            err.to_string()
                .contains("collides with the KernelBuiltin variant's display_name"),
            "expected whitespace-insensitive collision diagnostic, got: {err}"
        );
    }

    // -- validate_payload_workloads_dedup (commit 26352fd7) -----------

    #[test]
    fn validate_payload_workloads_dedup_empty_workloads_passes() {
        let payload: Option<syn::Path> = Some(syn::parse_quote!(FIO));
        validate_payload_workloads_dedup(&payload, &[]).expect("empty workloads is valid");
    }

    #[test]
    fn validate_payload_workloads_dedup_disjoint_passes() {
        let payload: Option<syn::Path> = Some(syn::parse_quote!(FIO));
        let workloads: Vec<syn::Path> =
            vec![syn::parse_quote!(STRESS_NG), syn::parse_quote!(NETPERF)];
        validate_payload_workloads_dedup(&payload, &workloads)
            .expect("disjoint workloads is valid");
    }

    #[test]
    fn validate_payload_workloads_dedup_primary_in_workloads_rejected() {
        let payload: Option<syn::Path> = Some(syn::parse_quote!(FIO));
        let workloads: Vec<syn::Path> = vec![syn::parse_quote!(STRESS_NG), syn::parse_quote!(FIO)];
        let err = validate_payload_workloads_dedup(&payload, &workloads).unwrap_err();
        assert!(
            err.to_string().contains("appears in both"),
            "expected payload-in-workloads diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_payload_workloads_dedup_pairwise_duplicate_rejected() {
        let payload: Option<syn::Path> = None;
        let workloads: Vec<syn::Path> = vec![syn::parse_quote!(FIO), syn::parse_quote!(FIO)];
        let err = validate_payload_workloads_dedup(&payload, &workloads).unwrap_err();
        assert!(
            err.to_string().contains("appears twice"),
            "expected pairwise-dupe diagnostic, got: {err}"
        );
    }
}
