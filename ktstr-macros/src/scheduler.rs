//! `declare_scheduler!` implementation: parses the scheduler-definition
//! DSL into a `Scheduler` static, validating kernel/binary sources,
//! const-eligibility, and topology constraints. The
//! `#[proc_macro] declare_scheduler` entry point lives in `lib.rs` (Rust
//! requires proc-macro fns at the crate root); this module holds the
//! parse + validation + codegen it calls.

use quote::{ToTokens, format_ident, quote};

pub(crate) fn declare_scheduler_inner(
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
    let mut sched_verifier_exclude_topologies: Vec<String> = Vec::new();
    let mut sched_verifier_exclude_topologies_set = false;
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
                                crate::kernel_path::KERNEL_ID_GRAMMAR,
                            ),
                        ));
                    }
                    // Run the same `KernelId::parse` + `validate` the
                    // verifier uses so any malformed entry — inverted
                    // range, suspicious `..` substring that fails the
                    // range grammar — is caught at the call site
                    // rather than as a confusing runtime "cache key
                    // not found" error.
                    let parsed = crate::kernel_path::KernelId::parse(&s);
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
                    if s.contains("..")
                        && matches!(parsed, crate::kernel_path::KernelId::CacheKey(_))
                    {
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
            "verifier_exclude_topologies" => {
                sched_verifier_exclude_topologies_set = true;
                let arr = expect_array(&value, &key, "verifier_exclude_topologies")?;
                for elem in &arr.elems {
                    let preset = expect_str_lit_element(elem, "verifier_exclude_topologies")?;
                    check_visible_lit(&preset, elem, "verifier_exclude_topologies entry")?;
                    if preset.contains('/') {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!(
                                "declare_scheduler!: verifier topology preset \
                                 `{preset}` must not contain `/` because verifier \
                                 cell names use `/` as a path separator"
                            ),
                        ));
                    }
                    if sched_verifier_exclude_topologies.contains(&preset) {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!(
                                "declare_scheduler!: duplicate \
                                 `verifier_exclude_topologies` entry `{preset}`"
                            ),
                        ));
                    }
                    sched_verifier_exclude_topologies.push(preset);
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
                // fns (`Assert::NO_OVERRIDES.check_not_stuck()...`),
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
    if sched_verifier_exclude_topologies_set {
        builder_chain = quote! {
            #builder_chain.verifier_exclude_topologies(
                &[#(#sched_verifier_exclude_topologies),*]
            )
        };
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
    // Capture the DECLARING crate's manifest dir. These tokens compile in
    // the invoking crate, so `env!("CARGO_MANIFEST_DIR")` resolves to that
    // crate's directory — a `binary = "pkg"` scheduler is built with
    // `cargo build -p pkg` run from there, i.e. in its own workspace.
    builder_chain = quote! {
        #builder_chain.manifest_dir(::core::env!("CARGO_MANIFEST_DIR"))
    };

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
pub(crate) fn check_visible_lit(lit: &str, value: &syn::Expr, field: &str) -> syn::Result<()> {
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
pub(crate) fn validate_kernel_builtin_pair(
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
pub(crate) fn validate_kernel_name_collision(
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
pub(crate) fn validate_exactly_one_source(
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
/// interpreter at `src/vmm/rust_init/dump.rs`'s `exec_shell_line`. Anything
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
                     `>` (`exec_shell_line` in `src/vmm/rust_init/dump.rs` \
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
    /// `Assert::NO_OVERRIDES.check_not_stuck()` chain pattern.
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
    crate::common::path_last_segment_ident(&ep.path).is_some_and(|ident| {
        let s = ident.to_string();
        s.chars().next().is_some_and(|c| c.is_ascii_uppercase()) || s == "new" || s == "default"
    })
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
    crate::common::path_last_segment_ident(&ep.path).is_some_and(|ident| {
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
     `Assert::NO_OVERRIDES.check_not_stuck().max_gap_ms(50)`.";

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
    if crate::common::path_last_segment_ident(&ep.path)? != "Some" {
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
