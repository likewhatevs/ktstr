//! `#[derive(Payload)]` implementation: emits a `Payload` const from a
//! marker struct's `#[payload]` / `#[default_args]` / `#[default_check]`
//! / `#[metric]` / `#[include_files]` attributes. The
//! `#[proc_macro_derive(Payload)]` entry point lives in `lib.rs` (Rust
//! requires proc-macro fns at the crate root); this module holds the
//! expansion it calls.

use quote::{format_ident, quote};
use syn::{Data, DeriveInput};

/// Convert a CamelCase identifier to SCREAMING_SNAKE_CASE.
///
/// Handles acronyms (consecutive uppercase): a separator is inserted
/// before the last letter of a run when followed by lowercase.
///
/// `Llc` -> `"LLC"`, `RejectPin` -> `"REJECT_PIN"`, `NoCtrl` -> `"NO_CTRL"`,
/// `LLC` -> `"LLC"`, `HTTPServer` -> `"HTTP_SERVER"`.
pub(crate) fn camel_to_screaming_snake(s: &str) -> String {
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

pub(crate) fn derive_payload_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
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
