//! `#[ktstr_scenario]` — the restricted, extractable sibling of
//! `#[ktstr_test]`.
//!
//! A `#[ktstr_test]` body is arbitrary Rust that receives a `&Ctx` and
//! returns an `AssertResult`. That generality is necessary for tests
//! that assert on run output, forward guest kmsg, or drive host-side
//! callbacks — and it is exactly what makes a test's *workload*
//! unreadable to anything but the guest that runs it.
//!
//! `#[ktstr_scenario]` narrows the entrypoint to the case where the
//! body does nothing but describe a workload. Its function takes no
//! arguments and returns anything convertible into a
//! `ktstr::scenario::ScenarioDef`, so the workload becomes a
//! host-buildable value. (Spelled plainly rather than as an intra-doc
//! link: `ktstr` depends on this crate, not the other way round, so
//! the type is not nameable from here.)
//!
//! # Expansion
//!
//! The macro does NOT fork `#[ktstr_test]`. It emits three items and
//! hands the third to [`crate::ktstr_test::ktstr_test_impl`]
//! verbatim:
//!
//! 1. the author's function, renamed `__ktstr_scenario_body_<name>`,
//!    body and attributes untouched;
//! 2. `__ktstr_scenario_def_<name>() -> ScenarioDef`, which coerces
//!    (1)'s return value via `Into`, plus a `ScenarioEntry`
//!    registration in the `KTSTR_SCENARIOS` slice pointing at it;
//! 3. `fn <name>(ctx: &Ctx) -> Result<AssertResult> {
//!    __ktstr_scenario_def_<name>().run(ctx) }`, passed to
//!    `ktstr_test_impl` with the attribute list unchanged.
//!
//! Because (3) goes through the ordinary `#[ktstr_test]` path, every
//! attribute, every cross-attribute validation, and every registration
//! the existing macro performs applies identically — there is no
//! second implementation to drift. The runtime path is likewise
//! unchanged: `ScenarioDef::run` calls `execute_steps_with`, which is
//! what a hand-written body calls.
//!
//! # Naming
//!
//! `ktstr_scenario` is PROVISIONAL. The name appears in exactly two
//! places that are not derived from it — the `#[proc_macro_attribute]`
//! function in `lib.rs` and its `pub use` in the `ktstr` crate root —
//! so renaming it is a mechanical change. The generated identifier
//! prefixes (`__ktstr_scenario_*`) are internal and need not track a
//! rename.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::{ItemFn, Meta};

use crate::ktstr_test::ktstr_test_impl;

/// Attributes that a scenario may not carry, and why. Each names a
/// host-side Rust callback: admitting one would put arbitrary
/// behaviour back into a form whose whole purpose is to have none.
const REJECTED_ATTRS: &[(&str, &str)] = &[
    (
        "post_vm",
        "a host-side callback is arbitrary Rust, which is what \
         `#[ktstr_scenario]` exists to exclude",
    ),
    (
        "post_vm_unconditional",
        "a host-side callback is arbitrary Rust, which is what \
         `#[ktstr_scenario]` exists to exclude",
    ),
];

/// Reject any attribute in [`REJECTED_ATTRS`].
///
/// This is a pre-pass over the same token stream `ktstr_test_impl`
/// parses; it deliberately does not validate anything else, so that
/// every other diagnostic (unknown attributes, duplicates, cross-attr
/// conflicts) comes from the one existing implementation and reads
/// identically for both entrypoints.
fn reject_callback_attrs(attr: &TokenStream) -> syn::Result<()> {
    let parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    // A parse failure here is not ours to report: hand the stream on
    // and let `ktstr_test_impl` produce its own diagnostic, which is
    // the one the author should see.
    let Ok(metas) = parser.parse2(attr.clone()) else {
        return Ok(());
    };
    for meta in &metas {
        let path = meta.path();
        let Some(ident) = path.get_ident() else {
            continue;
        };
        let name = ident.to_string();
        if let Some((_, why)) = REJECTED_ATTRS.iter().find(|(k, _)| *k == name) {
            return Err(syn::Error::new_spanned(
                path,
                format!(
                    "`{name}` is not available on `#[ktstr_scenario]` — {why}. \
                     Write the test with `#[ktstr_test]` instead: take a \
                     `ctx: &Ctx`, build the same `ScenarioDef`, and run it \
                     with `.run(ctx)`.",
                ),
            ));
        }
    }
    Ok(())
}

/// Reject a signature that cannot be evaluated on the host.
///
/// The parameter check is the load-bearing one: a builder that takes
/// `&Ctx` can only run inside a booted guest, so its workload is not
/// extractable and the entrypoint would be `#[ktstr_test]` wearing a
/// different name. The `async` / generic / `where` checks are not
/// about extractability — they are shapes the generated
/// `fn() -> ScenarioDef` wrapper cannot call, and rejecting them here
/// yields a diagnostic pointing at the signature rather than a type
/// error inside macro-generated code.
fn validate_signature(input: &ItemFn) -> syn::Result<()> {
    if let Some(first) = input.sig.inputs.first() {
        return Err(syn::Error::new_spanned(
            first,
            "a `#[ktstr_scenario]` function takes no arguments — in \
             particular no `ctx: &Ctx`. The context exists only inside \
             the running guest, so a body that reads it cannot be built \
             on the host, which is the property this entrypoint provides.\n\
             \n\
             Most `&Ctx` uses in scenario bodies do not need it: \
             `ctx.cgroup_def(\"cg_0\")` is `CgroupDef::named(\"cg_0\")` \
             with the worker count bound eagerly, and the step runner \
             already applies that same default to a `CgroupDef` that \
             leaves it unset — so `CgroupDef::named(\"cg_0\")` resolves \
             to an identical workload.\n\
             \n\
             If the body genuinely needs runtime topology, use \
             `#[ktstr_test]`.",
        ));
    }
    if let Some(asyncness) = input.sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            "a `#[ktstr_scenario]` function must not be `async` — the \
             generated builder calls it directly to produce a \
             `ScenarioDef`",
        ));
    }
    if !input.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.sig.generics,
            "a `#[ktstr_scenario]` function must not be generic — the \
             registry stores a plain `fn() -> ScenarioDef`, which has no \
             type parameters to instantiate",
        ));
    }
    if let Some(where_clause) = &input.sig.generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "a `#[ktstr_scenario]` function must not carry a `where` \
             clause — the registry stores a plain `fn() -> ScenarioDef`",
        ));
    }
    Ok(())
}

/// Expand `#[ktstr_scenario]`. See the module docs for the shape of
/// the expansion and why it delegates rather than forks.
pub(crate) fn ktstr_scenario_impl(
    attr: TokenStream,
    item: TokenStream,
) -> Result<TokenStream, syn::Error> {
    let input: ItemFn = syn::parse2(item)?;
    reject_callback_attrs(&attr)?;
    validate_signature(&input)?;

    let name = input.sig.ident.clone();
    let name_str = name.to_string();
    let body_fn = format_ident!("__ktstr_scenario_body_{}", name);
    let def_fn = format_ident!("__ktstr_scenario_def_{}", name);
    let registry_entry = format_ident!("__KTSTR_SCENARIO_{}", name_str.to_uppercase());

    // (1) The author's function, renamed. Attributes, visibility,
    // return type and body are preserved verbatim so the doc comment
    // and any `#[allow(...)]` stay attached to the code they describe.
    let mut renamed = input;
    renamed.sig.ident = body_fn.clone();

    // (3) The runner handed to `ktstr_test_impl`. Written out as
    // tokens rather than assembled from `renamed`'s signature because
    // it shares nothing with it: different arity, different return
    // type, generated body.
    let runner = quote! {
        fn #name(
            ctx: &::ktstr::scenario::Ctx,
        ) -> ::ktstr::prelude::Result<::ktstr::assert::AssertResult> {
            ::ktstr::scenario::ScenarioDef::run(&#def_fn(), ctx)
        }
    };
    let test_items = ktstr_test_impl(attr, runner)?;

    Ok(quote! {
        #renamed

        // (2) Canonical host-callable builder + its registration. The
        // `Into` coercion is what lets a body return `Step`,
        // `Vec<Step>`, or `ScenarioDef` interchangeably.
        #[doc(hidden)]
        fn #def_fn() -> ::ktstr::scenario::ScenarioDef {
            ::core::convert::Into::into(#body_fn())
        }

        #[::ktstr::distributed_slice(::ktstr::test_support::KTSTR_SCENARIOS)]
        #[linkme(crate = ::ktstr::linkme)]
        static #registry_entry: ::ktstr::test_support::ScenarioEntry =
            ::ktstr::test_support::ScenarioEntry {
                name: #name_str,
                build: #def_fn,
            };

        #test_items
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The attribute list shared by the equivalence tests. Deliberately
    /// broad — a scheduler path, topology, a threshold, a bool — so the
    /// delegation is exercised across every attribute category rather
    /// than just the trivial ones.
    const ATTRS: &str = "scheduler = MY_SCHED, llcs = 1, cores = 2, threads = 1, \
                         sustained_samples = 15, max_spread_pct = 80.0, \
                         performance_mode = true";

    fn ts(src: &str) -> TokenStream {
        src.parse().expect("test source parses as tokens")
    }

    fn expand_scenario(attrs: &str, item: &str) -> Result<TokenStream, syn::Error> {
        ktstr_scenario_impl(ts(attrs), ts(item))
    }

    fn err_of(attrs: &str, item: &str) -> String {
        match expand_scenario(attrs, item) {
            Ok(ts) => panic!("expansion must be rejected, but it produced: {ts}"),
            Err(e) => e.to_string(),
        }
    }

    /// THE equivalence pin, and the reason this macro delegates rather
    /// than forks.
    ///
    /// `#[ktstr_scenario]` must emit, verbatim, whatever
    /// `#[ktstr_test]` would have emitted for the synthesized runner —
    /// same `KtstrTestEntry` static, same manifest and admission
    /// stamps, same `#[test]` wrapper. Asserting containment of the
    /// hand-computed `ktstr_test_impl` output proves that with no
    /// field-by-field list to keep in sync: any future `#[ktstr_test]`
    /// field, stamp, or diagnostic is inherited by construction, and a
    /// scenario-side attempt to intercept or rewrite the delegation
    /// breaks this test immediately.
    #[test]
    fn delegates_to_ktstr_test_verbatim() {
        let expanded = expand_scenario(
            ATTRS,
            "fn my_scenario() -> ScenarioDef { ScenarioDef::default() }",
        )
        .expect("scenario expands");

        let runner = ts("fn my_scenario(\
             ctx: &::ktstr::scenario::Ctx,\
         ) -> ::ktstr::prelude::Result<::ktstr::assert::AssertResult> {\
             ::ktstr::scenario::ScenarioDef::run(&__ktstr_scenario_def_my_scenario(), ctx)\
         }");
        let direct = ktstr_test_impl(ts(ATTRS), runner).expect("runner expands via ktstr_test");

        assert!(
            expanded.to_string().contains(&direct.to_string()),
            "the #[ktstr_scenario] expansion must CONTAIN the \
             #[ktstr_test] expansion of its synthesized runner \
             verbatim — otherwise the two entrypoints can drift.\n\
             \n--- scenario expansion ---\n{expanded}\n\
             \n--- expected substring ---\n{direct}",
        );
    }

    /// The author's function survives renamed but otherwise untouched,
    /// and the canonical `fn() -> ScenarioDef` builder calls it through
    /// `Into` (which is what lets a body return `Step`, `Vec<Step>` or
    /// `ScenarioDef` interchangeably).
    #[test]
    fn emits_renamed_body_and_into_coercing_builder() {
        let expanded = expand_scenario(ATTRS, "fn my_scenario() -> Vec<Step> { vec![] }")
            .expect("scenario expands")
            .to_string();
        assert!(
            expanded.contains("fn __ktstr_scenario_body_my_scenario () -> Vec < Step >"),
            "author's fn is renamed, signature preserved: {expanded}",
        );
        assert!(
            expanded.contains(
                "fn __ktstr_scenario_def_my_scenario () -> :: ktstr :: scenario :: ScenarioDef"
            ),
            "canonical builder is emitted: {expanded}",
        );
        assert!(
            expanded.contains("Into :: into (__ktstr_scenario_body_my_scenario ())"),
            "builder coerces the body's return value via Into: {expanded}",
        );
    }

    /// Registration in `KTSTR_SCENARIOS` is what makes scenarios
    /// enumerable, and the `name` must be the test name so the entry
    /// joins to its `KtstrTestEntry`.
    #[test]
    fn registers_scenario_entry_under_the_test_name() {
        let expanded = expand_scenario(ATTRS, "fn my_scenario() -> ScenarioDef { d() }")
            .expect("scenario expands")
            .to_string();
        assert!(
            expanded.contains("KTSTR_SCENARIOS"),
            "scenario registers in the KTSTR_SCENARIOS slice: {expanded}",
        );
        assert!(
            expanded.contains("name : \"my_scenario\""),
            "registration is keyed by the test name: {expanded}",
        );
        assert!(
            expanded.contains("build : __ktstr_scenario_def_my_scenario"),
            "registration points at the canonical builder: {expanded}",
        );
    }

    /// A `&Ctx` parameter is the one rejection that is about
    /// extractability rather than codegen mechanics, so its diagnostic
    /// has to explain the `ctx.cgroup_def` equivalence — otherwise an
    /// author hits the error and concludes their test cannot be ported
    /// when in fact it can.
    #[test]
    fn rejects_ctx_parameter_with_a_porting_hint() {
        let msg = err_of(
            ATTRS,
            "fn my_scenario(ctx: &Ctx) -> ScenarioDef { ScenarioDef::default() }",
        );
        assert!(msg.contains("takes no arguments"), "{msg}");
        assert!(
            msg.contains("CgroupDef::named"),
            "the diagnostic must point at the ctx-free equivalent: {msg}",
        );
        assert!(
            msg.contains("#[ktstr_test]"),
            "the diagnostic must name the escape hatch: {msg}",
        );
    }

    /// Host-side callbacks are arbitrary Rust — admitting them would
    /// make a "declarative" scenario carry behaviour again.
    #[test]
    fn rejects_post_vm_callbacks() {
        for attr in ["post_vm = check", "post_vm_unconditional = check"] {
            let attrs = format!("{ATTRS}, {attr}");
            let msg = err_of(&attrs, "fn my_scenario() -> ScenarioDef { d() }");
            assert!(
                msg.contains("not available on `#[ktstr_scenario]`"),
                "{attr} must be rejected, got: {msg}",
            );
            assert!(
                msg.contains("#[ktstr_test]"),
                "{attr} rejection must name the escape hatch: {msg}",
            );
        }
    }

    /// Shapes the generated `fn() -> ScenarioDef` wrapper cannot call.
    /// Rejected here so the error points at the author's signature
    /// rather than at macro-generated code.
    #[test]
    fn rejects_uncallable_signatures() {
        let cases = [
            (
                "async fn my_scenario() -> ScenarioDef { d() }",
                "must not be `async`",
            ),
            (
                "fn my_scenario<T>() -> ScenarioDef { d() }",
                "must not be generic",
            ),
            (
                "fn my_scenario() -> ScenarioDef where u8: Copy { d() }",
                "`where` clause",
            ),
        ];
        for (item, expected) in cases {
            let msg = err_of(ATTRS, item);
            assert!(
                msg.contains(expected),
                "expected {expected:?} in diagnostic for {item:?}, got: {msg}",
            );
        }
    }

    /// Every diagnostic that is not one of this macro's own must come
    /// from `ktstr_test_impl`, so the two entrypoints report attribute
    /// problems identically. Pinned on the unknown-attribute path
    /// because its message is derived from the shared attribute-name
    /// registries — proof the delegation reaches the real parser and
    /// not a copy.
    #[test]
    fn attribute_diagnostics_come_from_ktstr_test() {
        let msg = err_of(
            "nonsuch_attr = 1",
            "fn my_scenario() -> ScenarioDef { d() }",
        );
        assert!(
            msg.contains("unknown attribute `nonsuch_attr`") && msg.contains("expected:"),
            "attribute errors must be ktstr_test's own: {msg}",
        );
    }
}
