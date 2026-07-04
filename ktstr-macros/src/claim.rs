//! `#[derive(Claim)]` — pointwise-claim accessor generator.
//!
//! Emits an extension trait `<StructName>Claim` with one
//! `claim_<field_name>` method per public scalar field (returning
//! `ClaimBuilder<'_, FieldType>`), plus container-typed accessors:
//!
//!   * `BTreeSet<T>` fields → `claim_<field_name>(&self, verdict: &mut Verdict)
//!     -> SetClaim<'_, T>`, dispatched through `Verdict::claim_set`.
//!   * `Vec<T>` fields → `claim_<field_name>(&self, verdict: &mut Verdict)
//!     -> SeqClaim<'_, T>`, dispatched through `Verdict::claim_seq`.
//!   * `BTreeMap<K, V>` / `HashMap<K, V>` fields → SKIPPED (no claim
//!     surface for maps in v1; users reach for `claim!(verdict,
//!     map.len())` or `claim!(verdict, map.contains_key(&k))` against
//!     the explicit expression). Maps that need first-class support
//!     can be added in a follow-up — the derive emits no method, so
//!     the user's call site fails to compile rather than silently
//!     dispatching to a wrong type.
//!
//! Per-field opt-out via `#[claim(skip)]`. Non-pub fields are skipped
//! automatically (the claim surface is for the test author's view of
//! the struct; private fields aren't part of that view).
//!
//! The generated trait is named `<StructName>Claim` with the same
//! visibility as the input struct. The single
//! `impl <StructName>Claim for <StructName>` lives in the same
//! expansion (the methods take `&self` on the stats struct plus a
//! `&mut Verdict` accumulator argument), so callers only need
//! `use ::ktstr::prelude::*` (which re-exports the trait) to bring the
//! methods into scope.
//!
//! Label source: every method body passes `stringify!(field)` as the
//! label to its dispatch helper (`verdict.claim` / `claim_set` /
//! `claim_seq`) — the label is the field's source-text identifier. Renaming
//! the field updates both the method name AND the rendered failure
//! label in lock-step; a stale call site that referenced the old
//! method name fails to compile. This is the compile-mechanical
//! drift-free axis that the design rejected manual-string labels in
//! favor of.
//!
//! The `#[proc_macro_derive(Claim)]` entry point lives in `lib.rs`
//! (Rust requires proc-macro fns at the crate root); this module holds
//! the expansion it calls.

use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

/// Detect whether a `syn::Type` is a path whose last segment matches
/// `name` (e.g. `BTreeSet<T>` → `is_path_named(ty, "BTreeSet") == true`).
/// Used to dispatch container fields onto `claim_set` / `claim_seq`
/// without forcing the caller to import `BTreeSet` / `Vec` from a
/// specific module path. Misses a field typed through a user alias
/// (e.g. `type MySet = BTreeSet<T>; field: MySet`), since only the last
/// path segment is compared; a fully-qualified `std::collections::BTreeSet<T>`
/// still matches (its last segment is `BTreeSet`) — acceptable in v1
/// because the project's stats structs use the canonical `BTreeSet` /
/// `Vec` names directly.
fn is_path_named(ty: &syn::Type, name: &str) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(ident) = crate::common::path_last_segment_ident(&tp.path)
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
/// See the module docs for the dispatch rules and label invariant.
/// Reject non-struct inputs and tuple-struct inputs — the claim API is
/// keyed on field names, which tuple structs do not have.
pub(crate) fn derive_claim_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
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
         structs do not collide. For prelude-exported stats types the trait is \
         preluded, so `use ktstr::prelude::*` brings the accessors into scope; \
         otherwise import the trait from the stats type's module.",
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
