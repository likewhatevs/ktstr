use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemStatic;

pub(crate) fn expand(item: ItemStatic) -> syn::Result<TokenStream> {
    if matches!(item.mutability, syn::StaticMutability::Mut(_)) {
        return Err(syn::Error::new_spanned(
            &item,
            "ktstr_test_entry requires an immutable static KtstrTestEntry",
        ));
    }

    let entry = &item.ident;
    let stamp = format_ident!(
        "__KTSTR_SCHED_MANIFEST_MANUAL_{}",
        entry.to_string().to_uppercase()
    );
    let admission_stamp = format_ident!(
        "__KTSTR_ADMISSION_MANUAL_{}",
        entry.to_string().to_uppercase()
    );
    let admission_key = format_ident!(
        "__KTSTR_ADMISSION_MANUAL_KEY_{}",
        entry.to_string().to_uppercase()
    );
    Ok(quote! {
        #[::ktstr::distributed_slice(::ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ::ktstr::linkme)]
        #item

        #[::ktstr::distributed_slice(
            ::ktstr::test_support::KTSTR_SCHEDULER_MANIFEST_TESTS_V1
        )]
        #[linkme(crate = ::ktstr::linkme)]
        static #stamp: ::ktstr::test_support::SchedulerManifestTestStampV1 =
            ::ktstr::test_support::SchedulerManifestTestStampV1::new(
                &#entry,
                &[
                    ::ktstr::test_support::SchedulerManifestUseStampV1::new(
                        #entry.scheduler
                    ),
                ],
            );

        #[::ktstr::distributed_slice(
            ::ktstr::test_support::KTSTR_ADMISSION_TESTS_V2
        )]
        #[linkme(crate = ::ktstr::linkme)]
        static #admission_stamp: ::ktstr::test_support::AdmissionTestStampV2 =
            ::ktstr::test_support::AdmissionTestStampV2::new(&#entry);

        #[::ktstr::distributed_slice(
            ::ktstr::test_support::KTSTR_ADMISSION_TEST_KEYS_V2
        )]
        #[linkme(crate = ::ktstr::linkme)]
        static #admission_key: ::ktstr::test_support::AdmissionTestKeyV2 =
            ::ktstr::test_support::AdmissionTestKeyV2::new(
                &#entry,
                &#admission_stamp,
            );
    })
}
