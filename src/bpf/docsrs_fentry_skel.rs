// docs.rs compile-only stub for the fentry/fexit BPF skeleton.
//
// Companion to `docsrs_probe_skel.rs`; see that file's header for why
// the real skeleton cannot be generated in the docs.rs sandbox and how
// `build.rs` substitutes this stub. Mirrors the shape
// `src/probe/process.rs` compiles against for the fentry probe path.

pub use self::imp::*;

#[allow(dead_code, non_snake_case, non_camel_case_types)]
mod imp {
    use libbpf_rs::libbpf_sys;
    use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};

    pub mod types {
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct rodata {
            pub ktstr_enabled: bool,
            pub ktstr_fentry_func_idx_0: u32,
            pub ktstr_fentry_func_idx_1: u32,
            pub ktstr_fentry_func_idx_2: u32,
            pub ktstr_fentry_func_idx_3: u32,
            pub ktstr_fentry_is_kernel_0: u8,
            pub ktstr_fentry_is_kernel_1: u8,
            pub ktstr_fentry_is_kernel_2: u8,
            pub ktstr_fentry_is_kernel_3: u8,
        }
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct bss {
            pub ktstr_fentry_probe_count: u64,
        }
    }

    pub struct OpenFentryProbeMaps<'obj> {
        pub func_meta_map: libbpf_rs::OpenMapMut<'obj>,
        pub probe_data: libbpf_rs::OpenMapMut<'obj>,
        pub probe_scratch: libbpf_rs::OpenMapMut<'obj>,
        pub rodata: libbpf_rs::OpenMapMut<'obj>,
        pub rodata_data: Option<&'obj mut types::rodata>,
        pub bss: libbpf_rs::OpenMapMut<'obj>,
        pub bss_data: Option<&'obj mut types::bss>,
    }
    pub struct OpenFentryProbeProgs<'obj> {
        pub ktstr_fentry_0: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fentry_1: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fentry_2: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fentry_3: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fexit_0: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fexit_1: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fexit_2: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_fexit_3: libbpf_rs::OpenProgramMut<'obj>,
    }
    pub struct OpenFentryProbeSkel<'obj> {
        pub maps: OpenFentryProbeMaps<'obj>,
        pub progs: OpenFentryProbeProgs<'obj>,
    }
    impl<'obj> OpenSkel<'obj> for OpenFentryProbeSkel<'obj> {
        type Output = FentryProbeSkel<'obj>;
        fn load(self) -> libbpf_rs::Result<FentryProbeSkel<'obj>> {
            unimplemented!("docs.rs stub skeleton is never loaded")
        }
        fn open_object(&self) -> &libbpf_rs::OpenObject {
            unimplemented!()
        }
        fn open_object_mut(&mut self) -> &mut libbpf_rs::OpenObject {
            unimplemented!()
        }
    }

    pub struct FentryProbeMaps<'obj> {
        pub func_meta_map: libbpf_rs::MapMut<'obj>,
        pub probe_data: libbpf_rs::MapMut<'obj>,
        pub probe_scratch: libbpf_rs::MapMut<'obj>,
        pub rodata: libbpf_rs::MapMut<'obj>,
        pub rodata_data: Option<&'obj types::rodata>,
        pub bss: libbpf_rs::MapMut<'obj>,
        pub bss_data: Option<&'obj mut types::bss>,
    }
    pub struct FentryProbeProgs<'obj> {
        pub ktstr_fentry_0: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fentry_1: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fentry_2: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fentry_3: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fexit_0: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fexit_1: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fexit_2: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_fexit_3: libbpf_rs::ProgramMut<'obj>,
    }
    #[derive(Default)]
    pub struct FentryProbeLinks {
        pub ktstr_fentry_0: Option<libbpf_rs::Link>,
        pub ktstr_fentry_1: Option<libbpf_rs::Link>,
        pub ktstr_fentry_2: Option<libbpf_rs::Link>,
        pub ktstr_fentry_3: Option<libbpf_rs::Link>,
        pub ktstr_fexit_0: Option<libbpf_rs::Link>,
        pub ktstr_fexit_1: Option<libbpf_rs::Link>,
        pub ktstr_fexit_2: Option<libbpf_rs::Link>,
        pub ktstr_fexit_3: Option<libbpf_rs::Link>,
    }
    pub struct FentryProbeSkel<'obj> {
        pub maps: FentryProbeMaps<'obj>,
        pub progs: FentryProbeProgs<'obj>,
        pub links: FentryProbeLinks,
    }
    unsafe impl Send for FentryProbeSkel<'_> {}
    unsafe impl Sync for FentryProbeSkel<'_> {}
    impl<'obj> Skel<'obj> for FentryProbeSkel<'obj> {
        fn object(&self) -> &libbpf_rs::Object {
            unimplemented!()
        }
        fn object_mut(&mut self) -> &mut libbpf_rs::Object {
            unimplemented!()
        }
    }

    #[derive(Default)]
    pub struct FentryProbeSkelBuilder {}
    impl<'obj> SkelBuilder<'obj> for FentryProbeSkelBuilder {
        type Output = OpenFentryProbeSkel<'obj>;
        fn open(
            self,
            _object: &'obj mut std::mem::MaybeUninit<libbpf_rs::OpenObject>,
        ) -> libbpf_rs::Result<OpenFentryProbeSkel<'obj>> {
            unimplemented!("docs.rs stub skeleton is never opened")
        }
        fn open_opts(
            self,
            _open_opts: libbpf_sys::bpf_object_open_opts,
            _object: &'obj mut std::mem::MaybeUninit<libbpf_rs::OpenObject>,
        ) -> libbpf_rs::Result<OpenFentryProbeSkel<'obj>> {
            unimplemented!()
        }
        fn object_builder(&self) -> &libbpf_rs::ObjectBuilder {
            unimplemented!()
        }
        fn object_builder_mut(&mut self) -> &mut libbpf_rs::ObjectBuilder {
            unimplemented!()
        }
    }
}
