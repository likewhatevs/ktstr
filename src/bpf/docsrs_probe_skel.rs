// docs.rs compile-only stub for the kprobe BPF skeleton.
//
// The real `probe_skel.rs` is generated at build time by `libbpf-cargo`
// from `src/bpf/probe.bpf.c`: it compiles the BPF object, resolves
// CO-RE relocations against kernel BTF, and embeds the object bytes.
// That pipeline needs a libbpf toolchain (clang + the vendored
// libbpf/libelf/zlib C stack) and a BTF source — none of which exist in
// the docs.rs sandbox (no flex/bison, no network). When ktstr is built
// without the `vendored` feature (only docs.rs does this, via
// `[package.metadata.docs.rs]` + `DOCS_RS`), `build.rs` copies this file
// into `$OUT_DIR/probe_skel.rs` in place of the generated skeleton.
//
// It reproduces exactly the public shape `src/probe/process.rs` compiles
// against — the map/prog/link wrappers are the real `libbpf_rs` types,
// and every method body is `unimplemented!()` because rustdoc compiles
// but never runs them. Keep it in sync with `probe.bpf.c`'s maps/progs
// and the `types::` structs `process.rs` reads; drift surfaces as a
// docs.rs build failure, never a runtime bug (this file is never linked
// into a real build).

pub use self::imp::*;

#[allow(dead_code, non_snake_case, non_camel_case_types)]
mod imp {
    use libbpf_rs::libbpf_sys;
    use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};

    pub mod types {
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct field_spec {
            pub param_idx: u32,
            pub offset: u32,
            pub size: u32,
            pub field_idx: u32,
            pub ptr_offset: u32,
        }
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct func_meta {
            pub func_idx: u32,
            pub nr_field_specs: u32,
            pub specs: [field_spec; 16],
            pub str_param_idx: u8,
        }
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct probe_key {
            pub func_ip: u64,
            pub task_ptr: u64,
        }
        #[derive(Debug, Copy, Clone)]
        #[repr(C)]
        pub struct probe_entry {
            pub ts: u64,
            pub args: [u64; 6],
            pub fields: [u64; 16],
            pub nr_fields: u32,
            pub str_val: [i8; 64],
            pub has_str: u8,
            pub str_param_idx: u8,
            pub exit_ts: u64,
            pub exit_fields: [u64; 16],
            pub nr_exit_fields: u32,
            pub has_exit: u8,
        }
        #[derive(Debug, Default, Copy, Clone)]
        #[repr(C)]
        pub struct rodata {
            pub ktstr_enabled: bool,
        }
        #[derive(Debug, Copy, Clone)]
        #[repr(C)]
        pub struct pcpu_counter {
            pub value: i64,
            pub __pad_8: [u8; 120],
        }
        #[derive(Debug, Copy, Clone)]
        #[repr(C)]
        pub struct bss {
            pub ktstr_err_exit_detected: u32,
            pub ktstr_last_trigger_ts: u64,
            pub ktstr_exit_kind_snap: u32,
            pub ktstr_miss_log: [u64; 16],
            pub ktstr_miss_log_idx: u32,
            pub ktstr_pcpu_counters: [[pcpu_counter; 15]; 256],
        }
    }

    pub struct OpenProbeMaps<'obj> {
        pub func_meta_map: libbpf_rs::OpenMapMut<'obj>,
        pub probe_scratch: libbpf_rs::OpenMapMut<'obj>,
        pub probe_data: libbpf_rs::OpenMapMut<'obj>,
        pub ktstr_events: libbpf_rs::OpenMapMut<'obj>,
        pub timeline_events: libbpf_rs::OpenMapMut<'obj>,
        pub pi_scratch: libbpf_rs::OpenMapMut<'obj>,
        pub preempt_disabled_per_cpu: libbpf_rs::OpenMapMut<'obj>,
        pub rodata: libbpf_rs::OpenMapMut<'obj>,
        pub rodata_data: Option<&'obj mut types::rodata>,
        pub bss: libbpf_rs::OpenMapMut<'obj>,
        pub bss_data: Option<&'obj mut types::bss>,
    }
    pub struct OpenProbeProgs<'obj> {
        pub ktstr_probe: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_trigger_tp: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_trigger_fexit: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_trigger_dump_fentry: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_tl_switch: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_tl_migrate: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_tl_wakeup: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_pi_fentry: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_pi_fexit: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_lock_contend: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_preempt_disable_tp: libbpf_rs::OpenProgramMut<'obj>,
        pub ktstr_preempt_enable_tp: libbpf_rs::OpenProgramMut<'obj>,
    }
    pub struct OpenProbeSkel<'obj> {
        pub maps: OpenProbeMaps<'obj>,
        pub progs: OpenProbeProgs<'obj>,
    }
    impl<'obj> OpenSkel<'obj> for OpenProbeSkel<'obj> {
        type Output = ProbeSkel<'obj>;
        fn load(self) -> libbpf_rs::Result<ProbeSkel<'obj>> {
            unimplemented!("docs.rs stub skeleton is never loaded")
        }
        fn open_object(&self) -> &libbpf_rs::OpenObject {
            unimplemented!()
        }
        fn open_object_mut(&mut self) -> &mut libbpf_rs::OpenObject {
            unimplemented!()
        }
    }

    pub struct ProbeMaps<'obj> {
        pub func_meta_map: libbpf_rs::MapMut<'obj>,
        pub probe_scratch: libbpf_rs::MapMut<'obj>,
        pub probe_data: libbpf_rs::MapMut<'obj>,
        pub ktstr_events: libbpf_rs::MapMut<'obj>,
        pub timeline_events: libbpf_rs::MapMut<'obj>,
        pub pi_scratch: libbpf_rs::MapMut<'obj>,
        pub preempt_disabled_per_cpu: libbpf_rs::MapMut<'obj>,
        pub rodata: libbpf_rs::MapMut<'obj>,
        pub rodata_data: Option<&'obj types::rodata>,
        pub bss: libbpf_rs::MapMut<'obj>,
        pub bss_data: Option<&'obj mut types::bss>,
    }
    pub struct ProbeProgs<'obj> {
        pub ktstr_probe: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_trigger_tp: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_trigger_fexit: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_trigger_dump_fentry: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_tl_switch: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_tl_migrate: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_tl_wakeup: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_pi_fentry: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_pi_fexit: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_lock_contend: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_preempt_disable_tp: libbpf_rs::ProgramMut<'obj>,
        pub ktstr_preempt_enable_tp: libbpf_rs::ProgramMut<'obj>,
    }
    #[derive(Default)]
    pub struct ProbeLinks {
        pub ktstr_probe: Option<libbpf_rs::Link>,
        pub ktstr_trigger_tp: Option<libbpf_rs::Link>,
        pub ktstr_trigger_fexit: Option<libbpf_rs::Link>,
        pub ktstr_trigger_dump_fentry: Option<libbpf_rs::Link>,
        pub ktstr_tl_switch: Option<libbpf_rs::Link>,
        pub ktstr_tl_migrate: Option<libbpf_rs::Link>,
        pub ktstr_tl_wakeup: Option<libbpf_rs::Link>,
        pub ktstr_pi_fentry: Option<libbpf_rs::Link>,
        pub ktstr_pi_fexit: Option<libbpf_rs::Link>,
        pub ktstr_lock_contend: Option<libbpf_rs::Link>,
        pub ktstr_preempt_disable_tp: Option<libbpf_rs::Link>,
        pub ktstr_preempt_enable_tp: Option<libbpf_rs::Link>,
    }
    pub struct ProbeSkel<'obj> {
        pub maps: ProbeMaps<'obj>,
        pub progs: ProbeProgs<'obj>,
        pub links: ProbeLinks,
    }
    unsafe impl Send for ProbeSkel<'_> {}
    unsafe impl Sync for ProbeSkel<'_> {}
    impl<'obj> Skel<'obj> for ProbeSkel<'obj> {
        fn object(&self) -> &libbpf_rs::Object {
            unimplemented!()
        }
        fn object_mut(&mut self) -> &mut libbpf_rs::Object {
            unimplemented!()
        }
    }

    #[derive(Default)]
    pub struct ProbeSkelBuilder {}
    impl<'obj> SkelBuilder<'obj> for ProbeSkelBuilder {
        type Output = OpenProbeSkel<'obj>;
        fn open(
            self,
            _object: &'obj mut std::mem::MaybeUninit<libbpf_rs::OpenObject>,
        ) -> libbpf_rs::Result<OpenProbeSkel<'obj>> {
            unimplemented!("docs.rs stub skeleton is never opened")
        }
        fn open_opts(
            self,
            _open_opts: libbpf_sys::bpf_object_open_opts,
            _object: &'obj mut std::mem::MaybeUninit<libbpf_rs::OpenObject>,
        ) -> libbpf_rs::Result<OpenProbeSkel<'obj>> {
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
