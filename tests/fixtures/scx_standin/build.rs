// Build the stand-in scheduler's BPF object without any scx bundled headers.
//
// vmlinux.h is generated from BTF at build time (libbpf's btf_dump via the
// vendored `vmlinux_gen.c`), exactly as ktstr's own guest-side BPF build does,
// rather than vendoring a multi-megabyte, arch-specific header. BPF CO-RE
// relocates field offsets against the guest kernel at load, so the build-time
// BTF only needs to contain the sched_ext types; each CI lane's build host
// shares the guest architecture and runs a sched_ext kernel, so its
// `/sys/kernel/btf/vmlinux` is a valid source.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use libbpf_cargo::SkeletonBuilder;

fn main() {
    println!("cargo:rerun-if-changed=src/bpf/standin.bpf.c");
    println!("cargo:rerun-if-changed=src/bpf/vmlinux_gen.c");
    println!("cargo:rerun-if-env-changed=KTSTR_BUILD_BTF");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let vmlinux_h = out_dir.join("vmlinux.h");

    // An explicitly content-keyed BTF is preferred when present; otherwise the
    // host BTF. CO-RE makes the exact source non-semantic beyond containing the
    // sched_ext types, so this needs no drift tracking.
    let btf_source = env::var_os("KTSTR_BUILD_BTF")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/sys/kernel/btf/vmlinux"));

    // libbpf-sys (links = "bpf", vendored) installs its headers at
    // DEP_BPF_INCLUDE and the static libbpf/elf/z archives in its parent.
    let libbpf_include =
        PathBuf::from(env::var("DEP_BPF_INCLUDE").expect("DEP_BPF_INCLUDE (libbpf-sys) not set"));
    let libbpf_lib_dir = libbpf_include
        .parent()
        .expect("DEP_BPF_INCLUDE has a parent");

    let driver_src = out_dir.join("vmlinux_gen_main.c");
    std::fs::write(
        &driver_src,
        format!(
            "extern int generate_vmlinux_h(const char *, const char *);\n\
             int main(void) {{ return generate_vmlinux_h(\"{btf}\", \"{out}\") == 0 ? 0 : 1; }}\n",
            btf = btf_source.display(),
            out = vmlinux_h.display(),
        ),
    )
    .expect("write vmlinux_gen driver");

    let gen_bin = out_dir.join("vmlinux_gen");
    let compiler = cc::Build::new().get_compiler();
    let status = Command::new(compiler.path())
        .args([
            "src/bpf/vmlinux_gen.c",
            driver_src.to_str().expect("driver path UTF-8"),
            "-o",
            gen_bin.to_str().expect("gen bin path UTF-8"),
            &format!("-I{}", libbpf_include.display()),
            &format!("-L{}", libbpf_lib_dir.display()),
            "-lbpf",
            "-lelf",
            "-lz",
        ])
        .status()
        .expect("compile vmlinux_gen");
    assert!(status.success(), "failed to compile vmlinux_gen");
    let status = Command::new(&gen_bin).status().expect("run vmlinux_gen");
    assert!(
        status.success(),
        "vmlinux_gen failed for BTF source {}",
        btf_source.display(),
    );

    // arm64 bpf_tracing.h casts through struct user_pt_regs, a UAPI type kernel
    // BTF may omit. Append it if absent, mirroring ktstr's own build.rs.
    if cfg!(target_arch = "aarch64") {
        let content = std::fs::read_to_string(&vmlinux_h).expect("read vmlinux.h");
        if !content.contains("struct user_pt_regs {") {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&vmlinux_h)
                .expect("open vmlinux.h for append");
            writeln!(
                file,
                "\nstruct user_pt_regs {{\n\t__u64 regs[31];\n\t__u64 sp;\n\t__u64 pc;\n\t__u64 pstate;\n}};\n"
            )
            .expect("append user_pt_regs");
        }
    }

    let arch_define = if cfg!(target_arch = "aarch64") {
        "-D__TARGET_ARCH_arm64"
    } else {
        "-D__TARGET_ARCH_x86"
    };
    let skel = out_dir.join("standin.skel.rs");
    SkeletonBuilder::new()
        .source("src/bpf/standin.bpf.c")
        .clang_args([format!("-I{}", out_dir.display()), arch_define.to_string()])
        .build_and_generate(&skel)
        .expect("build standin.bpf.c BPF skeleton");
}
