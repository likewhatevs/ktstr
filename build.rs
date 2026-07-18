// Generates vmlinux.h from kernel BTF using libbpf's btf_dump API.
// Uses the shared kernel resolver (src/kernel_path.rs) to find the
// BTF source. See resolve_btf() for the full search order.

// Without `vendored` (docs.rs only) the real build pipeline is compiled
// out, leaving its helpers and toolchain imports unused. Suppress the
// resulting warnings only in that configuration so a normal build still
// flags genuine dead code.
#![cfg_attr(not(feature = "vendored"), allow(dead_code, unused_imports))]

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[cfg(feature = "vendored")]
use libbpf_cargo::SkeletonBuilder;

#[cfg(feature = "wprof")]
use build_ahash as gix_acquire_ahash;
#[cfg(feature = "wprof")]
use build_fs2 as gix_acquire_fs2;
#[cfg(feature = "wprof")]
use build_gix as gix_acquire_gix;
#[cfg(feature = "wprof")]
use build_jobserver as gix_acquire_jobserver;
#[cfg(feature = "wprof")]
#[path = "build_support/gix_acquire.rs"]
mod gix_acquire;

include!("src/kernel_path.rs");
include!("src/build_helpers.rs");

/// Construct GNU make so it participates in Cargo's inherited jobserver.
///
/// Cargo exposes the jobserver authentication in `CARGO_MAKEFLAGS`, while
/// GNU make consumes it from `MAKEFLAGS`. Propagating it lets every native
/// build use otherwise-idle capacity without creating a second,
/// oversubscribed worker pool.
#[cfg(any(feature = "vendored", feature = "wprof"))]
fn cargo_coordinated_make() -> Command {
    let mut command = Command::new("make");
    // SAFETY: Cargo owns the authenticated jobserver descriptors inherited by
    // this build script. `configure()` marks those descriptors inheritable by
    // GNU make instead of merely copying their numeric names.
    if let Some(client) = unsafe { build_jobserver::Client::from_env() } {
        client.configure(&mut command);
    }
    // GNU make consumes MAKEFLAGS directly. Retain Cargo's compatibility
    // spelling after `configure()` has authenticated descriptor inheritance.
    if let Some(makeflags) = std::env::var_os("CARGO_MAKEFLAGS") {
        command.env("MAKEFLAGS", makeflags);
    }
    command
}

fn main() {
    #[cfg(feature = "wprof")]
    println!("cargo:rerun-if-changed=build_support/gix_acquire.rs");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // docs.rs / non-`vendored` builds cannot compile the vendored libbpf
    // C stack (the docs.rs sandbox has no flex/bison) or fetch build-time
    // blobs (no network), and without `vendored` there is no libbpf-cargo
    // to generate BPF skeletons. Emit the `$OUT_DIR` artifacts the crate
    // `include!`s / `env!`s (stub skeletons, shift registry, blob
    // placeholders, fingerprint env vars) so rustdoc still compiles the
    // full public API, then stop. Strict no-op for every normal build:
    // `vendored` is default-on, so this branch is compiled out and the
    // real pipeline below is byte-for-byte the historical `main`. The two
    // arms are mutually exclusive via cfg, so exactly one is compiled.
    #[cfg(not(feature = "vendored"))]
    emit_docsrs_stubs(&out_dir);
    #[cfg(feature = "vendored")]
    vendored_main(out_dir);
}

/// Emit the `$OUT_DIR` artifacts the crate `include!`s / `env!`s so
/// rustdoc can compile the whole crate without the BPF toolchain. Runs
/// on docs.rs and any `default-features = false` build that omits
/// `vendored`. The skeleton stubs (`src/bpf/docsrs_*.rs`) reproduce the
/// exact shape `src/probe/process.rs` compiles against; every other
/// artifact is an inert placeholder — this build cannot load BPF.
#[cfg(not(feature = "vendored"))]
fn emit_docsrs_stubs(out_dir: &std::path::Path) {
    if std::env::var_os("DOCS_RS").is_none() {
        println!(
            "cargo:warning=building ktstr without the `vendored` feature: BPF \
             skeletons are stubbed and cannot be loaded at runtime. Add \
             `features = [\"vendored\"]` for a functional build."
        );
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for (src, dst) in [
        ("src/bpf/docsrs_probe_skel.rs", "probe_skel.rs"),
        ("src/bpf/docsrs_fentry_skel.rs", "fentry_probe_skel.rs"),
    ] {
        let src = manifest_dir.join(src);
        println!("cargo:rerun-if-changed={}", src.display());
        std::fs::copy(&src, out_dir.join(dst))
            .unwrap_or_else(|e| panic!("copy docs.rs skeleton stub {}: {e}", src.display()));
    }

    // `budget.rs` `include!`s this; the generator is hermetic (it scans
    // `src/budget.rs`, no toolchain), so keep it real.
    generate_shift_registry(out_dir);

    // Inert blob placeholders. Only the cli-bins binaries `include_bytes!`
    // these and docs.rs drops cli-bins, but emit them anyway so a docs
    // build that opts cli-bins back in still compiles.
    for blob in ["busybox", "wprof"] {
        std::fs::write(out_dir.join(blob), b"")
            .unwrap_or_else(|e| panic!("write {blob} placeholder: {e}"));
    }

    // `persist.rs` consumes these via `env!`; they gate the on-disk cast
    // cache, which a docs build never touches, so stub them.
    println!("cargo:rustc-env=KTSTR_CAST_ANALYZER_FINGERPRINT=0000000000000000");
    println!("cargo:rustc-env=KTSTR_CARGO_LOCK_FINGERPRINT=0000000000000000");
}

/// The real build pipeline: vmlinux.h generation, BPF skeleton builds,
/// busybox/wprof compilation. Compiled only with the `vendored` feature
/// (default-on); docs.rs drops it. Unchanged from the historical `main`
/// body apart from taking `out_dir` as a parameter.
#[cfg(feature = "vendored")]
fn vendored_main(out_dir: PathBuf) {
    // Cache invalidation: track the env var that selects a kernel
    // and the build-script inputs (kernel_path resolver, C generator
    // source). Deliberately NOT emitting a `rerun-if-changed` on the
    // BTF source path itself:
    //
    //   1. `vmlinux` is consumed here only as the BTF source for
    //      `vmlinux.h` generation on the C side below, not as an
    //      input that the Rust compiler reads. BPF CO-RE (Compile
    //      Once Run Everywhere) relocates field offsets at LOAD
    //      time against the runtime kernel's BTF, so a field-layout
    //      drift between the compile-time `vmlinux.h` and the
    //      runtime kernel is resolved by libbpf on BPF object load
    //      — there is no compile-time correctness dependency on
    //      the exact byte content of the vmlinux used to generate
    //      `vmlinux.h`.
    //   2. `rerun-if-changed` on the BTF would force build.rs to
    //      re-run on every kernel rebuild. That runs the BPF
    //      skeleton generator unnecessarily when the drift (per
    //      (1)) has no compile-time correctness impact.
    //
    // However, WHEN build.rs does run (triggered by a watched
    // input — KTSTR_KERNEL change, kernel_path.rs edit, or a
    // previously-absent `vmlinux.h`), it SHOULD detect a BTF
    // content change and regenerate. The pre-hash design only
    // regenerated when `vmlinux.h` was absent entirely, which
    // meant a BTF-content change paired with an unrelated build-
    // script trigger would leave stale `vmlinux.h` in place. A
    // SipHasher13 hash of the BTF bytes is written alongside
    // `vmlinux.h` as `vmlinux.btf.hash`; regen fires when the
    // file is absent OR the stored hash differs from the current
    // BTF's hash. Operators who need to force regen unconditionally
    // still have `cargo clean` as the escape hatch. The algorithm
    // mirrors `src/test_support/sidecar/mod.rs::sidecar_variant_hash`
    // so the project uses a single stable hash family.
    println!("cargo:rerun-if-env-changed=KTSTR_KERNEL");
    println!("cargo:rerun-if-changed=src/kernel_path.rs");
    println!("cargo:rerun-if-changed=src/bpf/vmlinux_gen.c");
    let ktstr_kernel = env::var("KTSTR_KERNEL").ok();

    // Generate vmlinux.h from kernel BTF.
    let vmlinux_h = out_dir.join("vmlinux.h");
    let hash_path = out_dir.join("vmlinux.btf.hash");
    // Resolve BTF + compute content hash eagerly. `resolve_btf`
    // returns `Option` to degrade cleanly when no BTF is reachable
    // (no KTSTR_KERNEL + no host BTF): if `vmlinux.h` is already in
    // place from an earlier build, we keep it rather than panicking
    // — matches the CO-RE design (runtime BTF fixes field drift
    // anyway), so a disappearing source is not a build-blocking
    // event. A MISSING `vmlinux.h` still panics below because we
    // have nothing to fall back on.
    let current_btf = resolve_btf(ktstr_kernel.as_deref());
    // Hash the BTF source for drift detection. Fault-tolerant: a
    // BTF path that resolved but whose bytes cannot be read (EACCES,
    // or a race where the file vanished between resolve and read)
    // downgrades to `None` instead of panicking, so we fall back to
    // the existence-only gate for `vmlinux.h`. The eventual regen
    // path below re-reads the bytes via `vmlinux_gen` and fails
    // loudly there if the source is truly unusable.
    let current_hash: Option<String> = current_btf.as_ref().and_then(|p| match std::fs::read(p) {
        Ok(bytes) => Some(format!("{:016x}", siphash_13(&bytes))),
        Err(e) => {
            println!(
                "cargo:warning=BTF source {} present but unreadable \
                     ({e}); skipping hash check, reusing existing vmlinux.h",
                p.display(),
            );
            None
        }
    });
    let stored_hash: Option<String> = std::fs::read_to_string(&hash_path)
        .ok()
        .map(|s| s.trim().to_string());
    // Regen fires on any of three conditions:
    //   - `vmlinux.h` is absent (first build or post-`cargo clean`);
    //   - the stored hash is absent but we have a current hash (the
    //     vmlinux.h was generated by an older build.rs that didn't
    //     track hashes — upgrade in place);
    //   - current and stored hashes differ (real drift).
    // An unreadable BTF with vmlinux.h already in place falls
    // through to "no regen" per `current_hash.is_none()`.
    let should_regen =
        !vmlinux_h.exists() || (current_hash.is_some() && current_hash != stored_hash);
    if should_regen {
        let btf_source = current_btf.unwrap_or_else(|| {
            panic!(
                "no BTF source found. Set KTSTR_KERNEL to a kernel build \
                 directory, or ensure /sys/kernel/btf/vmlinux exists."
            );
        });
        println!("generating vmlinux.h from {}", btf_source.display());

        // libbpf-sys (links = "bpf") emits installed headers at
        // DEP_BPF_INCLUDE with bpf/ prefix (bpf/btf.h, bpf/libbpf.h).
        let libbpf_include =
            PathBuf::from(env::var("DEP_BPF_INCLUDE").expect("DEP_BPF_INCLUDE not set"));

        // Compile the C vmlinux generator + driver into a standalone binary.
        let vmlinux_gen_bin = out_dir.join("vmlinux_gen");
        let driver_src = out_dir.join("vmlinux_gen_main.c");
        std::fs::write(
            &driver_src,
            format!(
                r#"
extern int generate_vmlinux_h(const char *, const char *);
int main(void) {{
    return generate_vmlinux_h("{btf}", "{out}") == 0 ? 0 : 1;
}}
"#,
                btf = btf_source.display(),
                out = vmlinux_h.display(),
            ),
        )
        .expect("write driver source");

        // libbpf-sys with vendored feature installs static libraries
        // (libbpf.a, libelf.a, libz.a) in the parent of DEP_BPF_INCLUDE.
        let libbpf_lib_dir = libbpf_include.parent().unwrap();

        let compiler = cc::Build::new().get_compiler();
        let status = Command::new(compiler.path())
            .args([
                "src/bpf/vmlinux_gen.c",
                driver_src.to_str().unwrap(),
                "-o",
                vmlinux_gen_bin.to_str().unwrap(),
                &format!("-I{}", libbpf_include.display()),
                &format!("-L{}", libbpf_lib_dir.display()),
                "-lbpf",
                "-lelf",
                "-lz",
            ])
            .status()
            .expect("compile vmlinux_gen");
        assert!(status.success(), "failed to compile vmlinux_gen");

        let status = Command::new(&vmlinux_gen_bin)
            .status()
            .expect("run vmlinux_gen");
        assert!(
            status.success(),
            "vmlinux_gen failed — check BTF source: {}",
            btf_source.display()
        );

        // Record the BTF content hash alongside `vmlinux.h`. A
        // future build.rs invocation reads this file and compares
        // against the freshly-hashed BTF; a mismatch triggers
        // regeneration above.
        //
        // Normally `current_hash` was populated at the top of
        // `main`. The one path that leaves it `None` while still
        // reaching this regen branch is: `!vmlinux_h.exists()` AND
        // `std::fs::read(&btf_source)` failed during the eager hash
        // attempt. In that case, the generator above successfully
        // invoked `vmlinux_gen` against `btf_source`, which means
        // libbpf could read it — the earlier read failure was
        // transient or the generator accessed the file via a path
        // libbpf handles differently (e.g. sysfs BTF). Re-read and
        // hash here so the sidecar is always populated alongside a
        // successful regen; on a second-read failure, skip the
        // sidecar (the generator already succeeded — the build is
        // in a good state; a missing sidecar forces the next
        // build.rs run to regenerate conservatively, which is
        // correct).
        let hash_opt: Option<String> = match current_hash.as_deref() {
            Some(h) => Some(h.to_string()),
            None => match std::fs::read(&btf_source) {
                Ok(bytes) => Some(format!("{:016x}", siphash_13(&bytes))),
                Err(e) => {
                    println!(
                        "cargo:warning=post-regen BTF re-read failed ({e}); \
                         skipping hash sidecar — next build.rs run will \
                         regenerate conservatively"
                    );
                    None
                }
            },
        };
        if let Some(hash) = hash_opt {
            // Trailing newline so `cat` / editor-open produces a
            // clean single-line display. The reader at the top of
            // main uses `.trim()` on the stored value, so the
            // newline round-trips.
            std::fs::write(&hash_path, format!("{hash}\n"))
                .unwrap_or_else(|e| panic!("write BTF hash sidecar {}: {e}", hash_path.display()));
        }
    }

    // arm64 bpf_tracing.h casts pt_regs through struct user_pt_regs,
    // a UAPI type that kernel BTF may omit. Append it if absent so
    // PT_REGS_PARMn_CORE compiles on arm64 hosts.
    if cfg!(target_arch = "aarch64") {
        let content = std::fs::read_to_string(&vmlinux_h).expect("read vmlinux.h");
        if !content.contains("struct user_pt_regs {") {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&vmlinux_h)
                .expect("open vmlinux.h for append");
            writeln!(
                f,
                "\n/* Added by build.rs: arm64 UAPI type needed by bpf_tracing.h */\n\
                 struct user_pt_regs {{\n\
                 \t__u64 regs[31];\n\
                 \t__u64 sp;\n\
                 \t__u64 pc;\n\
                 \t__u64 pstate;\n\
                 }};\n"
            )
            .expect("append user_pt_regs to vmlinux.h");
        }
    }

    let clang_args = [
        format!("-I{}", out_dir.display()),
        format!("-I{}", "src/bpf"),
    ];

    // Build the kprobe BPF skeleton.
    let skel_path = out_dir.join("probe_skel.rs");
    SkeletonBuilder::new()
        .source("src/bpf/probe.bpf.c")
        .obj(out_dir.join("probe.o"))
        .clang_args(clang_args.clone())
        .reference_obj(true)
        .build_and_generate(&skel_path)
        .expect("build probe BPF skeleton");

    // Build the fentry BPF skeleton (separate for independent loading).
    let fentry_skel_path = out_dir.join("fentry_probe_skel.rs");
    SkeletonBuilder::new()
        .source("src/bpf/fentry_probe.bpf.c")
        .obj(out_dir.join("fentry_probe.o"))
        .clang_args(clang_args)
        .reference_obj(true)
        .build_and_generate(&fentry_skel_path)
        .expect("build fentry probe BPF skeleton");

    println!("cargo::rerun-if-changed=src/bpf/probe.bpf.c");
    println!("cargo::rerun-if-changed=src/bpf/fentry_probe.bpf.c");
    println!("cargo::rerun-if-changed=src/bpf/intf.h");

    // Generate ALL_SHIFTS registry from src/budget.rs so the
    // budget-feature tests can assert exhaustive classification
    // coverage. Scans `const NAME_SHIFT: u32 = N;` declarations and
    // emits a `pub(crate) const ALL_SHIFTS: &[(u32, &str)]` slice
    // into OUT_DIR. The test in budget.rs takes the union of its
    // one-bit and multi-bit shift enumerations and asserts equality
    // with this slice — a new SHIFT constant added without updating
    // either enumeration fails the union check.
    generate_shift_registry(&out_dir);

    // Fingerprint the cast-analysis source so the on-disk cast cache
    // (src/vmm/cast_analysis_load/persist.rs) self-invalidates whenever
    // the analyzer changes — with no manual SCHEMA_VERSION bump. Without
    // this, an analyzer-behavior change reuses a stale cached result and
    // masks a just-fixed analyzer bug as a flake (the 2026-06-01
    // arena_confirmed-drop bug hid this way for hours). The fn emits
    // `rerun-if-changed` for the watched dirs so cargo recomputes the env
    // when the analyzer source changes.
    println!(
        "cargo:rustc-env=KTSTR_CAST_ANALYZER_FINGERPRINT={:016x}",
        cast_analyzer_fingerprint()
    );

    // Fingerprint the whole Cargo.lock so the cast-analysis cache
    // self-invalidates on any dependency bump: persist::cache_path folds
    // this into the cache key. A btf-rs (BTF parsing) or libbpf-rs /
    // libbpf-sys (BPF-opcode constants) version change can alter the cast
    // map with no ktstr source change, so the analyzer-source fingerprint
    // alone would serve a stale result. Only the cast cache folds this in;
    // kernels / models / disk_template are dependency-independent.
    println!(
        "cargo:rustc-env=KTSTR_CARGO_LOCK_FINGERPRINT={:016x}",
        cargo_lock_fingerprint()
    );

    // Build busybox from source for guest shell mode.
    //
    // Hermeticity contract:
    //
    //  - The tarball is fetched ONCE per OUT_DIR and cached at
    //    `$OUT_DIR/busybox`. `cargo clean` forces a re-fetch.
    //  - The fetched bytes are SHA-256-verified against
    //    [`BUSYBOX_TARBALL_SHA256`] before extraction. A mismatch
    //    panics with the actual vs expected hash so the operator
    //    can decide between "the upstream changed (regenerate the
    //    pin)" and "the download was tampered (investigate)".
    //  - `KTSTR_BUSYBOX_TARBALL=<path>` points the build at a
    //    pre-fetched local tarball — for air-gapped CI runners and
    //    hermetic CI caches. The SHA pin still applies; the local
    //    path is a transport substitute, not a verification bypass.
    //  - `KTSTR_SKIP_BUSYBOX_BUILD=1` writes a 0-byte placeholder at
    //    `$OUT_DIR/busybox` and skips the compile entirely. Shell
    //    mode is unavailable in the resulting binary;
    //    `cargo_ktstr::blobs::install_env` detects the empty blob
    //    and leaves `KTSTR_BUSYBOX_PATH` unset so consumers fail
    //    with a clear "shell mode unavailable" rather than an
    //    opaque "exec format error" on the 0-byte file. Mirrors
    //    the existing `KTSTR_SKIP_WPROF_BUILD` escape hatch below.
    //  - `KTSTR_BUSYBOX_BIN=<path>` copies a pre-built busybox binary
    //    directly into `$OUT_DIR/busybox`, skipping fetch + compile.
    //    `cargo-ktstr` sets it (via `run_cargo.rs`) to the busybox it
    //    already embedded and extracted (`src/bin/cargo_ktstr/blobs.rs`),
    //    so a downstream `cargo ktstr test` reuses that binary instead
    //    of re-fetching. Falls through to the fetch path when the path
    //    is unset / missing / 0-byte. Precedence: SKIP > BIN > TARBALL
    //    > download.
    //
    // The pre-pin git-clone fallback was removed alongside this
    // refactor: a clone bypasses the SHA gate (no tarball to
    // verify), and `KTSTR_BUSYBOX_TARBALL` covers the
    // tarball-fetch-failed case more cleanly.
    let busybox_bin = out_dir.join("busybox");
    println!("cargo:rerun-if-env-changed=KTSTR_SKIP_BUSYBOX_BUILD");
    println!("cargo:rerun-if-env-changed=KTSTR_BUSYBOX_TARBALL");
    println!("cargo:rerun-if-env-changed=KTSTR_BUSYBOX_BIN");
    let skip_busybox = std::env::var("KTSTR_SKIP_BUSYBOX_BUILD")
        .ok()
        .filter(|v| !v.is_empty())
        .is_some();
    if skip_busybox {
        println!(
            "cargo:warning=KTSTR_SKIP_BUSYBOX_BUILD set — writing 0-byte \
             $OUT_DIR/busybox placeholder; shell mode will be unavailable \
             in the resulting cargo-ktstr binary"
        );
        if !busybox_bin.exists() {
            std::fs::write(&busybox_bin, b"").unwrap_or_else(|e| {
                panic!(
                    "write 0-byte busybox placeholder {}: {e}",
                    busybox_bin.display()
                )
            });
        }
    } else if !busybox_bin.exists()
        && !copy_prebuilt_blob(
            std::env::var("KTSTR_BUSYBOX_BIN").ok().as_deref(),
            &busybox_bin,
            "busybox",
        )
    {
        println!("cargo:warning=compiling busybox (first build only)...");

        // Check required tools before attempting build.
        if Command::new("make").arg("--version").output().is_err() {
            panic!(
                "busybox build requires 'make' — install build-essential \
                 (Debian/Ubuntu) or base-devel (Fedora/Arch)"
            );
        }
        if Command::new("gcc").arg("--version").output().is_err() {
            panic!(
                "busybox build requires 'gcc' — install build-essential \
                 (Debian/Ubuntu) or base-devel (Fedora/Arch)"
            );
        }

        let busybox_src = out_dir.join("busybox-src");

        // Recover from interrupted download: if the directory exists but
        // has no Makefile, the previous extraction was incomplete.
        if busybox_src.exists() && !busybox_src.join("Makefile").exists() {
            std::fs::remove_dir_all(&busybox_src).expect("remove incomplete busybox-src");
        }

        // Source the tarball: from a local path when
        // KTSTR_BUSYBOX_TARBALL is set, otherwise from the pinned
        // upstream URL with retry. Either path lands in
        // `tarball_bytes` which is then SHA-verified before any
        // extraction touches the filesystem.
        if !busybox_src.join("Makefile").exists() {
            const TARBALL_URL: &str =
                "https://github.com/mirror/busybox/archive/refs/tags/1_36_1.tar.gz";
            let tarball_bytes = match std::env::var("KTSTR_BUSYBOX_TARBALL")
                .ok()
                .filter(|v| !v.is_empty())
            {
                Some(local) => {
                    println!(
                        "cargo:warning=KTSTR_BUSYBOX_TARBALL set — reading {local} \
                         instead of fetching from {TARBALL_URL}"
                    );
                    std::fs::read(&local).unwrap_or_else(|e| {
                        panic!(
                            "read KTSTR_BUSYBOX_TARBALL={local}: {e} — the env \
                             var must point at a readable tarball matching the \
                             pinned SHA-256"
                        )
                    })
                }
                None => fetch_busybox_tarball(TARBALL_URL),
            };

            verify_busybox_tarball_sha256(&tarball_bytes);

            // Extract verified bytes into busybox-src/.
            let extract_dir = out_dir.join("busybox-extract");
            if extract_dir.exists() {
                let _ = std::fs::remove_dir_all(&extract_dir);
            }
            let gz = flate2::read::GzDecoder::new(std::io::Cursor::new(&tarball_bytes[..]));
            let mut archive = tar::Archive::new(gz);
            archive
                .unpack(&extract_dir)
                .unwrap_or_else(|e| panic!("extract busybox tarball: {e}"));
            let inner = extract_dir.join("busybox-1_36_1");
            std::fs::rename(&inner, &busybox_src).unwrap_or_else(|e| {
                panic!(
                    "expected extracted directory {} — tarball layout may have changed: {e}",
                    inner.display()
                )
            });
            std::fs::remove_dir_all(&extract_dir).ok();
        }

        // Configure busybox.
        let status = cargo_coordinated_make()
            .arg("defconfig")
            .current_dir(&busybox_src)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("make defconfig");
        assert!(status.success(), "busybox make defconfig failed");

        // Enable static linking, disable CONFIG_TC (requires iproute2 headers).
        let config_path = busybox_src.join(".config");
        let config = std::fs::read_to_string(&config_path).expect("read busybox .config");
        let config = config
            .replace("# CONFIG_STATIC is not set", "CONFIG_STATIC=y")
            .replace("CONFIG_TC=y", "# CONFIG_TC is not set");
        std::fs::write(&config_path, config).expect("write patched busybox .config");

        // Resolve patched config non-interactively. Busybox's Kbuild
        // lacks olddefconfig; pipe empty input to oldconfig so every
        // NEW prompt accepts its default without blocking on stdin.
        let status = cargo_coordinated_make()
            .arg("oldconfig")
            .current_dir(&busybox_src)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("make oldconfig");
        assert!(status.success(), "busybox make oldconfig failed");

        // Build through Cargo's jobserver. This shares the host-wide Cargo
        // concurrency budget instead of serializing the C build or starting
        // an unrelated, oversubscribed pool.
        let status = cargo_coordinated_make()
            .current_dir(&busybox_src)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("busybox make");
        assert!(status.success(), "busybox build failed");

        // Copy binary to OUT_DIR.
        std::fs::copy(busybox_src.join("busybox"), &busybox_bin)
            .expect("copy busybox binary to OUT_DIR");
    }

    // wprof build: gated behind the `wprof` cargo feature (default
    // off). When disabled, a 0-byte placeholder at $OUT_DIR/wprof
    // satisfies the `include_bytes!` site in cargo_ktstr/blobs.rs.
    // The KTSTR_SKIP_WPROF_BUILD env var remains as a secondary
    // escape hatch for builds that enable the feature but want to
    // skip the clone/compile (CI caching, cross-compilation, etc.).
    let wprof_bin = out_dir.join("wprof");
    #[cfg(not(feature = "wprof"))]
    if !wprof_bin.exists() {
        std::fs::write(&wprof_bin, b"").unwrap_or_else(|e| {
            panic!(
                "write 0-byte wprof placeholder {}: {e}",
                wprof_bin.display()
            )
        });
    }
    #[cfg(feature = "wprof")]
    {
        prepare_wprof(&out_dir, &wprof_bin);
    } // #[cfg(feature = "wprof")]
}

#[cfg(feature = "wprof")]
fn prepare_wprof(out_dir: &std::path::Path, wprof_bin: &std::path::Path) {
    const WPROF_URL: &str = "https://github.com/anakryiko/wprof.git";
    // v0.4 release commit. Every root/submodule checkout below is depth-one
    // and detached at the exact committed object id.
    const WPROF_REF: &str = "refs/tags/v0.4";
    const WPROF_REV: &str = "9afa9ee5493814c7791586f2179aa93528fde54a";
    const WPROF_STAMP: &str = ".wprof-content-key";

    println!("cargo:rerun-if-env-changed=KTSTR_SKIP_WPROF_BUILD");
    println!("cargo:rerun-if-env-changed=KTSTR_WPROF_BIN");
    for variable in [
        "CFLAGS",
        "CPPFLAGS",
        "EXTRA_CFLAGS",
        "EXTRA_LDFLAGS",
        "LDFLAGS",
        "CARGO",
        "RUSTC",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let skip = std::env::var("KTSTR_SKIP_WPROF_BUILD")
        .ok()
        .is_some_and(|value| !value.is_empty());
    let stamp_path = out_dir.join(WPROF_STAMP);
    let existing_stamp = std::fs::read_to_string(&stamp_path).unwrap_or_default();

    // A cargo-ktstr parent can hand this build the already embedded binary.
    // Keep that zero-network path ahead of tool probing and source acquisition.
    let copied = !skip
        && !wprof_bin.exists()
        && copy_prebuilt_blob(
            std::env::var("KTSTR_WPROF_BIN").ok().as_deref(),
            wprof_bin,
            "wprof",
        );
    if copied {
        std::fs::write(&stamp_path, format!("embedded:{WPROF_REV}"))
            .expect("stamp embedded wprof binary");
        return;
    }

    if skip {
        println!(
            "cargo:warning=KTSTR_SKIP_WPROF_BUILD set — writing 0-byte \
             $OUT_DIR/wprof placeholder; do NOT use the resulting \
             cargo-ktstr binary for wprof capture"
        );
        // The opt-out must remain authoritative when this OUT_DIR already
        // contains a real binary from an earlier non-skipped build.
        std::fs::write(wprof_bin, b"").unwrap_or_else(|err| {
            panic!(
                "write 0-byte wprof placeholder {}: {err}",
                wprof_bin.display()
            )
        });
        std::fs::write(&stamp_path, format!("skipped:{WPROF_REV}"))
            .expect("stamp skipped wprof placeholder");
        return;
    }

    if existing_stamp == format!("embedded:{WPROF_REV}")
        && std::fs::metadata(wprof_bin).is_ok_and(|meta| meta.len() > 0)
    {
        return;
    }

    let toolchain = wprof_toolchain_fingerprint();
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "unknown-host".to_string());
    let key_parts = [
        "wprof-binary-v1",
        WPROF_URL,
        WPROF_REF,
        WPROF_REV,
        target.as_str(),
        host.as_str(),
        toolchain.as_str(),
    ];
    let build_id = gix_acquire::content_id(&key_parts);
    let expected_stamp = format!("built:{build_id}");
    if existing_stamp == expected_stamp
        && std::fs::metadata(wprof_bin).is_ok_and(|meta| meta.len() > 0)
    {
        return;
    }
    if wprof_bin.exists() {
        std::fs::remove_file(wprof_bin).expect("remove stale wprof binary");
    }
    let _ = std::fs::remove_file(&stamp_path);
    // Clean up the old pre-CAS per-OUT_DIR clone once. The new builder always
    // uses a private staged checkout and publishes only its immutable binary.
    let old_source = out_dir.join("wprof-src");
    if old_source.exists() {
        if !is_wprof_clone_complete(&old_source) {
            println!("cargo:warning=removing incomplete legacy wprof source checkout");
        }
        std::fs::remove_dir_all(&old_source).expect("remove legacy wprof source checkout");
    }

    let binary_cache_root = gix_acquire::cache_root("wprof")
        .unwrap_or_else(|| out_dir.join(".ktstr-content-cache").join("wprof"));
    let source_cache_root = gix_acquire::cache_root("source-nodes")
        .unwrap_or_else(|| out_dir.join(".ktstr-content-cache").join("source-nodes"));
    let cached = gix_acquire::ensure_cached(
        &binary_cache_root,
        &key_parts,
        "wprof exact source + binary",
        |entry| std::fs::metadata(entry.join("wprof")).is_ok_and(|meta| meta.len() > 0),
        |stage, progress| {
            let source = stage.join("source");
            gix_acquire::assemble_exact_recursive_cached(
                &source_cache_root,
                WPROF_URL,
                WPROF_REF,
                WPROF_REV,
                &source,
                progress,
            )?;
            // Upstream's Makefile contains `git submodule update` fallbacks.
            // The exact source graph must make every guard true so the build
            // cannot cross back into an executable transport.
            for required in ["libbpf/src", "bpftool/src", "blazesym/src"] {
                if !source.join(required).is_dir() {
                    return Err(format!(
                        "exact wprof source graph is missing {required}; refusing \
                         upstream's executable submodule fallback"
                    ));
                }
            }
            isolate_wprof_subcrate_workspaces(&source);
            progress.set_phase("compiling wprof");
            let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
            let status = cargo_coordinated_make()
                // v0.4's outer Makefile is missing the prerequisite edge
                // between its recursive demangle Cargo build and the sibling
                // archive copy. Parallel outer recipes deterministically race
                // (`cp: cannot stat .../libdemangle_c.a`) in CI. Keep only
                // this outer make serial; recursive Cargo/sub-makes still use
                // the authenticated Cargo jobserver configured above.
                .arg("-j1")
                // Pin Makefile policy inputs that are not supported ktstr
                // overrides. Supported compiler/linker flags remain inherited
                // and are part of the content key below.
                .args([
                    "CLANG=clang",
                    "AWK=awk",
                    "DEBUG=",
                    "BLAZESYM_DEBUG=",
                    "STATIC=",
                    "LTO=1",
                    "DESTDIR=",
                    "CROSS_COMPILE=",
                    "AR=ar",
                    "LD=ld",
                    "NM=nm",
                    "OBJCOPY=objcopy",
                    "RANLIB=ranlib",
                    "STRIP=strip",
                ])
                .arg(format!("CARGO={cargo}"))
                .arg("CC=clang -fuse-ld=mold -Wno-unused-command-line-argument")
                .env(
                    "CC",
                    "clang -fuse-ld=mold -Wno-unused-command-line-argument",
                )
                .env_remove("ARCH")
                .env_remove("BPFTOOL")
                .env_remove("BPFTOOL_OUTPUT")
                .env_remove("BPFTOOL_OUTPUT_ABS")
                .env_remove("CLANG_BPF_SYS_INCLUDES")
                .env_remove("RUSTC_WORKSPACE_WRAPPER")
                .env_remove("RUSTC_WRAPPER")
                .env_remove("CARGO_BUILD_TARGET")
                .env_remove("CARGO_TARGET_DIR")
                .current_dir(source.join("src"))
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(|err| format!("spawn make for wprof: {err}"))?;
            if !status.success() {
                return Err(format!("wprof make exited {status}"));
            }
            let built = source.join("src/wprof");
            if !built.is_file() {
                return Err(format!(
                    "wprof build succeeded but binary is missing at {}",
                    built.display()
                ));
            }
            std::fs::copy(&built, stage.join("wprof"))
                .map_err(|err| format!("stage completed wprof binary: {err}"))?;
            // The cache publishes only the immutable result. Source/build state
            // is private to the elected builder and never shared with `make`.
            std::fs::remove_dir_all(&source)
                .map_err(|err| format!("remove private wprof build tree: {err}"))?;
            Ok(())
        },
    )
    .unwrap_or_else(|err| panic!("obtain exact wprof binary: {err}"));
    std::fs::copy(cached.join("wprof"), wprof_bin)
        .unwrap_or_else(|err| panic!("copy cached wprof binary: {err}"));
    std::fs::write(&stamp_path, expected_stamp).expect("stamp cached wprof binary");
}

#[cfg(feature = "wprof")]
fn wprof_toolchain_fingerprint() -> String {
    fn version(tool: &str, args: &[&str]) -> String {
        let output = Command::new(tool)
            .args(args)
            .output()
            .unwrap_or_else(|err| {
                panic!(
                    "wprof build requires '{tool}' on PATH: {err}. Install the \
                     build toolchain (make, gcc, clang, mold, and rustc)."
                )
            });
        if !output.status.success() {
            panic!(
                "wprof build requires a working '{tool}', but `{tool} {}` exited {}",
                args.join(" "),
                output.status
            );
        }
        format!(
            "{tool}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let versions = [
        version("make", &["--version"]),
        version("gcc", &["--version"]),
        version("clang", &["--version"]),
        version("mold", &["--version"]),
        version("ar", &["--version"]),
        version("ld", &["--version"]),
        version("nm", &["--version"]),
        version("objcopy", &["--version"]),
        version("ranlib", &["--version"]),
        version("strip", &["--version"]),
        version(&rustc, &["-vV"]),
        version(&cargo, &["-vV"]),
    ];
    let environment: Vec<String> = [
        "CFLAGS",
        "CPPFLAGS",
        "EXTRA_CFLAGS",
        "EXTRA_LDFLAGS",
        "LDFLAGS",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ]
    .into_iter()
    .map(|name| format!("{name}={}", std::env::var(name).unwrap_or_default()))
    .collect();
    let parts: Vec<&str> = versions
        .iter()
        .chain(environment.iter())
        .map(String::as_str)
        .collect();
    gix_acquire::content_id(&parts)
}

/// SHA-256 hex digest of the upstream busybox-1.36.1 release tarball
/// (`busybox-1_36_1.tar.gz` from the `mirror/busybox` github archive).
///
/// **Sentinel value**: `""` means the pin is not yet recorded for this
/// checkout. In that case [`verify_busybox_tarball_sha256`] emits the
/// computed digest as a `cargo:warning` and continues — first-build
/// integration. To activate the verification gate, replace the empty
/// string with the printed digest, then commit. Subsequent builds
/// fail on mismatch.
///
/// **Rotation**: bumping the busybox version requires updating BOTH
/// the URL in the `fetch_busybox_tarball` call site AND this pin in
/// lockstep — a partial edit produces a SHA mismatch on the next
/// fetch (fail-loud, not silent-pull-wrong-bytes).
///
/// **Why a custom pin instead of cargo's vendoring**: cargo's
/// vendoring covers crate sources, not arbitrary C-source tarballs
/// downloaded by a build script. The verification has to live in
/// `build.rs` itself.
const BUSYBOX_TARBALL_SHA256: &str = "";

/// Fetch the upstream busybox tarball with retry; return the raw
/// gzip-compressed bytes (NOT yet SHA-verified — caller passes the
/// returned buffer through [`verify_busybox_tarball_sha256`] before
/// extracting). Extracted from the prior in-line download so the
/// `KTSTR_BUSYBOX_TARBALL` operator override can read a local file
/// through the same downstream pipeline.
fn fetch_busybox_tarball(url: &str) -> Vec<u8> {
    // Authenticated GitHub requests get 1000/hr per token vs the
    // 60/hr IP-based unauth limit. GitHub Actions auto-issues
    // GITHUB_TOKEN per job; outside CI the env var is typically
    // absent and the request goes unauth, which still works for
    // public repos at low rate.
    let github_token = std::env::var("GITHUB_TOKEN").ok();
    let attempt = |attempt_idx: u32| -> Result<Vec<u8>, String> {
        // `timeout()` bounds the whole request including the body
        // when read via `.bytes()` (which uses `wait::timeout`
        // internally per `reqwest::blocking::Response::bytes`),
        // but does NOT apply when reading the response via the
        // `Read` trait -- streaming bypasses reqwest's timeout
        // machinery so a slow-drip server can hang the build
        // indefinitely. Buffer the body so the timeout actually
        // fires.
        //
        // Proxy support: reqwest automatically reads proxy configuration
        // from environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY
        // and their lowercase variants). In corporate or restricted
        // network environments, ensure these variables are set if a
        // proxy is required to reach github.com.
        let mut client_builder = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(30))
            .user_agent(concat!("ktstr-build/", env!("CARGO_PKG_VERSION")));

        // Explicitly configure proxy from environment if set.
        // reqwest reads these automatically, but we configure explicitly
        // to ensure proxy is used and to provide better error messages.
        // Supports: HTTP_PROXY, HTTPS_PROXY, NO_PROXY (and lowercase variants)
        if let Ok(proxy_url) = std::env::var("HTTPS_PROXY")
            .or_else(|_| std::env::var("https_proxy"))
            .or_else(|_| std::env::var("HTTP_PROXY"))
            .or_else(|_| std::env::var("http_proxy"))
        {
            let proxy = reqwest::Proxy::all(&proxy_url)
                .map_err(|e| format!("invalid proxy URL {proxy_url}: {e}"))?;
            client_builder = client_builder.proxy(proxy);
        }

        let client = client_builder
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        let mut req = client.get(url);
        if let Some(ref token) = github_token {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| format!("attempt {attempt_idx} request: {e}"))?;
        let body = resp
            .bytes()
            .map_err(|e| format!("attempt {attempt_idx} body: {e}"))?;
        Ok(body.to_vec())
    };

    println!("cargo:warning=downloading busybox source tarball from {url}");
    const MAX_TARBALL_ATTEMPTS: u32 = 4;
    retry_with_backoff("busybox tarball download", MAX_TARBALL_ATTEMPTS, attempt).unwrap_or_else(
        |e| {
            panic!(
                "failed to obtain busybox source after {MAX_TARBALL_ATTEMPTS} attempts.\n\
             tarball ({url}): {e}\n\
             Remediation:\n\
               • Check network connectivity (the build script needs HTTPS\n\
                 access to github.com to fetch the upstream tarball).\n\
               • If behind a proxy, ensure HTTP_PROXY/HTTPS_PROXY environment\n\
                 variables are set (e.g., export HTTPS_PROXY=http://proxy:8080).\n\
               • Or set KTSTR_BUSYBOX_TARBALL=<path> to point at a\n\
                 pre-fetched local copy of {url} — useful for air-gapped\n\
                 CI runners and hermetic build environments.\n\
               • Or set KTSTR_SKIP_BUSYBOX_BUILD=1 to skip the busybox\n\
                 compile entirely (shell mode will be unavailable in the\n\
                 resulting cargo-ktstr binary).",
            )
        },
    )
}

/// Verify the downloaded busybox tarball against [`BUSYBOX_TARBALL_SHA256`].
///
/// Three outcomes:
///
///   - **Pin empty**: log the computed digest as a `cargo:warning` and
///     continue. First-build bootstrap path — the operator pastes the
///     printed value into `BUSYBOX_TARBALL_SHA256` to lock the pin.
///   - **Pin matches**: silent pass.
///   - **Pin mismatches**: panic with both digests. The operator
///     investigates: a regenerated upstream archive (github does this
///     rarely; cf. the 2023 git-archive checksum change) requires a
///     pin refresh, whereas an unexplained mismatch on a fixed pin
///     indicates supply-chain tampering and warrants investigation
///     before the bytes hit the build.
fn verify_busybox_tarball_sha256(tarball_bytes: &[u8]) {
    use sha2::{Digest, Sha256};
    let actual = {
        let mut hasher = Sha256::new();
        hasher.update(tarball_bytes);
        hex_encode_lowercase(&hasher.finalize())
    };
    if BUSYBOX_TARBALL_SHA256.is_empty() {
        println!(
            "cargo:warning=BUSYBOX_TARBALL_SHA256 is unset — first-build \
             bootstrap. Computed SHA-256: {actual}\n\
             To lock the pin: update BUSYBOX_TARBALL_SHA256 in build.rs to\n\
             this value and commit. Subsequent builds will fail on mismatch."
        );
        return;
    }
    if !BUSYBOX_TARBALL_SHA256.eq_ignore_ascii_case(&actual) {
        panic!(
            "busybox tarball SHA-256 mismatch.\n\
             expected: {BUSYBOX_TARBALL_SHA256}\n\
             actual:   {actual}\n\
             \n\
             Diagnose:\n\
               • If the upstream archive was regenerated (rare — github\n\
                 changed archive generation in early 2023, otherwise these\n\
                 tarballs are stable for years), update BUSYBOX_TARBALL_SHA256\n\
                 in build.rs to the new digest after independently verifying\n\
                 the source.\n\
               • Otherwise treat as a supply-chain alert: compare against\n\
                 the upstream SHA published by the busybox maintainers\n\
                 before continuing."
        );
    }
}

/// Lowercase hex-encode a byte slice. Inlined to avoid pulling `hex`
/// into `[build-dependencies]` for a single 32-byte digest.
fn hex_encode_lowercase(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(&mut s, "{b:02x}").expect("write to String never fails");
    }
    s
}

/// Scan src/budget.rs for `const NAME_SHIFT: u32 = N;` declarations
/// and emit a `pub(crate) const ALL_SHIFTS: &[(u32, &str)]` slice
/// into `OUT_DIR/shift_registry.rs`. The slice is sorted by value
/// for stable test output.
///
/// Pattern: line.trim() starts with `const `, contains `: u32 = `
/// literal, name part ends with `_SHIFT`, value part parses as u32
/// (trailing `;` stripped). All four conditions must hold; a line
/// failing any one is skipped.
///
/// This is a deliberate text-scan, not a full Rust parser. Trade-offs:
/// - Full-line comments (`//`, `/* */`, `///`) start with `/`, not
///   `const` — never false-positive. Inline trailing comments on a
///   const line (e.g. `const X_SHIFT: u32 = 5; // foo`) leave the
///   comment text past the `;`; `trim_end_matches(';')` strips only
///   the trailing `;` so the parse-as-u32 step panics fail-loud
///   rather than silently dropping the entry.
/// - String literals containing `SHIFT:` live inside non-const lines
///   — never false-positive. EXCEPTION: a raw multi-line string
///   literal `r#"\nconst FOO_SHIFT: u32 = 4;\n"#` containing a
///   const-shaped line would false-positive (line.trim() yields the
///   raw const text). Low probability — budget.rs holds no such
///   literals today — and surfaces loudly: the false-positive grows
///   the registry by an entry no hand-classified enumeration
///   references, so the test's `unclassified` arm fires (asserts
///   `ALL_SHIFTS.filter(!classified.contains(v))` is empty), NOT a
///   silent drop.
/// - Macro-generated constants emit no source text — invisible to the
///   scan (false negative; documented by naming convention).
/// - `static FOO_SHIFT` and lowercase-named constants — both invisible
///   (false negative; violates Rust convention anyway).
/// - Const expressions whose RHS is non-integer (e.g.
///   `const X_SHIFT: u32 = OTHER + 1;`) — fail-loud panic, not silent
///   drop.
/// - The `: u32 = ` split anchor is rustfmt-canonical (single space
///   each side). A future rustfmt change to multi-space or no-space
///   formatting would cause the scan to miss every existing SHIFT
///   const. The test fails loudly on the first build after such a
///   change: registry shrinks, so each hand-classified SHIFT value
///   appears in `phantom_one_bit` (one_bit_values.difference(&registry))
///   or `phantom_multi_bit` (multi_bit_values.difference(&registry)),
///   tripping the phantom assertion. The regression surfaces
///   immediately, not on the next addition.
///
/// The hand-classified test enumerations in `src/budget.rs::tests`
/// are the consumer; the `all_shifts_classified_in_exactly_one_enumeration`
/// test asserts the union of the two hand-spelled lists equals this
/// generated set.
fn generate_shift_registry(out_dir: &std::path::Path) {
    use std::fmt::Write;
    println!("cargo::rerun-if-changed=src/budget.rs");
    let budget_rs = std::fs::read_to_string("src/budget.rs")
        .expect("read src/budget.rs for shift-registry scan");
    let mut shifts: Vec<(u32, String)> = Vec::new();
    for line in budget_rs.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("const ") else {
            continue;
        };
        let Some((name_part, val_part)) = rest.split_once(": u32 = ") else {
            continue;
        };
        let name = name_part.trim();
        if !name.ends_with("_SHIFT") {
            continue;
        }
        let val_str = val_part.trim_end_matches(';').trim();
        let val: u32 = val_str.parse().unwrap_or_else(|e| {
            panic!("shift-registry scan: parse `{val_str}` as u32 for {name}: {e}")
        });
        shifts.push((val, name.to_string()));
    }
    shifts.sort_by_key(|(v, _)| *v);

    let mut out = String::from(
        "// Generated by build.rs. Lists every `const *_SHIFT: u32 = N;`\n\
         // declaration in src/budget.rs, sorted by shift value. The\n\
         // budget tests assert their hand-classified one-bit and\n\
         // multi-bit enumerations cover every entry so a new SHIFT\n\
         // cannot land without being classified into the right test.\n\
         pub(crate) const ALL_SHIFTS: &[(u32, &str)] = &[\n",
    );
    for (v, name) in &shifts {
        writeln!(out, "    ({v}, \"{name}\"),").expect("write shift entry");
    }
    out.push_str("];\n");
    std::fs::write(out_dir.join("shift_registry.rs"), out).expect("write shift_registry.rs");
}

/// 64-bit SipHash-1-3 of `bytes`. Used to detect BTF content drift
/// between `vmlinux.h` regenerations.
///
/// Algorithm mirrors `src/test_support/sidecar/mod.rs::sidecar_variant_hash`
/// — `SipHasher13::new_with_keys(0, 0)` + `h.write(bytes)` +
/// `h.finish()`. Zero keys are deliberate: this is a drift hash, not
/// a DoS-mitigation hash, and stable (key-less) output lets a future
/// build.rs invocation compare against a sidecar written by a prior
/// run without coordinating on a key. SipHasher13 is faster than
/// SipHasher24 at the cost of reduced crypto strength — acceptable
/// because the hash is a build-artifact sidecar, not a signed
/// manifest.
fn siphash_13(bytes: &[u8]) -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(bytes);
    h.finish()
}

/// SipHasher13 fingerprint of every non-test `.rs` file under the
/// cast-analysis source dirs: the analyzer in `src/monitor/cast_analysis`,
/// its on-demand loader in `src/vmm/cast_analysis_load`,
/// `src/monitor/sdt_alloc` (whose `discover_payload_btf_id` +
/// `MAX_BTF_ID_PROBE` resolve the cached `alloc_size_types`), and
/// `src/monitor/btf_render` + `src/monitor/bpf_map` (whose
/// `peel_modifiers` / `type_size` / `resolve_to_struct_id` resolve every
/// cast's terminal type, 20+ call sites in cast_analysis/mod.rs). The
/// hash is folded into the disk-cache key (`persist.rs::cache_path`) so
/// the cache self-invalidates on any analyzer change without a manual
/// `SCHEMA_VERSION` bump. Files named `tests.rs` are excluded; inline
/// `#[cfg(test)]` modules in the watched `.rs` files are still hashed, so
/// a test-only edit to such a file does invalidate the cache — the safe,
/// over-conservative direction (never a stale serve). Each watched dir
/// gets a `rerun-if-changed` so cargo re-runs build.rs (recomputing the
/// env) when the analyzer source changes; a missing watched dir is a
/// hard error (see the loop body), not a silent skip. Crate-version
/// drift (btf-rs / libbpf) is handled separately by
/// [`cargo_lock_fingerprint`], which is folded alongside this into the
/// cast cache key — so this fingerprint covers only the analyzer's own
/// source.
fn cast_analyzer_fingerprint() -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in [
        "src/monitor/cast_analysis",
        "src/vmm/cast_analysis_load",
        // sdt_alloc feeds the cached output's `alloc_size_types` via
        // `discover_payload_btf_id` + `MAX_BTF_ID_PROBE` (see
        // cast_analysis_load::build_cast_analysis_from_bytes's alloc-size
        // resolution loop), so a change there alters the cached result and
        // must invalidate it -- same footgun class this fingerprint closes.
        "src/monitor/sdt_alloc",
        // cast_analysis resolves every cast's terminal type through
        // btf_render::{peel_modifiers,peel_modifiers_with_id,type_size}
        // and bpf_map::resolve_to_struct_id (20+ call sites in
        // cast_analysis/mod.rs); a change to either module's modifier-peel
        // / struct-resolve traversal alters the cached cast map for an
        // unchanged binary -- same footgun. Their callees stay within
        // btf-rs (a crate dep) + std, so the watched-source closure ends
        // here; btf-rs / libbpf crate-version drift is caught by the
        // whole-Cargo.lock fingerprint folded into the cast cache key
        // (cargo_lock_fingerprint + persist::cache_path).
        // Whole-subtree rather than per-fn because extracting
        // individual fns needs a parser; the extra invalidations on
        // unrelated edits in these modules are cheap (one BPF-object
        // re-analysis) and these modules already invalidate other caches
        // when they change.
        "src/monitor/btf_render",
        "src/monitor/bpf_map",
    ] {
        println!("cargo:rerun-if-changed={dir}");
        // Fail loud if a watched dir is missing: a typo or a layout move
        // would otherwise silently drop that dir's contribution and
        // resurrect the stale-cache footgun this fingerprint exists to
        // close. collect_fingerprint_files tolerates a missing dir for its
        // recursion case, so the top-level existence guard lives here.
        let path = std::path::Path::new(dir);
        assert!(
            path.is_dir(),
            "cast-analysis fingerprint dir missing: {dir} (layout moved? update build.rs)"
        );
        collect_fingerprint_files(path, &mut files);
    }
    // Sort for a deterministic hash independent of readdir order.
    files.sort();
    let mut h = SipHasher13::new_with_keys(0, 0);
    for f in &files {
        // Hash the path too so a rename (without content change) still
        // perturbs the fingerprint.
        h.write(f.to_string_lossy().as_bytes());
        let bytes = std::fs::read(f)
            .unwrap_or_else(|e| panic!("read {} for analyzer fingerprint: {e}", f.display()));
        h.write(&bytes);
    }
    h.finish()
}

/// SipHasher13 fingerprint of the entire `Cargo.lock`, emitted by
/// build.rs as `KTSTR_CARGO_LOCK_FINGERPRINT` and folded into the
/// cast-analysis cache key (see
/// `vmm::cast_analysis_load::persist::cache_path`). A dependency bump —
/// a `btf-rs` (BTF parsing) or `libbpf-rs` / `libbpf-sys` (BPF-opcode
/// constants) version change — can alter the cast map with no ktstr
/// source change, so the analyzer-source fingerprint alone would serve a
/// stale result. Only the cast cache folds this in; the kernels / models
/// / disk_template caches are produced by external tools and are
/// dependency-independent. Hashing the WHOLE lockfile invalidates the
/// cast cache on any dependency bump, even unrelated crates — the safe
/// over-conservative direction (never a stale serve), costing one cast
/// re-analysis per scheduler binary per lockfile change.
/// `rerun-if-changed` re-runs build.rs on a lockfile bump.
fn cargo_lock_fingerprint() -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    println!("cargo:rerun-if-changed=Cargo.lock");
    // Fail loud on an unreadable Cargo.lock rather than hashing the
    // empty-string default (a constant that would let machines with
    // different dependency sets share a cache entry) — mirrors
    // cast_analyzer_fingerprint's panic-on-read-failure posture.
    let lock = std::fs::read_to_string("Cargo.lock")
        .unwrap_or_else(|e| panic!("read Cargo.lock for dependency fingerprint: {e}"));
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(lock.as_bytes());
    h.finish()
}

/// Recursively collect non-test `.rs` files under `dir` into `out`.
/// A missing dir returns no files (tolerant for the recursion case);
/// the caller asserts each top-level watched dir exists, so a typo'd or
/// moved analyzer dir fails the build loudly rather than silently
/// dropping its fingerprint contribution.
fn collect_fingerprint_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_fingerprint_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && path.file_name().and_then(|n| n.to_str()) != Some("tests.rs")
        {
            out.push(path);
        }
    }
}
