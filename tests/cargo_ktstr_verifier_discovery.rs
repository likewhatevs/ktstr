//! End-to-end coverage for bare `cargo ktstr verifier` workspace discovery.
//!
//! The discovery fixture uses two workspace members whose scheduler
//! declarations live in differently named, feature-gated integration-test
//! binaries. Both declare the same scheduler name with intentionally different
//! guest execution settings. The real cargo-ktstr process must therefore:
//!
//! 1. widen a bare verifier invocation to the full workspace;
//! 2. infer and package-qualify both optional-ktstr feature roots;
//! 3. ask nextest to build both selected test binaries; and
//! 4. probe both linked scheduler registries.
//!
//! Discovery then fails on the deliberate declaration conflict, before any
//! scheduler prebuild or KVM cell can run. Observing both declaration payloads
//! in that conflict is the end-to-end proof of recursive discovery.
//!
//! A second fixture carries one compatible declaration plus a private copy of
//! ktstr's real `scx-ktstr` scheduler package. Its bare verifier invocation
//! must continue through recursive discovery, the parent-owned scheduler
//! prebuild and immutable artifact manifest, and one generated KVM cell. This
//! closes the success-path gap left intentionally by the conflict fixture.

use std::path::{Path, PathBuf};

const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

fn toml_string(path: &Path) -> String {
    serde_json::to_string(&path.to_string_lossy()).expect("path JSON string")
}

fn shared_target_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("CARGO_TARGET_DIR") {
        let path = PathBuf::from(path);
        return if path.is_absolute() {
            path
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
        };
    }

    // Integration tests live at <target>/<profile>/deps/<binary>. Reusing the
    // parent target directory lets the nested fixture build share the exact
    // ktstr/dependency artifacts which compiled this test.
    std::env::current_exe()
        .expect("current test executable")
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("integration test executable under target/<profile>/deps")
        .to_path_buf()
}

fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination)
        .unwrap_or_else(|error| panic!("create {}: {error}", destination.display()));
    for entry in std::fs::read_dir(source)
        .unwrap_or_else(|error| panic!("read {}: {error}", source.display()))
    {
        let entry = entry.expect("read fixture source entry");
        let source = entry.path();
        let destination = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .unwrap_or_else(|error| panic!("inspect {}: {error}", source.display()));
        if file_type.is_dir() {
            copy_tree(&source, &destination);
        } else if file_type.is_file() {
            std::fs::copy(&source, &destination).unwrap_or_else(|error| {
                panic!(
                    "copy {} to {}: {error}",
                    source.display(),
                    destination.display()
                )
            });
        } else {
            panic!("unsupported fixture source entry {}", source.display());
        }
    }
}

fn write_member(
    workspace: &Path,
    package: &str,
    dependency: &str,
    feature: &str,
    identity: &str,
    sysctl_value: &str,
    ktstr_root: &Path,
) {
    let root = workspace.join(package);
    std::fs::create_dir_all(root.join("tests")).expect("create member tests directory");
    let dependency_spec = if dependency == "ktstr" {
        format!(
            "ktstr = {{ path = {}, optional = true, default-features = false }}",
            toml_string(ktstr_root),
        )
    } else {
        format!(
            "{dependency} = {{ package = \"ktstr\", path = {}, optional = true, \
             default-features = false }}",
            toml_string(ktstr_root),
        )
    };
    std::fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "{package}"
version = "0.1.0"
edition = "2024"

[features]
{feature} = ["dep:{dependency}"]

[dependencies]
{dependency_spec}

[[test]]
name = "{package}_scheduler_declaration"
path = "tests/scheduler.rs"
required-features = ["{feature}"]
"#,
        ),
    )
    .expect("write member manifest");
    let dependency_alias = if dependency == "ktstr" {
        String::new()
    } else {
        format!("extern crate {dependency} as ktstr;\n\n")
    };
    std::fs::write(
        root.join("tests/scheduler.rs"),
        format!(
            r#"{dependency_alias}use ktstr::declare_scheduler;
use ktstr::test_support::Sysctl;

declare_scheduler!(RECURSIVE_DISCOVERY_SCHEDULER, {{
    name = "recursive-discovery-shared",
    binary_path = "/bin/true",
    sched_args = ["--shared"],
    sysctls = [Sysctl::new("kernel.numa_balancing", "{sysctl_value}")],
    kargs = ["identity={identity}"],
    cgroup_parent = "/{identity}",
    config_file = "{identity}.toml",
}});

#[test]
fn declaration_binary_was_built() {{}}
"#,
        ),
    )
    .expect("write feature-gated scheduler declaration");
}

fn write_success_member(workspace: &Path, ktstr_root: &Path) {
    let root = workspace.join("verifier-e2e");
    std::fs::create_dir_all(root.join("tests")).expect("create success member tests directory");
    std::fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "verifier-e2e"
version = "0.1.0"
edition = "2024"

[features]
verification-tests = ["dep:test_harness"]

[dependencies]
test_harness = {{ package = "ktstr", path = {}, optional = true, default-features = false, features = ["vendored"] }}

[[test]]
name = "recursive_scheduler_declaration"
path = "tests/scheduler.rs"
required-features = ["verification-tests"]
"#,
            toml_string(ktstr_root),
        ),
    )
    .expect("write success member manifest");
    std::fs::write(
        root.join("tests/scheduler.rs"),
        r#"extern crate test_harness as ktstr;

use ktstr::declare_scheduler;
use ktstr::test_support::TopologyConstraints;

declare_scheduler!(RECURSIVE_DISCOVERY_SUCCESS_SCHEDULER, {
    name = "recursive-discovery-success",
    binary = "scx-ktstr",
    topology = (1, 1, 4, 1),
    constraints = TopologyConstraints {
        min_numa_nodes: 1,
        max_numa_nodes: Some(1),
        min_llcs: 1,
        max_llcs: Some(1),
        requires_smt: false,
        min_cpus: 4,
        max_cpus: Some(4),
    },
});

#[test]
fn declaration_binary_was_built() {}
"#,
    )
    .expect("write successful feature-gated scheduler declaration");
}

#[test]
fn bare_verifier_recursively_discovers_feature_gated_workspace_test_binaries() {
    let temp = tempfile::tempdir().expect("fixture tempdir");
    let workspace = temp.path().join("workspace");
    let ktstr_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir(&workspace).expect("create fixture workspace");
    std::fs::write(
        workspace.join("Cargo.toml"),
        "[workspace]\nmembers = [\"alpha\", \"beta\"]\nresolver = \"2\"\n",
    )
    .expect("write workspace manifest");
    std::fs::copy(ktstr_root.join("Cargo.lock"), workspace.join("Cargo.lock"))
        .expect("seed fixture with ktstr's toolchain-compatible dependency lock");

    write_member(
        &workspace,
        "alpha",
        "ktstr",
        "scheduler-tests",
        "alpha",
        "0",
        &ktstr_root,
    );
    write_member(
        &workspace,
        "beta",
        "test_harness",
        "verifier-fixtures",
        "beta",
        "1",
        &ktstr_root,
    );

    // Bare verifier resolution accepts a preselected kernel directory through
    // KTSTR_KERNEL. The conflict is reached before a VM opens this image, so a
    // correctly named placeholder keeps the fixture host- and KVM-independent.
    let kernel = temp.path().join("kernel");
    std::fs::create_dir(&kernel).expect("create kernel fixture");
    #[cfg(target_arch = "x86_64")]
    std::fs::write(kernel.join("bzImage"), b"discovery-only").expect("write kernel placeholder");
    #[cfg(target_arch = "aarch64")]
    std::fs::write(kernel.join("Image"), b"discovery-only").expect("write kernel placeholder");

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .current_dir(&workspace)
        .args(["ktstr", "verifier"])
        .env("CARGO_TARGET_DIR", shared_target_dir())
        .env("CARGO_NET_OFFLINE", "true")
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .env("KTSTR_CACHE_DIR", temp.path().join("cache"))
        .env("KTSTR_RUNS_ROOT", temp.path().join("runs"))
        .env(ktstr::KTSTR_KERNEL_ENV, &kernel)
        // The parent nextest process exports its selected profile. This
        // nested, hermetic workspace deliberately has no project nextest
        // config, so inheriting (for example) CI's `ci` profile would make
        // discovery fail before cargo-ktstr can inspect either registry.
        .env_remove("NEXTEST_PROFILE")
        .output()
        .expect("run bare cargo ktstr verifier");

    assert!(
        !output.status.success(),
        "the deliberately conflicting declarations must fail discovery",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("conflicting declarations for scheduler")
            && stderr.contains("recursive-discovery-shared"),
        "both registries should meet at declaration conflict detection:\n{stderr}",
    );
    for marker in [
        "kernel.numa_balancing",
        "identity=alpha",
        "identity=beta",
        "/alpha",
        "/beta",
        "alpha.toml",
        "beta.toml",
    ] {
        assert!(
            stderr.contains(marker),
            "the conflict must include execution-identity marker {marker:?} from both \
             feature-gated test binaries:\n{stderr}",
        );
    }
    assert!(
        !stderr.contains("dispatching to nextest (verifier/ cells only)"),
        "the discovery conflict must abort before the KVM-running nextest phase:\n{stderr}",
    );
}

#[test]
fn bare_verifier_runs_recursively_discovered_scheduler_cell_end_to_end() {
    if let Err(error) = ktstr::cli::check_kvm() {
        eprintln!("skipping recursive verifier KVM e2e: {error:#}");
        return;
    }
    let Some(kernel) = ktstr::find_kernel().expect("resolve verifier e2e kernel") else {
        eprintln!("skipping recursive verifier KVM e2e: no bootable kernel is available");
        return;
    };
    #[cfg(target_arch = "x86_64")]
    let expected_image_name = "bzImage";
    #[cfg(target_arch = "aarch64")]
    let expected_image_name = "Image";
    if kernel.file_name().and_then(|name| name.to_str()) != Some(expected_image_name) {
        eprintln!(
            "skipping recursive verifier KVM e2e: {} is not a ktstr-built {expected_image_name}",
            kernel.display(),
        );
        return;
    }
    let kernel_dir = kernel
        .parent()
        .expect("a resolved ktstr kernel image has a parent directory");

    let temp = tempfile::tempdir().expect("fixture tempdir");
    let workspace = temp.path().join("workspace");
    let ktstr_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir(&workspace).expect("create success fixture workspace");
    std::fs::write(
        workspace.join("Cargo.toml"),
        r#"[workspace]
members = ["verifier-e2e", "scx-ktstr"]
resolver = "2"

[workspace.package]
edition = "2024"
rust-version = "1.94.1"
license = "GPL-2.0-only"
repository = "https://github.com/likewhatevs/ktstr"

[profile.release]
lto = "thin"
panic = "abort"
"#,
    )
    .expect("write success workspace manifest");
    std::fs::copy(ktstr_root.join("Cargo.lock"), workspace.join("Cargo.lock"))
        .expect("seed success fixture with ktstr's dependency lock");
    copy_tree(
        &ktstr_root.join("build_support"),
        &workspace.join("build_support"),
    );
    copy_tree(&ktstr_root.join("scx-ktstr"), &workspace.join("scx-ktstr"));
    write_success_member(&workspace, &ktstr_root);

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .current_dir(&workspace)
        // Intentionally bare: the already-resolved kernel is supplied through
        // ktstr's normal environment contract, while metadata must discover
        // the feature-gated declaration and scheduler workspace without any
        // Cargo or verifier selector.
        .args(["ktstr", "verifier"])
        .env("CARGO_TARGET_DIR", shared_target_dir())
        .env("KTSTR_RUNS_ROOT", temp.path().join("runs"))
        .env(ktstr::KTSTR_KERNEL_ENV, kernel_dir)
        // The parent nextest process exports its selected profile, but the
        // nested fixture has no matching project profile. The verifier command
        // owns the inner nextest profile independently.
        .env_remove("NEXTEST_PROFILE")
        .output()
        .expect("run successful bare cargo ktstr verifier");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "bare cargo ktstr verifier must execute its recursively discovered cell\n\
         status: {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code(),
    );
    assert!(
        stderr.contains("prebuilding 1 scheduler package(s)")
            && stderr.contains("dispatching to nextest (verifier/ cells only)"),
        "the successful path must cross scheduler prebuild and generated-cell dispatch:\n{stderr}",
    );
    assert!(
        stdout.contains("recursive-discovery-success: 1 ✅  0 ❌")
            && stdout.contains("4cpu-1llc-nosmt"),
        "the parent result grid must prove the generated KVM cell completed successfully:\n\
         stdout:\n{stdout}\nstderr:\n{stderr}",
    );
}
