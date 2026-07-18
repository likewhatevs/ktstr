//! End-to-end coverage for bare `cargo ktstr verifier` workspace discovery.
//!
//! The fixture uses two workspace members whose scheduler declarations live in
//! differently named, feature-gated integration-test binaries. Both declare
//! the same scheduler name with intentionally different arguments. The real
//! cargo-ktstr process must therefore:
//!
//! 1. widen a bare verifier invocation to the full workspace;
//! 2. infer and package-qualify both optional-ktstr feature roots;
//! 3. ask nextest to build both selected test binaries; and
//! 4. probe both linked scheduler registries.
//!
//! Discovery then fails on the deliberate declaration conflict, before any
//! scheduler prebuild or KVM cell can run. Observing both declaration payloads
//! in that conflict is the end-to-end proof of recursive discovery.

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

fn write_member(
    workspace: &Path,
    package: &str,
    dependency: &str,
    feature: &str,
    scheduler_args: &str,
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

declare_scheduler!(RECURSIVE_DISCOVERY_SCHEDULER, {{
    name = "recursive-discovery-shared",
    binary_path = "/bin/true",
    sched_args = ["{scheduler_args}"],
}});

#[test]
fn declaration_binary_was_built() {{}}
"#,
        ),
    )
    .expect("write feature-gated scheduler declaration");
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
        "--from-alpha",
        &ktstr_root,
    );
    write_member(
        &workspace,
        "beta",
        "test_harness",
        "verifier-fixtures",
        "--from-beta",
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
    assert!(
        stderr.contains("--from-alpha") && stderr.contains("--from-beta"),
        "the conflict must contain declarations from both feature-gated test binaries:\n{stderr}",
    );
    assert!(
        !stderr.contains("dispatching to nextest (verifier/ cells only)"),
        "the discovery conflict must abort before the KVM-running nextest phase:\n{stderr}",
    );
}
