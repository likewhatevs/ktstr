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
//! A second fixture links one compatible declaration into two separate test
//! binaries plus a private copy of ktstr's real `scx-ktstr` scheduler package.
//! Its bare verifier invocation must continue through recursive discovery, the
//! parent-owned scheduler prebuild and immutable artifact/ownership manifests,
//! and exactly one generated KVM cell. This closes both the success-path gap
//! left intentionally by the conflict fixture and the cross-binary duplicate
//! cell/record-writer regression.

use std::path::{Path, PathBuf};

const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");
const FIXTURE_RESOLVER: &str = "3";
const FIXTURE_TEST_DEBUG: &str = "line-tables-only";

fn toml_string(path: &Path) -> String {
    serde_json::to_string(&path.to_string_lossy()).expect("path JSON string")
}

fn shared_target_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("CARGO_TARGET_DIR") {
        let path = PathBuf::from(path);
        return if path.is_absolute() {
            path
        } else {
            ktstr::writable_source_path(env!("CARGO_MANIFEST_DIR")).join(path)
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

/// Keep nested workspaces below the repository so Cargo discovers the same
/// ancestor `.cargo/config.toml` (and therefore the same rustflags) as the
/// parent build and every sibling fixture sharing its target directory.
fn fixture_tempdir(ktstr_root: &Path) -> tempfile::TempDir {
    let scratch = ktstr_root.join("target/verifier-discovery-fixtures");
    std::fs::create_dir_all(&scratch).expect("create verifier fixture scratch directory");
    tempfile::Builder::new()
        .prefix("workspace-")
        .tempdir_in(scratch)
        .expect("fixture tempdir")
}

fn fixture_diagnostics_dir(temp: &tempfile::TempDir) -> PathBuf {
    std::env::var_os("KTSTR_BUILD_DIAGNOSTICS_DIR")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .map(|root| {
            root.join("verifier-discovery").join(
                temp.path()
                    .file_name()
                    .expect("verifier fixture tempdir has a basename"),
            )
        })
        .unwrap_or_else(|| temp.path().join("cargo-diagnostics"))
}

/// Render a fixture dependency with this parent test binary's ktstr features.
///
/// These nested workspaces intentionally share the parent target directory.
/// Keeping their feature fingerprint identical makes the conflict and success
/// fixtures share one dependency unit: narrower per-fixture dependencies made
/// Cargo build separate whole-ktstr variants while nextest saturated the host.
fn parent_ktstr_dependency(dependency: &str, ktstr_root: &Path) -> String {
    let mut features = Vec::new();
    for (name, enabled) in [
        ("integration", cfg!(feature = "integration")),
        ("wprof", cfg!(feature = "wprof")),
        ("pretty-labels", cfg!(feature = "pretty-labels")),
        ("remote-cache", cfg!(feature = "remote-cache")),
    ] {
        if enabled {
            features.push(name);
        }
    }
    format!(
        "{dependency} = {{ package = \"ktstr\", path = {}, optional = true, features = {} }}",
        toml_string(ktstr_root),
        serde_json::to_string(&features).expect("serialize fixture feature list"),
    )
}

/// Reproduce the parent integration-test dependency graph as well as its ktstr
/// feature set. Cargo fingerprints ktstr's direct dependency units, so root
/// dev-dependency feature unification (for example
/// `virtio-queue/test-utils`) is part of whether its library artifact can be
/// reused by the sibling nested build.
fn parent_dev_dependencies(ktstr_root: &Path) -> String {
    let manifest =
        std::fs::read_to_string(ktstr_root.join("Cargo.toml")).expect("read parent ktstr manifest");
    let marker = "\n[dev-dependencies]\n";
    let (_, table) = manifest
        .split_once(marker)
        .expect("parent manifest has a dev-dependencies table");
    let end = table.find("\n[").unwrap_or(table.len());
    format!("[dev-dependencies]\n{}", table[..end].trim_end())
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

/// Match nextest's one-cell success summary after removing presentation-only
/// ANSI CSI sequences. Parsing the count tokens (rather than merely searching
/// for a substring) keeps this an exact-once assertion: multi-cell summaries,
/// failed cells, and duplicate summary lines all reject.
fn exactly_one_verifier_cell_passed(stderr: &str) -> bool {
    let stderr = ktstr::test_support::strip_ansi_csi(stderr);
    stderr
        .lines()
        .filter(|line| {
            let Some((before, after)) = line.split_once(" test run: ") else {
                return false;
            };
            if before.split_whitespace().next_back() != Some("1") {
                return false;
            }
            let mut result = after.split_whitespace();
            result.next() == Some("1")
                && result
                    .next()
                    .is_some_and(|status| status.trim_end_matches(',') == "passed")
        })
        .count()
        == 1
}

/// The dependency renderer is driven directly by this test binary's `cfg!`
/// feature set. Keeping both aliases byte-identical statically prevents Cargo
/// from compiling two nested ktstr variants. The end-to-end tests below prove
/// smart feature activation independently: their declaration binaries have
/// `required-features`, so neither the conflict nor the generated cell can be
/// observed unless the package-qualified roots were enabled.
#[test]
fn nested_fixture_dependency_aliases_have_identical_cargo_fingerprint_inputs() {
    let ktstr_root = ktstr::writable_source_path(env!("CARGO_MANIFEST_DIR"));
    let direct = parent_ktstr_dependency("ktstr", &ktstr_root);
    let aliased = parent_ktstr_dependency("test_harness", &ktstr_root);
    let direct_value = direct
        .split_once(" = ")
        .expect("direct dependency assignment")
        .1;
    let aliased_value = aliased
        .split_once(" = ")
        .expect("aliased dependency assignment")
        .1;
    assert_eq!(
        direct_value, aliased_value,
        "dependency aliases must not create distinct nested ktstr variants",
    );

    let dev_dependencies = parent_dev_dependencies(&ktstr_root);
    assert!(
        dev_dependencies.starts_with("[dev-dependencies]\n")
            && !dev_dependencies["[dev-dependencies]\n".len()..].contains("\n["),
        "mirrored dev-dependencies must be exactly one narrowly delimited TOML table",
    );

    let root_manifest =
        std::fs::read_to_string(ktstr_root.join("Cargo.toml")).expect("read root manifest");
    assert!(
        root_manifest.contains(&format!(
            "[workspace]\nmembers = [\".\", \"ktstr-macros\", \"scx-ktstr\"]\n\
             resolver = \"{FIXTURE_RESOLVER}\"",
        )),
        "root and nested fixture workspaces must use the same resolver",
    );
    assert!(
        root_manifest.contains(&format!("[profile.test]\ndebug = \"{FIXTURE_TEST_DEBUG}\"",)),
        "root and nested fixture workspaces must use the same test profile",
    );
}

#[test]
fn exact_one_cell_summary_accepts_ansi_without_weakening_counts() {
    let colored = concat!(
        "\u{1b}[1mSummary\u{1b}[0m [ 28.848s] ",
        "\u{1b}[32m1 test run\u{1b}[0m: ",
        "\u{1b}[32m1 passed\u{1b}[0m, 306 skipped\n",
    );
    assert!(
        exactly_one_verifier_cell_passed(colored),
        "ANSI styling must not hide an exact one-cell success",
    );

    for not_exactly_one in [
        "Summary [ 1.000s] 2 test run: 2 passed",
        "Summary [ 1.000s] 11 test run: 1 passed",
        "Summary [ 1.000s] 1 test run: 0 passed",
        "Summary [ 1.000s] 1 test run: 1 failed",
        "Summary [ 1.000s] 1 test run: 1 passed\n\
         Summary [ 1.000s] 1 test run: 1 passed",
    ] {
        assert!(
            !exactly_one_verifier_cell_passed(not_exactly_one),
            "non-exact summary must reject: {not_exactly_one:?}",
        );
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
    let dependency_spec = parent_ktstr_dependency(dependency, ktstr_root);
    let dev_dependencies = parent_dev_dependencies(ktstr_root);
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

{dev_dependencies}

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
    std::fs::create_dir_all(root.join("src")).expect("create success member source directory");
    let dev_dependencies = parent_dev_dependencies(ktstr_root);
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
{}

{dev_dependencies}

[[test]]
name = "recursive_scheduler_declaration"
path = "tests/scheduler.rs"
required-features = ["verification-tests"]

[[test]]
name = "recursive_scheduler_declaration_duplicate"
path = "tests/scheduler_duplicate.rs"
required-features = ["verification-tests"]
"#,
            parent_ktstr_dependency("test_harness", ktstr_root),
        ),
    )
    .expect("write success member manifest");
    let declaration = r#"#![cfg(feature = "verification-tests")]

extern crate test_harness as ktstr;

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
"#;
    std::fs::write(root.join("src/lib.rs"), declaration)
        .expect("write dependency-level feature-gated scheduler declaration");

    let integration_test = r#"use verifier_e2e::RECURSIVE_DISCOVERY_SUCCESS_SCHEDULER;

#[test]
fn declaration_binary_was_built() {
    assert_eq!(
        RECURSIVE_DISCOVERY_SUCCESS_SCHEDULER.name,
        "recursive-discovery-success",
    );
}
"#;
    std::fs::write(root.join("tests/scheduler.rs"), integration_test)
        .expect("write successful feature-gated scheduler consumer");
    std::fs::write(root.join("tests/scheduler_duplicate.rs"), integration_test)
        .expect("write duplicate feature-gated scheduler consumer");
}

#[test]
fn bare_verifier_recursively_discovers_feature_gated_workspace_test_binaries() {
    let ktstr_root = ktstr::writable_source_path(env!("CARGO_MANIFEST_DIR"));
    let temp = fixture_tempdir(&ktstr_root);
    let workspace = temp.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create fixture workspace");
    std::fs::write(
        workspace.join("Cargo.toml"),
        format!(
            "[workspace]\n\
             members = [\"alpha\", \"beta\"]\n\
             resolver = \"{FIXTURE_RESOLVER}\"\n\n\
             [profile.test]\n\
             debug = \"{FIXTURE_TEST_DEBUG}\"\n",
        ),
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
    let diagnostics = fixture_diagnostics_dir(&temp);

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .current_dir(&workspace)
        .args(["ktstr", "verifier"])
        .env("CARGO_TARGET_DIR", shared_target_dir())
        .env("CARGO_NET_OFFLINE", "true")
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .env("KTSTR_CACHE_DIR", temp.path().join("cache"))
        .env("KTSTR_RUNS_ROOT", temp.path().join("runs"))
        .env("KTSTR_BUILD_DIAGNOSTICS_DIR", &diagnostics)
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

    let ktstr_root = ktstr::writable_source_path(env!("CARGO_MANIFEST_DIR"));
    let temp = fixture_tempdir(&ktstr_root);
    let workspace = temp.path().join("workspace");
    std::fs::create_dir(&workspace).expect("create success fixture workspace");
    std::fs::write(
        workspace.join("Cargo.toml"),
        format!(
            r#"[workspace]
members = ["verifier-e2e", "scx-ktstr"]
resolver = "{FIXTURE_RESOLVER}"

[workspace.package]
edition = "2024"
rust-version = "1.94.1"
license = "GPL-2.0-only"
repository = "https://github.com/likewhatevs/ktstr"

[profile.release]
lto = "thin"
panic = "abort"

[profile.test]
debug = "{FIXTURE_TEST_DEBUG}"
"#,
        ),
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
    let diagnostics = fixture_diagnostics_dir(&temp);

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .current_dir(&workspace)
        // Intentionally bare: the already-resolved kernel is supplied through
        // ktstr's normal environment contract, while metadata must discover
        // the feature-gated declaration and scheduler workspace without any
        // Cargo or verifier selector.
        .args(["ktstr", "verifier"])
        .env("CARGO_TARGET_DIR", shared_target_dir())
        .env("KTSTR_RUNS_ROOT", temp.path().join("runs"))
        .env("KTSTR_BUILD_DIAGNOSTICS_DIR", &diagnostics)
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
        exactly_one_verifier_cell_passed(&stderr),
        "two warmed binaries carrying the same full declaration must elect one lister, \
         one VM launch, and therefore one result writer:\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
    assert!(
        stdout.contains("recursive-discovery-success: 1 ✅  0 ❌")
            && stdout.contains("4cpu-1llc-nosmt"),
        "the parent result grid must prove the generated KVM cell completed successfully:\n\
         stdout:\n{stdout}\nstderr:\n{stderr}",
    );
}
