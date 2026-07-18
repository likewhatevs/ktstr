//! `cargo ktstr verifier` subcommand: thin wrapper around
//! `cargo nextest run` filtered to the `verifier/` test-name prefix.
//!
//! Each test binary that links the `ktstr` crate's `test_support`
//! module and has at least one `declare_scheduler!` declaration emits
//! one nextest test per (declared scheduler × kernel-list entry ×
//! accepted topology preset) cell — the verifier sweeps each declared
//! scheduler ACROSS topologies, because whether a scheduler attaches
//! and dispatches is topology-DEPENDENT (a scheduler can attach on one
//! topology and wedge on another). Every cell boots in no_perf_mode, so
//! a preset is emitted only when the scheduler's constraints accept it
//! under `accepts_no_perf_mode`. A cell PASSes only when its scheduler
//! (1) verifies (BPF loads — `verified_insns`), (2) attaches (the guest
//! attach gate confirms sched_ext `enabled`), AND (3) dispatches an
//! injected SpinWait workload (the guest emits a `WorkloadDispatched`
//! frame when a worker makes forward progress after attach). The cell
//! lister and handler live in
//! `src/test_support/dispatch.rs::list_verifier_cells_all` and
//! `run_verifier_cell`. Nextest provides per-cell parallelism, retries,
//! and failure isolation; this dispatcher resolves the `--kernel`
//! argument into the `KTSTR_KERNEL_LIST` env-var matrix dimension the
//! test binary's lister walks, plumbs `--raw` via `KTSTR_VERIFIER_RAW`,
//! restricts the sweep to one declared scheduler via `--scheduler
//! <NAME>` (plumbed through `KTSTR_VERIFIER_SCHEDULER`), and spawns
//! nextest filtered to the CELL names only (`test(/^verifier/) &
//! !test(/^verifier::/)` — the `verifier/...` cells, NOT the verifier
//! module's own `verifier::tests::*` unit tests, which also start with
//! "verifier"). The trailing `args` are forwarded verbatim to that
//! `cargo nextest run` (a nextest filterset, `--cargo-profile`, ...);
//! native flags may appear in any order relative to them and no `--`
//! separator is needed (see the bin's `argsplit` module). The
//! `declare_scheduler!` verifier cells carry no `required-features`,
//! but a consumer may place the declaration in a feature-gated test
//! target. For a direct compatible optional ktstr dependency, the dispatcher
//! follows ktstr-only feature aliases and auto-injects only their
//! package-qualified roots. Conventional gated targets therefore link their
//! declarations without a manual `--features` passthrough or an over-broad
//! `--all-features`.
//! The scheduler-under-test builds release by default, and each cell boots
//! with performance mode disabled (its `verified_insns` count is
//! perf-mode-independent, so cells take only a shared LLC reservation
//! and no longer starve each other on the LLC lock — see
//! `collect_verifier_output`). After nextest returns, the dispatcher
//! reads each cell's PASS/FAIL record (written under
//! `KTSTR_VERIFIER_RESULT_DIR`) and prints one `verified_insns` table
//! per declared scheduler followed by one PASS/FAIL grid per declared
//! scheduler (rows = topology, cols = kernel).
//!
//! `KTSTR_KERNEL_LIST` is ALWAYS populated by this dispatcher — even
//! with no `--kernel` flag the dispatcher auto-discovers one kernel
//! and synthesizes a single-entry list with a path-derived label.
//! That keeps the test-binary cell handler's lookup path unified
//! (always look up by label in the list, never fall through to a
//! resolve_test_kernel single-kernel fallback that would silently
//! run a cell against an unrelated kernel).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::semver::Version;
use cargo_metadata::{Metadata, PackageId};

use crate::feature_discovery::{
    MetadataMode, PackageFeatureActivation, VersionScope, has_package_selector,
    has_workspace_selector, infer_ktstr_feature_roots, inject_feature_activations,
    package_spec_name, query_metadata, selected_workspace_packages,
};
#[cfg(test)]
use crate::feature_discovery::{explicit_package_exclusions, explicit_package_selection};
use crate::kernel::{
    encode_kernel_list, path_kernel_label, resolve_kernel_image, resolve_kernel_set,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoverSchedulerRequest {
    scheduler: String,
    package: String,
    manifest_dir: String,
    workspace_root: PathBuf,
    package_id: PackageId,
}

#[derive(Debug)]
struct WorkspaceSchedulerBuild {
    root: PathBuf,
    /// package name -> Cargo package ID
    packages: BTreeMap<String, PackageId>,
    requests: Vec<DiscoverSchedulerRequest>,
}

/// Own one verifier result directory from creation through final report
/// rendering. `Drop` is the error-path cleanup: once this guard exists, any
/// later `?` (harness warm-up, declaration probe, scheduler prebuild,
/// snapshot, manifest write, or nextest spawn) removes the partially prepared
/// run instead of orphaning it.
struct VerifierResultDir {
    path: PathBuf,
    #[cfg(test)]
    cleanup_count: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

impl VerifierResultDir {
    fn create(temp_root: &Path) -> Result<Self, String> {
        sweep_stale_result_dirs(temp_root);
        let path = temp_root.join(format!("ktstr-verifier-results-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path)
            .map_err(|error| format!("create verifier result dir {}: {error}", path.display()))?;
        Ok(Self {
            path,
            #[cfg(test)]
            cleanup_count: None,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for VerifierResultDir {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(count) = &self.cleanup_count {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Sweep verifier result dirs orphaned by interrupted prior runs.
///
/// The result dir is keyed on the dispatcher pid. The outer interrupt guard
/// lets SIGINT/SIGTERM reach `VerifierResultDir::drop`, but SIGKILL and
/// abort-style crashes cannot run Rust cleanup and therefore orphan the dir.
/// This runs at startup and removes every
/// `ktstr-verifier-results-<pid>` dir whose owning pid is no longer alive
/// (`kill(pid, 0)` -> ESRCH), mirroring `cleanup_stale_shm`'s next-run
/// reclamation. A dir owned by a LIVE pid is a concurrent verifier run and
/// is left untouched.
fn sweep_stale_result_dirs(temp_root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(temp_root) else {
        return;
    };
    let self_pid = std::process::id();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|n| n.strip_prefix("ktstr-verifier-results-"))
            .and_then(|p| p.parse::<i32>().ok())
        else {
            continue;
        };
        // Skip non-positive pids (kill(0)/kill(-N) probe process GROUPS,
        // not a single process) and our own dir (owned by the run below).
        if pid <= 0 || pid == self_pid as i32 {
            continue;
        }
        // Only a definitively-dead owner (ESRCH) is an orphan. A live pid
        // (Ok) or a permission error (EPERM) is left alone.
        let dead = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH),
        );
        if dead {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// One workspace package whose test-link closure contains only the
/// cargo-ktstr CLI's ktstr version, so its verifier declarations are safe to
/// enumerate with this CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompatibleVerifierPackage {
    name: String,
    /// Minimal package feature roots whose definitions exclusively activate
    /// this package's optional dependency on the current ktstr.
    verifier_features: Vec<String>,
}

/// One workspace package whose test-link closure contains an older ktstr.
///
/// Exclusion is package-wide: cargo metadata cannot attribute distributed
/// scheduler registrations to one test target when a package links multiple
/// ktstr versions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OlderVerifierPackage {
    name: String,
    versions: Vec<Version>,
}

/// One workspace package whose test-link closure contains ktstr newer than
/// this cargo-ktstr CLI. Unlike an older package, this is an error when the
/// package is in the requested Cargo selection: the CLI predates its protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NewerVerifierPackage {
    name: String,
    versions: Vec<Version>,
}

/// Package-level compatibility partition for verifier test discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifierPackagePlan {
    compatible: Vec<CompatibleVerifierPackage>,
    older: Vec<OlderVerifierPackage>,
    newer: Vec<NewerVerifierPackage>,
}

/// Whether one cargo-metadata dependency edge can be linked into a test target.
///
/// The workspace member's own dev-dependencies participate in its tests.
/// Development dependencies of a dependency do not; normal dependencies keep
/// traversing at every depth. Empty `dep_kinds` is cargo metadata's
/// backwards-compatible spelling for a normal dependency.
fn test_link_edge(dep: &cargo_metadata::NodeDep, workspace_root: bool) -> bool {
    dep.dep_kinds.is_empty()
        || dep.dep_kinds.iter().any(|kind| {
            matches!(kind.kind, cargo_metadata::DependencyKind::Normal)
                || (workspace_root
                    && matches!(kind.kind, cargo_metadata::DependencyKind::Development))
        })
}

/// Collect every ktstr version in one workspace member's package-level test
/// link closure.
///
/// This deliberately walks beyond the direct edge. A current ktstr test
/// package can also link a dependency carrying an old ktstr and therefore an
/// old distributed scheduler registry. Package-level metadata cannot prove
/// which individual test binary retains that dependency, so any such mixed
/// package is excluded conservatively.
fn linked_ktstr_versions(
    member_id: &PackageId,
    packages: &HashMap<&PackageId, &cargo_metadata::Package>,
    nodes: &HashMap<&PackageId, &cargo_metadata::Node>,
) -> Vec<Version> {
    let mut versions = Vec::new();
    let mut seen = HashSet::new();
    let mut pending = vec![(member_id, true)];

    while let Some((package_id, workspace_root)) = pending.pop() {
        if !seen.insert(package_id.clone()) {
            continue;
        }
        let Some(package) = packages.get(package_id).copied() else {
            continue;
        };
        if package.name == "ktstr" {
            versions.push(package.version.clone());
            // A ktstr package cannot contribute a second scheduler registry
            // through its own ordinary dependencies.
            continue;
        }
        let Some(node) = nodes.get(package_id).copied() else {
            continue;
        };
        for dep in &node.deps {
            if !test_link_edge(dep, workspace_root) {
                continue;
            }
            pending.push((&dep.pkg, false));
        }
    }

    versions.sort_by(|a, b| a.cmp_precedence(b));
    versions.dedup_by(|a, b| a.cmp_precedence(b).is_eq());
    versions
}

/// Partition workspace members by the ktstr versions linked into their tests.
///
/// - no linked ktstr: irrelevant to verifier declaration discovery;
/// - exactly the CLI version: compatible;
/// - any older version (including old+current): skip the whole package;
/// - any newer version: record an error candidate; the caller first applies
///   explicit Cargo package selection so an unrelated newer package cannot
///   abort a deliberately scoped current-package run.
fn verifier_package_plan(meta: &Metadata, cli: &Version) -> Result<VerifierPackagePlan, String> {
    let resolve = meta
        .resolve
        .as_ref()
        .ok_or_else(|| "cargo metadata omitted the dependency resolve graph".to_string())?;
    let packages: HashMap<_, _> = meta.packages.iter().map(|p| (&p.id, p)).collect();
    let nodes: HashMap<_, _> = resolve.nodes.iter().map(|n| (&n.id, n)).collect();
    let mut compatible = Vec::new();
    let mut older = Vec::new();
    let mut newer = Vec::new();

    for member_id in &meta.workspace_members {
        let Some(member) = packages.get(member_id).copied() else {
            continue;
        };
        let versions = linked_ktstr_versions(member_id, &packages, &nodes);
        if versions.is_empty() {
            continue;
        }
        if versions.iter().any(|v| v.cmp_precedence(cli).is_gt()) {
            newer.push(NewerVerifierPackage {
                name: member.name.to_string(),
                versions,
            });
        } else if versions.iter().any(|v| v.cmp_precedence(cli).is_lt()) {
            older.push(OlderVerifierPackage {
                name: member.name.to_string(),
                versions,
            });
        } else {
            compatible.push(CompatibleVerifierPackage {
                name: member.name.to_string(),
                verifier_features: infer_ktstr_feature_roots(member, VersionScope::Matches(cli)),
            });
        }
    }

    compatible.sort_by(|a, b| a.name.cmp(&b.name));
    older.sort_by(|a, b| a.name.cmp(&b.name));
    newer.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(VerifierPackagePlan {
        compatible,
        older,
        newer,
    })
}

/// Resolve the verifier package partition before either nextest warm-up or run.
fn query_verifier_package_plan(args: &[String]) -> Result<VerifierPackagePlan, String> {
    let cli = Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("cargo-ktstr's own CARGO_PKG_VERSION is valid semver");
    // First inspect workspace manifests without resolving optional
    // dependencies, then resolve only the ktstr-specific gates inferred from
    // those manifests. This gives recursive linked-version classification the
    // exact graph compilation will use without a broad all-features resolve.
    let manifests = query_metadata(args, MetadataMode::NoDeps)
        .map_err(|error| format!("cargo ktstr verifier: {error}"))?;
    let member_ids = manifests.workspace_members.iter().collect::<HashSet<_>>();
    let activations = manifests
        .packages
        .iter()
        .filter(|package| member_ids.contains(&package.id))
        .filter_map(|package| {
            let features = infer_ktstr_feature_roots(package, VersionScope::Any);
            (!features.is_empty()).then(|| PackageFeatureActivation {
                package: package.name.to_string(),
                features,
            })
        })
        .collect::<Vec<_>>();
    let resolution_args = inject_feature_activations(args.to_vec(), &activations);
    let metadata = query_metadata(&resolution_args, MetadataMode::Default)
        .map_err(|error| format!("cargo ktstr verifier: {error}"))?;
    let mut plan = verifier_package_plan(&metadata, &cli)?;

    // A bare verifier deliberately widens beyond Cargo's default members.
    // Once the operator supplies package selection, however, classify only
    // the exact/globbed workspace packages Cargo selected. A lone --exclude
    // applies to that widened workspace selection, so synthesize --workspace
    // for metadata selection without changing the forwarded Cargo argv.
    if has_package_selector(args) {
        let selection_args;
        let args = if has_workspace_selector(args) || has_explicit_package_selector(args) {
            args
        } else {
            selection_args = std::iter::once("--workspace".to_string())
                .chain(args.iter().cloned())
                .collect::<Vec<_>>();
            &selection_args
        };
        if let Some(packages) = selected_workspace_packages(&metadata, args) {
            let selected = packages
                .into_iter()
                .map(|package| package.name.to_string())
                .collect::<HashSet<_>>();
            plan.compatible
                .retain(|package| selected.contains(&package.name));
            plan.older
                .retain(|package| selected.contains(&package.name));
            plan.newer
                .retain(|package| selected.contains(&package.name));
        }
    }
    Ok(plan)
}

fn has_explicit_package_selector(args: &[String]) -> bool {
    args.iter()
        .take_while(|argument| argument.as_str() != "--")
        .any(|argument| {
            matches!(argument.as_str(), "-p" | "--package")
                || argument.starts_with("--package=")
                || (argument.starts_with("-p") && argument.len() > 2)
        })
}

fn format_older_package_skip(package: &OlderVerifierPackage) -> String {
    let versions = package
        .versions
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("/");
    format!(
        "cargo ktstr verifier: skipping {} (ktstr {versions}): this test is older; update or exclude it",
        package.name,
    )
}

#[cfg(test)]
fn restrict_plan_to_explicit_selection(
    mut plan: VerifierPackagePlan,
    args: &[String],
) -> VerifierPackagePlan {
    let excluded = explicit_package_exclusions(args);
    plan.compatible
        .retain(|package| !excluded.contains(&package.name));
    plan.older
        .retain(|package| !excluded.contains(&package.name));
    plan.newer
        .retain(|package| !excluded.contains(&package.name));

    let Some(selected) = explicit_package_selection(args) else {
        return plan;
    };
    plan.compatible
        .retain(|package| selected.contains(&package.name));
    plan.older
        .retain(|package| selected.contains(&package.name));
    plan.newer
        .retain(|package| selected.contains(&package.name));
    plan
}

/// Remove exact `-p` selectors for packages already classified as old.
///
/// This matters for a mixed explicit selection: the invariant nextest package
/// gate prevents an old binary from listing, but dropping its Cargo selector
/// also prevents its obsolete dependency stack from being compiled at all.
/// A `--` separator ends Cargo option parsing; every token from that boundary
/// onward belongs to the test binary and must be preserved verbatim.
fn drop_older_package_selectors(args: &[String], older: &[OlderVerifierPackage]) -> Vec<String> {
    let old_names: HashSet<&str> = older.iter().map(|package| package.name.as_str()).collect();
    let mut out = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            out.extend_from_slice(&args[index..]);
            break;
        }
        if matches!(arg.as_str(), "-p" | "--package") {
            if let Some(spec) = args.get(index + 1)
                && package_spec_name(spec).is_some_and(|name| old_names.contains(name))
            {
                index += 2;
                continue;
            }
            out.push(arg.clone());
            if let Some(spec) = args.get(index + 1) {
                out.push(spec.clone());
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        let old_equals = arg
            .strip_prefix("--package=")
            .or_else(|| arg.strip_prefix("-p").filter(|spec| !spec.is_empty()))
            .and_then(package_spec_name)
            .is_some_and(|name| old_names.contains(name));
        if !old_equals {
            out.push(arg.clone());
        }
        index += 1;
    }
    out
}

/// Whether an explicit package selector needs Cargo-style pattern expansion.
///
/// Exact names and source-qualified `#name@version` IDs can be pruned one by
/// one above. Globs and legacy full IDs cannot, so a mixed current+old
/// selection is rewritten to the already-classified compatible packages.
fn has_non_exact_package_selector(args: &[String]) -> bool {
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            break;
        }
        let spec = if matches!(argument.as_str(), "-p" | "--package") {
            index += 1;
            args.get(index).map(String::as_str)
        } else {
            argument
                .strip_prefix("--package=")
                .or_else(|| argument.strip_prefix("-p").filter(|spec| !spec.is_empty()))
        };
        if spec.is_some_and(|spec| package_spec_name(spec).is_none()) {
            return true;
        }
        index += 1;
    }
    false
}

/// Replace every Cargo-side package selector with exact compatible packages.
fn replace_package_selectors(
    args: &[String],
    compatible: &[CompatibleVerifierPackage],
) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len() + compatible.len() * 2);
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            for package in compatible {
                out.push("-p".to_string());
                out.push(package.name.clone());
            }
            out.extend_from_slice(&args[index..]);
            return out;
        }
        if matches!(argument.as_str(), "-p" | "--package") {
            index += usize::from(args.get(index + 1).is_some()) + 1;
            continue;
        }
        if argument.starts_with("--package=") || (argument.starts_with("-p") && argument.len() > 2)
        {
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    for package in compatible {
        out.push("-p".to_string());
        out.push(package.name.clone());
    }
    out
}

/// Build the one invariant nextest filter used by both warm-up and run.
///
/// Nextest unions multiple `-E` arguments, so forwarded user filtersets are
/// extracted and intersected with the verifier/package gate. This prevents a
/// user filter from widening discovery back into a test binary linked against
/// old ktstr.
fn build_scoped_nextest_args(
    nextest_profile: Option<&str>,
    forward: &[String],
    plan: &VerifierPackagePlan,
) -> Vec<String> {
    let (user_filtersets, rest) = crate::run_cargo::extract_nextest_filtersets(forward.to_vec());
    let cell_gate = "test(/^verifier/) & !test(/^verifier::/)";
    let package_gate = plan
        .compatible
        .iter()
        .map(|package| format!("package(={})", package.name))
        .collect::<Vec<_>>()
        .join(" | ");
    let invariant = if package_gate.is_empty() {
        cell_gate.to_string()
    } else {
        format!("({cell_gate}) & ({package_gate})")
    };
    let filter = if user_filtersets.is_empty() {
        invariant
    } else {
        let user_union = user_filtersets
            .iter()
            .map(|filter| format!("({filter})"))
            .collect::<Vec<_>>()
            .join(" | ");
        format!("({invariant}) & ({user_union})")
    };

    let mut args = ktstr::verifier::build_nextest_args(nextest_profile, &[]);
    let filter_index = args
        .iter()
        .position(|arg| arg == "-E")
        .map(|index| index + 1)
        .expect("build_nextest_args always carries -E <filter>");
    args[filter_index] = filter;

    // Discovery resolved only inferred ktstr gates. Replay the compatible
    // package roots through the same targeted injector ordinary cargo-ktstr
    // test commands use.
    let activations = plan
        .compatible
        .iter()
        .map(|package| PackageFeatureActivation {
            package: package.name.clone(),
            features: package.verifier_features.clone(),
        })
        .collect::<Vec<_>>();
    let rest = inject_feature_activations(rest, &activations);

    // The unscoped one-shot path selects ONLY packages that can enumerate
    // current verifier declarations. This keeps old ktstr (and every unrelated
    // workspace package) out of the compile itself. Existing explicit package
    // selection is preserved rather than widened; the invariant package gate
    // above still prevents an old binary from listing or running cells.
    if has_workspace_selector(&rest) {
        for package in &plan.older {
            args.push("--exclude".to_string());
            args.push(package.name.clone());
        }
    } else if !has_package_selector(&rest) {
        for package in &plan.compatible {
            args.push("-p".to_string());
            args.push(package.name.clone());
        }
    } else if !has_explicit_package_selector(&rest) {
        // A lone --exclude applies to verifier's intentionally widened
        // workspace selection. Cargo requires --workspace alongside it.
        args.push("--workspace".to_string());
        for package in &plan.older {
            args.push("--exclude".to_string());
            args.push(package.name.clone());
        }
    }
    args.extend(rest);
    args
}

fn probe_scheduler_declarations(
    test_bins: &[PathBuf],
) -> Result<Vec<ktstr::test_support::SchedulerListEntry>, String> {
    if test_bins.is_empty() {
        return Ok(Vec::new());
    }
    let per_binary: Vec<Vec<ktstr::test_support::SchedulerListEntry>> =
        match crate::misc::probe_collect_from_bins(
            test_bins,
            |bin| {
                let mut command = Command::new(bin);
                command.arg("--ktstr-list-schedulers");
                command
            },
            |bin, output| {
                serde_json::from_slice(&output.stdout).map_err(|error| {
                    format!(
                        "parse --ktstr-list-schedulers output from {}: {error}",
                        bin.display()
                    )
                })
            },
        ) {
            Ok(entries) => entries,
            // None of the warmed executables linked the scheduler-list ctor.
            // Preserve verifier's established zero-cell diagnosis instead of
            // turning "no declare_scheduler!" into a probe setup failure.
            Err(crate::misc::ProbeError::Miss(_)) => Vec::new(),
            Err(error @ crate::misc::ProbeError::Setup(_)) => {
                return Err(format!(
                    "probe warmed test binaries for declared schedulers: {error:?}"
                ));
            }
        };
    Ok(per_binary.into_iter().flatten().collect())
}

/// Apply the scheduler-name filter and child-emission eligibility before
/// comparing declarations. Every surviving name must have one full
/// [`SchedulerJson`](ktstr::test_support::SchedulerJson) meaning; exact
/// duplicates from multiple warmed test binaries collapse, while any
/// differing field is an ambiguity because all cells share the same
/// `verifier/<scheduler>/...` namespace.
fn selected_emitting_scheduler_declarations(
    declarations: &[ktstr::test_support::SchedulerListEntry],
    scheduler_filter: Option<&str>,
    mut emits: impl FnMut(&ktstr::test_support::SchedulerJson) -> Result<bool, String>,
) -> Result<Vec<ktstr::test_support::SchedulerJson>, String> {
    let mut identities: BTreeMap<String, ktstr::test_support::SchedulerJson> = BTreeMap::new();
    for entry in declarations {
        let scheduler = &entry.scheduler;
        if scheduler_filter.is_some_and(|wanted| scheduler.name != wanted) {
            continue;
        }
        if !emits(scheduler)? {
            continue;
        }
        if let Some(previous) = identities.get(&scheduler.name) {
            if previous != scheduler {
                return Err(format!(
                    "conflicting declarations for scheduler {:?}: first {previous:?}, \
                     later {scheduler:?}; a verifier cell name must identify exactly one \
                     full scheduler declaration",
                    scheduler.name,
                ));
            }
        } else {
            identities.insert(scheduler.name.clone(), scheduler.clone());
        }
    }
    Ok(identities.into_values().collect())
}

fn scheduler_profile_for_run(cli_profile: Option<&str>) -> String {
    match cli_profile {
        Some(profile) if !profile.is_empty() => profile.to_string(),
        Some(_) => "release".to_string(),
        None => ktstr::scheduler_profile_name(),
    }
}

fn declaring_workspace(
    scheduler: &str,
    package: &str,
    manifest_dir: &str,
) -> Result<Option<(PathBuf, PackageId)>, String> {
    let manifest_dir = Path::new(manifest_dir);
    if !manifest_dir.is_dir() {
        return Err(format!(
            "scheduler {:?} declaration manifest directory does not exist or is not a \
             directory: {}",
            scheduler,
            manifest_dir.display(),
        ));
    }
    let mut command = cargo_metadata::MetadataCommand::new();
    command
        .cargo_path("cargo")
        .current_dir(manifest_dir)
        .no_deps();
    let metadata = command.exec().map_err(|error| {
        format!(
            "cargo metadata from scheduler {:?} manifest directory {} failed: {error}",
            scheduler,
            manifest_dir.display(),
        )
    })?;
    let member_ids: HashSet<&PackageId> = metadata.workspace_members.iter().collect();
    let package = metadata
        .packages
        .iter()
        .find(|candidate| member_ids.contains(&candidate.id) && candidate.name.as_str() == package);
    let Some(package) = package else {
        // Match child listing: fixture/nonmember Discover packages emit no
        // cells, so they are absent from both conflict detection and builds.
        return Ok(None);
    };
    let workspace_root =
        std::fs::canonicalize(metadata.workspace_root.as_std_path()).map_err(|error| {
            format!(
                "canonicalize scheduler {:?} workspace root {}: {error}",
                scheduler, metadata.workspace_root,
            )
        })?;
    Ok(Some((workspace_root, package.id.clone())))
}

/// Mirror every parent-visible child-listing gate before conflict detection or
/// scheduler build planning. Workspace metadata resolutions are cached by raw
/// declaring directory + package so exact declarations across test binaries
/// run Cargo metadata once.
fn selected_discover_requests(
    declarations: &[ktstr::test_support::SchedulerListEntry],
    scheduler_filter: Option<&str>,
    resolved_kernels: &[(String, String)],
    presets: &[ktstr::gauntlet::TopoPreset],
) -> Result<Vec<DiscoverSchedulerRequest>, String> {
    let mut workspaces: BTreeMap<(String, String), Option<(PathBuf, PackageId)>> = BTreeMap::new();
    let selected =
        selected_emitting_scheduler_declarations(declarations, scheduler_filter, |scheduler| {
            use ktstr::test_support::BinaryKindJson;

            if scheduler.name.contains('/') {
                return Ok(false);
            }
            if matches!(
                scheduler.binary_kind,
                BinaryKindJson::Eevdf | BinaryKindJson::KernelBuiltin
            ) {
                return Ok(false);
            }
            if !scheduler.has_accepted_verifier_cell(
                resolved_kernels
                    .iter()
                    .map(|(label, sanitized)| (label.as_str(), sanitized.as_str())),
                presets,
            ) {
                return Ok(false);
            }
            match &scheduler.binary_kind {
                BinaryKindJson::Path(path) => Ok(Path::new(path).exists()),
                BinaryKindJson::Discover(package) => {
                    if scheduler.manifest_dir.is_empty() {
                        return Err(format!(
                            "declared scheduler {:?} (package {:?}) did not report its manifest \
                             directory; rebuild its test binary with the current ktstr before \
                             running cargo ktstr verifier",
                            scheduler.name, package,
                        ));
                    }
                    let key = (scheduler.manifest_dir.clone(), package.clone());
                    if !workspaces.contains_key(&key) {
                        let resolution =
                            declaring_workspace(&scheduler.name, package, &scheduler.manifest_dir)?;
                        workspaces.insert(key.clone(), resolution);
                    }
                    Ok(workspaces
                        .get(&key)
                        .expect("workspace resolution inserted above")
                        .is_some())
                }
                BinaryKindJson::Eevdf | BinaryKindJson::KernelBuiltin => Ok(false),
            }
        })?;

    let mut requests = Vec::new();
    for scheduler in selected {
        let ktstr::test_support::BinaryKindJson::Discover(package) = scheduler.binary_kind else {
            continue;
        };
        let key = (scheduler.manifest_dir.clone(), package.clone());
        let (workspace_root, package_id) = workspaces
            .get(&key)
            .cloned()
            .flatten()
            .expect("emitting Discover declaration has a member resolution");
        requests.push(DiscoverSchedulerRequest {
            scheduler: scheduler.name,
            package,
            manifest_dir: scheduler.manifest_dir,
            workspace_root,
            package_id,
        });
    }
    requests.sort_by(|left, right| {
        (&left.scheduler, &left.package, &left.manifest_dir).cmp(&(
            &right.scheduler,
            &right.package,
            &right.manifest_dir,
        ))
    });
    Ok(requests)
}

fn plan_workspace_scheduler_builds(
    requests: &[DiscoverSchedulerRequest],
) -> Result<Vec<WorkspaceSchedulerBuild>, String> {
    let mut groups: BTreeMap<PathBuf, WorkspaceSchedulerBuild> = BTreeMap::new();
    for request in requests {
        let root = request.workspace_root.clone();
        let package_id = request.package_id.clone();
        let group = groups
            .entry(root.clone())
            .or_insert_with(|| WorkspaceSchedulerBuild {
                root,
                packages: BTreeMap::new(),
                requests: Vec::new(),
            });
        if let Some(previous) = group
            .packages
            .insert(request.package.clone(), package_id.clone())
            && previous != package_id
        {
            return Err(format!(
                "workspace {} resolved package {:?} to conflicting Cargo package IDs: \
                 {previous} and {package_id}",
                group.root.display(),
                request.package,
            ));
        }
        group.requests.push(request.clone());
    }
    Ok(groups.into_values().collect())
}

fn map_scheduler_artifacts(
    stdout: &[u8],
    group: &WorkspaceSchedulerBuild,
) -> Result<BTreeMap<String, PathBuf>, String> {
    let expected_by_id: HashMap<&PackageId, &str> = group
        .packages
        .iter()
        .map(|(name, id)| (id, name.as_str()))
        .collect();
    let mut artifacts: BTreeMap<String, PathBuf> = BTreeMap::new();
    for message in cargo_metadata::Message::parse_stream(BufReader::new(stdout)) {
        let message = message.map_err(|error| {
            format!(
                "parse cargo build JSON from workspace {}: {error}",
                group.root.display()
            )
        })?;
        let cargo_metadata::Message::CompilerArtifact(artifact) = message else {
            continue;
        };
        if !artifact.target.is_bin() || artifact.profile.test {
            continue;
        }
        let Some(package) = expected_by_id.get(&artifact.package_id) else {
            continue;
        };
        let Some(executable) = artifact.executable else {
            continue;
        };
        let path = PathBuf::from(executable.as_str());
        if let Some(previous) = artifacts.insert((*package).to_string(), path.clone()) {
            return Err(format!(
                "scheduler package {:?} in workspace {} emitted multiple [[bin]] \
                 executables ({} and {}); Discover requires one unambiguous binary",
                package,
                group.root.display(),
                previous.display(),
                path.display(),
            ));
        }
    }
    for package in group.packages.keys() {
        if !artifacts.contains_key(package) {
            return Err(format!(
                "cargo build succeeded but emitted no non-test [[bin]] executable for \
                 scheduler package {:?} in workspace {}",
                package,
                group.root.display(),
            ));
        }
    }
    Ok(artifacts)
}

fn build_scheduler_workspace(
    group: &WorkspaceSchedulerBuild,
    profile: &str,
) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut command = Command::new("cargo");
    command.current_dir(&group.root).args([
        "build",
        "--message-format=json-render-diagnostics",
        "--profile",
        profile,
    ]);
    for package in group.packages.keys() {
        command.args(["-p", package]);
    }
    let output = crate::run_cargo::run_reserved_build_output(
        command,
        "cargo ktstr verifier",
        "scheduler workspace pre-build",
    )?;
    if !output.status.success() {
        return Err(format!(
            "scheduler prebuild in workspace {} failed ({}) — see Cargo output above",
            group.root.display(),
            output
                .status
                .code()
                .map_or("signal".to_string(), |code| code.to_string()),
        ));
    }
    map_scheduler_artifacts(&output.stdout, group)
}

/// Copy one Cargo-emitted scheduler executable into the parent-owned result
/// directory through an already-open source descriptor.
///
/// Opening first pins the source inode against a concurrent Cargo atomic
/// replacement. The destination is populated under a private temporary name,
/// made read-only executable, synced, and renamed into place atomically; cells
/// never observe a partial file and the manifest never points back into a
/// mutable Cargo target directory.
fn snapshot_scheduler_artifact(
    source_path: &Path,
    result_dir: &Path,
    ordinal: usize,
) -> Result<PathBuf, String> {
    let source = std::fs::File::open(source_path).map_err(|error| {
        format!(
            "open scheduler artifact {} for immutable snapshot: {error}",
            source_path.display()
        )
    })?;
    snapshot_scheduler_artifact_from_open_file(source, source_path, result_dir, ordinal)
}

fn snapshot_scheduler_artifact_from_open_file(
    mut source: std::fs::File,
    source_path: &Path,
    result_dir: &Path,
    ordinal: usize,
) -> Result<PathBuf, String> {
    use std::os::unix::fs::PermissionsExt;

    let source_metadata = source.metadata().map_err(|error| {
        format!(
            "stat opened scheduler artifact {}: {error}",
            source_path.display()
        )
    })?;
    if !source_metadata.is_file() {
        return Err(format!(
            "scheduler artifact is not a file: {}",
            source_path.display()
        ));
    }
    if source_metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!(
            "scheduler artifact is not executable: {}",
            source_path.display()
        ));
    }

    let final_path = result_dir.join(format!("scheduler-executable-{ordinal}"));
    let mut temporary = tempfile::NamedTempFile::new_in(result_dir).map_err(|error| {
        format!(
            "create temporary scheduler snapshot in {}: {error}",
            result_dir.display()
        )
    })?;
    std::io::copy(&mut source, temporary.as_file_mut()).map_err(|error| {
        format!(
            "copy scheduler artifact {} into immutable snapshot: {error}",
            source_path.display()
        )
    })?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("flush scheduler snapshot: {error}"))?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o555))
        .map_err(|error| format!("make scheduler snapshot read-only executable: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("sync scheduler snapshot: {error}"))?;
    temporary.persist(&final_path).map_err(|error| {
        format!(
            "atomically install scheduler snapshot {}: {}",
            final_path.display(),
            error.error,
        )
    })?;
    std::fs::canonicalize(&final_path).map_err(|error| {
        format!(
            "canonicalize scheduler snapshot {}: {error}",
            final_path.display()
        )
    })
}

fn prebuild_scheduler_manifest(
    declarations: &[ktstr::test_support::SchedulerListEntry],
    scheduler_filter: Option<&str>,
    profile: &str,
    resolved_kernels: &[(String, String)],
    presets: &[ktstr::gauntlet::TopoPreset],
    result_dir: &Path,
    interrupt_guard: &crate::interrupt::InterruptGuard,
) -> Result<ktstr::verifier::VerifierSchedulerArtifactManifest, String> {
    if interrupt_guard.interrupted().is_some() {
        return Err("cargo ktstr verifier interrupted before scheduler prebuild".to_string());
    }
    let requests =
        selected_discover_requests(declarations, scheduler_filter, resolved_kernels, presets)?;
    let groups = plan_workspace_scheduler_builds(&requests)?;
    let package_count: usize = groups.iter().map(|group| group.packages.len()).sum();
    if package_count > 0 {
        eprintln!(
            "cargo ktstr verifier: prebuilding {package_count} scheduler package(s) \
             in {} workspace batch(es) with profile {profile:?}",
            groups.len(),
        );
    }
    let mut entries = Vec::with_capacity(requests.len());
    let mut snapshots: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    for group in &groups {
        if interrupt_guard.interrupted().is_some() {
            return Err("cargo ktstr verifier interrupted during scheduler prebuild".to_string());
        }
        let artifacts = build_scheduler_workspace(group, profile)?;
        if interrupt_guard.interrupted().is_some() {
            return Err("cargo ktstr verifier interrupted during scheduler prebuild".to_string());
        }
        for request in &group.requests {
            let emitted = artifacts
                .get(&request.package)
                .expect("artifact completeness checked above");
            let path = std::fs::canonicalize(emitted).map_err(|error| {
                format!(
                    "canonicalize scheduler {:?} artifact {}: {error}",
                    request.scheduler,
                    emitted.display(),
                )
            })?;
            let snapshot = if let Some(snapshot) = snapshots.get(&path) {
                snapshot.clone()
            } else {
                let snapshot = snapshot_scheduler_artifact(&path, result_dir, snapshots.len())?;
                snapshots.insert(path, snapshot.clone());
                snapshot
            };
            entries.push(ktstr::verifier::VerifierSchedulerArtifactEntry {
                scheduler: request.scheduler.clone(),
                package: request.package.clone(),
                manifest_dir: request.manifest_dir.clone(),
                path: snapshot,
            });
        }
    }
    entries.sort_by(|left, right| {
        (
            &left.scheduler,
            &left.package,
            &left.manifest_dir,
            &left.path,
        )
            .cmp(&(
                &right.scheduler,
                &right.package,
                &right.manifest_dir,
                &right.path,
            ))
    });
    Ok(ktstr::verifier::VerifierSchedulerArtifactManifest {
        version: ktstr::verifier::VERIFIER_SCHEDULER_ARTIFACT_MANIFEST_VERSION,
        profile: profile.to_string(),
        entries,
    })
}

fn write_scheduler_manifest(
    result_dir: &Path,
    manifest: &ktstr::verifier::VerifierSchedulerArtifactManifest,
) -> Result<PathBuf, String> {
    use std::os::unix::fs::PermissionsExt;

    let final_path = result_dir.join("scheduler-artifacts-v1.json");
    let mut temporary = tempfile::NamedTempFile::new_in(result_dir).map_err(|error| {
        format!(
            "create temporary scheduler artifact manifest in {}: {error}",
            result_dir.display()
        )
    })?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), manifest)
        .map_err(|error| format!("serialize scheduler artifact manifest: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("sync scheduler artifact manifest: {error}"))?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .map_err(|error| format!("make scheduler artifact manifest read-only: {error}"))?;
    temporary.persist(&final_path).map_err(|error| {
        format!(
            "atomically install scheduler artifact manifest {}: {}",
            final_path.display(),
            error.error,
        )
    })?;
    Ok(final_path)
}

/// Dispatch the `cargo ktstr verifier` subcommand.
///
/// The trailing `args` are forwarded verbatim to the inner
/// `cargo nextest run` (a nextest filterset, `--cargo-profile`, ...).
/// The `declare_scheduler!` verifier cells carry no `required-features`, but
/// declaration targets may themselves be feature-gated. From metadata, the
/// dispatcher follows and package-qualifies ktstr-only feature chains for a
/// direct compatible optional ktstr dependency, so conventional gates need no
/// manual `--features` passthrough or broad `--all-features`.
///
/// `profile` is the scheduler-under-test's cargo BUILD profile
/// (`--profile <NAME>`): set as `KTSTR_SCHEDULER_PROFILE` and used by the
/// parent-owned, workspace-batched scheduler prebuild. Omitted, schedulers
/// build with the `release` default from [`ktstr::scheduler_profile_name`].
/// `nextest_profile` is the
/// NEXTEST test profile (`--nextest-profile <NAME>`), emitted as
/// nextest's own `--profile <NAME>` before the user's trailing args.
pub(crate) fn run_verifier(
    kernel: Vec<String>,
    raw: bool,
    profile: Option<String>,
    nextest_profile: Option<String>,
    scheduler: Option<String>,
    include_eol: bool,
    args: Vec<String>,
) -> Result<(), String> {
    let package_plan = query_verifier_package_plan(&args)?;
    if let Some(package) = package_plan.newer.first() {
        let versions = package
            .versions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("/");
        return Err(format!(
            "package {} uses ktstr {versions}, newer than cargo-ktstr {}; update cargo-ktstr",
            package.name,
            env!("CARGO_PKG_VERSION"),
        ));
    }
    for package in &package_plan.older {
        eprintln!("{}", format_older_package_skip(package));
    }
    // Every verifier-bearing workspace member is old. Excluding those tests is
    // explicitly a non-error, and there is nothing current to ask nextest to
    // build or enumerate.
    if package_plan.compatible.is_empty() && !package_plan.older.is_empty() {
        return Ok(());
    }
    let args = if !package_plan.older.is_empty() && has_non_exact_package_selector(&args) {
        replace_package_selectors(&args, &package_plan.compatible)
    } else {
        drop_older_package_selectors(&args, &package_plan.older)
    };
    let nextest_args = build_scoped_nextest_args(nextest_profile.as_deref(), &args, &package_plan);
    let scheduler_profile = scheduler_profile_for_run(profile.as_deref());

    let mut cmd = Command::new("cargo");
    // The nextest argument vector — base flags (`--run-ignored all`, the
    // load-bearing `--no-tests pass`, and the `verifier/...`-cell filter),
    // then the optional `--nextest-profile` as nextest's `--profile`, then
    // the user's forwarded trailing args verbatim — is built by
    // `ktstr::verifier::build_nextest_args`, which documents each flag
    // and is unit-tested so the reachability-critical ones cannot be
    // silently dropped. The profile is emitted before the forwarded args
    // so a passthrough token cannot shadow it; no `--` separator is needed
    // (the bin's argsplit rewrite routes native flags to ktstr and the
    // passthrough to the `last = true` `args` field before clap parses).
    cmd.args(&nextest_args);

    // Hand the child build cargo-ktstr's embedded busybox / wprof
    // (mirrors run_cargo_sub): `build.rs` watches `KTSTR_BUSYBOX_BIN` /
    // `KTSTR_WPROF_BIN` via rerun-if-env-changed, so setting them here —
    // and identically on the reserved warm-up below — keeps the verifier
    // sweep on the SAME build fingerprint as `cargo ktstr test`, sharing
    // one cached harness build across both subcommands. Kept in
    // `blob_envs` for the warm-up's cache-parity application.
    let blob_envs = crate::run_cargo::prebuilt_blob_bin_envs(
        std::env::var_os(ktstr::KTSTR_BUSYBOX_PATH_ENV),
        std::env::var_os("KTSTR_WPROF_PATH"),
    );
    for (var, val) in &blob_envs {
        cmd.env(var, val);
    }

    if raw {
        cmd.env(ktstr::KTSTR_VERIFIER_RAW_ENV, "1");
    }

    // Export the EFFECTIVE profile, not the raw CLI token. The parent build,
    // immutable manifest, and every child validation therefore share one
    // source of truth. In particular `--profile ""` resolves to release and
    // overrides a non-empty inherited KTSTR_SCHEDULER_PROFILE everywhere.
    cmd.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, &scheduler_profile);

    // `--scheduler <NAME>` restricts the sweep to a single declared
    // scheduler: forwarded via KTSTR_VERIFIER_SCHEDULER so the cell
    // emission (`list_verifier_cells_all`, which runs in the test binary
    // where the `declare_scheduler!` registry is linked) skips every
    // other declared scheduler. Validation is emission-side: this CLI
    // bin does not link that registry, so a name typo surfaces as an
    // empty record set, reported after nextest returns.
    if let Some(s) = &scheduler {
        cmd.env(ktstr::KTSTR_VERIFIER_SCHEDULER_ENV, s);
    }

    // Always produce a non-empty kernel list. When --kernel is
    // omitted, auto-discover one kernel and synthesize a single
    // entry with a path-basename label. The test-binary cell
    // handler keys on this list as its single source of truth.
    let resolved: Vec<(String, PathBuf)> = if !kernel.is_empty() {
        let r = resolve_kernel_set(&kernel, include_eol)?;
        if r.is_empty() {
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        r
    } else {
        let path = resolve_kernel_image(None)?;
        let label = path_kernel_label(&path);
        vec![(label, path)]
    };

    cmd.env(ktstr::KTSTR_KERNEL_ENV, &resolved[0].1);
    let encoded = encode_kernel_list(&resolved)?;
    cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
    // Mark this test invocation as cargo-ktstr-orchestrated so
    // VM-boot tests can skip when run under raw nextest. Mirrors
    // the `cargo ktstr test` dispatcher in run_cargo.rs.
    cmd.env(ktstr::KTSTR_ORCHESTRATED_ENV, "1");

    // Reserve + cgroup-confine the harness COMPILE phase only, exactly as
    // `run_cargo_sub` does for `cargo ktstr test` (see the block comment
    // there): `cargo nextest run` builds then runs in one process, so an
    // explicit `--no-run` warm-up compiles every test binary under a
    // machine-global LLC LOCK_SH + cpuset cgroup (Consolidate placement —
    // a compile is throughput-elastic; packing leaves whole LLCs free for
    // exclusive perf-mode reservations), then releases BOTH before the
    // combined run below, whose cells take their own reservations. The
    // scheduler binaries are then prebuilt once per workspace batch under
    // the same compile-reservation semantics and passed to cells through an
    // immutable manifest. The verifier sweep is the
    // primary colocated-CI workload, so this is the path where an
    // unreserved harness compile would invade a peer runner's perf-mode
    // reservation. Cache parity with the combined run: identical nextest
    // argv (+`--no-run`), same `blob_envs`, and `KTSTR_KERNEL` (the only
    // kernel-resolution env `build.rs` fingerprints via
    // rerun-if-env-changed — KTSTR_KERNEL_LIST and the KTSTR_VERIFIER_*
    // vars are runtime-only). No BTF-anchor `BPF_EXTRA_CFLAGS_PRE_INCL`
    // handling: the verifier dispatcher injects none, so warm-up and run
    // inherit the identical process env.
    //
    // Install one outer signal guard BEFORE creating any run-owned state or
    // spawning any compile/probe child. Every `?` below exits this closure,
    // dropping `result_dir` before the guard is removed and the caught signal
    // is re-raised as 130/143. SIGKILL cannot run Rust cleanup; the existing
    // dead-pid sweep remains the next-run recovery for that case.
    let interrupt_guard = crate::interrupt::InterruptGuard::install();
    let guarded_result = (|| -> Result<Option<i32>, String> {
        // Per-cell result dir: each verifier cell writes its PASS/FAIL record
        // here (via KTSTR_VERIFIER_RESULT_DIR), and after nextest returns we
        // read them back to render the summary table. Creating it before the
        // warm-up makes the same RAII owner cover harness compilation,
        // declaration probes, scheduler prebuild/snapshots, manifest write,
        // and the final nextest run.
        let result_dir = VerifierResultDir::create(&std::env::temp_dir())?;
        if interrupt_guard.interrupted().is_some() {
            return Ok(None);
        }

        let mut warm = Command::new("cargo");
        warm.args(crate::run_cargo::prebuild_no_run_json_args(&nextest_args));
        for (var, val) in &blob_envs {
            warm.env(var, val);
        }
        warm.env(ktstr::KTSTR_KERNEL_ENV, &resolved[0].1);
        let test_bins = crate::run_cargo::run_reserved_prebuild_collect_test_bins(
            warm,
            "cargo ktstr verifier",
        )?;
        if interrupt_guard.interrupted().is_some() {
            return Ok(None);
        }
        let declarations = probe_scheduler_declarations(&test_bins)?;
        if interrupt_guard.interrupted().is_some() {
            return Ok(None);
        }
        cmd.env(ktstr::KTSTR_VERIFIER_RESULT_DIR_ENV, result_dir.path());
        let resolved_kernel_labels = resolved
            .iter()
            .map(|(label, _)| {
                (
                    label.clone(),
                    ktstr::test_support::sanitize_kernel_label(label),
                )
            })
            .collect::<Vec<_>>();
        let presets = ktstr::gauntlet::gauntlet_presets();
        let scheduler_manifest = prebuild_scheduler_manifest(
            &declarations,
            scheduler.as_deref(),
            &scheduler_profile,
            &resolved_kernel_labels,
            &presets,
            result_dir.path(),
            &interrupt_guard,
        )?;
        if interrupt_guard.interrupted().is_some() {
            return Ok(None);
        }
        let scheduler_manifest_path =
            write_scheduler_manifest(result_dir.path(), &scheduler_manifest)?;
        if interrupt_guard.interrupted().is_some() {
            return Ok(None);
        }
        cmd.env(
            ktstr::KTSTR_VERIFIER_SCHEDULER_MANIFEST_ENV,
            scheduler_manifest_path,
        );

        let kernel_count = resolved.len();

        eprintln!(
            "cargo ktstr verifier: dispatching to nextest (verifier/ cells only) \
         on {kernel_count} resolved kernel(s){raw}{fwd}",
            raw = if raw { " (raw output)" } else { "" },
            fwd = if args.is_empty() {
                String::new()
            } else {
                format!(" forwarding to nextest: {}", args.join(" "))
            },
        );

        // The outer guard survives Ctrl-C / SIGTERM so the result-dir cleanup
        // below runs; nextest tears down its own test children.
        if interrupt_guard.interrupted().is_some() {
            return Ok(None);
        }
        let status = cmd
            .status()
            .map_err(|e| format!("spawn cargo nextest run: {e}"))?;

        // From the records each cell wrote into `result_dir`: print the
        // per-scheduler verified_insns tables first, then the per-scheduler
        // topology × kernel PASS/FAIL grids LAST so the operator's final view
        // is the pass/fail matrix. Both print on success AND failure so
        // failing cells stay visible. Best-effort: no records (e.g. 0 cells
        // ran) -> the renderers return None and nothing prints.
        let records = ktstr::verifier::read_cell_records(result_dir.path());
        if let Some(tables) = ktstr::verifier::render_instruction_count_tables(&records) {
            print!("{tables}");
        }
        if let Some(table) = ktstr::verifier::render_result_table(&records) {
            print!("{table}");
        }
        // Decide the outcome from nextest's exit + the records. With
        // `--no-tests pass` a zero-cell selection exits 0, so an empty record
        // set on success is diagnosed here (a `--scheduler` typo, no scheduler
        // declared, or no topology preset fits this host) rather than
        // surfacing nextest's generic no-tests error. A real build/exec
        // failure still exits non-zero and is surfaced verbatim — EXCEPT when
        // the failure is a real cell failure the grid above already shows
        // (SilentExit): there the process exits with nextest's code but emits
        // no stderr error line, which would otherwise interleave into the
        // stdout report under CI's unordered pipes.
        match ktstr::verifier::classify_run_outcome(
            status.success(),
            records.is_empty(),
            records.iter().any(|r| !r.passed && !r.skipped),
            scheduler.as_deref(),
            status.code(),
        ) {
            ktstr::verifier::RunOutcome::Success => Ok(None),
            ktstr::verifier::RunOutcome::Failed(msg) => Err(msg),
            // Report + cleanup already ran above. Defer the silent exit until the
            // outer signal guard has restored the prior dispositions.
            ktstr::verifier::RunOutcome::SilentExit(code) => Ok(Some(code)),
        }
    })();

    // `guarded_result` owns no result-dir guard here: every success/error path
    // dropped it inside the closure. Restore the prior signal dispositions,
    // then preserve the first caught SIGINT/SIGTERM as the process outcome.
    let caught = crate::interrupt::restore_and_caught(interrupt_guard);
    if let Some(sig) = caught {
        crate::interrupt::reraise(sig);
    }
    match guarded_result {
        Ok(Some(code)) => std::process::exit(code),
        Ok(None) => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn discover_declaration(
        scheduler: &str,
        package: &str,
        manifest_dir: &str,
    ) -> ktstr::test_support::SchedulerListEntry {
        let mut scheduler_json = ktstr::test_support::SchedulerJson::from_scheduler(
            &ktstr::test_support::Scheduler::EEVDF,
        );
        scheduler_json.name = scheduler.to_string();
        scheduler_json.manifest_dir = manifest_dir.to_string();
        scheduler_json.binary_kind =
            ktstr::test_support::BinaryKindJson::Discover(package.to_string());
        ktstr::test_support::SchedulerListEntry {
            scheduler: scheduler_json,
            test_count: 1,
        }
    }

    #[test]
    fn emitting_declarations_compare_full_scheduler_json_semantics() {
        let a = discover_declaration("a", "scx_a", "/w/a");
        let b = discover_declaration("b", "scx_b", "/w/b");
        let mut duplicate_with_other_test_count = a.clone();
        duplicate_with_other_test_count.test_count = 999;
        let selected = selected_emitting_scheduler_declarations(
            &[a.clone(), duplicate_with_other_test_count, b],
            Some("a"),
            |_| Ok(true),
        )
        .expect("selected declarations");
        assert_eq!(
            selected,
            vec![a.scheduler.clone()],
            "exact duplicates collapse after the name filter",
        );

        let mut variants = Vec::new();
        let mut changed = a.scheduler.clone();
        changed.binary_kind = ktstr::test_support::BinaryKindJson::Discover("scx_other".into());
        variants.push(("binary", changed));
        let mut changed = a.scheduler.clone();
        changed.manifest_dir = "/w/other".into();
        variants.push(("manifest_dir", changed));
        let mut changed = a.scheduler.clone();
        changed.sched_args.push("--slice-us=5000".into());
        variants.push(("scheduler args", changed));
        let mut changed = a.scheduler.clone();
        changed.kernels.push("6.18".into());
        variants.push(("kernels", changed));
        let mut changed = a.scheduler.clone();
        changed.topology.cores_per_llc += 1;
        variants.push(("topology", changed));
        let mut changed = a.scheduler.clone();
        changed.constraints.min_cpus += 1;
        variants.push(("constraints", changed));
        let mut changed = a.scheduler.clone();
        changed
            .verifier_exclude_topologies
            .push("4cpu-1llc-nosmt".into());
        variants.push(("verifier exclusions", changed));

        for (field, conflicting) in variants {
            let error = selected_emitting_scheduler_declarations(
                &[
                    a.clone(),
                    ktstr::test_support::SchedulerListEntry {
                        scheduler: conflicting,
                        test_count: 99,
                    },
                ],
                None,
                |_| Ok(true),
            )
            .expect_err(field);
            assert!(
                error.contains("conflicting declarations for scheduler \"a\""),
                "{field} must be part of scheduler identity: {error}",
            );
        }
    }

    fn verifier_matrix() -> (Vec<(String, String)>, Vec<ktstr::gauntlet::TopoPreset>) {
        (
            vec![("6.14.2".into(), "kernel_6_14_2".into())],
            ktstr::gauntlet::gauntlet_presets(),
        )
    }

    #[test]
    fn discover_planning_skips_same_name_nonmember_before_conflict() {
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let member = workspace.path().join("member");
        std::fs::create_dir_all(member.join("src")).expect("create member source dir");
        std::fs::write(
            workspace.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"member\"]\nresolver = \"2\"\n",
        )
        .expect("write workspace manifest");
        std::fs::write(
            member.join("Cargo.toml"),
            "[package]\nname = \"valid_sched\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .expect("write member manifest");
        std::fs::write(member.join("src/main.rs"), "fn main() {}\n").expect("write member main");
        let manifest_dir = member.to_string_lossy().into_owned();
        let valid = discover_declaration("sched", "valid_sched", &manifest_dir);
        let fixture = discover_declaration("sched", "fixture_only", &manifest_dir);
        let (kernels, presets) = verifier_matrix();

        let selected = selected_discover_requests(&[fixture, valid], None, &kernels, &presets)
            .expect("nonmember declaration is an emission-side skip");

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].scheduler, "sched");
        assert_eq!(selected[0].package, "valid_sched");
        assert_eq!(
            selected[0].workspace_root,
            std::fs::canonicalize(workspace.path()).expect("canonical workspace"),
        );
    }

    #[test]
    fn every_nonemitting_declaration_is_filtered_before_conflict_or_build() {
        let (kernels, presets) = verifier_matrix();
        let slash = discover_declaration("ignored/name", "never_queried", "/missing");
        let mut missing_path = slash.clone();
        missing_path.scheduler.name = "ignored".into();
        missing_path.scheduler.binary_kind =
            ktstr::test_support::BinaryKindJson::Path("/definitely/missing/ktstr-sched".into());
        let mut eevdf = missing_path.clone();
        eevdf.scheduler.binary_kind = ktstr::test_support::BinaryKindJson::Eevdf;
        let mut builtin = missing_path.clone();
        builtin.scheduler.binary_kind = ktstr::test_support::BinaryKindJson::KernelBuiltin;
        let mut no_kernel = discover_declaration("ignored", "never_queried", "/missing");
        no_kernel.scheduler.kernels = vec!["9.99".into()];
        let mut no_preset = discover_declaration("ignored", "never_queried", "/missing");
        no_preset.scheduler.constraints.min_cpus = u32::MAX;

        let selected = selected_discover_requests(
            &[slash, missing_path, eevdf, builtin, no_kernel, no_preset],
            None,
            &kernels,
            &presets,
        )
        .expect("non-emitting declarations do not conflict");
        assert!(selected.is_empty());
    }

    #[test]
    fn scheduler_snapshot_uses_pinned_source_and_is_read_only_executable() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("cargo-target-scheduler");
        std::fs::write(&source, b"old scheduler bytes").expect("write source");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o755))
            .expect("chmod source");
        let pinned = std::fs::File::open(&source).expect("pin source fd");

        let replacement = dir.path().join("replacement");
        std::fs::write(&replacement, b"new scheduler bytes").expect("write replacement");
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o755))
            .expect("chmod replacement");
        std::fs::rename(&replacement, &source).expect("replace Cargo target path atomically");

        let result_dir = dir.path().join("results");
        std::fs::create_dir(&result_dir).expect("create result dir");
        let snapshot = snapshot_scheduler_artifact_from_open_file(pinned, &source, &result_dir, 0)
            .expect("snapshot pinned scheduler");

        assert_eq!(
            std::fs::read(&snapshot).expect("read snapshot"),
            b"old scheduler bytes",
            "the copy follows the pinned source inode, not a replaced target path",
        );
        let mode = std::fs::metadata(&snapshot)
            .expect("snapshot metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o222, 0, "snapshot must be read-only");
        assert_eq!(mode & 0o111, 0o111, "snapshot must remain executable");
        assert_eq!(
            snapshot.parent(),
            Some(
                std::fs::canonicalize(&result_dir)
                    .expect("canonical result dir")
                    .as_path()
            ),
            "manifestable path lives in the parent result directory",
        );
    }

    #[test]
    fn verifier_result_dir_guard_cleans_error_paths() {
        let root = tempfile::tempdir().expect("temp root");
        let path = {
            let guard = VerifierResultDir::create(root.path()).expect("create result dir");
            let path = guard.path().to_path_buf();
            std::fs::write(path.join("partial"), b"partial prebuild").expect("write partial file");
            path
        };
        assert!(!path.exists(), "Drop removes a partially prepared run");
    }

    #[test]
    fn outer_interrupt_guard_preserves_first_signal_and_cleans_once() {
        let _serial = crate::interrupt::test_serial_guard();
        let root = tempfile::tempdir().expect("temp root");
        let interrupt_guard = crate::interrupt::InterruptGuard::install();
        let cleanup_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let path = {
            let mut result_dir =
                VerifierResultDir::create(root.path()).expect("create guarded result dir");
            result_dir.cleanup_count = Some(cleanup_count.clone());
            let path = result_dir.path().to_path_buf();
            std::fs::write(path.join("partial-snapshot"), b"partial")
                .expect("write partial snapshot");
            crate::interrupt::record_for_test(libc::SIGTERM);
            crate::interrupt::record_for_test(libc::SIGINT);
            assert_eq!(interrupt_guard.interrupted(), Some(libc::SIGTERM));
            path
        };

        assert!(
            !path.exists(),
            "result-dir RAII runs while the outer signal guard is still live",
        );
        assert_eq!(
            cleanup_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the signal path owns exactly one result-dir cleanup",
        );
        assert_eq!(interrupt_guard.interrupted(), Some(libc::SIGTERM));
        drop(interrupt_guard);
        assert_eq!(
            crate::interrupt::caught(),
            Some(libc::SIGTERM),
            "the first signal remains readable across handler restoration",
        );
    }

    #[test]
    fn verifier_effective_scheduler_profile_normalizes_cli_empty() {
        assert_eq!(scheduler_profile_for_run(Some("")), "release");
        assert_eq!(scheduler_profile_for_run(Some("ci-release")), "ci-release");
    }

    #[test]
    fn scheduler_artifact_json_maps_by_package_id_not_target_name() {
        let package_id: PackageId = serde_json::from_str(r#""scx_a 1.0.0 (path+file:///w/scx_a)""#)
            .expect("PackageId fixture");
        let group = WorkspaceSchedulerBuild {
            root: PathBuf::from("/w"),
            packages: BTreeMap::from([("scx_a".to_string(), package_id)]),
            requests: Vec::new(),
        };
        let stream = concat!(
            r#"{"reason":"compiler-artifact","package_id":"scx_a 1.0.0 (path+file:///w/scx_a)","target":{"name":"different_bin_name","kind":["bin"],"src_path":"/w/scx_a/src/main.rs"},"profile":{"opt_level":"3","debug_assertions":false,"overflow_checks":false,"test":false},"features":[],"filenames":["/w/target/release/different_bin_name"],"executable":"/w/target/release/different_bin_name","fresh":true}"#,
            "\n",
            r#"{"reason":"build-finished","success":true}"#,
            "\n",
        );
        let artifacts =
            map_scheduler_artifacts(stream.as_bytes(), &group).expect("map Cargo artifacts");
        assert_eq!(
            artifacts.get("scx_a"),
            Some(&PathBuf::from("/w/target/release/different_bin_name")),
        );
    }

    fn package_json_with(
        name: &str,
        version: &str,
        id: &str,
        dependencies: &str,
        features: &str,
    ) -> String {
        format!(
            r#"{{"name":"{name}","version":"{version}","id":"{id}","source":null,"description":null,"dependencies":{dependencies},"license":null,"license_file":null,"targets":[],"features":{features},"manifest_path":"/w/{name}/Cargo.toml","readme":null,"repository":null,"homepage":null,"documentation":null,"links":null,"publish":null,"default_run":null}}"#
        )
    }

    fn package_json(name: &str, version: &str, id: &str) -> String {
        package_json_with(name, version, id, "[]", "{}")
    }

    fn optional_ktstr_package_json(
        name: &str,
        version: &str,
        id: &str,
        ktstr_req: &str,
        alias: Option<&str>,
        feature: &str,
    ) -> String {
        let dependency_alias = alias.unwrap_or("ktstr");
        let features = format!(r#"{{"{feature}":["dep:{dependency_alias}"],"unrelated-mode":[]}}"#);
        optional_ktstr_package_with_features_json(
            name, version, id, ktstr_req, alias, "null", &features,
        )
    }

    fn optional_ktstr_package_with_features_json(
        name: &str,
        version: &str,
        id: &str,
        ktstr_req: &str,
        alias: Option<&str>,
        kind: &str,
        features: &str,
    ) -> String {
        let rename = alias.map_or_else(|| "null".to_string(), |alias| format!(r#""{alias}""#));
        let dependencies = format!(
            r#"[{{"name":"ktstr","source":null,"req":"{ktstr_req}","kind":{kind},"rename":{rename},"optional":true,"uses_default_features":true,"features":[],"target":null,"registry":null,"path":null}}]"#
        );
        package_json_with(name, version, id, &dependencies, features)
    }

    fn scx_version_fixture() -> Metadata {
        let layered = "scx_layered 1.0.0 (path+file:///w/scx_layered)";
        let lavd = "scx_lavd 1.0.0 (path+file:///w/scx_lavd)";
        let mitosis = "scx_mitosis 1.0.0 (path+file:///w/scx_mitosis)";
        let unrelated = "unrelated 1.0.0 (path+file:///w/unrelated)";
        let helper = "helper 1.0.0 (path+file:///w/helper)";
        let current = "ktstr 0.41.0 (path+file:///checkout/ktstr)";
        let old = "ktstr 0.18.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let json = format!(
            r#"{{
              "packages":[{layered_pkg},{lavd_pkg},{mitosis_pkg},{unrelated_pkg},{helper_pkg},{current_pkg},{old_pkg}],
              "workspace_members":["{layered}","{lavd}","{mitosis}","{unrelated}"],
              "resolve":{{
                "root":null,
                "nodes":[
                  {{"id":"{layered}","deps":[{{"name":"ktstr","pkg":"{current}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{current}"],"features":["ktstr-tests"]}},
                  {{"id":"{lavd}","deps":[{{"name":"helper","pkg":"{helper}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{helper}"],"features":["ktstr-tests"]}},
                  {{"id":"{mitosis}","deps":[{{"name":"ktstr","pkg":"{old}","dep_kinds":[{{"kind":"dev","target":null}}]}}],"dependencies":["{old}"],"features":["ktstr-tests"]}},
                  {{"id":"{unrelated}","deps":[{{"name":"ktstr","pkg":"{old}","dep_kinds":[{{"kind":"build","target":null}}]}}],"dependencies":["{old}"],"features":[]}},
                  {{"id":"{helper}","deps":[{{"name":"ktstr","pkg":"{current}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{current}"],"features":[]}},
                  {{"id":"{current}","deps":[],"dependencies":[],"features":[]}},
                  {{"id":"{old}","deps":[],"dependencies":[],"features":[]}}
                ]
              }},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            layered_pkg = optional_ktstr_package_json(
                "scx_layered",
                "1.0.0",
                layered,
                "=0.41.0",
                None,
                "ktstr-tests",
            ),
            lavd_pkg = package_json("scx_lavd", "1.0.0", lavd),
            mitosis_pkg = package_json("scx_mitosis", "1.0.0", mitosis),
            unrelated_pkg = package_json("unrelated", "1.0.0", unrelated),
            helper_pkg = package_json("helper", "1.0.0", helper),
            current_pkg = package_json("ktstr", "0.41.0", current),
            old_pkg = package_json("ktstr", "0.18.0", old),
        );
        serde_json::from_str(&json).expect("cargo metadata fixture deserializes")
    }

    #[test]
    fn package_plan_keeps_current_and_skips_old_test_closures() {
        let plan =
            verifier_package_plan(&scx_version_fixture(), &Version::parse("0.41.0").unwrap())
                .expect("fixture has no newer ktstr");
        assert_eq!(
            plan.compatible,
            vec![
                CompatibleVerifierPackage {
                    name: "scx_lavd".to_string(),
                    verifier_features: Vec::new(),
                },
                CompatibleVerifierPackage {
                    name: "scx_layered".to_string(),
                    verifier_features: vec!["ktstr-tests".to_string()],
                },
            ],
            "direct and transitive current ktstr links are both eligible",
        );
        assert_eq!(
            plan.older,
            vec![OlderVerifierPackage {
                name: "scx_mitosis".to_string(),
                versions: vec![Version::parse("0.18.0").unwrap()],
            }],
            "the old dev edge is excluded; the unrelated build-only edge is ignored",
        );
    }

    #[test]
    fn inferred_verifier_features_follow_renamed_optional_ktstr_dependency() {
        let json = optional_ktstr_package_json(
            "scheduler",
            "1.0.0",
            "scheduler 1.0.0 (path+file:///w/scheduler)",
            "=0.41.0",
            Some("test-harness"),
            "verify-schedulers",
        );
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("package fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots(
                &package,
                VersionScope::Matches(&Version::parse("0.41.0").unwrap()),
            ),
            vec!["verify-schedulers"],
            "derive the declaring package's feature name from dep:<renamed-alias>, \
             without admitting its unrelated feature",
        );
    }

    #[test]
    fn inferred_verifier_features_follow_pure_aliases_not_composite_modes() {
        let json = optional_ktstr_package_with_features_json(
            "scheduler",
            "1.0.0",
            "scheduler 1.0.0 (path+file:///w/scheduler)",
            "=0.41.0",
            Some("test-harness"),
            "null",
            r#"{
                "verify":["dep:test-harness"],
                "ktstr-tests":["verify"],
                "gpu":[],
                "everything":["ktstr-tests","gpu"],
                "weak-only":["test-harness?/test-support"]
            }"#,
        );
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("package fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots(
                &package,
                VersionScope::Matches(&Version::parse("0.41.0").unwrap()),
            ),
            vec!["ktstr-tests"],
            "select the outer alias that activates only ktstr; reject composite and weak-only modes",
        );
    }

    #[test]
    fn inferred_verifier_features_handle_forwarding_and_cyclic_aliases() {
        let json = optional_ktstr_package_with_features_json(
            "scheduler",
            "1.0.0",
            "scheduler 1.0.0 (path+file:///w/scheduler)",
            "=0.41.0",
            Some("test-harness"),
            "null",
            r#"{
                "cycle-a":["cycle-b"],
                "cycle-b":["cycle-a","test-harness/test-support"]
            }"#,
        );
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("package fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots(
                &package,
                VersionScope::Matches(&Version::parse("0.41.0").unwrap()),
            ),
            vec!["cycle-a"],
            "dependency-feature forwarding activates ktstr, and one cyclic root is sufficient",
        );
    }

    #[test]
    fn inferred_verifier_features_reject_bare_alias_and_build_dependency() {
        let bare_alias = optional_ktstr_package_with_features_json(
            "scheduler",
            "1.0.0",
            "scheduler 1.0.0 (path+file:///w/scheduler)",
            "=0.41.0",
            Some("test-harness"),
            "null",
            r#"{"test-harness":["marker"],"marker":[],"verify":["test-harness"]}"#,
        );
        let build_dependency = optional_ktstr_package_with_features_json(
            "scheduler",
            "1.0.0",
            "scheduler 1.0.0 (path+file:///w/scheduler)",
            "=0.41.0",
            None,
            r#""build""#,
            r#"{"ktstr-tests":["dep:ktstr"]}"#,
        );
        for json in [bare_alias, build_dependency] {
            let package: cargo_metadata::Package =
                serde_json::from_str(&json).expect("package fixture deserializes");
            assert!(
                infer_ktstr_feature_roots(
                    &package,
                    VersionScope::Matches(&Version::parse("0.41.0").unwrap()),
                )
                .is_empty(),
                "a local feature name and a build-only dependency are not test-link activations",
            );
        }
    }

    #[test]
    fn package_plan_records_newer_ktstr_for_scope_aware_error() {
        let plan =
            verifier_package_plan(&scx_version_fixture(), &Version::parse("0.17.0").unwrap())
                .expect("fixture has a resolve graph");
        assert_eq!(
            plan.newer
                .iter()
                .map(|package| package.name.as_str())
                .collect::<Vec<_>>(),
            vec!["scx_lavd", "scx_layered", "scx_mitosis"],
        );
    }

    fn scoped_plan() -> VerifierPackagePlan {
        VerifierPackagePlan {
            compatible: vec![
                CompatibleVerifierPackage {
                    name: "scx_cosmos".to_string(),
                    verifier_features: vec!["ktstr-tests".to_string()],
                },
                CompatibleVerifierPackage {
                    name: "scx_lavd".to_string(),
                    verifier_features: vec!["ktstr-tests".to_string()],
                },
                CompatibleVerifierPackage {
                    name: "scx_layered".to_string(),
                    verifier_features: vec!["ktstr-tests".to_string()],
                },
            ],
            older: vec![OlderVerifierPackage {
                name: "scx_mitosis".to_string(),
                versions: vec![Version::parse("0.18.0").unwrap()],
            }],
            newer: Vec::new(),
        }
    }

    #[test]
    fn unscoped_verifier_selects_current_packages_with_targeted_features() {
        // Bare `cargo ktstr verifier` is the scx regression shape: its
        // declarations live in `#![cfg(feature = "ktstr-tests")]` test
        // targets. A manifest pass discovers each ktstr-only gate, then both
        // dependency resolution and nextest activate only the
        // package-qualified feature that links each declaration target.
        let args = build_scoped_nextest_args(None, &[], &scoped_plan());
        assert!(
            args.windows(2).any(|pair| pair == ["-p", "scx_cosmos"]),
            "{args:?}",
        );
        assert!(
            args.windows(2).any(|pair| pair == ["-p", "scx_layered"]),
            "{args:?}",
        );
        assert!(
            args.windows(2).any(|pair| pair == ["-p", "scx_lavd"]),
            "{args:?}",
        );
        assert!(!args.iter().any(|arg| arg == "scx_mitosis"), "{args:?}");
        assert!(!args.iter().any(|arg| arg == "--all-features"));
        assert!(args.windows(2).any(|pair| {
            pair == [
                "--features",
                "scx_cosmos/ktstr-tests,scx_lavd/ktstr-tests,scx_layered/ktstr-tests",
            ]
        }));
        assert!(
            !args
                .iter()
                .any(|arg| arg.contains("scx_mitosis/ktstr-tests")),
            "old-package features must stay disabled: {args:?}",
        );
        assert!(args.iter().any(|arg| {
            arg == "(test(/^verifier/) & !test(/^verifier::/)) & \
                    (package(=scx_cosmos) | package(=scx_lavd) | package(=scx_layered))"
        }));
    }

    #[test]
    fn explicit_package_selection_is_not_widened() {
        let forward = strings(&["-p", "scx_layered"]);
        let plan = restrict_plan_to_explicit_selection(scoped_plan(), &forward);
        let args = build_scoped_nextest_args(None, &forward, &plan);
        let selected = args
            .windows(2)
            .filter(|pair| pair[0] == "-p")
            .map(|pair| pair[1].as_str())
            .collect::<Vec<_>>();
        assert_eq!(selected, vec!["scx_layered"]);
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--features", "scx_layered/ktstr-tests"]),
            "only the explicitly selected package's verifier feature is enabled: {args:?}",
        );
    }

    #[test]
    fn explicit_old_package_becomes_non_error_empty_plan() {
        let selected =
            restrict_plan_to_explicit_selection(scoped_plan(), &strings(&["-p", "scx_mitosis"]));
        assert!(selected.compatible.is_empty());
        assert_eq!(selected.older[0].name, "scx_mitosis");
        assert!(selected.newer.is_empty());
    }

    #[test]
    fn mixed_explicit_selection_drops_old_cargo_package() {
        let args = strings(&[
            "-p",
            "scx_layered",
            "--package=scx_mitosis@1.1.2",
            "--features",
            "ktstr-tests",
        ]);
        let selected = restrict_plan_to_explicit_selection(scoped_plan(), &args);
        let rewritten = drop_older_package_selectors(&args, &selected.older);
        assert_eq!(
            rewritten,
            strings(&["-p", "scx_layered", "--features", "ktstr-tests"]),
        );
    }

    #[test]
    fn mixed_glob_selection_rewrites_to_exact_compatible_packages() {
        let plan = scoped_plan();
        let args = strings(&[
            "-p",
            "scx_*",
            "--all-features",
            "--",
            "--package",
            "scx_mitosis",
        ]);
        assert!(has_non_exact_package_selector(&args));
        let rewritten = replace_package_selectors(&args, &plan.compatible);
        assert!(!rewritten.iter().any(|argument| argument == "scx_*"));
        for package in ["scx_cosmos", "scx_lavd", "scx_layered"] {
            assert!(
                rewritten.windows(2).any(|pair| pair == ["-p", package]),
                "{rewritten:?}",
            );
        }
        assert!(
            !rewritten[..rewritten.iter().position(|arg| arg == "--").unwrap()]
                .iter()
                .any(|argument| argument == "scx_mitosis")
        );
        assert_eq!(
            &rewritten[rewritten.iter().position(|arg| arg == "--").unwrap()..],
            strings(&["--", "--package", "scx_mitosis"]),
            "test-binary arguments remain untouched",
        );
    }

    #[test]
    fn old_package_pruning_preserves_test_binary_args_after_separator() {
        let args = strings(&[
            "-p",
            "scx_mitosis",
            "-p",
            "scx_layered",
            "--",
            "--package",
            "scx_mitosis",
            "-pscx_mitosis",
            "--package=scx_mitosis",
            "payload",
        ]);
        let selected = restrict_plan_to_explicit_selection(scoped_plan(), &args);
        assert_eq!(
            drop_older_package_selectors(&args, &selected.older),
            strings(&[
                "-p",
                "scx_layered",
                "--",
                "--package",
                "scx_mitosis",
                "-pscx_mitosis",
                "--package=scx_mitosis",
                "payload",
            ]),
            "only Cargo-side selectors are pruned; test-binary argv is opaque",
        );
    }

    #[test]
    fn package_spec_name_handles_exact_version_and_full_id() {
        assert_eq!(package_spec_name("scx_layered"), Some("scx_layered"));
        assert_eq!(package_spec_name("scx_layered@1.1.2"), Some("scx_layered"),);
        assert_eq!(
            package_spec_name("path+file:///w#scx_layered@1.1.2"),
            Some("scx_layered"),
        );
        assert_eq!(package_spec_name("scx_*"), None);
    }

    #[test]
    fn explicit_current_scope_ignores_unrelated_newer_package() {
        let mut plan = scoped_plan();
        plan.newer.push(NewerVerifierPackage {
            name: "future_tests".to_string(),
            versions: vec![Version::parse("0.42.0").unwrap()],
        });
        let selected = restrict_plan_to_explicit_selection(plan, &strings(&["-p", "scx_layered"]));
        assert_eq!(
            selected.compatible,
            vec![CompatibleVerifierPackage {
                name: "scx_layered".to_string(),
                verifier_features: vec!["ktstr-tests".to_string()],
            }],
        );
        assert!(selected.older.is_empty());
        assert!(selected.newer.is_empty());
    }

    #[test]
    fn workspace_exclude_removes_old_and_newer_from_plan() {
        let mut plan = scoped_plan();
        plan.newer.push(NewerVerifierPackage {
            name: "future_tests".to_string(),
            versions: vec![Version::parse("0.42.0").unwrap()],
        });
        let selected = restrict_plan_to_explicit_selection(
            plan,
            &strings(&[
                "--workspace",
                "--exclude",
                "scx_mitosis",
                "--exclude=future_tests@1.0.0",
            ]),
        );
        assert!(selected.older.is_empty());
        assert!(selected.newer.is_empty());
        assert_eq!(selected.compatible.len(), 3);
    }

    #[test]
    fn workspace_selection_excludes_old_package_from_compile() {
        let args = build_scoped_nextest_args(None, &strings(&["--workspace"]), &scoped_plan());
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--exclude", "scx_mitosis"]),
            "{args:?}",
        );
        assert!(args.windows(2).any(|pair| {
            pair == [
                "--features",
                "scx_cosmos/ktstr-tests,scx_lavd/ktstr-tests,scx_layered/ktstr-tests",
            ]
        }));
        assert!(
            !args
                .iter()
                .any(|arg| arg.contains("scx_mitosis/ktstr-tests")),
            "workspace mode must not activate the excluded old package: {args:?}",
        );
    }

    #[test]
    fn lone_exclude_gets_required_workspace_scope() {
        let forward = strings(&["--exclude", "scx_cosmos"]);
        let plan = restrict_plan_to_explicit_selection(scoped_plan(), &forward);
        let args = build_scoped_nextest_args(None, &forward, &plan);
        assert!(args.iter().any(|argument| argument == "--workspace"));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--exclude", "scx_mitosis"]),
            "older verifier packages remain outside the widened compile: {args:?}",
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--exclude", "scx_cosmos"]),
            "the user's exclusion remains forwarded: {args:?}",
        );
    }

    #[test]
    fn explicit_all_features_is_preserved_without_inferred_feature_flag() {
        let args = build_scoped_nextest_args(None, &strings(&["--all-features"]), &scoped_plan());
        assert_eq!(
            args.iter().filter(|arg| *arg == "--all-features").count(),
            1,
        );
        assert!(
            !args.iter().any(|arg| arg == "--features"),
            "the user's broad feature choice already subsumes inferred roots: {args:?}",
        );
    }

    #[test]
    fn user_filtersets_are_intersected_with_package_gate() {
        let args = build_scoped_nextest_args(
            None,
            &strings(&[
                "-p",
                "scx_layered",
                "-E",
                "test(large)",
                "--filterset=test(kernel_gke)",
            ]),
            &scoped_plan(),
        );
        assert_eq!(args.iter().filter(|arg| *arg == "-E").count(), 1);
        let filter = args
            .windows(2)
            .find(|pair| pair[0] == "-E")
            .map(|pair| pair[1].as_str())
            .expect("one folded filter");
        assert!(
            filter.contains("package(=scx_cosmos) | package(=scx_lavd) | package(=scx_layered)")
        );
        assert!(filter.contains("(test(large)) | (test(kernel_gke))"));
    }

    #[test]
    fn older_package_message_stays_short_and_actionable() {
        assert_eq!(
            format_older_package_skip(&scoped_plan().older[0]),
            "cargo ktstr verifier: skipping scx_mitosis (ktstr 0.18.0): \
             this test is older; update or exclude it",
        );
    }

    #[test]
    fn metadata_passthrough_keeps_only_resolution_options() {
        assert_eq!(
            crate::feature_discovery::metadata_passthrough_options(&strings(&[
                "--locked",
                "--features",
                "ktstr-tests",
                "--config",
                "patch.crates-io.ktstr.path='../ktstr'",
                "--offline",
                "-p",
                "scx_layered",
            ])),
            strings(&[
                "--locked",
                "--config",
                "patch.crates-io.ktstr.path='../ktstr'",
                "--offline",
            ]),
        );
    }

    #[test]
    fn sweep_removes_dead_pid_result_dirs_keeps_live() {
        // A result dir owned by a dead pid is reclaimed; one owned by our
        // own (live) pid is kept. pid 2147483647 (i32::MAX) is above
        // pid_max, so kill() returns ESRCH. Each nextest test is its own
        // process, so the shared temp dir is test-isolated by pid.
        let base = std::env::temp_dir().join(format!("ktstr-vsweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk temp root");
        let dead = base.join("ktstr-verifier-results-2147483647");
        let live = base.join(format!("ktstr-verifier-results-{}", std::process::id()));
        std::fs::create_dir(&dead).expect("mk dead dir");
        std::fs::create_dir(&live).expect("mk live dir");

        sweep_stale_result_dirs(&base);

        assert!(!dead.exists(), "a dead-owner result dir is reclaimed");
        assert!(live.exists(), "our own (live) result dir is kept");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The verifier's reserved warm-up argv is the combined run's
    /// `build_nextest_args` output plus exactly a trailing `--no-run` —
    /// same nextest profile, cell filter, and forwarded user args — so
    /// the warm-up compiles the identical target set and the combined
    /// run finds everything cached. Mirrors run_cargo's
    /// `warmup_command_mirrors_run_argv_plus_no_run`.
    #[test]
    fn verifier_warmup_argv_is_run_argv_plus_no_run() {
        let forwarded = vec!["--cargo-profile".to_string(), "release".to_string()];
        let run_argv = build_scoped_nextest_args(
            Some("ci"),
            &forwarded,
            &VerifierPackagePlan {
                compatible: Vec::new(),
                older: Vec::new(),
                newer: Vec::new(),
            },
        );
        let warm_argv = crate::run_cargo::prebuild_no_run_args(&run_argv);
        assert_eq!(
            warm_argv[..warm_argv.len() - 1],
            run_argv[..],
            "warm-up must share the run's argv prefix verbatim",
        );
        assert_eq!(
            warm_argv.last().map(String::as_str),
            Some("--no-run"),
            "warm-up must append exactly --no-run",
        );
        // The reachability-critical flags survive into the warm-up: the
        // cell filter bounds what gets compiled-for-run selection, and
        // the profiles pin the same build fingerprint.
        assert!(
            warm_argv
                .iter()
                .any(|a| a == "test(/^verifier/) & !test(/^verifier::/)"),
            "verifier-cell filter present in warm-up: {warm_argv:?}",
        );
        assert!(
            warm_argv
                .windows(2)
                .any(|w| w[0] == "--profile" && w[1] == "ci"),
            "nextest profile present in warm-up: {warm_argv:?}",
        );
        assert!(
            warm_argv
                .windows(2)
                .any(|w| w[0] == "--cargo-profile" && w[1] == "release"),
            "forwarded cargo profile present in warm-up: {warm_argv:?}",
        );
    }

    #[test]
    fn verifier_warmup_preserves_targeted_features() {
        let run_argv = build_scoped_nextest_args(None, &[], &scoped_plan());
        let warm_argv = crate::run_cargo::prebuild_no_run_args(&run_argv);
        let expected = [
            "--features",
            "scx_cosmos/ktstr-tests,scx_lavd/ktstr-tests,scx_layered/ktstr-tests",
        ];
        assert!(
            run_argv.windows(2).any(|pair| pair == expected),
            "the run must enable only inferred verifier features: {run_argv:?}",
        );
        assert!(
            warm_argv.windows(2).any(|pair| pair == expected),
            "the warm-up must compile the same declaration targets: {warm_argv:?}",
        );
        assert!(!run_argv.iter().any(|arg| arg == "--all-features"));
        assert!(!warm_argv.iter().any(|arg| arg == "--all-features"));
    }
}
