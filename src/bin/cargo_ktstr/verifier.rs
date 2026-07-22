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
//! when multiple warmed test binaries report the same full scheduler
//! declaration, the parent records one canonical binary owner and both
//! listing and exact dispatch enforce that ownership, so the declaration
//! produces one VM cell and one result writer rather than one of each per
//! binary.
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
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::semver::Version;
use cargo_metadata::{Metadata, PackageId};

use crate::feature_discovery::{
    MetadataMode, PackageFeatureActivation, TargetContext, VersionScope,
    declaring_metadata_options, effective_target_context, has_package_selector,
    has_workspace_selector, infer_ktstr_feature_roots_for_context, inject_feature_activations,
    package_spec_name, query_metadata_for_target, query_resolved_metadata, scheduler_build_options,
    selected_activations_for_context, selected_workspace_packages,
};
#[cfg(test)]
use crate::feature_discovery::{
    explicit_package_exclusions, explicit_package_selection, infer_ktstr_feature_roots,
    selected_activations,
};
use crate::kernel::{
    encode_kernel_list, path_kernel_label, resolve_kernel_image, resolve_kernel_set,
};

#[derive(Debug, Clone)]
struct SchedulerWorkspace {
    metadata: Metadata,
    root: PathBuf,
    target_dir: PathBuf,
    package_id: PackageId,
}

type SchedulerWorkspaceCache = BTreeMap<(String, String), Option<SchedulerWorkspace>>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoverSchedulerRequest {
    scheduler: String,
    package: String,
    manifest_dir: String,
    workspace_root: PathBuf,
    target_dir: PathBuf,
    package_id: PackageId,
    metadata: Metadata,
}

#[derive(Debug, Clone)]
struct WorkspaceSchedulerBuild {
    root: PathBuf,
    target_dir: PathBuf,
    /// package name -> Cargo package ID
    packages: BTreeMap<String, PackageId>,
    requests: Vec<DiscoverSchedulerRequest>,
    metadata: Option<Metadata>,
}

fn scheduler_group_remapped_to_stable_source(
    group: &WorkspaceSchedulerBuild,
    stable_workspace: &Path,
    build_options: &[String],
) -> Result<WorkspaceSchedulerBuild, String> {
    let mut command = cargo_metadata::MetadataCommand::new();
    let mut options = crate::feature_discovery::metadata_passthrough_options(build_options);
    options.extend(crate::feature_discovery::metadata_resolution_options(
        build_options,
    ));
    command
        .cargo_path("cargo")
        .current_dir(stable_workspace)
        .no_deps()
        .other_options(options);
    let metadata = command.exec().map_err(|error| {
        format!(
            "cargo metadata from stable scheduler workspace {} failed: {error}",
            stable_workspace.display()
        )
    })?;
    let members = metadata.workspace_members.iter().collect::<HashSet<_>>();
    let mut remapped = group.clone();
    for (name, id) in &mut remapped.packages {
        let package = metadata
            .packages
            .iter()
            .find(|package| members.contains(&package.id) && package.name.as_str() == name)
            .ok_or_else(|| {
                format!(
                    "stable scheduler workspace {} has no member package {name:?}",
                    stable_workspace.display()
                )
            })?;
        *id = package.id.clone();
    }
    Ok(remapped)
}

/// Scheduler declarations retained with the exact warmed test executable
/// which reported them.
///
/// Keeping this provenance past the declaration probe is what lets the parent
/// elect one child owner when multiple binaries link an identical scheduler.
#[derive(Debug, Clone)]
struct TestBinarySchedulerDeclarations {
    executable: PathBuf,
    declarations: Vec<ktstr::test_support::SchedulerListEntry>,
}

type TestBinarySchedulerManifest = crate::misc::ProbedSchedulerManifest;

#[derive(Debug)]
struct SelectedSchedulerPlan {
    schedulers: Vec<ktstr::test_support::SchedulerJson>,
    discover_requests: Vec<DiscoverSchedulerRequest>,
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
#[cfg(test)]
fn test_link_edge(
    dep: &cargo_metadata::NodeDep,
    workspace_root: bool,
    requested_target: Option<&str>,
) -> bool {
    let target = requested_target.map(|target| {
        // Focused parser tests below exercise named target-table entries.
        // Production always supplies rustc's complete effective context.
        TargetContext::named_for_test(target)
    });
    test_link_edge_for_context(
        dep,
        workspace_root,
        target.as_ref(),
        requested_target.is_none(),
    )
}

fn test_link_edge_for_context(
    dep: &cargo_metadata::NodeDep,
    workspace_root: bool,
    target: Option<&TargetContext>,
    unfiltered_without_context: bool,
) -> bool {
    dep.dep_kinds.is_empty()
        || dep.dep_kinds.iter().any(|kind| {
            let linked_kind = matches!(kind.kind, cargo_metadata::DependencyKind::Normal)
                || (workspace_root
                    && matches!(kind.kind, cargo_metadata::DependencyKind::Development));
            linked_kind && dep_kind_matches_target_context(kind, target, unfiltered_without_context)
        })
}

fn dep_kind_matches_target_context(
    kind: &cargo_metadata::DepKindInfo,
    target: Option<&TargetContext>,
    unfiltered_without_context: bool,
) -> bool {
    let Some(platform) = kind.target.as_ref() else {
        return true;
    };
    let Some(target) = target else {
        return unfiltered_without_context;
    };
    target.matches_platform(platform)
}

/// Collect every ktstr version in one workspace member's package-level test
/// link closure.
///
/// This deliberately walks beyond the direct edge. A current ktstr test
/// package can also link a dependency carrying an old ktstr and therefore an
/// old distributed scheduler registry. Package-level metadata cannot prove
/// which individual test binary retains that dependency, so any such mixed
/// package is excluded conservatively.
fn linked_ktstr_versions_for_context(
    member_id: &PackageId,
    packages: &HashMap<&PackageId, &cargo_metadata::Package>,
    nodes: &HashMap<&PackageId, &cargo_metadata::Node>,
    target: Option<&TargetContext>,
    unfiltered_without_context: bool,
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
            if !test_link_edge_for_context(dep, workspace_root, target, unfiltered_without_context)
            {
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
#[cfg(test)]
fn verifier_package_plan(
    meta: &Metadata,
    cli: &Version,
    requested_target: Option<&str>,
) -> Result<VerifierPackagePlan, String> {
    let target = requested_target.map(TargetContext::named_for_test);
    verifier_package_plan_for_context(meta, cli, target.as_ref(), requested_target.is_none())
}

fn verifier_package_plan_for_context(
    meta: &Metadata,
    cli: &Version,
    target: Option<&TargetContext>,
    unfiltered_without_context: bool,
) -> Result<VerifierPackagePlan, String> {
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
        let versions = linked_ktstr_versions_for_context(
            member_id,
            &packages,
            &nodes,
            target,
            unfiltered_without_context,
        );
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
                verifier_features: infer_ktstr_feature_roots_for_context(
                    member,
                    VersionScope::Matches(cli),
                    target,
                ),
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

/// Cargo package selection used by verifier discovery.
///
/// A bare verifier intentionally scans all members, unlike ordinary test
/// commands' default-member semantics. A lone exclusion likewise needs an
/// implicit workspace base. Exact/glob package selectors remain untouched.
fn verifier_selection_args(args: &[String]) -> Vec<String> {
    if has_workspace_selector(args) || has_explicit_package_selector(args) {
        return args.to_vec();
    }
    std::iter::once("--workspace".to_string())
        .chain(args.iter().cloned())
        .collect()
}

/// Resolve the verifier package partition before either nextest warm-up or run.
fn query_verifier_package_plan(
    args: &[String],
) -> Result<(VerifierPackagePlan, Option<Metadata>), String> {
    let cli = Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("cargo-ktstr's own CARGO_PKG_VERSION is valid semver");
    let target = effective_target_context(args)
        .map_err(|error| format!("cargo ktstr verifier: determine Cargo target: {error}"))?;
    // First inspect workspace manifests without resolving optional
    // dependencies, then resolve only the ktstr-specific gates inferred from
    // those manifests. This gives recursive linked-version classification the
    // exact graph compilation will use without a broad all-features resolve.
    let manifests = query_metadata_for_target(args, MetadataMode::NoDeps, &target)
        .map_err(|error| format!("cargo ktstr verifier: {error}"))?;
    // Selection must precede optional feature activation. Otherwise an
    // unrelated workspace member can pull an old/new ktstr into Default
    // metadata even though Cargo was asked to verify one different package.
    let selection_args = verifier_selection_args(args);
    if selected_workspace_packages(&manifests, &selection_args).is_none() {
        // Do not guess at malformed/missing Cargo selector syntax. An empty
        // compatibility plan injects no features or package widening; the
        // unchanged selector reaches nextest/Cargo, which remains the
        // authoritative parser and diagnostic source.
        return Ok((
            VerifierPackagePlan {
                compatible: Vec::new(),
                older: Vec::new(),
                newer: Vec::new(),
            },
            None,
        ));
    }
    let activations = selected_activations_for_context(
        &manifests,
        &selection_args,
        VersionScope::Any,
        Some(&target),
    );
    let resolution_args = inject_feature_activations(args.to_vec(), &activations);
    let metadata = query_resolved_metadata(&resolution_args, &selection_args, &manifests, &target)
        .map_err(|error| format!("cargo ktstr verifier: {error}"))?;
    let mut plan = verifier_package_plan_for_context(&metadata, &cli, Some(&target), false)?;

    // A bare verifier deliberately widens beyond Cargo's default members.
    // Once the operator supplies package selection, however, classify only
    // the exact/globbed workspace packages Cargo selected. A lone --exclude
    // applies to that widened workspace selection, so synthesize --workspace
    // for metadata selection without changing the forwarded Cargo argv.
    if let Some(packages) = selected_workspace_packages(&metadata, &selection_args) {
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
    Ok((plan, Some(metadata)))
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

/// Build the verifier run argv and apply ktstr's low-priority nextest policy.
///
/// Kept as one operation so the production run cannot accidentally inject
/// after cloning a separate warm-up argv. The caller derives both the JSON
/// no-run discovery command and the final sweep from the returned vector.
fn build_injected_scoped_nextest_args_with(
    nextest_profile: Option<&str>,
    forward: &[String],
    plan: &VerifierPackagePlan,
    inject: impl FnOnce(Vec<String>) -> Result<Vec<String>, String>,
) -> Result<Vec<String>, String> {
    inject(build_scoped_nextest_args(nextest_profile, forward, plan))
        .map(crate::run_cargo::normalize_nextest_command_admission)
}

fn probe_scheduler_manifests(
    test_bins: &[PathBuf],
    loader_paths: &[PathBuf],
    probe_provenance: Option<&HashMap<PathBuf, PathBuf>>,
) -> Result<Vec<TestBinarySchedulerManifest>, String> {
    crate::misc::probe_scheduler_manifests_from_bins(
        test_bins,
        loader_paths,
        probe_provenance,
        "warmed test binaries for scheduler manifests",
    )
    .map_err(|error| format!("probe warmed test binaries for scheduler manifests: {error}"))
}

fn probe_scheduler_declarations(
    test_bins: &[PathBuf],
    loader_paths: &[PathBuf],
    probe_provenance: &HashMap<PathBuf, PathBuf>,
) -> Result<Vec<TestBinarySchedulerDeclarations>, String> {
    probe_scheduler_manifests(test_bins, loader_paths, Some(probe_provenance)).map(|manifests| {
        manifests
            .into_iter()
            .map(|binary| TestBinarySchedulerDeclarations {
                executable: binary.executable,
                declarations: binary.manifest.declarations,
            })
            .collect()
    })
}

fn cached_scheduler_declarations(
    build_directory: &Path,
    stamps: &[crate::nextest_artifact_cache::CachedBinaryStamp],
) -> Vec<TestBinarySchedulerDeclarations> {
    stamps
        .iter()
        .map(|stamp| TestBinarySchedulerDeclarations {
            executable: build_directory.join(&stamp.relative_binary),
            declarations: stamp.manifest.declarations.clone(),
        })
        .collect()
}

fn probe_scheduler_artifact_requirements(
    test_bins: &[PathBuf],
    loader_paths: &[PathBuf],
) -> Result<Vec<ktstr::test_support::SchedulerArtifactRequirement>, String> {
    if test_bins.is_empty() {
        return Ok(Vec::new());
    }
    let per_binary = probe_scheduler_manifests(test_bins, loader_paths, None)?;
    merge_scheduler_artifact_requirements(per_binary.into_iter().map(|binary| binary.manifest))
}

fn merge_scheduler_artifact_requirements(
    manifests: impl IntoIterator<Item = ktstr::test_support::SchedulerManifestProbe>,
) -> Result<Vec<ktstr::test_support::SchedulerArtifactRequirement>, String> {
    struct RequirementAccumulator {
        binary_kind: ktstr::test_support::BinaryKindJson,
        manifest_dir: String,
        schedulers: BTreeMap<String, ()>,
        use_count: usize,
    }
    let mut merged: BTreeMap<(u8, String, String), RequirementAccumulator> = BTreeMap::new();
    for requirement in manifests
        .into_iter()
        .flat_map(|manifest| manifest.artifact_requirements)
    {
        let (kind_order, value) = match &requirement.binary_kind {
            ktstr::test_support::BinaryKindJson::Discover(value) => (0, value.clone()),
            ktstr::test_support::BinaryKindJson::Path(value) => (1, value.clone()),
            ktstr::test_support::BinaryKindJson::Eevdf
            | ktstr::test_support::BinaryKindJson::KernelBuiltin => {
                return Err(
                    "scheduler artifact requirements probe emitted a kernel-only scheduler"
                        .to_string(),
                );
            }
        };
        let key = (kind_order, value, requirement.manifest_dir.clone());
        let accumulator = merged.entry(key).or_insert_with(|| RequirementAccumulator {
            binary_kind: requirement.binary_kind.clone(),
            manifest_dir: requirement.manifest_dir.clone(),
            schedulers: BTreeMap::new(),
            use_count: 0,
        });
        accumulator.use_count = accumulator
            .use_count
            .checked_add(requirement.use_count)
            .ok_or_else(|| "scheduler artifact requirement use_count overflow".to_string())?;
        for scheduler in requirement.schedulers {
            accumulator.schedulers.insert(scheduler, ());
        }
    }
    Ok(merged
        .into_values()
        .map(
            |requirement| ktstr::test_support::SchedulerArtifactRequirement {
                binary_kind: requirement.binary_kind,
                manifest_dir: requirement.manifest_dir,
                schedulers: requirement.schedulers.into_keys().collect(),
                use_count: requirement.use_count,
            },
        )
        .collect())
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

pub(crate) fn scheduler_profile_for_run(cli_profile: Option<&str>) -> String {
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
    metadata_options: &[String],
) -> Result<Option<SchedulerWorkspace>, String> {
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
        .other_options(metadata_options.to_vec());
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
    let target_dir = metadata.target_directory.clone().into_std_path_buf();
    let package_id = package.id.clone();
    Ok(Some(SchedulerWorkspace {
        metadata,
        root: workspace_root,
        target_dir,
        package_id,
    }))
}

/// Mirror every parent-visible child-listing gate before conflict detection,
/// owner election, or scheduler build planning. Workspace metadata
/// resolutions are cached by raw declaring directory + package so exact
/// declarations across test binaries run Cargo metadata once.
fn selected_scheduler_plan(
    declarations: &[ktstr::test_support::SchedulerListEntry],
    scheduler_filter: Option<&str>,
    resolved_kernels: &[(String, String)],
    presets: &[ktstr::gauntlet::TopoPreset],
    metadata_options: &[String],
) -> Result<SelectedSchedulerPlan, String> {
    let mut workspaces = SchedulerWorkspaceCache::new();
    let scheduler_override = std::env::var_os(ktstr::KTSTR_SCHEDULER_ENV);
    let schedulers =
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
                BinaryKindJson::Discover(_) if scheduler_override.is_some() => Ok(true),
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
                        let resolution = declaring_workspace(
                            &scheduler.name,
                            package,
                            &scheduler.manifest_dir,
                            metadata_options,
                        )?;
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

    let mut discover_requests = Vec::new();
    for scheduler in &schedulers {
        let ktstr::test_support::BinaryKindJson::Discover(package) = &scheduler.binary_kind else {
            continue;
        };
        if scheduler_override.is_some() {
            continue;
        }
        let key = (scheduler.manifest_dir.clone(), package.clone());
        let workspace = workspaces
            .get(&key)
            .cloned()
            .flatten()
            .expect("emitting Discover declaration has a member resolution");
        discover_requests.push(DiscoverSchedulerRequest {
            scheduler: scheduler.name.clone(),
            package: package.clone(),
            manifest_dir: scheduler.manifest_dir.clone(),
            workspace_root: workspace.root,
            target_dir: workspace.target_dir,
            package_id: workspace.package_id,
            metadata: workspace.metadata,
        });
    }
    discover_requests.sort_by(|left, right| {
        (&left.scheduler, &left.package, &left.manifest_dir).cmp(&(
            &right.scheduler,
            &right.package,
            &right.manifest_dir,
        ))
    });
    Ok(SelectedSchedulerPlan {
        schedulers,
        discover_requests,
    })
}

#[cfg(test)]
fn selected_discover_requests(
    declarations: &[ktstr::test_support::SchedulerListEntry],
    scheduler_filter: Option<&str>,
    resolved_kernels: &[(String, String)],
    presets: &[ktstr::gauntlet::TopoPreset],
    metadata_options: &[String],
) -> Result<Vec<DiscoverSchedulerRequest>, String> {
    Ok(selected_scheduler_plan(
        declarations,
        scheduler_filter,
        resolved_kernels,
        presets,
        metadata_options,
    )?
    .discover_requests)
}

fn plan_verifier_cell_ownership(
    binaries: &[TestBinarySchedulerDeclarations],
    selected: &[ktstr::test_support::SchedulerJson],
) -> Result<ktstr::verifier::VerifierCellOwnershipManifest, String> {
    let mut entries = Vec::with_capacity(selected.len());
    for scheduler in selected {
        let owner = binaries
            .iter()
            .filter(|binary| {
                binary
                    .declarations
                    .iter()
                    .any(|entry| entry.scheduler.eq(scheduler))
            })
            .min_by(|left, right| left.executable.cmp(&right.executable))
            .ok_or_else(|| {
                format!(
                    "selected scheduler {:?} has no warmed test-binary owner",
                    scheduler.name,
                )
            })?;
        entries.push(ktstr::verifier::VerifierCellOwnershipEntry {
            scheduler: scheduler.clone(),
            executable: owner.executable.clone(),
        });
    }
    entries.sort_by(|left, right| left.scheduler.name.cmp(&right.scheduler.name));
    Ok(ktstr::verifier::VerifierCellOwnershipManifest {
        version: ktstr::verifier::VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION,
        entries,
    })
}

fn plan_workspace_scheduler_builds(
    requests: &[DiscoverSchedulerRequest],
) -> Result<Vec<WorkspaceSchedulerBuild>, String> {
    let mut groups: BTreeMap<(PathBuf, PathBuf), WorkspaceSchedulerBuild> = BTreeMap::new();
    for request in requests {
        let root = request.workspace_root.clone();
        let target_dir = request.target_dir.clone();
        let package_id = request.package_id.clone();
        let group = groups
            .entry((root.clone(), target_dir.clone()))
            .or_insert_with(|| WorkspaceSchedulerBuild {
                root,
                target_dir,
                packages: BTreeMap::new(),
                requests: Vec::new(),
                metadata: Some(request.metadata.clone()),
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

fn scheduler_workspace_build_args(
    group: &WorkspaceSchedulerBuild,
    profile: &str,
    build_options: &[String],
) -> Vec<String> {
    let mut args = [
        "build",
        "--message-format=json-render-diagnostics",
        "--profile",
        profile,
    ]
    .into_iter()
    .map(ToString::to_string)
    .collect::<Vec<_>>();
    for package in group.packages.keys() {
        args.extend(["-p".to_string(), package.clone()]);
    }
    // Cargo accepts build options after package selectors. Keeping the
    // planner-owned argv prefix stable makes the replayed context visibly
    // separate from package/profile ownership.
    args.extend_from_slice(build_options);
    args
}

fn explicit_scheduler_target_dir(
    group: &WorkspaceSchedulerBuild,
    build_options: &[String],
) -> Option<PathBuf> {
    let mut explicit = None;
    let mut index = 0;
    while index < build_options.len() {
        let argument = &build_options[index];
        if argument == "--target-dir" {
            if let Some(value) = build_options.get(index + 1) {
                explicit = Some(PathBuf::from(value));
            }
            index += 2;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--target-dir=") {
            explicit = Some(PathBuf::from(value));
            index += 1;
            continue;
        }
        index += if matches!(argument.as_str(), "--config" | "--target" | "-Z") {
            2
        } else {
            1
        };
    }
    explicit.map(|path| {
        if path.is_absolute() {
            path
        } else {
            group.root.join(path)
        }
    })
}

fn effective_scheduler_target_dir(
    group: &WorkspaceSchedulerBuild,
    build_options: &[String],
) -> PathBuf {
    explicit_scheduler_target_dir(group, build_options).unwrap_or_else(|| group.target_dir.clone())
}

fn scheduler_workspace_execution(
    group: &WorkspaceSchedulerBuild,
    profile: &str,
    build_options: &[String],
) -> (Vec<String>, PathBuf) {
    let explicit_target_dir = explicit_scheduler_target_dir(group, build_options);
    let target_dir = effective_scheduler_target_dir(group, build_options);
    let mut build_args = scheduler_workspace_build_args(group, profile, build_options);
    if explicit_target_dir.is_none() {
        // Metadata resolves an inherited relative CARGO_TARGET_DIR against the
        // declaring member directory. The batched build runs from the
        // workspace root, where inheriting that same relative value would
        // silently select a different directory than the lock. Pin Cargo to
        // metadata's already-resolved absolute path so writer and lease are
        // necessarily identical.
        build_args.extend(["--target-dir".to_string(), target_dir.display().to_string()]);
    }
    (build_args, target_dir)
}

fn force_scheduler_stable_target(mut args: Vec<String>, target: &Path) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len() + 2);
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--target-dir" {
            index += 2;
        } else if args[index].starts_with("--target-dir=") {
            index += 1;
        } else {
            out.push(std::mem::take(&mut args[index]));
            index += 1;
        }
    }
    out.extend(["--target-dir".to_string(), target.display().to_string()]);
    out
}

struct WorkspaceSchedulerArtifacts {
    paths: BTreeMap<String, PathBuf>,
    tree: ktstr::cache::artifact_tree::MaterializedArtifactTree,
    stable_source: Option<crate::nextest_artifact_cache::StableCargoSource>,
}

fn scheduler_identity_not_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), String> {
    if cancelled() {
        Err("scheduler build identity planning interrupted".to_string())
    } else {
        Ok(())
    }
}

/// Runtime-only ktstr state that must neither split a Cargo artifact identity
/// nor reach the corresponding stable Cargo producer.
///
/// `cargo-ktstr` installs several of these values itself before planning an
/// artifact (notably the extracted busybox/wprof paths and project/runtime
/// coordinates). They affect how a finished test or scheduler is executed,
/// but not the bytes Cargo is asked to build. Keeping one shared list for
/// identity planning and child sanitization prevents a newly added runtime
/// coordinate from being fixed on only one side of that contract.
const CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT: &[&str] = &[
    "KTSTR_ADMISSION_CHAINED_RUNNER",
    "KTSTR_ADMISSION_ORIGINAL_RUNNER",
    "KTSTR_ADMISSION_TARGET_ENV_KEY",
    "KTSTR_BUDGET_SECS",
    "KTSTR_BUILD_DIAGNOSTICS_DIR",
    "KTSTR_BUSYBOX_PATH",
    "KTSTR_BYPASS_LLC_LOCKS",
    "KTSTR_CARGO_TEST_MODE",
    "KTSTR_CGROUP_WALK_ROOT",
    "KTSTR_CONTENTION_BYPASS",
    "KTSTR_CPU_CAP",
    "KTSTR_DEBUG",
    "KTSTR_GUEST_INIT",
    "KTSTR_HOST_CGROUP_PARENT",
    "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
    "KTSTR_JEMALLOC_PROBE_BINARY",
    "KTSTR_KERNEL",
    "KTSTR_KERNEL_COMMIT",
    "KTSTR_KERNEL_LIST",
    "KTSTR_KERNEL_PARALLELISM",
    "KTSTR_LOCK_DIR",
    "KTSTR_LOG_PASSES",
    "KTSTR_NO_PERF_MODE",
    "KTSTR_NO_SKIP_MODE",
    "KTSTR_ORCHESTRATED",
    "KTSTR_PERF_ONLY",
    "KTSTR_PROJECT_COMMIT",
    "KTSTR_RUN_EPOCH",
    "KTSTR_RUNS_ROOT",
    "KTSTR_SCHEDULER",
    "KTSTR_SCHEDULER_MANIFEST",
    "KTSTR_SCHEDULER_PROFILE",
    "KTSTR_SIDECAR_DIR",
    "KTSTR_SOURCE_ROOT_REMAPS",
    "KTSTR_STUCK_POLL_MS",
    "KTSTR_TEST_KERNEL",
    "KTSTR_VERBOSE",
    "KTSTR_VERIFIER_CELL_OWNERSHIP_MANIFEST",
    "KTSTR_VERIFIER_RAW",
    "KTSTR_VERIFIER_RESULT_DIR",
    "KTSTR_VERIFIER_SCHEDULER",
    "KTSTR_WPROF_PATH",
];

/// Per-service coordinates installed by systemd for the current runner.
///
/// The directory variables are the complete family documented by
/// `systemd.exec` for the corresponding `*Directory=` settings, plus the
/// credentials directory. The memory-pressure values similarly identify
/// service-manager-owned runtime endpoints. Their paths differ per runner and
/// service instance, but none describe bytes requested from a Cargo producer.
pub(crate) const CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT: &[&str] = &[
    "CACHE_DIRECTORY",
    "CONFIGURATION_DIRECTORY",
    "CREDENTIALS_DIRECTORY",
    "LOGS_DIRECTORY",
    "MEMORY_PRESSURE_WATCH",
    "MEMORY_PRESSURE_WRITE",
    "RUNTIME_DIRECTORY",
    "STATE_DIRECTORY",
];

/// Per-job GitHub Actions and runner logging coordinates.
///
/// These values differ between matrix jobs which otherwise compile the exact
/// same Cargo closure. They only identify the Actions orchestration stream and
/// state/log sinks; cached producers neither consume nor publish through those
/// endpoints, so strip them from both the identity and the child environment.
pub(crate) const CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT: &[&str] =
    &["ACTIONS_ORCHESTRATION_ID", "GITHUB_STATE", "LOG_NAMESPACE"];

/// Operational cache controls are intentionally inherited by a producer, but
/// cannot describe its output bytes and therefore must not enter its key.
/// In particular, ghars supplies one trust-zone-wide `KTSTR_CACHE_DIR`; the
/// scheduler build script uses it to share exact gix source nodes across
/// runner homes.
const SCHEDULER_BUILD_PRESERVED_OPERATIONAL_ENVIRONMENT: &[&str] =
    &["KTSTR_CACHE_DIR", "KTSTR_GHA_CACHE"];

/// Whether an inherited coordinate belongs to test/runtime orchestration
/// rather than any cached Cargo producer. Ordinary nextest, llvm-cov nextest,
/// recursive verifier, and scheduler builds share this classification.
pub(crate) fn cached_cargo_build_environment_is_runtime(name: &std::ffi::OsStr) -> bool {
    crate::nextest_process::is_runtime_environment(name)
        || name.to_str().is_some_and(|name| {
            CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT.contains(&name)
                || CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT.contains(&name)
                || CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT.contains(&name)
        })
}

fn scheduler_build_environment_is_nonsemantic(name: &std::ffi::OsStr) -> bool {
    if cached_cargo_build_environment_is_runtime(name) {
        return true;
    }
    let Some(name) = name.to_str() else {
        return false;
    };
    if name == "LLVM_PROFILE_FILE"
        || SCHEDULER_BUILD_PRESERVED_OPERATIONAL_ENVIRONMENT.contains(&name)
    {
        return true;
    }

    // This denylist is intentionally operational rather than build-oriented:
    // output placement, concurrency/jobserver controls, terminal/session
    // plumbing, CI per-run endpoints, credentials, and compiler paths whose
    // resolved tool contents are hashed separately. Every other inherited
    // variable is conservatively treated as a build-script input.
    let exact = [
        "_",
        "ACTIONS_CACHE_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_RUNTIME_URL",
        "CARGO_BUILD_BUILD_DIR",
        "CARGO_BUILD_DEP_INFO_BASEDIR",
        "CARGO_BUILD_JOBS",
        "CARGO_BUILD_RUSTC",
        "CARGO_BUILD_RUSTC_WRAPPER",
        "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
        "CARGO_MAKEFLAGS",
        "CARGO_TARGET_DIR",
        "CCACHE_DIR",
        "CLICOLOR",
        "CLICOLOR_FORCE",
        "COLORTERM",
        "DBUS_SESSION_BUS_ADDRESS",
        "FORCE_COLOR",
        "GH_TOKEN",
        "GITHUB_ACTION",
        "GITHUB_ACTION_PATH",
        "GITHUB_ACTION_REPOSITORY",
        "GITHUB_ACTOR",
        "GITHUB_ACTOR_ID",
        "GITHUB_ENV",
        "GITHUB_EVENT_PATH",
        "GITHUB_JOB",
        "GITHUB_OUTPUT",
        "GITHUB_PATH",
        "GITHUB_RETENTION_DAYS",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_STEP_SUMMARY",
        "GITHUB_TOKEN",
        "GITHUB_TRIGGERING_ACTOR",
        "GITHUB_WORKSPACE",
        "GIT_ASKPASS",
        "GIT_OPTIONAL_LOCKS",
        "HOSTNAME",
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "LESS",
        "LOGNAME",
        "LS_COLORS",
        "MAKEFLAGS",
        "MFLAGS",
        "NO_COLOR",
        "NUM_JOBS",
        "OLDPWD",
        "OUT_DIR",
        "PAGER",
        "PWD",
        "RAYON_NUM_THREADS",
        "RUSTC",
        "RUSTC_BACKTRACE",
        "RUSTC_LOG",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "SCCACHE_DIR",
        "SHELL",
        "SHLVL",
        "SSH_ASKPASS",
        "SSH_AUTH_SOCK",
        "SSH_CLIENT",
        "SSH_CONNECTION",
        "SSH_TTY",
        "SYSTEMD_EXEC_PID",
        "TEMP",
        "TERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TMP",
        "TMPDIR",
        "TOKIO_WORKER_THREADS",
        "USER",
        "VISUAL",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
    ];
    let prefixes = [
        "ACTIONS_ID_TOKEN_",
        "ACTIONS_RUNTIME_",
        "CCACHE_",
        "RUNNER_",
        "SCCACHE_",
    ];
    exact.contains(&name)
        || prefixes.iter().any(|prefix| name.starts_with(prefix))
        || (name.starts_with("CARGO_TARGET_") && name.ends_with("_RUNNER"))
        || (name.starts_with("CARGO_REGISTRIES_") && name.ends_with("_TOKEN"))
}

fn scheduler_replace_identity_path(bytes: &[u8], path: &[u8], replacement: &[u8]) -> Vec<u8> {
    if path.len() <= 1 || bytes.len() < path.len() {
        return bytes.to_vec();
    }
    let mut output = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    while let Some(offset) = bytes[cursor..]
        .windows(path.len())
        .position(|window| window == path)
    {
        let found = cursor + offset;
        output.extend_from_slice(&bytes[cursor..found]);
        output.extend_from_slice(replacement);
        cursor = found + path.len();
    }
    output.extend_from_slice(&bytes[cursor..]);
    output
}

fn scheduler_build_environment_from(
    workspace_root: &Path,
    mut environment: Vec<(std::ffi::OsString, std::ffi::OsString)>,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<(std::ffi::OsString, Vec<u8>)>, String> {
    use std::os::unix::ffi::OsStrExt as _;

    // Scheduler artifacts are complete, immutable cache entries. Incremental
    // dep graphs cannot be reused after publication, so keeping them only
    // consumes cache space. Normalize the value before hashing the producer
    // environment so inherited settings and actual execution cannot diverge.
    environment.retain(|(name, _)| name != "CARGO_INCREMENTAL");
    environment.push(("CARGO_INCREMENTAL".into(), "0".into()));
    let workspace_root =
        std::fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let home = environment
        .iter()
        .find(|(name, _)| name == "HOME")
        .map(|(_, value)| value.as_os_str().as_bytes().to_vec());
    let mut semantic = Vec::with_capacity(environment.len());
    for (name, value) in environment {
        scheduler_identity_not_cancelled(cancelled)?;
        if scheduler_build_environment_is_nonsemantic(&name) {
            continue;
        }
        let mut value = scheduler_replace_identity_path(
            value.as_os_str().as_bytes(),
            workspace_root.as_os_str().as_bytes(),
            b"$WORKSPACE",
        );
        if let Some(home) = &home {
            value = scheduler_replace_identity_path(&value, home, b"$HOME");
        }
        semantic.push((name, value));
    }
    semantic.sort_by(|left, right| left.0.cmp(&right.0));
    scheduler_identity_not_cancelled(cancelled)?;
    Ok(semantic)
}

pub(crate) fn scheduler_build_environment(
    workspace_root: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<(std::ffi::OsString, Vec<u8>)>, String> {
    scheduler_build_environment_from(workspace_root, std::env::vars_os().collect(), cancelled)
}

/// Remove run-instance state from a scheduler Cargo child.
///
/// These variables select kernels, result directories, admission modes, and
/// verifier ownership after the scheduler artifact already exists, or point at
/// systemd runtime resources belonging to one runner service. Letting those
/// unique paths reach the child both exposes irrelevant inputs to build scripts
/// and defeats the machine-wide content-addressed scheduler cache. Compiler
/// wrappers, operational cache controls, and build-affecting variables remain
/// inherited.
fn sanitize_scheduler_build_child_environment(command: &mut Command) {
    for &name in CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT
        .iter()
        .chain(CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT)
        .chain(CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT)
    {
        command.env_remove(name);
    }
    command.env_remove("LLVM_PROFILE_FILE");
    // OUT_DIR belongs to the Cargo invocation which built/launched
    // cargo-ktstr. The scheduler Cargo child supplies its own build-script
    // OUT_DIR and must not inherit or key on the parent's feature-specific
    // target path.
    command.env_remove("OUT_DIR");
    crate::nextest_process::remove_runtime_environment(command);
    for (name, _) in std::env::vars_os() {
        if name
            .to_str()
            .is_some_and(|name| name.starts_with("CARGO_TARGET_") && name.ends_with("_RUNNER"))
        {
            command.env_remove(name);
        }
    }
    command.env("CARGO_INCREMENTAL", "0");
}

fn build_scheduler_workspace(
    group: &WorkspaceSchedulerBuild,
    profile: &str,
    build_options: &[String],
    cli_label: &str,
    containing_source: Option<&crate::nextest_artifact_cache::StableCargoSource>,
) -> Result<WorkspaceSchedulerArtifacts, String> {
    let metadata = group.metadata.as_ref().ok_or_else(|| {
        format!(
            "scheduler workspace {} has no Cargo metadata for stable source materialization",
            group.root.display()
        )
    })?;
    let (identity_build_args, identity_target_dir) =
        scheduler_workspace_execution(group, profile, build_options);
    let reused_identity = containing_source
        .map(|source| {
            source.contained_invocation_identity(
                metadata,
                "scheduler-workspace",
                &identity_build_args,
                std::slice::from_ref(&identity_target_dir),
                &group.root,
            )
        })
        .transpose()?
        .flatten();
    let mut owned_stable_source = None;
    let (identity, stable_source, source_is_already_stable) =
        if let Some(identity) = reused_identity {
            (
                identity,
                containing_source.expect("reused identity has a containing source"),
                true,
            )
        } else {
            let identity_plan = crate::nextest_artifact_cache::identity_plan_for_invocation(
                metadata,
                "scheduler-workspace",
                &identity_build_args,
                std::slice::from_ref(&identity_target_dir),
                &group.root,
            )?;
            let identity = identity_plan.identity;
            owned_stable_source = Some(identity_plan.stable_source(cli_label)?);
            (
                identity,
                owned_stable_source
                    .as_ref()
                    .expect("stable scheduler source materialized above"),
                false,
            )
        };
    let stable_build_options = stable_source.remap_cargo_args(build_options);
    let (build_args, _identity_target_dir) =
        scheduler_workspace_execution(group, profile, &stable_build_options);
    let packages = group.packages.keys().cloned().collect::<Vec<_>>();
    let cached = ktstr::scheduler_artifact::load_or_build_scheduler_workspace_artifacts_stable(
        identity,
        &packages,
        cli_label,
        || Ok(true),
        || crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire),
        |stable_build| {
            // The build consumes only the immutable stable source whose
            // identity was validated before publication. There is no live
            // checkout left to rewalk after Cargo exits.
            let (stable_group, stable_workspace_root) = if source_is_already_stable {
                (group.clone(), group.root.as_path())
            } else {
                (
                    scheduler_group_remapped_to_stable_source(
                        group,
                        &stable_source.workspace_root,
                        &stable_build_options,
                    )?,
                    stable_source.workspace_root.as_path(),
                )
            };
            let mut command = Command::new("cargo");
            let stable_build_dir = stable_build.root.join("build");
            command
                .current_dir(stable_workspace_root)
                .args(force_scheduler_stable_target(
                    build_args.clone(),
                    &stable_build.target_directory,
                ))
                .env("CARGO_BUILD_BUILD_DIR", &stable_build_dir)
                .env("GIT_OPTIONAL_LOCKS", "0");
            sanitize_scheduler_build_child_environment(&mut command);
            crate::run_cargo::run_reserved_build_output_under_lease(
                command,
                cli_label,
                "scheduler workspace pre-build",
                crate::reserved_build_progress::ReservedBuildOutputKind::CargoJson,
                &stable_build.target_directory,
                |output| {
                    if !output.status.success() {
                        return Err(format!(
                            "scheduler prebuild in workspace {} failed ({}) — see Cargo \
                             output above",
                            group.root.display(),
                            output
                                .status
                                .code()
                                .map_or("signal".to_string(), |code| code.to_string()),
                        ));
                    }
                    map_scheduler_artifacts(&output.stdout, &stable_group)?
                        .into_iter()
                        .map(|(package, path)| {
                            let pinned = ktstr::cache::pin_content_file(&path).map_err(|error| {
                                format!(
                                    "pin Cargo-emitted scheduler package {package:?} artifact {} \
                                     while target output is exclusively owned: {error:#}",
                                    path.display(),
                                )
                            })?;
                            Ok((package, pinned))
                        })
                        .collect::<Result<BTreeMap<_, _>, String>>()
                },
            )
        },
    )?;
    if cached.cache_hit {
        eprintln!("{cli_label}: reused input-addressed scheduler workspace build {identity:016x}");
    }
    Ok(WorkspaceSchedulerArtifacts {
        paths: cached.paths,
        tree: cached.tree,
        stable_source: owned_stable_source,
    })
}

/// Parent-owned scheduler handoff kept alive through one nextest run.
pub(crate) struct PreparedSchedulerArtifacts {
    _owner: tempfile::TempDir,
    _snapshots: Vec<ktstr::scheduler_artifact::SchedulerArtifactSnapshot>,
    _scheduler_trees: Vec<ktstr::cache::artifact_tree::MaterializedArtifactTree>,
    _scheduler_sources: Vec<crate::nextest_artifact_cache::StableCargoSource>,
    pub(crate) manifest_path: PathBuf,
    pub(crate) binaries: Vec<PathBuf>,
}

/// Probe every primary and staged scheduler requirement from the exact warmed
/// test executables, build each distinct Discover package once, and snapshot
/// every Discover and Path source into the shared immutable content CAS.
///
/// The returned owner and CAS leases must remain alive through nextest. Its
/// manifest is complete and authoritative for all child resolution; there is
/// no target-directory scan, declaration-only shortcut, or child Cargo
/// fallback.
pub(crate) fn prepare_scheduler_artifacts(
    test_bins: &[PathBuf],
    loader_paths: &[PathBuf],
    cli_profile: Option<&str>,
    cargo_args: &[String],
    invocation_dir: &Path,
) -> Result<PreparedSchedulerArtifacts, String> {
    let requirements = probe_scheduler_artifact_requirements(test_bins, loader_paths)?;
    prepare_scheduler_artifacts_from_requirements(
        requirements,
        cli_profile,
        cargo_args,
        invocation_dir,
        None,
    )
}

/// Prepare scheduler artifacts from scheduler/admission manifests captured in
/// the generic nextest artifact closure. Cache hits use this path and never
/// reopen the large test executables merely to rediscover the same stamps.
pub(crate) fn prepare_scheduler_artifacts_from_cached_manifests(
    manifests: &[ktstr::test_support::SchedulerManifestProbe],
    stable_source: &crate::nextest_artifact_cache::StableCargoSource,
    cli_profile: Option<&str>,
    cargo_args: &[String],
    invocation_dir: &Path,
) -> Result<PreparedSchedulerArtifacts, String> {
    let requirements = merge_scheduler_artifact_requirements(manifests.iter().cloned())?;
    prepare_scheduler_artifacts_from_requirements(
        requirements,
        cli_profile,
        cargo_args,
        invocation_dir,
        Some(stable_source),
    )
}

fn prepare_scheduler_artifacts_from_requirements(
    requirements: Vec<ktstr::test_support::SchedulerArtifactRequirement>,
    cli_profile: Option<&str>,
    cargo_args: &[String],
    invocation_dir: &Path,
    stable_source: Option<&crate::nextest_artifact_cache::StableCargoSource>,
) -> Result<PreparedSchedulerArtifacts, String> {
    use ktstr::scheduler_artifact::{
        SCHEDULER_ARTIFACT_MANIFEST_VERSION, SchedulerArtifactEntry, SchedulerArtifactManifest,
        SchedulerArtifactSpec,
    };

    let profile = scheduler_profile_for_run(cli_profile);
    let metadata_options = declaring_metadata_options(cargo_args, invocation_dir);
    let build_options = scheduler_build_options(cargo_args, invocation_dir);
    let scheduler_override = std::env::var_os(ktstr::KTSTR_SCHEDULER_ENV).map(PathBuf::from);
    let mut workspaces = SchedulerWorkspaceCache::new();
    let owner = tempfile::Builder::new()
        .prefix("ktstr-scheduler-artifacts-")
        .tempdir()
        .map_err(|error| format!("create scheduler artifact handoff directory: {error}"))?;
    let mut requests = Vec::new();
    let mut pending: BTreeMap<(SchedulerArtifactSpec, String), (Vec<String>, PathBuf)> =
        BTreeMap::new();
    let mut discover_scheduler_names: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for requirement in requirements {
        if requirement.manifest_dir.is_empty() || requirement.schedulers.is_empty() {
            return Err(format!(
                "scheduler artifact requirement has empty manifest_dir or scheduler names: \
                 {requirement:?}"
            ));
        }
        match requirement.binary_kind {
            ktstr::test_support::BinaryKindJson::Eevdf
            | ktstr::test_support::BinaryKindJson::KernelBuiltin => {
                return Err(
                    "scheduler artifact requirements contained a kernel-only scheduler".to_string(),
                );
            }
            ktstr::test_support::BinaryKindJson::Path(path) => {
                let source = std::fs::canonicalize(&path).map_err(|error| {
                    format!(
                        "canonicalize required scheduler path {}: {error}",
                        Path::new(&path).display()
                    )
                })?;
                pending.insert(
                    (SchedulerArtifactSpec::Path(path), requirement.manifest_dir),
                    (requirement.schedulers, source),
                );
            }
            ktstr::test_support::BinaryKindJson::Discover(package)
                if scheduler_override.is_some() =>
            {
                let source = scheduler_override
                    .as_ref()
                    .expect("Discover override checked above");
                let source = std::fs::canonicalize(source).map_err(|error| {
                    format!(
                        "canonicalize KTSTR_SCHEDULER override {}: {error}",
                        source.display()
                    )
                })?;
                pending.insert(
                    (
                        SchedulerArtifactSpec::Discover(package),
                        requirement.manifest_dir,
                    ),
                    (requirement.schedulers, source),
                );
            }
            ktstr::test_support::BinaryKindJson::Discover(package) => {
                let scheduler = requirement.schedulers[0].clone();
                let key = (requirement.manifest_dir.clone(), package.clone());
                if !workspaces.contains_key(&key) {
                    let resolution = declaring_workspace(
                        &scheduler,
                        &package,
                        &requirement.manifest_dir,
                        &metadata_options,
                    )?;
                    workspaces.insert(key.clone(), resolution);
                }
                let Some(workspace) = workspaces.get(&key).cloned().flatten() else {
                    return Err(format!(
                        "required scheduler package {package:?} for {scheduler:?} is not a \
                         member of the workspace declared by {}",
                        requirement.manifest_dir,
                    ));
                };
                requests.push(DiscoverSchedulerRequest {
                    scheduler,
                    package: package.clone(),
                    manifest_dir: requirement.manifest_dir.clone(),
                    workspace_root: workspace.root,
                    target_dir: workspace.target_dir,
                    package_id: workspace.package_id,
                    metadata: workspace.metadata,
                });
                discover_scheduler_names
                    .insert((requirement.manifest_dir, package), requirement.schedulers);
            }
        }
    }

    let groups = plan_workspace_scheduler_builds(&requests)?;
    let package_count: usize = groups.iter().map(|group| group.packages.len()).sum();
    if package_count > 0 {
        eprintln!(
            "cargo ktstr: prebuilding {package_count} declared scheduler package(s) \
             in {} workspace batch(es) with profile {profile:?}",
            groups.len(),
        );
    }
    let mut snapshot_paths: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    let mut snapshots = Vec::new();
    let mut scheduler_trees = Vec::new();
    let mut scheduler_sources = Vec::new();
    for group in &groups {
        let artifacts = build_scheduler_workspace(
            group,
            &profile,
            &build_options,
            "cargo ktstr",
            stable_source,
        )?;
        for path in artifacts.paths.values() {
            snapshot_paths.insert(path.clone(), path.clone());
        }
        scheduler_trees.push(artifacts.tree);
        scheduler_sources.extend(artifacts.stable_source);
        for request in &group.requests {
            let emitted = artifacts
                .paths
                .get(&request.package)
                .expect("artifact completeness checked above");
            let schedulers = discover_scheduler_names
                .get(&(request.manifest_dir.clone(), request.package.clone()))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "lost scheduler names for required package {:?}, manifest_dir {:?}",
                        request.package, request.manifest_dir,
                    )
                })?;
            pending.insert(
                (
                    SchedulerArtifactSpec::Discover(request.package.clone()),
                    request.manifest_dir.clone(),
                ),
                (schedulers, emitted.clone()),
            );
        }
    }

    let mut entries = Vec::with_capacity(pending.len());
    for ((binary, manifest_dir), (schedulers, source)) in pending {
        let snapshot_path = if let Some(snapshot_path) = snapshot_paths.get(&source) {
            snapshot_path.clone()
        } else {
            let snapshot = ktstr::scheduler_artifact::snapshot_scheduler_artifact(&source)?;
            let snapshot_path = snapshot.path().to_path_buf();
            snapshot_paths.insert(source, snapshot_path.clone());
            snapshots.push(snapshot);
            snapshot_path
        };
        entries.push(SchedulerArtifactEntry {
            binary,
            manifest_dir,
            schedulers,
            path: snapshot_path,
        });
    }
    let mut binaries = snapshot_paths.into_values().collect::<Vec<_>>();
    binaries.sort();
    binaries.dedup();
    let manifest = SchedulerArtifactManifest {
        version: SCHEDULER_ARTIFACT_MANIFEST_VERSION,
        profile,
        entries,
    };
    let manifest_path =
        ktstr::scheduler_artifact::write_scheduler_artifact_manifest(owner.path(), &manifest)?;
    Ok(PreparedSchedulerArtifacts {
        _owner: owner,
        _snapshots: snapshots,
        _scheduler_trees: scheduler_trees,
        _scheduler_sources: scheduler_sources,
        manifest_path,
        binaries,
    })
}

struct SchedulerPrebuildContext<'a> {
    build_options: &'a [String],
    interrupted: &'a std::sync::atomic::AtomicBool,
}

struct PreparedSchedulerManifest {
    manifest: ktstr::scheduler_artifact::SchedulerArtifactManifest,
    _snapshots: Vec<ktstr::scheduler_artifact::SchedulerArtifactSnapshot>,
    _scheduler_trees: Vec<ktstr::cache::artifact_tree::MaterializedArtifactTree>,
    _scheduler_sources: Vec<crate::nextest_artifact_cache::StableCargoSource>,
}

fn prebuild_scheduler_manifest(
    requests: &[DiscoverSchedulerRequest],
    selected_schedulers: &[ktstr::test_support::SchedulerJson],
    profile: &str,
    context: SchedulerPrebuildContext<'_>,
) -> Result<PreparedSchedulerManifest, String> {
    use ktstr::scheduler_artifact::{
        SCHEDULER_ARTIFACT_MANIFEST_VERSION, SchedulerArtifactEntry, SchedulerArtifactManifest,
        SchedulerArtifactSpec,
    };

    if context
        .interrupted
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return Err("cargo ktstr verifier interrupted before scheduler prebuild".to_string());
    }
    let groups = plan_workspace_scheduler_builds(requests)?;
    let package_count: usize = groups.iter().map(|group| group.packages.len()).sum();
    if package_count > 0 {
        eprintln!(
            "cargo ktstr verifier: prebuilding {package_count} scheduler package(s) \
             in {} workspace batch(es) with profile {profile:?}",
            groups.len(),
        );
    }
    let mut sources: BTreeMap<(SchedulerArtifactSpec, String), (BTreeMap<String, ()>, PathBuf)> =
        BTreeMap::new();
    let mut snapshot_paths: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    let mut snapshots = Vec::new();
    let mut scheduler_trees = Vec::new();
    let mut scheduler_sources = Vec::new();
    for group in &groups {
        if context
            .interrupted
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err("cargo ktstr verifier interrupted during scheduler prebuild".to_string());
        }
        let artifacts = build_scheduler_workspace(
            group,
            profile,
            context.build_options,
            "cargo ktstr verifier",
            None,
        )?;
        for path in artifacts.paths.values() {
            snapshot_paths.insert(path.clone(), path.clone());
        }
        scheduler_trees.push(artifacts.tree);
        scheduler_sources.extend(artifacts.stable_source);
        if context
            .interrupted
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err("cargo ktstr verifier interrupted during scheduler prebuild".to_string());
        }
        for request in &group.requests {
            let emitted = artifacts
                .paths
                .get(&request.package)
                .expect("artifact completeness checked above");
            let path = emitted.clone();
            let key = (
                SchedulerArtifactSpec::Discover(request.package.clone()),
                request.manifest_dir.clone(),
            );
            let source = sources
                .entry(key)
                .or_insert_with(|| (BTreeMap::new(), path.clone()));
            if source.1 != path {
                return Err(format!(
                    "scheduler artifact identity for package {:?}, manifest_dir {:?} \
                     resolved to both {} and {}",
                    request.package,
                    request.manifest_dir,
                    source.1.display(),
                    path.display(),
                ));
            }
            source.0.insert(request.scheduler.clone(), ());
        }
    }

    let scheduler_override = std::env::var_os(ktstr::KTSTR_SCHEDULER_ENV)
        .map(PathBuf::from)
        .map(|path| {
            std::fs::canonicalize(&path).map_err(|error| {
                format!(
                    "canonicalize KTSTR_SCHEDULER override {}: {error}",
                    path.display()
                )
            })
        })
        .transpose()?;
    for scheduler in selected_schedulers {
        let (binary, source_path) = match &scheduler.binary_kind {
            ktstr::test_support::BinaryKindJson::Path(path) => (
                SchedulerArtifactSpec::Path(path.clone()),
                std::fs::canonicalize(path).map_err(|error| {
                    format!(
                        "canonicalize verifier scheduler {:?} path {}: {error}",
                        scheduler.name,
                        Path::new(path).display(),
                    )
                })?,
            ),
            ktstr::test_support::BinaryKindJson::Discover(package)
                if scheduler_override.is_some() =>
            {
                (
                    SchedulerArtifactSpec::Discover(package.clone()),
                    scheduler_override
                        .as_ref()
                        .expect("Discover override checked above")
                        .clone(),
                )
            }
            ktstr::test_support::BinaryKindJson::Discover(_) => continue,
            ktstr::test_support::BinaryKindJson::Eevdf
            | ktstr::test_support::BinaryKindJson::KernelBuiltin => continue,
        };
        let key = (binary, scheduler.manifest_dir.clone());
        let source = sources
            .entry(key)
            .or_insert_with(|| (BTreeMap::new(), source_path.clone()));
        if source.1 != source_path {
            return Err(format!(
                "scheduler artifact identity for {:?}, manifest_dir {:?} resolved to \
                 both {} and {}",
                scheduler.binary_kind,
                scheduler.manifest_dir,
                source.1.display(),
                source_path.display(),
            ));
        }
        source.0.insert(scheduler.name.clone(), ());
    }

    let mut entries = Vec::with_capacity(sources.len());
    for ((binary, manifest_dir), (scheduler_names, source_path)) in sources {
        let snapshot_path = if let Some(snapshot_path) = snapshot_paths.get(&source_path) {
            snapshot_path.clone()
        } else {
            let snapshot = ktstr::scheduler_artifact::snapshot_scheduler_artifact(&source_path)?;
            let snapshot_path = snapshot.path().to_path_buf();
            snapshot_paths.insert(source_path, snapshot_path.clone());
            snapshots.push(snapshot);
            snapshot_path
        };
        entries.push(SchedulerArtifactEntry {
            binary,
            manifest_dir,
            schedulers: scheduler_names.into_keys().collect(),
            path: snapshot_path,
        });
    }
    Ok(PreparedSchedulerManifest {
        manifest: SchedulerArtifactManifest {
            version: SCHEDULER_ARTIFACT_MANIFEST_VERSION,
            profile: profile.to_string(),
            entries,
        },
        _snapshots: snapshots,
        _scheduler_trees: scheduler_trees,
        _scheduler_sources: scheduler_sources,
    })
}

fn write_cell_ownership_manifest(
    result_dir: &Path,
    manifest: &ktstr::verifier::VerifierCellOwnershipManifest,
) -> Result<PathBuf, String> {
    use std::os::unix::fs::PermissionsExt;

    let final_path = result_dir.join("cell-ownership-v1.json");
    let mut temporary = tempfile::NamedTempFile::new_in(result_dir).map_err(|error| {
        format!(
            "create temporary verifier cell ownership manifest in {}: {error}",
            result_dir.display()
        )
    })?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), manifest)
        .map_err(|error| format!("serialize verifier cell ownership manifest: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("sync verifier cell ownership manifest: {error}"))?;
    temporary
        .as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .map_err(|error| format!("make verifier cell ownership manifest read-only: {error}"))?;
    temporary.persist(&final_path).map_err(|error| {
        format!(
            "atomically install verifier cell ownership manifest {}: {}",
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
    let invocation_dir = std::env::current_dir()
        .map_err(|error| format!("cargo ktstr verifier: read invocation directory: {error}"))?;
    let (package_plan, verifier_metadata) = query_verifier_package_plan(&args)?;
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
    let nextest_args = build_injected_scoped_nextest_args_with(
        nextest_profile.as_deref(),
        &args,
        &package_plan,
        crate::nextest_config::inject,
    )?;
    let harness_target_dir =
        crate::run_cargo::resolve_cargo_target_dir_for_args(&nextest_args, &invocation_dir)?;
    let scheduler_profile = scheduler_profile_for_run(profile.as_deref());
    let declaring_cargo_options = declaring_metadata_options(&args, &invocation_dir);
    let scheduler_cargo_options = scheduler_build_options(&args, &invocation_dir);

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
    cmd.env("GIT_OPTIONAL_LOCKS", "0");
    // Mark this test invocation as cargo-ktstr-orchestrated so
    // VM-boot tests can skip when run under raw nextest. Mirrors
    // the `cargo ktstr test` dispatcher in run_cargo.rs.
    cmd.env(ktstr::KTSTR_ORCHESTRATED_ENV, "1");
    let base_command_environment = cmd
        .get_envs()
        .filter_map(|(name, value)| value.map(|value| (name.to_os_string(), value.to_os_string())))
        .collect::<Vec<_>>();

    // Reserve + cgroup-confine the harness COMPILE phase only, exactly as
    // `run_cargo_sub` does for `cargo ktstr test` (see the block comment
    // there): `cargo nextest run` builds then runs in one process, so an
    // explicit `--no-run` warm-up compiles every test binary under a
    // machine-global LLC LOCK_SH + cpuset cgroup (a consolidated LLC
    // footprint leaves whole cache domains free for exclusive perf-mode
    // reservations while least-held CPU selection keeps compatible builds
    // from piling onto one CPU prefix), then releases BOTH before the
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
    // Everything above is preflight: package metadata and kernel resolution
    // retain terminate-immediately signal behavior. Cross into cleanup
    // ownership immediately before the first result directory/reservation.
    // The one top-level guard remains installed while every `?` below drops
    // `result_dir`, then re-raises the first caught signal as 130/143.
    crate::interrupt::enter_cleanup_phase()
        .map_err(|error| format!("cargo ktstr verifier: enter cleanup phase: {error}"))?;
    let guarded_result = (|| -> Result<Option<i32>, String> {
        // Per-cell result dir: each verifier cell writes its PASS/FAIL record
        // here (via KTSTR_VERIFIER_RESULT_DIR), and after nextest returns we
        // read them back to render the summary table. Creating it before the
        // warm-up makes the same RAII owner cover harness compilation,
        // declaration probes, scheduler prebuild/snapshots, manifest write,
        // and the final nextest run.
        let result_dir = VerifierResultDir::create(&std::env::temp_dir())?;
        if crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }

        let mut producer_environment = blob_envs
            .iter()
            .map(|(name, value)| (std::ffi::OsString::from(*name), value.clone()))
            .collect::<Vec<_>>();
        producer_environment.push((
            std::ffi::OsString::from("GIT_OPTIONAL_LOCKS"),
            std::ffi::OsString::from("0"),
        ));
        producer_environment.push((
            std::ffi::OsString::from(ktstr::KTSTR_KERNEL_ENV),
            resolved[0].1.as_os_str().to_os_string(),
        ));
        let cached_artifacts = if let Some(metadata) = verifier_metadata.as_ref() {
            let build_args = crate::run_cargo::nextest_command_build_surface(&nextest_args)?;
            Some(crate::run_cargo::load_or_build_nextest_artifacts(
                metadata,
                crate::run_cargo::CachedNextestMode::Plain,
                &build_args,
                &[],
                false,
                std::slice::from_ref(&harness_target_dir),
                &producer_environment,
                Some(&resolved[0].1),
                "cargo ktstr verifier nextest artifact cache",
            )?)
        } else {
            None
        };
        let binary_declarations = if let Some(cached) = &cached_artifacts {
            let nextest_args = cached.remap_cargo_args(&nextest_args);
            let nextest_args = crate::run_cargo::remap_nextest_store_output(
                &nextest_args,
                &cached.stable_workspace_root,
                &cached.stable_invocation_root,
                &harness_target_dir,
            )?;
            let cached_command = crate::run_cargo::inject_nextest_command_reuse_args(
                nextest_args,
                &cached.reuse_build_args(),
            )?;
            cmd = Command::new("cargo");
            cmd.args(cached_command);
            for (name, value) in &base_command_environment {
                cmd.env(name, value);
            }
            cached.apply_execution_context(&mut cmd)?;
            cached_scheduler_declarations(&cached.build_directory, &cached.scheduler_stamps)
        } else {
            let mut warm = Command::new("cargo");
            warm.args(crate::run_cargo::prebuild_no_run_json_args(&nextest_args));
            for (var, val) in &blob_envs {
                warm.env(var, val);
            }
            warm.env(ktstr::KTSTR_KERNEL_ENV, &resolved[0].1);
            let pinned_test_bins = crate::run_cargo::run_reserved_prebuild_collect_test_bins(
                warm,
                "cargo ktstr verifier",
                &harness_target_dir,
            )?;
            if crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire) {
                return Ok(None);
            }
            let test_bins = pinned_test_bins.probe_paths();
            let probe_provenance = pinned_test_bins.probe_provenance();
            let binary_declarations =
                probe_scheduler_declarations(&test_bins, &[], &probe_provenance)?;
            // Ownership now records Cargo's canonical emitted paths; no later
            // phase reads through the warmed executable descriptors.
            drop(pinned_test_bins);
            binary_declarations
        };
        if crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        let declarations = binary_declarations
            .iter()
            .flat_map(|binary| binary.declarations.iter().cloned())
            .collect::<Vec<_>>();
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
        let selected_plan = selected_scheduler_plan(
            &declarations,
            scheduler.as_deref(),
            &resolved_kernel_labels,
            &presets,
            &declaring_cargo_options,
        )?;
        let cell_ownership =
            plan_verifier_cell_ownership(&binary_declarations, &selected_plan.schedulers)?;
        let scheduler_manifest = prebuild_scheduler_manifest(
            &selected_plan.discover_requests,
            &selected_plan.schedulers,
            &scheduler_profile,
            SchedulerPrebuildContext {
                build_options: &scheduler_cargo_options,
                interrupted: &crate::interrupt::INTERRUPTED,
            },
        )?;
        let mut scheduler_binaries = scheduler_manifest
            .manifest
            .entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect::<Vec<_>>();
        scheduler_binaries.sort();
        scheduler_binaries.dedup();
        crate::run_cargo::precompute_cast_cache(&scheduler_binaries)?;
        if crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        let scheduler_manifest_path = ktstr::scheduler_artifact::write_scheduler_artifact_manifest(
            result_dir.path(),
            &scheduler_manifest.manifest,
        )?;
        let cell_ownership_manifest_path =
            write_cell_ownership_manifest(result_dir.path(), &cell_ownership)?;
        if crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        cmd.env(ktstr::KTSTR_SCHEDULER_MANIFEST_ENV, scheduler_manifest_path);
        cmd.env(
            ktstr::KTSTR_VERIFIER_CELL_OWNERSHIP_MANIFEST_ENV,
            cell_ownership_manifest_path,
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

        // The top-level guard and shared group runner survive Ctrl-C/SIGTERM
        // so nextest descendants and the result directory both finish
        // teardown before the signal becomes the parent outcome.
        if crate::interrupt::INTERRUPTED.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        let status = crate::nextest_process::run_status(cmd)
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
        // declared, or declaration-level topology constraints / verifier-only
        // exclusions reject every selected preset) rather than
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
    // dropped it inside the closure. The top-level interrupt owner restores
    // dispositions and re-raises only after this function returns.
    match guarded_result {
        Ok(Some(code)) => {
            crate::interrupt::defer_exit_code(code);
            Ok(())
        }
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

    #[test]
    fn recursive_verifier_cached_reuse_strips_build_surface_and_keeps_cell_filter() {
        let command = strings(&[
            "nextest",
            "run",
            "--workspace",
            "--features",
            "integration,verifier-tests",
            "--run-ignored",
            "all",
            "--test-threads=1000000",
            "-E",
            "test(/^verifier/)",
            "--",
            "--nocapture",
        ]);
        let reuse = strings(&[
            "--cargo-metadata",
            "/cache/cargo.json",
            "--binaries-metadata",
            "/cache/binaries.json",
        ]);
        assert_eq!(
            crate::run_cargo::inject_nextest_command_reuse_args(command, &reuse).unwrap(),
            strings(&[
                "nextest",
                "run",
                "--run-ignored",
                "all",
                "--test-threads=1000000",
                "-E",
                "test(/^verifier/)",
                "--cargo-metadata",
                "/cache/cargo.json",
                "--binaries-metadata",
                "/cache/binaries.json",
                "--",
                "--nocapture",
            ]),
            "cached verifier runs must not combine reuse metadata with conflicting Cargo opts",
        );
    }

    #[test]
    fn recursive_verifier_cache_miss_builds_only_the_cargo_surface() {
        assert_eq!(
            crate::run_cargo::nextest_command_build_surface(&strings(&[
                "nextest",
                "run",
                "--workspace",
                "--features",
                "integration,verifier-tests",
                "--run-ignored",
                "all",
                "--test-threads=1000000",
                "--retries",
                "2",
                "-E",
                "test(/^verifier/)",
            ]))
            .unwrap(),
            strings(&["--workspace", "--features", "integration,verifier-tests",]),
            "the elected cache producer must compile once without executing/listing cells",
        );
    }

    #[test]
    fn recursive_verifier_cached_stamp_restores_binary_owner_without_elf_probe() {
        let stamps = vec![crate::nextest_artifact_cache::CachedBinaryStamp {
            relative_binary: PathBuf::from("debug/deps/verifier_fixture-deadbeef"),
            manifest: ktstr::test_support::SchedulerManifestProbe {
                declarations: Vec::new(),
                artifact_requirements: Vec::new(),
                tests: Vec::new(),
            },
        }];
        let declarations = cached_scheduler_declarations(Path::new("/cow/build"), &stamps);
        assert_eq!(declarations.len(), 1);
        assert_eq!(
            declarations[0].executable,
            Path::new("/cow/build/debug/deps/verifier_fixture-deadbeef"),
        );
        assert!(declarations[0].declarations.is_empty());
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

    #[test]
    fn identical_multi_binary_declarations_elect_one_read_only_cell_owner() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("ownership tempdir");
        let first = dir.path().join("a-test-bin");
        let later = dir.path().join("z-test-bin");
        std::fs::write(&first, b"first").expect("write first test bin");
        std::fs::write(&later, b"later").expect("write later test bin");
        let first = std::fs::canonicalize(first).expect("canonical first test bin");
        let later = std::fs::canonicalize(later).expect("canonical later test bin");
        let declaration = discover_declaration("shared", "scx_shared", "/workspace/member");
        let mut path_declaration =
            discover_declaration("shared-path", "unused", "/workspace/member");
        path_declaration.scheduler.binary_kind =
            ktstr::test_support::BinaryKindJson::Path("/scheduler/path".into());
        let binaries = vec![
            TestBinarySchedulerDeclarations {
                executable: later,
                declarations: vec![declaration.clone(), path_declaration.clone()],
            },
            TestBinarySchedulerDeclarations {
                executable: first.clone(),
                // A repeated local registration must not create another
                // manifest owner either.
                declarations: vec![
                    declaration.clone(),
                    declaration.clone(),
                    path_declaration.clone(),
                ],
            },
        ];

        let manifest = plan_verifier_cell_ownership(
            &binaries,
            &[
                declaration.scheduler.clone(),
                path_declaration.scheduler.clone(),
            ],
        )
        .expect("elect one owner for Discover and Path declarations");
        assert_eq!(manifest.entries.len(), 2);
        assert!(
            manifest
                .entries
                .iter()
                .all(|entry| entry.executable == first),
            "owner election is canonical-path ordered for Discover and Path declarations, \
             independent of probe order",
        );

        let result_dir = dir.path().join("results");
        std::fs::create_dir(&result_dir).expect("create result dir");
        let path = write_cell_ownership_manifest(&result_dir, &manifest)
            .expect("write ownership manifest");
        let roundtrip: ktstr::verifier::VerifierCellOwnershipManifest =
            serde_json::from_slice(&std::fs::read(&path).expect("read ownership manifest"))
                .expect("parse ownership manifest");
        assert_eq!(roundtrip, manifest);
        assert_eq!(
            std::fs::metadata(path)
                .expect("ownership manifest metadata")
                .permissions()
                .mode()
                & 0o222,
            0,
            "published ownership is immutable before nextest starts",
        );
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

        let selected = selected_discover_requests(&[fixture, valid], None, &kernels, &presets, &[])
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
            &[],
        )
        .expect("non-emitting declarations do not conflict");
        assert!(selected.is_empty());
    }

    #[test]
    fn scheduler_snapshot_uses_pinned_source_and_is_read_only_executable() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("cargo-target-scheduler");
        std::fs::write(&source, b"old scheduler bytes").expect("write source");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o755))
            .expect("chmod source");
        let pinned = ktstr::cache::pin_content_file(&source).expect("pin source fd");

        let replacement = dir.path().join("replacement");
        std::fs::write(&replacement, b"new scheduler bytes").expect("write replacement");
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o755))
            .expect("chmod replacement");
        std::fs::rename(&replacement, &source).expect("replace Cargo target path atomically");

        let snapshot = ktstr::scheduler_artifact::snapshot_pinned_scheduler_artifact(pinned)
            .expect("snapshot pinned scheduler");

        assert_eq!(
            std::fs::read(snapshot.path()).expect("read snapshot"),
            b"old scheduler bytes",
            "the copy follows the pinned source inode, not a replaced target path",
        );
        let mode = std::fs::metadata(snapshot.path())
            .expect("snapshot metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o222, 0, "snapshot must be read-only");
        assert_eq!(mode & 0o111, 0o111, "snapshot must remain executable");
        assert_ne!(
            std::fs::metadata(snapshot.path())
                .expect("snapshot metadata")
                .ino(),
            std::fs::metadata(&source)
                .expect("replacement metadata")
                .ino(),
            "the manifest path must retain the pinned source revision",
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
            target_dir: PathBuf::from("/w/target"),
            packages: BTreeMap::from([("scx_a".to_string(), package_id)]),
            requests: Vec::new(),
            metadata: None,
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

    #[test]
    fn scheduler_workspace_argv_replays_only_safe_rebased_cargo_context() {
        let a: PackageId = serde_json::from_str(r#""scx_a 1.0.0 (path+file:///w/scx_a)""#).unwrap();
        let b: PackageId = serde_json::from_str(r#""scx_b 1.0.0 (path+file:///w/scx_b)""#).unwrap();
        let group = WorkspaceSchedulerBuild {
            root: PathBuf::from("/w"),
            target_dir: PathBuf::from("/w/target"),
            packages: BTreeMap::from([("scx_b".to_string(), b), ("scx_a".to_string(), a)]),
            requests: Vec::new(),
            metadata: None,
        };
        let outer = strings(&[
            "--locked",
            "--offline",
            "--config",
            "config/ci.toml",
            "--config",
            "patch.crates-io.ktstr.path='../ktstr'",
            "--target",
            "aarch64-unknown-linux-gnu",
            "--target-dir",
            "target/ci",
            "--features",
            "consumer/ktstr-tests",
            "--all-features",
            "--no-default-features",
        ]);
        let controls = scheduler_build_options(&outer, Path::new("/invoke"));
        assert_eq!(
            scheduler_workspace_build_args(&group, "release-ci", &controls),
            strings(&[
                "build",
                "--message-format=json-render-diagnostics",
                "--profile",
                "release-ci",
                "-p",
                "scx_a",
                "-p",
                "scx_b",
                "--locked",
                "--offline",
                "--config",
                "/invoke/config/ci.toml",
                "--config",
                r#"patch.crates-io.ktstr.path="/invoke/../ktstr""#,
                "--target",
                "aarch64-unknown-linux-gnu",
                "--target-dir",
                "/invoke/target/ci",
            ]),
            "consumer feature modes never leak into the parent-owned scheduler workspace",
        );
    }

    #[test]
    fn scheduler_workspace_pins_metadata_target_for_relative_cargo_target_dir() {
        let package: PackageId =
            serde_json::from_str(r#""scx_a 1.0.0 (path+file:///w/member/scx_a)""#).unwrap();
        let group = WorkspaceSchedulerBuild {
            root: PathBuf::from("/w"),
            // This is the absolute target directory Cargo metadata reports
            // when invoked from `/w/member` with CARGO_TARGET_DIR=relative.
            // Inheriting the same env from the batched `/w` build would
            // otherwise write `/w/relative`, outside the output lease.
            target_dir: PathBuf::from("/w/member/relative"),
            packages: BTreeMap::from([("scx_a".to_string(), package)]),
            requests: Vec::new(),
            metadata: None,
        };

        let (args, locked_target) = scheduler_workspace_execution(&group, "release", &[]);
        assert_eq!(locked_target, PathBuf::from("/w/member/relative"));
        assert_eq!(
            &args[args.len() - 2..],
            strings(&["--target-dir", "/w/member/relative"]).as_slice(),
            "the Cargo writer must be forced onto the exact metadata/lock path",
        );

        let (args, locked_target) = scheduler_workspace_execution(
            &group,
            "release",
            &strings(&["--target-dir=/invoke/explicit"]),
        );
        assert_eq!(locked_target, PathBuf::from("/invoke/explicit"));
        assert_eq!(
            args.iter()
                .filter(|argument| argument.starts_with("--target-dir"))
                .count(),
            1,
            "an explicit rebased target-dir remains authoritative",
        );
    }

    #[test]
    fn scheduler_build_environment_hashes_arbitrary_inputs_but_normalizes_runner_locations() {
        let workspace = Path::new("/runner/work/ktstr");
        let mut environment = vec![
            ("HOME".into(), "/runner/home".into()),
            (
                "PATH".into(),
                "/runner/home/.cargo/bin:/runner/work/ktstr/tools:/usr/bin".into(),
            ),
            ("SCHEDULER_FIXTURE_MODE".into(), "semantic-value".into()),
            ("CARGO_INCREMENTAL".into(), "1".into()),
            ("PWD".into(), "/runner/work/ktstr".into()),
            ("GITHUB_RUN_ID".into(), "123456".into()),
            ("SCCACHE_IDLE_TIMEOUT".into(), "0".into()),
            (
                "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER".into(),
                "/proc/1234/exe __ktstr_admission_runner".into(),
            ),
        ];
        environment.extend(
            CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT
                .iter()
                .map(|name| {
                    (
                        std::ffi::OsString::from(*name),
                        std::ffi::OsString::from(format!("runtime-{name}")),
                    )
                }),
        );
        environment.extend(
            CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT
                .iter()
                .map(|name| {
                    (
                        std::ffi::OsString::from(*name),
                        std::ffi::OsString::from(format!("systemd-runtime-{name}")),
                    )
                }),
        );
        environment.extend(
            CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT
                .iter()
                .map(|name| {
                    (
                        std::ffi::OsString::from(*name),
                        std::ffi::OsString::from(format!("ci-runtime-{name}")),
                    )
                }),
        );
        environment.push(("LLVM_PROFILE_FILE".into(), "/tmp/profile-%p.profraw".into()));
        environment.extend(
            SCHEDULER_BUILD_PRESERVED_OPERATIONAL_ENVIRONMENT
                .iter()
                .map(|name| {
                    (
                        std::ffi::OsString::from(*name),
                        std::ffi::OsString::from(format!("operational-{name}")),
                    )
                }),
        );
        let semantic =
            scheduler_build_environment_from(workspace, environment, &|| false).expect("env key");
        let semantic = semantic.into_iter().collect::<BTreeMap<_, _>>();

        assert_eq!(
            semantic.get(std::ffi::OsStr::new("SCHEDULER_FIXTURE_MODE")),
            Some(&b"semantic-value".to_vec()),
            "an arbitrary inherited build-script input must enter the cache key",
        );
        assert_eq!(
            semantic.get(std::ffi::OsStr::new("PATH")),
            Some(&b"$HOME/.cargo/bin:$WORKSPACE/tools:/usr/bin".to_vec()),
            "runner-specific home and checkout roots must retain stable semantic spellings",
        );
        assert_eq!(
            semantic.get(std::ffi::OsStr::new("CARGO_INCREMENTAL")),
            Some(&b"0".to_vec()),
            "the scheduler identity must describe the forced non-incremental producer",
        );
        for operational in ["PWD", "GITHUB_RUN_ID", "SCCACHE_IDLE_TIMEOUT"]
            .into_iter()
            .chain(CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT.iter().copied())
            .chain(
                CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT
                    .iter()
                    .copied(),
            )
            .chain(CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT.iter().copied())
            .chain(
                SCHEDULER_BUILD_PRESERVED_OPERATIONAL_ENVIRONMENT
                    .iter()
                    .copied(),
            )
            .chain([
                "LLVM_PROFILE_FILE",
                "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
            ])
        {
            assert!(
                !semantic.contains_key(std::ffi::OsStr::new(operational)),
                "{operational} is operational plumbing rather than a scheduler build input",
            );
        }
    }

    #[test]
    fn scheduler_identity_ignores_nextest_retry_coordinates_but_tracks_build_inputs() {
        let workspace = Path::new("/runner/work/ktstr");
        let identity = |attempt: &str, slot: &str, test_name: &str, fixture: &str| {
            scheduler_build_environment_from(
                workspace,
                vec![
                    ("NEXTEST".into(), "1".into()),
                    ("NEXTEST_ATTEMPT".into(), attempt.into()),
                    ("NEXTEST_TEST_GLOBAL_SLOT".into(), slot.into()),
                    ("NEXTEST_TEST_NAME".into(), test_name.into()),
                    ("SCHEDULER_FIXTURE_MODE".into(), fixture.into()),
                ],
                &|| false,
            )
            .expect("scheduler environment identity")
        };

        let first = identity("1", "3", "ktstr::nested_retry", "semantic-a");
        let retry = identity("7", "91", "ktstr::nested_retry/retry", "semantic-a");
        assert_eq!(
            retry, first,
            "nextest attempt, slot, and test-name coordinates must not split the Cargo cache",
        );
        assert_ne!(
            identity("7", "91", "ktstr::nested_retry/retry", "semantic-b"),
            first,
            "an arbitrary inherited build-script input must still split the Cargo cache",
        );
    }

    #[test]
    fn scheduler_parent_out_dir_is_nonsemantic() {
        let workspace = Path::new("/runner/work/ktstr");
        let identity = |out_dir: &str, fixture: &str| {
            scheduler_build_environment_from(
                workspace,
                vec![
                    ("OUT_DIR".into(), out_dir.into()),
                    ("SCHEDULER_FIXTURE_MODE".into(), fixture.into()),
                ],
                &|| false,
            )
            .expect("scheduler environment identity")
        };

        let base = identity(
            "/runner/work/ktstr/target/debug/build/ktstr-base/out",
            "semantic-a",
        );
        let wprof = identity(
            "/runner/work/ktstr/target/debug/build/ktstr-wprof/out",
            "semantic-a",
        );
        assert_eq!(
            wprof, base,
            "the parent cargo-ktstr feature build must not split scheduler artifacts",
        );
        assert_ne!(
            identity(
                "/runner/work/ktstr/target/debug/build/ktstr-wprof/out",
                "semantic-b",
            ),
            base,
            "real inherited build-script inputs must still split scheduler artifacts",
        );

        let mut command = Command::new("cargo");
        command
            .env(
                "OUT_DIR",
                "/runner/work/ktstr/target/debug/build/ktstr-wprof/out",
            )
            .env("SCHEDULER_FIXTURE_MODE", "semantic-a");
        sanitize_scheduler_build_child_environment(&mut command);
        let environment = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(std::ffi::OsStr::to_owned)))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            environment.get(std::ffi::OsStr::new("OUT_DIR")),
            Some(&None),
            "the scheduler Cargo child must not inherit cargo-ktstr's OUT_DIR",
        );
        assert_eq!(
            environment.get(std::ffi::OsStr::new("SCHEDULER_FIXTURE_MODE")),
            Some(&Some("semantic-a".into())),
            "real inherited build-script inputs must remain available to the scheduler build",
        );
    }

    #[test]
    fn scheduler_identity_ignores_service_and_ci_coordinates() {
        let workspace = Path::new("/runner/work/ktstr");
        let identity = |runner: &str| {
            let mut environment = vec![
                ("RUSTC_WRAPPER".into(), "/usr/local/bin/sccache".into()),
                ("SCHEDULER_FIXTURE_MODE".into(), "semantic".into()),
            ];
            environment.extend(
                CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT
                    .iter()
                    .map(|name| ((*name).into(), format!("/run/{runner}/{name}").into())),
            );
            environment.extend(
                CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT
                    .iter()
                    .map(|name| ((*name).into(), format!("{runner}-{name}").into())),
            );
            scheduler_build_environment_from(workspace, environment, &|| false)
                .expect("scheduler environment identity")
        };

        assert_eq!(
            identity("runner-1"),
            identity("runner-9"),
            "service and CI control-plane coordinates must not split the scheduler cache",
        );
    }

    #[test]
    fn scheduler_build_child_removes_runtime_orchestration_but_keeps_sccache() {
        let mut command = Command::new("cargo");
        command
            .env("NEXTEST", "1")
            .env("NEXTEST_ATTEMPT", "7")
            .env("NEXTEST_TEST_GLOBAL_SLOT", "41")
            .env("NEXTEST_TEST_NAME", "ktstr::nested_retry")
            .env("CARGO_INCREMENTAL", "1")
            .env("RUSTC_WRAPPER", "/usr/local/bin/sccache")
            .env("SCHEDULER_FIXTURE_MODE", "semantic");
        for &name in CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT {
            command.env(name, format!("runtime-{name}"));
        }
        for &name in CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT {
            command.env(name, format!("systemd-runtime-{name}"));
        }
        for &name in CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT {
            command.env(name, format!("ci-runtime-{name}"));
        }
        command
            .env("KTSTR_CACHE_DIR", "/var/cache/ktstr")
            .env("KTSTR_GHA_CACHE", "1")
            .env("LLVM_PROFILE_FILE", "/tmp/profile-%p.profraw");

        sanitize_scheduler_build_child_environment(&mut command);
        let environment = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(std::ffi::OsStr::to_owned)))
            .collect::<BTreeMap<_, _>>();

        for removed in CACHED_CARGO_BUILD_KTSTR_RUNTIME_ENVIRONMENT
            .iter()
            .copied()
            .chain(
                CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT
                    .iter()
                    .copied(),
            )
            .chain(CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT.iter().copied())
            .chain([
                "LLVM_PROFILE_FILE",
                "NEXTEST",
                "NEXTEST_ATTEMPT",
                "NEXTEST_TEST_GLOBAL_SLOT",
                "NEXTEST_TEST_NAME",
            ])
        {
            assert_eq!(
                environment.get(std::ffi::OsStr::new(removed)),
                Some(&None),
                "{removed} must be explicitly removed from the scheduler Cargo child",
            );
        }
        assert_eq!(
            environment.get(std::ffi::OsStr::new("KTSTR_CACHE_DIR")),
            Some(&Some("/var/cache/ktstr".into())),
            "the machine-wide content cache must remain available to scheduler build scripts",
        );
        assert_eq!(
            environment.get(std::ffi::OsStr::new("KTSTR_GHA_CACHE")),
            Some(&Some("1".into())),
            "remote-cache policy is operational but remains available to the producer",
        );
        assert_eq!(
            environment.get(std::ffi::OsStr::new("RUSTC_WRAPPER")),
            Some(&Some("/usr/local/bin/sccache".into())),
            "scheduler cache reuse must not disable the configured compiler cache",
        );
        assert_eq!(
            environment.get(std::ffi::OsStr::new("CARGO_INCREMENTAL")),
            Some(&Some("0".into())),
            "stable scheduler producers must override inherited incremental compilation",
        );
        assert_eq!(
            environment.get(std::ffi::OsStr::new("SCHEDULER_FIXTURE_MODE")),
            Some(&Some("semantic".into())),
            "arbitrary scheduler build inputs remain inherited",
        );
    }

    #[test]
    fn resolved_dep_kinds_reject_other_named_targets() {
        let dependency: cargo_metadata::NodeDep = serde_json::from_str(
            r#"{
                "name":"ktstr",
                "pkg":"ktstr 0.42.0 (path+file:///w/ktstr)",
                "dep_kinds":[{
                    "kind":null,
                    "target":"aarch64-unknown-linux-gnu"
                }]
            }"#,
        )
        .expect("target-specific NodeDep fixture");
        assert!(test_link_edge(
            &dependency,
            true,
            Some("aarch64-unknown-linux-gnu"),
        ));
        assert!(
            !test_link_edge(&dependency, true, Some("x86_64-unknown-linux-gnu"),),
            "a target-specific old/current ktstr edge cannot classify another --target",
        );
        assert!(
            test_link_edge(&dependency, true, None),
            "an unfiltered host graph remains conservative",
        );
    }

    #[test]
    fn resolved_dep_kinds_evaluate_cfg_against_the_effective_target() {
        let dependency: cargo_metadata::NodeDep = serde_json::from_str(
            r#"{
                "name":"ktstr",
                "pkg":"ktstr 0.42.0 (path+file:///w/ktstr)",
                "dep_kinds":[{
                    "kind":null,
                    "target":"cfg(target_os = \"linux\")"
                }]
            }"#,
        )
        .expect("cfg-target NodeDep fixture");
        let linux = TargetContext::named(
            "x86_64-unknown-linux-gnu",
            vec![cargo_platform::Cfg::KeyPair(
                "target_os".to_string(),
                "linux".to_string(),
            )],
        );
        let windows = TargetContext::named(
            "x86_64-pc-windows-msvc",
            vec![cargo_platform::Cfg::KeyPair(
                "target_os".to_string(),
                "windows".to_string(),
            )],
        );
        assert!(test_link_edge_for_context(
            &dependency,
            true,
            Some(&linux),
            false,
        ));
        assert!(
            !test_link_edge_for_context(&dependency, true, Some(&windows), false),
            "an opposite-target ktstr edge cannot taint the selected test closure",
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
        let plan = verifier_package_plan(
            &scx_version_fixture(),
            &Version::parse("0.41.0").unwrap(),
            None,
        )
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
    fn verifier_feature_inference_is_scoped_before_default_resolution() {
        let manifests = scx_version_fixture();
        let all =
            selected_activations(&manifests, &verifier_selection_args(&[]), VersionScope::Any);
        assert_eq!(
            all,
            vec![PackageFeatureActivation {
                package: "scx_layered".to_string(),
                features: vec!["ktstr-tests".to_string()],
            }],
            "a bare verifier intentionally considers every workspace declaration gate",
        );
        assert!(
            selected_activations(
                &manifests,
                &verifier_selection_args(&strings(&["-p", "scx_lavd"])),
                VersionScope::Any,
            )
            .is_empty(),
            "an optional gate from unselected scx_layered must not enter Default metadata",
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
        let plan = verifier_package_plan(
            &scx_version_fixture(),
            &Version::parse("0.17.0").unwrap(),
            None,
        )
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
        let forwarded = strings(&[
            "--cargo-profile",
            "release",
            "-j",
            "192",
            "--test-threads=8",
        ]);
        let run_argv = build_injected_scoped_nextest_args_with(
            Some("ci"),
            &forwarded,
            &VerifierPackagePlan {
                compatible: Vec::new(),
                older: Vec::new(),
                newer: Vec::new(),
            },
            |args| {
                crate::nextest_config::inject_with_path(
                    args,
                    std::path::Path::new("/tmp/ktstr-nextest.toml"),
                )
            },
        )
        .expect("verifier nextest tool-config injection succeeds");
        let warm_argv = crate::run_cargo::prebuild_no_run_args(&run_argv);
        assert_eq!(
            warm_argv[..warm_argv.len() - 1],
            run_argv[..],
            "warm-up must share the run's argv prefix verbatim",
        );
        assert!(
            !run_argv
                .iter()
                .any(|argument| argument == "-j" || argument == "--test-threads=8"),
            "verifier must remove forwarded nextest run-slot controls"
        );
        assert_eq!(
            run_argv
                .iter()
                .filter(|argument| argument.as_str() == "--test-threads=1000000")
                .count(),
            1,
            "verifier must use the same single admission scheduler as other routes"
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
        for argv in [&run_argv, &warm_argv] {
            assert_eq!(
                argv.iter()
                    .filter(|argument| {
                        argument.as_str() == "--tool-config-file=ktstr:/tmp/ktstr-nextest.toml"
                    })
                    .count(),
                1,
                "run and warm-up must share one identical tool config: {argv:?}",
            );
        }
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
