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
//! `declare_scheduler!` verifier cells carry no
//! `required-features`, so they build without a feature flag — no
//! `--features` passthrough is needed to collect verifier statistics.
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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Command;

use cargo_metadata::semver::Version;
use cargo_metadata::{CargoOpt, Metadata, MetadataCommand, PackageId};

use crate::kernel::{
    encode_kernel_list, path_kernel_label, resolve_kernel_image, resolve_kernel_set,
};

/// Sweep verifier result dirs orphaned by interrupted prior runs.
///
/// The result dir is keyed on the dispatcher pid, so a run killed before
/// its post-run `remove_dir_all` (Ctrl-C / SIGKILL / crash) orphans its
/// dir — and nothing reclaims it later, since the per-run wipe only
/// targets the CURRENT pid. This runs at startup and removes every
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

/// Cargo options which affect dependency resolution and are safe to replay on
/// the verifier's metadata preflight. Feature selection is intentionally not
/// copied: the preflight always uses all features so optional ktstr test edges
/// (such as scx's `ktstr-tests`) are visible before nextest builds anything.
fn metadata_passthrough_options(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if matches!(arg.as_str(), "--locked" | "--offline" | "--frozen") {
            out.push(arg.clone());
        } else if matches!(arg.as_str(), "--config" | "--manifest-path") {
            out.push(arg.clone());
            if let Some(value) = it.next() {
                out.push(value.clone());
            }
        } else if arg.starts_with("--config=") || arg.starts_with("--manifest-path=") {
            out.push(arg.clone());
        }
    }
    out
}

/// Resolve the verifier package partition before either nextest warm-up or run.
///
/// `cargo_path("cargo")` is load-bearing for local development: unlike
/// cargo_metadata's `$CARGO` default, it honors the PATH cargo wrapper used to
/// patch crates.io ktstr to a checkout.
fn query_verifier_package_plan(args: &[String]) -> Result<VerifierPackagePlan, String> {
    let cli = Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("cargo-ktstr's own CARGO_PKG_VERSION is valid semver");
    let mut command = MetadataCommand::new();
    command
        .cargo_path("cargo")
        .features(CargoOpt::AllFeatures)
        .other_options(metadata_passthrough_options(args));
    let metadata = command
        .exec()
        .map_err(|e| format!("cargo ktstr verifier: cargo metadata failed: {e}"))?;
    verifier_package_plan(&metadata, &cli)
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

/// Recover a Cargo package name from the common `-p` package-id spellings.
///
/// Exact names and `name@version` cover normal verifier use. A full package ID
/// (`registry+...#name@version` / `path+...#name@version`) is reduced to its
/// fragment. Anything with Cargo package-spec syntax this small parser cannot
/// prove is left unresolved, causing selection filtering to stay conservative.
fn package_spec_name(spec: &str) -> Option<&str> {
    let tail = spec.rsplit_once('#').map_or(spec, |(_, tail)| tail);
    let name = tail.split(['@', ':']).next()?;
    (!name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')))
    .then_some(name)
}

/// Exact package names requested by `-p` / `--package`, or `None` for the
/// unscoped / `--workspace` case (and for package-id syntax we cannot prove).
fn explicit_package_selection(args: &[String]) -> Option<HashSet<String>> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--workspace" | "--all"))
    {
        return None;
    }
    let mut selected = HashSet::new();
    let mut saw_package = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let spec = if matches!(arg.as_str(), "-p" | "--package") {
            saw_package = true;
            index += 1;
            args.get(index).map(String::as_str)
        } else if let Some(spec) = arg.strip_prefix("--package=") {
            saw_package = true;
            Some(spec)
        } else if let Some(spec) = arg.strip_prefix("-p")
            && !spec.is_empty()
        {
            saw_package = true;
            Some(spec)
        } else {
            None
        };
        if let Some(spec) = spec {
            let name = package_spec_name(spec)?;
            selected.insert(name.to_string());
        } else if saw_package && matches!(arg.as_str(), "-p" | "--package") {
            // A missing selector value is invalid Cargo syntax. Do not infer a
            // narrower compatibility scope from it.
            return None;
        }
        index += 1;
    }
    saw_package.then_some(selected)
}

fn explicit_package_exclusions(args: &[String]) -> HashSet<String> {
    let mut excluded = HashSet::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let spec = if arg == "--exclude" {
            index += 1;
            args.get(index).map(String::as_str)
        } else {
            arg.strip_prefix("--exclude=")
        };
        if let Some(name) = spec.and_then(package_spec_name) {
            excluded.insert(name.to_string());
        }
        index += 1;
    }
    excluded
}

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
fn drop_older_package_selectors(args: &[String], older: &[OlderVerifierPackage]) -> Vec<String> {
    let old_names: HashSet<&str> = older.iter().map(|package| package.name.as_str()).collect();
    let mut out = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
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

fn has_package_selector(args: &[String]) -> bool {
    args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "-p" | "--package" | "--workspace" | "--all" | "--exclude"
        ) || arg.starts_with("--package=")
            || arg.starts_with("--exclude=")
            || (arg.starts_with("-p") && arg.len() > 2)
    })
}

fn has_workspace_selector(args: &[String]) -> bool {
    args.iter()
        .any(|arg| matches!(arg.as_str(), "--workspace" | "--all"))
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
    }
    args.extend(rest);
    args
}

/// Dispatch the `cargo ktstr verifier` subcommand.
///
/// The trailing `args` are forwarded verbatim to the inner
/// `cargo nextest run` (a nextest filterset, `--cargo-profile`, ...).
/// The `declare_scheduler!` verifier cells carry no `required-features`,
/// so they build without a feature flag — no `--features` passthrough is
/// needed for the cell-only filter to match and collect verifier
/// statistics.
///
/// `profile` is the scheduler-under-test's cargo BUILD profile
/// (`--profile <NAME>`): set as `KTSTR_SCHEDULER_PROFILE` so
/// [`ktstr::build_and_find_binary`] passes `cargo build -p <scheduler>
/// --profile <name>`. Omitted, the scheduler builds `release` (that
/// default lives in `build_and_find_binary`). `nextest_profile` is the
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
    let package_plan =
        restrict_plan_to_explicit_selection(query_verifier_package_plan(&args)?, &args);
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
    let args = drop_older_package_selectors(&args, &package_plan.older);
    let nextest_args = build_scoped_nextest_args(nextest_profile.as_deref(), &args, &package_plan);

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

    // `--profile <NAME>` sets the scheduler-under-test's cargo BUILD
    // profile via `KTSTR_SCHEDULER_PROFILE`; absent, `build_and_find_binary`
    // defaults it to `release`.
    if let Some(p) = &profile {
        cmd.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, p);
    }

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
    // combined run below, whose cells take their own reservations
    // (the in-cell `build_and_find_binary` scheduler build is test-runtime
    // work those reservations already cover). The verifier sweep is the
    // primary colocated-CI workload, so this is the path where an
    // unreserved harness compile would invade a peer runner's perf-mode
    // reservation. Cache parity with the combined run: identical nextest
    // argv (+`--no-run`), same `blob_envs`, and `KTSTR_KERNEL` (the only
    // kernel-resolution env `build.rs` fingerprints via
    // rerun-if-env-changed — KTSTR_KERNEL_LIST and the KTSTR_VERIFIER_*
    // vars are runtime-only). No BTF-anchor `BPF_EXTRA_CFLAGS_PRE_INCL`
    // handling: the verifier dispatcher injects none, so warm-up and run
    // inherit the identical process env.
    let mut warm = Command::new("cargo");
    warm.args(crate::run_cargo::prebuild_no_run_args(&nextest_args));
    for (var, val) in &blob_envs {
        warm.env(var, val);
    }
    warm.env(ktstr::KTSTR_KERNEL_ENV, &resolved[0].1);
    crate::run_cargo::run_reserved_prebuild(warm, "cargo ktstr verifier")?;

    // Per-cell result dir: each verifier cell writes its PASS/FAIL record
    // here (via KTSTR_VERIFIER_RESULT_DIR), and after nextest returns we
    // read them back to render the summary table. Unique per dispatcher pid
    // so concurrent `cargo ktstr verifier` runs don't cross-read. First
    // sweep dirs orphaned by dead-pid prior runs (an interrupted run skips
    // the post-run wipe below), then wipe our own pid's dir in case a prior
    // run reused this pid.
    let temp_root = std::env::temp_dir();
    sweep_stale_result_dirs(&temp_root);
    let result_dir = temp_root.join(format!("ktstr-verifier-results-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&result_dir);
    if let Err(e) = std::fs::create_dir_all(&result_dir) {
        return Err(format!(
            "create verifier result dir {}: {e}",
            result_dir.display()
        ));
    }
    cmd.env(ktstr::KTSTR_VERIFIER_RESULT_DIR_ENV, &result_dir);

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

    // Survive Ctrl-C / SIGTERM so the result-dir cleanup below runs; nextest
    // tears down its own test children (see crate::interrupt).
    let interrupt_guard = crate::interrupt::InterruptGuard::install();
    let status = cmd
        .status()
        .map_err(|e| format!("spawn cargo nextest run: {e}"))?;

    // From the records each cell wrote into `result_dir`: print the
    // per-scheduler verified_insns tables first, then the per-scheduler
    // topology × kernel PASS/FAIL grids LAST so the operator's final view
    // is the pass/fail matrix. Both print on success AND failure so
    // failing cells stay visible. Best-effort: no records (e.g. 0 cells
    // ran) -> the renderers return None and nothing prints.
    let records = ktstr::verifier::read_cell_records(&result_dir);
    if let Some(tables) = ktstr::verifier::render_instruction_count_tables(&records) {
        print!("{tables}");
    }
    if let Some(table) = ktstr::verifier::render_result_table(&records) {
        print!("{table}");
    }
    let _ = std::fs::remove_dir_all(&result_dir);
    // Cleanup done; if interrupted, propagate as 128+signal now.
    let caught = interrupt_guard.interrupted();
    drop(interrupt_guard);
    if let Some(sig) = caught {
        crate::interrupt::reraise(sig);
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
        ktstr::verifier::RunOutcome::Success => Ok(()),
        ktstr::verifier::RunOutcome::Failed(msg) => Err(msg),
        // Report + cleanup already ran above; exit silently with nextest's
        // own code.
        ktstr::verifier::RunOutcome::SilentExit(code) => std::process::exit(code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn package_json(name: &str, version: &str, id: &str) -> String {
        format!(
            r#"{{"name":"{name}","version":"{version}","id":"{id}","source":null,"description":null,"dependencies":[],"license":null,"license_file":null,"targets":[],"features":{{}},"manifest_path":"/w/{name}/Cargo.toml","readme":null,"repository":null,"homepage":null,"documentation":null,"links":null,"publish":null,"default_run":null}}"#
        )
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
            layered_pkg = package_json("scx_layered", "1.0.0", layered),
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
                },
                CompatibleVerifierPackage {
                    name: "scx_layered".to_string(),
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
                },
                CompatibleVerifierPackage {
                    name: "scx_layered".to_string(),
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
    fn unscoped_verifier_selects_only_current_packages() {
        let args = build_scoped_nextest_args(
            None,
            &strings(&["--features", "ktstr-tests"]),
            &scoped_plan(),
        );
        assert!(
            args.windows(2).any(|pair| pair == ["-p", "scx_cosmos"]),
            "{args:?}",
        );
        assert!(
            args.windows(2).any(|pair| pair == ["-p", "scx_layered"]),
            "{args:?}",
        );
        assert!(!args.iter().any(|arg| arg == "scx_mitosis"), "{args:?}");
        assert!(args.iter().any(|arg| {
            arg == "(test(/^verifier/) & !test(/^verifier::/)) & \
                    (package(=scx_cosmos) | package(=scx_layered))"
        }));
    }

    #[test]
    fn explicit_package_selection_is_not_widened() {
        let args = build_scoped_nextest_args(
            None,
            &strings(&["-p", "scx_layered", "--features", "ktstr-tests"]),
            &scoped_plan(),
        );
        let selected = args
            .windows(2)
            .filter(|pair| pair[0] == "-p")
            .map(|pair| pair[1].as_str())
            .collect::<Vec<_>>();
        assert_eq!(selected, vec!["scx_layered"]);
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
        assert_eq!(selected.compatible.len(), 2);
    }

    #[test]
    fn workspace_selection_excludes_old_package_from_compile() {
        let args = build_scoped_nextest_args(
            None,
            &strings(&["--workspace", "--features", "ktstr-tests"]),
            &scoped_plan(),
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--exclude", "scx_mitosis"]),
            "{args:?}",
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
        assert!(filter.contains("package(=scx_cosmos) | package(=scx_layered)"));
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
            metadata_passthrough_options(&strings(&[
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
}
