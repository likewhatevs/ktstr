//! `cargo ktstr affected` — which declared schedulers a `base..HEAD` diff
//! affects, as a flat JSON array of package names for a GitHub dynamic matrix
//! (`strategy.matrix.scheduler: ${{ fromJSON(...) }}` -> one job per
//! scheduler).
//!
//! ## Attribution (a union of two layers)
//!
//! A scheduler is affected if a changed path is reachable by EITHER layer:
//!
//! 1. **`.d` input set** ([`dep_info`]): the scheduler is built once and its
//!    cargo `<artifact>.d` dep-info is parsed into the exact set of source
//!    files that compiled into it -- the Rust sources, the generated BPF
//!    skeletons, and (via clang's `-M`) every `.bpf.c` / header it includes,
//!    including cross-scheduler text-includes (e.g. `scx_chaos` including
//!    `scx_p2dq`'s BPF) and shared headers under the `bpf_h -> scheds/include`
//!    symlink. File-precise, and always the fresh ground truth (uncached --
//!    see [`scheduler_input_set`]).
//!
//! 2. **cargo dep-closure** (cargo_metadata): a changed path is attributed to
//!    its owning workspace crate; the scheduler is affected if that crate is
//!    in the scheduler's transitive dependency closure. Catches shared *Rust*
//!    library changes, which rustc's per-crate `.d` does NOT list.
//!
//! The `.d` build is skipped for a pure-Rust change (every changed path is a
//! `.rs` owned by a workspace crate): Rust has no text-include, so the
//! crate-closure alone is sound and no scheduler need be built. It runs only
//! when a native (`.c`/`.h`) source or a crate-orphan path changed.
//!
//! ## Fail-safe (a false-negative is the worst outcome)
//!
//! Every uncertainty widens to RunAll, never to a skip: an unresolvable base,
//! a diff failure, a workspace-root / infra change, or ANY changed non-docs
//! path attributed to neither a scheduler `.d` nor a workspace crate. A
//! per-scheduler build/read failure marks that scheduler affected. Only a
//! strictly docs-only change yields the empty set.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow};

mod dep_info;
mod diff;

/// A declared scheduler discovered from the target repo's test binaries.
/// Keyed by scheduler name during enumeration; only the package + test count
/// are needed downstream, so the name itself is not retained.
struct SchedulerInfo {
    /// The cargo package to build; `Some` only for `Discover` schedulers
    /// (`Path`/`Eevdf`/`KernelBuiltin` have no target-repo package).
    package: Option<String>,
    /// Count of `#[ktstr_test]`s declared against this scheduler.
    test_count: usize,
}

/// Outcome of the affected computation.
#[derive(Debug, PartialEq)]
enum AffectedOutcome {
    /// Run every testable scheduler (a broad/infra change or any attribution
    /// failure -- over-run, never under-run).
    RunAll,
    /// Run nothing (a strictly docs-only change).
    Empty,
    /// Run exactly these scheduler packages.
    Subset(Vec<String>),
}

/// Entry point: print the flat JSON array of affected scheduler package names.
///
/// The array holds only cargo-PACKAGE (`Discover`) schedulers with >=1 test.
/// Package-less schedulers (`Eevdf` / `Path` / `KernelBuiltin`) have no cargo
/// package to key a matrix job on, so they are NOT in this array and cannot be
/// change-scoped -- CI must run their tests in a SEPARATE UNCONDITIONAL leg.
pub(crate) fn run(base: Option<&str>, base_ref: Option<&str>, default_branch: &str) -> Result<()> {
    let schedulers = enumerate_schedulers().context("enumerate declared schedulers")?;
    let names = compute(base, base_ref, default_branch, &schedulers)?;
    println!(
        "{}",
        serde_json::to_string(&names).context("serialize affected JSON")?
    );
    Ok(())
}

/// Every Discover-package scheduler carrying >=1 test, sorted+deduped. The
/// pre-workspace-graph fallback set (used only if `cargo metadata` fails);
/// once the graph is built, [`compute`] narrows this to workspace members.
fn package_schedulers(schedulers: &[SchedulerInfo]) -> Vec<String> {
    let mut v: Vec<String> = schedulers
        .iter()
        .filter(|s| s.test_count > 0)
        .filter_map(|s| s.package.clone())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Enumerate declared schedulers by probing the target's built test binaries
/// with `--ktstr-list-schedulers`. Deduplicates by scheduler name, summing
/// `test_count` across binaries (a scheduler may be registered in more than
/// one test binary).
fn enumerate_schedulers() -> Result<Vec<SchedulerInfo>> {
    let per_binary: Vec<Vec<ktstr::test_support::SchedulerListEntry>> =
        crate::misc::probe_collect(
            None,
            false,
            |bin| {
                let mut c = Command::new(bin);
                c.arg("--ktstr-list-schedulers");
                c
            },
            |_bin, out| {
                serde_json::from_slice::<Vec<ktstr::test_support::SchedulerListEntry>>(&out.stdout)
                    .map_err(|e| format!("parse --ktstr-list-schedulers output: {e}"))
            },
        )
        .map_err(|e| anyhow!("probe test binaries for declared schedulers: {e:?}"))?;

    let mut by_name: BTreeMap<String, SchedulerInfo> = BTreeMap::new();
    for entry in per_binary.into_iter().flatten() {
        let name = entry.scheduler.name.clone();
        let package = match entry.scheduler.binary_kind {
            ktstr::test_support::BinaryKindJson::Discover(pkg) => Some(pkg),
            _ => None,
        };
        by_name
            .entry(name)
            .and_modify(|si| si.test_count += entry.test_count)
            .or_insert(SchedulerInfo {
                package,
                test_count: entry.test_count,
            });
    }
    Ok(by_name.into_values().collect())
}

/// The core fallback + attribution engine. Returns the flat list of affected
/// scheduler packages: empty for a strictly docs-only change (or base == HEAD),
/// the full testable set for RunAll, otherwise the attributed subset.
fn compute(
    base: Option<&str>,
    base_ref: Option<&str>,
    default_branch: &str,
    schedulers: &[SchedulerInfo],
) -> Result<Vec<String>> {
    // Build the workspace graph FIRST so the RunAll set and the Subset set draw
    // from the SAME ws-filtered testable universe. A `cargo metadata` failure
    // is the one case that falls back to the unfiltered enumeration set.
    let meta = match cargo_metadata::MetadataCommand::new().exec() {
        Ok(m) => m,
        Err(_) => return Ok(package_schedulers(schedulers)),
    };
    let ws_root = PathBuf::from(meta.workspace_root.as_str());
    let repo_root = std::fs::canonicalize(&ws_root).unwrap_or(ws_root);
    let ws = WorkspaceGraph::build(&meta, &repo_root);

    // Testable = a Discover scheduler with >=1 test whose package is a
    // workspace member (buildable). A tested Discover package that is NOT a
    // member (a fixture, or a mis-declared scheduler) is DIAGNOSED to stderr,
    // never silently dropped.
    let mut testable_pkgs: Vec<String> = Vec::new();
    for s in schedulers {
        if s.test_count == 0 {
            continue;
        }
        let Some(pkg) = s.package.as_deref() else {
            continue; // package-less scheduler -- see run() doc.
        };
        if ws.is_member(pkg) {
            testable_pkgs.push(pkg.to_string());
        } else {
            eprintln!(
                "ktstr affected: declared scheduler package `{pkg}` (with tests) \
                 is not a workspace member; excluded from the matrix"
            );
        }
    }
    testable_pkgs.sort();
    testable_pkgs.dedup();
    let run_all = || testable_pkgs.clone();

    // --- Base resolution + diff; any failure -> RunAll (over-run beats a CI
    // error or a silently-empty gate). The gix workdir MUST equal the cargo
    // workspace root, or the diff paths and the repo-relative .d/crate paths
    // live in different spaces (attribution unsafe) -> RunAll. ---
    let cwd = std::env::current_dir().context("resolve current dir")?;
    let repo = match gix::discover(&cwd) {
        Ok(r) => r,
        Err(_) => return Ok(run_all()),
    };
    let workdir_matches_root = repo
        .workdir()
        .and_then(|w| std::fs::canonicalize(w).ok())
        .is_some_and(|w| w == repo_root);
    if !workdir_matches_root {
        return Ok(run_all());
    }
    let sel = crate::perf_delta::select_base(
        base,
        base_ref,
        std::env::var("GITHUB_BASE_REF").ok().as_deref(),
        default_branch,
    );
    let base_oid = match crate::perf_delta::resolve_baseline(&repo, &sel) {
        Ok(o) => o,
        Err(_) => return Ok(run_all()),
    };
    let head_oid = match repo.head_id() {
        Ok(h) => h.detach(),
        Err(_) => return Ok(run_all()),
    };
    let changed = match diff::changed_paths_committed(&repo, base_oid, head_oid) {
        Ok(c) => c,
        Err(_) => return Ok(run_all()),
    };

    // --- Cheap whole-set verdicts. ---
    if changed.is_empty() {
        return Ok(Vec::new()); // base == HEAD
    }
    if changed.iter().all(|p| is_docs_only(p)) {
        return Ok(Vec::new());
    }
    if changed.iter().any(|p| is_infra_path(p)) {
        return Ok(run_all());
    }

    // --- Layer 1 (.d): built only when a native source or a crate-orphan path
    // changed (a pure-Rust change needs only the crate-closure). ---
    let need_dot_d = changed
        .iter()
        .any(|p| is_native_source(p) || ws.owning_crate(p).is_none());
    let mut input_sets: BTreeMap<String, Option<BTreeSet<String>>> = BTreeMap::new();
    if need_dot_d {
        for pkg in &testable_pkgs {
            // Err (build/read failure) -> None -> conservatively affected.
            let set = scheduler_input_set(pkg, &repo_root, &ws).ok();
            input_sets.insert(pkg.clone(), set);
        }
    }

    Ok(match attribute(&changed, &testable_pkgs, &input_sets, &ws) {
        AffectedOutcome::RunAll => run_all(),
        AffectedOutcome::Empty => Vec::new(),
        AffectedOutcome::Subset(pkgs) => pkgs,
    })
}

/// Pure attribution: given the changed paths, the testable scheduler packages,
/// each scheduler's `.d` input set (`None` = could not compute -> conservatively
/// affected), and the workspace graph, decide the outcome. Extracted from
/// [`compute`] so the layer-1 ∪ layer-2 union and the fail-safe
/// (unattributed -> RunAll) are unit-testable without git or a build.
fn attribute(
    changed: &BTreeSet<String>,
    testable_pkgs: &[String],
    input_sets: &BTreeMap<String, Option<BTreeSet<String>>>,
    ws: &WorkspaceGraph,
) -> AffectedOutcome {
    let mut affected: BTreeSet<String> = BTreeSet::new();
    let mut unattributed = false;
    for path in changed {
        if is_docs_only(path) {
            continue;
        }
        let mut attributed = false;

        // Layer 2: owned by a workspace crate -> attributed; affected
        // schedulers are those whose dependency closure contains the owner.
        if let Some(owner) = ws.owning_crate(path) {
            attributed = true;
            for pkg in testable_pkgs {
                if ws.closure_contains(pkg, owner) {
                    affected.insert(pkg.clone());
                }
            }
        }

        // Layer 1: present in a scheduler's .d input set.
        for (pkg, set) in input_sets {
            if set.as_ref().is_some_and(|s| s.contains(path)) {
                attributed = true;
                affected.insert(pkg.clone());
            }
        }

        // A non-docs path attributed to neither a crate nor a .d has an
        // unknown blast radius.
        if !attributed {
            unattributed = true;
        }
    }

    // A scheduler whose .d could not be computed is conservatively affected --
    // never silently dropped.
    for (pkg, set) in input_sets {
        if set.is_none() {
            affected.insert(pkg.clone());
        }
    }

    if unattributed {
        return AffectedOutcome::RunAll;
    }
    let affected: Vec<String> = affected.into_iter().collect();
    if affected.is_empty() {
        AffectedOutcome::Empty
    } else {
        AffectedOutcome::Subset(affected)
    }
}

/// A scheduler's `.d` input set (canonicalized, repo-relative). Builds the
/// scheduler (a fast up-to-date check if the target dir is warm), reads its
/// dep-info, and folds in the scheduler's own manifest + build.rs paths (build
/// inputs the `.d` never lists, so a dep bump in its Cargo.toml still marks it
/// affected).
///
/// Deliberately UNCACHED: the `.d` is the only ground truth for the input set,
/// and any cache keyed on a cheaper proxy (source-path set, Cargo.lock) risks a
/// stale set that silently under-runs when a shared header gains a new include
/// -- the worst outcome. Rebuilding every run is the accepted cost.
fn scheduler_input_set(
    pkg: &str,
    repo_root: &Path,
    ws: &WorkspaceGraph,
) -> Result<BTreeSet<String>> {
    let (manifest_rel, build_rs_rel) = ws.scheduler_manifest_inputs(pkg, repo_root)?;
    let artifact = build_scheduler(pkg)?;
    let dot_d = artifact.with_extension("d");
    let contents = std::fs::read_to_string(&dot_d)
        .with_context(|| format!("read dep-info {}", dot_d.display()))?;
    let mut set: BTreeSet<String> = dep_info::parse_dep_info(&contents)
        .iter()
        .filter_map(|p| dep_info::normalize_to_repo_relative(p, repo_root))
        .collect();
    // The manifest + build.rs are build inputs the .d never lists.
    set.insert(manifest_rel);
    if let Some(b) = build_rs_rel {
        set.insert(b);
    }
    Ok(set)
}

/// Build a TARGET-repo scheduler package in the CURRENT directory and return
/// its `[[bin]]` artifact path (its `.d` dep-info sits alongside).
///
/// Deliberately NOT [`ktstr::build_and_find_binary`], which pins cargo's cwd to
/// ktstr's own manifest dir (baked via `env!("CARGO_MANIFEST_DIR")` at compile
/// time) -- wrong here, because `affected` runs INSIDE the target repo, whose
/// schedulers are what we build. Uses the SAME profile
/// (`ktstr::scheduler_profile_name`: `KTSTR_SCHEDULER_PROFILE` or the release
/// default) the scheduler-under-test is built with, so the `.d` lands in the
/// matching `target/<profile>/` and a warm build is reused rather than a
/// redundant release build.
fn build_scheduler(pkg: &str) -> Result<PathBuf> {
    let profile = ktstr::scheduler_profile_name();
    let output = Command::new("cargo")
        .args([
            "build",
            "-p",
            pkg,
            "--message-format=json",
            "--profile",
            profile.as_str(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| format!("spawn cargo build -p {pkg}"))?;
    if !output.status.success() {
        anyhow::bail!(
            "cargo build -p {pkg} failed (exit {})",
            output.status.code().unwrap_or(-1)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let is_bin = msg
            .get("target")
            .and_then(|t| t.get("kind"))
            .and_then(|k| k.as_array())
            .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("bin")));
        let is_test = msg
            .get("profile")
            .and_then(|p| p.get("test"))
            .and_then(|t| t.as_bool())
            == Some(true);
        if is_bin
            && !is_test
            && let Some(path) = msg
                .get("filenames")
                .and_then(|f| f.as_array())
                .and_then(|a| a.first())
                .and_then(|f| f.as_str())
        {
            return Ok(PathBuf::from(path));
        }
    }
    anyhow::bail!("cargo build -p {pkg} produced no [[bin]] artifact")
}

/// Workspace crate graph: member manifest dirs (repo-relative) + dependency
/// closures, all derived once from `cargo metadata`.
struct WorkspaceGraph {
    /// crate name -> repo-relative manifest directory (with a trailing `/`).
    crate_dirs: BTreeMap<String, String>,
    /// crate name -> its transitive workspace dependency closure (crate names,
    /// including itself).
    closures: BTreeMap<String, BTreeSet<String>>,
    /// crate name -> absolute manifest path.
    manifest_paths: BTreeMap<String, PathBuf>,
    member_names: BTreeSet<String>,
}

impl WorkspaceGraph {
    fn build(meta: &cargo_metadata::Metadata, repo_root: &Path) -> Self {
        let member_ids: BTreeSet<&cargo_metadata::PackageId> =
            meta.workspace_members.iter().collect();
        let mut member_names = BTreeSet::new();
        let mut crate_dirs = BTreeMap::new();
        let mut manifest_paths = BTreeMap::new();
        // id -> name, for the resolve-graph walk.
        let mut id_to_name: BTreeMap<&cargo_metadata::PackageId, String> = BTreeMap::new();

        for p in &meta.packages {
            let name = p.name.to_string();
            if member_ids.contains(&p.id) {
                member_names.insert(name.clone());
                let manifest = PathBuf::from(p.manifest_path.as_str());
                if let Some(dir) = manifest.parent() {
                    // Canonicalize so the crate dir strips against the canonical
                    // repo root. ASSUMES crate dirs are not reached via an
                    // in-repo symlink: git reports the symlink path while
                    // canonicalize resolves it, so a symlinked crate dir would
                    // break owning_crate's prefix match (not reachable in scx;
                    // a future one would need the diff paths canonicalized too).
                    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
                    if let Ok(rel) = canon.strip_prefix(repo_root) {
                        let rel = rel.to_string_lossy().into_owned();
                        // A root package strips to "" and would prefix-match
                        // EVERY path; skip it so a repo-root file falls through
                        // to unattributed -> RunAll rather than being wrongly
                        // owned by the root package.
                        if !rel.is_empty() {
                            crate_dirs.insert(name.clone(), format!("{rel}/"));
                        }
                    }
                }
                manifest_paths.insert(name.clone(), manifest);
            }
            id_to_name.insert(&p.id, name);
        }

        // Member->member dependency edges (Normal + Build; a pure dev-dep is
        // not linked into the scheduler binary). Empty dep_kinds (older cargo)
        // -> treat as Normal.
        let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        if let Some(resolve) = &meta.resolve {
            for node in &resolve.nodes {
                let Some(from) = id_to_name.get(&node.id).filter(|n| member_names.contains(*n))
                else {
                    continue;
                };
                for dep in &node.deps {
                    let Some(to) = id_to_name.get(&dep.pkg).filter(|n| member_names.contains(*n))
                    else {
                        continue;
                    };
                    let linked = dep.dep_kinds.is_empty()
                        || dep.dep_kinds.iter().any(|dk| {
                            matches!(
                                dk.kind,
                                cargo_metadata::DependencyKind::Normal
                                    | cargo_metadata::DependencyKind::Build
                            )
                        });
                    if linked {
                        edges.entry((*from).clone()).or_default().insert((*to).clone());
                    }
                }
            }
        }

        let closures = member_names
            .iter()
            .map(|name| (name.clone(), transitive_closure(name, &edges)))
            .collect();

        Self {
            crate_dirs,
            closures,
            manifest_paths,
            member_names,
        }
    }

    fn is_member(&self, pkg: &str) -> bool {
        self.member_names.contains(pkg)
    }

    /// The workspace crate that owns `path` (longest repo-relative
    /// manifest-dir prefix wins, so a crate nested under another is
    /// attributed to the inner one).
    fn owning_crate(&self, path: &str) -> Option<&str> {
        self.crate_dirs
            .iter()
            .filter(|(_, dir)| path.starts_with(dir.as_str()))
            .max_by_key(|(_, dir)| dir.len())
            .map(|(name, _)| name.as_str())
    }

    fn closure_contains(&self, scheduler: &str, crate_name: &str) -> bool {
        self.closures
            .get(scheduler)
            .is_some_and(|c| c.contains(crate_name))
    }

    /// (manifest repo-rel, build.rs repo-rel) for the scheduler -- build inputs
    /// the `.d` never lists, folded into the scheduler's input set so a change
    /// to its Cargo.toml / build.rs marks it affected.
    fn scheduler_manifest_inputs(
        &self,
        pkg: &str,
        repo_root: &Path,
    ) -> Result<(String, Option<String>)> {
        let manifest = self
            .manifest_paths
            .get(pkg)
            .ok_or_else(|| anyhow!("{pkg} not in workspace metadata"))?;
        let dir = manifest
            .parent()
            .ok_or_else(|| anyhow!("{pkg} manifest has no parent dir"))?;
        let build_rs_path = dir.join("build.rs");
        let manifest_rel = dep_info::normalize_to_repo_relative(manifest, repo_root)
            .ok_or_else(|| anyhow!("{pkg} manifest is outside the repo"))?;
        let build_rs_rel = if build_rs_path.exists() {
            dep_info::normalize_to_repo_relative(&build_rs_path, repo_root)
        } else {
            None
        };
        Ok((manifest_rel, build_rs_rel))
    }
}

/// Transitive closure of `start` over `edges` (includes `start`).
fn transitive_closure(
    start: &str,
    edges: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start.to_string());
    while let Some(n) = queue.pop_front() {
        if !seen.insert(n.clone()) {
            continue;
        }
        if let Some(next) = edges.get(&n) {
            for m in next {
                if !seen.contains(m) {
                    queue.push_back(m.clone());
                }
            }
        }
    }
    seen
}

/// A change confined to documentation: a `.md` file, or anything under a
/// top-level `doc`/`docs` directory. STRICT (matched with `.all()` over the
/// full changed set) -- `src/docs_render.rs` is NOT docs.
fn is_docs_only(path: &str) -> bool {
    path.ends_with(".md")
        || matches!(path.split('/').next(), Some("doc" | "docs"))
}

/// A change to workspace-root / build-graph infrastructure whose blast radius
/// spans every scheduler -> RunAll. The kernel tree (`../linux`) lives OUTSIDE
/// the target repo and cannot appear here; a kernel bump must force RunAll
/// out-of-band in CI. `ktstr.kconfig` (in-repo) IS detectable.
fn is_infra_path(path: &str) -> bool {
    matches!(
        path,
        "Cargo.toml" | "Cargo.lock" | "rust-toolchain.toml" | "rust-toolchain" | "ktstr.kconfig"
    ) || path.starts_with(".cargo/")
}

/// A native (C / BPF) source or header -- a change to one may be text-included
/// by a scheduler other than its owner, so the `.d` layer must run.
fn is_native_source(path: &str) -> bool {
    path.ends_with(".c") || path.ends_with(".h")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_only_classification() {
        assert!(is_docs_only("README.md"));
        assert!(is_docs_only("docs/guide/x.md"));
        assert!(is_docs_only("doc/notes.txt"));
        // Not docs: a source file whose name merely contains "docs".
        assert!(!is_docs_only("src/docs_render.rs"));
        assert!(!is_docs_only("scheds/rust/scx_x/src/main.rs"));
    }

    #[test]
    fn infra_classification() {
        assert!(is_infra_path("Cargo.lock"));
        assert!(is_infra_path("Cargo.toml"));
        assert!(is_infra_path("rust-toolchain.toml"));
        assert!(is_infra_path("ktstr.kconfig"));
        assert!(is_infra_path(".cargo/config.toml"));
        // A per-crate Cargo.toml is NOT the root infra file.
        assert!(!is_infra_path("scheds/rust/scx_x/Cargo.toml"));
    }

    #[test]
    fn native_source_classification() {
        assert!(is_native_source("scheds/rust/scx_x/src/bpf/main.bpf.c"));
        assert!(is_native_source("scheds/include/scx/common.bpf.h"));
        assert!(!is_native_source("scheds/rust/scx_x/src/main.rs"));
    }

    fn graph() -> WorkspaceGraph {
        let mut crate_dirs = BTreeMap::new();
        crate_dirs.insert("scx_a".to_string(), "scheds/rust/scx_a/".to_string());
        crate_dirs.insert("scx_b".to_string(), "scheds/rust/scx_b/".to_string());
        crate_dirs.insert("scx_common".to_string(), "rust/scx_common/".to_string());
        let mut closures = BTreeMap::new();
        closures.insert(
            "scx_a".to_string(),
            ["scx_a", "scx_common"].iter().map(|s| s.to_string()).collect(),
        );
        closures.insert("scx_b".to_string(), ["scx_b"].iter().map(|s| s.to_string()).collect());
        closures.insert(
            "scx_common".to_string(),
            ["scx_common"].iter().map(|s| s.to_string()).collect(),
        );
        WorkspaceGraph {
            crate_dirs,
            closures,
            manifest_paths: BTreeMap::new(),
            member_names: ["scx_a", "scx_b", "scx_common"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    #[test]
    fn owning_crate_longest_prefix() {
        let g = graph();
        assert_eq!(g.owning_crate("scheds/rust/scx_a/src/main.rs"), Some("scx_a"));
        assert_eq!(g.owning_crate("rust/scx_common/src/lib.rs"), Some("scx_common"));
        // A shared header outside every crate dir has no owner (the .d layer
        // attributes it instead).
        assert_eq!(g.owning_crate("scheds/include/scx/common.bpf.h"), None);
    }

    #[test]
    fn closure_reflects_dependency_edges() {
        let g = graph();
        // scx_a depends on scx_common; scx_b does not.
        assert!(g.closure_contains("scx_a", "scx_common"));
        assert!(!g.closure_contains("scx_b", "scx_common"));
        assert!(g.closure_contains("scx_a", "scx_a"));
    }

    #[test]
    fn transitive_closure_follows_chain() {
        let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        edges.insert("a".into(), ["b"].iter().map(|s| s.to_string()).collect());
        edges.insert("b".into(), ["c"].iter().map(|s| s.to_string()).collect());
        let c = transitive_closure("a", &edges);
        assert_eq!(
            c,
            ["a", "b", "c"].iter().map(|s| s.to_string()).collect::<BTreeSet<_>>()
        );
    }

    fn changed(paths: &[&str]) -> BTreeSet<String> {
        paths.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn attribute_docs_only_is_empty() {
        let g = graph();
        let pkgs = vec!["scx_a".to_string()];
        let out = attribute(
            &changed(&["README.md", "docs/x.md"]),
            &pkgs,
            &BTreeMap::new(),
            &g,
        );
        assert_eq!(out, AffectedOutcome::Empty);
    }

    #[test]
    fn attribute_layer2_closure_hit() {
        // A pure-Rust change to scx_common (no .d) affects scx_a (depends on
        // it) but not scx_b.
        let g = graph();
        let pkgs = vec!["scx_a".to_string(), "scx_b".to_string()];
        let out = attribute(
            &changed(&["rust/scx_common/src/lib.rs"]),
            &pkgs,
            &BTreeMap::new(),
            &g,
        );
        assert_eq!(out, AffectedOutcome::Subset(vec!["scx_a".to_string()]));
    }

    #[test]
    fn attribute_layer1_shared_header_via_dot_d() {
        // A shared BPF header owned by NO crate, but present in scx_a's .d,
        // affects scx_a alone -- attributed, no run_all.
        let g = graph();
        let pkgs = vec!["scx_a".to_string(), "scx_b".to_string()];
        let mut input_sets = BTreeMap::new();
        input_sets.insert(
            "scx_a".to_string(),
            Some(changed(&["scheds/include/scx/common.bpf.h"])),
        );
        input_sets.insert("scx_b".to_string(), Some(BTreeSet::new()));
        let out = attribute(
            &changed(&["scheds/include/scx/common.bpf.h"]),
            &pkgs,
            &input_sets,
            &g,
        );
        assert_eq!(out, AffectedOutcome::Subset(vec!["scx_a".to_string()]));
    }

    #[test]
    fn attribute_unattributed_non_docs_is_runall() {
        // A non-docs path owned by no crate and in no .d -> unknown blast
        // radius -> RunAll (the fail-safe).
        let g = graph();
        let pkgs = vec!["scx_a".to_string()];
        let mut input_sets = BTreeMap::new();
        input_sets.insert("scx_a".to_string(), Some(BTreeSet::new()));
        let out = attribute(&changed(&["tools/random_script.sh"]), &pkgs, &input_sets, &g);
        assert_eq!(out, AffectedOutcome::RunAll);
    }

    #[test]
    fn attribute_uncomputable_dot_d_is_conservatively_affected() {
        // A scheduler whose .d could not be computed (None) is affected
        // regardless -- never silently dropped.
        let g = graph();
        let pkgs = vec!["scx_a".to_string(), "scx_b".to_string()];
        let mut input_sets = BTreeMap::new();
        input_sets.insert("scx_a".to_string(), None); // build/read failed
        input_sets.insert("scx_b".to_string(), Some(BTreeSet::new()));
        // A change owned by scx_b's crate keeps the path attributed.
        let out = attribute(
            &changed(&["scheds/rust/scx_b/src/bpf/main.bpf.c"]),
            &pkgs,
            &input_sets,
            &g,
        );
        assert_eq!(
            out,
            AffectedOutcome::Subset(vec!["scx_a".to_string(), "scx_b".to_string()])
        );
    }

    /// changed_paths_committed captures BOTH the source and destination of a
    /// rename (the location + source_location union) -- else a scheduler whose
    /// `.d` referenced the pre-rename path is wrongly judged unaffected. Shells
    /// `git` to build the fixture (same dependency perf-delta's tests use).
    #[test]
    fn changed_paths_committed_captures_rename_source_and_dest() {
        use std::process::Command;
        let dir = std::env::temp_dir().join(format!("ktstr-aff-gitfix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mk tempdir");
        let run = |args: &[&str]| {
            let ok = Command::new("git")
                .current_dir(&dir)
                .args([
                    "-c",
                    "user.email=t@example.invalid",
                    "-c",
                    "user.name=t",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            assert!(ok, "git {args:?} failed");
        };
        run(&["init", "-q"]);
        run(&["config", "diff.renames", "true"]);
        std::fs::write(dir.join("a.rs"), "fn a() {}\n").expect("write a.rs");
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "first"]);
        run(&["mv", "a.rs", "b.rs"]);
        run(&["commit", "-q", "-am", "rename a.rs -> b.rs"]);

        let repo = gix::discover(&dir).expect("discover temp repo");
        let oid = |spec: &str| -> gix::ObjectId {
            match repo.rev_parse(spec).expect("rev-parse").detach() {
                gix::revision::plumbing::Spec::Include(id)
                | gix::revision::plumbing::Spec::ExcludeParents(id) => id,
                other => panic!("{spec} did not resolve to a commit: {other:?}"),
            }
        };
        let changed =
            diff::changed_paths_committed(&repo, oid("HEAD~1"), oid("HEAD")).expect("changed paths");
        assert!(changed.contains("a.rs"), "rename source captured: {changed:?}");
        assert!(changed.contains("b.rs"), "rename dest captured: {changed:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// WorkspaceGraph::build derives crate dirs + dependency closures from real
    /// `cargo metadata`: a virtual root contributes no `""` crate_dir (so a
    /// top-level file is unowned), owning_crate maps member paths, and the
    /// closure reflects the `user -> dep` edge.
    #[test]
    fn workspace_graph_from_metadata() {
        let dir = std::env::temp_dir().join(format!("ktstr-aff-wsfix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("dep/src")).unwrap();
        std::fs::create_dir_all(dir.join("user/src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers = [\"dep\", \"user\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("dep/Cargo.toml"),
            "[package]\nname = \"dep\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("dep/src/lib.rs"), "").unwrap();
        std::fs::write(
            dir.join("user/Cargo.toml"),
            "[package]\nname = \"user\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\
             [dependencies]\ndep = { path = \"../dep\" }\n",
        )
        .unwrap();
        std::fs::write(dir.join("user/src/lib.rs"), "").unwrap();

        let repo_root = std::fs::canonicalize(&dir).unwrap();
        let meta = cargo_metadata::MetadataCommand::new()
            .manifest_path(dir.join("Cargo.toml"))
            .exec()
            .expect("cargo metadata");
        let ws = WorkspaceGraph::build(&meta, &repo_root);

        assert!(ws.is_member("dep") && ws.is_member("user"));
        assert_eq!(ws.owning_crate("dep/src/lib.rs"), Some("dep"));
        assert_eq!(ws.owning_crate("user/src/lib.rs"), Some("user"));
        // A virtual-workspace root contributes no "" crate_dir, so a top-level
        // file is UNOWNED (falls through to unattributed -> RunAll).
        assert_eq!(ws.owning_crate("README.md"), None);
        assert!(ws.closure_contains("user", "dep"));
        assert!(!ws.closure_contains("dep", "user"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
