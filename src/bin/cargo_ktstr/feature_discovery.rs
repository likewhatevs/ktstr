//! Targeted Cargo feature discovery for downstream ktstr test binaries.
//!
//! A consumer commonly declares ktstr as an optional dependency and gates its
//! test registry behind a feature such as `ktstr-tests = ["dep:ktstr"]`.
//! Cargo metadata exposes both the optional dependency declaration and the
//! complete feature graph even when that feature is disabled. This module
//! follows only ktstr-specific feature chains and emits package-qualified
//! selectors, avoiding a broad `--all-features` build.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::process::Command;
use std::str::FromStr;

use cargo_metadata::semver::{Version, VersionReq};
use cargo_metadata::{Metadata, MetadataCommand};
use cargo_platform::{Cfg, Platform};
use glob::Pattern;

/// How much dependency resolution a metadata caller needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetadataMode {
    /// Workspace manifests only. This is sufficient for ordinary feature
    /// inference and avoids resolving every optional dependency.
    NoDeps,
    /// Cargo's normal, requested-feature resolve graph. `test` and `coverage`
    /// (including raw `llvm-cov nextest`) use this for their version guard;
    /// when inference adds a previously inactive optional ktstr, they resolve
    /// the targeted result once more.
    Default,
}

/// Which ktstr dependency versions are eligible for automatic activation.
#[derive(Clone, Copy, Debug)]
pub(crate) enum VersionScope<'a> {
    /// Ordinary test commands can run a consumer's own ktstr version.
    Any,
    /// The verifier dispatcher can enumerate declarations only from the ktstr
    /// version linked into this cargo-ktstr binary.
    Matches(&'a Version),
}

/// One selected workspace package and the minimal feature roots that activate
/// only its optional ktstr dependency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackageFeatureActivation {
    pub(crate) package: String,
    pub(crate) features: Vec<String>,
}

/// Cargo's effective compilation target and the cfg set rustc reports for it.
///
/// Cargo's metadata graph is otherwise deliberately target-agnostic. Keeping
/// this context explicit lets manifest feature inference and resolved-graph
/// version classification apply the same target-table decision.
#[derive(Clone, Debug)]
pub(crate) struct TargetContext {
    name: String,
    cfg: Vec<Cfg>,
    metadata_filter: Option<String>,
}

impl TargetContext {
    pub(crate) fn named(name: impl Into<String>, cfg: Vec<Cfg>) -> Self {
        let name = name.into();
        Self {
            metadata_filter: Some(name.clone()),
            name,
            cfg,
        }
    }

    fn custom(name: impl Into<String>, cfg: Vec<Cfg>) -> Self {
        Self {
            name: name.into(),
            cfg,
            metadata_filter: None,
        }
    }

    pub(crate) fn matches_platform(&self, platform: &Platform) -> bool {
        platform.matches(&self.name, &self.cfg)
    }

    #[cfg(test)]
    pub(crate) fn named_for_test(name: impl Into<String>) -> Self {
        Self::named(name, Vec::new())
    }
}

/// The explicitly requested Cargo compilation target, if any.
///
/// Stop at `--`, after which a `--target` token belongs to the test binary.
/// Named target triples are reused both for Cargo metadata's
/// `--filter-platform` and for target-aware dependency-edge classification.
/// Custom target JSON paths remain available to scheduler builds but cannot
/// safely be interpreted as cargo-platform expressions.
pub(crate) fn requested_target(args: &[String]) -> Option<String> {
    let args = cargo_args(args);
    let mut requested = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--target" {
            index += 1;
            requested = args.get(index).cloned();
        } else if let Some(target) = arg.strip_prefix("--target=")
            && !target.is_empty()
        {
            requested = Some(target.to_string());
        }
        index += 1;
    }
    requested
}

fn rustc_program() -> std::ffi::OsString {
    std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into())
}

fn rustc_stdout(arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(rustc_program())
        .args(arguments)
        .output()
        .map_err(|error| format!("run rustc {}: {error}", arguments.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "rustc {} failed ({}): {}",
            arguments.join(" "),
            output
                .status
                .code()
                .map_or("signal".to_string(), |code| code.to_string()),
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| {
        format!(
            "rustc {} emitted non-UTF-8 output: {error}",
            arguments.join(" ")
        )
    })
}

fn rustc_host_target() -> Result<String, String> {
    let verbose = rustc_stdout(&["-vV"])?;
    verbose
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(ToString::to_string))
        .filter(|host| !host.is_empty())
        .ok_or_else(|| "rustc -vV omitted its host target".to_string())
}

fn rustc_target_cfg(target: Option<&str>) -> Result<Vec<Cfg>, String> {
    let output = match target {
        Some(target) => rustc_stdout(&["--print", "cfg", "--target", target])?,
        None => rustc_stdout(&["--print", "cfg"])?,
    };
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            Cfg::from_str(line).map_err(|error| format!("parse rustc target cfg {line:?}: {error}"))
        })
        .collect()
}

fn custom_target_name(target: &str) -> Result<String, String> {
    Path::new(target)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| format!("custom target path has no UTF-8 file stem: {target:?}"))
}

/// Derive the target Cargo will compile tests for.
///
/// An omitted `--target` means rustc's host target, not an unfiltered union of
/// every target table in the manifest. For a custom JSON target Cargo metadata
/// cannot accept the path as `--filter-platform`, but rustc can still report
/// its cfg set and the manual dependency-edge matcher remains exact.
pub(crate) fn effective_target_context(args: &[String]) -> Result<TargetContext, String> {
    match requested_target(args) {
        Some(target) => {
            let cfg = rustc_target_cfg(Some(&target))?;
            if named_filter_platform(&target) {
                Ok(TargetContext::named(target, cfg))
            } else {
                // Cargo identifies a custom target by the JSON file stem for
                // exact `[target.<name>]` tables; the path is only rustc's
                // input locator.
                Ok(TargetContext::custom(custom_target_name(&target)?, cfg))
            }
        }
        None => {
            let host = rustc_host_target()?;
            let cfg = rustc_target_cfg(None)?;
            Ok(TargetContext::named(host, cfg))
        }
    }
}

/// The Cargo arguments relevant to a top-level metadata preflight.
///
/// Feature selection is deliberately omitted: package manifests expose every
/// feature definition without activating it. Stop at `--`, after which tokens
/// belong to the test binary rather than Cargo. Cargo metadata names the
/// compilation-target option `--filter-platform`, so an explicit Cargo
/// `--target` is translated rather than forwarded verbatim.
pub(crate) fn metadata_passthrough_options(args: &[String]) -> Vec<String> {
    metadata_context_options(args, true, None)
}

/// Cargo context for metadata run from a declaration's own manifest directory.
///
/// The caller deliberately supplies that workspace through `current_dir`, so
/// forwarding the outer command's `--manifest-path` would point Cargo back at
/// the wrong workspace. Resolution controls and target filtering still apply.
pub(crate) fn declaring_metadata_options(args: &[String], invocation_dir: &Path) -> Vec<String> {
    metadata_context_options(args, false, Some(invocation_dir))
}

fn metadata_context_options(
    args: &[String],
    include_manifest_path: bool,
    rebase_from: Option<&Path>,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            break;
        }
        if matches!(arg.as_str(), "--locked" | "--offline" | "--frozen") {
            out.push(arg.clone());
        } else if arg == "--config" {
            out.push(arg.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                out.push(rebase_config_value(value, rebase_from));
            }
        } else if arg == "--manifest-path" {
            index += 1;
            if include_manifest_path && let Some(value) = args.get(index) {
                out.push(arg.clone());
                out.push(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix("--config=") {
            out.push(format!(
                "--config={}",
                rebase_config_value(value, rebase_from)
            ));
        } else if arg == "-Z" {
            out.push(arg.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                out.push(value.clone());
            }
        } else if (arg.starts_with("-Z") && arg.len() > 2)
            || (include_manifest_path && arg.starts_with("--manifest-path="))
        {
            out.push(arg.clone());
        } else if arg == "--target" {
            index += 1;
            if let Some(value) = args.get(index)
                && named_filter_platform(value)
            {
                out.push("--filter-platform".to_string());
                out.push(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix("--target=")
            && named_filter_platform(value)
        {
            out.push(format!("--filter-platform={value}"));
        }
        index += 1;
    }
    out
}

fn named_filter_platform(target: &str) -> bool {
    !target.is_empty()
        && !target.ends_with(".json")
        && !target.contains('/')
        && !target.contains('\\')
}

fn rebase_path(value: &str, invocation_dir: &Path) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        invocation_dir.join(path).to_string_lossy().into_owned()
    }
}

fn rebase_config_value(value: &str, invocation_dir: Option<&Path>) -> String {
    match invocation_dir {
        Some(invocation_dir) if !value.contains('=') => rebase_path(value, invocation_dir),
        Some(invocation_dir) => {
            rebase_inline_config_path(value, invocation_dir).unwrap_or_else(|| value.to_string())
        }
        _ => value.to_string(),
    }
}

/// Rebase scalar Cargo config values whose schema defines a filesystem path.
///
/// `--config KEY=VALUE` resolves relative paths against Cargo's current
/// directory. Scheduler discovery/build changes that directory, so the common
/// path-bearing keys (especially `[patch].*.path`) need the same treatment as
/// a `--config path.toml` argument. Bare executable names remain PATH lookups.
fn rebase_inline_config_path(value: &str, invocation_dir: &Path) -> Option<String> {
    let (key, raw_value) = value.split_once('=')?;
    let key = key.trim();
    let raw_value = raw_value.trim();
    let decoded = if raw_value.starts_with('"') && raw_value.ends_with('"') {
        serde_json::from_str::<String>(raw_value).ok()?
    } else if raw_value.len() >= 2 && raw_value.starts_with('\'') && raw_value.ends_with('\'') {
        raw_value[1..raw_value.len() - 1].to_string()
    } else {
        return None;
    };
    let non_executable_path = key.ends_with(".path")
        || matches!(
            key,
            "build.target-dir" | "build.build-dir" | "build.dep-info-basedir"
        );
    let executable_path = matches!(
        key,
        "build.rustc"
            | "build.rustc-wrapper"
            | "build.rustc-workspace-wrapper"
            | "build.rustdoc"
            | "doc.browser"
    ) || key.ends_with(".linker")
        || key.ends_with(".runner");
    if !non_executable_path
        && !(executable_path && (decoded.contains('/') || decoded.contains('\\')))
    {
        return None;
    }
    let rebased = rebase_path(&decoded, invocation_dir);
    Some(format!(
        "{key}={}",
        serde_json::to_string(&rebased).expect("a Rust string always serializes as JSON")
    ))
}

/// Cargo controls whose semantics must also govern parent-owned scheduler
/// builds.
///
/// Package selection and scheduler profile are supplied by the prebuild
/// planner itself. Resolution, target, and artifact-placement controls remain
/// relevant. Feature flags deliberately do not: declaration-test features
/// name the consumer workspace and can fail or broaden a distinct scheduler
/// package, so they stay on metadata/nextest rather than leaking into this
/// parent-owned build.
///
/// Relative path-valued controls are rebased before the scheduler command
/// changes its current directory to the declaring workspace.
pub(crate) fn scheduler_build_options(args: &[String], invocation_dir: &Path) -> Vec<String> {
    let args = cargo_args(args);
    let mut out = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if matches!(
            arg.as_str(),
            "--locked" | "--offline" | "--frozen" | "--ignore-rust-version"
        ) {
            out.push(arg.clone());
        } else if matches!(
            arg.as_str(),
            "--config" | "--target" | "--target-dir" | "-Z"
        ) {
            out.push(arg.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                let value = match arg.as_str() {
                    "--config" => rebase_config_value(value, Some(invocation_dir)),
                    "--target-dir" => rebase_path(value, invocation_dir),
                    "--target"
                        if value.ends_with(".json")
                            || value.contains('/')
                            || value.contains('\\') =>
                    {
                        rebase_path(value, invocation_dir)
                    }
                    _ => value.clone(),
                };
                out.push(value);
            }
        } else if let Some(value) = arg.strip_prefix("--config=") {
            out.push(format!(
                "--config={}",
                rebase_config_value(value, Some(invocation_dir))
            ));
        } else if let Some(value) = arg.strip_prefix("--target-dir=") {
            out.push(format!(
                "--target-dir={}",
                rebase_path(value, invocation_dir)
            ));
        } else if let Some(value) = arg.strip_prefix("--target=") {
            let value = if value.ends_with(".json") || value.contains('/') || value.contains('\\') {
                rebase_path(value, invocation_dir)
            } else {
                value.to_string()
            };
            out.push(format!("--target={value}"));
        } else if arg.starts_with("-Z") && arg.len() > 2 {
            out.push(arg.clone());
        }
        index += 1;
    }
    out
}

/// Cargo feature-selection arguments that shape a normal metadata resolve.
///
/// Unlike [`metadata_passthrough_options`], these are replayed only by
/// [`MetadataMode::Default`]. Manifest-only discovery needs feature
/// definitions, not an activated graph. Stop at `--`, where tokens become
/// test-binary arguments.
pub(crate) fn metadata_resolution_options(args: &[String]) -> Vec<String> {
    let args = cargo_args(args);
    let mut out = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if matches!(arg.as_str(), "--features" | "-F") {
            out.push(arg.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                out.push(value.clone());
            }
        } else if matches!(arg.as_str(), "--all-features" | "--no-default-features")
            || arg.starts_with("--features=")
            || (arg.starts_with("-F") && arg.len() > 2)
        {
            out.push(arg.clone());
        }
        index += 1;
    }
    out
}

/// Cargo options that must shape a registry-discovery `cargo test --no-run`
/// exactly like the caller's eventual nextest build.
///
/// `cargo ktstr test` and `cargo ktstr coverage` accept nextest (or
/// cargo-llvm-cov-nextest) argv, not raw `cargo test` argv. Most build-shaping
/// spellings are shared, but nextest calls the harness profile
/// `--cargo-profile`, its compile parallelism `--build-jobs`, and
/// cargo-llvm-cov calls a test-only workspace exclusion
/// `--exclude-from-test`. Normalize those three onto Cargo's spellings while
/// preserving package, target, feature, resolution, and artifact-placement
/// controls byte-for-byte. Runtime filters and test-binary arguments are
/// deliberately omitted: they cannot change which distributed registry is
/// linked, and `--` makes the remaining suffix opaque.
///
/// The caller passes the argv *after* metadata-driven smart feature
/// preparation. Consequently inferred package-qualified roots and explicit
/// feature modes (`--all-features`, `--no-default-features`, composite
/// `--features` expressions) all reach the registry build through the same
/// path as the final nextest invocation.
pub(crate) fn test_registry_build_options(args: &[String]) -> Vec<String> {
    let args = cargo_args(args);
    let mut out = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];

        let (output_flag, takes_value) = match argument.as_str() {
            // Package selection.
            "-p" | "--package" | "--exclude" => (Some(argument.as_str()), true),
            "--exclude-from-test" => (Some("--exclude"), true),
            "--workspace" => (Some("--workspace"), false),
            "--all" => (Some("--workspace"), false),

            // Cargo target selection. `cargo test --no-run` gives these the
            // same unit/integration harness semantics as nextest.
            "--lib" | "--bins" | "--examples" | "--tests" | "--benches" | "--all-targets" => {
                (Some(argument.as_str()), false)
            }
            "--bin" | "--example" | "--test" | "--bench" => (Some(argument.as_str()), true),

            // Feature modes.
            "-F" | "--features" => (Some(argument.as_str()), true),
            "--all-features" | "--no-default-features" => (Some(argument.as_str()), false),

            // Compilation context. `-r` is nextest's short spelling too.
            "-r" | "--release" => (Some("--release"), false),
            "--cargo-profile" => (Some("--profile"), true),
            "--target" | "--target-dir" => (Some(argument.as_str()), true),
            "--build-jobs" => (Some("--jobs"), true),

            // Cargo resolution / manifest context.
            "--manifest-path" | "--config" | "-Z" => (Some(argument.as_str()), true),
            "--frozen" | "--locked" | "--offline" | "--ignore-rust-version" => {
                (Some(argument.as_str()), false)
            }
            _ => (None, false),
        };

        if let Some(output_flag) = output_flag {
            out.push(output_flag.to_string());
            if takes_value {
                index += 1;
                if let Some(value) = args.get(index) {
                    out.push(value.clone());
                }
            }
            index += 1;
            continue;
        }

        // Attached long forms.
        if argument.starts_with("--package=")
            || argument.starts_with("--exclude=")
            || argument.starts_with("--bin=")
            || argument.starts_with("--example=")
            || argument.starts_with("--test=")
            || argument.starts_with("--bench=")
            || argument.starts_with("--features=")
            || argument.starts_with("--target=")
            || argument.starts_with("--target-dir=")
            || argument.starts_with("--manifest-path=")
            || argument.starts_with("--config=")
        {
            out.push(argument.clone());
        } else if let Some(value) = argument.strip_prefix("--exclude-from-test=") {
            out.push(format!("--exclude={value}"));
        } else if let Some(value) = argument.strip_prefix("--cargo-profile=") {
            out.push(format!("--profile={value}"));
        } else if let Some(value) = argument.strip_prefix("--build-jobs=") {
            out.push(format!("--jobs={value}"));
        } else if (argument.starts_with("-p") && argument.len() > 2)
            || (argument.starts_with("-F") && argument.len() > 2)
            || (argument.starts_with("-Z") && argument.len() > 2)
        {
            out.push(argument.clone());
        }
        index += 1;
    }
    out
}

#[cfg(test)]
fn metadata_other_options(args: &[String], mode: MetadataMode) -> Vec<String> {
    let mut options = metadata_passthrough_options(args);
    if mode == MetadataMode::Default {
        options.extend(metadata_resolution_options(args));
    }
    options
}

fn parsed_feature_modes(args: &[String]) -> Option<(Vec<String>, bool, bool)> {
    let args = cargo_args(args);
    let mut features = Vec::new();
    let mut all_features = false;
    let mut no_default_features = false;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let value = if matches!(argument.as_str(), "--features" | "-F") {
            index += 1;
            Some(args.get(index)?.as_str())
        } else if let Some(value) = argument.strip_prefix("--features=") {
            Some(value)
        } else if let Some(value) = argument.strip_prefix("-F")
            && !value.is_empty()
        {
            Some(value.strip_prefix('=').unwrap_or(value))
        } else {
            None
        };
        if let Some(value) = value {
            if value.is_empty() {
                return None;
            }
            features.extend(
                value
                    .split(|character: char| character == ',' || character.is_whitespace())
                    .filter(|feature| !feature.is_empty())
                    .map(ToString::to_string),
            );
        } else if argument == "--all-features" {
            all_features = true;
        } else if argument == "--no-default-features" {
            no_default_features = true;
        }
        index += 1;
    }
    Some((features, all_features, no_default_features))
}

/// Re-express caller feature modes in Cargo metadata's workspace-global
/// feature namespace without widening beyond Cargo's selected packages.
///
/// `cargo metadata` has no `-p`; an unqualified `--features ktstr-tests` or
/// `--all-features` therefore activates every workspace member that defines
/// the feature. The real nextest/build command still has its `-p` selection.
/// Package-qualifying the metadata-only replay keeps its resolved graph equal
/// to that command while leaving the forwarded argv untouched.
fn scoped_metadata_resolution_options(
    args: &[String],
    selection_args: &[String],
    manifests: &Metadata,
) -> Vec<String> {
    let Some((explicit, all_features, no_default_features)) = parsed_feature_modes(args) else {
        return metadata_resolution_options(args);
    };
    let Some(selected) = selected_workspace_packages(manifests, selection_args) else {
        return metadata_resolution_options(args);
    };

    let mut selectors = Vec::new();
    if !no_default_features {
        for package in &selected {
            if package.features.contains_key("default") {
                selectors.push(format!("{}/default", package.name));
            }
        }
    }
    if all_features {
        for package in &selected {
            selectors.extend(
                package
                    .features
                    .keys()
                    .map(|feature| format!("{}/{feature}", package.name)),
            );
        }
    }
    for feature in explicit {
        if feature.contains('/') {
            selectors.push(feature);
            continue;
        }
        let mut matched = false;
        for package in &selected {
            if package.features.contains_key(&feature) {
                selectors.push(format!("{}/{feature}", package.name));
                matched = true;
            }
        }
        if !matched {
            if let Some(package) = selected.first() {
                // Preserve Cargo's unknown-feature diagnostic without letting
                // an unrelated workspace member of the same feature name make
                // metadata succeed spuriously.
                selectors.push(format!("{}/{feature}", package.name));
            } else {
                selectors.push(feature);
            }
        }
    }
    selectors.sort();
    selectors.dedup();

    let mut options = Vec::new();
    if !selectors.is_empty() {
        options.push("--features".to_string());
        options.push(selectors.join(","));
    }
    // Cargo metadata cannot carry `-p`, so its ordinary default activation is
    // workspace-wide. Disable that global behavior unconditionally, then the
    // selected-package `/default` selectors above reconstruct Cargo's actual
    // defaults unless the caller explicitly disabled them.
    options.push("--no-default-features".to_string());
    options
}

fn metadata_passthrough_for_target(args: &[String], target: &TargetContext) -> Vec<String> {
    let mut options = metadata_passthrough_options(args);
    let already_filtered = options
        .iter()
        .any(|option| option == "--filter-platform" || option.starts_with("--filter-platform="));
    if !already_filtered && let Some(platform) = &target.metadata_filter {
        options.push("--filter-platform".to_string());
        options.push(platform.clone());
    }
    options
}

fn execute_metadata(
    args: &[String],
    mode: MetadataMode,
    target: &TargetContext,
    resolution_options: &[String],
) -> Result<Metadata, String> {
    let mut options = metadata_passthrough_for_target(args, target);
    if mode == MetadataMode::Default {
        options.extend_from_slice(resolution_options);
    }
    let mut command = MetadataCommand::new();
    command.cargo_path("cargo").other_options(options);
    if mode == MetadataMode::NoDeps {
        command.no_deps();
    }
    command
        .exec()
        .map_err(|error| format!("cargo metadata failed: {error}"))
}

pub(crate) fn query_metadata_for_target(
    args: &[String],
    mode: MetadataMode,
    target: &TargetContext,
) -> Result<Metadata, String> {
    match mode {
        MetadataMode::NoDeps => execute_metadata(args, mode, target, &[]),
        MetadataMode::Default => {
            let manifests = execute_metadata(args, MetadataMode::NoDeps, target, &[])?;
            query_resolved_metadata(args, args, &manifests, target)
        }
    }
}

/// Resolve one test selection using a manifest pass the caller already owns.
pub(crate) fn query_resolved_metadata(
    args: &[String],
    selection_args: &[String],
    manifests: &Metadata,
    target: &TargetContext,
) -> Result<Metadata, String> {
    let resolution = scoped_metadata_resolution_options(args, selection_args, manifests);
    execute_metadata(args, MetadataMode::Default, target, &resolution)
}

/// Classify a Cargo feature member that addresses one ktstr dependency alias.
///
/// `true` means the member strongly activates the optional dependency
/// (`dep:alias` or `alias/feature`). `false` is Cargo's weak
/// `alias?/feature` forwarding syntax, which is safe within a ktstr-only
/// feature but does not itself activate the dependency.
fn compatible_ktstr_member(member: &str, aliases: &HashSet<&str>) -> Option<bool> {
    if let Some(alias) = member.strip_prefix("dep:") {
        return aliases.contains(alias).then_some(true);
    }
    let (alias, _) = member.split_once('/')?;
    if let Some(alias) = alias.strip_suffix('?') {
        aliases.contains(alias).then_some(false)
    } else {
        aliases.contains(alias).then_some(true)
    }
}

/// Whether a manifest dependency can participate in the requested target.
///
/// Production callers supply rustc's effective target name and complete cfg
/// set, so both exact target tables and `cfg(...)` expressions are evaluated
/// with Cargo's own platform matcher. The optional legacy test wrapper keeps
/// `None` as "unknown" and therefore declines target-specific activation.
fn dependency_matches_target_context(
    dependency: &cargo_metadata::Dependency,
    target: Option<&TargetContext>,
) -> bool {
    let Some(platform) = dependency.target.as_ref() else {
        return true;
    };
    let Some(target) = target else {
        return false;
    };
    target.matches_platform(platform)
}

/// Whether one target participates in `cargo test`/nextest harness discovery.
fn target_has_test_harness(target: &cargo_metadata::Target) -> bool {
    target.test
}

fn semantic_name_mentions_ktstr(name: &str, aliases: &HashSet<&str>) -> bool {
    fn normalized(value: &str) -> String {
        value
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect()
    }

    let name = format!("-{}-", normalized(name));
    std::iter::once("ktstr")
        .chain(aliases.iter().copied())
        .map(normalized)
        .any(|alias| !alias.is_empty() && name.contains(&format!("-{alias}-")))
}

/// A target-required feature can supplement graph-based ktstr activation only
/// when enabling it remains narrow.
///
/// An empty required feature is a source-level gate with no Cargo dependency
/// side effects, but metadata cannot otherwise associate it with ktstr. It is
/// therefore admitted only when the feature or integration-test target name
/// contains `ktstr` (or the dependency's rename) as a delimited component. A
/// non-empty gate must walk exclusively through local features and ktstr
/// dependency-feature forwarding, and must actually mention ktstr; an empty
/// descendant is rejected so a composite mode cannot smuggle an unrelated
/// source-only feature into an otherwise ktstr-looking wrapper.
fn required_feature_is_narrow(
    start: &str,
    features: &BTreeMap<String, Vec<String>>,
    aliases: &HashSet<&str>,
    allow_empty: bool,
) -> bool {
    let Some(start_members) = features.get(start) else {
        return false;
    };
    if start_members.is_empty() {
        return allow_empty;
    }

    let mut pending = vec![start];
    let mut seen = HashSet::new();
    let mut mentions_ktstr = false;
    while let Some(feature) = pending.pop() {
        if !seen.insert(feature) {
            continue;
        }
        let Some(members) = features.get(feature) else {
            return false;
        };
        if feature != start && members.is_empty() {
            return false;
        }
        for member in members {
            if compatible_ktstr_member(member, aliases).is_some() {
                mentions_ktstr = true;
            } else if features.contains_key(member) {
                pending.push(member);
            } else {
                return false;
            }
        }
    }
    mentions_ktstr
}

/// Whether `start` transitively enables the local Cargo feature `target`.
fn local_feature_reaches<'a>(
    start: &'a str,
    target: &str,
    edges: &HashMap<&'a str, Vec<&'a str>>,
) -> bool {
    let mut pending = vec![start];
    let mut seen = HashSet::new();
    while let Some(feature) = pending.pop() {
        if feature == target {
            return true;
        }
        if seen.insert(feature) {
            pending.extend(edges.get(feature).into_iter().flatten().copied());
        }
    }
    false
}

/// Infer narrow feature roots that activate one package's optional ktstr.
///
/// This follows renamed dependencies and local feature aliases. A feature is
/// eligible only when every member in its reachable graph is ktstr-specific
/// and at least one member strongly activates ktstr. Thus
/// `ktstr-tests = ["verify"]` can be inferred through
/// `verify = ["dep:ktstr"]`, while
/// `everything = ["ktstr-tests", "gpu"]` remains opt-in.
///
/// `default` is never selected as an automatic root. Treating it as a normal
/// root would silently undo an operator's `--no-default-features`; a narrower
/// ktstr-only descendant is selected instead when one exists.
#[cfg(test)]
pub(crate) fn infer_ktstr_feature_roots(
    package: &cargo_metadata::Package,
    scope: VersionScope<'_>,
) -> Vec<String> {
    infer_ktstr_feature_roots_for_target(package, scope, None)
}

/// Target-aware feature-root inference.
#[cfg(test)]
pub(crate) fn infer_ktstr_feature_roots_for_target(
    package: &cargo_metadata::Package,
    scope: VersionScope<'_>,
    requested_target: Option<&str>,
) -> Vec<String> {
    let target = requested_target.map(|target| TargetContext::named(target, Vec::new()));
    infer_ktstr_feature_roots_for_context(package, scope, target.as_ref())
}

pub(crate) fn infer_ktstr_feature_roots_for_context(
    package: &cargo_metadata::Package,
    scope: VersionScope<'_>,
    target: Option<&TargetContext>,
) -> Vec<String> {
    let eligible_dependencies = package
        .dependencies
        .iter()
        .filter(|dependency| {
            dependency.name == "ktstr"
                && dependency_matches_target_context(dependency, target)
                && matches!(
                    dependency.kind,
                    cargo_metadata::DependencyKind::Normal
                        | cargo_metadata::DependencyKind::Development
                )
                && match scope {
                    VersionScope::Any => true,
                    VersionScope::Matches(version) => dependency.req.matches(version),
                }
        })
        .collect::<Vec<_>>();
    let has_nonoptional_ktstr = eligible_dependencies
        .iter()
        .any(|dependency| !dependency.optional);
    let aliases = eligible_dependencies
        .into_iter()
        .map(|dependency| {
            dependency
                .rename
                .as_deref()
                .unwrap_or(dependency.name.as_str())
        })
        .collect::<HashSet<_>>();

    let feature_names = package
        .features
        .keys()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut local_edges = HashMap::<&str, Vec<&str>>::new();
    let mut activates = HashSet::new();
    let mut impure = HashSet::new();
    if feature_names.contains("default") {
        impure.insert("default");
    }

    for (feature, members) in &package.features {
        let feature = feature.as_str();
        for member in members {
            match compatible_ktstr_member(member, &aliases) {
                Some(true) => {
                    activates.insert(feature);
                }
                Some(false) => {}
                None if feature_names.contains(member.as_str()) => {
                    local_edges.entry(feature).or_default().push(member);
                }
                None => {
                    impure.insert(feature);
                }
            }
        }
    }

    // Find every feature with a path to a strong ktstr activation.
    let mut reaches_ktstr = activates;
    loop {
        let before = reaches_ktstr.len();
        for (feature, dependencies) in &local_edges {
            if dependencies
                .iter()
                .any(|dependency| reaches_ktstr.contains(dependency))
            {
                reaches_ktstr.insert(feature);
            }
        }
        if reaches_ktstr.len() == before {
            break;
        }
    }

    // A local member without its own ktstr path may represent an unrelated
    // mode. Taint its parents, then propagate that taint through the graph.
    for (feature, dependencies) in &local_edges {
        if dependencies
            .iter()
            .any(|dependency| !reaches_ktstr.contains(dependency))
        {
            impure.insert(feature);
        }
    }
    loop {
        let before = impure.len();
        for (feature, dependencies) in &local_edges {
            if dependencies
                .iter()
                .any(|dependency| impure.contains(dependency))
            {
                impure.insert(feature);
            }
        }
        if impure.len() == before {
            break;
        }
    }

    let mut pure = reaches_ktstr
        .difference(&impure)
        .copied()
        .collect::<Vec<_>>();
    pure.sort_unstable();

    let mut roots: Vec<&str> = Vec::new();
    for feature in &pure {
        // A strict ancestor is a different feature component that enables
        // this one without being enabled in return.
        let has_strict_ancestor = pure.iter().any(|other| {
            other != feature
                && local_feature_reaches(other, feature, &local_edges)
                && !local_feature_reaches(feature, other, &local_edges)
        });
        if has_strict_ancestor {
            continue;
        }
        // Cyclic features enable one another, so one sorted representative
        // of a root component is sufficient.
        let root_component_already_selected = roots.iter().any(|other| {
            local_feature_reaches(other, feature, &local_edges)
                && local_feature_reaches(feature, other, &local_edges)
        });
        if !root_component_already_selected {
            roots.push(*feature);
        }
    }

    let mut selected = roots
        .into_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    // A nonoptional ktstr is already linked, so a test target may use a local
    // required feature solely as its source-level declaration gate. Add the
    // exact required feature only when its Cargo graph is empty or exclusively
    // ktstr/local forwarding, and only when no graph-derived root already
    // enables it. This admits `required-features = ["ktstr-tests"]` with
    // `ktstr-tests = []` without approximating `--all-features`.
    if has_nonoptional_ktstr {
        let mut required = package
            .targets
            .iter()
            .filter(|target| target_has_test_harness(target))
            .filter(|target| {
                let target_mentions_ktstr = semantic_name_mentions_ktstr(&target.name, &aliases);
                target.required_features.iter().all(|feature| {
                    feature == "default"
                        || required_feature_is_narrow(
                            feature,
                            &package.features,
                            &aliases,
                            target_mentions_ktstr
                                || semantic_name_mentions_ktstr(feature, &aliases),
                        )
                })
            })
            .flat_map(|target| target.required_features.iter())
            .filter(|feature| feature.as_str() != "default")
            .cloned()
            .collect::<Vec<_>>();
        required.sort();
        required.dedup();
        for feature in required {
            if !selected
                .iter()
                .any(|root| local_feature_reaches(root, &feature, &local_edges))
            {
                selected.push(feature);
            }
        }
    }
    selected.sort();
    selected.dedup();
    selected
}

/// Recover a Cargo package name from common `-p` package-id spellings.
pub(crate) fn package_spec_name(spec: &str) -> Option<&str> {
    let spec = spec.strip_prefix('=').unwrap_or(spec);
    let tail = spec.rsplit_once('#').map_or(spec, |(_, tail)| tail);
    let name = tail.split(['@', ':']).next()?;
    (!name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')))
    .then_some(name)
}

/// The name-pattern and optional exact version carried by a Cargo package
/// specification.
///
/// Cargo accepts `name`, `name@version`, the legacy `name:version`, and full
/// source-qualified forms such as `path+file:///w#name@version`. A full package
/// ID is also checked byte-for-byte by [`package_matches_spec`] before this
/// parser is used.
fn package_spec_pattern(spec: &str) -> Option<(&str, Option<&str>)> {
    let spec = spec.strip_prefix('=').unwrap_or(spec);
    let tail = spec.rsplit_once('#').map_or(spec, |(_, tail)| tail);
    if tail.is_empty() {
        return None;
    }
    let (name, version) = tail
        .split_once('@')
        .or_else(|| tail.split_once(':'))
        .map_or((tail, None), |(name, version)| (name, Some(version)));
    if name.is_empty() || version.is_some_and(str::is_empty) {
        return None;
    }
    Some((name, version))
}

/// Whether a workspace package matches one Cargo `-p` / `--exclude` spec.
///
/// Match the exact metadata package ID first, then apply Cargo's documented
/// Unix glob syntax to the package-name component. A supplied version remains
/// exact so `foo@1.0.0` cannot accidentally select another workspace version.
pub(crate) fn package_matches_spec(package: &cargo_metadata::Package, spec: &str) -> bool {
    let spec = spec.strip_prefix('=').unwrap_or(spec);
    if package.id.repr == spec {
        return true;
    }
    let Some((name_pattern, version)) = package_spec_pattern(spec) else {
        return false;
    };
    if let Some(version) = version {
        // Cargo package specs accept partial exact versions (`foo@1`,
        // `foo@1.2`) in addition to a full semver. Semver's `=`
        // requirement has the same prefix semantics.
        let Ok(requirement) = VersionReq::parse(&format!("={version}")) else {
            return false;
        };
        if !requirement.matches(&package.version) {
            return false;
        }
    }
    Pattern::new(name_pattern).is_ok_and(|pattern| pattern.matches(package.name.as_str()))
}

fn cargo_args(args: &[String]) -> &[String] {
    args.iter()
        .position(|arg| arg == "--")
        .map_or(args, |separator| &args[..separator])
}

fn has_explicit_package_request(args: &[String]) -> bool {
    cargo_args(args).iter().any(|arg| {
        matches!(arg.as_str(), "-p" | "--package")
            || arg.starts_with("--package=")
            || (arg.starts_with("-p") && arg.len() > 2)
    })
}

/// Raw Cargo package specifications requested via `-p` / `--package`.
///
/// `None` means either no explicit request or invalid/missing selector syntax;
/// callers can distinguish those cases with [`has_explicit_package_request`].
fn explicit_package_specs(args: &[String]) -> Option<Vec<String>> {
    let args = cargo_args(args);
    let mut specs = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let spec = if matches!(arg.as_str(), "-p" | "--package") {
            index += 1;
            args.get(index).map(String::as_str)?
        } else if let Some(spec) = arg.strip_prefix("--package=") {
            spec
        } else if let Some(spec) = arg.strip_prefix("-p")
            && !spec.is_empty()
        {
            spec
        } else {
            index += 1;
            continue;
        };
        if spec.is_empty() {
            return None;
        }
        specs.push(spec.to_string());
        index += 1;
    }
    (!specs.is_empty()).then_some(specs)
}

/// Exact package names requested by `-p` / `--package`, or `None` for
/// unscoped/workspace/unsupported package-spec cases.
#[cfg(test)]
pub(crate) fn explicit_package_selection(args: &[String]) -> Option<HashSet<String>> {
    if has_workspace_selector(args) {
        return None;
    }
    explicit_package_specs(args)?
        .into_iter()
        .map(|spec| package_spec_name(&spec).map(ToString::to_string))
        .collect()
}

fn explicit_package_exclusion_specs(args: &[String]) -> Option<Vec<String>> {
    let args = cargo_args(args);
    let mut excluded = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let spec = if matches!(arg.as_str(), "--exclude" | "--exclude-from-test") {
            index += 1;
            Some(args.get(index)?.as_str())
        } else {
            arg.strip_prefix("--exclude=")
                .or_else(|| arg.strip_prefix("--exclude-from-test="))
        };
        if let Some(spec) = spec {
            if spec.is_empty() {
                return None;
            }
            excluded.push(spec.to_string());
        }
        index += 1;
    }
    Some(excluded)
}

#[cfg(test)]
pub(crate) fn explicit_package_exclusions(args: &[String]) -> HashSet<String> {
    explicit_package_exclusion_specs(args)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|spec| package_spec_name(&spec).map(ToString::to_string))
        .collect()
}

pub(crate) fn has_package_selector(args: &[String]) -> bool {
    cargo_args(args).iter().any(|arg| {
        matches!(
            arg.as_str(),
            "-p" | "--package" | "--workspace" | "--all" | "--exclude" | "--exclude-from-test"
        ) || arg.starts_with("--package=")
            || arg.starts_with("--exclude=")
            || arg.starts_with("--exclude-from-test=")
            || (arg.starts_with("-p") && arg.len() > 2)
    })
}

pub(crate) fn has_workspace_selector(args: &[String]) -> bool {
    cargo_args(args)
        .iter()
        .any(|arg| matches!(arg.as_str(), "--workspace" | "--all"))
}

fn explicit_all_features(args: &[String]) -> bool {
    cargo_args(args).iter().any(|arg| arg == "--all-features")
}

/// Resolve the workspace packages selected by Cargo package arguments.
///
/// This is the common selection source for feature activation and version
/// compatibility checks: exact/full package IDs, name globs, default members,
/// workspace-wide selection, and exclusions therefore cannot drift between
/// those callers. `None` means an explicit selector was malformed or could not
/// be interpreted safely; an empty `Some` means the specs matched no package
/// (which the eventual Cargo command will diagnose).
pub(crate) fn selected_workspace_packages<'metadata>(
    metadata: &'metadata Metadata,
    args: &[String],
) -> Option<Vec<&'metadata cargo_metadata::Package>> {
    let member_ids = metadata.workspace_members.iter().collect::<HashSet<_>>();
    let explicit_specs = explicit_package_specs(args);
    if has_explicit_package_request(args) && explicit_specs.is_none() {
        // An unsupported package spec is safer left explicit than guessed.
        return None;
    }
    let default_ids: &[cargo_metadata::PackageId] =
        if metadata.workspace_default_members.is_available()
            && !metadata.workspace_default_members.is_empty()
        {
            &metadata.workspace_default_members
        } else {
            &metadata.workspace_members
        };
    let default_ids = default_ids.iter().collect::<HashSet<_>>();
    let exclusion_specs = explicit_package_exclusion_specs(args)?;

    let mut packages = metadata
        .packages
        .iter()
        .filter(|package| member_ids.contains(&package.id))
        .filter(|package| {
            if has_workspace_selector(args) {
                true
            } else if let Some(specs) = &explicit_specs {
                specs.iter().any(|spec| package_matches_spec(package, spec))
            } else {
                default_ids.contains(&package.id)
            }
        })
        .filter(|package| {
            !exclusion_specs
                .iter()
                .any(|spec| package_matches_spec(package, spec))
        })
        .collect::<Vec<_>>();
    packages.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.version.cmp(&right.version))
            .then_with(|| left.id.repr.cmp(&right.id.repr))
    });
    Some(packages)
}

/// Infer activations only for the workspace packages Cargo will select.
#[cfg(test)]
pub(crate) fn selected_activations(
    metadata: &Metadata,
    args: &[String],
    scope: VersionScope<'_>,
) -> Vec<PackageFeatureActivation> {
    let requested_target = requested_target(args);
    let target = requested_target
        .as_deref()
        .map(|target| TargetContext::named(target, Vec::new()));
    selected_activations_for_context(metadata, args, scope, target.as_ref())
}

pub(crate) fn selected_activations_for_context(
    metadata: &Metadata,
    args: &[String],
    scope: VersionScope<'_>,
    target: Option<&TargetContext>,
) -> Vec<PackageFeatureActivation> {
    let mut activations = selected_workspace_packages(metadata, args)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|package| {
            let features = infer_ktstr_feature_roots_for_context(package, scope, target);
            (!features.is_empty()).then(|| PackageFeatureActivation {
                package: package.name.to_string(),
                features,
            })
        })
        .collect::<Vec<_>>();
    activations.sort_by(|left, right| left.package.cmp(&right.package));
    activations
}

/// Add one package-qualified `--features` flag before any `--` separator.
///
/// Existing `--features` flags are preserved and union naturally in Cargo.
/// An explicit Cargo-side `--all-features` is authoritative and suppresses
/// automatic selectors; a token after `--` is test-binary input and does not.
pub(crate) fn inject_feature_activations(
    mut args: Vec<String>,
    activations: &[PackageFeatureActivation],
) -> Vec<String> {
    if explicit_all_features(&args) {
        return args;
    }
    let mut selectors = activations
        .iter()
        .flat_map(|activation| {
            activation
                .features
                .iter()
                .map(|feature| format!("{}/{feature}", activation.package))
        })
        .collect::<Vec<_>>();
    selectors.sort();
    selectors.dedup();
    if selectors.is_empty() {
        return args;
    }

    let insertion = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    args.splice(
        insertion..insertion,
        ["--features".to_string(), selectors.join(",")],
    );
    args
}

/// Select features on one workspace package only when Cargo will build that
/// package, retaining only requested features its manifest actually declares.
///
/// This is used for cargo-ktstr's own compile-time capabilities when a probe is
/// building the ktstr workspace itself. Package qualification is load-bearing:
/// forwarding `--features wprof` into an arbitrary consumer workspace can
/// either fail on an unknown feature or accidentally enable an unrelated
/// same-named feature. A dependency named `ktstr` is not enough; it must be a
/// selected workspace member.
fn selected_workspace_package_feature_activation(
    metadata: &Metadata,
    args: &[String],
    package_name: &str,
    requested_features: &[&str],
) -> Option<PackageFeatureActivation> {
    let package = selected_workspace_packages(metadata, args)?
        .into_iter()
        .find(|package| package.name.as_str() == package_name)?;
    let mut features = requested_features
        .iter()
        .copied()
        .filter(|feature| package.features.contains_key(*feature))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    features.sort();
    features.dedup();
    (!features.is_empty()).then(|| PackageFeatureActivation {
        package: package.name.to_string(),
        features,
    })
}

/// Discover ordinary ktstr registry gates and, when present in the selected
/// workspace, safely preserve a named package's already-active capabilities.
///
/// `workspace_package_features` never become unqualified Cargo features. They
/// are injected as `PACKAGE/FEATURE` only after metadata proves that PACKAGE is
/// a selected workspace member and declares FEATURE.
pub(crate) fn augment_test_features_with_workspace_package_features(
    args: Vec<String>,
    workspace_package: &str,
    workspace_package_features: &[&str],
) -> Result<Vec<String>, String> {
    if explicit_all_features(&args) {
        return Ok(args);
    }
    let target = effective_target_context(&args)?;
    let metadata = query_metadata_for_target(&args, MetadataMode::NoDeps, &target)?;
    let mut activations =
        selected_activations_for_context(&metadata, &args, VersionScope::Any, Some(&target));
    if let Some(activation) = selected_workspace_package_feature_activation(
        &metadata,
        &args,
        workspace_package,
        workspace_package_features,
    ) {
        activations.push(activation);
    }
    Ok(inject_feature_activations(args, &activations))
}

/// Inject ordinary test features from metadata a caller already queried.
#[cfg(test)]
pub(crate) fn augment_test_features_from_metadata(
    args: Vec<String>,
    metadata: &Metadata,
) -> Vec<String> {
    let activations = selected_activations(metadata, &args, VersionScope::Any);
    inject_feature_activations(args, &activations)
}

pub(crate) fn augment_test_features_from_metadata_for_context(
    args: Vec<String>,
    metadata: &Metadata,
    target: Option<&TargetContext>,
) -> Vec<String> {
    let activations = selected_activations_for_context(metadata, &args, VersionScope::Any, target);
    inject_feature_activations(args, &activations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn activations() -> Vec<PackageFeatureActivation> {
        vec![
            PackageFeatureActivation {
                package: "cosmos".to_string(),
                features: vec!["ktstr-tests".to_string()],
            },
            PackageFeatureActivation {
                package: "lavd".to_string(),
                features: vec!["verify".to_string()],
            },
        ]
    }

    fn optional_ktstr_package_json(
        name: &str,
        version: &str,
        requirement: &str,
        features: &str,
    ) -> String {
        let id = format!("{name} {version} (path+file:///w/{name})");
        format!(
            r#"{{
                "name":"{name}",
                "version":"{version}",
                "id":"{id}",
                "source":null,
                "description":null,
                "dependencies":[{{
                    "name":"ktstr",
                    "source":null,
                    "req":"{requirement}",
                    "kind":null,
                    "rename":null,
                    "optional":true,
                    "uses_default_features":true,
                    "features":[],
                    "target":null,
                    "registry":null,
                    "path":null
                }}],
                "license":null,
                "license_file":null,
                "targets":[],
                "features":{features},
                "manifest_path":"/w/{name}/Cargo.toml",
                "readme":null,
                "repository":null,
                "homepage":null,
                "documentation":null,
                "links":null,
                "publish":null,
                "default_run":null
            }}"#
        )
    }

    fn selection_metadata() -> Metadata {
        let cosmos = "cosmos 1.0.0 (path+file:///w/cosmos)";
        let lavd = "lavd 1.0.0 (path+file:///w/lavd)";
        let json = format!(
            r#"{{
                "packages":[{cosmos_package},{lavd_package}],
                "workspace_members":["{cosmos}","{lavd}"],
                "workspace_default_members":["{cosmos}"],
                "resolve":null,
                "workspace_root":"/w",
                "target_directory":"/w/target",
                "version":1
            }}"#,
            cosmos_package = optional_ktstr_package_json(
                "cosmos",
                "1.0.0",
                "=0.42.0",
                r#"{"ktstr-tests":["dep:ktstr"]}"#,
            ),
            lavd_package = optional_ktstr_package_json(
                "lavd",
                "1.0.0",
                "=0.18.0",
                r#"{"verify":["dep:ktstr"]}"#,
            ),
        );
        serde_json::from_str(&json).expect("metadata fixture deserializes")
    }

    fn write_package(root: &Path, relative: &str, manifest: &str) {
        let package = root.join(relative);
        std::fs::create_dir_all(package.join("src")).expect("create fixture package source");
        std::fs::write(package.join("Cargo.toml"), manifest).expect("write fixture manifest");
        std::fs::write(package.join("src/lib.rs"), "pub fn marker() {}\n")
            .expect("write fixture source");
    }

    fn real_feature_workspace() -> tempfile::TempDir {
        let workspace = tempfile::tempdir().expect("feature workspace");
        std::fs::write(
            workspace.path().join("Cargo.toml"),
            r#"[workspace]
members = ["selected", "unrelated"]
exclude = ["deps/ktstr-current", "deps/ktstr-old", "deps/selected-extra"]
resolver = "2"
"#,
        )
        .expect("write workspace manifest");
        write_package(
            workspace.path(),
            "selected",
            r#"[package]
name = "selected"
version = "1.0.0"
edition = "2024"

[dependencies]
selected-extra = { path = "../deps/selected-extra", optional = true }

[target.'cfg(target_os = "linux")'.dependencies]
ktstr = { path = "../deps/ktstr-current", optional = true }

[features]
default = ["selected-default"]
selected-default = []
ktstr-tests = ["dep:ktstr"]
extra = ["dep:selected-extra"]
"#,
        );
        write_package(
            workspace.path(),
            "unrelated",
            r#"[package]
name = "unrelated"
version = "1.0.0"
edition = "2024"

[dependencies]
ktstr = { path = "../deps/ktstr-old", optional = true }

[features]
default = ["ktstr-tests"]
ktstr-tests = ["dep:ktstr"]
unrelated-mode = []
"#,
        );
        write_package(
            workspace.path(),
            "deps/ktstr-current",
            "[package]\nname = \"ktstr\"\nversion = \"0.42.0\"\nedition = \"2024\"\n",
        );
        write_package(
            workspace.path(),
            "deps/ktstr-old",
            "[package]\nname = \"ktstr\"\nversion = \"0.18.0\"\nedition = \"2024\"\n",
        );
        write_package(
            workspace.path(),
            "deps/selected-extra",
            "[package]\nname = \"selected-extra\"\nversion = \"1.0.0\"\nedition = \"2024\"\n",
        );
        workspace
    }

    fn package_node<'a>(metadata: &'a Metadata, name: &str) -> &'a cargo_metadata::Node {
        let package = metadata
            .packages
            .iter()
            .find(|package| package.name.as_str() == name)
            .unwrap_or_else(|| panic!("metadata contains package {name}"));
        metadata
            .resolve
            .as_ref()
            .expect("resolved metadata")
            .nodes
            .iter()
            .find(|node| node.id == package.id)
            .unwrap_or_else(|| panic!("resolve contains node {name}"))
    }

    #[test]
    fn injection_preserves_user_features_and_precedes_test_args() {
        assert_eq!(
            inject_feature_activations(
                strings(&["--features", "integration", "--", "--all-features"]),
                &activations(),
            ),
            strings(&[
                "--features",
                "integration",
                "--features",
                "cosmos/ktstr-tests,lavd/verify",
                "--",
                "--all-features",
            ]),
        );
    }

    #[test]
    fn cargo_side_all_features_suppresses_inference() {
        let args = strings(&["--workspace", "--all-features"]);
        assert_eq!(
            inject_feature_activations(args.clone(), &activations()),
            args
        );
    }

    #[test]
    fn selection_respects_defaults_packages_workspace_and_version_scope() {
        let metadata = selection_metadata();
        assert_eq!(
            selected_activations(&metadata, &[], VersionScope::Any),
            vec![PackageFeatureActivation {
                package: "cosmos".to_string(),
                features: vec!["ktstr-tests".to_string()],
            }],
            "an unscoped ordinary command follows Cargo's default members",
        );
        assert_eq!(
            selected_activations(&metadata, &strings(&["-p", "lavd"]), VersionScope::Any,),
            vec![PackageFeatureActivation {
                package: "lavd".to_string(),
                features: vec!["verify".to_string()],
            }],
            "ordinary commands activate the selected consumer's own ktstr version",
        );
        assert!(
            selected_activations(
                &metadata,
                &strings(&["-p", "lavd"]),
                VersionScope::Matches(&Version::parse("0.42.0").unwrap()),
            )
            .is_empty(),
            "the verifier's version scope rejects an old declaration package",
        );
        assert_eq!(
            selected_activations(
                &metadata,
                &strings(&["--workspace", "--exclude", "lavd"]),
                VersionScope::Any,
            ),
            vec![PackageFeatureActivation {
                package: "cosmos".to_string(),
                features: vec!["ktstr-tests".to_string()],
            }],
        );
    }

    #[test]
    fn workspace_package_capabilities_are_selected_declared_and_package_qualified() {
        let metadata = selection_metadata();
        let selected = selected_workspace_package_feature_activation(
            &metadata,
            &[],
            "cosmos",
            &["missing", "ktstr-tests"],
        );
        assert_eq!(
            selected,
            Some(PackageFeatureActivation {
                package: "cosmos".to_string(),
                features: vec!["ktstr-tests".to_string()],
            }),
            "only a declared feature on Cargo's default member is retained",
        );
        assert_eq!(
            inject_feature_activations(
                strings(&["--workspace"]),
                std::slice::from_ref(selected.as_ref().expect("selected activation")),
            ),
            strings(&["--workspace", "--features", "cosmos/ktstr-tests"]),
            "the capability is emitted only in package-qualified form",
        );
        assert_eq!(
            selected_workspace_package_feature_activation(&metadata, &[], "lavd", &["verify"],),
            None,
            "an unselected workspace member must not receive a feature",
        );
        assert_eq!(
            selected_workspace_package_feature_activation(
                &metadata,
                &strings(&["-p", "lavd"]),
                "lavd",
                &["verify"],
            ),
            Some(PackageFeatureActivation {
                package: "lavd".to_string(),
                features: vec!["verify".to_string()],
            }),
            "an explicitly selected member may retain its declared capability",
        );
        assert_eq!(
            selected_workspace_package_feature_activation(
                &metadata,
                &strings(&["--workspace"]),
                "not-a-member",
                &["verify"],
            ),
            None,
            "a dependency or unrelated package name can never receive consumer features",
        );
    }

    #[test]
    fn malformed_package_selectors_decline_inference_and_remain_unchanged() {
        let metadata = selection_metadata();
        for args in [
            strings(&["-p"]),
            strings(&["--package="]),
            strings(&["--workspace", "--exclude"]),
            strings(&["--workspace", "--exclude="]),
        ] {
            assert!(
                selected_workspace_packages(&metadata, &args).is_none(),
                "malformed Cargo selector must not be guessed: {args:?}",
            );
            assert_eq!(
                augment_test_features_from_metadata(args.clone(), &metadata),
                args,
                "Cargo's eventual parser receives the original malformed argv",
            );
        }
    }

    #[test]
    fn default_wrapper_yields_narrow_descendant_not_default_feature() {
        let json = optional_ktstr_package_json(
            "scheduler",
            "1.0.0",
            "=0.42.0",
            r#"{"default":["ktstr-tests"],"ktstr-tests":["dep:ktstr"]}"#,
        );
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("package fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots(&package, VersionScope::Any),
            vec!["ktstr-tests"],
        );
    }

    #[test]
    fn nonoptional_ktstr_forwarding_feature_is_inferred_narrowly() {
        let json = optional_ktstr_package_json(
            "scheduler",
            "1.0.0",
            "=0.42.0",
            r#"{
                "verify":["ktstr/test-support"],
                "unrelated":[],
                "everything":["verify","unrelated"]
            }"#,
        )
        .replacen(r#""optional":true"#, r#""optional":false"#, 1);
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("nonoptional package fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots(&package, VersionScope::Any),
            vec!["verify"],
            "dependency-feature forwarding is a narrow gate even when ktstr itself is nonoptional",
        );
    }

    #[test]
    fn nonoptional_ktstr_test_targets_add_only_wholly_narrow_required_gates() {
        let targets = r#"[{
            "name":"verifier_cells",
            "kind":["test"],
            "crate_types":["bin"],
            "required-features":["ktstr-tests"],
            "src_path":"/w/scheduler/tests/verifier_cells.rs",
            "edition":"2024",
            "doctest":false,
            "test":true,
            "doc":false
        },{
            "name":"mixed_mode",
            "kind":["test"],
            "crate_types":["bin"],
            "required-features":["ktstr-tests","mixed-mode"],
            "src_path":"/w/scheduler/tests/mixed_mode.rs",
            "edition":"2024",
            "doctest":false,
            "test":true,
            "doc":false
        },{
            "name":"other_cells",
            "kind":["test"],
            "crate_types":["bin"],
            "required-features":["other-tests"],
            "src_path":"/w/scheduler/tests/other_cells.rs",
            "edition":"2024",
            "doctest":false,
            "test":true,
            "doc":false
        },{
            "name":"scheduler_test_cli",
            "kind":["bin"],
            "crate_types":["bin"],
            "required-features":["ktstr-bin-tests"],
            "src_path":"/w/scheduler/src/bin/scheduler_test_cli.rs",
            "edition":"2024",
            "doctest":false,
            "test":true,
            "doc":false
        },{
            "name":"unrelated_cli",
            "kind":["bin"],
            "crate_types":["bin"],
            "required-features":["bin-mode"],
            "src_path":"/w/scheduler/src/main.rs",
            "edition":"2024",
            "doctest":false,
            "test":true,
            "doc":false
        }]"#;
        let json = optional_ktstr_package_json(
            "scheduler",
            "1.0.0",
            "=0.42.0",
            r#"{
                "ktstr-tests":[],
                "ktstr-bin-tests":[],
                "other-tests":[],
                "bin-mode":[],
                "mixed-mode":["unrelated"],
                "unrelated":[]
            }"#,
        )
        .replacen(r#""optional":true"#, r#""optional":false"#, 1)
        .replacen(r#""targets":[]"#, &format!(r#""targets":{targets}"#), 1);
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("required-feature package fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots(&package, VersionScope::Any),
            vec!["ktstr-bin-tests", "ktstr-tests"],
            "semantically named empty source gates are enabled for integration-test and \
             test-enabled binary harnesses, while unrelated empty/composite gates cannot \
             broaden inference",
        );
    }

    #[test]
    fn legacy_target_name_helpers_do_not_guess_missing_rustc_cfgs() {
        let json = optional_ktstr_package_json(
            "scheduler",
            "1.0.0",
            "=0.42.0",
            r#"{"ktstr-tests":["dep:ktstr"]}"#,
        )
        .replacen(
            r#""target":null"#,
            r#""target":"cfg(target_os = \"linux\")""#,
            1,
        );
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("target-specific fixture deserializes");
        assert!(
            infer_ktstr_feature_roots(&package, VersionScope::Any).is_empty(),
            "the legacy context-free helper must not guess whether a target-specific dependency is active",
        );
        assert!(
            infer_ktstr_feature_roots_for_target(
                &package,
                VersionScope::Any,
                Some("x86_64-unknown-linux-gnu"),
            )
            .is_empty(),
            "the legacy target-name-only helper deliberately lacks rustc's full cfg set; \
             production inference uses TargetContext with rustc-reported cfgs",
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_workspace_host_cfg_inference_and_package_scoped_feature_resolution() {
        let workspace = real_feature_workspace();
        let manifest = workspace.path().join("Cargo.toml");
        let args = strings(&[
            "--manifest-path",
            manifest.to_str().expect("UTF-8 fixture path"),
            "-p",
            "selected",
            "--features",
            "ktstr-tests",
        ]);
        let target = effective_target_context(&args).expect("effective host target");
        let manifests = query_metadata_for_target(&args, MetadataMode::NoDeps, &target)
            .expect("manifest metadata");
        assert_eq!(
            selected_activations_for_context(&manifests, &args, VersionScope::Any, Some(&target),),
            vec![PackageFeatureActivation {
                package: "selected".to_string(),
                features: vec!["ktstr-tests".to_string()],
            }],
            "the host-matching cfg(target_os = linux) ktstr edge is inferred",
        );

        let windows = TargetContext::named(
            "x86_64-pc-windows-msvc",
            vec![
                Cfg::Name("windows".to_string()),
                Cfg::KeyPair("target_os".to_string(), "windows".to_string()),
            ],
        );
        assert!(
            selected_activations_for_context(&manifests, &args, VersionScope::Any, Some(&windows),)
                .is_empty(),
            "the opposite target must not activate a Linux-only ktstr edge",
        );

        let resolved = query_resolved_metadata(&args, &args, &manifests, &target)
            .expect("selection-scoped resolved metadata");
        let selected = package_node(&resolved, "selected");
        assert!(
            selected
                .features
                .iter()
                .any(|feature| feature == "ktstr-tests")
        );
        assert!(
            ["default", "selected-default"]
                .iter()
                .all(|feature| selected.features.iter().any(|active| active == feature)),
            "selected package defaults remain active: {:?}",
            selected.features,
        );
        assert!(
            selected.deps.iter().any(|dependency| {
                resolved
                    .packages
                    .iter()
                    .find(|package| package.id == dependency.pkg)
                    .is_some_and(|package| {
                        package.name.as_str() == "ktstr"
                            && package.version == Version::parse("0.42.0").unwrap()
                    })
            }),
            "the selected package links current ktstr",
        );
        let unrelated = package_node(&resolved, "unrelated");
        assert!(
            unrelated.features.is_empty()
                && unrelated.deps.iter().all(|dependency| {
                    resolved
                        .packages
                        .iter()
                        .find(|package| package.id == dependency.pkg)
                        .is_none_or(|package| package.name.as_str() != "ktstr")
                }),
            "unqualified --features is metadata-scoped to -p selected",
        );

        let no_default_args = strings(&[
            "--manifest-path",
            manifest.to_str().expect("UTF-8 fixture path"),
            "-p",
            "selected",
            "--features",
            "ktstr-tests",
            "--no-default-features",
        ]);
        let no_default =
            query_resolved_metadata(&no_default_args, &no_default_args, &manifests, &target)
                .expect("no-default selection metadata");
        let selected = package_node(&no_default, "selected");
        assert!(
            selected
                .features
                .iter()
                .any(|feature| feature == "ktstr-tests")
                && !selected.features.iter().any(|feature| feature == "default")
                && !selected
                    .features
                    .iter()
                    .any(|feature| feature == "selected-default"),
            "caller --no-default-features remains authoritative: {:?}",
            selected.features,
        );
    }

    #[test]
    fn real_workspace_package_scopes_caller_all_features_for_metadata_only() {
        let workspace = real_feature_workspace();
        let manifest = workspace.path().join("Cargo.toml");
        let args = strings(&[
            "--manifest-path",
            manifest.to_str().expect("UTF-8 fixture path"),
            "-p",
            "selected",
            "--all-features",
        ]);
        let target = effective_target_context(&args).expect("effective host target");
        let manifests = query_metadata_for_target(&args, MetadataMode::NoDeps, &target)
            .expect("manifest metadata");
        let resolved = query_resolved_metadata(&args, &args, &manifests, &target)
            .expect("selection-scoped all-features metadata");
        let selected = package_node(&resolved, "selected");
        assert!(
            ["default", "extra", "ktstr-tests", "selected-default"]
                .iter()
                .all(|feature| selected.features.iter().any(|active| active == feature)),
            "all selected-package features are active: {:?}",
            selected.features,
        );
        let unrelated = package_node(&resolved, "unrelated");
        assert!(
            unrelated.features.is_empty()
                && unrelated.deps.iter().all(|dependency| {
                    resolved
                        .packages
                        .iter()
                        .find(|package| package.id == dependency.pkg)
                        .is_none_or(|package| package.name.as_str() != "ktstr")
                }),
            "caller --all-features must not become workspace-global in metadata",
        );
        assert_eq!(
            inject_feature_activations(args.clone(), &activations()),
            args,
            "caller all-features remains untouched; no automatic all-features is introduced",
        );
    }

    #[test]
    fn named_target_specific_ktstr_matches_only_explicit_cargo_target() {
        let json = optional_ktstr_package_json(
            "scheduler",
            "1.0.0",
            "=0.42.0",
            r#"{"ktstr-tests":["dep:ktstr"]}"#,
        )
        .replacen(
            r#""target":null"#,
            r#""target":"aarch64-unknown-linux-gnu""#,
            1,
        );
        let package: cargo_metadata::Package =
            serde_json::from_str(&json).expect("named-target fixture deserializes");
        assert_eq!(
            infer_ktstr_feature_roots_for_target(
                &package,
                VersionScope::Any,
                Some("aarch64-unknown-linux-gnu"),
            ),
            vec!["ktstr-tests"],
        );
        assert!(
            infer_ktstr_feature_roots_for_target(
                &package,
                VersionScope::Any,
                Some("x86_64-unknown-linux-gnu"),
            )
            .is_empty(),
            "a dependency from another named target cannot activate verifier source",
        );
    }

    #[test]
    fn package_spec_parser_handles_common_exact_forms() {
        assert_eq!(package_spec_name("scx_layered"), Some("scx_layered"));
        assert_eq!(package_spec_name("=scx_layered"), Some("scx_layered"));
        assert_eq!(package_spec_name("scx_layered@1.1.2"), Some("scx_layered"));
        assert_eq!(
            package_spec_name("path+file:///w#scx_layered@1.1.2"),
            Some("scx_layered"),
        );
        assert_eq!(package_spec_name("scx_*"), None);
    }

    #[test]
    fn selection_supports_cargo_package_globs_and_full_ids() {
        let metadata = selection_metadata();
        let lavd = PackageFeatureActivation {
            package: "lavd".to_string(),
            features: vec!["verify".to_string()],
        };
        for spec in [
            "l*",
            "l?vd",
            "l[ae]vd",
            "lavd@1",
            "lavd@1.0",
            "lavd@1.0.0",
            "lavd:1.0",
            "path+file:///w/lavd#lavd@1.0.0",
            "lavd 1.0.0 (path+file:///w/lavd)",
        ] {
            assert_eq!(
                selected_activations(&metadata, &strings(&["-p", spec]), VersionScope::Any,),
                vec![lavd.clone()],
                "Cargo package spec {spec:?} should select lavd",
            );
        }
        assert_eq!(
            selected_activations(&metadata, &strings(&["-p=lavd"]), VersionScope::Any),
            vec![lavd],
            "Cargo's short equals package form is normalized",
        );
        assert_eq!(
            selected_activations(
                &metadata,
                &strings(&["-p=lavd 1.0.0 (path+file:///w/lavd)"]),
                VersionScope::Any,
            )
            .len(),
            1,
            "short equals also accepts Cargo's canonical full package ID",
        );
        assert!(
            selected_activations(
                &metadata,
                &strings(&["-p", "lavd@9.9.9"]),
                VersionScope::Any,
            )
            .is_empty(),
            "an exact version mismatch must not select a same-named package",
        );
    }

    #[test]
    fn workspace_exclusions_support_cargo_package_globs_and_full_ids() {
        let metadata = selection_metadata();
        let cosmos = vec![PackageFeatureActivation {
            package: "cosmos".to_string(),
            features: vec!["ktstr-tests".to_string()],
        }];
        for spec in [
            "l*",
            "l?vd",
            "l[ae]vd",
            "lavd@1",
            "lavd@1.0",
            "lavd@1.0.0",
            "lavd:1.0",
            "path+file:///w/lavd#lavd@1.0.0",
            "lavd 1.0.0 (path+file:///w/lavd)",
        ] {
            assert_eq!(
                selected_activations(
                    &metadata,
                    &strings(&["--workspace", "--exclude", spec]),
                    VersionScope::Any,
                ),
                cosmos,
                "Cargo exclusion spec {spec:?} should exclude lavd",
            );
        }
    }

    #[test]
    fn metadata_passthrough_stops_before_test_binary_arguments() {
        assert_eq!(
            metadata_passthrough_options(&strings(&[
                "--locked",
                "--manifest-path",
                "consumer/Cargo.toml",
                "--features",
                "manual",
                "--",
                "--offline",
            ])),
            strings(&["--locked", "--manifest-path", "consumer/Cargo.toml"]),
        );
    }

    #[test]
    fn cargo_context_replay_is_scoped_and_rebases_changed_current_dirs() {
        let args = strings(&[
            "--manifest-path",
            "consumer/Cargo.toml",
            "--locked",
            "--config",
            "config/ci.toml",
            "--config=net.offline=true",
            "--config",
            "patch.crates-io.ktstr.path='../ktstr'",
            "-Z",
            "bindeps",
            "--target",
            "aarch64-unknown-linux-gnu",
            "--target-dir=artifacts",
            "--features",
            "consumer/ktstr-tests",
            "--all-features",
            "--no-default-features",
            "--ignore-rust-version",
            "--",
            "--offline",
        ]);
        assert_eq!(
            metadata_passthrough_options(&args),
            strings(&[
                "--manifest-path",
                "consumer/Cargo.toml",
                "--locked",
                "--config",
                "config/ci.toml",
                "--config=net.offline=true",
                "--config",
                "patch.crates-io.ktstr.path='../ktstr'",
                "-Z",
                "bindeps",
                "--filter-platform",
                "aarch64-unknown-linux-gnu",
            ]),
            "top-level metadata keeps its own manifest and current-directory semantics",
        );
        assert_eq!(
            declaring_metadata_options(&args, Path::new("/invoke")),
            strings(&[
                "--locked",
                "--config",
                "/invoke/config/ci.toml",
                "--config=net.offline=true",
                "--config",
                r#"patch.crates-io.ktstr.path="/invoke/../ktstr""#,
                "-Z",
                "bindeps",
                "--filter-platform",
                "aarch64-unknown-linux-gnu",
            ]),
            "declaration metadata omits the outer manifest and rebases config paths",
        );
        assert_eq!(
            scheduler_build_options(&args, Path::new("/invoke")),
            strings(&[
                "--locked",
                "--config",
                "/invoke/config/ci.toml",
                "--config=net.offline=true",
                "--config",
                r#"patch.crates-io.ktstr.path="/invoke/../ktstr""#,
                "-Z",
                "bindeps",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--target-dir=/invoke/artifacts",
                "--ignore-rust-version",
            ]),
            "scheduler builds replay only resolution/platform controls, never consumer features",
        );
    }

    #[test]
    fn cargo_build_dir_config_is_rebased_with_scheduler_current_dir() {
        assert_eq!(
            rebase_config_value(
                "build.build-dir='artifacts/build'",
                Some(Path::new("/invoke")),
            ),
            r#"build.build-dir="/invoke/artifacts/build""#,
        );
        assert_eq!(
            rebase_config_value(
                r#"build.build-dir="/absolute/build""#,
                Some(Path::new("/invoke")),
            ),
            r#"build.build-dir="/absolute/build""#,
        );
    }

    #[test]
    fn custom_target_json_is_rebased_for_build_but_not_metadata_filtering() {
        let args = strings(&["--target", "targets/custom.json"]);
        assert!(
            metadata_passthrough_options(&args).is_empty(),
            "a JSON path is not a cargo-platform expression",
        );
        assert_eq!(
            scheduler_build_options(&args, Path::new("/invoke")),
            strings(&["--target", "/invoke/targets/custom.json"]),
        );
        assert_eq!(
            custom_target_name("targets/custom.json").expect("custom target name"),
            "custom",
            "Cargo matches exact target tables against the JSON file stem",
        );
        let context = TargetContext::custom(
            custom_target_name("targets/custom.json").unwrap(),
            vec![Cfg::KeyPair("target_os".to_string(), "linux".to_string())],
        );
        assert!(context.matches_platform(&Platform::from_str("custom").unwrap()));
        assert!(
            context.matches_platform(&Platform::from_str(r#"cfg(target_os = "linux")"#).unwrap())
        );
    }

    #[test]
    fn metadata_resolution_options_replay_cargo_features_before_separator() {
        let args = strings(&[
            "--locked",
            "--features",
            "manual",
            "-Fshort",
            "-F",
            "split",
            "--features=equals",
            "--no-default-features",
            "--all-features",
            "--",
            "--features",
            "test-binary",
        ]);
        assert_eq!(
            metadata_resolution_options(&args),
            strings(&[
                "--features",
                "manual",
                "-Fshort",
                "-F",
                "split",
                "--features=equals",
                "--no-default-features",
                "--all-features",
            ]),
        );
        assert_eq!(
            metadata_other_options(&args, MetadataMode::Default),
            strings(&[
                "--locked",
                "--features",
                "manual",
                "-Fshort",
                "-F",
                "split",
                "--features=equals",
                "--no-default-features",
                "--all-features",
            ]),
        );
        assert_eq!(
            metadata_other_options(&args, MetadataMode::NoDeps),
            strings(&["--locked"]),
        );
    }
}
