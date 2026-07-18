//! Targeted Cargo feature discovery for downstream ktstr test binaries.
//!
//! A consumer commonly declares ktstr as an optional dependency and gates its
//! test registry behind a feature such as `ktstr-tests = ["dep:ktstr"]`.
//! Cargo metadata exposes both the optional dependency declaration and the
//! complete feature graph even when that feature is disabled. This module
//! follows only ktstr-specific feature chains and emits package-qualified
//! selectors, avoiding a broad `--all-features` build.

use std::collections::{HashMap, HashSet};

use cargo_metadata::semver::{Version, VersionReq};
use cargo_metadata::{Metadata, MetadataCommand};
use glob::Pattern;

/// How much dependency resolution a metadata caller needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetadataMode {
    /// Workspace manifests only. This is sufficient for ordinary feature
    /// inference and avoids resolving every optional dependency.
    NoDeps,
    /// Cargo's normal, requested-feature resolve graph. `test` and `coverage`
    /// use this for their version guard; when inference adds a previously
    /// inactive optional ktstr, they resolve the targeted result once more.
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

/// The Cargo arguments relevant to a metadata preflight.
///
/// Feature selection is deliberately omitted: package manifests expose every
/// feature definition without activating it. Stop at `--`, after which tokens
/// belong to the test binary rather than Cargo.
pub(crate) fn metadata_passthrough_options(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            break;
        }
        if matches!(arg.as_str(), "--locked" | "--offline" | "--frozen") {
            out.push(arg.clone());
        } else if matches!(arg.as_str(), "--config" | "--manifest-path") {
            out.push(arg.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                out.push(value.clone());
            }
        } else if arg.starts_with("--config=") || arg.starts_with("--manifest-path=") {
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

fn metadata_other_options(args: &[String], mode: MetadataMode) -> Vec<String> {
    let mut options = metadata_passthrough_options(args);
    if mode == MetadataMode::Default {
        options.extend(metadata_resolution_options(args));
    }
    options
}

/// Run the one metadata preflight used for feature inference.
///
/// `cargo_path("cargo")` is load-bearing for local development: unlike
/// cargo_metadata's `$CARGO` default, it honors a PATH cargo wrapper used to
/// patch crates.io ktstr to a checkout.
pub(crate) fn query_metadata(args: &[String], mode: MetadataMode) -> Result<Metadata, String> {
    let mut command = MetadataCommand::new();
    command
        .cargo_path("cargo")
        .other_options(metadata_other_options(args, mode));
    match mode {
        MetadataMode::NoDeps => {
            command.no_deps();
        }
        MetadataMode::Default => {}
    }
    command
        .exec()
        .map_err(|error| format!("cargo metadata failed: {error}"))
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
pub(crate) fn infer_ktstr_feature_roots(
    package: &cargo_metadata::Package,
    scope: VersionScope<'_>,
) -> Vec<String> {
    let aliases = package
        .dependencies
        .iter()
        .filter(|dependency| {
            dependency.name == "ktstr"
                && dependency.optional
                // Cargo features are package-global, while a target-specific
                // dependency may be absent for the requested host/target. A
                // metadata-only preflight does not have Cargo's full cfg set,
                // so leave such gates explicit instead of risking activation
                // of source that cannot link its ktstr dependency.
                && dependency.target.is_none()
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

    roots.into_iter().map(ToString::to_string).collect()
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

fn explicit_package_exclusion_specs(args: &[String]) -> Vec<String> {
    let args = cargo_args(args);
    let mut excluded = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let spec = if matches!(arg.as_str(), "--exclude" | "--exclude-from-test") {
            index += 1;
            args.get(index).map(String::as_str)
        } else {
            arg.strip_prefix("--exclude=")
                .or_else(|| arg.strip_prefix("--exclude-from-test="))
        };
        if let Some(spec) = spec {
            excluded.push(spec.to_string());
        }
        index += 1;
    }
    excluded
}

#[cfg(test)]
pub(crate) fn explicit_package_exclusions(args: &[String]) -> HashSet<String> {
    explicit_package_exclusion_specs(args)
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
    let exclusion_specs = explicit_package_exclusion_specs(args);

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
pub(crate) fn selected_activations(
    metadata: &Metadata,
    args: &[String],
    scope: VersionScope<'_>,
) -> Vec<PackageFeatureActivation> {
    let mut activations = selected_workspace_packages(metadata, args)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|package| {
            let features = infer_ktstr_feature_roots(package, scope);
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

/// Discover and inject targeted ktstr test features for an ordinary Cargo test
/// build/run from a manifest-only metadata pass.
pub(crate) fn augment_test_features(args: Vec<String>) -> Result<Vec<String>, String> {
    if explicit_all_features(&args) {
        return Ok(args);
    }
    let metadata = query_metadata(&args, MetadataMode::NoDeps)?;
    Ok(augment_test_features_from_metadata(args, &metadata))
}

/// Inject ordinary test features from metadata a caller already queried.
pub(crate) fn augment_test_features_from_metadata(
    args: Vec<String>,
    metadata: &Metadata,
) -> Vec<String> {
    let activations = selected_activations(metadata, &args, VersionScope::Any);
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
    fn target_specific_optional_ktstr_remains_explicit() {
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
            "metadata inference must not guess whether a target-specific dependency is active",
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
