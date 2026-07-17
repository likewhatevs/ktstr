// Shared kernel directory resolution.
//
// This file lives in two places: `src/kernel_path.rs` (the canonical
// source in the parent ktstr crate, included into its `build.rs` via
// `include!("src/kernel_path.rs")` and exposed to its lib via
// `pub mod kernel_path`) and `ktstr-macros/src/kernel_path.rs` (the
// bundled mirror, declared `mod kernel_path` in the macros crate's
// `src/lib.rs` and verified byte-identical to the canonical by
// `ktstr-macros/build.rs`'s drift check). The `include!` in the
// parent's build.rs is deliberate — build.rs runs before the crate
// compiles, so it cannot `use ktstr::...`. Duplicating the resolution
// logic across the three consumers (parent build.rs, parent lib,
// macros crate) would drift between build-time BTF discovery
// (vmlinux.h generation), run-time kernel selection, and the macros
// crate's parse-time KernelId construction.
//
// Constraints that every edit to this file must satisfy — breaking
// any of them surfaces as either a cryptic build-script error or a
// runtime/build-time behaviour mismatch:
//
// 1. **No non-std imports.** build.rs has its own dependency graph
//    (`libbpf-cargo`, `tempfile`, etc.). A `use foo::bar` here would
//    compile inside lib.rs (via the `pub mod` path) but fail inside
//    build.rs because build.rs hasn't declared `foo` as a build-dep.
// 2. **No `pub(crate)` items.** `pub(crate)` is meaningless inside
//    an `include!`'d fragment — build.rs isn't a crate, so the item
//    resolves at crate-root visibility there. Use `pub` for items
//    build.rs needs, `fn` (private) for items lib.rs alone uses.
// 3. **`#[cfg(test)]` blocks may use non-std test helpers freely.**
//    Cargo does not set `cfg(test)` when compiling build scripts, so
//    `#[cfg(test)]` items inside this file are simply elided from the
//    build.rs view of the fragment — `tempfile`, `proptest`, etc. are
//    safe to import inside `#[cfg(test)] mod tests { ... }`. The
//    std-only rule (#1 above) applies to non-`cfg(test)` items only.
// 4. **All functions are pure.** Callers supply inputs and handle
//    caching — no global state, no `std::env::set_var`, no FS
//    writes outside the caller-provided paths. Pure is what makes
//    the double-consumer (build + runtime) safe.
// 5. **Bundled into ktstr-macros.** Everything before the first
//    `#[cfg(test)]` is mirrored verbatim between
//    `src/kernel_path.rs` (the canonical source) and
//    `ktstr-macros/src/kernel_path.rs` (the bundled mirror);
//    `ktstr-macros/build.rs` panics on any drift. Edits to
//    non-test items must update both copies in lock-step. The
//    `#[cfg(test)]` portion lives in `src/kernel_path.rs` only.

/// Human-readable enumeration of every form `KernelId::parse` accepts.
/// The macro-time rejection in `declare_scheduler!(kernels = […])`
/// cites this const verbatim. The runtime cache-lookup bails in
/// `ktstr` / `cargo-ktstr` cite the `KTSTR_KERNEL_HINT` const, a manual
/// mirror of this same wording — const composition cannot `concat!` a
/// `const &str`, only literals — so keep the two in sync when either
/// changes.
pub const KERNEL_ID_GRAMMAR: &str = "exact version (`6.14`), inclusive range (`6.14..7.0` or \
     `6.14..=7.0`), git source (`git+URL#tag=NAME`, `git+URL#branch=NAME`, or \
     `git+URL#sha=<40-hex>`), absolute or `~`-prefixed path, local kernel package \
     (`*.rpm`, `*.deb`, or `*.pkg.tar.zst`), distro kernel (`fedora`/`fedora-44`/`f44`, \
     `ubuntu`/`ubuntu-24.04`, `amazonlinux`/`amazonlinux-2023`/`al2023`, \
     `steamos`/`steamos-3.8`, `gke`/`gke-129`), or cache key";

/// Kernel identifier: filesystem path, version string, cache key,
/// stable-release range, or git source.
///
/// Parsing heuristic (see [`KernelId::parse`]):
/// - Starts with `git+`: [`KernelId::Git`] (form `git+URL#tag=NAME` /
///   `git+URL#branch=NAME` / `git+URL#sha=<40-hex>`)
/// - Ends with `.rpm`, `.deb`, or `.pkg.tar.zst`: [`KernelId::Package`]
/// - Contains `/` (without a `git+` prefix) or starts with `.` or `~`:
///   [`KernelId::Path`]
/// - Contains `..` between two version-shaped tokens:
///   [`KernelId::Range`] (inclusive on both endpoints)
/// - Matches `MAJOR.MINOR[.PATCH][-rcN]`: [`KernelId::Version`]
/// - A distro name (`fedora` / `ubuntu` / `amazonlinux` /
///   `steamos` / `gke`, an explicit-release `NAME-REL`, or shorthand `f44` /
///   `al2023`): [`KernelId::Distro`]
/// - Otherwise: [`KernelId::CacheKey`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelId {
    /// Filesystem path to kernel source/build directory.
    Path(std::path::PathBuf),
    /// Kernel version string (e.g. "6.14.2", "6.15-rc3").
    Version(String),
    /// Cache key (e.g. "6.14.2-tarball-x86_64-kc...").
    CacheKey(String),
    /// Inclusive range of stable kernel versions, expanded against
    /// kernel.org's release index at resolve time. `start` and `end`
    /// are both [`KernelId::Version`]-shaped strings (e.g. "6.10",
    /// "6.13"); the resolver fans this out to every release in
    /// [start, end] inclusive on both endpoints regardless of whether
    /// the parser saw `..` or `..=`. A version present in the range
    /// but missing from the upstream index is a hard error before any
    /// boot — partial expansions are not silently dropped. The
    /// `syntax_inclusive` flag preserves the original separator for
    /// round-trip [`std::fmt::Display`] and operator-facing error
    /// messages; it does not change resolution semantics.
    Range {
        /// Inclusive lower bound, version-shaped.
        start: String,
        /// Inclusive upper bound, version-shaped.
        end: String,
        /// `true` when the parser saw `..=` (or the construction site
        /// asked for it); `false` for the `..` form. Both are
        /// resolved as inclusive ranges; the flag exists so
        /// [`std::fmt::Display`] and the inverted-range error
        /// message round-trip the operator's typed form.
        syntax_inclusive: bool,
    },
    /// Git source: acquire the source at `git_ref` per `ref_kind`
    /// (tag / branch / sha), chosen explicitly by the operator's
    /// `#tag=` / `#branch=` / `#sha=` fragment — no DWIM. Stored
    /// verbatim by `KernelId::parse` with no remote contact. At
    /// cache-resolution time `resolve_git_kernel` resolves `git_ref`
    /// to its full commit hash (a kind-directed ls-remote) and probes
    /// the cache before fetching, so a re-run against an unchanged tip
    /// skips the download.
    ///
    /// Acquisition is routed by host (see `resolve_git_kernel` /
    /// `crate::fetch`):
    /// - GitHub (`github.com/OWNER/REPO`): a codeload `tar.gz` snapshot
    ///   of the RESOLVED COMMIT (the ls-remote-resolved commit for a
    ///   tag/branch, the sha itself for a sha) — no clone; the
    ///   exact-commit snapshot matches the cache key even if a branch
    ///   tip moves mid-resolve. A tag/branch whose ls-remote resolution
    ///   fails falls back to the clone path below (like a non-GitHub
    ///   source).
    /// - Non-GitHub: a kind-directed shallow clone — `Tag` fetches
    ///   `refs/tags/{git_ref}` (annotated tags peel to the commit),
    ///   `Branch` fetches `refs/heads/{git_ref}`. `Sha` is unsupported off
    ///   GitHub (gix cannot fetch a bare commit and the remote lacks
    ///   allow-sha-in-want) and errors.
    Git {
        /// Remote URL (https or git@). GitHub sources are fetched from
        /// codeload; non-GitHub sources are shallow-cloned from here.
        url: String,
        /// The ref value after `kind=` (verbatim, no `refs/` prefix) for
        /// `Tag` / `Branch` / `Sha`; the whole unrecognized fragment for
        /// `Unknown`. For `ref_kind == Sha` this is the 40-hex commit id.
        git_ref: String,
        /// Which git namespace `git_ref` names, from the explicit
        /// `#tag=` / `#branch=` / `#sha=` selector. `Unknown` marks a
        /// bare `#REF` or unrecognized selector that `validate` rejects.
        ref_kind: GitRefKind,
    },
    /// Local kernel package: an `.rpm`, `.deb`, or pacman
    /// `.pkg.tar.zst` file on disk, to be unpacked for its prebuilt
    /// kernel image and modules. Classified by [`KernelId::parse`] on
    /// the case-sensitive `.rpm` / `.deb` / `.pkg.tar.zst` suffix
    /// ahead of the path check, so `./foo.rpm`, `/abs/foo.deb`, and a
    /// bare `foo.rpm` all land here rather than as
    /// [`KernelId::Path`]. The `path` is `~`-expanded identically to
    /// the Path variant.
    Package {
        /// Path to the `.rpm` / `.deb` / `.pkg.tar.zst` file.
        path: std::path::PathBuf,
    },
    /// Distro-provided prebuilt kernel: download a specific build for a
    /// named distribution and (optional) release. A bare distro name
    /// (`fedora`) leaves `release` `None` — the resolver picks the
    /// distro's default; an explicit release (`fedora-44`, `f44`,
    /// `ubuntu-24.04`, `amazonlinux-2023`, `al2023`,
    /// `steamos-3.8`, `gke-129`) pins one. The
    /// release string's grammar is distro-specific and enforced by
    /// [`KernelId::validate`], not `parse`: a distro name with a
    /// malformed release parses to this variant and is rejected at
    /// validate time (mirroring the git arm's parse/validate split),
    /// never silently demoted to [`KernelId::CacheKey`].
    Distro {
        /// Which distribution.
        kind: DistroKind,
        /// Release identifier (e.g. "44", "24.04", "2023"), or `None`
        /// for the distro's default.
        release: Option<String>,
    },
}

/// Which distribution a [`KernelId::Distro`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistroKind {
    /// Fedora (`fedora`, `fedora-44`, or shorthand `f44`).
    Fedora,
    /// Ubuntu (`ubuntu`, `ubuntu-24.04`).
    Ubuntu,
    /// Amazon Linux (`amazonlinux`, `amazonlinux-2023`, or shorthand
    /// `al2023`).
    AmazonLinux,
    /// SteamOS (`steamos`, or a pinned channel `steamos-3.8`).
    /// x86_64 only — Valve publishes no other architecture.
    SteamOs,
    /// Google Kubernetes Engine COS node image (`gke`, or a constrained
    /// COS milestone such as `gke-129`). Bare `gke` tracks the newest
    /// GKE-promoted image; the milestone form tracks the newest promoted
    /// revision within that milestone. x86_64 only today.
    Gke,
}

/// Which git ref namespace a [`KernelId::Git`]'s `git_ref` names,
/// chosen explicitly by the operator via the `#tag=` / `#branch=` /
/// `#sha=` fragment — never DWIM-inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitRefKind {
    /// `#tag=v6.14` → an annotated or lightweight tag (`refs/tags/`).
    Tag,
    /// `#branch=for-next` → a branch head (`refs/heads/`).
    Branch,
    /// `#sha=<40-hex>` → a raw commit id.
    Sha,
    /// A bare `#REF` fragment (no `kind=`) or an unrecognized selector;
    /// carried so [`KernelId::validate`] can emit an actionable error.
    /// Never resolved.
    Unknown,
}

impl KernelId {
    /// Parse a string into a kernel identifier.
    ///
    /// Recognizes (in order):
    /// - `git+`-prefixed → [`KernelId::Git`]. ANY `git+…` string is a
    ///   Git source (the `git+` prefix takes precedence over the range
    ///   and `/`-contains tests below), so a typo such as a missing
    ///   `#fragment` never silently becomes a `Path`. The fragment
    ///   selects the ref kind: `#tag=NAME` / `#branch=NAME` /
    ///   `#sha=<40-hex>`; a missing/empty fragment or unrecognized
    ///   selector yields `GitRefKind::Unknown`, and an empty URL an
    ///   empty `url` — both of which [`KernelId::validate`] rejects
    ///   with an actionable error rather than the resolver later
    ///   reporting a confusing "path not found".
    /// - `.rpm`/`.deb`/`.pkg.tar.zst`-suffixed → [`KernelId::Package`].
    ///   Checked ahead of the path test so `./foo.rpm`, `/abs/foo.deb`,
    ///   and a bare `foo.rpm` all classify as a package rather than a
    ///   path.
    /// - `START..=END` or `START..END` where both endpoints are
    ///   version-shaped → [`KernelId::Range`]. The endpoints are
    ///   ALWAYS inclusive — both `..` and `..=` spellings produce a
    ///   closed range, regardless of Rust's exclusive-`..` /
    ///   inclusive-`..=` distinction. Both forms are accepted so test
    ///   authors and CLI users can write whichever feels natural.
    /// - `/`-containing or `.`/`~`-prefixed → [`KernelId::Path`].
    /// - Version-shaped → [`KernelId::Version`].
    /// - A distro name (`fedora` / `ubuntu` / `amazonlinux` /
    ///   `steamos`, an explicit-release `NAME-REL`, or
    ///   shorthand `f44` / `al2023`) →
    ///   [`KernelId::Distro`]. The release grammar is checked by
    ///   `validate`, not here — a malformed release still classifies as
    ///   `Distro` (not `CacheKey`) so `validate` can reject it.
    /// - Anything else → [`KernelId::CacheKey`].
    pub fn parse(s: &str) -> Self {
        if let Some(rest) = s.strip_prefix("git+") {
            // ANY `git+…` string is a Git source — split off the
            // `#fragment` (a missing one leaves an empty fragment) and
            // NEVER fall through to Path/CacheKey, so a typo such as
            // `git+URL` (no fragment) surfaces the actionable
            // git-grammar error from `validate` rather than a confusing
            // "path not found". `parse` returns `Self`, not `Result`, so
            // all structural rejection (Unknown kind, empty url/ref, bad
            // sha) is deferred to `validate`.
            let (url, frag) = match rest.rsplit_once('#') {
                Some((url, frag)) => (url, frag),
                None => (rest, ""),
            };
            // Explicit ref-kind grammar: `#tag=NAME` / `#branch=NAME` /
            // `#sha=<40-hex>`. A bare `#REF` (no `kind=`), an empty
            // fragment, or an unrecognized selector parses as `Unknown`.
            let (ref_kind, git_ref) = match frag.split_once('=') {
                Some(("tag", v)) => (GitRefKind::Tag, v.to_string()),
                Some(("branch", v)) => (GitRefKind::Branch, v.to_string()),
                Some(("sha", v)) => (GitRefKind::Sha, v.to_string()),
                _ => (GitRefKind::Unknown, frag.to_string()),
            };
            return KernelId::Git {
                url: url.to_string(),
                git_ref,
                ref_kind,
            };
        }
        // Case-sensitive suffix, checked before the path arm so a
        // package spec with directory separators (`/abs/foo.deb`) or a
        // `.`-prefix (`./foo.rpm`) still classifies as a package.
        // Classification requires a non-empty file stem before the
        // extension: a bare `.deb` / `.rpm` / `.pkg.tar.zst` (or a
        // path ending in such a component) is a HIDDEN-FILE name, not
        // a package spec — it falls through to the Path arm via its
        // `.` prefix or `/` content, preserving the dot-prefix → Path
        // contract the property tests pin.
        let base = s.rsplit('/').next().unwrap_or(s);
        if [".rpm", ".deb", ".pkg.tar.zst"]
            .iter()
            .any(|ext| base.ends_with(ext) && base.len() > ext.len())
        {
            return KernelId::Package {
                path: expand_tilde(s),
            };
        }
        if let Some((start, end)) = s.split_once("..=")
            && _is_version_string(start)
            && _is_version_string(end)
        {
            return KernelId::Range {
                start: start.to_string(),
                end: end.to_string(),
                syntax_inclusive: true,
            };
        }
        if let Some((start, end)) = s.split_once("..")
            && _is_version_string(start)
            && _is_version_string(end)
        {
            return KernelId::Range {
                start: start.to_string(),
                end: end.to_string(),
                syntax_inclusive: false,
            };
        }
        if s.contains('/') || s.starts_with('.') || s.starts_with('~') {
            return KernelId::Path(expand_tilde(s));
        }
        if _is_version_string(s) {
            return KernelId::Version(s.to_string());
        }
        if let Some(distro) = parse_distro(s) {
            return distro;
        }
        KernelId::CacheKey(s.to_string())
    }

    /// Parse a comma-separated list of kernel specs into a vector of
    /// identifiers. Empty entries are silently skipped (so trailing
    /// commas or repeated separators are forgiving). Each non-empty
    /// segment is fed through [`KernelId::parse`] verbatim — so
    /// `parse_list("6.10,git+URL#branch=main,/srv/linux")` returns three
    /// distinct variants. Deduplication is the resolver's
    /// responsibility (after canonicalization to a cache key); this
    /// function preserves order and duplicates as written.
    pub fn parse_list(s: &str) -> Vec<KernelId> {
        s.split(',')
            .map(str::trim)
            .filter(|seg| !seg.is_empty())
            .map(KernelId::parse)
            .collect()
    }

    /// Validate a parsed `KernelId` for resolve-time legality. Returns
    /// `Err(message)` when the identifier carries a structural problem
    /// the parser couldn't catch on its own — currently:
    ///
    /// - [`KernelId::Range`] with `start > end` after numeric
    ///   component-wise comparison. The parser cannot reject this at
    ///   parse time because both endpoints are valid version strings
    ///   in isolation; the inversion only surfaces when the two are
    ///   compared.
    /// - [`KernelId::Git`] with an `Unknown` ref kind, an empty
    ///   url/ref, or a `#sha=` that isn't a full 40-hex id.
    /// - [`KernelId::Distro`] whose explicit release does not match the
    ///   distro's grammar (Fedora `\d{2,3}`, Ubuntu `YY.MM`, Amazon
    ///   Linux `\d{4}`, SteamOS
    ///   `\d{1,2}.\d{1,2}`). As with git, the parser defers the
    ///   release
    ///   check here so a malformed release classifies as `Distro`
    ///   rather than silently becoming a `CacheKey`.
    ///
    /// The remaining variants (Path, Version, CacheKey, Package) always
    /// return `Ok(())` — this is a hook for per-variant invariants, not
    /// a general-purpose validator.
    /// Use `Result<(), String>` rather than `anyhow::Result` because
    /// this file is included from `build.rs` (see file header rule
    /// #1, no non-std imports outside `cfg(test)`).
    ///
    /// Comparison semantics: each endpoint decomposes to a
    /// `(major, minor, patch, rc)` tuple where missing patch maps to
    /// `0` and missing `-rc` maps to `u64::MAX` so a release
    /// (`6.10`) sorts strictly above any pre-release (`6.10-rc3`) of
    /// the same major.minor.patch. Inverted ranges include
    /// `7.0..6.99`, `6.10..6.5`, `6.10..6.10-rc3` (release > rc), and
    /// `6.10-rc3..6.10-rc1`. Equal endpoints (`6.10..6.10`) pass
    /// validation as a single-element range.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            KernelId::Range {
                start,
                end,
                syntax_inclusive,
            } => {
                let start_key = decompose_version_for_compare(start).ok_or_else(|| {
                    format!(
                        "kernel range start `{start}` is not a parseable version. \
                         Expected `MAJOR.MINOR[.PATCH][-rc<num>]` (e.g. \"6.10\", \
                         \"6.14.2\", \"6.15-rc3\"). Range examples: `6.10..6.15`, \
                         `6.10-rc1..=6.10`.",
                    )
                })?;
                // END is series-inclusive: a 2-component `MAJOR.MINOR`
                // END names the whole series (see `range_end_key`), so
                // the inversion check must use the SAME widened bound the
                // expansion does — otherwise a valid same-series range
                // like `6.14.5..6.14` (= 6.14.5 .. end of the 6.14
                // series) is falsely rejected as inverted.
                let end_key = range_end_key(end).ok_or_else(|| {
                    format!(
                        "kernel range end `{end}` is not a parseable version. \
                         Expected `MAJOR.MINOR[.PATCH][-rc<num>]` (e.g. \"6.10\", \
                         \"6.14.2\", \"6.15-rc3\"). Range examples: `6.10..6.15`, \
                         `6.10-rc1..=6.10`.",
                    )
                })?;
                if start_key > end_key {
                    let sep = if *syntax_inclusive { "..=" } else { ".." };
                    return Err(format!(
                        "inverted kernel range `{start}{sep}{end}`: start version is greater \
                         than end version. Swap the endpoints (`{end}{sep}{start}`) or use \
                         a single version (no range) to test just one release.",
                    ));
                }
                Ok(())
            }
            KernelId::Git {
                url,
                git_ref,
                ref_kind,
            } => {
                if url.is_empty() {
                    return Err(format!(
                        "git source `git+#{git_ref}`: empty URL — write \
                         `git+<url>#tag=<name>` (or `#branch=` / `#sha=`)."
                    ));
                }
                if *ref_kind == GitRefKind::Unknown {
                    return Err(format!(
                        "git source `git+{url}#{git_ref}`: the ref kind must be \
                         explicit — write `#tag=<name>`, `#branch=<name>`, or \
                         `#sha=<40-hex>` (a bare `#REF` is no longer accepted).",
                    ));
                }
                if git_ref.is_empty() {
                    return Err(format!(
                        "git source `git+{url}#...`: empty ref value — write \
                         `#tag=<name>`, `#branch=<name>`, or `#sha=<40-hex>`.",
                    ));
                }
                if *ref_kind == GitRefKind::Sha
                    && (git_ref.len() != 40 || !git_ref.bytes().all(|b| b.is_ascii_hexdigit()))
                {
                    return Err(format!(
                        "git source `git+{url}#sha={git_ref}`: a sha must be the full \
                         40-hex commit id (abbreviated shas can't be fetched); use \
                         `#tag=` or `#branch=` for a name.",
                    ));
                }
                Ok(())
            }
            KernelId::Distro { kind, release } => {
                let Some(rel) = release.as_deref() else {
                    return Ok(());
                };
                let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
                let (ok, expected) = match kind {
                    // 2–3 digits: `44` today, `100`+ future-proofed.
                    DistroKind::Fedora => (
                        (2..=3).contains(&rel.len()) && all_digits(rel),
                        "a 2- or 3-digit release number (e.g. `fedora-44` or `f44`)",
                    ),
                    // `YY.MM`, e.g. `24.04`.
                    DistroKind::Ubuntu => (
                        matches!(rel.split_once('.'), Some((y, m))
                            if y.len() == 2 && m.len() == 2 && all_digits(y) && all_digits(m)),
                        "a `YY.MM` release (e.g. `ubuntu-24.04`)",
                    ),
                    // 4-digit year, e.g. `2023`.
                    DistroKind::AmazonLinux => (
                        rel.len() == 4 && all_digits(rel),
                        "a 4-digit year release (e.g. `amazonlinux-2023` or `al2023`)",
                    ),
                    // `MAJOR.MINOR` channel version, e.g. `3.8`.
                    DistroKind::SteamOs => (
                        matches!(rel.split_once('.'), Some((maj, min))
                            if (1..=2).contains(&maj.len()) && (1..=2).contains(&min.len())
                                && all_digits(maj) && all_digits(min)),
                        "a `MAJOR.MINOR` channel version (e.g. `steamos-3.8`)",
                    ),
                    // COS milestone, e.g. `129`.
                    DistroKind::Gke => (
                        (2..=3).contains(&rel.len()) && all_digits(rel),
                        "a 2- or 3-digit COS milestone (e.g. `gke-129`)",
                    ),
                };
                if ok {
                    Ok(())
                } else {
                    Err(format!(
                        "distro kernel `{self}`: `{rel}` is not a valid \
                         {kind} release — expected {expected}.",
                        kind = kind.as_str(),
                    ))
                }
            }
            KernelId::Path(_)
            | KernelId::Version(_)
            | KernelId::CacheKey(_)
            | KernelId::Package { .. } => Ok(()),
        }
    }
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelId::Path(p) => write!(f, "{}", p.display()),
            KernelId::Version(v) => write!(f, "{v}"),
            KernelId::CacheKey(k) => write!(f, "{k}"),
            KernelId::Range {
                start,
                end,
                syntax_inclusive,
            } => {
                let sep = if *syntax_inclusive { "..=" } else { ".." };
                write!(f, "{start}{sep}{end}")
            }
            KernelId::Git {
                url,
                git_ref,
                ref_kind,
            } => match ref_kind {
                GitRefKind::Tag => write!(f, "git+{url}#tag={git_ref}"),
                GitRefKind::Branch => write!(f, "git+{url}#branch={git_ref}"),
                GitRefKind::Sha => write!(f, "git+{url}#sha={git_ref}"),
                // Round-trip a rejected bare fragment verbatim so
                // `parse(Display(x)) == x` still holds for the Unknown case.
                GitRefKind::Unknown => write!(f, "git+{url}#{git_ref}"),
            },
            KernelId::Package { path } => write!(f, "{}", path.display()),
            KernelId::Distro { kind, release } => {
                let name = kind.as_str();
                match release {
                    // Shorthand inputs (`f44`, `al2023`) render in long
                    // form; `parse(Display(x))` still yields an equal id
                    // (both spellings parse to the same variant).
                    Some(rel) => write!(f, "{name}-{rel}"),
                    None => write!(f, "{name}"),
                }
            }
        }
    }
}

impl DistroKind {
    /// The canonical long-form distro name.
    fn as_str(self) -> &'static str {
        match self {
            DistroKind::Fedora => "fedora",
            DistroKind::Ubuntu => "ubuntu",
            DistroKind::AmazonLinux => "amazonlinux",
            DistroKind::SteamOs => "steamos",
            DistroKind::Gke => "gke",
        }
    }
}

/// Classify a distro kernel spec, or `None` so the caller falls
/// through to [`KernelId::CacheKey`].
///
/// Recognizes the long form — an exact distro name (`fedora`, release
/// `None`) or `NAME-REL` (`fedora-44`) — and the shorthands `f<rel>`
/// (Fedora) and `al<rel>` (Amazon Linux), where `<rel>` is a run of
/// digits. A bare `al<digits>` (`al2023`, or a malformed `al9`) is
/// Amazon Linux; a non-digit remainder (`alma9`, whose `al`-stripped
/// tail `ma9` is not all digits) is not a distro spec and stays a
/// cache key.
/// The release portion is carried verbatim; its per-distro
/// digit grammar is enforced by [`KernelId::validate`], not here — so a
/// name with a malformed release (`fedora-abc`, `f4`) classifies as
/// `Distro` and surfaces an actionable validation error rather than a
/// silent cache-key miss (mirroring the git arm's parse/validate
/// split). A shorthand whose remainder is not all digits (`foo`,
/// `alpha`) is not a distro spec and stays a cache key.
fn parse_distro(s: &str) -> Option<KernelId> {
    for (name, kind) in [
        ("fedora", DistroKind::Fedora),
        ("ubuntu", DistroKind::Ubuntu),
        ("amazonlinux", DistroKind::AmazonLinux),
        ("steamos", DistroKind::SteamOs),
        ("gke", DistroKind::Gke),
    ] {
        if s == name {
            return Some(KernelId::Distro {
                kind,
                release: None,
            });
        }
        if let Some(rel) = s.strip_prefix(name).and_then(|r| r.strip_prefix('-')) {
            return Some(KernelId::Distro {
                kind,
                release: Some(rel.to_string()),
            });
        }
    }
    for (prefix, kind) in [("f", DistroKind::Fedora), ("al", DistroKind::AmazonLinux)] {
        if let Some(rel) = s.strip_prefix(prefix)
            && !rel.is_empty()
            && rel.bytes().all(|b| b.is_ascii_digit())
        {
            return Some(KernelId::Distro {
                kind,
                release: Some(rel.to_string()),
            });
        }
    }
    None
}

/// Check if a string matches a kernel version pattern.
///
/// Matches: `6` (bare major prefix), `6.14`, `6.14.2`, `6.15-rc3`,
/// `6.14.2-rc1`. Does not match: `v6.14` (git tag prefix), `6.`
/// (trailing dot), `6.14.2-tarball-x86_64-kc...` (cache key with
/// extra segments).
fn _is_version_string(s: &str) -> bool {
    let (version_part, rc_part) = match s.split_once("-rc") {
        Some((v, rc)) => (v, Some(rc)),
        None => (s, None),
    };

    // The part after -rc must be a non-empty digit string.
    if let Some(rc) = rc_part
        && (rc.is_empty() || !rc.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }

    let mut parts = version_part.split('.');

    // Major: required, non-empty digits.
    match parts.next() {
        Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {}
        _ => return false,
    }
    // Minor: OPTIONAL — a bare MAJOR (`6`) is a valid version prefix
    // that resolves to the highest patch across all minors (see
    // `crate::fetch::fetch_version_for_prefix`). If present it must be
    // non-empty digits; `6.` (trailing dot) is rejected as an empty
    // component.
    match parts.next() {
        None => {}
        Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {}
        _ => return false,
    }
    // Patch: optional, non-empty digits.
    if let Some(patch) = parts.next()
        && (patch.is_empty() || !patch.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    // No more segments allowed (rejects `1.2.3.4`).
    parts.next().is_none()
}

/// Decompose a version-shaped string into a `(major, minor, patch,
/// rc)` tuple suitable for `Ord` comparison. Returns `None` when the
/// input doesn't match the kernel-version grammar — same predicate as
/// [`_is_version_string`] but extracting numeric components rather
/// than just yes/no.
///
/// Comparison semantics:
/// - Missing patch defaults to `0` so `6.10` and `6.10.0` compare
///   equal.
/// - Missing `-rcN` defaults to `u64::MAX` so a release
///   (`6.10`, `6.10.5`) sorts strictly above any pre-release
///   (`6.10-rc3`, `6.10.5-rc1`) of the same `major.minor.patch`. A
///   future major/minor/patch bump still dominates because the tuple
///   is compared in declaration order — the rc-as-MAX trick only
///   resolves ties on the leading three components.
///
/// Used by [`KernelId::validate`] to detect inverted ranges
/// (`6.16..6.12`, `6.10..6.10-rc3`, `7.0..6.99`), and by
/// the `cli` module's range-expansion helper to filter and sort
/// kernel.org release rows that fall inside a `start..end` interval.
pub(crate) fn decompose_version_for_compare(s: &str) -> Option<(u64, u64, u64, u64)> {
    let (version_part, rc_part) = match s.split_once("-rc") {
        Some((v, rc)) => (v, Some(rc)),
        None => (s, None),
    };
    // rc must be a non-empty digit string when present.
    let rc: u64 = match rc_part {
        Some(rc) if rc.is_empty() || !rc.bytes().all(|b| b.is_ascii_digit()) => return None,
        Some(rc) => rc.parse().ok()?,
        None => u64::MAX,
    };
    let mut parts = version_part.split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    // Minor optional: a bare MAJOR (`6`) decomposes to `(major, 0, 0,
    // ..)` so it behaves as a series-floor prefix, consistent with
    // `_is_version_string` accepting a bare major.
    let minor: u64 = match parts.next() {
        None => 0,
        Some("") => return None,
        Some(m) => m.parse().ok()?,
    };
    let patch: u64 = match parts.next() {
        Some("") => return None,
        Some(p) => p.parse().ok()?,
        None => 0,
    };
    // Reject `1.2.3.4` and similar — only major.minor[.patch] is grammar.
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch, rc))
}

/// The upper-bound comparison key for a range's END endpoint,
/// series-inclusive: a 2-component `MAJOR.MINOR` (or bare-major) END
/// with no `-rc` names the WHOLE series, so its patch and rc slots are
/// widened to `u64::MAX`. [`decompose_version_for_compare`] alone maps a
/// missing patch to 0, which as an inclusive upper bound would exclude
/// every `6.14.N` (N >= 1) from an END of `6.14`. An explicit-patch END
/// (`6.14.2`) or an `-rc` END keeps its exact key. Shared by
/// [`KernelId::validate`]'s inversion check and the `cli` module's
/// range expansion (`range_bounds`) so the two agree on where a range
/// ends. START needs no such widening — its `.0` is the series floor.
pub(crate) fn range_end_key(end: &str) -> Option<(u64, u64, u64, u64)> {
    let key = decompose_version_for_compare(end)?;
    // Same predicate as `crate::fetch::is_major_minor_prefix`, inlined
    // because this file forbids non-std imports (file-header rule #1):
    // a 2-component (or bare-major) endpoint with no `-rc` is a whole
    // series and widens to its ceiling.
    if end.matches('.').count() < 2 && !end.contains("-rc") {
        Some((key.0, key.1, u64::MAX, u64::MAX))
    } else {
        Some(key)
    }
}

/// Expand a leading `~` or `~/...` in `s` against `$HOME` and
/// return the resulting [`std::path::PathBuf`]. Any other shape (no leading
/// `~`, `~user/...` for a different user, `$HOME` unset or empty)
/// passes through verbatim — the caller's downstream `is_dir()`
/// surfaces a regular "no such directory" error instead of being
/// silently rewritten.
///
/// Cases handled:
/// - `"~"` → `$HOME`
/// - `"~/"` → `$HOME` (same as bare `"~"`; the empty suffix
///   after `~/` yields no trailing separator)
/// - `"~/linux"` → `$HOME/linux`
/// - `"~user/..."` → unchanged (std has no `getpwnam`; a
///   different-user expansion would require shelling out, which
///   the file's "no non-std imports outside cfg(test)" rule
///   forbids; the operator who wants a peer's home dir can spell
///   it absolutely)
/// - any input not starting with `~` → unchanged
/// - `~`-prefix with `$HOME` unset / empty → unchanged (the
///   downstream `is_dir()` failure is the clearest error path
///   we can produce without a logging dep)
///
/// Pure with respect to filesystem writes; reads `$HOME` once. Env
/// reads are consistent with the existing
/// [`kernel_release_from_procfs`] pattern (FS read at resolve time)
/// and explicitly outside the file-header `std::env::set_var` ban.
///
/// Called from [`KernelId::parse`]'s Path arm so the Path variant
/// stores an absolute (or filesystem-resolvable) path. Without this,
/// `KernelId::parse("~/linux")` stores the literal `"~/linux"`,
/// which `is_dir()` rejects unconditionally — there is no shell to
/// perform the standard tilde expansion on the operator's behalf at
/// CLI invocation time.
fn expand_tilde(s: &str) -> std::path::PathBuf {
    // Bare `~` and `~/...` are the only shapes we expand. Anything
    // else falls through verbatim.
    if s != "~" && !s.starts_with("~/") {
        return std::path::PathBuf::from(s);
    }
    // `$HOME` empty or unset is treated identically to "no
    // expansion possible" — the caller's `is_dir()` check will
    // surface the missing-path error normally. We do NOT panic
    // here because `KernelId::parse` is `pub` and on a hot CLI
    // path; failing to expand a single arg is not a fatal
    // condition for the whole CLI.
    let home = match std::env::var("HOME") {
        Ok(h) if !h.is_empty() => h,
        _ => return std::path::PathBuf::from(s),
    };
    if s == "~" {
        return std::path::PathBuf::from(home);
    }
    // s starts with "~/", so the suffix we want to splice on is
    // the slice starting AFTER the `/` separator. Joining `home`
    // with `&s[1..]` would land an absolute path inside `home`
    // (PathBuf::push of an absolute path RESETS the buffer to that
    // absolute path), so we strip the leading `/` first. Doubled
    // separators in the rest portion (`~//foo` → `s[2..] = "/foo"`)
    // would also reset the buffer; loop the strip so any run of
    // leading `/`s is consumed before the push.
    let mut rest = &s[2..]; // skip "~/"
    while let Some(stripped) = rest.strip_prefix('/') {
        rest = stripped;
    }
    let mut p = std::path::PathBuf::from(home);
    if !rest.is_empty() {
        p.push(rest);
    }
    p
}

/// Read the running kernel release from `/proc/sys/kernel/osrelease`.
///
/// Returns `None` if the procfs entry is unreadable, empty, or missing.
/// Callers that need the release string for `/lib/modules/{release}/…`
/// fallbacks use this rather than shelling out to `uname -r`: the
/// procfs entry exposes the raw utsname release (served by
/// `proc_do_uts_string` in linux/kernel/utsname_sysctl.c), the same
/// field uname(2) reads — modulo the UNAME26 personality shim
/// `override_release` applies on the uname(2) path only — and only
/// costs a small read.
fn kernel_release_from_procfs() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Resolve a kernel source/build directory.
///
/// `kernel_dir`: value of `KTSTR_KERNEL` env var (if set).
///
/// Search order:
/// 1. `kernel_dir` parameter (from env var)
/// 2. `./linux` (workspace-local build tree)
/// 3. `../linux` (sibling directory)
/// 4. `/lib/modules/{release}/build` (installed kernel headers)
///
/// Returns the directory path if a kernel tree is found.
#[allow(dead_code)]
pub fn resolve_kernel(kernel_dir: Option<&str>) -> Option<std::path::PathBuf> {
    // 1. Explicit directory.
    if let Some(dir) = kernel_dir {
        let p = std::path::PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }

    // 2-3. Local build trees.
    for rel in &["./linux", "../linux"] {
        let p = std::path::PathBuf::from(rel);
        if p.is_dir() && has_kernel_artifacts(&p) {
            return Some(p);
        }
    }

    // 4. Installed kernel build dir — use the running release from
    // procfs to locate `/lib/modules/{release}/build`.
    if let Some(rel) = kernel_release_from_procfs() {
        let p = std::path::PathBuf::from(format!("/lib/modules/{rel}/build"));
        if p.is_dir() {
            return Some(p);
        }
    }

    None
}

/// Derive the kernel directory (holding `vmlinux` and related build
/// artifacts) from a kernel image path.
///
/// Recognizes two layouts:
///
/// - **Build tree**: `<root>/arch/x86/boot/bzImage` (or
///   `arch/arm64/boot/Image`) → `<root>`. Suffix match on the
///   canonical path.
/// - **Cache entry**: `<cache_dir>/bzImage` (or `Image`) with a
///   sibling `vmlinux` → `<cache_dir>`. Lets probe source-location
///   resolution walk a cached kernel's stripped ELF.
///
/// Returns `None` when neither layout matches or the input path
/// doesn't canonicalize.
///
/// Cache entries carry stripped vmlinux (no DWARF) — `strip_vmlinux_debug`
/// drops `.debug_*` on every cache entry regardless of source type.
/// file:line resolution works only for build-tree paths where the
/// unstripped vmlinux is still present, or when the caller layers
/// `cache::prefer_source_tree_for_dwarf` on top to re-route
/// `cache::KernelSource::Local` entries at their original source tree.
#[allow(dead_code)]
pub fn derive_kernel_dir(image: &std::path::Path) -> Option<std::path::PathBuf> {
    let canon = std::fs::canonicalize(image).ok()?;

    #[cfg(target_arch = "x86_64")]
    let build_suffix = "/arch/x86/boot/bzImage";
    #[cfg(target_arch = "aarch64")]
    let build_suffix = "/arch/arm64/boot/Image";

    if let Some(canon_str) = canon.to_str()
        && let Some(root) = canon_str.strip_suffix(build_suffix)
    {
        return Some(std::path::PathBuf::from(root));
    }

    let parent = canon.parent()?;
    // is_file (not exists) matches cache::prefer_source_tree_for_dwarf's
    // sibling probe, so a `vmlinux` directory or symlink-to-directory
    // cannot satisfy either check.
    if parent.join("vmlinux").is_file() {
        return Some(parent.to_path_buf());
    }

    None
}

/// Find a bootable kernel image within a directory.
///
/// Checks the arch-specific build tree path first (`arch/x86/boot/bzImage`
/// or `arch/arm64/boot/Image`), then falls back to the directory root
/// (for cache entries that store the boot image directly).
#[allow(dead_code)]
pub fn find_image_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    // Build tree layout: arch-specific subdirectory.
    #[cfg(target_arch = "x86_64")]
    {
        let p = dir.join("arch/x86/boot/bzImage");
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let p = dir.join("arch/arm64/boot/Image");
        if p.exists() {
            return Some(p);
        }
    }
    // Cache entry layout: boot image at directory root.
    #[cfg(target_arch = "x86_64")]
    {
        let p = dir.join("bzImage");
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let p = dir.join("Image");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Find a bootable kernel image on the host.
///
/// `kernel_dir`: explicit kernel directory (e.g. from `KTSTR_KERNEL`).
/// When set, only that directory is searched — no fallback to local
/// build trees or host paths.
///
/// `release`: kernel release string (e.g. from `uname -r`). When
/// `None`, falls back to reading `/proc/sys/kernel/osrelease` — the
/// same value the kernel exposes via the `uname(2)` syscall, without
/// the shell-out cost.
///
/// Without `kernel_dir`, searches local build trees (`./linux`,
/// `../linux`), `/lib/modules/{release}/build`, then host paths
/// (`/lib/modules/{release}/vmlinuz`, `/boot/vmlinuz-{release}`,
/// `/boot/vmlinuz`).
#[allow(dead_code)]
pub fn find_image(kernel_dir: Option<&str>, release: Option<&str>) -> Option<std::path::PathBuf> {
    // When kernel_dir is explicit, only check that directory.
    if let Some(dir_str) = kernel_dir {
        let dir = std::path::PathBuf::from(dir_str);
        if !dir.is_dir() {
            return None;
        }
        return find_image_in_dir(&dir);
    }

    // No explicit dir: search local build trees via resolve_kernel.
    if let Some(dir) = resolve_kernel(None)
        && let Some(img) = find_image_in_dir(&dir)
    {
        return Some(img);
    }

    // Host fallback paths. When `release` is not supplied, pull the
    // running kernel release from procfs via
    // [`kernel_release_from_procfs`].
    let owned_release;
    let rel = match release {
        Some(r) => Some(r),
        None => {
            owned_release = kernel_release_from_procfs();
            owned_release.as_deref()
        }
    };

    if let Some(rel) = rel {
        let p = std::path::PathBuf::from(format!("/lib/modules/{rel}/vmlinuz"));
        if std::fs::File::open(&p).is_ok() {
            return Some(p);
        }
        let p = std::path::PathBuf::from(format!("/boot/vmlinuz-{rel}"));
        if std::fs::File::open(&p).is_ok() {
            return Some(p);
        }
    }

    let p = std::path::PathBuf::from("/boot/vmlinuz");
    if std::fs::File::open(&p).is_ok() {
        return Some(p);
    }

    None
}

/// Resolve the BTF source file for vmlinux.h generation.
///
/// `kernel_dir`: explicit kernel directory (e.g. from `KTSTR_KERNEL`).
///
/// Prefers `{resolved_dir}/vmlinux`, then `/sys/kernel/btf/vmlinux`.
#[allow(dead_code)]
pub fn resolve_btf(kernel_dir: Option<&str>) -> Option<std::path::PathBuf> {
    if let Some(dir) = resolve_kernel(kernel_dir) {
        let vmlinux = dir.join("vmlinux");
        if vmlinux.exists() {
            return Some(vmlinux);
        }
    }
    let sysfs = std::path::Path::new("/sys/kernel/btf/vmlinux");
    if sysfs.exists() {
        return Some(sysfs.to_path_buf());
    }
    None
}

/// Check if a directory contains kernel build artifacts.
///
/// Checks both build tree layout (arch subdirectories) and cache
/// entry layout (boot image at directory root).
fn has_kernel_artifacts(dir: &std::path::Path) -> bool {
    if dir.join("vmlinux").exists() {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if dir.join("arch/x86/boot/bzImage").exists() || dir.join("bzImage").exists() {
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if dir.join("arch/arm64/boot/Image").exists() || dir.join("Image").exists() {
        return true;
    }
    false
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // -- resolve_kernel --

    #[test]
    fn kernel_path_resolve_explicit_dir_exists() {
        let tmp = TempDir::new().unwrap();
        let result = resolve_kernel(Some(tmp.path().to_str().unwrap()));
        assert_eq!(result, Some(tmp.path().to_path_buf()));
    }

    #[test]
    fn kernel_path_resolve_explicit_dir_not_exists() {
        let result = resolve_kernel(Some("/nonexistent/kernel/dir/that/does/not/exist"));
        // The explicit dir doesn't exist, so resolve_kernel skips it.
        // It may still find a kernel via fallback paths (./linux, ../linux,
        // /lib/modules). The key invariant: the nonexistent path must never
        // be returned.
        assert_ne!(
            result,
            Some(PathBuf::from("/nonexistent/kernel/dir/that/does/not/exist"))
        );
    }

    #[test]
    fn kernel_path_resolve_none_falls_through() {
        // With None, resolve_kernel skips the explicit branch and tries
        // ./linux, ../linux, then /lib/modules. The result depends on
        // the host, but the function must not panic.
        let _ = resolve_kernel(None);
    }

    #[test]
    fn kernel_path_resolve_none_returns_osrelease_build_dir_when_present() {
        // resolve_kernel(None) reads `/proc/sys/kernel/osrelease` and
        // checks `/lib/modules/{rel}/build` as its last fallback. The
        // earlier branches (`./linux`, `../linux`) cannot be controlled
        // from a parallel-safe unit test (`set_current_dir` is process-
        // wide), so this test is strong only when those local trees are
        // absent. When `/lib/modules/{rel}/build` is absent on the host
        // (typical CI without installed kernel headers), skip via early
        // return — the panic-free contract is already covered by
        // `kernel_path_resolve_none_falls_through`.
        let release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .expect("host /proc/sys/kernel/osrelease must be readable for this test")
            .trim()
            .to_string();
        let expected = std::path::PathBuf::from(format!("/lib/modules/{release}/build"));
        if !expected.is_dir() {
            return;
        }

        let resolved = resolve_kernel(None).unwrap_or_else(|| {
            panic!(
                "resolve_kernel(None) must return Some when {} exists",
                expected.display(),
            )
        });
        assert!(
            resolved.is_dir(),
            "resolved path must be a directory, got {}",
            resolved.display(),
        );
        // Strong pin only when no earlier branch (`./linux`, `../linux`)
        // shadowed the osrelease path. When an earlier branch matched,
        // the panic-free + valid-dir contract above is what we get.
        let local_shadowed = std::path::PathBuf::from("./linux").is_dir()
            || std::path::PathBuf::from("../linux").is_dir();
        if !local_shadowed {
            assert_eq!(
                resolved, expected,
                "with no local trees, resolve_kernel(None) must return the osrelease build dir",
            );
        }
    }

    #[test]
    fn kernel_path_resolve_empty_string() {
        // Empty string creates a PathBuf("") which is_dir() returns false,
        // so it falls through to search paths.
        let result = resolve_kernel(Some(""));
        // "" is not a directory, so it must not be returned as the explicit path.
        assert_ne!(result, Some(PathBuf::from("")));
    }

    // -- has_kernel_artifacts --

    #[test]
    fn kernel_path_has_artifacts_vmlinux() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_has_artifacts_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_path_has_artifacts_image() {
        // aarch64 build tree layout: arch/arm64/boot/Image.
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/arm64/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("Image"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    #[test]
    fn kernel_path_has_artifacts_empty_dir() {
        let tmp = TempDir::new().unwrap();
        assert!(!has_kernel_artifacts(tmp.path()));
    }

    // -- find_image_in_dir --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_path_find_image_in_dir_image() {
        // aarch64 build tree layout: find_image_in_dir returns
        // arch/arm64/boot/Image.
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/arm64/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("Image"), b"fake").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("Image")));
    }

    #[test]
    fn kernel_path_find_image_in_dir_empty() {
        let tmp = TempDir::new().unwrap();
        assert!(find_image_in_dir(tmp.path()).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_cache_layout() {
        // Cache entries store bzImage at directory root (no arch/ subdir).
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("bzImage"), b"fake").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(tmp.path().join("bzImage")));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_path_find_image_in_dir_cache_layout_image() {
        // Cache entries store Image at directory root on aarch64.
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Image"), b"fake").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(tmp.path().join("Image")));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_in_dir_prefers_build_tree() {
        // When both arch/ and root-level bzImage exist, prefer arch/.
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"build-tree").unwrap();
        std::fs::write(tmp.path().join("bzImage"), b"root-level").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_path_find_image_in_dir_prefers_build_tree_image() {
        // When both arch/arm64/boot/Image and root-level Image exist,
        // prefer the build-tree path. Pins the same precedence on
        // aarch64 as the x86_64 sibling — the build-tree branch in
        // `find_image_in_dir` runs before the cache-entry branch on
        // either arch.
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/arm64/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("Image"), b"build-tree").unwrap();
        std::fs::write(tmp.path().join("Image"), b"root-level").unwrap();
        let result = find_image_in_dir(tmp.path());
        assert_eq!(result, Some(boot.join("Image")));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_has_artifacts_root_bzimage() {
        // Cache entry layout: bzImage at directory root.
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("bzImage"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_path_has_artifacts_root_image() {
        // Cache entry layout on aarch64: Image at directory root.
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Image"), b"fake").unwrap();
        assert!(has_kernel_artifacts(tmp.path()));
    }

    // -- derive_kernel_dir --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_build_tree_x86() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        let image = boot.join("bzImage");
        std::fs::write(&image, b"fake").unwrap();

        let canon_root = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon_root));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn derive_kernel_dir_build_tree_aarch64() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/arm64/boot");
        std::fs::create_dir_all(&boot).unwrap();
        let image = boot.join("Image");
        std::fs::write(&image, b"fake").unwrap();

        let canon_root = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon_root));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_cache_entry_x86_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("bzImage");
        std::fs::write(&image, b"fake").unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake-elf").unwrap();

        let canon = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn derive_kernel_dir_cache_entry_aarch64_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("Image");
        std::fs::write(&image, b"fake").unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake-elf").unwrap();

        let canon = std::fs::canonicalize(tmp.path()).unwrap();
        assert_eq!(derive_kernel_dir(&image), Some(canon));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_cache_entry_without_vmlinux() {
        // bzImage at root with no vmlinux sibling — neither layout
        // applies, return None.
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("bzImage");
        std::fs::write(&image, b"fake").unwrap();
        assert_eq!(derive_kernel_dir(&image), None);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn derive_kernel_dir_cache_entry_without_vmlinux_aarch64() {
        // Image at root with no vmlinux sibling — neither the build-
        // tree suffix nor the cache-entry vmlinux probe applies,
        // return None. Mirror of the x86_64 sibling so a future
        // refactor that loosened the predicate on either arch trips
        // a test rather than silently mapping arbitrary
        // `Image`-named files to their parent directories.
        let tmp = TempDir::new().unwrap();
        let image = tmp.path().join("Image");
        std::fs::write(&image, b"fake").unwrap();
        assert_eq!(derive_kernel_dir(&image), None);
    }

    #[test]
    fn derive_kernel_dir_nonexistent_path() {
        // canonicalize fails on a nonexistent path.
        let p = std::path::Path::new("/nonexistent/kernel/bzImage");
        assert_eq!(derive_kernel_dir(p), None);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn derive_kernel_dir_arbitrary_image_no_vmlinux_sibling() {
        // A file named bzImage but in a dir without a vmlinux sibling
        // and not under arch/x86/boot — no match.
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("somewhere/else");
        std::fs::create_dir_all(&sub).unwrap();
        let image = sub.join("bzImage");
        std::fs::write(&image, b"fake").unwrap();
        assert_eq!(derive_kernel_dir(&image), None);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn derive_kernel_dir_arbitrary_image_no_vmlinux_sibling_aarch64() {
        // A file named Image but neither under arch/arm64/boot nor
        // alongside a vmlinux sibling — no match. Mirror of the
        // x86_64 sibling.
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("somewhere/else");
        std::fs::create_dir_all(&sub).unwrap();
        let image = sub.join("Image");
        std::fs::write(&image, b"fake").unwrap();
        assert_eq!(derive_kernel_dir(&image), None);
    }

    // -- resolve_btf --

    #[test]
    fn kernel_path_resolve_btf_with_vmlinux_in_dir() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("vmlinux"), b"fake").unwrap();
        let result = resolve_btf(Some(tmp.path().to_str().unwrap()));
        assert_eq!(result, Some(tmp.path().join("vmlinux")));
    }

    #[test]
    fn kernel_path_resolve_btf_dir_without_vmlinux() {
        let tmp = TempDir::new().unwrap();
        // No vmlinux in dir; falls through to /sys/kernel/btf/vmlinux check.
        let result = resolve_btf(Some(tmp.path().to_str().unwrap()));
        // Result depends on host: either /sys/kernel/btf/vmlinux exists or None.
        if let Some(ref p) = result {
            assert!(p.exists());
        }
    }

    #[test]
    fn kernel_path_resolve_btf_nonexistent_dir() {
        let result = resolve_btf(Some("/nonexistent/btf/dir/xyz"));
        // Dir doesn't exist so resolve_kernel returns None; falls to sysfs.
        if let Some(ref p) = result {
            assert!(p.exists());
        }
    }

    // -- find_image --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_path_find_image_explicit_dir_with_bzimage() {
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/x86/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("bzImage"), b"fake").unwrap();
        let result = find_image(Some(tmp.path().to_str().unwrap()), None);
        assert_eq!(result, Some(boot.join("bzImage")));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_path_find_image_explicit_dir_with_image() {
        // `find_image(Some(dir), _)` must short-circuit to the
        // directory-local search and return the build-tree
        // arch/arm64/boot/Image without falling through to host
        // fallback paths. Mirror of the x86_64 sibling.
        let tmp = TempDir::new().unwrap();
        let boot = tmp.path().join("arch/arm64/boot");
        std::fs::create_dir_all(&boot).unwrap();
        std::fs::write(boot.join("Image"), b"fake").unwrap();
        let result = find_image(Some(tmp.path().to_str().unwrap()), None);
        assert_eq!(result, Some(boot.join("Image")));
    }

    #[test]
    fn kernel_path_find_image_nonexistent_dir() {
        // Nonexistent explicit dir: `!dir.is_dir()` in the explicit-dir
        // branch hits `return None` (find_image ~line 592) with no
        // fallthrough to resolve_kernel / host fallback paths. The
        // return value None — not merely "did not panic" — is the
        // documented short-circuit contract, so pin it. A regression
        // that fell through to local/host search instead of
        // short-circuiting would be caught here.
        assert_eq!(
            find_image(Some("/nonexistent/image/dir/xyz"), None),
            None,
            "nonexistent explicit dir must short-circuit to None with no fallthrough",
        );
    }

    #[test]
    fn kernel_path_find_image_release_none_matches_osrelease() {
        // The `/proc/sys/kernel/osrelease` path is hardcoded in
        // find_image and cannot be mocked, so the fallback can only
        // be verified by equivalence: read osrelease the way the
        // function does, then assert that find_image(None, None)
        // equals find_image(None, Some(<that value>)). Identical
        // post-`rel` logic in both calls means equal outputs prove
        // the None branch derived `rel` from osrelease (or both
        // short-circuited via resolve_kernel(None), which is also
        // a contract — no panic, no divergence).
        let host_release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .expect("host /proc/sys/kernel/osrelease must be readable for this test")
            .trim()
            .to_string();
        assert!(
            !host_release.is_empty(),
            "/proc/sys/kernel/osrelease must be non-empty for this test",
        );

        let derived = find_image(None, None);
        let explicit = find_image(None, Some(&host_release));
        assert_eq!(
            derived, explicit,
            "find_image(None, None) must equal find_image(None, Some(osrelease)); fallback diverged",
        );
    }

    // -- KernelId parsing --

    #[test]
    fn kernel_id_parse_path_with_slash() {
        assert_eq!(
            KernelId::parse("../linux"),
            KernelId::Path(PathBuf::from("../linux"))
        );
        assert_eq!(
            KernelId::parse("/boot/vmlinuz"),
            KernelId::Path(PathBuf::from("/boot/vmlinuz"))
        );
    }

    #[test]
    fn kernel_id_parse_path_dot_prefix() {
        assert_eq!(
            KernelId::parse("./linux"),
            KernelId::Path(PathBuf::from("./linux"))
        );
        assert_eq!(KernelId::parse("."), KernelId::Path(PathBuf::from(".")));
    }

    /// `~`-prefixed paths are expanded against `$HOME` at parse
    /// time so a downstream `is_dir()` sees the resolved path
    /// rather than the literal `~/...` (which the libc / Rust
    /// path APIs do not interpret — only shells do). Pins the
    /// expansion against an explicit `HOME` value so the test is
    /// independent of the running operator's environment.
    ///
    /// Uses the env-mutation `lock_env` helper to serialise the
    /// `HOME` write against any sibling test that reads
    /// `$HOME` — see `crate::test_support::test_helpers::lock_env`
    /// for the locking rationale. `EnvVarGuard::set` restores the
    /// prior value when the guard drops so the test does not leak
    /// the override into peers.
    #[test]
    fn kernel_id_parse_path_tilde_prefix_expands() {
        let _lock = crate::test_support::test_helpers::lock_env();
        let _home_guard =
            crate::test_support::test_helpers::EnvVarGuard::set("HOME", "/home/fixture-user");
        assert_eq!(
            KernelId::parse("~/linux"),
            KernelId::Path(PathBuf::from("/home/fixture-user/linux")),
        );
    }

    /// Bare `~` (no slash) expands to `$HOME` exactly. Pins the
    /// degenerate single-character case so a future regression
    /// that special-cased "must contain `/` after `~`" lands
    /// here instead of silently leaving the literal `~`.
    #[test]
    fn kernel_id_parse_path_bare_tilde_expands() {
        let _lock = crate::test_support::test_helpers::lock_env();
        let _home_guard =
            crate::test_support::test_helpers::EnvVarGuard::set("HOME", "/home/fixture-user");
        assert_eq!(
            KernelId::parse("~"),
            KernelId::Path(PathBuf::from("/home/fixture-user")),
        );
    }

    /// `$HOME` unset → no expansion possible. The literal `~/...`
    /// passes through verbatim. Downstream `is_dir()` will reject
    /// it normally, surfacing the missing-directory error rather
    /// than panicking inside the parser.
    #[test]
    fn kernel_id_parse_path_tilde_with_home_unset_passes_through() {
        let _lock = crate::test_support::test_helpers::lock_env();
        let _home_guard = crate::test_support::test_helpers::EnvVarGuard::remove("HOME");
        assert_eq!(
            KernelId::parse("~/linux"),
            KernelId::Path(PathBuf::from("~/linux")),
        );
    }

    /// `~user/...` (different user) is NOT expanded because std
    /// has no `getpwnam`. Operator who wants a peer's home dir
    /// can spell it absolutely. Pin the no-op behavior so a
    /// future "shell out to getpwnam" addition has to update
    /// this test deliberately.
    #[test]
    fn kernel_id_parse_path_tilde_user_passes_through() {
        let _lock = crate::test_support::test_helpers::lock_env();
        let _home_guard =
            crate::test_support::test_helpers::EnvVarGuard::set("HOME", "/home/fixture-user");
        assert_eq!(
            KernelId::parse("~peer/linux"),
            KernelId::Path(PathBuf::from("~peer/linux")),
        );
    }

    #[test]
    fn kernel_id_parse_version_stable() {
        assert_eq!(
            KernelId::parse("6.14.2"),
            KernelId::Version("6.14.2".to_string())
        );
        assert_eq!(
            KernelId::parse("6.14"),
            KernelId::Version("6.14".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_version_rc() {
        assert_eq!(
            KernelId::parse("6.15-rc3"),
            KernelId::Version("6.15-rc3".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_version_patch_rc() {
        assert_eq!(
            KernelId::parse("6.14.2-rc1"),
            KernelId::Version("6.14.2-rc1".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_cache_key() {
        assert_eq!(
            KernelId::parse("6.14.2-tarball-x86_64"),
            KernelId::CacheKey("6.14.2-tarball-x86_64".to_string())
        );
        assert_eq!(
            KernelId::parse("local-deadbeef-x86_64"),
            KernelId::CacheKey("local-deadbeef-x86_64".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_v_prefix_not_version() {
        // "v6.14" starts with 'v', not a digit -- cache key.
        assert_eq!(
            KernelId::parse("v6.14"),
            KernelId::CacheKey("v6.14".to_string())
        );
    }

    #[test]
    fn kernel_id_parse_bare_major_is_version() {
        // "6" is a bare-major version prefix (resolves to the highest
        // 6.x.y patch via fetch_version_for_prefix), NOT a cache key.
        assert_eq!(KernelId::parse("6"), KernelId::Version("6".to_string()));
    }

    #[test]
    fn kernel_id_display() {
        assert_eq!(
            KernelId::Version("6.14.2".to_string()).to_string(),
            "6.14.2"
        );
        assert_eq!(
            KernelId::Path(PathBuf::from("../linux")).to_string(),
            "../linux"
        );
        assert_eq!(
            KernelId::CacheKey("my-key".to_string()).to_string(),
            "my-key"
        );
        assert_eq!(
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.13".to_string(),
                syntax_inclusive: false,
            }
            .to_string(),
            "6.10..6.13",
        );
        assert_eq!(
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.13".to_string(),
                syntax_inclusive: true,
            }
            .to_string(),
            "6.10..=6.13",
        );
        assert_eq!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "for-next".to_string(),
                ref_kind: GitRefKind::Branch,
            }
            .to_string(),
            "git+https://example.com/r.git#branch=for-next",
        );
        assert_eq!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "v6.14".to_string(),
                ref_kind: GitRefKind::Tag,
            }
            .to_string(),
            "git+https://example.com/r.git#tag=v6.14",
        );
        assert_eq!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "abc123".to_string(),
                ref_kind: GitRefKind::Sha,
            }
            .to_string(),
            "git+https://example.com/r.git#sha=abc123",
        );
        // Unknown (a bare `#REF`) round-trips verbatim so
        // parse(Display(x)) == x still holds for the reject case.
        assert_eq!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
                ref_kind: GitRefKind::Unknown,
            }
            .to_string(),
            "git+https://example.com/r.git#main",
        );
    }

    // -- KernelId::parse — Range arm --

    #[test]
    fn kernel_id_parse_range_versions() {
        assert_eq!(
            KernelId::parse("6.10..6.15"),
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.15".to_string(),
                syntax_inclusive: false,
            },
        );
    }

    /// `..=` (inclusive Rust syntax) and `..` both produce a Range
    /// with the SAME endpoints (and the same closed-range resolution
    /// semantics — both endpoints inclusive); they differ only in
    /// the `syntax_inclusive` flag, which round-trips through
    /// [`std::fmt::Display`] and the inverted-range error message
    /// so the operator-typed form is preserved verbatim. The `..=`
    /// arm is checked first in `parse`, so even though `..=` contains
    /// `..` as a substring the version-shaped split lands on `6.10`
    /// / `6.15`, not on `6.10` / `=6.15`.
    #[test]
    fn kernel_id_parse_range_inclusive_eq_syntax() {
        assert_eq!(
            KernelId::parse("6.10..=6.15"),
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.15".to_string(),
                syntax_inclusive: true,
            },
        );
    }

    #[test]
    fn kernel_id_parse_range_patch_versions() {
        assert_eq!(
            KernelId::parse("6.10.5..6.10.10"),
            KernelId::Range {
                start: "6.10.5".to_string(),
                end: "6.10.10".to_string(),
                syntax_inclusive: false,
            },
        );
    }

    #[test]
    fn kernel_id_parse_range_rc() {
        assert_eq!(
            KernelId::parse("6.10..6.10-rc3"),
            KernelId::Range {
                start: "6.10".to_string(),
                end: "6.10-rc3".to_string(),
                syntax_inclusive: false,
            },
        );
    }

    /// Both endpoints non-version: not a Range. The `/`-contains
    /// test fails too, so this falls to the version-shaped check
    /// (also fails on the `..`) and lands as CacheKey.
    #[test]
    fn kernel_id_parse_range_non_version_falls_through() {
        assert_eq!(
            KernelId::parse("foo..bar"),
            KernelId::CacheKey("foo..bar".to_string()),
        );
    }

    /// One endpoint version-shaped, the other not: the Range arm
    /// requires BOTH endpoints to pass `_is_version_string`, so
    /// `6.10..foo` falls through to CacheKey.
    #[test]
    fn kernel_id_parse_range_one_non_version() {
        assert_eq!(
            KernelId::parse("6.10..foo"),
            KernelId::CacheKey("6.10..foo".to_string()),
        );
    }

    /// Trailing `..` with no second endpoint: `_is_version_string("")`
    /// is false, so the Range arm doesn't fire. Falls to CacheKey
    /// (the version-shaped check also fails because the trailing `..`
    /// means a parts-iter sees an empty patch component).
    #[test]
    fn kernel_id_parse_range_empty_endpoint() {
        assert_eq!(
            KernelId::parse("6.10.."),
            KernelId::CacheKey("6.10..".to_string()),
        );
    }

    // -- KernelId::parse — Git arm (explicit #tag= / #branch= / #sha=) --

    #[test]
    fn kernel_id_parse_git_branch() {
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#branch=main"),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
                ref_kind: GitRefKind::Branch,
            },
        );
    }

    #[test]
    fn kernel_id_parse_git_tag() {
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#tag=v6.14"),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "v6.14".to_string(),
                ref_kind: GitRefKind::Tag,
            },
        );
    }

    #[test]
    fn kernel_id_parse_git_sha() {
        let sha = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(
            KernelId::parse(&format!("git+https://example.com/r.git#sha={sha}")),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: sha.to_string(),
                ref_kind: GitRefKind::Sha,
            },
        );
    }

    /// A bare `#REF` (no `kind=`) parses to `GitRefKind::Unknown` so
    /// `validate` — not `parse` — surfaces the actionable error; `parse`
    /// returns `Self`, never `Result`. An unrecognized `#foo=` selector
    /// lands the same way (the whole fragment is kept as the ref).
    #[test]
    fn kernel_id_parse_git_bare_ref_is_unknown() {
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#main"),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
                ref_kind: GitRefKind::Unknown,
            },
        );
        assert_eq!(
            KernelId::parse("git+https://example.com/r.git#foo=bar"),
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "foo=bar".to_string(),
                ref_kind: GitRefKind::Unknown,
            },
        );
    }

    /// The Git arm splits the URL/fragment on the LAST `#` (so a `#`
    /// inside the URL survives), then the fragment on the FIRST `=`
    /// into kind/value — `git+https://x#frag#branch=main` parses as
    /// url=`https://x#frag`, branch=`main`. A regression to
    /// `split_once('#')` would flip the URL/ref.
    #[test]
    fn kernel_id_parse_git_multi_hash_url() {
        assert_eq!(
            KernelId::parse("git+https://x#frag#branch=main"),
            KernelId::Git {
                url: "https://x#frag".to_string(),
                git_ref: "main".to_string(),
                ref_kind: GitRefKind::Branch,
            },
        );
    }

    /// Empty fragment after the `#` (`git+URL#`): the Git arm now
    /// claims ANY `git+…` string, so this parses to a Git with an empty
    /// `git_ref` and `GitRefKind::Unknown` rather than falling through
    /// to Path. That routes the typo into `validate`, which surfaces the
    /// actionable "ref kind must be explicit" error instead of a
    /// confusing filesystem "path not found".
    #[test]
    fn kernel_id_parse_git_empty_ref_is_git_unknown() {
        let id = KernelId::parse("git+https://example.com/r.git#");
        assert_eq!(
            id,
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: String::new(),
                ref_kind: GitRefKind::Unknown,
            },
        );
        assert!(
            id.validate().is_err(),
            "an empty/kind-less git fragment must be rejected by validate",
        );
    }

    /// Empty URL before the `#`: the Git arm claims it (git+ takes
    /// precedence), producing a Git with an empty `url`. A valid `#tag=`
    /// selector isolates the empty-URL `validate` check (a kind-less
    /// fragment would also trip the Unknown check), pinning that the
    /// empty URL — not a silent CacheKey miss — is what gets rejected.
    #[test]
    fn kernel_id_parse_git_empty_url_is_rejected() {
        let id = KernelId::parse("git+#tag=x");
        assert_eq!(
            id,
            KernelId::Git {
                url: String::new(),
                git_ref: "x".to_string(),
                ref_kind: GitRefKind::Tag,
            },
        );
        let err = id
            .validate()
            .expect_err("an empty git URL must be rejected by validate");
        assert!(
            err.contains("empty URL"),
            "empty-URL git spec must surface the empty-URL diagnostic, got: {err}",
        );
    }

    /// `git+` prefix takes precedence over the `/`-contains Path
    /// test. A user pointing at a local clone via `git+/local/repo#v1`
    /// should get a Git, not a Path. This pins the parse-arm
    /// ordering — flipping the Path check above the Git check would
    /// land here as KernelId::Path("git+/local/repo#v1").
    #[test]
    fn kernel_id_parse_git_beats_path() {
        assert_eq!(
            KernelId::parse("git+/local/repo#tag=v1"),
            KernelId::Git {
                url: "/local/repo".to_string(),
                git_ref: "v1".to_string(),
                ref_kind: GitRefKind::Tag,
            },
        );
    }

    // -- KernelId::parse — Package arm (`.rpm` / `.deb` / `.pkg.tar.zst`) --

    /// A bare package filename (no separators) classifies as Package,
    /// NOT a CacheKey — the suffix check runs before the version and
    /// cache-key fallbacks.
    #[test]
    fn kernel_id_parse_package_bare_filename() {
        assert_eq!(
            KernelId::parse("linux-6.14.rpm"),
            KernelId::Package {
                path: PathBuf::from("linux-6.14.rpm"),
            },
        );
        assert_eq!(
            KernelId::parse("linux-image.deb"),
            KernelId::Package {
                path: PathBuf::from("linux-image.deb"),
            },
        );
        assert_eq!(
            KernelId::parse("linux-neptune-618-x86_64.pkg.tar.zst"),
            KernelId::Package {
                path: PathBuf::from("linux-neptune-618-x86_64.pkg.tar.zst"),
            },
        );
    }

    /// `./foo.rpm` and `/abs/foo.deb` classify as Package, NOT Path —
    /// the `.rpm` / `.deb` suffix check runs ahead of the
    /// `/`-contains / `.`-prefix path test.
    #[test]
    fn kernel_id_parse_package_beats_path() {
        assert_eq!(
            KernelId::parse("./foo.rpm"),
            KernelId::Package {
                path: PathBuf::from("./foo.rpm"),
            },
        );
        assert_eq!(
            KernelId::parse("/abs/dir/foo.deb"),
            KernelId::Package {
                path: PathBuf::from("/abs/dir/foo.deb"),
            },
        );
        assert_eq!(
            KernelId::parse("./local.pkg.tar.zst"),
            KernelId::Package {
                path: PathBuf::from("./local.pkg.tar.zst"),
            },
        );
    }

    /// A bare extension with no stem is a HIDDEN-FILE name, not a
    /// package spec: `.deb` / `.rpm` / `.pkg.tar.zst` classify as
    /// Path via the dot-prefix arm, and a path whose last component
    /// is such a name classifies as Path via the `/` arm. Regression:
    /// the suffix check used to run without a stem requirement, so
    /// proptest's `\.[a-z]{1,10}` generator (which pins dot-prefix →
    /// Path) failed the moment it drew `.deb`.
    #[test]
    fn kernel_id_parse_bare_extension_is_path_not_package() {
        assert_eq!(
            KernelId::parse(".deb"),
            KernelId::Path(PathBuf::from(".deb"))
        );
        assert_eq!(
            KernelId::parse(".rpm"),
            KernelId::Path(PathBuf::from(".rpm"))
        );
        assert_eq!(
            KernelId::parse(".pkg.tar.zst"),
            KernelId::Path(PathBuf::from(".pkg.tar.zst")),
        );
        assert_eq!(
            KernelId::parse("/abs/dir/.deb"),
            KernelId::Path(PathBuf::from("/abs/dir/.deb")),
        );
    }

    /// The suffix is case-sensitive lowercase — `.RPM` is not a
    /// package; `FOO.RPM` (no separators) falls through to CacheKey.
    #[test]
    fn kernel_id_parse_package_case_sensitive() {
        assert_eq!(
            KernelId::parse("FOO.RPM"),
            KernelId::CacheKey("FOO.RPM".to_string()),
        );
    }

    /// `git+…` takes precedence over the package suffix — a git URL
    /// that happens to end in `.rpm` is still a Git source.
    #[test]
    fn kernel_id_parse_git_beats_package() {
        assert!(matches!(
            KernelId::parse("git+https://example.com/r.git#tag=v1.rpm"),
            KernelId::Git { .. },
        ));
    }

    /// A `~`-prefixed package path is expanded against `$HOME` at parse
    /// time, exactly like the Path arm.
    #[test]
    fn kernel_id_parse_package_tilde_expands() {
        let _lock = crate::test_support::test_helpers::lock_env();
        let _home_guard =
            crate::test_support::test_helpers::EnvVarGuard::set("HOME", "/home/fixture-user");
        assert_eq!(
            KernelId::parse("~/pkgs/foo.rpm"),
            KernelId::Package {
                path: PathBuf::from("/home/fixture-user/pkgs/foo.rpm"),
            },
        );
    }

    /// Package always validates — resolve-time legality is the
    /// integration step's concern, not the parser's.
    #[test]
    fn kernel_id_validate_package_ok() {
        assert!(
            KernelId::parse("linux.rpm").validate().is_ok(),
            "package spec must validate",
        );
    }

    // -- KernelId::parse — Distro arm --

    #[test]
    fn kernel_id_parse_distro_bare_names() {
        for (spec, kind) in [
            ("fedora", DistroKind::Fedora),
            ("ubuntu", DistroKind::Ubuntu),
            ("amazonlinux", DistroKind::AmazonLinux),
            ("steamos", DistroKind::SteamOs),
            ("gke", DistroKind::Gke),
        ] {
            assert_eq!(
                KernelId::parse(spec),
                KernelId::Distro {
                    kind,
                    release: None,
                },
                "bare distro name {spec:?}",
            );
        }
    }

    #[test]
    fn kernel_id_parse_distro_explicit_release() {
        assert_eq!(
            KernelId::parse("fedora-44"),
            KernelId::Distro {
                kind: DistroKind::Fedora,
                release: Some("44".to_string()),
            },
        );
        assert_eq!(
            KernelId::parse("ubuntu-24.04"),
            KernelId::Distro {
                kind: DistroKind::Ubuntu,
                release: Some("24.04".to_string()),
            },
        );
        assert_eq!(
            KernelId::parse("amazonlinux-2023"),
            KernelId::Distro {
                kind: DistroKind::AmazonLinux,
                release: Some("2023".to_string()),
            },
        );
        assert_eq!(
            KernelId::parse("steamos-3.8"),
            KernelId::Distro {
                kind: DistroKind::SteamOs,
                release: Some("3.8".to_string()),
            },
        );
        assert_eq!(
            KernelId::parse("gke-129"),
            KernelId::Distro {
                kind: DistroKind::Gke,
                release: Some("129".to_string()),
            },
        );
    }

    /// `f44` and `al2023` shorthands map to the same variants their
    /// long forms produce.
    #[test]
    fn kernel_id_parse_distro_shorthand() {
        assert_eq!(
            KernelId::parse("f44"),
            KernelId::Distro {
                kind: DistroKind::Fedora,
                release: Some("44".to_string()),
            },
        );
        assert_eq!(
            KernelId::parse("al2023"),
            KernelId::Distro {
                kind: DistroKind::AmazonLinux,
                release: Some("2023".to_string()),
            },
        );
    }

    /// `al<digits>` is Amazon Linux — `al9` parses as Amazon Linux
    /// release `9` and is then REJECTED by validate (not a 4-digit
    /// year). AlmaLinux is not supported (RHEL-family kernels disable
    /// CONFIG_VIRTIO_MMIO, which ktstr's console needs), so its former
    /// `alma<digits>` shorthand is now an ordinary cache key: `al`
    /// strips to `ma9`, which is not all digits, so the distro arm
    /// declines and the token falls through to `CacheKey`.
    #[test]
    fn kernel_id_parse_distro_shorthand_al_amazon_and_alma_is_cache_key() {
        assert_eq!(
            KernelId::parse("al9"),
            KernelId::Distro {
                kind: DistroKind::AmazonLinux,
                release: Some("9".to_string()),
            },
        );
        let err = KernelId::parse("al9")
            .validate()
            .expect_err("al9 must fail amazonlinux release validation");
        assert!(
            err.contains("amazonlinux") && err.contains("4-digit"),
            "al9 rejection must steer to the amazonlinux grammar: {err}",
        );
        // Former AlmaLinux spellings fall back to CacheKey like any
        // unknown token — parse never demotes them to a distro spec.
        for spec in ["alma", "alma9", "alma10", "almalinux", "almalinux-10"] {
            assert_eq!(
                KernelId::parse(spec),
                KernelId::CacheKey(spec.to_string()),
                "{spec:?} must classify as a cache key, not a distro",
            );
        }
    }

    /// A shorthand-shaped prefix whose remainder is not all digits is
    /// NOT a distro — `foo` / `alpha` stay CacheKey, and a bare `f`
    /// (empty remainder) too.
    #[test]
    fn kernel_id_parse_distro_shorthand_non_digits_are_cache_keys() {
        assert_eq!(
            KernelId::parse("foo"),
            KernelId::CacheKey("foo".to_string()),
        );
        assert_eq!(
            KernelId::parse("alpha"),
            KernelId::CacheKey("alpha".to_string()),
        );
        assert_eq!(KernelId::parse("f"), KernelId::CacheKey("f".to_string()));
    }

    /// A real cache-key shape must still classify as CacheKey, not get
    /// captured by the distro arm.
    #[test]
    fn kernel_id_parse_distro_does_not_shadow_cache_keys() {
        assert_eq!(
            KernelId::parse("6.14-tarball-x86_64-kc12345678"),
            KernelId::CacheKey("6.14-tarball-x86_64-kc12345678".to_string()),
        );
    }

    /// A distro name with a malformed release classifies as Distro (so
    /// `validate` can reject it) rather than silently becoming a
    /// CacheKey — mirrors the git parse/validate split.
    #[test]
    fn kernel_id_parse_distro_malformed_release_is_distro_not_cache_key() {
        assert_eq!(
            KernelId::parse("fedora-abc"),
            KernelId::Distro {
                kind: DistroKind::Fedora,
                release: Some("abc".to_string()),
            },
        );
        assert_eq!(
            KernelId::parse("ubuntu-24"),
            KernelId::Distro {
                kind: DistroKind::Ubuntu,
                release: Some("24".to_string()),
            },
        );
    }

    #[test]
    fn kernel_id_validate_distro_accepts_valid_releases() {
        for spec in [
            "fedora",
            "fedora-44",
            "f44",
            "fedora-100",
            "ubuntu",
            "ubuntu-24.04",
            "amazonlinux",
            "amazonlinux-2023",
            "al2023",
            "steamos",
            "steamos-3.7",
            "steamos-3.8",
            "gke",
            "gke-129",
        ] {
            assert!(
                KernelId::parse(spec).validate().is_ok(),
                "distro spec {spec:?} must validate",
            );
        }
    }

    #[test]
    fn kernel_id_validate_distro_rejects_malformed_releases() {
        // Fedora: 1 digit too short, 4 digits too long, non-digits.
        for spec in ["fedora-4", "f4", "fedora-4444", "fedora-abc"] {
            let err = KernelId::parse(spec)
                .validate()
                .expect_err("fedora spec must be rejected");
            assert!(
                err.contains("not a valid") && err.contains("fedora"),
                "fedora reject for {spec:?} must name the distro: {err}",
            );
        }
        // Ubuntu: bare year, wrong shape.
        for spec in ["ubuntu-24", "ubuntu-2404", "ubuntu-24.4"] {
            assert!(
                KernelId::parse(spec).validate().is_err(),
                "ubuntu spec {spec:?} must be rejected",
            );
        }
        // Amazon Linux: not 4 digits.
        for spec in ["amazonlinux-23", "al23", "amazonlinux-abcd"] {
            assert!(
                KernelId::parse(spec).validate().is_err(),
                "amazonlinux spec {spec:?} must be rejected",
            );
        }
        // SteamOS: bare major, no dot, non-digits.
        for spec in ["steamos-3", "steamos-38", "steamos-3.x", "steamos-3.8.1"] {
            let err = KernelId::parse(spec)
                .validate()
                .expect_err("steamos spec must be rejected");
            assert!(
                err.contains("steamos") && err.contains("MAJOR.MINOR"),
                "steamos reject for {spec:?} must cite the grammar: {err}",
            );
        }
        // GKE: milestone only, 2-3 decimal digits.
        for spec in ["gke-1", "gke-129.1", "gke-1234", "gke-latest"] {
            let err = KernelId::parse(spec)
                .validate()
                .expect_err("gke spec must be rejected");
            assert!(
                err.contains("gke") && err.contains("milestone"),
                "gke reject for {spec:?} must cite the grammar: {err}",
            );
        }
    }

    /// Distro Display renders the canonical long form; shorthand and
    /// long-form inputs round-trip semantically (Display then re-parse
    /// yields an equal id).
    #[test]
    fn kernel_id_distro_display_and_round_trip() {
        assert_eq!(
            KernelId::Distro {
                kind: DistroKind::Fedora,
                release: None,
            }
            .to_string(),
            "fedora",
        );
        assert_eq!(
            KernelId::parse("f44").to_string(),
            "fedora-44",
            "shorthand renders in long form",
        );
        for spec in [
            "fedora",
            "fedora-44",
            "f44",
            "ubuntu-24.04",
            "al2023",
            "steamos",
            "steamos-3.8",
            "gke",
            "gke-129",
        ] {
            let id = KernelId::parse(spec);
            assert_eq!(
                KernelId::parse(&id.to_string()),
                id,
                "distro spec {spec:?} must round-trip through Display",
            );
        }
    }

    // -- KernelId::parse_list --

    #[test]
    fn kernel_id_parse_list_basic() {
        let list = KernelId::parse_list("6.10,6.13");
        assert_eq!(
            list,
            vec![
                KernelId::Version("6.10".to_string()),
                KernelId::Version("6.13".to_string()),
            ],
        );
    }

    #[test]
    fn kernel_id_parse_list_mixed() {
        let list = KernelId::parse_list("6.10,git+url#branch=main,/srv/linux");
        assert_eq!(list.len(), 3, "expected 3 entries, got {list:?}");
        assert!(matches!(list[0], KernelId::Version(ref v) if v == "6.10"));
        assert!(matches!(
            list[1],
            KernelId::Git { ref url, ref git_ref, ref_kind: GitRefKind::Branch }
                if url == "url" && git_ref == "main"
        ));
        assert!(matches!(list[2], KernelId::Path(ref p) if p == &PathBuf::from("/srv/linux")));
    }

    #[test]
    fn kernel_id_parse_list_empty() {
        assert_eq!(KernelId::parse_list(""), Vec::<KernelId>::new());
    }

    /// Trailing / leading / repeated commas are forgiving — empty
    /// segments are silently dropped so `,6.10,,` yields just one
    /// entry. Spec says: defer dedup to the resolver but do not
    /// inject empty Cache-key entries from an operator typo.
    #[test]
    fn kernel_id_parse_list_trailing_comma() {
        assert_eq!(
            KernelId::parse_list(",6.10,,"),
            vec![KernelId::Version("6.10".to_string())],
        );
    }

    /// Whitespace around comma-separated entries gets trimmed before
    /// `parse` runs so `"6.10 , 6.13"` produces clean Version variants
    /// rather than CacheKey entries with embedded spaces.
    #[test]
    fn kernel_id_parse_list_whitespace() {
        assert_eq!(
            KernelId::parse_list("6.10 , 6.13"),
            vec![
                KernelId::Version("6.10".to_string()),
                KernelId::Version("6.13".to_string()),
            ],
        );
    }

    /// A single-entry list with no commas falls through `split(',')`
    /// as one segment and produces the same Variant `parse` would
    /// have produced directly. Pins the parse_list/parse equivalence
    /// for the trivial case so a future regression that special-cased
    /// "must contain comma" lands here.
    #[test]
    fn kernel_id_parse_list_single() {
        assert_eq!(
            KernelId::parse_list("6.10"),
            vec![KernelId::Version("6.10".to_string())],
        );
    }

    /// Duplicate entries are PRESERVED at parse time — `parse_list`
    /// is a pure splitter, and dedup is the resolver's job (after
    /// canonicalization to a cache key, since `6.10` and `v6.10` and
    /// a tag pointing at the same sha all collapse). Pin the count
    /// AND the index of each occurrence so a future regression that
    /// added an early dedup at parse time (which would silently
    /// collapse `6.10,6.10` to one entry and lose the operator's
    /// "run twice" intent if they later added that semantic) lands
    /// here.
    #[test]
    fn kernel_id_parse_list_preserves_dups() {
        let list = KernelId::parse_list("6.10,6.10,6.13");
        assert_eq!(list.len(), 3, "expected 3 entries, got {list:?}");
        assert_eq!(list[0], KernelId::Version("6.10".to_string()));
        assert_eq!(list[1], KernelId::Version("6.10".to_string()));
        assert_eq!(list[2], KernelId::Version("6.13".to_string()));
    }

    // -- KernelId::validate — inverted-range rejection --

    /// Forward range `6.10..6.13` validates fine — the most common
    /// happy-path case, here as a baseline for the failure tests
    /// below.
    #[test]
    fn kernel_id_validate_range_forward_ok() {
        let id = KernelId::parse("6.10..6.13");
        assert!(id.validate().is_ok(), "forward range must validate: {id:?}");
    }

    /// Equal endpoints `6.10..6.10` validate fine — degenerate
    /// single-element range, not inverted.
    #[test]
    fn kernel_id_validate_range_equal_endpoints_ok() {
        let id = KernelId::parse("6.10..6.10");
        assert!(
            id.validate().is_ok(),
            "equal endpoints must validate: {id:?}"
        );
    }

    /// `6.16..6.12` — same major, minor decreases. Reject. The error
    /// message must name both endpoints AND suggest the swapped
    /// spelling so the operator can fix the typo without re-reading
    /// the help.
    #[test]
    fn kernel_id_validate_range_inverted_minor() {
        let id = KernelId::parse("6.16..6.12");
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("inverted kernel range"),
            "error must say 'inverted kernel range', got: {err}",
        );
        assert!(
            err.contains("6.16..6.12"),
            "error must cite the spec, got: {err}"
        );
        assert!(
            err.contains("6.12..6.16"),
            "error must suggest the swapped form, got: {err}",
        );
        // Load-bearing negative: the operator typed `..`, so the error
        // must NOT silently substitute `..=`. Pins the
        // `syntax_inclusive: false` branch of validate's separator
        // selection — a regression that always emitted `..=`
        // (regardless of the typed form) would still pass the
        // positive substring checks above, but trips this assertion.
        assert!(
            !err.contains("..="),
            "operator typed `..`, error must not silently switch to `..=`: {err}",
        );
    }

    /// Mirror of `kernel_id_validate_range_inverted_minor` for the
    /// `..=` typed form. Pins the `syntax_inclusive: true` branch of
    /// validate's separator selection — the inverted-range error must
    /// preserve the operator's `..=` separator in both the as-typed
    /// citation and the swap suggestion, and must NOT silently
    /// substitute `..` (load-bearing negative). A regression that
    /// flipped the ternary in only one direction (Display correct,
    /// validate wrong, or vice versa) trips one of the four
    /// assertions.
    #[test]
    fn kernel_id_validate_range_inverted_minor_inclusive_eq_syntax() {
        let id = KernelId::parse("6.16..=6.12");
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("inverted kernel range"),
            "error must say 'inverted kernel range', got: {err}",
        );
        assert!(
            err.contains("6.16..=6.12"),
            "error must cite the spec verbatim (typed `..=`), got: {err}",
        );
        assert!(
            err.contains("6.12..=6.16"),
            "error must suggest the swapped form preserving `..=`, got: {err}",
        );
        // Load-bearing negative: operator typed `..=`, so the error
        // must NOT contain a bare `..` separator between version-shaped
        // tokens. The "bare `..` between digits" pattern only appears
        // when the separator is wrong; `..=` substrings always have
        // the trailing `=`, so checking for any `..` not followed by
        // `=` catches the bug. Use a substring check on both renderings
        // we expect to be inclusive.
        assert!(
            !err.contains("6.16..6.12"),
            "operator typed `..=`, error must not switch the cited spec to `..`: {err}",
        );
        assert!(
            !err.contains("6.12..6.16"),
            "operator typed `..=`, error must not switch the swap suggestion to `..`: {err}",
        );
    }

    /// Direct (non-parse) construction with `syntax_inclusive: true`
    /// still produces a `..=`-embedded error. Catches the gap where a
    /// caller bypasses [`KernelId::parse`] — e.g. config-file
    /// deserialization or a future builder API — and sets the flag
    /// directly. Validates that the flag drives the error format
    /// regardless of construction path.
    #[test]
    fn kernel_id_validate_range_inverted_inclusive_direct_construction() {
        let id = KernelId::Range {
            start: "6.16".to_string(),
            end: "6.12".to_string(),
            syntax_inclusive: true,
        };
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("6.16..=6.12"),
            "direct-construction Range with syntax_inclusive=true must emit `..=` cite in error, got: {err}",
        );
        assert!(
            err.contains("6.12..=6.16"),
            "direct-construction Range with syntax_inclusive=true must emit `..=` swap in error, got: {err}",
        );
    }

    /// End-to-end round-trip: parse → Display round-trips the typed
    /// form verbatim, AND validate's error message embeds the typed
    /// form in both the as-typed cite and the swap suggestion.
    /// Parameterized over both separator spellings so a regression
    /// that broke ONE direction trips here. Integration test for the
    /// `syntax_inclusive` contract across parse + Display + validate.
    #[test]
    fn kernel_id_display_inverted_range_round_trips_syntax_through_validate() {
        for (input, sep) in [("6.16..6.12", ".."), ("6.16..=6.12", "..=")] {
            let id = KernelId::parse(input);
            assert_eq!(
                id.to_string(),
                input,
                "Display must round-trip input verbatim for {input:?}",
            );
            let err = id.validate().unwrap_err();
            let cite = format!("6.16{sep}6.12");
            let suggest = format!("6.12{sep}6.16");
            assert!(
                err.contains(&cite),
                "validate error must cite `{cite}` for input {input:?}, got: {err}",
            );
            assert!(
                err.contains(&suggest),
                "validate error must suggest swap `{suggest}` for input {input:?}, got: {err}",
            );
        }
    }

    /// `7.0..6.99` — major decreases. Reject.
    #[test]
    fn kernel_id_validate_range_inverted_major() {
        let id = KernelId::parse("7.0..6.99");
        assert!(id.validate().is_err(), "inverted major must reject: {id:?}");
    }

    /// `6.10.5..6.10.3` — same major.minor, patch decreases. Reject.
    #[test]
    fn kernel_id_validate_range_inverted_patch() {
        let id = KernelId::parse("6.10.5..6.10.3");
        assert!(id.validate().is_err(), "inverted patch must reject: {id:?}");
    }

    /// `6.10..6.10-rc3` — release > rc per the rc-as-MAX rule, so
    /// pre-release on the upper end is inverted. Reject. Catches the
    /// common operator mistake of "I want 6.10 latest stable up
    /// through the rc series" written in reverse order.
    #[test]
    fn kernel_id_validate_range_inverted_rc_below_release() {
        let id = KernelId::parse("6.10..6.10-rc3");
        assert!(
            id.validate().is_err(),
            "release > rc — `6.10..6.10-rc3` must reject: {id:?}",
        );
    }

    /// `6.10-rc3..6.10` — pre-release < release. Forward direction;
    /// validate passes. The companion to `inverted_rc_below_release`.
    #[test]
    fn kernel_id_validate_range_rc_below_release_forward_ok() {
        let id = KernelId::parse("6.10-rc3..6.10");
        assert!(
            id.validate().is_ok(),
            "rc < release — `6.10-rc3..6.10` must validate: {id:?}",
        );
    }

    /// `6.14.5..6.14` — an explicit-patch START into a 2-component END
    /// that names the WHOLE 6.14 series. `range_end_key` widens the END
    /// to the series ceiling (6,14,MAX,MAX), so this is the valid range
    /// "6.14.5 through the end of the 6.14 series", NOT an inversion.
    /// Regression pin: before END-series-inclusivity, validate used the
    /// un-widened end (6,14,0,MAX) and falsely rejected this while the
    /// expansion accepted it — the two disagreed; now they must agree.
    #[test]
    fn kernel_id_validate_range_explicit_start_patch_into_series_end_ok() {
        let id = KernelId::parse("6.14.5..6.14");
        assert!(
            id.validate().is_ok(),
            "explicit-patch start into a series END must validate: {id:?}",
        );
    }

    /// `6.15..6.14` — a higher START minor than the END series must
    /// still reject even with the END widened to (6,14,MAX,MAX): the
    /// widening must not mask a real inversion.
    #[test]
    fn kernel_id_validate_range_series_end_still_rejects_higher_start() {
        let id = KernelId::parse("6.15..6.14");
        assert!(
            id.validate().is_err(),
            "higher start minor must still reject as inverted: {id:?}",
        );
    }

    /// `6.10-rc3..6.10-rc1` — same major.minor.patch but rc decreases.
    /// Reject. Pre-release ordering must follow numeric rcN order.
    #[test]
    fn kernel_id_validate_range_inverted_rc_to_rc() {
        let id = KernelId::parse("6.10-rc3..6.10-rc1");
        assert!(id.validate().is_err(), "rc3..rc1 must reject: {id:?}");
    }

    /// `6.10..6.10.5` — `6.10` decomposes to (6,10,0,MAX), `6.10.5`
    /// to (6,10,5,MAX). Forward direction. Validates.
    #[test]
    fn kernel_id_validate_range_missing_patch_treated_as_zero() {
        let id = KernelId::parse("6.10..6.10.5");
        assert!(
            id.validate().is_ok(),
            "missing patch defaults to 0, so `6.10..6.10.5` is forward: {id:?}",
        );
    }

    /// All non-Range variants validate trivially — Path, Version,
    /// CacheKey, Git all return Ok. Pins the "validate is currently
    /// only meaningful for Range" contract: a future field with its
    /// own resolve-time invariant should add an arm here, not slip
    /// through silently.
    #[test]
    fn kernel_id_validate_non_range_variants_ok() {
        assert!(KernelId::Version("6.14.2".to_string()).validate().is_ok());
        assert!(KernelId::CacheKey("my-key".to_string()).validate().is_ok());
        assert!(KernelId::Path(PathBuf::from("../linux")).validate().is_ok(),);
        assert!(
            KernelId::Git {
                url: "https://example.com/r.git".to_string(),
                git_ref: "main".to_string(),
                ref_kind: GitRefKind::Branch,
            }
            .validate()
            .is_ok(),
        );
    }

    /// The explicit git ref grammar rejects, at `validate`, a bare
    /// `#REF` (Unknown kind), an empty ref value, and a `#sha=` that
    /// is not a full 40-hex commit id. `parse` returns the variant;
    /// `validate` is the error channel (there is no `KernelId::Invalid`).
    #[test]
    fn kernel_id_validate_git_rejects_malformed_ref() {
        // Bare `#main` → Unknown → rejected with the "use #tag=/…" hint.
        let bare = KernelId::parse("git+https://example.com/r.git#main");
        let err = bare.validate().expect_err("bare #REF must be rejected");
        assert!(
            err.contains("#tag=") && err.contains("#branch=") && err.contains("#sha="),
            "error must name the accepted grammar: {err}"
        );
        // Empty value after a valid kind.
        assert!(
            KernelId::parse("git+https://example.com/r.git#tag=")
                .validate()
                .is_err(),
            "empty #tag= must be rejected"
        );
        // `#sha=` that is not 40 hex.
        assert!(
            KernelId::parse("git+https://example.com/r.git#sha=abc123")
                .validate()
                .is_err(),
            "short (non-40-hex) #sha= must be rejected"
        );
        let full = "0123456789abcdef0123456789abcdef01234567";
        assert!(
            KernelId::parse(&format!("git+https://example.com/r.git#sha={full}"))
                .validate()
                .is_ok(),
            "a full 40-hex #sha= must validate"
        );
    }

    /// Direct construction with an unparseable `start` endpoint
    /// (callers that build `KernelId::Range` outside `KernelId::parse`
    /// can put any string in either slot — the Display round-trip
    /// gives them the spelling back, but `validate()` is the safety
    /// net for resolve-time legality). Asserts the error names the
    /// "not a parseable version" condition so a downstream tool can
    /// distinguish this from the inverted-range message above.
    #[test]
    fn kernel_id_validate_range_unparseable_start() {
        let id = KernelId::Range {
            start: "garbage".to_string(),
            end: "6.10".to_string(),
            syntax_inclusive: false,
        };
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("not a parseable version"),
            "error must say 'not a parseable version', got: {err}",
        );
        assert!(
            err.contains("garbage"),
            "error must cite the bad endpoint, got: {err}"
        );
    }

    /// Companion to `unparseable_start` for the `end` slot.
    #[test]
    fn kernel_id_validate_range_unparseable_end() {
        let id = KernelId::Range {
            start: "6.10".to_string(),
            end: "garbage".to_string(),
            syntax_inclusive: false,
        };
        let err = id.validate().unwrap_err();
        assert!(
            err.contains("not a parseable version"),
            "error must say 'not a parseable version', got: {err}",
        );
        assert!(
            err.contains("garbage"),
            "error must cite the bad endpoint, got: {err}"
        );
    }

    // -- _is_version_string --

    #[test]
    fn kernel_id_is_version_string_valid() {
        assert!(_is_version_string("6"), "bare major is a version prefix");
        assert!(_is_version_string("6.14"));
        assert!(_is_version_string("6.14.2"));
        assert!(_is_version_string("6.15-rc3"));
        assert!(_is_version_string("6.14.0-rc1"));
        assert!(_is_version_string("5.0"));
        assert!(_is_version_string("5.0.0"));
        assert!(_is_version_string("5.4.0"));
    }

    #[test]
    fn kernel_id_is_version_string_invalid() {
        // Bare major `6` is now VALID (optional minor), but a non-digit
        // minor must still reject — pins the optional-minor boundary.
        assert!(!_is_version_string("6.x"));
        assert!(!_is_version_string("v6.14"));
        assert!(!_is_version_string(""));
        assert!(!_is_version_string("6.14.2-tarball-x86_64"));
        assert!(!_is_version_string("6.14.2.3"));
        assert!(!_is_version_string("6.14-rc"));
        assert!(!_is_version_string("6.14-rcX"));
        // rc_part contains non-digits after splitting on "-rc".
        assert!(!_is_version_string("6.14-rc3-tarball-x86_64"));
        assert!(!_is_version_string("abc"));
        assert!(!_is_version_string(".14"));
        assert!(!_is_version_string("6."));
        assert!(!_is_version_string("linux"));
        assert!(!_is_version_string(".6"));
    }

    // -- proptest --

    use proptest::prop_assert;

    proptest::proptest! {
        /// Arbitrary input must parse into a `KernelId` variant whose
        /// payload round-trips to the original string where a
        /// round-trip is defined (Path / Version / CacheKey). Bumped
        /// the input range from 30 to 120 characters to exercise long
        /// paths and pathological multi-dot strings.
        ///
        /// Path payload round-trip is conditional on the leading
        /// `~`-expansion path: a `~`- or `~/...`-prefixed input
        /// that resolves against `$HOME` lands in `Path` with the
        /// expanded form, NOT the literal `~`-prefix the input
        /// carried. The proptest detects that case and asserts the
        /// expanded equivalence instead of the literal one. Other
        /// `~`-prefix shapes (`~user/...`, or any prefix when
        /// `$HOME` is empty/unset) pass through verbatim and the
        /// strict round-trip assertion still holds — see
        /// [`super::expand_tilde`] for the case table.
        #[test]
        fn prop_kernel_id_parse_never_panics(s in "\\PC{0,120}") {
            // Hold the env lock for each proptest iteration so a
            // parallel test that mutates `HOME` cannot race the
            // two `expand_tilde` reads (one inside `parse`, one
            // for the expected value below) and produce a
            // false-positive payload-drift failure on a
            // `~`-prefixed input.
            let _env_lock = crate::test_support::test_helpers::lock_env();
            match KernelId::parse(&s) {
                KernelId::Path(p) => {
                    let expected = expand_tilde(&s);
                    prop_assert!(
                        p == expected,
                        "Path payload drift for {s:?}: got {p:?}, expected {expected:?}",
                    );
                }
                KernelId::Version(v) => prop_assert!(v == s, "Version payload drift for {s:?}"),
                KernelId::CacheKey(k) => prop_assert!(k == s, "CacheKey payload drift for {s:?}"),
                KernelId::Range { start, end, syntax_inclusive } => {
                    // Range is constructed only when both endpoints
                    // are version-shaped, so the payload round-trips
                    // through the `start..end` (or `start..=end`)
                    // rendering. Display emits the same separator
                    // the parser consumed, tracked via
                    // `syntax_inclusive`.
                    let sep = if syntax_inclusive { "..=" } else { ".." };
                    prop_assert!(
                        format!("{start}{sep}{end}") == s,
                        "Range payload drift for {s:?}",
                    );
                }
                KernelId::Git { .. } => {
                    // Any `git+…` string parses to a Git value. A STRING
                    // round-trip cannot hold for every input: `git+URL`
                    // and `git+URL#` both parse to the same value
                    // (`Unknown` kind, empty ref), and `Display` always
                    // renders a `#`, so `git+URL` != Display(parse("git+URL")).
                    // Assert the SEMANTIC round-trip instead — Display
                    // then re-parse yields an equal id — which holds for
                    // every kind including the `#`-free / empty-fragment
                    // spellings.
                    let id = KernelId::parse(&s);
                    let reparsed = KernelId::parse(&id.to_string());
                    prop_assert!(
                        reparsed == id,
                        "Git semantic round-trip drift for {s:?}: \
                         {id:?} -> {:?} -> {reparsed:?}",
                        id.to_string(),
                    );
                }
                KernelId::Package { path } => {
                    // Package mirrors Path: the payload is the
                    // `~`-expanded input, so it round-trips against
                    // `expand_tilde` exactly like the Path arm.
                    let expected = expand_tilde(&s);
                    prop_assert!(
                        path == expected,
                        "Package payload drift for {s:?}: got {path:?}, expected {expected:?}",
                    );
                }
                KernelId::Distro { .. } => {
                    // Shorthand spellings (`f44`, `al2023`) render in
                    // long form, so a STRING round-trip need not hold;
                    // assert the SEMANTIC round-trip (Display then
                    // re-parse yields an equal id) as the Git arm does.
                    let id = KernelId::parse(&s);
                    let reparsed = KernelId::parse(&id.to_string());
                    prop_assert!(
                        reparsed == id,
                        "Distro semantic round-trip drift for {s:?}: \
                         {id:?} -> {:?} -> {reparsed:?}",
                        id.to_string(),
                    );
                }
            }
        }

        #[test]
        fn prop_kernel_id_path_on_slash(
            prefix in "[a-z]{1,5}",
            suffix in "[a-z]{1,5}",
        ) {
            let s = format!("{prefix}/{suffix}");
            assert!(matches!(KernelId::parse(&s), KernelId::Path(_)));
        }

        #[test]
        fn prop_kernel_id_path_on_dot_prefix(s in "\\.[a-z]{1,10}") {
            assert!(matches!(KernelId::parse(&s), KernelId::Path(_)));
        }

        #[test]
        fn prop_kernel_id_version_roundtrip(
            major in 1u32..20,
            minor in 0u32..50,
            patch in 0u32..100,
        ) {
            let v = format!("{major}.{minor}.{patch}");
            assert_eq!(KernelId::parse(&v), KernelId::Version(v.clone()));
        }

        #[test]
        fn prop_kernel_id_version_rc(major in 1u32..20, minor in 0u32..50, rc in 1u32..10) {
            let v = format!("{major}.{minor}-rc{rc}");
            assert_eq!(KernelId::parse(&v), KernelId::Version(v.clone()));
        }
    }
}
