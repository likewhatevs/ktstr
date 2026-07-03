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
     `git+URL#sha=<40-hex>`), absolute or `~`-prefixed path, or cache key";

/// Kernel identifier: filesystem path, version string, cache key,
/// stable-release range, or git source.
///
/// Parsing heuristic (see [`KernelId::parse`]):
/// - Contains `/` (without a `git+` prefix) or starts with `.` or `~`:
///   [`KernelId::Path`]
/// - Starts with `git+`: [`KernelId::Git`] (form `git+URL#tag=NAME` /
///   `git+URL#branch=NAME` / `git+URL#sha=<40-hex>`)
/// - Contains `..` between two version-shaped tokens:
///   [`KernelId::Range`] (inclusive on both endpoints)
/// - Matches `MAJOR.MINOR[.PATCH][-rcN]`: [`KernelId::Version`]
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
    /// - `START..=END` or `START..END` where both endpoints are
    ///   version-shaped → [`KernelId::Range`]. The endpoints are
    ///   ALWAYS inclusive — both `..` and `..=` spellings produce a
    ///   closed range, regardless of Rust's exclusive-`..` /
    ///   inclusive-`..=` distinction. Both forms are accepted so test
    ///   authors and CLI users can write whichever feels natural.
    /// - `/`-containing or `.`/`~`-prefixed → [`KernelId::Path`].
    /// - Version-shaped → [`KernelId::Version`].
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
    ///
    /// All other variants always return `Ok(())` — this is a hook for
    /// future per-variant invariants, not a general-purpose validator.
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
            KernelId::Path(_) | KernelId::Version(_) | KernelId::CacheKey(_) => Ok(()),
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
        }
    }
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
