//! Minimal initramfs (cpio newc format) creation via the `cpio` crate.
//! Packs the test binary as `/init` along with scheduler binaries,
//! shared libraries, optional busybox, and user-provided include files
//! into a cpio archive for use as Linux initrd.
//! Init setup is handled by Rust code in `vmm::rust_init`.
use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
#[cfg(test)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// Result of shared library resolution for a binary.
#[derive(Debug, Clone)]
pub(crate) struct SharedLibs {
    /// Resolved `(guest_path, host_path)` pairs.
    pub found: Vec<(String, PathBuf)>,
    /// Library sonames that could not be resolved to a host path.
    pub missing: Vec<MissingLib>,
    /// The binary's PT_INTERP path, if present (e.g. `/lib64/ld-linux-x86-64.so.2`).
    pub interpreter: Option<String>,
    /// Stable identities of every ELF file parsed while resolving this
    /// closure. The cache preparer uses these to prove that the fds it pins
    /// for keying/building are the exact revisions the resolver inspected.
    pub observed_files: Vec<(PathBuf, ResolverFileIdentity)>,
    /// Exact candidate paths observed, in priority order up to each
    /// resolution decision. Both existing and missing candidates are
    /// recorded so adding a higher-priority soname invalidates a persistent
    /// closure memo without coupling it to unrelated directory churn.
    pub search_paths: Vec<ResolverPathObservation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolverFileIdentity {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_secs: i64,
    pub mtime_nsecs: i64,
    pub ctime_secs: i64,
    pub ctime_nsecs: i64,
}

impl ResolverFileIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            size: metadata.size(),
            mtime_secs: metadata.mtime(),
            mtime_nsecs: metadata.mtime_nsec(),
            ctime_secs: metadata.ctime(),
            ctime_nsecs: metadata.ctime_nsec(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolverPathObservation {
    pub path: PathBuf,
    pub identity: Option<ResolverFileIdentity>,
}

/// A shared library dependency that could not be resolved.
#[derive(Debug, Clone)]
pub(crate) struct MissingLib {
    /// The soname (e.g. `libssl.so.1.1`).
    pub soname: String,
}

/// Parsed soname-to-path mappings from `/etc/ld.so.cache`.
///
/// The binary cache is the authoritative lookup used by `ld-linux.so`.
/// It contains entries for every library indexed by `ldconfig`, including
/// libraries in directories added via `ldconfig /path` that may not
/// appear in the text-based `/etc/ld.so.conf` files. Parsing the cache
/// catches libraries the conf-based directory scan misses.
///
/// Format (glibc new format, `glibc-ld.so.cache1.1`):
///   - 48-byte header: `magic[20] + nlibs[4] + len_strings[4] + flags[4] + unused[16]`
///   - nlibs entries of 24 bytes: `flags[4] + key[4] + value[4] + osversion[4] + hwcap[8]`
///   - String table: key/value are absolute byte offsets from file start
///
/// Magic bytes at the start of the glibc new-format `ld.so.cache`.
const LD_CACHE_MAGIC: &[u8; 20] = b"glibc-ld.so.cache1.1";
/// Header size: magic(20) + nlibs(4) + len_strings(4) + flags(4) + unused(16).
const LD_CACHE_HEADER_SIZE: usize = 48;
/// Per-entry size: flags(4) + key(4) + value(4) + osversion(4) + hwcap(8).
const LD_CACHE_ENTRY_SIZE: usize = 24;

#[derive(Debug, Default)]
struct LdSoCache {
    /// First currently usable path per soname. Retained as the map-like
    /// compatibility surface for parser tests and diagnostics.
    selected: HashMap<String, PathBuf>,
    /// Every absolute cache candidate in file order, including missing and
    /// dangling paths whose later appearance must invalidate a closure.
    candidates: HashMap<String, Vec<PathBuf>>,
}

impl std::ops::Deref for LdSoCache {
    type Target = HashMap<String, PathBuf>;

    fn deref(&self) -> &Self::Target {
        &self.selected
    }
}

/// Parse the binary `/etc/ld.so.cache` file into a soname->path map.
///
/// Scans for the new-format magic because some systems prepend the
/// old format (`ld.so-1.7.0`) before the new-format section.
fn parse_ld_so_cache(path: &Path) -> LdSoCache {
    let mut cache = LdSoCache::default();
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return cache,
    };
    // Scan for new-format magic. Usually at offset 0, but old-format
    // systems prepend the legacy section.
    let Some(magic_pos) = data
        .windows(LD_CACHE_MAGIC.len())
        .position(|w| w == LD_CACHE_MAGIC)
    else {
        return cache;
    };
    let hdr = magic_pos;
    if data.len() < hdr + LD_CACHE_HEADER_SIZE {
        return cache;
    }
    let nlibs = u32::from_le_bytes(data[hdr + 20..hdr + 24].try_into().unwrap()) as usize;
    let min_size = hdr + LD_CACHE_HEADER_SIZE + nlibs * LD_CACHE_ENTRY_SIZE;
    if data.len() < min_size {
        return cache;
    }
    for i in 0..nlibs {
        let off = hdr + LD_CACHE_HEADER_SIZE + i * LD_CACHE_ENTRY_SIZE;
        // key and value are absolute byte offsets from file start.
        let key_off = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
        let val_off = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()) as usize;
        if key_off >= data.len() || val_off >= data.len() {
            continue;
        }
        let soname = match read_cstr(&data, key_off) {
            Some(s) => s,
            None => continue,
        };
        let path_str = match read_cstr(&data, val_off) {
            Some(s) => s,
            None => continue,
        };
        // Keep every absolute path, including a currently missing or
        // dangling candidate. Resolution records followed metadata for the
        // candidate before deciding whether to use it, so a target that
        // appears later invalidates the persistent closure.
        if path_str.starts_with('/') {
            let p = PathBuf::from(path_str);
            cache
                .candidates
                .entry(soname.to_string())
                .or_default()
                .push(p.clone());
            if p.is_file() {
                cache.selected.entry(soname.to_string()).or_insert(p);
            }
        }
    }
    cache
}

/// Read a null-terminated C string from `data` at `offset`.
fn read_cstr(data: &[u8], offset: usize) -> Option<&str> {
    let end = data[offset..].iter().position(|&b| b == 0)?;
    std::str::from_utf8(&data[offset..offset + end]).ok()
}

fn read_file_stable(path: &Path) -> Result<(Vec<u8>, ResolverFileIdentity)> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open ELF source {}", path.display()))?;
    let before_metadata = file
        .metadata()
        .with_context(|| format!("stat ELF source {}", path.display()))?;
    anyhow::ensure!(
        before_metadata.is_file(),
        "ELF source is not a regular file: {}",
        path.display()
    );
    let before = ResolverFileIdentity::from_metadata(&before_metadata);
    let mut data = Vec::with_capacity(usize::try_from(before.size).unwrap_or(0));
    file.read_to_end(&mut data)
        .with_context(|| format!("read ELF source {}", path.display()))?;
    let after = ResolverFileIdentity::from_metadata(
        &file
            .metadata()
            .with_context(|| format!("restat ELF source {}", path.display()))?,
    );
    anyhow::ensure!(
        before == after,
        "ELF source changed while resolving dependencies: {}",
        path.display()
    );
    Ok((data, before))
}

/// Resolve shared library dependencies for a dynamically-linked ELF binary.
/// Parses the ELF dynamic section to read DT_NEEDED entries, then resolves
/// each soname to a host path matching the host dynamic linker's search
/// order: DT_RPATH (legacy; only when DT_RUNPATH is absent) →
/// LD_LIBRARY_PATH → DT_RUNPATH → interp-relative hints → /etc/ld.so.cache
/// → default library paths. A separate /etc/ld.so.conf walk is omitted
/// because ldconfig ingests conf paths into ld.so.cache. When the binary
/// uses a non-standard PT_INTERP, the interpreter's parent and sibling
/// lib dirs feed the interp hints and are propagated to transitive deps.
/// Walks transitive deps via level-parallel BFS. Returns empty result
/// for static binaries or non-ELF files.
#[cfg(test)]
pub(crate) fn resolve_shared_libs(binary: &Path) -> Result<SharedLibs> {
    let loader_cwd = loader_current_dir()?;
    let ld_library_path_dirs = current_ld_library_path_dirs();
    let ld_so_cache = parse_ld_so_cache(Path::new("/etc/ld.so.cache"));
    resolve_shared_libs_inner(
        binary,
        binary,
        &[],
        &loader_cwd,
        &ld_library_path_dirs,
        &ld_so_cache,
    )
}

/// Resolve a pinned binary fd while retaining its original pathname for
/// `$ORIGIN` expansion and diagnostics. The loader inputs are supplied by
/// the cache preparer so the resolution, persistent key, and archive build
/// all observe the same pinned `/etc/ld.so.cache` revision and raw
/// `LD_LIBRARY_PATH` value.
pub(crate) fn resolve_shared_libs_from_pinned(
    pinned_source: &Path,
    original_path: &Path,
    loader_cwd: &Path,
    ld_library_path_dirs: &[PathBuf],
    pinned_ld_so_cache: Option<&Path>,
) -> Result<SharedLibs> {
    let ld_so_cache = pinned_ld_so_cache
        .map(parse_ld_so_cache)
        .unwrap_or_default();
    resolve_shared_libs_inner(
        pinned_source,
        original_path,
        &[],
        loader_cwd,
        ld_library_path_dirs,
        &ld_so_cache,
    )
}

/// Like [`resolve_shared_libs`] but seeds the BFS with additional
/// interp-relative hint directories on top of the ones derived from
/// the binary's own PT_INTERP. Use this when walking the interpreter
/// itself: the linker has no PT_INTERP of its own, so the auto-derived
/// hint set is empty and toolchain-local libs (interp→libA→libB chains
/// through `/opt/toolchain/lib`) would otherwise fall off the BFS at
/// libA's resolution step.
#[cfg(test)]
fn resolve_shared_libs_with_extra_interp_hints(
    binary: &Path,
    extra_interp_hints: &[PathBuf],
) -> Result<SharedLibs> {
    let loader_cwd = loader_current_dir()?;
    let ld_library_path_dirs = current_ld_library_path_dirs();
    let ld_so_cache = parse_ld_so_cache(Path::new("/etc/ld.so.cache"));
    resolve_shared_libs_inner(
        binary,
        binary,
        extra_interp_hints,
        &loader_cwd,
        &ld_library_path_dirs,
        &ld_so_cache,
    )
}

/// Resolve a non-standard interpreter's *own* shared-library dependencies.
///
/// A standard system linker (glibc/musl `ld.so`) is statically linked with no
/// deps, so this returns an empty set for it (`is_standard_interpreter`). A
/// custom-toolchain linker can instead be dynamically linked with its own
/// `libA -> libB` chain alongside it; the base archive packs that chain
/// into the base and `prepared_base_semantic_key` hashes it, so both go
/// through this one helper and the cache key covers exactly the set the base
/// packs.
///
/// Seeds the BFS with the linker's parent and sibling lib dirs (the linker has
/// no PT_INTERP of its own, so the auto-derived hint set would be empty) so the
/// linker's toolchain-local deps resolve against the same dirs the parent
/// binary's resolution used.
#[cfg(test)]
pub(crate) fn resolve_interpreter_deps(interp: &str) -> Result<SharedLibs> {
    if is_standard_interpreter(interp) {
        return Ok(SharedLibs {
            found: vec![],
            missing: vec![],
            interpreter: None,
            observed_files: vec![],
            search_paths: vec![],
        });
    }
    let interp_path = Path::new(interp);
    let mut interp_hints: Vec<PathBuf> = Vec::new();
    if let Some(parent) = interp_path.parent() {
        interp_hints.push(parent.to_path_buf());
        if let Some(grandparent) = parent.parent() {
            interp_hints.push(grandparent.join("lib"));
            interp_hints.push(grandparent.join("lib64"));
        }
    }
    resolve_shared_libs_with_extra_interp_hints(interp_path, &interp_hints)
}

/// Resolve the dependency chain of an interpreter through a pinned fd and
/// the same loader environment used for its parent binary.
pub(crate) fn resolve_interpreter_deps_from_pinned(
    pinned_source: &Path,
    original_path: &Path,
    loader_cwd: &Path,
    ld_library_path_dirs: &[PathBuf],
    pinned_ld_so_cache: Option<&Path>,
) -> Result<SharedLibs> {
    let mut interp_hints = Vec::new();
    if let Some(parent) = original_path.parent() {
        interp_hints.push(parent.to_path_buf());
        if let Some(grandparent) = parent.parent() {
            interp_hints.push(grandparent.join("lib"));
            interp_hints.push(grandparent.join("lib64"));
        }
    }
    let ld_so_cache = pinned_ld_so_cache
        .map(parse_ld_so_cache)
        .unwrap_or_default();
    resolve_shared_libs_inner(
        pinned_source,
        original_path,
        &interp_hints,
        loader_cwd,
        ld_library_path_dirs,
        &ld_so_cache,
    )
}

#[cfg(test)]
fn current_ld_library_path_dirs() -> Vec<PathBuf> {
    std::env::var_os("LD_LIBRARY_PATH")
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

#[cfg(test)]
fn loader_current_dir() -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("read loader current directory")?;
    std::fs::canonicalize(&cwd)
        .with_context(|| format!("canonicalize loader current directory {}", cwd.display()))
}

#[tracing::instrument(skip_all, fields(binary = %binary_origin.display(), extra_hints = extra_interp_hints.len()))]
fn resolve_shared_libs_inner(
    binary_source: &Path,
    binary_origin: &Path,
    extra_interp_hints: &[PathBuf],
    loader_cwd: &Path,
    ld_library_path_dirs: &[PathBuf],
    ld_so_cache: &LdSoCache,
) -> Result<SharedLibs> {
    // Cross-process memoisation and builder election live in
    // initramfs_cache. Keeping no path-only process cache here avoids stale
    // closures after a long-lived process observes a rebuilt ELF at the same
    // pathname.
    let (data, root_identity) = read_file_stable(binary_source)
        .with_context(|| format!("read binary: {}", binary_origin.display()))?;
    let mut observed_files = vec![(binary_origin.to_path_buf(), root_identity)];
    let mut search_observations = Vec::new();
    let elf = match goblin::elf::Elf::parse(&data) {
        Ok(e) => e,
        Err(_) => {
            // Not a valid ELF (or 32-bit) — treat as static/non-dynamic.
            return Ok(SharedLibs {
                found: vec![],
                missing: vec![],
                interpreter: None,
                observed_files,
                search_paths: search_observations,
            });
        }
    };

    let interpreter = elf.interpreter.map(|s| s.to_string());

    if elf.libraries.is_empty() && elf.dynamic.is_none() {
        // No dynamic section — static binary.
        return Ok(SharedLibs {
            found: vec![],
            missing: vec![],
            interpreter,
            observed_files,
            search_paths: search_observations,
        });
    }

    // Extract DT_NEEDED, DT_RUNPATH, and DT_RPATH from the root binary.
    let root_needed: Vec<String> = elf.libraries.iter().map(|s| s.to_string()).collect();
    let root_search = elf_search_paths(&elf, binary_origin);
    // When the binary uses a non-standard interpreter (custom toolchain),
    // collect the interpreter's parent dir and sibling lib dirs. These
    // are passed to resolve_soname as `interp_hints` alongside the
    // RPATH/RUNPATH split so the custom environment's libs are
    // consulted before system libs (ld.so.cache, /etc/ld.so.conf,
    // default paths). LD_LIBRARY_PATH still overrides them per the
    // resolve_soname order contract.
    // Without this, the system libc gets resolved first, causing version
    // mismatches when the custom ld.so loads a libc that requires GLIBC
    // symbols the custom ld.so doesn't provide.
    let mut interp_search_dirs: Vec<PathBuf> = match interpreter {
        Some(ref interp) if !is_standard_interpreter(interp) => {
            let interp_path = Path::new(interp);
            let mut dirs = Vec::new();
            if let Some(parent) = interp_path.parent() {
                dirs.push(parent.to_path_buf());
                // Sibling lib dirs: e.g. for /opt/toolchain/lib64/ld.so,
                // parent is lib64, so siblings are at parent.parent()/lib
                // and parent.parent()/lib64.
                if let Some(grandparent) = parent.parent() {
                    dirs.push(grandparent.join("lib"));
                    dirs.push(grandparent.join("lib64"));
                }
            }
            dirs
        }
        _ => Vec::new(),
    };

    // Caller-supplied hints (used when walking the linker itself: the
    // linker has no PT_INTERP of its own, so the match arm above
    // produces an empty set and the caller passes the toolchain dirs
    // computed at the call site). Extras are appended after the
    // auto-derived dirs so a binary's own PT_INTERP-derived hints take
    // precedence; resolve_soname iterates in order.
    for hint in extra_interp_hints {
        if !interp_search_dirs.contains(hint) {
            interp_search_dirs.push(hint.clone());
        }
    }

    // Resolve the full transitive closure level by level. Keep the
    // per-level reads sequential: a normal userspace payload has only a
    // handful of DT_NEEDED entries, while starting rayon's process-global
    // pool creates one worker per host CPU. Under a verifier sweep every
    // nextest cell is a separate process, so the old `par_iter()` turned a
    // few tiny ELF reads into thousands of runnable helper threads before
    // the cross-process initramfs cache gate was even reached.

    let mut found: Vec<(String, PathBuf)> = Vec::new();
    let mut missing: Vec<MissingLib> = Vec::new();
    let mut visited = std::collections::HashSet::new();

    // Current level: (soname, search_paths_from_parent)
    let mut level: Vec<(String, ElfSearchPaths)> = root_needed
        .iter()
        .map(|s| (s.clone(), root_search.clone()))
        .collect();

    while !level.is_empty() {
        // Phase 1: resolve sonames to host paths (sequential, cheap).
        let mut resolved: Vec<(String, PathBuf, PathBuf)> = Vec::new();
        for (soname, search_paths) in &level {
            if !visited.insert(soname.clone()) {
                continue;
            }
            if let Some(host_path) = resolve_soname_with_loader(
                soname,
                search_paths,
                &interp_search_dirs,
                loader_cwd,
                ld_library_path_dirs,
                ld_so_cache,
                &mut search_observations,
            )? {
                let canonical =
                    std::fs::canonicalize(&host_path).unwrap_or_else(|_| host_path.clone());
                let canon_str = canonical.to_string_lossy();
                let canon_guest = canon_str
                    .strip_prefix('/')
                    .unwrap_or(&canon_str)
                    .to_string();
                found.push((canon_guest.clone(), canonical.clone()));

                // Also add the non-canonical path if it differs, so the
                // guest dynamic linker can find the lib via either path.
                let host_str = host_path.to_string_lossy();
                let host_guest = host_str.strip_prefix('/').unwrap_or(&host_str).to_string();
                let host_guest_for_alias_check = host_guest.clone();
                if host_guest != canon_guest {
                    found.push((host_guest, canonical.clone()));
                }

                // Always emit a standard-path alias at `lib64/<soname>`
                // so the guest dynamic linker can find the lib via its
                // default search path even when host resolution picked
                // a non-standard path (e.g. a build artifact reached
                // via the host's `LD_LIBRARY_PATH` — common when
                // `cargo nextest` runs a test binary linked against
                // libbpf-sys's vendored libelf). Without this alias,
                // an executable like `/bin/wprof` that does
                // `dlopen("libelf.so.1")` inside the guest can't see
                // the lib at all because the guest has no
                // `/etc/ld.so.cache` and no `LD_LIBRARY_PATH` —
                // glibc falls back to `/lib64`, `/usr/lib64`, and
                // nothing else.
                let standard_alias = format!("lib64/{soname}");
                if standard_alias != canon_guest && standard_alias != host_guest_for_alias_check {
                    found.push((standard_alias, canonical.clone()));
                }

                resolved.push((soname.clone(), host_path, canonical));
            } else {
                missing.push(MissingLib {
                    soname: soname.clone(),
                });
            }
        }

        // Phase 2: read + parse resolved libs to discover their DT_NEEDED
        // entries and search paths. The interp-relative dirs apply
        // uniformly to every resolve_soname call (via the top-level
        // `interp_search_dirs` slice), so transitive deps don't need them
        // threaded through per-level.
        let mut next_deps: Vec<(String, ElfSearchPaths)> = Vec::new();
        for (_, _, canonical) in &resolved {
            let (lib_data, identity) = read_file_stable(canonical)
                .with_context(|| format!("read resolved shared library {}", canonical.display()))?;
            observed_files.push((canonical.clone(), identity));
            let lib_elf = goblin::elf::Elf::parse(&lib_data).with_context(|| {
                format!("parse resolved shared library {}", canonical.display())
            })?;
            let lib_search = elf_search_paths(&lib_elf, canonical);
            next_deps.extend(
                lib_elf
                    .libraries
                    .iter()
                    .map(|name| (name.to_string(), lib_search.clone())),
            );
        }

        // Build next level from discovered deps, skipping already-visited.
        level = next_deps
            .into_iter()
            .filter(|(soname, _)| !visited.contains(soname))
            .collect();
    }

    let result = SharedLibs {
        found,
        missing,
        interpreter,
        observed_files,
        search_paths: search_observations,
    };

    Ok(result)
}

/// DT_RPATH / DT_RUNPATH directories for a single binary.
///
/// glibc's `ld.so` treats these differently:
/// - **DT_RUNPATH** (modern): consulted AFTER `LD_LIBRARY_PATH`.
/// - **DT_RPATH** (legacy): consulted BEFORE `LD_LIBRARY_PATH`, but
///   only when `DT_RUNPATH` is absent (DT_RUNPATH presence causes the
///   loader to ignore DT_RPATH entirely).
///
/// Collapsing these into a single list (as the prior code did)
/// silently demoted legacy DT_RPATH binaries to DT_RUNPATH order,
/// which can produce different library resolution than the real
/// dynamic linker.
#[derive(Debug, Clone, Default)]
struct ElfSearchPaths {
    /// DT_RPATH directories, with dynamic tokens expanded. Non-empty
    /// only when the binary has DT_RPATH and no DT_RUNPATH (glibc
    /// ignores DT_RPATH when DT_RUNPATH is present).
    rpath: Vec<PathBuf>,
    /// DT_RUNPATH directories, with dynamic tokens expanded.
    runpath: Vec<PathBuf>,
}

/// Extract search paths from DT_RUNPATH and DT_RPATH, with dynamic
/// string tokens expanded:
/// - `$ORIGIN` / `${ORIGIN}`: binary's parent directory
/// - `$LIB` / `${LIB}`: `lib` or `lib64` based on ELF class
/// - `$PLATFORM` / `${PLATFORM}`: `x86_64` or `aarch64`
///
/// Returns the two sets separately so `resolve_soname` can apply
/// glibc's ordering rules; see [`ElfSearchPaths`].
fn elf_search_paths(elf: &goblin::elf::Elf, binary: &Path) -> ElfSearchPaths {
    let origin = binary
        .parent()
        .and_then(|p| std::fs::canonicalize(p).ok())
        .unwrap_or_default();

    let origin_str = origin.to_string_lossy();
    let lib_str = if elf.is_64 { "lib64" } else { "lib" };
    let platform_str = std::env::consts::ARCH;

    let expand = |raw: &str| -> Vec<PathBuf> {
        raw.split(':')
            .filter(|s| !s.is_empty())
            .map(|p| {
                let expanded = p
                    .replace("$ORIGIN", &origin_str)
                    .replace("${ORIGIN}", &origin_str)
                    .replace("$LIB", lib_str)
                    .replace("${LIB}", lib_str)
                    .replace("$PLATFORM", platform_str)
                    .replace("${PLATFORM}", platform_str);
                PathBuf::from(expanded)
            })
            .collect()
    };

    // Modern: DT_RUNPATH is honored and overrides DT_RPATH completely.
    if !elf.runpaths.is_empty() {
        return ElfSearchPaths {
            rpath: Vec::new(),
            runpath: expand(&elf.runpaths.join(":")),
        };
    }
    // Legacy: only DT_RPATH. Searched before LD_LIBRARY_PATH.
    if !elf.rpaths.is_empty() {
        return ElfSearchPaths {
            rpath: expand(&elf.rpaths.join(":")),
            runpath: Vec::new(),
        };
    }
    ElfSearchPaths::default()
}

/// Well-known system dynamic linker paths. If a binary's PT_INTERP
/// canonicalizes to the same file as one of these, it uses the standard
/// linker and does not need the interpreter packed separately.
const STANDARD_INTERPRETERS: &[&str] = &[
    "/lib/ld-linux.so.2",
    "/lib/ld-linux-aarch64.so.1",
    "/lib/ld-linux-armhf.so.3",
    "/lib64/ld-linux-x86-64.so.2",
    "/lib/ld-musl-x86_64.so.1",
    "/lib/ld-musl-aarch64.so.1",
    "/libexec/ld-elf.so.1",
];

/// Check if `interp` is a standard system linker. Compares the
/// canonicalized path against canonicalized well-known linker paths
/// to catch symlinks (e.g. `/opt/toolchain/lib/ld-linux-x86-64.so.2`
/// symlinking to `/lib64/ld-linux-x86-64.so.2`).
fn is_standard_interpreter(interp: &str) -> bool {
    let interp_path = Path::new(interp);
    // Direct match first (avoids syscalls for common case).
    if STANDARD_INTERPRETERS.contains(&interp) {
        return true;
    }
    // Canonicalize and compare against canonical standard paths.
    let Ok(canon) = std::fs::canonicalize(interp_path) else {
        return false;
    };
    STANDARD_INTERPRETERS.iter().any(|std_interp| {
        std::fs::canonicalize(std_interp).is_ok_and(|std_canon| std_canon == canon)
    })
}

/// Default library search paths used by the dynamic linker.
const DEFAULT_LIB_PATHS: &[&str] = &[
    "/lib",
    "/usr/lib",
    "/lib64",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
];

fn observe_search_candidate(
    path: &Path,
    observations: &mut Vec<ResolverPathObservation>,
) -> Result<()> {
    let identity = std::fs::metadata(path)
        .ok()
        .map(|metadata| ResolverFileIdentity::from_metadata(&metadata));
    if let Some(previous) = observations
        .iter()
        .find(|observation| observation.path == path)
    {
        anyhow::ensure!(
            previous.identity == identity,
            "dynamic-library search candidate changed during resolution: {}",
            path.display()
        );
    } else {
        observations.push(ResolverPathObservation {
            path: path.to_path_buf(),
            identity,
        });
    }
    Ok(())
}

fn normalize_loader_search_dir(path: &Path, loader_cwd: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        loader_cwd.to_path_buf()
    } else if path.is_absolute() {
        path.to_path_buf()
    } else {
        loader_cwd.join(path)
    }
}

/// Resolve a soname to a host path.
/// Search order matches the host dynamic linker (ld.so):
///   1. DT_RPATH (ONLY if DT_RUNPATH is absent — legacy order)
///   2. LD_LIBRARY_PATH
///   3. DT_RUNPATH (modern; ignored when DT_RPATH was used above)
///   4. interp-relative hints (custom toolchain support, not part of
///      glibc; treated as "RUNPATH-adjacent" to keep LD_LIBRARY_PATH
///      able to override them)
///   5. /etc/ld.so.cache (binary cache from ldconfig — already
///      covers everything in /etc/ld.so.conf, so no separate
///      conf-walk step)
///   6. Default library paths (/lib, /usr/lib, etc.)
///
/// This matches glibc ld.so(8) — specifically, DT_RPATH takes
/// priority over `LD_LIBRARY_PATH` when it is the binary's only
/// rpath-style entry, and DT_RUNPATH is consulted only AFTER
/// `LD_LIBRARY_PATH` so an admin override still wins.
fn resolve_soname_with_loader(
    soname: &str,
    elf_paths: &ElfSearchPaths,
    interp_hints: &[PathBuf],
    loader_cwd: &Path,
    ld_library_path_dirs: &[PathBuf],
    ld_so_cache: &LdSoCache,
    observations: &mut Vec<ResolverPathObservation>,
) -> Result<Option<PathBuf>> {
    // 1. DT_RPATH (legacy). Non-empty only when DT_RUNPATH is absent
    //    per `elf_search_paths`; matches glibc's "DT_RPATH before
    //    LD_LIBRARY_PATH" rule for pre-RUNPATH binaries.
    for dir in &elf_paths.rpath {
        let dir = normalize_loader_search_dir(dir, loader_cwd);
        let candidate = dir.join(soname);
        observe_search_candidate(&candidate, observations)?;
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }

    // 2. LD_LIBRARY_PATH.
    for dir in ld_library_path_dirs {
        let dir = normalize_loader_search_dir(dir, loader_cwd);
        let candidate = dir.join(soname);
        observe_search_candidate(&candidate, observations)?;
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }

    // 3. DT_RUNPATH (modern).
    for dir in &elf_paths.runpath {
        let dir = normalize_loader_search_dir(dir, loader_cwd);
        let candidate = dir.join(soname);
        observe_search_candidate(&candidate, observations)?;
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }

    // 4. Interp-relative hints for non-standard dynamic linkers.
    //    Not a glibc concept — ktstr uses these to keep custom
    //    toolchain libs resolvable without requiring the user to
    //    set LD_LIBRARY_PATH.
    for dir in interp_hints {
        let dir = normalize_loader_search_dir(dir, loader_cwd);
        let candidate = dir.join(soname);
        observe_search_candidate(&candidate, observations)?;
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }

    // 5. ld.so.cache — the binary cache is the real dynamic linker's
    //    primary lookup mechanism. Catches libraries in directories
    //    added via `ldconfig /path` that don't appear in ld.so.conf.
    if let Some(candidates) = ld_so_cache.candidates.get(soname) {
        for cached_path in candidates {
            observe_search_candidate(cached_path, observations)?;
            if cached_path.is_file() {
                return Ok(Some(cached_path.clone()));
            }
        }
    }

    // 6. Default paths. (ld.so.conf step dropped: ldconfig already
    //    ingests conf paths into ld.so.cache above, so a separate
    //    walk here was redundant per glibc's search algorithm.)
    for dir in DEFAULT_LIB_PATHS {
        let dir = Path::new(dir);
        let candidate = dir.join(soname);
        observe_search_candidate(&candidate, observations)?;
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }

    Ok(None)
}

#[cfg(test)]
fn resolve_soname(
    soname: &str,
    elf_paths: &ElfSearchPaths,
    interp_hints: &[PathBuf],
) -> Option<PathBuf> {
    let loader_cwd = loader_current_dir().ok()?;
    let ld_library_path_dirs = current_ld_library_path_dirs();
    let ld_so_cache = parse_ld_so_cache(Path::new("/etc/ld.so.cache"));
    let mut observations = Vec::new();
    resolve_soname_with_loader(
        soname,
        elf_paths,
        interp_hints,
        &loader_cwd,
        &ld_library_path_dirs,
        &ld_so_cache,
        &mut observations,
    )
    .ok()
    .flatten()
}

/// ELF magic bytes: `\x7fELF`.
#[cfg(test)]
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// Check if the first 4 bytes of a file match ELF magic.
#[cfg(test)]
fn is_elf(path: &Path) -> bool {
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            Ok(magic)
        })
        .is_ok_and(|m| m == *ELF_MAGIC)
}

/// Write one entry (file or directory) into the cpio archive.
fn write_entry(archive: &mut Vec<u8>, name: &str, data: &[u8], mode: u32) -> Result<()> {
    let builder = cpio::newc::Builder::new(name).mode(mode).nlink(1);
    let mut writer = builder.write(archive as &mut dyn Write, data.len() as u32);
    writer
        .write_all(data)
        .with_context(|| format!("write cpio entry '{name}'"))?;
    writer.finish().context("finish cpio entry")?;
    Ok(())
}

/// Write a cpio symlink entry. `name` is the symlink path, `target` is the
/// absolute path it points to. Mode is S_IFLNK | 0777 = 0o120777.
fn write_symlink_entry(archive: &mut Vec<u8>, name: &str, target: &str) -> Result<()> {
    let target_bytes = target.as_bytes();
    let builder = cpio::newc::Builder::new(name).mode(0o120777).nlink(1);
    let mut writer = builder.write(archive as &mut dyn Write, target_bytes.len() as u32);
    writer
        .write_all(target_bytes)
        .with_context(|| format!("write cpio symlink '{name}' -> '{target}'"))?;
    writer.finish().context("finish cpio symlink entry")?;
    Ok(())
}

/// Section names removed during debug stripping. These contain debug
/// info and compiler metadata (`.comment`) that inflate the binary but
/// are not needed inside the VM. Coverage profiling sections
/// (`__llvm_prf_*`) are deliberately absent from this list — guest
/// coverage capture needs them resident.
const DEBUG_SECTIONS: &[&[u8]] = &[
    b".debug_info",
    b".debug_abbrev",
    b".debug_line",
    b".debug_line_str",
    b".debug_str",
    b".debug_ranges",
    b".debug_aranges",
    b".debug_frame",
    b".debug_loc",
    b".debug_loclists",
    b".debug_rnglists",
    b".debug_str_offsets",
    b".debug_addr",
    b".debug_pubtypes",
    b".debug_pubnames",
    b".debug_types",
    b".debug_macro",
    b".debug_macinfo",
    b".comment",
];

/// Strip debug sections from an ELF binary to reduce initramfs size.
/// Debug info can be 10-50x the loadable segment size and is not needed
/// inside the VM. Uses the `object` crate to parse and rewrite the ELF,
/// removing non-loadable debug sections. Falls back to the original
/// binary on parse or write failure.
///
/// When the binary has been deleted (e.g. by `cargo llvm-cov`),
/// retries via `/proc/self/exe` which remains valid as long as the
/// process is alive.
fn strip_debug(path: &Path) -> Result<Vec<u8>> {
    // Try the original path first, then /proc/self/exe if the binary
    // was deleted (cargo llvm-cov deletes binaries after instrumenting).
    let paths_to_try: Vec<&Path> = if is_deleted_self(path) {
        vec![path, Path::new("/proc/self/exe")]
    } else {
        vec![path]
    };

    for src in &paths_to_try {
        if let Ok(data) = std::fs::read(src) {
            match strip_debug_sections(&data) {
                Ok(stripped) => return Ok(stripped),
                Err(e) => {
                    // object crate failed to parse/write — log and
                    // fall back to the unstripped binary so the VM
                    // boot proceeds. Without the warn the operator
                    // sees a 900 MiB initramfs with no clue why
                    // stripping silently no-op'd.
                    tracing::warn!(
                        binary = %src.display(),
                        error = %e,
                        "strip_debug_sections failed, using unstripped binary"
                    );
                    return Ok(data);
                }
            }
        }
    }

    std::fs::read(path).with_context(|| format!("read binary: {}", path.display()))
}

/// Remove debug sections from ELF data using the shared
/// [`crate::elf_strip::rewrite`] primitive. Thin filter — delete
/// sections whose name is in the explicit [`DEBUG_SECTIONS`] list.
fn strip_debug_sections(data: &[u8]) -> std::result::Result<Vec<u8>, object::build::Error> {
    crate::elf_strip::rewrite(data, |name| DEBUG_SECTIONS.contains(&name))
}

/// Check if `path` is the current executable and has been deleted.
fn is_deleted_self(path: &Path) -> bool {
    let proc_exe = Path::new("/proc/self/exe");
    let Ok(target) = std::fs::read_link(proc_exe) else {
        return false;
    };
    let target_str = target.to_string_lossy();
    target_str.ends_with(" (deleted)")
        && target_str.trim_end_matches(" (deleted)") == path.to_string_lossy().as_ref()
}

/// Expand `guest_path`'s parent into every ancestor directory component
/// and insert each into `dirs`. No-op when the path has no parent
/// (e.g. top-level files like `init`). The component walk produces every
/// intermediate directory, e.g. `include-files/sub/f` registers
/// `include-files` and `include-files/sub`.
fn register_parent_dirs(dirs: &mut BTreeSet<String>, guest_path: &str) {
    let Some(parent) = Path::new(guest_path).parent() else {
        return;
    };
    let mut dir = PathBuf::new();
    for component in parent.components() {
        dir.push(component);
        dirs.insert(dir.to_string_lossy().to_string());
    }
}

/// Build a base archive from an already-resolved, caller-pinned dependency
/// closure.
///
/// Every host path is normally `/proc/<preparer-pid>/fd/N`. The persistent
/// closure record and [`crate::vmm::initramfs_cache`] retain those exact fds
/// through semantic key construction and this write, so the builder never
/// reopens a mutable original pathname or repeats ELF resolution on a CAS hit.
pub(crate) fn build_initramfs_base_from_resolved(
    extra_binaries: &[(&str, &Path)],
    include_files: &[(&str, &Path, u32)],
    busybox_bytes: Option<&[u8]>,
    shared_libs: &[(String, PathBuf, u64, u64)],
) -> Result<Vec<u8>> {
    let mut dirs = BTreeSet::new();
    if busybox_bytes.is_some() {
        dirs.insert("bin".to_string());
    }
    for (archive_path, _, mode) in include_files {
        validate_include_archive_path(archive_path)?;
        anyhow::ensure!(
            mode & libc::S_IFMT == libc::S_IFREG,
            "resolved include file '{}' has non-regular archive mode {mode:#o}",
            archive_path
        );
        register_parent_dirs(&mut dirs, archive_path);
    }
    for (archive_path, _) in extra_binaries {
        register_parent_dirs(&mut dirs, archive_path);
    }
    for (guest_path, _, _, _) in shared_libs {
        register_parent_dirs(&mut dirs, guest_path);
    }
    let shared_sources: Vec<(String, PathBuf)> = shared_libs
        .iter()
        .map(|(guest, host, _, _)| (guest.clone(), host.clone()))
        .collect();
    let content_keys: Vec<(u64, u64)> = shared_libs
        .iter()
        .map(|(_, _, size, hash)| (*size, *hash))
        .collect();
    let mut archive = Vec::new();
    write_archive_entries(
        &mut archive,
        &dirs,
        busybox_bytes,
        extra_binaries,
        include_files,
        &shared_sources,
        Some(&content_keys),
    )?;
    Ok(archive)
}

fn validate_include_archive_path(archive_path: &str) -> Result<()> {
    if Path::new(archive_path)
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        anyhow::bail!("include_files archive path contains '..': {archive_path}");
    }
    if archive_path.starts_with(".ktstr_") {
        anyhow::bail!("include_files archive path must not start with '.ktstr_': {archive_path}");
    }
    Ok(())
}

/// Write all cpio entries for the base archive in extractor-safe order:
/// directories first, then busybox, extra binaries (stripped), include
/// files (verbatim), and shared libraries.
///
/// Shared libraries are deduplicated by canonical host path: the first
/// guest path for a host file is written as a regular file, later guest
/// paths mapping to the same file become cpio symlinks. The
/// `write_cpio` span brackets these writes.
fn write_archive_entries(
    archive: &mut Vec<u8>,
    dirs: &BTreeSet<String>,
    busybox_bytes: Option<&[u8]>,
    extra_binaries: &[(&str, &Path)],
    validated_includes: &[(&str, &Path, u32)],
    shared_libs: &[(String, PathBuf)],
    shared_lib_content_keys: Option<&[(u64, u64)]>,
) -> Result<()> {
    let _s_write = tracing::debug_span!("write_cpio").entered();
    // Directory entries
    for dir in dirs {
        write_entry(archive, dir, &[], 0o40755)?;
    }

    // Shell mode: embed busybox bytes provided by the caller. The
    // ktstr library does not own the bytes — they come from the
    // `KTSTR_BUSYBOX_PATH` env var that cargo-ktstr sets at startup
    // (see [`crate::vmm::blobs::load_busybox_bytes`]).
    if let Some(busybox_bytes) = busybox_bytes {
        write_entry(archive, "bin/busybox", busybox_bytes, 0o100755)?;
    }

    // Extra binaries (stripped to reduce initramfs size)
    for (name, path) in extra_binaries {
        let data = strip_debug(path)
            .with_context(|| format!("strip/read extra binary '{}': {}", name, path.display()))?;
        write_entry(archive, name, &data, 0o100755)?;
    }

    // Include files: copied verbatim, preserving original content and
    // debug symbols. No strip_debug — included files are user-provided
    // and may be non-ELF.
    for (archive_path, host_path, mode) in validated_includes {
        let data = std::fs::read(host_path).with_context(|| {
            format!(
                "read include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        write_entry(archive, archive_path, &data, *mode)?;
    }

    // Shared libraries — write each canonical host file once as a regular
    // file, then write subsequent guest paths that map to the same host
    // file as cpio symlinks. This avoids duplicating large libraries in
    // the initramfs (e.g. libc appearing under both lib64/ and usr/lib64/).
    {
        // Open-file identity -> first guest_path written for this file. This
        // remains stable for `/proc/<pid>/fd/N` sources even after the
        // original pathname is replaced or unlinked.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        enum SharedLibDedupKey {
            Identity(u64, u64),
            Content(u64, u64),
        }
        if let Some(keys) = shared_lib_content_keys {
            anyhow::ensure!(
                keys.len() == shared_libs.len(),
                "prepared shared-library content-key count mismatch"
            );
        }
        let mut written_files: HashMap<SharedLibDedupKey, String> = HashMap::new();
        for (index, (guest_path, host_path)) in shared_libs.iter().enumerate() {
            let key = if let Some(keys) = shared_lib_content_keys {
                let (size, hash) = keys[index];
                SharedLibDedupKey::Content(size, hash)
            } else {
                let metadata = std::fs::metadata(host_path).with_context(|| {
                    format!("stat shared lib '{}': {}", guest_path, host_path.display())
                })?;
                SharedLibDedupKey::Identity(metadata.dev(), metadata.ino())
            };
            if let Some(first_guest) = written_files.get(&key) {
                // Already written — emit a symlink to the first guest path.
                let target = format!("/{first_guest}");
                write_symlink_entry(archive, guest_path, &target)?;
            } else {
                let data = std::fs::read(host_path).with_context(|| {
                    format!("read shared lib '{}': {}", guest_path, host_path.display())
                })?;
                write_entry(archive, guest_path, &data, 0o100755)?;
                written_files.insert(key, guest_path.clone());
            }
        }
    }

    drop(_s_write);

    Ok(())
}

/// Per-invocation inputs that turn a cached base archive into a
/// complete initramfs. Borrows all slices so callers can build a
/// `SuffixParams` without copying `Vec<String>` fields.
#[derive(Default)]
pub struct SuffixParams<'a> {
    /// The test binary, packed as `/init` (the kernel's rdinit entry).
    /// `strip_debug`'d into the suffix — not the cached base — so a
    /// payload recompile that leaves the shared-lib set unchanged is a
    /// base-cache hit. `None` only in suffix-shape unit tests; every
    /// real boot path sets it from the VM's `init_binary`.
    pub payload: Option<&'a Path>,
    /// `/args` contents — one entry per line.
    pub args: &'a [String],
    /// `/sched_args` contents, or empty to skip the entry.
    pub sched_args: &'a [String],
    /// `/sched_enable` shell-script lines for kernel-built
    /// schedulers, or empty to skip the entry.
    pub sched_enable: &'a [String],
    /// `/sched_disable` shell-script lines, or empty to skip.
    pub sched_disable: &'a [String],
    /// `/exec_cmd` contents when `--exec` is used; `None` otherwise.
    pub exec_cmd: Option<&'a str>,
    /// Per-staged-scheduler args files. Each tuple is
    /// `(scheduler_name, args)`; the suffix emits a
    /// `staging/schedulers/<name>/sched_args` cpio entry per
    /// non-empty `args` slice, joined by `\n` to match the
    /// boot-time `/sched_args` parser shape.
    ///
    /// Parent directories (`staging`, `staging/schedulers`,
    /// `staging/schedulers/<name>`) are registered when the
    /// matching staged scheduler binary is packed into the base
    /// archive by the prepared base builder (its
    /// `register_parent_dirs` loop walks every extras entry's
    /// path components). The base + suffix split is deliberate:
    /// binary content is stable enough to benefit from the
    /// content-hash cache; args vary per-run and pay the
    /// recompress cost regardless. The name MUST satisfy
    /// [`crate::test_support::staged::validate_staged_scheduler_name`]
    /// — empty slice (zero staged schedulers) is the common case
    /// and emits no entries.
    pub staged_sched_args: &'a [(String, Vec<String>)],
    /// `/workload_root_cgroup` file contents when set. Sourced from
    /// [`KtstrTestEntry::workload_root_cgroup`](crate::test_support::KtstrTestEntry::workload_root_cgroup);
    /// the guest's `rust_init` reads this file BEFORE starting the
    /// scheduler and mkdir's `/sys/fs/cgroup{path}` so the workload
    /// CgroupManager has its root in place. Absent ⇒ no file
    /// emitted ⇒ guest falls back to the legacy
    /// `--cell-parent-cgroup`-or-default resolution.
    pub workload_root_cgroup: Option<&'a str>,
    /// `/scheduler_cgroup_parent` file contents when set. Sourced
    /// from [`crate::test_support::Scheduler::cgroup_parent`]; the
    /// guest mkdir's `/sys/fs/cgroup{path}` + enables `+cpuset +cpu`
    /// on every ancestor BEFORE starting the scheduler so the
    /// scheduler attaches into a ready cgroup tree. Absent ⇒ no
    /// file emitted ⇒ scheduler runs at the cgroup root without
    /// explicit framework placement.
    pub scheduler_cgroup_parent: Option<&'a str>,
    /// Kernel modules to load in the guest, in caller-determined load
    /// order. Each is a raw (already-decompressed) `.ko`; `build_suffix`
    /// embeds them under `modules/NNN-<filename>` (zero-padded index) so
    /// a lexical sort in the guest reproduces this slice's order. The
    /// guest's `rust_init::load_kernel_modules` loads them via
    /// `finit_module(2)` right after mounting devtmpfs and BEFORE
    /// touching any virtio device — required for prebuilt distro kernels
    /// that ship virtio (blk/console/net) as modules. Empty ⇒ no
    /// `modules/` entries are emitted and the suffix bytes are identical
    /// to the pre-module output; the ktstr-built kernels pin virtio =y
    /// and pass an empty slice. Which modules to include is the caller's
    /// decision (host integration derives them from the target kernel).
    pub kernel_modules: &'a [PathBuf],
}

/// Compatibility helper for suffix-shape tests.
///
/// Production prepares three independent archive fragments: the stripped
/// `/init` entry, the stable kernel-module set, and the tiny per-cell tail.
/// Keeping this wrapper in tests preserves the old "one suffix Vec" fixtures
/// while exercising the same split builders used by the CAS.
#[cfg(test)]
pub fn build_suffix(base_len: usize, params: &SuffixParams<'_>) -> Result<Vec<u8>> {
    let mut suffix = Vec::new();
    if let Some(payload) = params.payload {
        suffix.extend(build_payload_part_from_pinned(payload)?);
    }
    let modules: Vec<(String, PathBuf)> = params
        .kernel_modules
        .iter()
        .map(|module| {
            let file_name = module
                .file_name()
                .and_then(|name| name.to_str())
                .with_context(|| {
                    format!(
                        "kernel module path has no valid filename: {}",
                        module.display()
                    )
                })?
                .to_owned();
            Ok((file_name, module.clone()))
        })
        .collect::<Result<_>>()?;
    suffix.extend(build_modules_part_from_pinned(&modules)?);
    suffix.extend(build_dynamic_tail(base_len + suffix.len(), params)?);
    Ok(suffix)
}

/// Build the immutable archive fragment containing only `/init`.
///
/// `payload` normally names `/proc/<preparer-pid>/fd/N`; the preparer keeps
/// that fd open and verifies its identity again before publication. This
/// expensive strip is therefore elected and cached independently of per-cell
/// arguments.
pub(crate) fn build_payload_part_from_pinned(payload: &Path) -> Result<Vec<u8>> {
    let binary = strip_debug(payload)
        .with_context(|| format!("strip/read pinned init binary: {}", payload.display()))?;
    let mut part = Vec::new();
    write_entry(&mut part, "init", &binary, 0o100755)?;
    Ok(part)
}

/// Build the immutable archive fragment containing a stable module set.
///
/// Archive names stay separate from fd paths because the latter end in a
/// descriptor number. Empty input deliberately returns an empty fragment.
pub(crate) fn build_modules_part_from_pinned(modules: &[(String, PathBuf)]) -> Result<Vec<u8>> {
    if modules.is_empty() {
        return Ok(Vec::new());
    }
    let mut part = Vec::new();
    write_entry(&mut part, "modules", &[], 0o40755)?;
    for (idx, (file_name, module)) in modules.iter().enumerate() {
        let data = std::fs::read(module)
            .with_context(|| format!("read pinned kernel module: {}", module.display()))?;
        let archive_path = format!("modules/{idx:03}-{file_name}");
        write_entry(&mut part, &archive_path, &data, 0o100644)?;
    }
    Ok(part)
}

/// The guest reads `/args` and `/sched_args` one argument per line
/// (`test_support::parse_line_framed_args`), so an argument containing a
/// line break would silently re-split into separate arguments after the
/// `join("\n")` below. The declaration macros reject such literals at
/// compile time; this gate catches every non-literal route.
fn reject_unframable_args(kind: &str, args: &[String]) -> Result<()> {
    for argument in args {
        anyhow::ensure!(
            !argument.contains(['\n', '\r']),
            "{kind} element {argument:?} contains a line break, which would \
             split it into separate guest argv entries",
        );
    }
    Ok(())
}

/// Build the tiny per-cell tail that closes the cpio archive.
///
/// This contains every invocation-varying byte, the completion sentinel,
/// trailer, and final 512-byte archive padding. `prefix_len` is the exact
/// uncompressed length of all immutable fragments before this tail.
pub(crate) fn build_dynamic_tail(prefix_len: usize, params: &SuffixParams<'_>) -> Result<Vec<u8>> {
    let mut suffix = Vec::new();

    // Args file
    reject_unframable_args("args", params.args)?;
    let args_data = params.args.join("\n");
    write_entry(&mut suffix, "args", args_data.as_bytes(), 0o100644)?;

    // Scheduler args file
    if !params.sched_args.is_empty() {
        reject_unframable_args("sched_args", params.sched_args)?;
        let sched_args_data = params.sched_args.join("\n");
        write_entry(
            &mut suffix,
            "sched_args",
            sched_args_data.as_bytes(),
            0o100644,
        )?;
    }

    // Kernel-built scheduler enable/disable scripts
    if !params.sched_enable.is_empty() {
        let data = params.sched_enable.join("\n");
        write_entry(&mut suffix, "sched_enable", data.as_bytes(), 0o100755)?;
    }
    if !params.sched_disable.is_empty() {
        let data = params.sched_disable.join("\n");
        write_entry(&mut suffix, "sched_disable", data.as_bytes(), 0o100755)?;
    }

    if let Some(cmd) = params.exec_cmd {
        write_entry(&mut suffix, "exec_cmd", cmd.as_bytes(), 0o100644)?;
    }

    // `/workload_root_cgroup` carries the per-test workload cgroup
    // root the guest reads in Phase 3 of `rust_init` (just before
    // `start_scheduler`). The host writes the validated absolute
    // path verbatim — the guest's `create_workload_root_cgroup_from_file`
    // re-validates the leading `/` + "not bare /" gate before
    // calling mkdir, so a stale/hand-edited image with a bad value
    // still fails closed rather than corrupting unrelated cgroup
    // state.
    if let Some(path) = params.workload_root_cgroup {
        write_entry(
            &mut suffix,
            "workload_root_cgroup",
            path.as_bytes(),
            0o100644,
        )?;
    }

    // `/scheduler_cgroup_parent` carries the per-scheduler cgroup
    // placement target (from `Scheduler::cgroup_parent`). Same
    // re-validation flow as workload_root_cgroup — the guest's
    // `create_scheduler_cgroup_parent_from_file` re-checks the
    // path-shape gate before mkdir so a stale or hand-edited image
    // carrying a bad value still fails closed rather than
    // corrupting unrelated cgroup state.
    if let Some(path) = params.scheduler_cgroup_parent {
        write_entry(
            &mut suffix,
            "scheduler_cgroup_parent",
            path.as_bytes(),
            0o100644,
        )?;
    }

    // Per-staged-scheduler args files. The staged binary itself is
    // packed into the base archive by
    // `build_initramfs_base_from_resolved` as an
    // extras entry under `staging/schedulers/<name>/scheduler`;
    // its `register_parent_dirs` loop populates the directory
    // chain so this suffix entry lands inside a pre-existing tree.
    // Names are pre-validated upstream by
    // `validate_staged_scheduler_name` (rejects path separators,
    // NUL bytes, reserved names, dot prefix), so the format! here
    // cannot produce a path that escapes the staging scope or
    // collides with a boot-time slot.
    for (name, args) in params.staged_sched_args {
        if args.is_empty() {
            continue;
        }
        reject_unframable_args("staged sched_args", args)?;
        let archive_path = format!(
            "{}/sched_args",
            crate::test_support::staged::staged_scheduler_archive_dir(name)
        );
        let data = args.join("\n");
        write_entry(&mut suffix, &archive_path, data.as_bytes(), 0o100644)?;
    }

    // Sentinel: the LAST entry before the trailer. The kernel extracts the
    // whole initramfs before exec'ing /init, so this file's ABSENCE means
    // the kernel silently dropped late cpio entries under memory pressure
    // (do_name's filp_open ENOMEM returns without error — skips the entry
    // but keeps consuming the stream). The guest's `rust_init` checks for it.
    write_entry(&mut suffix, ".ktstr_init_ok", &[], 0o100644)?;

    // Trailer
    cpio::newc::trailer(&mut suffix as &mut dyn Write).context("write cpio trailer")?;

    // Pad to 512-byte boundary (initramfs convention)
    let total = prefix_len + suffix.len();
    let pad = (512 - (total % 512)) % 512;
    suffix.extend(std::iter::repeat_n(0u8, pad));

    Ok(suffix)
}

/// RAII guard for a live COW-overlay mapping.
///
/// A prepared COW extent is a `MAP_PRIVATE | MAP_FIXED_NOREPLACE` file VMA
/// installed into a prevalidated hole. `MAP_PRIVATE` pages are lazily read
/// from the page cache on first access. Holding the fd with `LOCK_SH` for the
/// mapping's lifetime prevents the prepared-initrd GC from unlinking the
/// backing object until after the mapping is torn down.
///
/// Drop order: the guard releases `LOCK_UN` and `close` only. The
/// file VMA itself is owned by the caller's VA reservation
/// (e.g. `ReservationGuard` in the VMM) and is munmapped when that
/// reservation drops — which must happen BEFORE this guard drops,
/// so the lock protects the mapping right up until tear-down.
pub(crate) struct CowOverlayGuard {
    fd: std::os::fd::OwnedFd,
}

impl CowOverlayGuard {
    pub(crate) fn new(fd: std::os::fd::OwnedFd) -> Self {
        Self { fd }
    }
}

impl Drop for CowOverlayGuard {
    fn drop(&mut self) {
        // Release LOCK_SH explicitly so GC waiting on LOCK_EX observes
        // ordering with the VM's reads. OwnedFd closes the descriptor
        // after this function returns.
        let _ = super::initramfs_cache::flock_retry(&self.fd, rustix::fs::FlockOperation::Unlock);
    }
}

/// Install one immutable prepared-initrd extent into an already-unmapped
/// destination hole.
///
/// `MAP_FIXED_NOREPLACE` makes the final VMA layout constructive rather than
/// destructive: primaries are split around every boundary overlay, and no
/// file mapping is ever overwritten by a later mapping. A stale or accidental
/// mapping in the destination is therefore an error instead of being silently
/// replaced.
///
/// # Safety
///
/// The caller must own the complete destination range, have already unmapped
/// exactly `[host_addr, host_addr + len)`, validate the source extent and
/// alignment, and retain `backing_fd` until the containing reservation is
/// torn down.
pub(crate) unsafe fn cow_map_file_into_hole_borrowed(
    host_addr: *mut u8,
    len: usize,
    backing_fd: &std::os::fd::OwnedFd,
    file_offset: u64,
) -> Result<()> {
    use std::os::fd::AsRawFd;

    let ptr = unsafe {
        libc::mmap(
            host_addr.cast(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_FIXED_NOREPLACE,
            backing_fd.as_raw_fd(),
            libc::off_t::try_from(file_offset).context("COW file offset exceeds off_t")?,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error())
            .context("MAP_PRIVATE prepared initrd mapping into reserved hole");
    }
    if ptr != host_addr.cast() {
        // Linux before MAP_FIXED_NOREPLACE support may ignore the flag and
        // choose a different free address. Never leak that mapping or accept
        // a non-exact guest-memory layout.
        unsafe {
            libc::munmap(ptr, len);
        }
        anyhow::bail!("MAP_FIXED_NOREPLACE was ignored: requested {host_addr:p}, got {ptr:p}");
    }
    Ok(())
}

/// Write one or more byte slices sequentially into guest memory as a
/// single contiguous initramfs. Passing a one-element `parts` slice
/// matches the "single blob" caller; multi-element slices avoid the
/// copy into a monolithic `Vec` that the split base/suffix production
/// path would otherwise need. Returns (address, total_size) for
/// boot_params.
#[cfg(test)]
pub fn load_initramfs_parts(
    guest_mem: &vm_memory::GuestMemoryMmap,
    parts: &[&[u8]],
    load_addr: u64,
) -> Result<(u64, u32)> {
    use vm_memory::{Bytes, GuestAddress};
    let mut offset = 0u64;
    for part in parts {
        guest_mem
            .write_slice(part, GuestAddress(load_addr + offset))
            .context("write initramfs part to guest memory")?;
        offset += part.len() as u64;
    }
    Ok((load_addr, offset as u32))
}

/// Which compression the guest kernel's initramfs unpacker can decode
/// for the initrd ktstr assembles.
///
/// ktstr-built kernels pin `CONFIG_RD_LZ4=y` (ktstr.kconfig), but a
/// prebuilt distro kernel chooses its own `CONFIG_RD_*` set — AL2023,
/// for instance, ships only `RD_GZIP` + `RD_ZSTD` — and an initrd in a
/// format the kernel lacks dies at boot with "Initramfs unpacking
/// failed: decompressor failed". The boot path selects a variant from
/// the kernel's extracted `.config`
/// (`crate::cache::initrd_compression_for_image`), defaulting to
/// [`InitrdCompression::Lz4`] when no config is available (built
/// kernels, raw images — today's behavior).
///
/// The base and suffix are compressed as SEPARATE streams of the
/// chosen format and concatenated; the kernel's `unpack_to_rootfs`
/// loop (init/initramfs.c) decodes one compressed archive per
/// iteration, advancing by the decompressor-reported consumed length,
/// so back-to-back frames of any supported format unpack like a single
/// initramfs. `Uncompressed` is plain newc cpio, which every kernel
/// unpacks with no `CONFIG_RD_*` at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitrdCompression {
    /// LZ4 legacy frames (`CONFIG_RD_LZ4`-gated) — the default for
    /// ktstr-built kernels.
    Lz4,
    /// One zstd frame per part (`CONFIG_RD_ZSTD`). Level 1: the
    /// window log stays far under the unpacker's 8 MiB
    /// `ZSTD_WINDOWSIZE_MAX` cap (lib/decompress_unzstd.c).
    Zstd,
    /// One gzip member per part (`CONFIG_RD_GZIP`).
    Gzip,
    /// Plain newc cpio — universally bootable last resort.
    Uncompressed,
}

/// Compress one immutable initrd part in `comp`'s format. Every variant is
/// published through the prepared-initrd CAS and can back direct COW mappings.
pub(crate) fn compress_initrd_part(comp: InitrdCompression, data: &[u8]) -> Result<Vec<u8>> {
    match comp {
        InitrdCompression::Lz4 => Ok(lz4_legacy_compress(data)),
        InitrdCompression::Zstd => {
            zstd::stream::encode_all(data, 1).context("zstd-compress initrd part")
        }
        InitrdCompression::Gzip => {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            enc.write_all(data).context("gzip-compress initrd part")?;
            enc.finish().context("finish gzip initrd part")
        }
        InitrdCompression::Uncompressed => Ok(data.to_vec()),
    }
}

/// LZ4 legacy format magic number (`0x184C2102` little-endian).
/// This is the format the kernel's initramfs decompressor expects
/// (CONFIG_RD_LZ4 / lib/decompress_unlz4.c).
pub(crate) const LZ4_LEGACY_MAGIC: [u8; 4] = 0x184C2102u32.to_le_bytes();

/// Maximum uncompressed chunk size for LZ4 legacy format.
/// Must match `LZ4_DEFAULT_UNCOMPRESSED_CHUNK_SIZE` in the kernel
/// (lib/decompress_unlz4.c: `8 << 20`).
const LZ4_CHUNK_SIZE: usize = 8 << 20;

/// Compress `data` into LZ4 legacy frame format for the kernel's
/// initramfs decompressor. The format is:
///   [4-byte magic] ([4-byte compressed_size LE] [compressed block])*
///
/// Input is split into `LZ4_CHUNK_SIZE` (8MB) chunks and compressed
/// sequentially. CAS publication already elects one compressor for an exact
/// object; using rayon here created a host-sized process-global worker pool
/// in every distinct winner process and left those workers alive for the
/// whole VM, collapsing colocated verifier storms.
pub(crate) fn lz4_legacy_compress(data: &[u8]) -> Vec<u8> {
    let compressed_chunks: Vec<Vec<u8>> = data
        .chunks(LZ4_CHUNK_SIZE)
        .map(lz4_flex::block::compress)
        .collect();

    // Assemble: magic + (size + data) per chunk.
    let total: usize = 4 + compressed_chunks.iter().map(|c| 4 + c.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&LZ4_LEGACY_MAGIC);
    for chunk in &compressed_chunks {
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out
}

#[cfg(test)]
#[path = "initramfs_tests.rs"]
mod tests;
