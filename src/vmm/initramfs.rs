//! Minimal initramfs (cpio newc format) creation via the `cpio` crate.
//! Packs the test binary as `/init` along with scheduler binaries,
//! shared libraries, optional busybox, and user-provided include files
//! into a cpio archive for use as Linux initrd.
//! Init setup is handled by Rust code in `vmm::rust_init`.
use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Result of shared library resolution for a binary.
#[derive(Debug, Clone)]
pub(crate) struct SharedLibs {
    /// Resolved `(guest_path, host_path)` pairs.
    pub found: Vec<(String, PathBuf)>,
    /// Library sonames that could not be resolved to a host path.
    pub missing: Vec<MissingLib>,
    /// The binary's PT_INTERP path, if present (e.g. `/lib64/ld-linux-x86-64.so.2`).
    pub interpreter: Option<String>,
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
static LD_SO_CACHE: LazyLock<HashMap<String, PathBuf>> =
    LazyLock::new(|| parse_ld_so_cache(Path::new("/etc/ld.so.cache")));

/// Magic bytes at the start of the glibc new-format `ld.so.cache`.
const LD_CACHE_MAGIC: &[u8; 20] = b"glibc-ld.so.cache1.1";
/// Header size: magic(20) + nlibs(4) + len_strings(4) + flags(4) + unused(16).
const LD_CACHE_HEADER_SIZE: usize = 48;
/// Per-entry size: flags(4) + key(4) + value(4) + osversion(4) + hwcap(8).
const LD_CACHE_ENTRY_SIZE: usize = 24;

/// Parse the binary `/etc/ld.so.cache` file into a soname->path map.
///
/// Scans for the new-format magic because some systems prepend the
/// old format (`ld.so-1.7.0`) before the new-format section.
fn parse_ld_so_cache(path: &Path) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return map,
    };
    // Scan for new-format magic. Usually at offset 0, but old-format
    // systems prepend the legacy section.
    let Some(magic_pos) = data
        .windows(LD_CACHE_MAGIC.len())
        .position(|w| w == LD_CACHE_MAGIC)
    else {
        return map;
    };
    let hdr = magic_pos;
    if data.len() < hdr + LD_CACHE_HEADER_SIZE {
        return map;
    }
    let nlibs = u32::from_le_bytes(data[hdr + 20..hdr + 24].try_into().unwrap()) as usize;
    let min_size = hdr + LD_CACHE_HEADER_SIZE + nlibs * LD_CACHE_ENTRY_SIZE;
    if data.len() < min_size {
        return map;
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
        // Only accept absolute paths that exist as files.
        if path_str.starts_with('/') {
            let p = PathBuf::from(path_str);
            if p.is_file() {
                map.entry(soname.to_string()).or_insert(p);
            }
        }
    }
    map
}

/// Read a null-terminated C string from `data` at `offset`.
fn read_cstr(data: &[u8], offset: usize) -> Option<&str> {
    let end = data[offset..].iter().position(|&b| b == 0)?;
    std::str::from_utf8(&data[offset..offset + end]).ok()
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
pub(crate) fn resolve_shared_libs(binary: &Path) -> Result<SharedLibs> {
    resolve_shared_libs_inner(binary, &[])
}

/// Like [`resolve_shared_libs`] but seeds the BFS with additional
/// interp-relative hint directories on top of the ones derived from
/// the binary's own PT_INTERP. Use this when walking the interpreter
/// itself: the linker has no PT_INTERP of its own, so the auto-derived
/// hint set is empty and toolchain-local libs (interp→libA→libB chains
/// through `/opt/toolchain/lib`) would otherwise fall off the BFS at
/// libA's resolution step.
fn resolve_shared_libs_with_extra_interp_hints(
    binary: &Path,
    extra_interp_hints: &[PathBuf],
) -> Result<SharedLibs> {
    resolve_shared_libs_inner(binary, extra_interp_hints)
}

/// Resolve a non-standard interpreter's *own* shared-library dependencies.
///
/// A standard system linker (glibc/musl `ld.so`) is statically linked with no
/// deps, so this returns an empty set for it (`is_standard_interpreter`). A
/// custom-toolchain linker can instead be dynamically linked with its own
/// `libA -> libB` chain alongside it; `build_initramfs_base` packs that chain
/// into the base and `BaseKey::hash_shared_libs` hashes it, so both go through
/// this one helper and the cache key covers exactly the set the base packs.
///
/// Seeds the BFS with the linker's parent and sibling lib dirs (the linker has
/// no PT_INTERP of its own, so the auto-derived hint set would be empty) so the
/// linker's toolchain-local deps resolve against the same dirs the parent
/// binary's resolution used.
pub(crate) fn resolve_interpreter_deps(interp: &str) -> Result<SharedLibs> {
    if is_standard_interpreter(interp) {
        return Ok(SharedLibs {
            found: vec![],
            missing: vec![],
            interpreter: None,
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

#[tracing::instrument(skip_all, fields(binary = %binary.display(), extra_hints = extra_interp_hints.len()))]
fn resolve_shared_libs_inner(binary: &Path, extra_interp_hints: &[PathBuf]) -> Result<SharedLibs> {
    // Cache results by canonical path AND extra-hint set — avoids
    // re-resolving the same binary across concurrent initramfs builds
    // (nextest parallelism). Hints are part of the key so a second
    // call on the same binary with different hint sets does not return
    // the prior result.
    type LibCache = LazyLock<std::sync::Mutex<HashMap<(PathBuf, Vec<PathBuf>), SharedLibs>>>;
    static CACHE: LibCache = LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

    let canon = std::fs::canonicalize(binary).unwrap_or_else(|_| binary.to_path_buf());
    let cache_key = (canon.clone(), extra_interp_hints.to_vec());
    if let Ok(cache) = CACHE.lock()
        && let Some(cached) = cache.get(&cache_key)
    {
        return Ok(cached.clone());
    }

    let data =
        std::fs::read(binary).with_context(|| format!("read binary: {}", binary.display()))?;
    let elf = match goblin::elf::Elf::parse(&data) {
        Ok(e) => e,
        Err(_) => {
            // Not a valid ELF (or 32-bit) — treat as static/non-dynamic.
            return Ok(SharedLibs {
                found: vec![],
                missing: vec![],
                interpreter: None,
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
        });
    }

    // Extract DT_NEEDED, DT_RUNPATH, and DT_RPATH from the root binary.
    let root_needed: Vec<String> = elf.libraries.iter().map(|s| s.to_string()).collect();
    let root_search = elf_search_paths(&elf, binary);

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

    // Resolve the full transitive closure via level-parallel BFS.
    // Each level's file reads (read + ELF parse) run in parallel via
    // rayon. Soname resolution (resolve_soname) is cheap (cache lookups
    // + stat calls), so it stays sequential per level.
    use rayon::prelude::*;

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
            if let Some(host_path) = resolve_soname(soname, search_paths, &interp_search_dirs) {
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

        // Phase 2: read + parse resolved libs in parallel to discover
        // their DT_NEEDED entries and search paths. The interp-relative
        // dirs apply uniformly to every resolve_soname call (via the
        // top-level `interp_search_dirs` slice), so transitive deps
        // don't need them threaded through per-level.
        let next_deps: Vec<(String, ElfSearchPaths)> = resolved
            .par_iter()
            .flat_map(|(_, _, canonical)| {
                let Ok(lib_data) = std::fs::read(canonical) else {
                    return Vec::new();
                };
                let Ok(lib_elf) = goblin::elf::Elf::parse(&lib_data) else {
                    return Vec::new();
                };
                let lib_search = elf_search_paths(&lib_elf, canonical);
                lib_elf
                    .libraries
                    .iter()
                    .map(|name| (name.to_string(), lib_search.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

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
    };

    if let Ok(mut cache) = CACHE.lock() {
        cache.insert(cache_key, result.clone());
    }

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

/// Directories from the `LD_LIBRARY_PATH` environment variable, parsed
/// once on first access. Empty when the variable is unset or empty.
static LD_LIBRARY_PATH_DIRS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
});

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
fn resolve_soname(
    soname: &str,
    elf_paths: &ElfSearchPaths,
    interp_hints: &[PathBuf],
) -> Option<PathBuf> {
    // 1. DT_RPATH (legacy). Non-empty only when DT_RUNPATH is absent
    //    per `elf_search_paths`; matches glibc's "DT_RPATH before
    //    LD_LIBRARY_PATH" rule for pre-RUNPATH binaries.
    for dir in &elf_paths.rpath {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 2. LD_LIBRARY_PATH.
    for dir in LD_LIBRARY_PATH_DIRS.iter() {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 3. DT_RUNPATH (modern).
    for dir in &elf_paths.runpath {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 4. Interp-relative hints for non-standard dynamic linkers.
    //    Not a glibc concept — ktstr uses these to keep custom
    //    toolchain libs resolvable without requiring the user to
    //    set LD_LIBRARY_PATH.
    for dir in interp_hints {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 5. ld.so.cache — the binary cache is the real dynamic linker's
    //    primary lookup mechanism. Catches libraries in directories
    //    added via `ldconfig /path` that don't appear in ld.so.conf.
    if let Some(cached_path) = LD_SO_CACHE.get(soname) {
        return Some(cached_path.clone());
    }

    // 6. Default paths. (ld.so.conf step dropped: ldconfig already
    //    ingests conf paths into ld.so.cache above, so a separate
    //    walk here was redundant per glibc's search algorithm.)
    for dir in DEFAULT_LIB_PATHS {
        let candidate = Path::new(dir).join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

/// ELF magic bytes: `\x7fELF`.
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// Check if the first 4 bytes of a file match ELF magic.
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

/// Build the base cpio archive: directories, `bin/busybox` (shell mode),
/// extra binaries, include files, and shared libraries. Does NOT include
/// `/init`, `/args`, the sentinel, trailer, or 512-byte padding — those
/// are the per-run suffix ([`build_suffix`]). The returned bytes are a
/// valid cpio prefix `build_suffix` completes. `/init` is deliberately
/// NOT here: keeping the payload's bytes out of the base means a payload
/// recompile that leaves the shared-lib set unchanged hits the base cache
/// (the base depends only on the lib SET, resolved from `payload`, not
/// its content).
///
/// When `busybox_bytes` is `Some`, embeds the provided bytes at
/// `bin/busybox` for shell mode. Bytes are sourced from
/// [`crate::vmm::blobs::load_busybox_bytes`] (which reads the
/// `KTSTR_BUSYBOX_PATH` env var that `cargo-ktstr` sets at startup);
/// the library does not embed busybox itself.
///
/// `include_files` adds files verbatim to the archive (no strip_debug).
/// Each entry is `(archive_path, host_path)`. ELF files get shared library
/// resolution; non-ELF files are copied as-is. Symlinks are followed to
/// their target; the target must be a regular file (FIFOs, device nodes,
/// and sockets are rejected). Archive paths must not contain `..`
/// components. Callers expand directories into individual file entries
/// before calling this function (see `cli::resolve_include_files`).
///
/// Extra binaries are `strip_debug`'d before being written. `payload` is
/// read only for shared-lib resolution (its `/init` bytes are written by
/// [`build_suffix`]).
#[tracing::instrument(skip_all, fields(payload = %payload.display(), includes = include_files.len()))]
pub fn build_initramfs_base(
    payload: &Path,
    extra_binaries: &[(&str, &Path)],
    include_files: &[(&str, &Path)],
    busybox_bytes: Option<&[u8]>,
) -> Result<Vec<u8>> {
    // Validate include_files and collect metadata (reused in the write
    // loop to avoid a second stat syscall per file).
    let mut validated_includes: Vec<(&str, &Path, u32)> = Vec::with_capacity(include_files.len());
    for (archive_path, host_path) in include_files {
        // Reject path traversal.
        if Path::new(archive_path)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            anyhow::bail!("include_files archive path contains '..': {}", archive_path);
        }
        // Reject paths that collide with internal sentinel files.
        if archive_path.starts_with(".ktstr_") {
            anyhow::bail!(
                "include_files archive path must not start with '.ktstr_': {}",
                archive_path
            );
        }
        // Follow symlinks: include_files entries are explicitly
        // specified by the test author, so a symlink to a regular
        // file is intentional (e.g. package-manager symlinks in
        // /usr/local/bin). The is_file() check below catches
        // non-regular targets (directories, FIFOs, devices).
        let meta = std::fs::metadata(host_path).with_context(|| {
            format!(
                "stat include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        // Reject non-regular files (FIFOs, device nodes, sockets block or
        // produce garbage).
        if !meta.file_type().is_file() {
            anyhow::bail!(
                "include_files entry '{}' is not a regular file: {}",
                archive_path,
                host_path.display()
            );
        }
        validated_includes.push((archive_path, host_path, meta.permissions().mode()));
    }

    let mut archive = Vec::new();

    // Collect directory entries needed for shared libraries and includes.
    let mut dirs = BTreeSet::new();

    // Resolve shared library dependencies for init binary and extras.
    let mut shared_libs: Vec<(String, PathBuf)> = Vec::new();
    let mut all_binaries: Vec<&Path> = std::iter::once(payload)
        .chain(extra_binaries.iter().map(|(_, p)| *p))
        .collect();

    // ELF files from include_files join the shared lib resolution chain.
    let mut include_elf_paths: Vec<&Path> = Vec::new();
    for (_, host_path) in include_files {
        if is_elf(host_path) {
            include_elf_paths.push(host_path);
            all_binaries.push(host_path);
        }
    }

    let _s_resolve = tracing::debug_span!("resolve_all_libs", count = all_binaries.len()).entered();
    for path in &all_binaries {
        let _s_one =
            tracing::debug_span!("resolve_shared_libs", binary = %path.display()).entered();
        let result = resolve_shared_libs(path)
            .with_context(|| format!("resolve libs for {}", path.display()))?;
        drop(_s_one);

        // Include-file ELFs must have all shared libs resolvable.
        if !result.missing.is_empty() && include_elf_paths.contains(path) {
            let names: Vec<&str> = result.missing.iter().map(|m| m.soname.as_str()).collect();
            anyhow::bail!(
                "{}: missing shared libraries: {}",
                path.display(),
                names.join(", ")
            );
        }

        // Pack PT_INTERP (dynamic linker) into the initramfs. The
        // interpreter is not a DT_NEEDED entry and won't appear in the
        // resolved shared libs, so it must be added explicitly.
        // For non-standard interpreters, also resolve their own deps.
        tracing::debug!(
            binary = %path.display(),
            interpreter = ?result.interpreter,
            is_include = include_elf_paths.contains(path),
            "resolved interpreter for binary"
        );
        if let Some(ref interp) = result.interpreter {
            let interp_path = Path::new(interp);
            let is_standard = is_standard_interpreter(interp);
            tracing::debug!(
                interp = %interp_path.display(),
                exists = interp_path.is_file(),
                is_standard,
                "interpreter details"
            );
            if interp_path.is_file() {
                let canonical = std::fs::canonicalize(interp_path)
                    .unwrap_or_else(|_| interp_path.to_path_buf());
                let canon_str = canonical.to_string_lossy();
                let guest = canon_str
                    .strip_prefix('/')
                    .unwrap_or(&canon_str)
                    .to_string();
                register_parent_dirs(&mut dirs, &guest);
                tracing::debug!(
                    canonical_guest = %guest,
                    canonical_host = %canonical.display(),
                    "packing interpreter canonical path"
                );
                shared_libs.push((guest.clone(), canonical.clone()));

                // Also add the non-canonical path if it differs.
                let orig_guest = interp.strip_prefix('/').unwrap_or(interp).to_string();
                if orig_guest != guest {
                    tracing::debug!(
                        orig_guest = %orig_guest,
                        canonical_guest = %guest,
                        "packing interpreter original (non-canonical) path"
                    );
                    register_parent_dirs(&mut dirs, &orig_guest);
                    shared_libs.push((orig_guest, canonical));
                } else {
                    tracing::debug!("interpreter original path matches canonical, no alias needed");
                }

                // A non-standard interpreter (custom toolchain linker) can
                // itself be dynamically linked with its own libA->libB chain
                // alongside it; resolve_interpreter_deps walks that chain
                // (seeding the linker's parent + sibling lib dirs, since the
                // linker has no PT_INTERP of its own) and returns empty for a
                // standard, statically-linked ld.so. The cache key hashes the
                // same set (BaseKey::hash_shared_libs) so an interp-dep change
                // invalidates the base.
                if let Ok(interp_result) = resolve_interpreter_deps(interp) {
                    for (g, h) in interp_result.found {
                        register_parent_dirs(&mut dirs, &g);
                        shared_libs.push((g, h));
                    }
                }
            }
        }

        for (guest_path, host_path) in result.found {
            register_parent_dirs(&mut dirs, &guest_path);
            shared_libs.push((guest_path, host_path));
        }
    }
    let pre_dedup_count = shared_libs.len();
    shared_libs.sort_by(|a, b| a.0.cmp(&b.0));
    shared_libs.dedup_by(|a, b| a.0 == b.0);
    tracing::debug!(
        pre_dedup = pre_dedup_count,
        post_dedup = shared_libs.len(),
        removed = pre_dedup_count - shared_libs.len(),
        "shared_libs dedup"
    );

    // Busybox goes under bin/. wprof, when set, rides
    // include_files (which registers its own parent dirs).
    if busybox_bytes.is_some() {
        dirs.insert("bin".to_string());
    }
    // Include files need their parent directories in the cpio archive.
    // The component walk produces all ancestors (e.g. "include-files/sub/f"
    // yields "include-files" and "include-files/sub").
    for (archive_path, _, _) in &validated_includes {
        register_parent_dirs(&mut dirs, archive_path);
    }
    // Extras with a path component (e.g. "bin/ktstr-jemalloc-probe")
    // need their parent directory registered too, otherwise the
    // kernel's cpio extractor sees the file before the directory
    // entry exists and silently drops it. The `("scheduler",
    // path)` case writes to archive root and has no parent to
    // register, so `register_parent_dirs` no-ops on it.
    for (name, _) in extra_binaries {
        register_parent_dirs(&mut dirs, name);
    }

    drop(_s_resolve);

    tracing::debug!(
        shared_libs_count = shared_libs.len(),
        dirs_count = dirs.len(),
        dirs = ?dirs,
        shared_libs_guests = ?shared_libs.iter().map(|(g, _)| g.as_str()).collect::<Vec<_>>(),
        "pre-write archive contents"
    );

    let _s_write = tracing::debug_span!("write_cpio").entered();
    // Directory entries
    for dir in &dirs {
        write_entry(&mut archive, dir, &[], 0o40755)?;
    }

    // Shell mode: embed busybox bytes provided by the caller. The
    // ktstr library does not own the bytes — they come from the
    // `KTSTR_BUSYBOX_PATH` env var that cargo-ktstr sets at startup
    // (see [`crate::vmm::blobs::load_busybox_bytes`]).
    if let Some(busybox_bytes) = busybox_bytes {
        write_entry(&mut archive, "bin/busybox", busybox_bytes, 0o100755)?;
    }

    // Extra binaries (stripped to reduce initramfs size)
    for (name, path) in extra_binaries {
        let data = strip_debug(path)
            .with_context(|| format!("strip/read extra binary '{}': {}", name, path.display()))?;
        write_entry(&mut archive, name, &data, 0o100755)?;
    }

    // Include files: copied verbatim, preserving original content and
    // debug symbols. No strip_debug — included files are user-provided
    // and may be non-ELF.
    for (archive_path, host_path, mode) in &validated_includes {
        let data = std::fs::read(host_path).with_context(|| {
            format!(
                "read include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        write_entry(&mut archive, archive_path, &data, *mode)?;
    }

    // Shared libraries — write each canonical host file once as a regular
    // file, then write subsequent guest paths that map to the same host
    // file as cpio symlinks. This avoids duplicating large libraries in
    // the initramfs (e.g. libc appearing under both lib64/ and usr/lib64/).
    {
        // canonical host path -> first guest_path written for this file
        let mut written_files: HashMap<PathBuf, String> = HashMap::new();
        for (guest_path, host_path) in &shared_libs {
            let canonical = std::fs::canonicalize(host_path).unwrap_or_else(|_| host_path.clone());
            if let Some(first_guest) = written_files.get(&canonical) {
                // Already written — emit a symlink to the first guest path.
                let target = format!("/{first_guest}");
                write_symlink_entry(&mut archive, guest_path, &target)?;
            } else {
                let data = std::fs::read(host_path).with_context(|| {
                    format!("read shared lib '{}': {}", guest_path, host_path.display())
                })?;
                write_entry(&mut archive, guest_path, &data, 0o100755)?;
                written_files.insert(canonical, guest_path.clone());
            }
        }
    }

    drop(_s_write);

    Ok(archive)
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
    /// archive by [`build_initramfs_base`] (its
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
}

/// Build the suffix that completes a base archive: `/init` (the
/// `strip_debug`'d test binary), `/args` and `/sched_args` entries,
/// optional `/sched_enable` and `/sched_disable` shell scripts for
/// kernel-built schedulers, optional `/exec_cmd`, the `.ktstr_init_ok`
/// sentinel (last entry — its absence flags incomplete extraction),
/// trailer, and 512-byte
/// padding. `base_len` is needed to compute the padding. `/init` lives
/// here, not the base, so a payload recompile is a base-cache hit; it
/// dominates the suffix size (the other entries are ~200 bytes total).
pub fn build_suffix(base_len: usize, params: &SuffixParams<'_>) -> Result<Vec<u8>> {
    let mut suffix = Vec::new();

    // Test binary as /init — detects PID 1 and runs all setup before the
    // test (see `vmm::rust_init`). It lives in the per-run suffix, not the
    // cached base, so the base archive is independent of payload
    // recompiles; `strip_debug` here is the per-run cost. `None` only in
    // suffix-shape unit tests.
    if let Some(payload) = params.payload {
        let binary = strip_debug(payload)
            .with_context(|| format!("strip/read init binary: {}", payload.display()))?;
        write_entry(&mut suffix, "init", &binary, 0o100755)?;
    }

    // Args file
    let args_data = params.args.join("\n");
    write_entry(&mut suffix, "args", args_data.as_bytes(), 0o100644)?;

    // Scheduler args file
    if !params.sched_args.is_empty() {
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
    // packed into the base archive by `build_initramfs_base` as an
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
    let total = base_len + suffix.len();
    let pad = (512 - (total % 512)) % 512;
    suffix.extend(std::iter::repeat_n(0u8, pad));

    Ok(suffix)
}

// ---------------------------------------------------------------------------
// POSIX shared-memory cache for base initramfs
// ---------------------------------------------------------------------------

/// Target arch tag embedded in shm segment names so caches built on
/// a host running a foreign-arch ktstr binary (cross-arch developer
/// boxes that share `/dev/shm`) cannot collide with ours. Selected
/// at compile time so a single binary always emits one tag.
#[cfg(target_arch = "x86_64")]
pub(crate) const SHM_ARCH_TAG: &str = "x86_64";
#[cfg(target_arch = "aarch64")]
pub(crate) const SHM_ARCH_TAG: &str = "aarch64";

/// Derive an shm segment name from a content hash. Each distinct
/// combination of payload + scheduler binaries gets its own segment.
/// The target arch is included so an x86_64 binary cannot mmap a
/// segment written by an aarch64 binary on the same host.
pub(crate) fn shm_segment_name(content_hash: u64) -> String {
    format!("/ktstr-base-{SHM_ARCH_TAG}-{content_hash:016x}")
}

/// Read-only mmap of a POSIX shared-memory segment. The mapping stays
/// live until the struct is dropped, so callers can borrow the bytes
/// without copying the entire base archive.
///
/// Holds the shared flock (`LOCK_SH`) for the lifetime of the mapping
/// so that a concurrent writer cannot `ftruncate` the segment beneath
/// us, which would cause `SIGBUS` on access to the truncated pages.
///
/// # Drop under `panic = "abort"`
///
/// Cargo.toml release profile sets `panic = "abort"`, so a panic in
/// the release binary aborts the process without unwinding — Drop
/// impls are **not run**. For `MappedShm` this means `munmap` and
/// the `flock` unlock are skipped on the abort path. Neither is a
/// resource leak in practice:
///
/// - **mmap**: the kernel reclaims all mappings on process exit.
/// - **flock**: the lock is released when the fd is closed, which
///   the kernel does on process exit.
///
/// The **SHM segment itself**, however, persists in `/dev/shm` until
/// something calls `shm_unlink` on it. An aborted process never
/// reaches its normal unlink sites, and `libc::atexit` handlers do
/// not run on `abort()` either — so the segment leaks *from this
/// run's perspective*. Cleanup is deferred: the next ktstr run
/// sweeps orphans via `crate::vmm::cleanup_stale_shm`, which
/// non-blockingly `LOCK_EX`'s each stale entry and unlinks it when
/// no other process holds a lock.
///
/// The resulting accumulation is bounded in the common case:
/// repeated aborts during iterative local development trigger the
/// next-run sweep, and /dev/shm is typically sized at ~50% of RAM
/// (tmpfs default) which can hold many orphans before disk pressure.
/// The pathological case is **one-shot CI jobs** that panic-abort
/// and are never re-run: the segment remains until host reboot or
/// manual `rm /dev/shm/ktstr-*` cleanup. An `atexit`-based cleanup
/// would not help (`abort()` bypasses atexit); a signal handler
/// that runs before the abort could, but would have to be
/// async-signal-safe and cannot reliably walk the tracked-segment
/// set under arbitrary signal arrival. Accepting the bounded leak
/// is the design tradeoff from the panic-strategy decision
/// (panic=abort with audited threads).
pub(crate) struct MappedShm {
    ptr: *const u8,
    len: usize,
    fd: std::os::fd::OwnedFd,
}

// SAFETY: The mmap is MAP_SHARED|PROT_READ over a shm segment whose
// contents are held stable for the mapping's lifetime by a shared
// flock retained in `fd`. The pointer and length are valid for the
// lifetime of the mapping.
unsafe impl Send for MappedShm {}
unsafe impl Sync for MappedShm {}

impl AsRef<[u8]> for MappedShm {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: ptr/len are set by a successful mmap in
        // shm_load_base, and the SHM segment's contents are held
        // stable for the mapping's lifetime by the shared flock
        // retained in self.fd — a cooperating writer cannot
        // ftruncate the segment out from under us.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MappedShm {
    fn drop(&mut self) {
        // Note: under `panic = "abort"` this Drop is skipped entirely.
        // See `MappedShm`'s type doc for the /dev/shm accumulation
        // discussion and the `cleanup_stale_shm` recovery path.
        //
        // SAFETY: ptr/len are from mmap; fd is an OwnedFd opened via
        // rustix::shm::open with LOCK_SH held. Release the mapping
        // first so the lock still protects it, then drop the flock
        // explicitly. The OwnedFd's own drop closes the descriptor
        // after this function returns.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
        let _ = rustix::fs::flock(&self.fd, rustix::fs::FlockOperation::Unlock);
    }
}

/// Try to mmap a base initramfs from a POSIX shared-memory segment
/// identified by `content_hash`. Returns a `MappedShm` that borrows
/// the data without copying. Returns `None` on miss or error.
///
/// Acquires a shared flock (`LOCK_SH`) before mmap and keeps it held
/// for the lifetime of the returned `MappedShm`. A concurrent writer
/// calls `ftruncate` under `LOCK_EX` in `shm_store` (and in
/// `shm_write_and_release`, which also `ftruncate`s to 0 on mmap
/// failure); holding `LOCK_SH` for the mapping's lifetime prevents
/// either writer from truncating the segment out from under us, which
/// would turn any access to the truncated pages into `SIGBUS`.
///
/// Note: `flock` is advisory — it only protects against other
/// processes that also call `flock`. A process that writes the
/// segment without taking `LOCK_EX` (e.g. `rm /dev/shm/…` + recreate
/// by an unrelated tool) bypasses this scheme. All callers within
/// this crate cooperate, which is the closed-world guarantee we
/// rely on.
pub(crate) fn shm_load_base(content_hash: u64) -> Option<MappedShm> {
    use std::os::fd::AsRawFd;

    let name = shm_segment_name(content_hash);
    let fd = rustix::shm::open(
        name.as_str(),
        rustix::shm::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    )
    .ok()?;

    // Shared lock — blocks until any concurrent writer releases
    // LOCK_EX. Held for the mapping's lifetime; released in
    // MappedShm::drop. The fd is dropped on the early-return paths,
    // which implicitly closes the descriptor.
    rustix::fs::flock(&fd, rustix::fs::FlockOperation::LockShared).ok()?;

    let stat = rustix::fs::fstat(&fd).ok()?;
    if stat.st_size <= 0 {
        let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
        return None;
    }
    let len = stat.st_size as usize;

    // SAFETY: mmap consumes the raw fd value but does not take
    // ownership; the OwnedFd lives on in `fd` and ultimately closes
    // in `MappedShm::drop` after munmap.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
        return None;
    }

    Some(MappedShm {
        ptr: ptr as *const u8,
        len,
        fd,
    })
}

/// Write `data` to a POSIX SHM segment identified by `name`.
///
/// Creates (or opens existing) the segment with `O_CREAT | O_RDWR`,
/// takes an exclusive flock, `ftruncate`s to `data.len()`, `mmap`s
/// `PROT_WRITE | MAP_SHARED`, copies, and cleans up.
///
/// Concurrency: the write takes a NON-BLOCKING `LOCK_EX`
/// (`NonBlockingLockExclusive`); on contention it SKIPS rather than
/// blocking. This preserves the `SIGBUS`-prevention invariant (never
/// `ftruncate` a segment a reader has COW-mapped) by construction:
/// `LOCK_EX` is granted only when no reader holds `LOCK_SH`, so a
/// successful acquire guarantees no live mapping, and on contention we
/// skip (never truncate). The reader-side flock lifetime lives in
/// `shm_load_base` (via `MappedShm`), `shm_open_lz4`, and
/// `cow_overlay`.
///
/// A blocking `LOCK_EX` here starved indefinitely: Linux flock grants
/// no writer preference, so under sustained host concurrency fresh
/// `LOCK_SH` readers (each held for a VM's lifetime) perpetually jump
/// ahead of the waiting writer, wedging the suite. Skipping is correct
/// because writes are content-addressed (below): a held segment has
/// the same hash, hence the same bytes, so the skipped rewrite was
/// redundant; `try_cow_overlay` re-opens by hash, so a peer's write is
/// still COW-shared.
///
/// Writes are content-addressed at the caller: callers hash `data`
/// to form the segment name. When two callers write the same content
/// to the same hash, the payload length and bytes are identical, so
/// the second `ftruncate(same_len)` is a no-op on page contents and
/// the second memcpy writes the same bytes. A third-party caller
/// that writes DIFFERENT data to an already-used hash (e.g. the
/// rename-test pattern) will overwrite — the store does not enforce
/// idempotence itself.
fn shm_store(name: &str, data: &[u8]) -> Result<()> {
    use std::os::fd::AsRawFd;

    let fd = rustix::shm::open(
        name,
        rustix::shm::OFlags::CREATE | rustix::shm::OFlags::RDWR,
        rustix::fs::Mode::from_raw_mode(0o644),
    )
    .map_err(|e| anyhow::anyhow!("shm_open: {e}"))?;

    // Post-flock error paths (ftruncate/mmap failure) call shm_unlink
    // before the OwnedFd drop so a half-initialized segment (empty /
    // corrupt / wrong size) does not persist in /dev/shm where the
    // next caller would read it. Those paths hold LOCK_EX, so the
    // segment is either newly created by this call or being
    // exclusively rewritten — destroying it is safe. Unlinking may
    // also remove a pre-existing valid segment that the caller had
    // just opened then failed to rewrite; content-addressing makes
    // recovery cheap — the next writer re-creates with the same hash.
    //
    // Flock failure (shm_open succeeded, LOCK_EX not acquired) does
    // NOT unlink: the segment may be a pre-existing valid one a peer
    // holds (a LOCK_SH reader via a live COW overlay, or a LOCK_EX
    // writer), and we must not destroy another process's cache entry
    // on our own contention (EWOULDBLOCK) or flock error.
    //
    // Success path does NOT unlink — the segment IS the cache. The
    // OwnedFd's drop handles close() on every path.
    match rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {}
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => {
            // A reader holds LOCK_SH (live COW overlay / MappedShm) or
            // a peer holds LOCK_EX. The segment is in use or being
            // written by the peer; skip the write. Content-addressing
            // makes our rewrite redundant, and try_cow_overlay re-opens
            // by hash so the peer's segment is still COW-shared. Do NOT
            // unlink: the segment may be a valid one a peer is reading.
            tracing::debug!(
                segment = name,
                data_len = data.len(),
                "shm_store: LOCK_EX contended, skipping cache write"
            );
            return Ok(());
        }
        Err(e) => return Err(anyhow::anyhow!("flock: {e}")),
    }

    let raw_fd = fd.as_raw_fd();
    unsafe {
        if libc::ftruncate(raw_fd, data.len() as libc::off_t) != 0 {
            let err = std::io::Error::last_os_error();
            if let Err(e) = rustix::shm::unlink(name) {
                tracing::warn!(
                    err = %e,
                    segment = name,
                    "shm_unlink failed on ftruncate error path"
                );
            }
            let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
            anyhow::bail!("ftruncate: {err}");
        }

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            raw_fd,
            0,
        );
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            if let Err(e) = rustix::shm::unlink(name) {
                tracing::warn!(
                    err = %e,
                    segment = name,
                    "shm_unlink failed on mmap error path"
                );
            }
            let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
            anyhow::bail!("mmap: {err}");
        }

        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        libc::munmap(ptr, data.len());
    }
    let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
    // fd drops here → close(fd).
    Ok(())
}

pub(crate) fn shm_store_base(content_hash: u64, data: &[u8]) -> Result<()> {
    shm_store(&shm_segment_name(content_hash), data)
}

/// Test helper — remove the POSIX shared-memory segment identified
/// by `content_hash`.
#[cfg(test)]
pub(crate) fn shm_unlink_base(content_hash: u64) {
    let _ = rustix::shm::unlink(shm_segment_name(content_hash).as_str());
}

// ---------------------------------------------------------------------------
// Compressed SHM cache — stores LZ4-compressed base for COW overlay into
// guest RAM
// ---------------------------------------------------------------------------

/// Segment name for the LZ4-compressed version of a base initramfs.
/// Uses `lz4` prefix to avoid collisions with segments written by
/// previous compression formats (zstd, gzip). The target arch tag
/// keeps cross-arch caches separate on hosts where both ktstr
/// binaries share `/dev/shm`.
pub(crate) fn shm_lz4_segment_name(content_hash: u64) -> String {
    format!("/ktstr-lz4-{SHM_ARCH_TAG}-{content_hash:016x}")
}

/// Open the compressed SHM segment and return a held OwnedFd + size.
/// The fd has a shared flock held; drop the OwnedFd (via
/// [`shm_close_fd`] or scope exit) to release the lock and close.
/// Returns `None` on miss or error.
pub(crate) fn shm_open_lz4(content_hash: u64) -> Option<(std::os::fd::OwnedFd, usize)> {
    let name = shm_lz4_segment_name(content_hash);
    let fd = rustix::shm::open(
        name.as_str(),
        rustix::shm::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    )
    .ok()?;
    rustix::fs::flock(&fd, rustix::fs::FlockOperation::LockShared).ok()?;
    let stat = rustix::fs::fstat(&fd).ok()?;
    if stat.st_size <= 0 {
        let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
        return None;
    }
    Some((fd, stat.st_size as usize))
}

/// Store compressed initramfs data into an LZ4 SHM segment.
pub(crate) fn shm_store_lz4(content_hash: u64, data: &[u8]) -> Result<()> {
    shm_store(&shm_lz4_segment_name(content_hash), data)
}

/// Load + copy the LZ4-compressed base from its SHM segment into an
/// owned `Vec`, validating the LZ4 legacy magic. Returns `None` on
/// miss, mmap failure, or a stale/wrong-format segment (rejecting
/// data written by a previous compression format, e.g. zstd/gzip, or
/// a truncated write). The shared flock taken by [`shm_open_lz4`] is
/// released before returning; the bytes are copied out, so the
/// segment need not stay pinned. This is the owned-`Vec` analog of
/// [`shm_load_base`] (which returns a borrowed `MappedShm`); the LZ4
/// GET path hands the bytes back to the caller and does not retain
/// the fd.
pub(crate) fn shm_load_lz4(content_hash: u64) -> Option<Vec<u8>> {
    use std::os::fd::AsRawFd;
    let (fd, len) = shm_open_lz4(content_hash)?;
    let mut buf = vec![0u8; len];
    // SAFETY: `fd` is a live RDONLY shm fd from shm_open_lz4 holding a
    // shared flock; the segment is at least `len` bytes (its fstat).
    // The mmap is PROT_READ/MAP_SHARED over exactly `len` bytes, fully
    // copied out before munmap; no pointer escapes the block.
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        );
        if ptr == libc::MAP_FAILED {
            shm_close_fd(fd);
            return None;
        }
        std::ptr::copy_nonoverlapping(ptr as *const u8, buf.as_mut_ptr(), len);
        libc::munmap(ptr, len);
    }
    shm_close_fd(fd);

    if buf.len() >= 4 && buf[..4] == LZ4_LEGACY_MAGIC {
        tracing::debug!(bytes = len, "lz4_base cache hit (shm)");
        Some(buf)
    } else {
        let head = &buf[..buf.len().min(8)];
        tracing::warn!(
            bytes = len,
            ?head,
            "stale/truncated compressed shm segment (bad LZ4 magic), recompressing"
        );
        None
    }
}

/// RAII guard for a live COW-overlay mapping.
///
/// A COW overlay is `MAP_PRIVATE | MAP_FIXED` onto guest memory from
/// a SHM segment fd. `MAP_PRIVATE` pages are lazily read from the
/// backing file on first access; if the SHM segment is truncated or
/// unlinked-with-retruncate between `mmap` and the guest's first
/// read, the access SIGBUSes (see Linux `filemap_fault` against
/// `i_size`). Holding the fd with `LOCK_SH` for the mapping's
/// lifetime blocks any cooperating writer from taking `LOCK_EX` and
/// `ftruncate`ing the segment until after the mapping is torn down.
///
/// Drop order: the guard releases `LOCK_UN` and `close` only. The
/// MAP_FIXED region itself is owned by the caller's VA reservation
/// (e.g. `ReservationGuard` in the VMM) and is munmapped when that
/// reservation drops — which must happen BEFORE this guard drops,
/// so the lock protects the mapping right up until tear-down.
pub(crate) struct CowOverlayGuard {
    fd: std::os::fd::OwnedFd,
}

impl CowOverlayGuard {
    fn new(fd: std::os::fd::OwnedFd) -> Self {
        Self { fd }
    }
}

impl Drop for CowOverlayGuard {
    fn drop(&mut self) {
        // fd was obtained via shm_open in the COW overlay path; release
        // LOCK_SH explicitly so cooperating writers waiting on LOCK_EX
        // observe ordering with the VM's reads. The OwnedFd's own drop
        // closes the descriptor after this function returns.
        let _ = rustix::fs::flock(&self.fd, rustix::fs::FlockOperation::Unlock);
    }
}

/// COW-overlay `len` bytes from `shm_fd` at `host_addr` using
/// `MAP_PRIVATE | MAP_FIXED | MAP_POPULATE`. The guest sees the SHM
/// content but writes go to private anonymous pages (copy-on-write).
/// `MAP_POPULATE` pre-faults the pages so the initial accesses skip
/// the filemap fault path; the lock guard still protects against
/// truncate of pages that may be refaulted from the page cache
/// (MAP_POPULATE alone is not sufficient — truncate invalidates the
/// page cache via `unmap_mapping_range`).
///
/// On success, returns `Some(CowOverlayGuard)` — the guard owns
/// `shm_fd` and holds `LOCK_SH` for the mapping's lifetime. The
/// caller MUST keep the guard alive for as long as the MAP_FIXED
/// mapping is in use (typically the VM lifetime) and drop the guard
/// AFTER the VA region is munmapped.
///
/// On failure, returns `None` and CLOSES `shm_fd` (releasing
/// `LOCK_SH`) so the caller does not need to clean it up.
///
/// # Safety
///
/// The caller MUST have validated that the entire range
/// `[host_addr, host_addr + len)` lies within one contiguous guest
/// memory region. `MAP_FIXED` unmaps whatever is already present
/// across the full range and replaces it with the new mapping; if
/// `len` extends past the region, `MAP_FIXED` silently corrupts
/// unrelated host mappings (kernel-defined behaviour at the mmap
/// layer) and violates Rust's aliasing invariants at the process
/// level. The caller is also responsible for ensuring `shm_fd` is a
/// valid, open file descriptor with `LOCK_SH` already held (the
/// guard inherits both).
pub(crate) unsafe fn cow_overlay(
    host_addr: *mut u8,
    len: usize,
    shm_fd: std::os::fd::OwnedFd,
) -> Option<CowOverlayGuard> {
    use std::os::fd::AsRawFd;

    // SAFETY: caller guarantees [host_addr, host_addr + len) is
    // entirely within a single valid guest memory region and shm_fd
    // is a valid fd holding LOCK_SH. See function-level docs.
    let ptr = unsafe {
        libc::mmap(
            host_addr as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_FIXED | libc::MAP_POPULATE,
            shm_fd.as_raw_fd(),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        // Drop the OwnedFd on the failure path so the caller expects
        // no cleanup responsibility on the None branch; drop releases
        // the LOCK_SH and closes the descriptor.
        let _ = rustix::fs::flock(&shm_fd, rustix::fs::FlockOperation::Unlock);
        return None;
    }
    Some(CowOverlayGuard::new(shm_fd))
}

/// Close a SHM fd and release its shared flock.
pub(crate) fn shm_close_fd(fd: std::os::fd::OwnedFd) {
    // Explicit flock-unlock so a cooperating writer waiting on LOCK_EX
    // observes ordering with our earlier reads (LOCK_SH); the OwnedFd
    // drop at scope exit then closes the descriptor.
    let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
}

/// Write one or more byte slices sequentially into guest memory as a
/// single contiguous initramfs. Passing a one-element `parts` slice
/// matches the "single blob" caller; multi-element slices avoid the
/// copy into a monolithic `Vec` that the split base/suffix production
/// path would otherwise need. Returns (address, total_size) for
/// boot_params.
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
/// Input is split into `LZ4_CHUNK_SIZE` (8MB) chunks, compressed in
/// parallel with rayon, then assembled sequentially.
pub(crate) fn lz4_legacy_compress(data: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;

    // Compress all chunks in parallel.
    let compressed_chunks: Vec<Vec<u8>> = data
        .par_chunks(LZ4_CHUNK_SIZE)
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
