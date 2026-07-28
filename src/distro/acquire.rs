//! Orchestration for prebuilt distro / local-package kernels.
//!
//! Ties the three foundation pieces together into a cache entry the
//! rest of ktstr boots like any built kernel:
//!
//! 1. [`repo::resolve_distro_kernel`](crate::distro::repo) — spec →
//!    concrete package URLs + sha256 (distro path only).
//! 2. [`crate::fetch::download_verified_file`] — stream each package
//!    (and its debuginfo) with the shared watchdog + progress + retry.
//! 3. [`extract`](crate::distro::extract) — unpack the packages and
//!    locate image / modules / config / vmlinux.
//!
//! then a config gate (boot-critical options must be `=y`/`=m`, other
//! ktstr-feature options warn), a boot-module selection (each modular
//! virtio driver is `module_closure`'d and embedded in the entry), and
//! an atomic install into the kernel cache.
//!
//! ## Config gate & boot modules — why these options
//!
//! The ktstr guest boots off the initramfs (no root disk — see
//! `src/vmm/setup/mod.rs` `rdinit=/init`), so the only devices EVERY
//! boot touches are the virtio-console control ports
//! (the named `ktstr-bulk` sys_rdy channel and `/dev/hvc0`), presented over
//! the virtio-MMIO transport on both arches
//! (`src/vmm/setup/mod.rs` console `virtio_mmio.device=`,
//! `src/vmm/aarch64/fdt.rs` `virtio,mmio` nodes) and loaded before any
//! virtio access by guest init
//! (`src/vmm/rust_init/init.rs` "load modules BEFORE any virtio device
//! is touched"). So `BOOT_CRITICAL` — virtio core, the MMIO
//! transport, and the console driver — is what a kernel cannot boot in
//! ktstr without, in ANY form; absent (neither `=y` nor `=m`) is a
//! hard error. `FEATURE_MODULES` (blk / net / pci / btrfs) back
//! optional devices attached only by tests that request a disk or NIC
//! (`self.disk` / `self.networks` gating in `setup/mod.rs`), so a
//! missing one only warns. `virtio_pci` is x86-only (aarch64 has no
//! PCI bus).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
use crate::distro::extract::{
    ExtractedKernel, ensure_arch_matches_host, extract_kernel_packages, extract_vmlinux,
    module_closure, package_arch,
};
use crate::distro::repo::{DistroKind, PackageRef, ResolvedDistroKernel, resolve_distro_kernel};

/// Kernel config options whose driver the ktstr VMM needs present in
/// SOME form for EVERY guest boot, mapped to the module ktstr must load
/// when the option is `=m`. See the module docs for the code-verified
/// justification (virtio-console control port over the MMIO transport).
const BOOT_CRITICAL: &[(&str, &str)] = &[
    ("CONFIG_VIRTIO", "virtio"),
    ("CONFIG_VIRTIO_MMIO", "virtio_mmio"),
    ("CONFIG_VIRTIO_CONSOLE", "virtio_console"),
];

/// Optional-device options bundled as boot modules when `=m` (so the
/// corresponding ktstr feature works) but not required to boot — the
/// device is only presented when a test attaches a disk / NIC / btrfs
/// disk. `virtio_pci` is x86-only (aarch64 routes virtio over MMIO with
/// no PCI bus — `src/vmm/setup/mod.rs`).
const FEATURE_MODULES: &[(&str, &str)] = &[
    #[cfg(target_arch = "x86_64")]
    ("CONFIG_VIRTIO_PCI", "virtio_pci"),
    ("CONFIG_VIRTIO_BLK", "virtio_blk"),
    ("CONFIG_VIRTIO_NET", "virtio_net"),
    ("CONFIG_BTRFS_FS", "btrfs"),
];

/// ktstr monitor/debug niceties that are not on the build path's
/// critical list but degrade functionality when absent from a prebuilt
/// kernel. Warn-only, with the concrete effect the operator loses.
const WARN_EXTRA: &[(&str, &str)] = &[
    (
        "CONFIG_KALLSYMS_ALL",
        "monitor kallsyms symbol resolution will be less complete",
    ),
    (
        "CONFIG_IKCONFIG_PROC",
        "/proc/config.gz unavailable for in-guest config inspection",
    ),
    (
        "CONFIG_IKHEADERS",
        "/sys/kernel/kheaders.tar.xz absent — bpftrace/BCC in shell mode \
         cannot fall back to in-kernel headers",
    ),
    (
        "CONFIG_FTRACE_SYSCALLS",
        "syscall tracepoints (tracepoint:syscalls:*) unavailable",
    ),
];

/// State of a kconfig symbol in an extracted `.config`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OptState {
    /// `CONFIG_X=y` (or any non-`m` value assignment) — compiled in.
    Builtin,
    /// `CONFIG_X=m` — loadable module.
    Module,
    /// `# CONFIG_X is not set`, `=n`, or absent entirely.
    Off,
}

/// Parsed states of the symbols in a kernel `.config`, for the gate.
struct ConfigStates(HashMap<String, OptState>);

impl ConfigStates {
    fn parse(text: &str) -> Self {
        let mut map = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("# ")
                && let Some(sym) = rest.strip_suffix(" is not set")
                && sym.starts_with("CONFIG_")
            {
                map.insert(sym.to_string(), OptState::Off);
            } else if line.starts_with("CONFIG_")
                && let Some((sym, val)) = line.split_once('=')
            {
                let state = match val {
                    "m" => OptState::Module,
                    "n" => OptState::Off,
                    _ => OptState::Builtin,
                };
                map.insert(sym.to_string(), state);
            }
        }
        ConfigStates(map)
    }

    fn state(&self, opt: &str) -> OptState {
        self.0.get(opt).copied().unwrap_or(OptState::Off)
    }

    fn is_enabled(&self, opt: &str) -> bool {
        matches!(self.state(opt), OptState::Builtin | OptState::Module)
    }
}

/// The module NAMES a gate run selected, split by criticality.
#[derive(Debug)]
struct GateResult {
    /// `=m` boot-critical drivers — every one MUST resolve to a `.ko`
    /// or acquisition fails (the kernel cannot boot in ktstr).
    critical: Vec<&'static str>,
    /// `=m` feature drivers — best-effort: a module the kernel declares
    /// `=m` but does not actually ship in the resolved packages only
    /// disables that feature, not the boot.
    feature: Vec<&'static str>,
}

/// Apply the config gate to an extracted kernel and return the boot-
/// critical and feature module NAMES to load. Boot-critical options
/// absent in any form (neither `=y` nor `=m`) are a hard error; every
/// other classified option warns and proceeds.
///
/// A package that shipped no `.config` (SteamOS's linux-neptune
/// carries neither a config file nor an embedded IKCONFIG — verified
/// against a live jupiter-3.8 package) but did ship a module tree is
/// gated on the TREE instead: every boot-critical and feature
/// driver is listed for [`select_boot_modules`], whose
/// `module_closure` run skips drivers built in per `modules.builtin`,
/// resolves shipped `.ko`s, hard-errors on a boot-critical driver
/// found in neither, and warns per missing feature driver — the same
/// outcomes the config gate produces, derived from what the package
/// actually carries. With neither config nor module tree, the gate is
/// skipped with a warning and the kernel is trusted to have virtio
/// built in.
fn gate_config(extracted: &ExtractedKernel) -> Result<GateResult> {
    let kernel_label = &extracted.kernel_release;
    let Some(config_path) = extracted.config.as_deref() else {
        if extracted.modules_dir.is_some() {
            tracing::warn!(
                kernel = %kernel_label,
                "prebuilt kernel shipped no .config — gating boot-critical \
                 virtio drivers on the module tree (modules.builtin + \
                 modules.dep) instead; config-derived feature warnings are \
                 unavailable for this kernel"
            );
            return Ok(GateResult {
                critical: BOOT_CRITICAL.iter().map(|(_, m)| *m).collect(),
                feature: FEATURE_MODULES.iter().map(|(_, m)| *m).collect(),
            });
        }
        tracing::warn!(
            kernel = %kernel_label,
            "prebuilt kernel shipped no .config — skipping config gate. If the \
             guest hangs at boot, the kernel likely lacks CONFIG_VIRTIO_MMIO / \
             CONFIG_VIRTIO_CONSOLE (which ktstr needs for its console/control port)."
        );
        return Ok(GateResult {
            critical: Vec::new(),
            feature: Vec::new(),
        });
    };
    let text = fs::read_to_string(config_path)
        .with_context(|| format!("read extracted config {}", config_path.display()))?;
    let states = ConfigStates::parse(&text);

    let mut critical: Vec<&'static str> = Vec::new();
    let mut missing_critical: Vec<&'static str> = Vec::new();
    for (opt, module) in BOOT_CRITICAL {
        match states.state(opt) {
            OptState::Builtin => {}
            OptState::Module => critical.push(module),
            OptState::Off => missing_critical.push(opt),
        }
    }
    if !missing_critical.is_empty() {
        bail!(
            "prebuilt kernel {kernel_label} cannot boot under ktstr: \
             boot-critical option(s) {missing_critical:?} are neither =y nor =m. \
             ktstr presents its console + host control port over the virtio-MMIO \
             transport, so without these drivers the guest hangs before reaching \
             userspace. This kernel is unsupported."
        );
    }

    let mut feature: Vec<&'static str> = Vec::new();
    for (opt, module) in FEATURE_MODULES {
        match states.state(opt) {
            OptState::Module => feature.push(module),
            OptState::Builtin => {}
            OptState::Off => tracing::warn!(
                kernel = kernel_label,
                option = opt,
                "not enabled — the corresponding ktstr feature (disk / NIC / btrfs \
                 disk) will not work with this kernel; plain boots are unaffected"
            ),
        }
    }

    // Warn on every other option the build path treats as critical
    // (SCHED_CLASS_EXT, DEBUG_INFO_BTF, BPF, FTRACE, tracepoints, ACPI,
    // PCI, ...) that this prebuilt kernel lacks — reusing that list so
    // the two never drift. Options already classified above are skipped.
    let handled: HashSet<&str> = BOOT_CRITICAL
        .iter()
        .chain(FEATURE_MODULES.iter())
        .map(|(opt, _)| *opt)
        .collect();
    for (opt, hint) in crate::cli::critical_config_options() {
        if handled.contains(opt) || states.is_enabled(opt) {
            continue;
        }
        tracing::warn!(
            kernel = kernel_label,
            option = opt,
            "not enabled in this prebuilt kernel — {hint}"
        );
    }
    for (opt, effect) in WARN_EXTRA {
        if !states.is_enabled(opt) {
            tracing::warn!(
                kernel = kernel_label,
                option = opt,
                "not enabled in this prebuilt kernel — {effect}"
            );
        }
    }

    Ok(GateResult { critical, feature })
}

/// Resolve the gate's module NAMES into the ordered, dependency-first,
/// already-decompressed `.ko` paths to embed. Boot-critical modules
/// must all resolve; a feature module the kernel declares `=m` but does
/// not actually ship in the extracted packages is warned and skipped
/// rather than failing the whole acquisition.
fn select_boot_modules(extracted: &ExtractedKernel, gate: &GateResult) -> Result<Vec<PathBuf>> {
    if gate.critical.is_empty() && gate.feature.is_empty() {
        return Ok(Vec::new());
    }
    let modules_dir = extracted.modules_dir.as_ref().ok_or_else(|| {
        anyhow!(
            "kernel needs loadable modules but the package(s) carried no module \
             tree — cannot assemble a bootable module set"
        )
    })?;
    let release = &extracted.kernel_release;

    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    // Boot-critical: every module must resolve, or this kernel cannot
    // reach the ktstr console/control port and is unsupported.
    let crit = module_closure(modules_dir, release, &gate.critical).with_context(|| {
        format!(
            "prebuilt kernel {release} neither builds in nor ships a \
             boot-critical virtio module among {:?} — ktstr cannot reach its \
             console/host control port without it, so this kernel is unsupported \
             (some distros split virtio-console into a subpackage the resolver \
             must include)",
            gate.critical
        )
    })?;
    for p in crit {
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }

    // Feature drivers: resolve each independently so a missing one only
    // disables its feature (disk / NIC / btrfs) instead of blocking boot.
    for module in &gate.feature {
        match module_closure(modules_dir, release, std::slice::from_ref(module)) {
            Ok(mods) => {
                for p in mods {
                    if seen.insert(p.clone()) {
                        out.push(p);
                    }
                }
            }
            Err(e) => tracing::warn!(
                module = module,
                "declared =m but not shipped by this kernel ({e:#}); the \
                 corresponding disk/NIC/btrfs feature will not work"
            ),
        }
    }

    Ok(out)
}

/// Atomically install the extracted kernel + already-selected boot
/// modules into the cache under `cache_key`. Returns the cache entry
/// directory. Shared tail of the distro and local-package paths; the
/// config gate and boot-module selection run in the caller (before this)
/// so the distro path can gate a kernel before downloading its
/// debuginfo.
pub(crate) fn install_extracted(
    cache_key: &str,
    extracted: &ExtractedKernel,
    modules: &[PathBuf],
    source: KernelSource,
) -> Result<PathBuf> {
    let (arch, image_name) = crate::fetch::arch_info();
    let meta = KernelMetadata::new(source, arch, image_name, crate::test_support::now_iso8601())
        .with_version(&extracted.kernel_release);

    let mut artifacts = CacheArtifacts::new(&extracted.image).with_modules(modules);
    if let Some(vmlinux) = extracted.vmlinux.as_deref() {
        artifacts = artifacts.with_vmlinux(vmlinux);
    }
    if let Some(config) = extracted.config.as_deref() {
        artifacts = artifacts.with_config(config);
    }

    let cache = CacheDir::new()?;
    let entry = cache.store(cache_key, &artifacts, &meta).with_context(|| {
        format!(
            "install prebuilt kernel {} into cache",
            extracted.kernel_release
        )
    })?;
    Ok(entry.path)
}

/// A disk-backed scratch directory under the kernel cache root, so
/// large debuginfo downloads (~1 GiB) don't land in a tmpfs `/tmp` and
/// so the final reflink-copy into the cache stays on one filesystem.
/// Dot-prefixed, which `CacheDir::list` skips, so it never surfaces as
/// a cache entry even mid-run.
pub(crate) fn scratch_dir() -> Result<tempfile::TempDir> {
    let root = crate::cache::resolve_cache_root_with_suffix("kernels")?;
    fs::create_dir_all(&root).with_context(|| format!("create cache root {}", root.display()))?;
    tempfile::Builder::new()
        .prefix(".acquire-")
        .tempdir_in(&root)
        .with_context(|| "create acquire scratch dir")
}

/// Cache key for a distro kernel:
/// `distro-{tag}-{release}-{arch}-pkg{sha12}` — deliberately with NO
/// `kc{...}` kconfig suffix (ktstr did not build this kernel, so its
/// kconfig hash is irrelevant). `sha12` is the first 12 hex of the
/// primary (first) package's declared sha256, so a distro republish of
/// the same version under a new build lands at a distinct entry.
fn distro_cache_key(resolved: &ResolvedDistroKernel) -> Result<String> {
    let primary = resolved
        .packages
        .first()
        .ok_or_else(|| anyhow!("distro resolver returned no kernel packages"))?;
    let sha12 = primary.sha256.get(..12).unwrap_or(&primary.sha256);
    Ok(format!(
        "distro-{}-{}-{}-pkg{}",
        sanitize_key_part(&resolved.distro),
        sanitize_key_part(&resolved.kernel_release),
        resolved.arch,
        sha12,
    ))
}

/// Cache key for local packages: `pkg-{combined12}-{arch}`, where
/// `combined12` is the first 12 hex of the sha256 over the sorted
/// per-file sha256 digests (order-independent, content-addressed).
fn local_cache_key(file_hashes: &[String], arch: &str) -> String {
    let mut sorted: Vec<&str> = file_hashes.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut h = Sha256::new();
    for fh in sorted {
        h.update(fh.as_bytes());
    }
    let combined = hex::encode(h.finalize());
    format!("pkg-{}-{arch}", &combined[..12])
}

/// Reduce a key part to the `validate_cache_key` alphabet
/// (`[A-Za-z0-9._-]`), folding every other byte to `_`. Distro tags and
/// kernel releases already fit; this is a guard against a future distro
/// tag with an out-of-alphabet byte producing an invalid key.
fn sanitize_key_part(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// sha256 (lowercase hex) of a file's contents, streamed.
fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Try every equivalent official origin for one package while keeping one
/// repository-declared digest as the authority for all of them.
///
/// An origin gets the complete shared transport retry policy before advancing
/// to the next one. Interruption never looks like origin failure: it aborts the
/// sequence immediately instead of beginning another potentially large
/// download after Ctrl-C/SIGTERM.
fn try_package_urls<T, F>(package: &PackageRef, attempt: F) -> Result<T>
where
    F: FnMut(&str) -> Result<T>,
{
    try_package_urls_with(package, crate::fetch::git_operation_interrupted, attempt)
}

fn try_package_urls_with<T, I, F>(
    package: &PackageRef,
    mut interrupted: I,
    mut attempt: F,
) -> Result<T>
where
    I: FnMut() -> bool,
    F: FnMut(&str) -> Result<T>,
{
    let urls: Vec<&str> = std::iter::once(package.url.as_str())
        .chain(package.alternate_urls.iter().map(String::as_str))
        .collect();
    let mut failures = Vec::with_capacity(urls.len());

    for (index, url) in urls.iter().copied().enumerate() {
        match attempt(url) {
            Ok(value) => return Ok(value),
            Err(error) if interrupted() => {
                return Err(error).with_context(|| format!("download {} from {url}", package.name));
            }
            Err(error) => {
                failures.push(format!("{url}: {error:#}"));
                if index + 1 < urls.len() {
                    tracing::warn!(
                        package = %package.name,
                        failed_url = url,
                        next_url = urls[index + 1],
                        error = %format!("{error:#}"),
                        "official package origin failed; trying independent origin",
                    );
                }
            }
        }
    }

    bail!(
        "all official download origins failed for {}: {}",
        package.name,
        failures.join("; ")
    )
}

/// Acquire a prebuilt distro kernel (`--kernel fedora` / `ubuntu` /
/// `amazonlinux`, with optional pinned release) into the cache and
/// return its cache-entry directory.
///
/// Resolves repo metadata, composes the cache key, and on a hit returns
/// immediately. On a miss: downloads the kernel image/modules packages
/// with sha256 verification, checks each against the host arch,
/// extracts, and runs the config gate + boot-module selection — all
/// BEFORE fetching the debuginfo, so an unsupported kernel fails without
/// any (up to ~1 GiB) debuginfo bytes crossing the wire. Only once the
/// gate passes does it download + extract the debuginfo, merge its
/// vmlinux, and install atomically. Debuginfo is mandatory for every
/// distro whose repos publish it (the resolvers fail hard on a missing
/// package); a distro that publishes none (SteamOS) resolves with an
/// empty debuginfo set and proceeds with a warning naming what
/// degrades.
pub fn acquire_distro_kernel(
    kind: DistroKind,
    release: Option<&str>,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<PathBuf> {
    if kind == DistroKind::Gke {
        return crate::distro::gke::acquire_gke_kernel(release, cli_label, mp);
    }
    let resolved =
        resolve_distro_kernel(kind, release).with_context(|| "resolve distro kernel metadata")?;
    let cache_key = distro_cache_key(&resolved)?;

    let cache = CacheDir::new()?;
    if let Some(entry) = cache.lookup(&cache_key) {
        tracing::info!(cache_key = %cache_key, "distro kernel cache hit");
        return Ok(entry.path);
    }

    let msg = format!(
        "{cli_label}: resolving {} kernel {} ({} package(s) + {} debuginfo)",
        resolved.distro,
        resolved.kernel_release,
        resolved.packages.len(),
        resolved.debuginfo.len(),
    );
    match mp {
        Some(fp) => fp.println(&msg),
        None => eprintln!("{msg}"),
    }
    if resolved.debuginfo.is_empty() {
        // A distro with no published debug packages (SteamOS) is exempt
        // from the otherwise-mandatory debuginfo: the cache entry gets
        // no vmlinux, so the monitor's vmlinux-dependent features
        // (symbolized stacks, probe file:line resolution) surface their
        // normal "no vmlinux" error if actually used.
        let warn = format!(
            "{cli_label}: {} publishes no kernel debuginfo — proceeding \
             without vmlinux; monitor features that need it (symbolized \
             stacks, probe file:line) will be unavailable for this kernel",
            resolved.distro,
        );
        match mp {
            Some(fp) => fp.println(&warn),
            None => eprintln!("{warn}"),
        }
        tracing::warn!(distro = %resolved.distro, "no debuginfo published; caching kernel without vmlinux");
    }

    let scratch = scratch_dir()?;
    let download_dir = scratch.path().join("pkgs");
    fs::create_dir_all(&download_dir)?;

    // Download one package with its own progress line + sha256 check,
    // then guard it against the host arch before it is unpacked.
    let download = |pkg: &PackageRef| -> Result<PathBuf> {
        let file_name = pkg
            .url
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or(&pkg.name);
        let dest = download_dir.join(file_name);
        try_package_urls(pkg, |url| {
            crate::fetch::download_verified_file(url, &dest, &pkg.sha256, &pkg.name, cli_label, mp)
        })
        .with_context(|| format!("download {}", pkg.name))?;
        if let Some(expected) = pkg.size {
            let actual = fs::metadata(&dest)
                .with_context(|| format!("stat downloaded package {}", dest.display()))?
                .len();
            if actual != expected {
                bail!(
                    "downloaded package {} has size {actual}, expected {expected}",
                    pkg.name
                );
            }
        }
        ensure_arch_matches_host(&package_arch(&dest)?, &dest)?;
        Ok(dest)
    };

    // Download + extract the kernel image/modules packages FIRST, then
    // gate the config and select boot modules — all before touching the
    // mandatory (up to ~1 GiB) debuginfo. A kernel that cannot boot
    // under ktstr (e.g. one built without CONFIG_VIRTIO_MMIO) fails at
    // the gate here with no debuginfo bytes fetched.
    let extract_dir = scratch.path().join("tree");
    let kernel_paths: Vec<PathBuf> = resolved
        .packages
        .iter()
        .map(&download)
        .collect::<Result<_>>()?;
    let kernel_refs: Vec<&Path> = kernel_paths.iter().map(PathBuf::as_path).collect();
    let mut extracted = extract_kernel_packages(&kernel_refs, &extract_dir)
        .with_context(|| format!("extract {} packages", resolved.distro))?;

    let gate = gate_config(&extracted)?;
    let modules = select_boot_modules(&extracted, &gate)?;

    // Gate passed — only now fetch the debuginfo and merge its vmlinux
    // into the extracted set. SteamOS publishes none (debuginfo empty,
    // warned about above) and is skipped here.
    if !resolved.debuginfo.is_empty() {
        let debug_paths: Vec<PathBuf> = resolved
            .debuginfo
            .iter()
            .map(&download)
            .collect::<Result<_>>()?;
        let debug_refs: Vec<&Path> = debug_paths.iter().map(PathBuf::as_path).collect();
        extracted.vmlinux = extract_vmlinux(&debug_refs, &extract_dir, &extracted.kernel_release)
            .with_context(|| format!("extract {} debuginfo", resolved.distro))?;
    }

    let packages_meta: Vec<String> = resolved
        .packages
        .iter()
        .chain(resolved.debuginfo.iter())
        .map(|p| format!("{}-{}", p.name, p.version))
        .collect();
    let source = KernelSource::DistroPackage {
        distro: resolved.distro.clone(),
        packages: packages_meta,
    };
    install_extracted(&cache_key, &extracted, &modules, source)
}

/// Acquire a kernel from one or more local `.rpm`/`.deb` files into the
/// cache and return its cache-entry directory. No debuginfo requirement
/// — `vmlinux` is cached only when a debuginfo package is among the
/// files; otherwise the entry has no vmlinux and the monitor's
/// vmlinux-dependent features surface their normal "no vmlinux" error
/// only if actually used.
pub fn acquire_package_kernel(paths: &[PathBuf]) -> Result<PathBuf> {
    if paths.is_empty() {
        bail!("acquire_package_kernel: no package files supplied");
    }
    for p in paths {
        if !p.is_file() {
            bail!("kernel package not found: {}", p.display());
        }
    }

    let (arch, _) = crate::fetch::arch_info();
    let hashes: Vec<String> = paths
        .iter()
        .map(|p| sha256_file(p))
        .collect::<Result<_>>()?;
    let cache_key = local_cache_key(&hashes, arch);

    let cache = CacheDir::new()?;
    if let Some(entry) = cache.lookup(&cache_key) {
        tracing::info!(cache_key = %cache_key, "local package kernel cache hit");
        return Ok(entry.path);
    }

    // Arch guard up front for a clear message before extraction work.
    for p in paths {
        ensure_arch_matches_host(&package_arch(p)?, p)?;
    }

    let scratch = scratch_dir()?;
    let extract_dir = scratch.path().join("tree");
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let extracted = extract_kernel_packages(&refs, &extract_dir)
        .with_context(|| "extract local kernel packages")?;
    let gate = gate_config(&extracted)?;
    let modules = select_boot_modules(&extracted, &gate)?;

    let packages_meta: Vec<String> = paths
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    let source = KernelSource::LocalPackage {
        packages: packages_meta,
    };
    install_extracted(&cache_key, &extracted, &modules, source)
}

#[cfg(test)]
#[path = "acquire_tests.rs"]
mod tests;
