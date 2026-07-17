//! Extraction of prebuilt distro kernel packages (`.rpm` / `.deb`).
//!
//! Given one or more downloaded/local kernel packages, this module
//! materializes the filesystem tree they carry into a destination
//! directory and locates the artifacts ktstr needs to boot a VM:
//! the bootable image, module tree, kernel config, `System.map`, and
//! (from a debuginfo package) `vmlinux`.
//!
//! ## Package shapes
//!
//! - **Fedora / RHEL**: `kernel-core` carries `vmlinuz`, `config`,
//!   `System.map`, and `modules.builtin` under
//!   `/lib/modules/<release>/`; the loadable `.ko.xz` modules live in
//!   separate `kernel-modules-core` / `kernel-modules` rpms (verified
//!   against Fedora 44 packages). A `kernel-debuginfo` rpm carries
//!   `vmlinux` under `/usr/lib/debug/`.
//! - **Ubuntu / Debian**: the image lives in `linux-image-*` (under
//!   `/boot/`) and the modules in `linux-modules-*` (under
//!   `/lib/modules/`, or `/usr/lib/modules/` on usrmerge packages);
//!   the two debs merge into one tree. A `-dbgsym` ddeb contributes
//!   `vmlinux`.
//! - **Arch / SteamOS**: a pacman `.pkg.tar.zst` is a plain
//!   zstd-compressed tarball; `linux-neptune-*` carries `vmlinuz` and
//!   the module tree under `/usr/lib/modules/<release>/` (usrmerge)
//!   with the package arch in the `.PKGINFO` member's `arch = ` field
//!   (verified against a live jupiter-3.8 package). No kernel
//!   `.config` is shipped and there is no debuginfo package.
//!
//! None of these distros ship `modules.dep` in the package — depmod
//! generates it at install time — so extraction synthesizes it in
//! process from each module's `.modinfo` section (see
//! `ensure_modules_dep`), without relying on a host `depmod`.
//!
//! ## Compression
//!
//! The `rpm` crate parses package headers; the raw cpio payload is
//! decompressed here with the already-vendored xz2/zstd/flate2 stack
//! (Fedora kernels ship xz payloads, newer ones zstd) rather than
//! pulling rpm's own liblzma backend. deb `data.tar[.xz|.zst|.gz]`
//! members are streamed through the same vendored decoders.
//!
//! ## Hardening
//!
//! Every archive entry path is sanitized before it touches disk:
//! `..` components are a hard error and absolute anchors are stripped
//! so a payload path like `/lib/modules/...` lands *under* the
//! destination rather than at the filesystem root. Symlinks whose
//! target would resolve outside the destination tree are skipped
//! rather than created. This mirrors the tarball-extraction rigor in
//! [`crate::fetch`] and [`crate::cgroup`].

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::OsString;
use std::fs;
use std::io::{self, BufReader, Cursor, Read};
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

/// The artifacts recovered from one or more kernel packages.
///
/// Paths are absolute and point inside the destination directory
/// passed to [`extract_kernel_packages`], except where a field is
/// `None` because the corresponding package (config, debuginfo, …)
/// was not supplied.
#[derive(Debug, Clone)]
pub struct ExtractedKernel {
    /// `uname -r`-style kernel release string derived from the
    /// extracted tree (e.g. `6.14.6-300.fc42.x86_64`).
    pub kernel_release: String,
    /// Raw bootable image (x86_64 bzImage, aarch64 flat `Image`),
    /// normalized from whatever form the package shipped.
    pub image: PathBuf,
    /// Kernel `.config`, when the package carries one.
    pub config: Option<PathBuf>,
    /// `System.map`, when present.
    pub system_map: Option<PathBuf>,
    /// Root directory containing `lib/modules/<release>/…`, suitable
    /// for [`module_closure`]. `None` when no module tree was found.
    pub modules_dir: Option<PathBuf>,
    /// Uncompressed `vmlinux` from a debuginfo package, when supplied.
    pub vmlinux: Option<PathBuf>,
}

/// Extract one or more kernel packages into `dest` and locate the
/// bootable artifacts.
///
/// Multiple packages are merged into a single tree under `dest`
/// (Ubuntu's split image/modules debs, or a Fedora kernel plus its
/// debuginfo). Every package is checked against the host architecture
/// before extraction. The kernel release is derived from the merged
/// tree and it is an error for the packages to disagree on it.
pub fn extract_kernel_packages(packages: &[&Path], dest: &Path) -> Result<ExtractedKernel> {
    if packages.is_empty() {
        bail!("extract_kernel_packages: no packages supplied");
    }
    fs::create_dir_all(dest).with_context(|| format!("create destination {}", dest.display()))?;
    let dest = dest
        .canonicalize()
        .with_context(|| format!("canonicalize destination {}", dest.display()))?;

    for pkg in packages {
        let arch = package_arch(pkg)?;
        ensure_arch_matches_host(&arch, pkg)?;
        match package_kind(pkg)? {
            PackageKind::Rpm => extract_rpm(pkg, &dest)?,
            PackageKind::Deb => extract_deb(pkg, &dest)?,
            PackageKind::Pacman => extract_pacman(pkg, &dest)?,
        }
    }

    unify_modules_layout(&dest)?;
    let release = discover_release(&dest)?;
    let image_src = find_image(&dest, &release).ok_or_else(|| {
        anyhow!(
            "no kernel image (lib/modules/{release}/vmlinuz or boot/vmlinuz-{release}) \
             found under {}",
            dest.display()
        )
    })?;
    let image = normalize_image(&image_src, &dest, &release)?;

    let modules_root = dest.join(format!("lib/modules/{release}"));
    let modules_dir = modules_root.is_dir().then(|| dest.clone());
    if modules_dir.is_some() {
        ensure_modules_dep(&dest, &release);
    }
    Ok(ExtractedKernel {
        kernel_release: release.clone(),
        image,
        config: find_first(
            &dest,
            &[
                format!("boot/config-{release}"),
                format!("lib/modules/{release}/config"),
            ],
        ),
        system_map: find_first(
            &dest,
            &[
                format!("boot/System.map-{release}"),
                format!("lib/modules/{release}/System.map"),
            ],
        ),
        vmlinux: find_first(
            &dest,
            &[
                format!("usr/lib/debug/lib/modules/{release}/vmlinux"),
                format!("usr/lib/debug/boot/vmlinux-{release}"),
            ],
        ),
        modules_dir,
    })
}

/// Extract debuginfo package(s) into `dest` — merging onto a kernel
/// tree already materialized there by [`extract_kernel_packages`] — and
/// return the `vmlinux` path they carry, if any.
///
/// Split out from [`extract_kernel_packages`] so the distro acquisition
/// path can run its config gate on the kernel packages BEFORE fetching
/// the (mandatory, up to ~1 GiB) debuginfo: a kernel that cannot boot
/// under ktstr fails at the gate with no debuginfo bytes crossing the
/// wire. `release` is the kernel release already discovered from the
/// kernel tree; the lookup uses the same `vmlinux` candidates
/// [`extract_kernel_packages`] checks. A second extraction pass into the
/// same `dest` is safe — the debuginfo packages only add
/// `usr/lib/debug/…` paths and are arch-checked like any package.
pub fn extract_vmlinux(packages: &[&Path], dest: &Path, release: &str) -> Result<Option<PathBuf>> {
    let dest = dest
        .canonicalize()
        .with_context(|| format!("canonicalize destination {}", dest.display()))?;
    for pkg in packages {
        let arch = package_arch(pkg)?;
        ensure_arch_matches_host(&arch, pkg)?;
        match package_kind(pkg)? {
            PackageKind::Rpm => extract_rpm(pkg, &dest)?,
            PackageKind::Deb => extract_deb(pkg, &dest)?,
            PackageKind::Pacman => extract_pacman(pkg, &dest)?,
        }
    }
    Ok(find_first(
        &dest,
        &[
            format!("usr/lib/debug/lib/modules/{release}/vmlinux"),
            format!("usr/lib/debug/boot/vmlinux-{release}"),
        ],
    ))
}

/// Fold the usrmerge module layout onto the classic one.
///
/// Newer Ubuntu debs ship modules under `usr/lib/modules/<rel>/`
/// (usrmerge) where Fedora rpms use `lib/modules/<rel>/`; on an
/// installed system `/lib` is a symlink into `/usr`, but the
/// extracted tree has no such link. Bridge the two `modules`
/// directories with a relative symlink so every downstream lookup —
/// image discovery, `modules.dep` synthesis, and `module_closure` —
/// resolves through the classic `lib/modules/<rel>` path (and vice
/// versa) regardless of which layout the package shipped.
fn unify_modules_layout(dest: &Path) -> Result<()> {
    let classic = dest.join("lib/modules");
    let usrmerge = dest.join("usr/lib/modules");
    if usrmerge.is_dir() && !classic.exists() {
        fs::create_dir_all(dest.join("lib"))?;
        std::os::unix::fs::symlink("../usr/lib/modules", &classic)
            .with_context(|| format!("symlink {}", classic.display()))?;
    } else if classic.is_dir() && !usrmerge.exists() {
        fs::create_dir_all(dest.join("usr/lib"))?;
        std::os::unix::fs::symlink("../../lib/modules", &usrmerge)
            .with_context(|| format!("symlink {}", usrmerge.display()))?;
    }
    Ok(())
}

/// Synthesize `modules.dep` for a package that doesn't ship one.
///
/// Neither Fedora's kernel-core rpm nor Ubuntu's linux-modules deb
/// contains `modules.dep` — depmod generates it at package install
/// time. Rather than shell out to the host `depmod` (whose kmod may be
/// unable to decompress the tree's `.ko.xz` modules, silently emitting
/// an incomplete file), build it in process from each module's
/// `.modinfo` (see [`build_modules_dep`]). A failure is a warning, not
/// a hard error: [`module_closure`] still emits an actionable message
/// if the file is missing when modules are actually requested.
fn ensure_modules_dep(dest: &Path, release: &str) {
    let modules_dir = dest.join(format!("lib/modules/{release}"));
    let dep_path = modules_dir.join("modules.dep");
    if dep_path.exists() {
        return;
    }
    match build_modules_dep(&modules_dir) {
        Ok(text) => {
            if let Err(e) = fs::write(&dep_path, text) {
                tracing::warn!(
                    release,
                    error = %e,
                    "could not write synthesized modules.dep; module_closure will not resolve dependencies for this tree"
                );
            }
        }
        Err(e) => tracing::warn!(
            release,
            error = %e,
            "could not synthesize modules.dep; module_closure will not resolve dependencies for this tree"
        ),
    }
}

/// `.modinfo` fields ktstr needs to reconstruct `modules.dep`.
struct ModInfo {
    /// The module's registered name (`name=` field), if present.
    name: Option<String>,
    /// Direct dependency module names (`depends=`, comma-separated).
    depends: Vec<String>,
}

/// Build `modules.dep` text for the module tree rooted at
/// `modules_dir` (a `lib/modules/<release>/` directory).
///
/// Walks every module file, reads its `.modinfo` for the module name
/// and direct `depends`, and emits one depmod-format line per module:
/// the module's path, a colon, then the space-separated paths of its
/// direct dependencies — all relative to `modules_dir`, exactly the
/// shape [`module_closure`] parses. Dependency names are resolved to
/// paths through the name→path map; a `depends` entry with no matching
/// module file (a built-in, or one outside the tree) is dropped, as
/// real depmod does. Lines are sorted for a deterministic file.
fn build_modules_dep(modules_dir: &Path) -> Result<String> {
    let mut entries: Vec<(String, Vec<String>)> = Vec::new();
    let mut name_to_path: HashMap<String, String> = HashMap::new();

    for entry in walkdir::WalkDir::new(modules_dir).follow_links(false) {
        let entry = entry.with_context(|| format!("walk {}", modules_dir.display()))?;
        if !entry.file_type().is_file() || !is_module_file(entry.path()) {
            continue;
        }
        let path = entry.path();
        let rel = path
            .strip_prefix(modules_dir)
            .expect("walkdir entry is under modules_dir")
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 module path {}", path.display()))?
            .to_string();

        let raw = fs::read(path).with_context(|| format!("read module {}", path.display()))?;
        let elf = match path.extension().and_then(|e| e.to_str()) {
            Some("ko") => raw,
            Some(ext) => decompress_bytes(ext, &raw)
                .with_context(|| format!("decompress module {}", path.display()))?,
            None => raw,
        };
        let info =
            parse_modinfo(&elf).with_context(|| format!("read .modinfo of {}", path.display()))?;
        let name = info.name.unwrap_or_else(|| module_key(&rel));
        name_to_path.insert(module_key(&name), rel.clone());
        entries.push((rel, info.depends));
    }

    let mut lines: Vec<String> = entries
        .iter()
        .map(|(rel, depends)| {
            let deps: Vec<&str> = depends
                .iter()
                .filter_map(|d| name_to_path.get(&module_key(d)).map(String::as_str))
                .collect();
            if deps.is_empty() {
                format!("{rel}:")
            } else {
                format!("{rel}: {}", deps.join(" "))
            }
        })
        .collect();
    lines.sort();
    Ok(lines.join("\n") + "\n")
}

/// Whether `path`'s basename is a loadable module (`.ko` or a
/// `.ko.{xz,zst,gz}` compressed module).
fn is_module_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.ends_with(".ko")
        || name.ends_with(".ko.xz")
        || name.ends_with(".ko.zst")
        || name.ends_with(".ko.gz")
}

/// Read the `name=` and `depends=` fields from a module's `.modinfo`
/// ELF section (NUL-separated `key=value` pairs).
fn parse_modinfo(elf: &[u8]) -> Result<ModInfo> {
    let obj = goblin::elf::Elf::parse(elf).context("parse module ELF")?;
    let section = obj
        .section_headers
        .iter()
        .find(|sh| obj.shdr_strtab.get_at(sh.sh_name) == Some(".modinfo"))
        .ok_or_else(|| anyhow!("module has no .modinfo section"))?;
    let start = section.sh_offset as usize;
    let end = start
        .checked_add(section.sh_size as usize)
        .ok_or_else(|| anyhow!(".modinfo section size overflows"))?;
    let blob = elf
        .get(start..end)
        .ok_or_else(|| anyhow!(".modinfo section [{start:#x}..{end:#x}] out of bounds"))?;

    let mut name = None;
    let mut depends = Vec::new();
    for field in blob.split(|&b| b == 0).filter(|f| !f.is_empty()) {
        let field =
            std::str::from_utf8(field).map_err(|_| anyhow!(".modinfo field is not UTF-8"))?;
        if let Some(v) = field.strip_prefix("name=") {
            name = Some(v.to_string());
        } else if let Some(v) = field.strip_prefix("depends=") {
            depends = v
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        }
    }
    Ok(ModInfo { name, depends })
}

// ---------------------------------------------------------------------------
// Architecture guard
// ---------------------------------------------------------------------------

/// Package architecture as spelled by the package itself
/// (`x86_64`/`aarch64` for rpm, `amd64`/`arm64` for deb).
pub fn package_arch(path: &Path) -> Result<String> {
    match package_kind(path)? {
        PackageKind::Rpm => {
            let meta = rpm::PackageMetadata::open(path)
                .with_context(|| format!("parse rpm header {}", path.display()))?;
            Ok(meta.get_arch()?.to_string())
        }
        PackageKind::Deb => deb_arch(path),
        PackageKind::Pacman => pacman_arch(path),
    }
}

/// Canonicalize a distro arch token to ktstr's host-arch spelling,
/// mapping the deb `amd64`/`arm64` aliases onto `x86_64`/`aarch64`.
fn canonical_arch(arch: &str) -> Option<&'static str> {
    match arch.trim().to_ascii_lowercase().as_str() {
        "x86_64" | "amd64" => Some("x86_64"),
        "aarch64" | "arm64" => Some("aarch64"),
        _ => None,
    }
}

/// Error unless `pkg_arch` matches the compile-time host architecture
/// (see [`crate::fetch::arch_info`]).
pub fn ensure_arch_matches_host(pkg_arch: &str, pkg: &Path) -> Result<()> {
    let host = crate::fetch::arch_info().0;
    let canon = canonical_arch(pkg_arch).ok_or_else(|| {
        anyhow!(
            "unsupported package architecture {pkg_arch:?} for {}",
            pkg.display()
        )
    })?;
    if canon != host {
        bail!(
            "package {} targets {canon} (declared {pkg_arch:?}), but this host is {host}",
            pkg.display()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Package-kind dispatch
// ---------------------------------------------------------------------------

enum PackageKind {
    Rpm,
    Deb,
    Pacman,
}

/// Classify a package by extension (`.pkg.tar.zst` is matched on the
/// full name — `Path::extension` only sees the final `.zst`), falling
/// back to the leading magic bytes (rpm lead `ed ab ee db`, ar/deb
/// `!<arch>\n`, zstd frame `28 b5 2f fd` for a pacman package).
fn package_kind(path: &Path) -> Result<PackageKind> {
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".pkg.tar.zst"))
    {
        return Ok(PackageKind::Pacman);
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("rpm") => return Ok(PackageKind::Rpm),
        Some("deb" | "ddeb" | "udeb") => return Ok(PackageKind::Deb),
        _ => {}
    }
    let mut magic = [0u8; 8];
    let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let n = f.read(&mut magic)?;
    let magic = &magic[..n];
    if magic.starts_with(&[0xed, 0xab, 0xee, 0xdb]) {
        Ok(PackageKind::Rpm)
    } else if magic.starts_with(b"!<arch>\n") {
        Ok(PackageKind::Deb)
    } else if magic.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Ok(PackageKind::Pacman)
    } else {
        bail!("{} is not an rpm, deb, or pacman package", path.display())
    }
}

// ---------------------------------------------------------------------------
// RPM extraction
// ---------------------------------------------------------------------------

fn extract_rpm(path: &Path, dest: &Path) -> Result<()> {
    let pkg = rpm::Package::open(path).with_context(|| format!("parse rpm {}", path.display()))?;
    let compressor = pkg.metadata.get_payload_compressor()?;
    let reader = rpm_payload_decoder(compressor, &pkg.payload)
        .with_context(|| format!("open payload of {}", path.display()))?;
    extract_cpio(reader, dest)
        .with_context(|| format!("extract cpio payload of {}", path.display()))
}

/// Wrap the raw (compressed) rpm payload in the matching vendored
/// decoder. rpm payloads are cpio streams compressed with the codec
/// named by `RPMTAG_PAYLOADCOMPRESSOR`; Fedora/RHEL kernels use xz,
/// newer Fedora uses zstd. bzip2 (long obsolete for rpm) is rejected.
fn rpm_payload_decoder<'a>(
    compressor: rpm::CompressionType,
    payload: &'a [u8],
) -> Result<Box<dyn Read + 'a>> {
    let cursor = Cursor::new(payload);
    Ok(match compressor {
        rpm::CompressionType::None => Box::new(cursor),
        rpm::CompressionType::Gzip => Box::new(flate2::read::GzDecoder::new(cursor)),
        rpm::CompressionType::Zstd => Box::new(zstd::stream::read::Decoder::new(cursor)?),
        rpm::CompressionType::Xz => Box::new(xz2::read::XzDecoder::new(cursor)),
        other => bail!("unsupported rpm payload compression {other}"),
    })
}

/// Extract a newc-format cpio stream (SVR4 ASCII, magic `070701` /
/// `070702`) with the same path hardening as [`extract_tar`].
///
/// The rpm-specific "stripped" variant (`07070X`), which stores file
/// names in the header rather than the payload, is not produced by the
/// distro kernel packages ktstr targets and is rejected explicitly.
fn extract_cpio<R: Read>(mut r: R, dest: &Path) -> Result<()> {
    loop {
        let mut hdr = [0u8; 110];
        r.read_exact(&mut hdr).context("read cpio header")?;
        match &hdr[0..6] {
            b"070701" | b"070702" => {}
            b"07070X" => bail!("rpm 'stripped' cpio payloads are not supported"),
            magic => bail!("unknown cpio magic {:?}", String::from_utf8_lossy(magic)),
        }
        let field = |i: usize| cpio_hex(&hdr[6 + i * 8..6 + i * 8 + 8]);
        let mode = field(1)?;
        let filesize = field(6)? as usize;
        let namesize = field(11)? as usize;

        let mut name = vec![0u8; namesize];
        r.read_exact(&mut name).context("read cpio name")?;
        while name.last() == Some(&0) {
            name.pop();
        }
        skip_exact(&mut r, pad4(110 + namesize))?;
        let name = String::from_utf8(name).context("cpio entry name is not utf-8")?;
        if name == "TRAILER!!!" {
            break;
        }

        let rel = sanitize_relative(Path::new(&name))?;
        write_cpio_entry(&mut r, dest, rel.as_deref(), mode, filesize)?;
        skip_exact(&mut r, pad4(filesize))?;
    }
    Ok(())
}

/// Materialize one cpio entry, consuming exactly `filesize` payload
/// bytes regardless of whether the entry is kept (so the stream stays
/// aligned for the next header).
fn write_cpio_entry<R: Read>(
    r: &mut R,
    dest: &Path,
    rel: Option<&Path>,
    mode: u32,
    filesize: usize,
) -> Result<()> {
    const S_IFMT: u32 = 0o170000;
    match (mode & S_IFMT, rel) {
        (0o040000, Some(rel)) => {
            fs::create_dir_all(dest.join(rel))?;
        }
        (0o100000, Some(rel)) => {
            let out = dest.join(rel);
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut f =
                fs::File::create(&out).with_context(|| format!("create {}", out.display()))?;
            let copied = io::copy(&mut r.by_ref().take(filesize as u64), &mut f)?;
            if copied != filesize as u64 {
                bail!(
                    "truncated cpio entry {} (wanted {filesize} bytes, got {copied})",
                    out.display()
                );
            }
            let _ = f.set_permissions(fs::Permissions::from_mode(mode & 0o7777));
            return Ok(());
        }
        (0o120000, rel) => {
            let mut target = vec![0u8; filesize];
            r.read_exact(&mut target)?;
            if let (Some(rel), Ok(target)) = (rel, std::str::from_utf8(&target)) {
                create_symlink_safe(dest, rel, Path::new(target))?;
            }
            return Ok(());
        }
        _ => {}
    }
    // Dropped or unhandled entry: drain its data so the stream stays aligned.
    skip_exact(r, filesize)
}

/// Parse an 8-char ASCII hex cpio header field.
fn cpio_hex(bytes: &[u8]) -> Result<u32> {
    let s = std::str::from_utf8(bytes).map_err(|_| anyhow!("non-ascii cpio header field"))?;
    u32::from_str_radix(s.trim(), 16).map_err(|_| anyhow!("malformed cpio header field {s:?}"))
}

/// newc pads each of (header+name) and (data) to a 4-byte boundary.
fn pad4(len: usize) -> usize {
    (4 - (len % 4)) % 4
}

/// Read and discard exactly `n` bytes, erroring on a short stream.
fn skip_exact<R: Read>(r: &mut R, n: usize) -> Result<()> {
    let copied = io::copy(&mut r.by_ref().take(n as u64), &mut io::sink())?;
    if copied != n as u64 {
        bail!("truncated archive stream (wanted {n} bytes, got {copied})");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DEB extraction
// ---------------------------------------------------------------------------

fn extract_deb(path: &Path, dest: &Path) -> Result<()> {
    let (name, data) = find_ar_member(path, |n| n.starts_with("data.tar"))?
        .ok_or_else(|| anyhow!("{} has no data.tar member", path.display()))?;
    extract_data_tar(&name, data, dest)
        .with_context(|| format!("extract {name} from {}", path.display()))
}

/// Stream a `data.tar.{xz,zst,gz}` (or plain `data.tar`) member through
/// the matching vendored decoder into the hardened tar extractor.
fn extract_data_tar(name: &str, data: Vec<u8>, dest: &Path) -> Result<()> {
    match compression_ext(name) {
        "xz" => extract_tar(xz2::read::XzDecoder::new(Cursor::new(data)), dest),
        "gz" => extract_tar(flate2::read::GzDecoder::new(Cursor::new(data)), dest),
        "zst" | "zstd" => extract_tar(zstd::stream::read::Decoder::new(Cursor::new(data))?, dest),
        "tar" => extract_tar(Cursor::new(data), dest),
        other => bail!("unsupported data.tar compression {other:?} ({name})"),
    }
}

/// Hardened tar extraction: absolute anchors are stripped, `..` is a
/// hard error, and escaping symlinks are skipped.
pub(crate) fn extract_tar<R: Read>(reader: R, dest: &Path) -> Result<()> {
    extract_tar_matching(reader, dest, |_| true)
}

/// Hardened selective tar extraction using the same path and symlink
/// checks as [`extract_tar`]. `include` receives the sanitized relative
/// path, so callers can extract a small source subset from a large
/// archive without duplicating archive-hardening logic.
pub(crate) fn extract_tar_matching<R: Read, F: FnMut(&Path) -> bool>(
    reader: R,
    dest: &Path,
    mut include: F,
) -> Result<()> {
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let rel = match sanitize_relative(&path)? {
            Some(rel) => rel,
            None => continue,
        };
        if !include(&rel) {
            continue;
        }
        let out = dest.join(&rel);
        match entry.header().entry_type() {
            tar::EntryType::Directory => {
                fs::create_dir_all(&out)?;
            }
            tar::EntryType::Symlink | tar::EntryType::Link => {
                if let Some(target) = entry.link_name()? {
                    create_symlink_safe(dest, &rel, &target)?;
                }
            }
            _ => {
                let mode = entry.header().mode().unwrap_or(0o644);
                if let Some(parent) = out.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut f =
                    fs::File::create(&out).with_context(|| format!("create {}", out.display()))?;
                io::copy(&mut entry, &mut f)?;
                let _ = f.set_permissions(fs::Permissions::from_mode(mode & 0o7777));
            }
        }
    }
    Ok(())
}

/// The last dotted component of an archive member name
/// (`data.tar.xz` → `xz`), used to pick a decompressor.
fn compression_ext(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or("")
}

// ---------------------------------------------------------------------------
// `ar` container (deb) reader
// ---------------------------------------------------------------------------

/// Return the bytes of the first `ar` member whose name satisfies
/// `want`. The deb `ar` format is a fixed 8-byte global header
/// (`!<arch>\n`) followed by 60-byte member headers; member data is
/// padded to an even boundary.
fn find_ar_member<F: FnMut(&str) -> bool>(
    path: &Path,
    mut want: F,
) -> Result<Option<(String, Vec<u8>)>> {
    let mut f =
        BufReader::new(fs::File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut global = [0u8; 8];
    f.read_exact(&mut global)?;
    if &global != b"!<arch>\n" {
        bail!("{} is not an ar archive", path.display());
    }
    loop {
        let mut hdr = [0u8; 60];
        match f.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        if &hdr[58..60] != b"`\n" {
            bail!("{}: malformed ar member header", path.display());
        }
        let name = std::str::from_utf8(&hdr[0..16])
            .map_err(|_| anyhow!("{}: non-utf8 ar member name", path.display()))?
            .trim_end()
            .trim_end_matches('/')
            .to_string();
        let size: u64 = std::str::from_utf8(&hdr[48..58])
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| anyhow!("{}: malformed ar member size", path.display()))?;
        if want(&name) {
            let mut buf = vec![0u8; size as usize];
            f.read_exact(&mut buf)?;
            return Ok(Some((name, buf)));
        }
        let pad = size % 2;
        io::copy(&mut (&mut f).take(size + pad), &mut io::sink())?;
    }
    Ok(None)
}

/// Parse the deb architecture from the `control.tar.*` member's
/// `control` file, falling back to the `_<arch>.deb` filename token.
fn deb_arch(path: &Path) -> Result<String> {
    if let Some((name, data)) = find_ar_member(path, |n| n.starts_with("control.tar"))?
        && let Some(arch) = control_arch(&name, data)
    {
        return Ok(arch);
    }
    filename_arch(path)
}

fn control_arch(name: &str, data: Vec<u8>) -> Option<String> {
    let control = read_tar_entry(name, data, "control")?;
    let text = String::from_utf8(control).ok()?;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("Architecture:") {
            return Some(v.trim().to_string());
        }
    }
    None
}

/// Read one named file (matched by basename) out of a possibly
/// compressed tar member into memory.
fn read_tar_entry(member_name: &str, data: Vec<u8>, basename: &str) -> Option<Vec<u8>> {
    let reader: Box<dyn Read> = match compression_ext(member_name) {
        "xz" => Box::new(xz2::read::XzDecoder::new(Cursor::new(data))),
        "gz" => Box::new(flate2::read::GzDecoder::new(Cursor::new(data))),
        "zst" | "zstd" => Box::new(zstd::stream::read::Decoder::new(Cursor::new(data)).ok()?),
        _ => Box::new(Cursor::new(data)),
    };
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries().ok()? {
        let mut entry = entry.ok()?;
        let path = entry.path().ok()?.into_owned();
        if path.file_name().and_then(|n| n.to_str()) == Some(basename) {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).ok()?;
            return Some(buf);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Pacman (.pkg.tar.zst) extraction
// ---------------------------------------------------------------------------

/// A pacman package is one zstd-compressed tarball; stream it through
/// the vendored decoder into the hardened tar extractor. The
/// `.PKGINFO`/`.MTREE`/`.BUILDINFO` metadata members land as dot-files
/// in the destination root, where no artifact lookup ever visits them.
fn extract_pacman(path: &Path, dest: &Path) -> Result<()> {
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = zstd::stream::read::Decoder::new(BufReader::new(f))
        .with_context(|| format!("init zstd decoder for {}", path.display()))?;
    extract_tar(reader, dest).with_context(|| format!("extract pacman {}", path.display()))
}

/// Parse the pacman package architecture from the `.PKGINFO` member's
/// `arch = <arch>` line.
fn pacman_arch(path: &Path) -> Result<String> {
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = zstd::stream::read::Decoder::new(BufReader::new(f))
        .with_context(|| format!("init zstd decoder for {}", path.display()))?;
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries()? {
        let mut entry = entry?;
        if entry.path()?.as_os_str() != ".PKGINFO" {
            continue;
        }
        let mut text = String::new();
        entry.read_to_string(&mut text)?;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("arch") {
                let v = v.trim_start();
                if let Some(v) = v.strip_prefix('=') {
                    return Ok(v.trim().to_string());
                }
            }
        }
        bail!("{}: .PKGINFO has no `arch =` field", path.display());
    }
    bail!("{}: no .PKGINFO member found", path.display())
}

fn filename_arch(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|stem| stem.rsplit('_').next())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("cannot infer architecture from filename {}", path.display()))
}

// ---------------------------------------------------------------------------
// Path hardening
// ---------------------------------------------------------------------------

/// Reduce an archive entry path to a safe, destination-relative path.
///
/// Absolute anchors and `.` components are dropped; a `..` component is
/// a hard error (traversal attempt). Returns `Ok(None)` for a path that
/// reduces to nothing (e.g. `.` / `/`).
fn sanitize_relative(entry: &Path) -> Result<Option<PathBuf>> {
    let mut out = PathBuf::new();
    for comp in entry.components() {
        match comp {
            Component::Normal(c) => out.push(c),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
            Component::ParentDir => {
                bail!(
                    "archive entry escapes destination via `..`: {}",
                    entry.display()
                )
            }
        }
    }
    Ok((!out.as_os_str().is_empty()).then_some(out))
}

/// Create `dest/link_rel` -> `target`, but only when the link target
/// resolves to a path inside `dest`. Escaping targets (absolute, or
/// `..` climbing above the root) are silently skipped — they are never
/// the `.ko`/image/config artifacts ktstr consumes, and honoring them
/// would let a package plant a link outside the tree.
fn create_symlink_safe(dest: &Path, link_rel: &Path, target: &Path) -> Result<()> {
    let link_dir = link_rel.parent().unwrap_or_else(|| Path::new(""));
    if resolve_within_root(link_dir, target).is_none() {
        tracing::debug!(
            link = %link_rel.display(),
            target = %target.display(),
            "skipping symlink whose target escapes the extraction root"
        );
        return Ok(());
    }
    let link_path = dest.join(link_rel);
    if let Some(parent) = link_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = fs::remove_file(&link_path);
    std::os::unix::fs::symlink(target, &link_path)
        .with_context(|| format!("symlink {} -> {}", link_path.display(), target.display()))?;
    Ok(())
}

/// Lexically resolve `target` relative to `base` (both destination-
/// relative) and return the normalized result, or `None` if it is
/// absolute or climbs above the root.
fn resolve_within_root(base: &Path, target: &Path) -> Option<PathBuf> {
    let mut stack: Vec<OsString> = base
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_os_string()),
            _ => None,
        })
        .collect();
    for comp in target.components() {
        match comp {
            Component::Normal(s) => stack.push(s.to_os_string()),
            Component::CurDir => {}
            Component::ParentDir => {
                stack.pop()?;
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(stack.iter().collect())
}

// ---------------------------------------------------------------------------
// Tree discovery
// ---------------------------------------------------------------------------

/// Derive the kernel release from the extracted tree, erroring if the
/// packages disagree.
fn discover_release(dest: &Path) -> Result<String> {
    let mut releases = BTreeSet::new();
    collect_subdir_names(&dest.join("lib/modules"), &mut releases);
    collect_subdir_names(&dest.join("usr/lib/debug/lib/modules"), &mut releases);
    collect_suffixed(
        &dest.join("boot"),
        &["vmlinuz-", "config-", "System.map-"],
        &mut releases,
    );
    collect_suffixed(
        &dest.join("usr/lib/debug/boot"),
        &["vmlinux-"],
        &mut releases,
    );

    match releases.len() {
        0 => bail!(
            "could not determine kernel release: no lib/modules/<rel> or boot/*-<rel> \
             entries under {}",
            dest.display()
        ),
        1 => Ok(releases.into_iter().next().unwrap()),
        _ => bail!("packages disagree on kernel release: {releases:?}"),
    }
}

fn collect_subdir_names(dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
            && let Some(name) = entry.file_name().to_str()
        {
            out.insert(name.to_string());
        }
    }
}

fn collect_suffixed(dir: &Path, prefixes: &[&str], out: &mut BTreeSet<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        for prefix in prefixes {
            if let Some(rel) = name.strip_prefix(prefix) {
                out.insert(rel.to_string());
            }
        }
    }
}

fn find_image(dest: &Path, release: &str) -> Option<PathBuf> {
    find_first(
        dest,
        &[
            format!("lib/modules/{release}/vmlinuz"),
            format!("boot/vmlinuz-{release}"),
        ],
    )
}

/// Return the first of `candidates` (relative to `dest`) that exists.
fn find_first(dest: &Path, candidates: &[String]) -> Option<PathBuf> {
    candidates.iter().map(|c| dest.join(c)).find(|p| p.exists())
}

// ---------------------------------------------------------------------------
// Image normalization
// ---------------------------------------------------------------------------

/// "HdrS" setup-header magic at offset 0x202 of an x86 bzImage.
/// Consumed only by the x86_64 `normalize_image` and by the unit tests
/// (which synthesize host images on every arch), so gate accordingly to
/// stay dead-code-free in the aarch64 non-test build under `-D warnings`.
#[cfg(any(target_arch = "x86_64", test))]
const BZIMAGE_MAGIC_OFFSET: usize = 0x202;
#[cfg(any(target_arch = "x86_64", test))]
const BZIMAGE_MAGIC: [u8; 4] = *b"HdrS";
/// arm64 flat `Image` magic ("ARM\x64") at offset 0x38.
const ARM64_MAGIC_OFFSET: usize = 0x38;
const ARM64_MAGIC: [u8; 4] = [b'A', b'R', b'M', 0x64];

/// EFI zboot container header, from linux
/// `drivers/firmware/efi/libstub/zboot-header.S`: a PE/COFF DOS
/// signature ("MZ") at 0x00 and a "zimg" marker at 0x04, the compressed
/// payload's offset and size as u32-LE at 0x08 and 0x0c, and the
/// NUL-terminated compression-type string ("gzip"/"zstd") at 0x18. The
/// size field already excludes the trailing uncompressed-length word
/// (`ZBOOT_SIZE_LEN`), so it delimits exactly the compressed stream.
const ZBOOT_ZIMG_OFFSET: usize = 0x04;
const ZBOOT_PAYLOAD_OFFSET_FIELD: usize = 0x08;
const ZBOOT_PAYLOAD_SIZE_FIELD: usize = 0x0c;
const ZBOOT_COMP_TYPE_OFFSET: usize = 0x18;

fn has_magic(bytes: &[u8], offset: usize, magic: &[u8; 4]) -> bool {
    bytes
        .get(offset..offset + 4)
        .is_some_and(|slice| slice == magic)
}

fn has_gzip_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x1f, 0x8b])
}

fn gunzip(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_end(&mut out)
        .context("gunzip compressed kernel image")?;
    Ok(out)
}

/// Normalize a distro image to a raw bootable image for the host arch.
///
/// x86_64 distro images (including EFI-signed ones) are already valid
/// bzImages and pass through after a magic check. aarch64 images are a
/// flat `Image` that may be gzip-wrapped or carried in an EFI zboot
/// container (Fedora's aarch64 kernels ship the latter); either wrapper
/// is decompressed to a new file beside the tree.
#[cfg(target_arch = "x86_64")]
fn normalize_image(src: &Path, _dest: &Path, _release: &str) -> Result<PathBuf> {
    let prefix = read_prefix(src, BZIMAGE_MAGIC_OFFSET + 4)?;
    if !has_magic(&prefix, BZIMAGE_MAGIC_OFFSET, &BZIMAGE_MAGIC) {
        bail!(
            "{} is not a valid x86_64 bzImage (missing HdrS magic)",
            src.display()
        );
    }
    Ok(src.to_path_buf())
}

#[cfg(target_arch = "aarch64")]
fn normalize_image(src: &Path, dest: &Path, release: &str) -> Result<PathBuf> {
    let raw = fs::read(src).with_context(|| format!("read image {}", src.display()))?;
    // A bare flat Image boots as-is; a gzip- or EFI-zboot-wrapped image
    // must be decompressed to a new file beside the tree.
    if has_magic(&raw, ARM64_MAGIC_OFFSET, &ARM64_MAGIC) {
        return Ok(src.to_path_buf());
    }
    let image = normalize_arm64_bytes(&raw)?;
    let out = dest.join(format!("vmlinuz-{release}.img"));
    fs::write(&out, &image).with_context(|| format!("write image {}", out.display()))?;
    Ok(out)
}

/// Decompress (if gzip-wrapped) and verify an aarch64 flat `Image`,
/// returning the raw image bytes. Arch-independent so it is unit-
/// testable on any host.
#[cfg_attr(not(test), allow(dead_code))]
fn normalize_arm64_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let image = if is_zboot(bytes) {
        decompress_zboot(bytes)?
    } else if has_gzip_magic(bytes) {
        gunzip(bytes)?
    } else {
        bytes.to_vec()
    };
    if !has_magic(&image, ARM64_MAGIC_OFFSET, &ARM64_MAGIC) {
        bail!("decompressed image is not a valid aarch64 Image (missing ARM\\x64 magic)");
    }
    Ok(image)
}

/// Detect an EFI zboot container by its PE "MZ" signature and "zimg"
/// marker (see [`ZBOOT_ZIMG_OFFSET`] and the constants above it).
#[cfg_attr(not(test), allow(dead_code))]
fn is_zboot(bytes: &[u8]) -> bool {
    bytes.starts_with(b"MZ")
        && bytes.get(ZBOOT_ZIMG_OFFSET..ZBOOT_ZIMG_OFFSET + 4) == Some(b"zimg".as_slice())
}

/// Slice the compressed payload out of an EFI zboot container and
/// decompress it per the header's compression-type string. Every offset
/// and length taken from the (attacker-controlled) header is
/// bounds-checked, so a truncated or malformed container errors rather
/// than panics.
#[cfg_attr(not(test), allow(dead_code))]
fn decompress_zboot(bytes: &[u8]) -> Result<Vec<u8>> {
    let read_u32 = |off: usize| -> Result<usize> {
        let raw = bytes
            .get(off..off + 4)
            .ok_or_else(|| anyhow!("EFI zboot header truncated at offset {off:#x}"))?;
        Ok(u32::from_le_bytes(raw.try_into().unwrap()) as usize)
    };
    let payload_offset = read_u32(ZBOOT_PAYLOAD_OFFSET_FIELD)?;
    let payload_size = read_u32(ZBOOT_PAYLOAD_SIZE_FIELD)?;
    let comp_region = bytes
        .get(ZBOOT_COMP_TYPE_OFFSET..)
        .ok_or_else(|| anyhow!("EFI zboot header truncated before the compression type"))?;
    let comp_len = comp_region
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| anyhow!("EFI zboot compression type is not NUL-terminated"))?;
    let comp = std::str::from_utf8(&comp_region[..comp_len])
        .map_err(|_| anyhow!("EFI zboot compression type is not valid UTF-8"))?;
    let end = payload_offset
        .checked_add(payload_size)
        .ok_or_else(|| anyhow!("EFI zboot payload bounds overflow"))?;
    let payload = bytes.get(payload_offset..end).ok_or_else(|| {
        anyhow!(
            "EFI zboot payload [{payload_offset:#x}..{end:#x}] extends past the {}-byte image",
            bytes.len()
        )
    })?;
    decompress_bytes(comp, payload).with_context(|| format!("decompress EFI zboot {comp} payload"))
}

/// Used only by the x86_64 `normalize_image` (the aarch64 path reads
/// the whole image via `fs::read`), so gate it off the aarch64 build.
#[cfg(target_arch = "x86_64")]
fn read_prefix(path: &Path, len: usize) -> Result<Vec<u8>> {
    let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = vec![0u8; len];
    let mut filled = 0;
    while filled < len {
        let n = f.read(&mut buf[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled);
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Module dependency closure
// ---------------------------------------------------------------------------

/// Resolve the transitive `.ko` dependency closure for `wanted`
/// modules, in load order (dependencies first).
///
/// `modules_dir` is the tree root (containing `lib/modules/<release>/`).
/// Module names are matched modprobe-style: dashes and underscores are
/// interchangeable and any `.ko[.comp]` suffix is ignored. Modules that
/// are built into the kernel (listed in `modules.builtin`) are skipped;
/// a name found in neither `modules.dep` nor `modules.builtin` is an
/// error. Compressed `.ko.{xz,zst,gz}` files are decompressed in place
/// and the returned paths point at the raw `.ko` (the guest loads it
/// via `finit_module`).
pub fn module_closure(
    modules_dir: &Path,
    kernel_release: &str,
    wanted: &[&str],
) -> Result<Vec<PathBuf>> {
    let base = modules_dir.join(format!("lib/modules/{kernel_release}"));
    let dep_text = fs::read_to_string(base.join("modules.dep"))
        .with_context(|| format!("read modules.dep for {kernel_release}"))?;

    let mut name_to_path: HashMap<String, String> = HashMap::new();
    let mut path_deps: HashMap<String, Vec<String>> = HashMap::new();
    for line in dep_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (key, rest) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("malformed modules.dep line: {line:?}"))?;
        let key = key.trim().to_string();
        name_to_path.insert(module_key(&key), key.clone());
        path_deps.insert(
            key,
            rest.split_whitespace().map(|s| s.to_string()).collect(),
        );
    }

    let mut builtin: HashSet<String> = HashSet::new();
    if let Ok(text) = fs::read_to_string(base.join("modules.builtin")) {
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() {
                builtin.insert(module_key(line));
            }
        }
    }

    let mut ordered: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    for name in wanted {
        let key = module_key(name);
        if builtin.contains(&key) {
            continue;
        }
        let path = name_to_path.get(&key).ok_or_else(|| {
            anyhow!(
                "module `{name}` not found in modules.dep or modules.builtin for {kernel_release}"
            )
        })?;
        visit_deps(path, &path_deps, &mut visited, &mut ordered);
    }

    ordered
        .into_iter()
        .map(|rel| decompress_module(&base.join(&rel)))
        .collect()
}

/// Post-order DFS over the dependency graph so a module's dependencies
/// precede it and each module appears once.
fn visit_deps(
    path: &str,
    deps: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    ordered: &mut Vec<String>,
) {
    if !visited.insert(path.to_string()) {
        return;
    }
    if let Some(children) = deps.get(path) {
        for dep in children {
            visit_deps(dep, deps, visited, ordered);
        }
    }
    ordered.push(path.to_string());
}

/// modprobe-style module key: basename with any `.ko[.comp]` suffix
/// stripped and dashes folded to underscores.
fn module_key(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    let stem = base
        .strip_suffix(".ko.xz")
        .or_else(|| base.strip_suffix(".ko.zst"))
        .or_else(|| base.strip_suffix(".ko.gz"))
        .or_else(|| base.strip_suffix(".ko"))
        .unwrap_or(base);
    stem.replace('-', "_")
}

/// Decompress an in-memory buffer with the codec named by `codec`,
/// accepting both the file-extension spelling (`gz`/`zst`/`xz`) and the
/// zboot `COMP_TYPE` spelling (`gzip`/`zstd`); an unrecognized codec is
/// an error naming it. Shares the flate2/zstd/xz2 stack the rest of the
/// module already streams `.ko` and archive payloads through.
fn decompress_bytes(codec: &str, data: &[u8]) -> Result<Vec<u8>> {
    let mut reader: Box<dyn Read> = match codec {
        "gz" | "gzip" => Box::new(flate2::read::GzDecoder::new(data)),
        "zst" | "zstd" => Box::new(zstd::stream::read::Decoder::new(data)?),
        "xz" => Box::new(xz2::read::XzDecoder::new(data)),
        other => bail!("unsupported compression {other:?}"),
    };
    let mut out = Vec::new();
    reader
        .read_to_end(&mut out)
        .with_context(|| format!("decompress {codec} stream"))?;
    Ok(out)
}

/// Decompress a compressed `.ko` in place (writing the raw `.ko`
/// beside it) and return the path to the raw module; an already-raw
/// `.ko` is returned unchanged.
fn decompress_module(path: &Path) -> Result<PathBuf> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let decoder: Option<Box<dyn Read>> = match ext {
        "xz" => Some(Box::new(xz2::read::XzDecoder::new(fs::File::open(path)?))),
        "zst" => Some(Box::new(zstd::stream::read::Decoder::new(fs::File::open(
            path,
        )?)?)),
        "gz" => Some(Box::new(flate2::read::GzDecoder::new(fs::File::open(
            path,
        )?))),
        _ => None,
    };
    let Some(mut decoder) = decoder else {
        return Ok(path.to_path_buf());
    };
    let out = path.with_extension("");
    let mut buf = Vec::new();
    decoder
        .read_to_end(&mut buf)
        .with_context(|| format!("decompress module {}", path.display()))?;
    fs::write(&out, &buf).with_context(|| format!("write {}", out.display()))?;
    Ok(out)
}

#[cfg(test)]
#[path = "extract_tests.rs"]
mod tests;
