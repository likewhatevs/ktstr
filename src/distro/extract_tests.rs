//! Tests for [`super`] (the distro package `extract` module).
//!
//! Source-shared into `extract.rs` via `#[path]` so this becomes the
//! `extract::tests` submodule and `use super::*` reaches the module's
//! private helpers (`sanitize_relative`, `normalize_arm64_bytes`,
//! `extract_tar`, `module_key`, …).
//!
//! Fixtures are synthesized at test time — a minimal rpm is built with
//! the `rpm` crate's builder and a minimal deb is hand-assembled from
//! `ar` + `tar` + a vendored compressor — so no binary blobs are
//! checked into the tree and no network access is required.

use super::*;
use std::io::Write;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Kernel-image bytes carrying the valid boot magic for the *host*
/// architecture, so `normalize_image` accepts the fixture wherever the
/// test suite runs.
fn host_image_bytes() -> Vec<u8> {
    let mut v = vec![0u8; 0x400];
    match crate::fetch::arch_info().0 {
        "x86_64" => {
            v[BZIMAGE_MAGIC_OFFSET..BZIMAGE_MAGIC_OFFSET + 4].copy_from_slice(&BZIMAGE_MAGIC)
        }
        _ => v[ARM64_MAGIC_OFFSET..ARM64_MAGIC_OFFSET + 4].copy_from_slice(&ARM64_MAGIC),
    }
    v
}

/// Host rpm arch spelling.
fn host_rpm_arch() -> &'static str {
    crate::fetch::arch_info().0
}

/// Host deb arch spelling (the `amd64`/`arm64` alias).
fn host_deb_arch() -> &'static str {
    match crate::fetch::arch_info().0 {
        "x86_64" => "amd64",
        _ => "arm64",
    }
}

/// The arch spelling that does NOT match the host, for mismatch tests.
fn foreign_rpm_arch() -> &'static str {
    match crate::fetch::arch_info().0 {
        "x86_64" => "aarch64",
        _ => "x86_64",
    }
}

fn build_rpm(
    out: &Path,
    name: &str,
    arch: &str,
    files: &[(&str, Vec<u8>)],
    compression: rpm::CompressionType,
) {
    let mut builder =
        rpm::PackageBuilder::new(name, "1.0.0", "GPL-2.0-only", arch, "ktstr fixture");
    builder.using_config(rpm::BuildConfig::default().compression(compression));
    for (path, content) in files {
        builder
            .with_file_contents(
                content.clone(),
                rpm::FileOptions::new(*path).permissions(0o644),
            )
            .expect("add rpm file");
    }
    let pkg = builder.build().expect("build rpm");
    pkg.write_file(out).expect("write rpm");
}

fn xz_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = xz2::write::XzEncoder::new(Vec::new(), 6);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn gz_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn zst_compress(data: &[u8]) -> Vec<u8> {
    zstd::stream::encode_all(data, 3).unwrap()
}

/// Build an uncompressed tar from `(path, bytes)` entries.
fn make_tar(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (path, data) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        builder.append_data(&mut header, path, *data).unwrap();
    }
    builder.into_inner().unwrap()
}

fn push_ar_member(out: &mut Vec<u8>, name: &str, data: &[u8]) {
    let header = format!(
        "{name:<16}{mtime:<12}{uid:<6}{gid:<6}{mode:<8}{size:<10}`\n",
        mtime = 0,
        uid = 0,
        gid = 0,
        mode = "100644",
        size = data.len(),
    );
    assert_eq!(header.len(), 60, "ar member header must be 60 bytes");
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    if data.len() % 2 == 1 {
        out.push(b'\n');
    }
}

/// Assemble a `.deb`: `ar(debian-binary, control.tar.gz, data.tar.<ext>)`.
fn build_deb(out: &Path, arch: &str, data_ext: &str, data_entries: &[(&str, &[u8])]) {
    let control_body = format!("Package: fixture\nArchitecture: {arch}\nVersion: 1.0.0\n");
    let control_tar = make_tar(&[("./control", control_body.as_bytes())]);
    let control_gz = gz_compress(&control_tar);

    let data_tar = make_tar(data_entries);
    let (data_name, data_bytes) = match data_ext {
        "xz" => ("data.tar.xz", xz_compress(&data_tar)),
        "gz" => ("data.tar.gz", gz_compress(&data_tar)),
        "zst" => ("data.tar.zst", zst_compress(&data_tar)),
        _ => panic!("unsupported fixture data ext {data_ext}"),
    };

    let mut ar = Vec::new();
    ar.extend_from_slice(b"!<arch>\n");
    push_ar_member(&mut ar, "debian-binary", b"2.0\n");
    push_ar_member(&mut ar, "control.tar.gz", &control_gz);
    push_ar_member(&mut ar, data_name, &data_bytes);
    fs::write(out, &ar).unwrap();
}

/// Assemble a pacman `.pkg.tar.zst`: one zstd tarball whose first
/// member is `.PKGINFO` (carrying the `arch = ` field) followed by the
/// payload tree — the shape of a live `linux-neptune` package.
fn build_pacman(out: &Path, arch: &str, data_entries: &[(&str, &[u8])]) {
    let pkginfo =
        format!("# Generated by makepkg\npkgname = fixture\npkgver = 1.0.0-1\narch = {arch}\n");
    let mut entries: Vec<(&str, &[u8])> = vec![(".PKGINFO", pkginfo.as_bytes())];
    entries.extend_from_slice(data_entries);
    let tar = make_tar(&entries);
    fs::write(out, zst_compress(&tar)).unwrap();
}

/// A hand-built tar whose sole member escapes via `../evil`, bypassing
/// the `tar` crate's own path guard on the write side.
fn evil_tar() -> Vec<u8> {
    let mut block = [0u8; 512];
    let name = b"../evil";
    block[..name.len()].copy_from_slice(name);
    block[100..108].copy_from_slice(b"0000644\0");
    block[108..116].copy_from_slice(b"0000000\0");
    block[116..124].copy_from_slice(b"0000000\0");
    block[124..136].copy_from_slice(b"00000000000\0"); // size 0
    block[136..148].copy_from_slice(b"00000000000\0");
    block[156] = b'0'; // regular file
    block[257..263].copy_from_slice(b"ustar\0");
    block[263..265].copy_from_slice(b"00");
    for b in &mut block[148..156] {
        *b = b' ';
    }
    let sum: u32 = block.iter().map(|&b| b as u32).sum();
    block[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());

    let mut out = block.to_vec();
    out.extend_from_slice(&[0u8; 512]);
    out.extend_from_slice(&[0u8; 512]);
    out
}

const REL: &str = "6.14.0-1.fixture";

fn kernel_rpm_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (concat_leak("lib/modules/", "/vmlinuz"), host_image_bytes()),
        (
            concat_leak("lib/modules/", "/config"),
            b"CONFIG_FIXTURE=y\n".to_vec(),
        ),
        (
            concat_leak("lib/modules/", "/System.map"),
            b"0 T _stext\n".to_vec(),
        ),
        (
            concat_leak("lib/modules/", "/modules.dep"),
            b"kernel/x.ko:\n".to_vec(),
        ),
        (
            concat_leak("lib/modules/", "/modules.builtin"),
            b"kernel/ext4.ko\n".to_vec(),
        ),
        (
            concat_leak("lib/modules/", "/kernel/x.ko"),
            b"fake-ko\n".to_vec(),
        ),
    ]
}

/// Build an absolute install path `"/{prefix}{REL}{suffix}"`, leaking it
/// to `&'static str` so it fits the `FileOptions::new(&str)` fixtures.
fn concat_leak(prefix: &str, suffix: &str) -> &'static str {
    Box::leak(format!("/{prefix}{REL}{suffix}").into_boxed_str())
}

// ---------------------------------------------------------------------------
// End-to-end extraction
// ---------------------------------------------------------------------------

#[test]
fn single_rpm_extraction_zstd() {
    let tmp = TempDir::new().unwrap();
    let rpm = tmp.path().join("kernel-core.rpm");
    build_rpm(
        &rpm,
        "kernel-core",
        host_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Zstd,
    );

    let dest = tmp.path().join("out");
    let k = extract_kernel_packages(&[&rpm], &dest).expect("extract");

    assert_eq!(k.kernel_release, REL);
    assert!(k.image.exists(), "image must exist");
    assert_eq!(fs::read(&k.image).unwrap(), host_image_bytes());
    assert!(k.config.is_some(), "config must be found");
    assert!(k.system_map.is_some(), "System.map must be found");
    assert!(k.modules_dir.is_some(), "modules_dir must be found");
    assert!(k.vmlinux.is_none(), "no debuginfo package supplied");
}

#[test]
fn single_rpm_extraction_gzip() {
    // Exercise the gzip payload path too (Gzip is the builder default,
    // but pin it explicitly).
    let tmp = TempDir::new().unwrap();
    let rpm = tmp.path().join("kernel-core.rpm");
    build_rpm(
        &rpm,
        "kernel-core",
        host_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Gzip,
    );
    let dest = tmp.path().join("out");
    let k = extract_kernel_packages(&[&rpm], &dest).expect("extract");
    assert_eq!(k.kernel_release, REL);
    assert!(k.image.exists());
}

#[test]
fn multi_deb_merge() {
    let tmp = TempDir::new().unwrap();
    let vmlinuz = format!("./boot/vmlinuz-{REL}");
    let config = format!("./boot/config-{REL}");
    let sysmap = format!("./boot/System.map-{REL}");
    let dep = format!("./lib/modules/{REL}/modules.dep");
    let ko = format!("./lib/modules/{REL}/kernel/foo.ko");

    let image = host_image_bytes();
    let image_deb = tmp.path().join("linux-image.deb");
    build_deb(
        &image_deb,
        host_deb_arch(),
        "xz",
        &[
            (vmlinuz.as_str(), image.as_slice()),
            (config.as_str(), b"CONFIG_FIXTURE=y\n"),
            (sysmap.as_str(), b"0 T _stext\n"),
        ],
    );

    let modules_deb = tmp.path().join("linux-modules.deb");
    build_deb(
        &modules_deb,
        host_deb_arch(),
        "zst",
        &[
            (dep.as_str(), b"kernel/foo.ko:\n"),
            (ko.as_str(), b"fake-ko\n"),
        ],
    );

    let dest = tmp.path().join("out");
    let k = extract_kernel_packages(&[&image_deb, &modules_deb], &dest).expect("extract");

    assert_eq!(k.kernel_release, REL);
    assert_eq!(
        fs::read(&k.image).unwrap(),
        image,
        "image from linux-image deb"
    );
    assert!(k.config.is_some());
    assert!(k.system_map.is_some());
    let modules_dir = k.modules_dir.expect("modules from linux-modules deb");
    assert!(
        modules_dir
            .join(format!("lib/modules/{REL}/kernel/foo.ko"))
            .exists()
    );
}

#[test]
fn usrmerge_deb_layout_is_unified() {
    // Newer Ubuntu debs ship modules under usr/lib/modules/<rel>/
    // (verified against a real linux-modules 7.1.0-5 deb); the
    // extractor must bridge that onto the classic lib/modules path.
    let tmp = TempDir::new().unwrap();
    let vmlinuz = format!("./boot/vmlinuz-{REL}");
    let dep = format!("./usr/lib/modules/{REL}/modules.dep");
    let ko = format!("./usr/lib/modules/{REL}/kernel/foo.ko");

    let image = host_image_bytes();
    let deb = tmp.path().join("linux-modules.deb");
    build_deb(
        &deb,
        host_deb_arch(),
        "zst",
        &[
            (vmlinuz.as_str(), image.as_slice()),
            (dep.as_str(), b"kernel/foo.ko:\n"),
            (ko.as_str(), b"fake-ko\n"),
        ],
    );

    let dest = tmp.path().join("out");
    let k = extract_kernel_packages(&[&deb], &dest).expect("extract");
    assert_eq!(k.kernel_release, REL);
    let root = k.modules_dir.expect("usrmerge modules must be discovered");
    let closure =
        module_closure(&root, REL, &["foo"]).expect("closure through lib/modules symlink");
    assert_eq!(closure.len(), 1);
    assert!(closure[0].exists());
}

#[test]
fn pacman_extraction_usrmerge_layout() {
    // linux-neptune shape: usrmerge module dir carrying vmlinuz +
    // modules.builtin, one zstd-compressed module, no config, no
    // modules.dep.
    let tmp = TempDir::new().unwrap();
    let pkg = tmp
        .path()
        .join("linux-neptune-618-1.0.0-1-x86_64.pkg.tar.zst");
    let rel = "6.18.33-valve2-1.1-neptune-618-gabc";
    let image = host_image_bytes();
    let ko = zst_compress(b"fake module");
    let builtin = b"kernel/drivers/virtio/virtio.ko
";
    let entries: Vec<(String, &[u8])> = vec![
        (format!("usr/lib/modules/{rel}/vmlinuz"), image.as_slice()),
        (format!("usr/lib/modules/{rel}/modules.builtin"), builtin),
        (
            format!("usr/lib/modules/{rel}/kernel/drivers/virtio/virtio_mmio.ko.zst"),
            ko.as_slice(),
        ),
    ];
    let refs: Vec<(&str, &[u8])> = entries.iter().map(|(p, d)| (p.as_str(), *d)).collect();
    build_pacman(&pkg, host_rpm_arch(), &refs);

    assert_eq!(package_arch(&pkg).unwrap(), host_rpm_arch());

    let dest = tmp.path().join("out");
    let extracted = extract_kernel_packages(&[&pkg], &dest).unwrap();
    assert_eq!(extracted.kernel_release, rel);
    // usrmerge bridging: the image resolves through the classic
    // lib/modules path via the compatibility symlink.
    assert!(extracted.image.exists());
    assert!(extracted.config.is_none(), "pacman kernels ship no config");
    assert!(extracted.modules_dir.is_some());
    assert!(
        dest.join(format!("lib/modules/{rel}/modules.builtin"))
            .exists(),
        "classic path must reach the usrmerge tree"
    );
}

#[test]
fn pacman_arch_mismatch_is_rejected() {
    let tmp = TempDir::new().unwrap();
    let pkg = tmp.path().join("foreign.pkg.tar.zst");
    build_pacman(
        &pkg,
        foreign_rpm_arch(),
        &[("usr/lib/modules/6.1.0/vmlinuz", b"x")],
    );
    let arch = package_arch(&pkg).unwrap();
    let err = ensure_arch_matches_host(&arch, &pkg).unwrap_err();
    assert!(format!("{err}").contains("this host is"));
}

#[test]
fn pacman_kind_detected_by_name_and_magic() {
    let tmp = TempDir::new().unwrap();
    // Magic fallback: a zstd stream under a name without the
    // .pkg.tar.zst suffix still classifies as pacman.
    let odd = tmp.path().join("mystery.bin");
    fs::write(
        &odd,
        zst_compress(&make_tar(&[(
            ".PKGINFO",
            b"arch = x86_64
",
        )])),
    )
    .unwrap();
    assert!(matches!(package_kind(&odd).unwrap(), PackageKind::Pacman));
    // Name-based: the full compound suffix wins even though
    // Path::extension only sees `zst`.
    let named = tmp.path().join("k.pkg.tar.zst");
    fs::write(&named, b"garbage-not-zstd").unwrap();
    assert!(matches!(package_kind(&named).unwrap(), PackageKind::Pacman));
}

#[test]
fn debuginfo_vmlinux_pickup() {
    let tmp = TempDir::new().unwrap();
    let kernel = tmp.path().join("kernel-core.rpm");
    build_rpm(
        &kernel,
        "kernel-core",
        host_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Zstd,
    );

    let debug = tmp.path().join("kernel-debuginfo.rpm");
    let vmlinux_path = concat_leak("usr/lib/debug/lib/modules/", "/vmlinux");
    build_rpm(
        &debug,
        "kernel-debuginfo",
        host_rpm_arch(),
        &[(vmlinux_path, b"\x7fELF-fake-vmlinux".to_vec())],
        rpm::CompressionType::Zstd,
    );

    let dest = tmp.path().join("out");
    let k = extract_kernel_packages(&[&kernel, &debug], &dest).expect("extract");
    let vmlinux = k.vmlinux.expect("vmlinux from debuginfo");
    assert_eq!(fs::read(vmlinux).unwrap(), b"\x7fELF-fake-vmlinux");
}

/// The distro acquire path extracts the kernel packages first (gating
/// before any debuginfo download), then runs a SECOND extraction pass
/// for the debuginfo via `extract_vmlinux` into the same dest. Pin that
/// the second pass merges onto the already-populated tree and recovers
/// vmlinux at the known release.
#[test]
fn extract_vmlinux_second_pass_merges_into_existing_tree() {
    let tmp = TempDir::new().unwrap();
    let kernel = tmp.path().join("kernel-core.rpm");
    build_rpm(
        &kernel,
        "kernel-core",
        host_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Zstd,
    );
    let dest = tmp.path().join("out");
    let k = extract_kernel_packages(&[&kernel], &dest).expect("extract kernel");
    assert!(
        k.vmlinux.is_none(),
        "no debuginfo supplied to the first pass"
    );

    let debug = tmp.path().join("kernel-debuginfo.rpm");
    let vmlinux_path = concat_leak("usr/lib/debug/lib/modules/", "/vmlinux");
    build_rpm(
        &debug,
        "kernel-debuginfo",
        host_rpm_arch(),
        &[(vmlinux_path, b"\x7fELF-fake-vmlinux".to_vec())],
        rpm::CompressionType::Zstd,
    );
    let vmlinux = extract_vmlinux(&[&debug], &dest, &k.kernel_release)
        .expect("second pass")
        .expect("vmlinux recovered");
    assert_eq!(fs::read(&vmlinux).unwrap(), b"\x7fELF-fake-vmlinux");
    // The first pass's artifacts must still be present under the tree.
    assert!(k.image.exists(), "kernel image survives the second pass");
}

#[test]
fn arch_mismatch_is_rejected() {
    let tmp = TempDir::new().unwrap();
    let rpm = tmp.path().join("kernel-core.rpm");
    build_rpm(
        &rpm,
        "kernel-core",
        foreign_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Zstd,
    );
    let dest = tmp.path().join("out");
    let err = extract_kernel_packages(&[&rpm], &dest).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("host is") || msg.contains("targets"),
        "arch mismatch error should name the arch: {msg}"
    );
}

#[test]
fn conflicting_releases_are_rejected() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("a.rpm");
    build_rpm(
        &a,
        "kernel-core",
        host_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Zstd,
    );
    // Second rpm ships a different release under lib/modules.
    let other = "/lib/modules/9.9.9-other/vmlinuz";
    let b = tmp.path().join("b.rpm");
    build_rpm(
        &b,
        "kernel-core-2",
        host_rpm_arch(),
        &[(other, host_image_bytes())],
        rpm::CompressionType::Zstd,
    );
    let dest = tmp.path().join("out");
    let err = extract_kernel_packages(&[&a, &b], &dest).unwrap_err();
    assert!(err.to_string().contains("disagree"), "{err}");
}

// ---------------------------------------------------------------------------
// Path hardening
// ---------------------------------------------------------------------------

#[test]
fn sanitize_relative_rejects_parent_dir() {
    assert!(sanitize_relative(Path::new("../evil")).is_err());
    assert!(sanitize_relative(Path::new("a/../../evil")).is_err());
    // Absolute anchors are stripped, not rejected.
    assert_eq!(
        sanitize_relative(Path::new("/lib/modules/x")).unwrap(),
        Some(PathBuf::from("lib/modules/x"))
    );
    assert_eq!(
        sanitize_relative(Path::new("./boot/vmlinuz")).unwrap(),
        Some(PathBuf::from("boot/vmlinuz"))
    );
    assert_eq!(sanitize_relative(Path::new("/")).unwrap(), None);
}

#[test]
fn extract_tar_rejects_traversal_entry() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("out");
    fs::create_dir_all(&dest).unwrap();
    let err = extract_tar(Cursor::new(evil_tar()), &dest).unwrap_err();
    assert!(err.to_string().contains(".."), "{err}");
    assert!(
        !tmp.path().join("evil").exists(),
        "traversal target must not be written outside dest"
    );
}

#[test]
fn resolve_within_root_containment() {
    // Relative link staying inside the tree resolves.
    assert!(resolve_within_root(Path::new("lib/modules"), Path::new("../firmware")).is_some());
    // Absolute target escapes.
    assert!(resolve_within_root(Path::new("lib/modules"), Path::new("/usr/src")).is_none());
    // Too many `..` climbs above the root.
    assert!(resolve_within_root(Path::new("lib"), Path::new("../../etc/passwd")).is_none());
}

#[test]
fn rpm_payload_decoder_handles_each_codec() {
    // The real Fedora kernel rpm ships an xz payload; pin that xz2 is
    // wired up (plus the gzip/zstd/none branches) without needing a
    // full fixture rpm per codec.
    let payload = b"cpio-payload-bytes";
    for (codec, bytes) in [
        (rpm::CompressionType::Xz, xz_compress(payload)),
        (rpm::CompressionType::Gzip, gz_compress(payload)),
        (rpm::CompressionType::Zstd, zst_compress(payload)),
        (rpm::CompressionType::None, payload.to_vec()),
    ] {
        let mut out = Vec::new();
        rpm_payload_decoder(codec, &bytes)
            .unwrap()
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, payload, "codec {codec:?}");
    }
}

// ---------------------------------------------------------------------------
// Image normalization
// ---------------------------------------------------------------------------

#[test]
fn arm64_gzip_image_is_decompressed_and_verified() {
    let mut image = vec![0u8; 0x100];
    image[ARM64_MAGIC_OFFSET..ARM64_MAGIC_OFFSET + 4].copy_from_slice(&ARM64_MAGIC);

    // gzip-wrapped Image round-trips to the raw bytes.
    let gz = gz_compress(&image);
    assert!(has_gzip_magic(&gz));
    assert_eq!(normalize_arm64_bytes(&gz).unwrap(), image);

    // A bare (uncompressed) Image passes through unchanged.
    assert_eq!(normalize_arm64_bytes(&image).unwrap(), image);

    // Missing magic is rejected (both raw and gzip-wrapped).
    let bad = vec![0u8; 0x100];
    assert!(normalize_arm64_bytes(&bad).is_err());
    assert!(normalize_arm64_bytes(&gz_compress(&bad)).is_err());
}

#[test]
fn magic_helpers() {
    let mut bz = vec![0u8; 0x400];
    bz[BZIMAGE_MAGIC_OFFSET..BZIMAGE_MAGIC_OFFSET + 4].copy_from_slice(&BZIMAGE_MAGIC);
    assert!(has_magic(&bz, BZIMAGE_MAGIC_OFFSET, &BZIMAGE_MAGIC));
    assert!(!has_magic(&[0u8; 4], BZIMAGE_MAGIC_OFFSET, &BZIMAGE_MAGIC));
}

// ---------------------------------------------------------------------------
// Module closure
// ---------------------------------------------------------------------------

/// Lay down a synthetic module tree and return its root.
fn make_module_tree(tmp: &Path) -> PathBuf {
    let base = tmp.join(format!("lib/modules/{REL}/kernel"));
    fs::create_dir_all(&base).unwrap();
    fs::write(base.join("a.ko"), b"a").unwrap();
    fs::write(base.join("b.ko"), b"b").unwrap();
    fs::write(base.join("c.ko"), b"c").unwrap();
    fs::write(base.join("my_mod.ko"), b"m").unwrap();
    fs::write(base.join("d.ko.xz"), xz_compress(b"decompressed-d")).unwrap();

    let modroot = tmp.join(format!("lib/modules/{REL}"));
    fs::write(
        modroot.join("modules.dep"),
        "kernel/a.ko: kernel/b.ko kernel/c.ko\n\
         kernel/b.ko: kernel/c.ko\n\
         kernel/c.ko:\n\
         kernel/my_mod.ko:\n\
         kernel/d.ko.xz: kernel/c.ko\n",
    )
    .unwrap();
    fs::write(modroot.join("modules.builtin"), "kernel/fs/ext4/ext4.ko\n").unwrap();
    tmp.to_path_buf()
}

#[test]
fn module_closure_orders_dependencies_first() {
    let tmp = TempDir::new().unwrap();
    let root = make_module_tree(tmp.path());
    let closure = module_closure(&root, REL, &["a"]).unwrap();
    let names: Vec<String> = closure
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec!["c.ko", "b.ko", "a.ko"],
        "deps precede dependents, deduped"
    );
}

#[test]
fn module_closure_skips_builtins() {
    let tmp = TempDir::new().unwrap();
    let root = make_module_tree(tmp.path());
    // ext4 is builtin; requesting it yields nothing.
    assert!(module_closure(&root, REL, &["ext4"]).unwrap().is_empty());
}

#[test]
fn module_closure_errors_on_missing_module() {
    let tmp = TempDir::new().unwrap();
    let root = make_module_tree(tmp.path());
    let err = module_closure(&root, REL, &["does_not_exist"]).unwrap_err();
    assert!(err.to_string().contains("does_not_exist"), "{err}");
}

#[test]
fn module_closure_dash_underscore_equivalence() {
    let tmp = TempDir::new().unwrap();
    let root = make_module_tree(tmp.path());
    // File is my_mod.ko; request it as my-mod.
    let closure = module_closure(&root, REL, &["my-mod"]).unwrap();
    assert_eq!(closure.len(), 1);
    assert_eq!(closure[0].file_name().unwrap(), "my_mod.ko");
}

#[test]
fn module_closure_decompresses_compressed_ko() {
    let tmp = TempDir::new().unwrap();
    let root = make_module_tree(tmp.path());
    let closure = module_closure(&root, REL, &["d"]).unwrap();
    // Closure is [c.ko, d.ko]; the compressed d.ko.xz is decompressed.
    let d = closure.last().unwrap();
    assert_eq!(d.file_name().unwrap(), "d.ko");
    assert_eq!(fs::read(d).unwrap(), b"decompressed-d");
}

#[test]
fn module_key_normalizes() {
    assert_eq!(module_key("kernel/fs/xfs/xfs.ko.xz"), "xfs");
    assert_eq!(module_key("kernel/drivers/foo-bar.ko"), "foo_bar");
    assert_eq!(module_key("nvme_core"), "nvme_core");
}

// ---------------------------------------------------------------------------
// Arch helpers
// ---------------------------------------------------------------------------

#[test]
fn canonical_arch_maps_aliases() {
    assert_eq!(canonical_arch("amd64"), Some("x86_64"));
    assert_eq!(canonical_arch("x86_64"), Some("x86_64"));
    assert_eq!(canonical_arch("arm64"), Some("aarch64"));
    assert_eq!(canonical_arch("aarch64"), Some("aarch64"));
    assert_eq!(canonical_arch("riscv64"), None);
}

#[test]
fn package_arch_reads_rpm_and_deb() {
    let tmp = TempDir::new().unwrap();
    let rpm = tmp.path().join("k.rpm");
    build_rpm(
        &rpm,
        "kernel-core",
        host_rpm_arch(),
        &kernel_rpm_files(),
        rpm::CompressionType::Zstd,
    );
    assert_eq!(package_arch(&rpm).unwrap(), host_rpm_arch());

    let deb = tmp.path().join("k.deb");
    build_deb(
        &deb,
        host_deb_arch(),
        "gz",
        &[(concat_deb_path().as_str(), b"x")],
    );
    assert_eq!(package_arch(&deb).unwrap(), host_deb_arch());
}

fn concat_deb_path() -> String {
    format!("./lib/modules/{REL}/kernel/foo.ko")
}
