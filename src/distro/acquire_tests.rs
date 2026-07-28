//! Unit tests for distro/local-package acquisition: cache-key
//! composition, the config gate's classification, boot-module name
//! selection, and the `modules/`-dir boot-wiring discovery. All offline
//! — no network, no real packages.

use super::*;
use crate::cache::{boot_modules_for_image, ordered_boot_modules_in};
use crate::distro::repo::{PackageRef, ResolvedDistroKernel};

fn pkg(name: &str, sha: &str) -> PackageRef {
    PackageRef {
        name: name.to_string(),
        version: "1.0-1".to_string(),
        url: format!("https://example/{name}.rpm"),
        alternate_urls: Vec::new(),
        sha256: sha.to_string(),
        size: None,
    }
}

fn resolved(distro: &str, release: &str, arch: &str, primary_sha: &str) -> ResolvedDistroKernel {
    ResolvedDistroKernel {
        distro: distro.to_string(),
        kernel_release: release.to_string(),
        arch: arch.to_string(),
        packages: vec![
            pkg("kernel-core", primary_sha),
            pkg("kernel-modules", "ffff"),
        ],
        debuginfo: vec![pkg("kernel-debuginfo", "aaaa")],
    }
}

#[test]
fn package_download_origins_fail_over_in_declared_order() {
    let mut package = pkg("kernel-debuginfo", "aaaa");
    package.url = "https://ddebs.ubuntu.com/package.ddeb".to_string();
    package.alternate_urls =
        vec!["https://launchpad.net/ubuntu/+archive/primary/+files/package.ddeb".to_string()];
    let mut attempts = Vec::new();

    let result = try_package_urls_with(
        &package,
        || false,
        |url| {
            attempts.push(url.to_string());
            if url == package.url.as_str() {
                bail!("primary unavailable");
            }
            Ok(17)
        },
    )
    .unwrap();

    assert_eq!(result, 17);
    assert_eq!(
        attempts,
        [package.url.clone(), package.alternate_urls[0].clone()]
    );
}

// -- cache-key composition --

#[test]
fn distro_key_shape_and_no_kconfig_suffix() {
    let r = resolved(
        "fedora44",
        "7.1.3-200.fc44.x86_64",
        "x86_64",
        "0123456789abcdef0123456789abcdef",
    );
    let key = distro_cache_key(&r).unwrap();
    assert_eq!(
        key,
        "distro-fedora44-7.1.3-200.fc44.x86_64-x86_64-pkg0123456789ab"
    );
    // Deliberately NO kconfig-hash suffix — ktstr did not build this.
    assert!(
        !key.contains("-kc"),
        "distro key must carry no kc{{...}} suffix: {key}"
    );
}

#[test]
fn distro_key_uses_primary_package_sha_first12() {
    let r = resolved(
        "al2023",
        "6.12.0-1.al2023",
        "aarch64",
        "deadbeefcafef00d1234",
    );
    let key = distro_cache_key(&r).unwrap();
    assert!(key.starts_with("distro-al2023-6.12.0-1.al2023-aarch64-pkgdeadbeefcafe"));
}

#[test]
fn distro_key_sanitizes_out_of_alphabet_bytes() {
    // A hypothetical tag with a `/` must fold to `_` so the key stays
    // valid for `validate_cache_key`.
    let mut r = resolved("weird/tag", "6.1", "x86_64", "abcdef012345");
    r.distro = "weird/tag".to_string();
    let key = distro_cache_key(&r).unwrap();
    assert!(
        !key.contains('/'),
        "key must not contain a path separator: {key}"
    );
    assert!(key.contains("weird_tag"));
}

#[test]
fn local_key_shape_is_order_independent() {
    let a = local_cache_key(&["1111".to_string(), "2222".to_string()], "x86_64");
    let b = local_cache_key(&["2222".to_string(), "1111".to_string()], "x86_64");
    assert_eq!(a, b, "local key must not depend on file order");
    assert!(a.starts_with("pkg-"));
    assert!(a.ends_with("-x86_64"));
    // pkg- (4) + 12 hex + -x86_64
    assert_eq!(a.len(), 4 + 12 + "-x86_64".len());
    assert!(!a.contains("kc"), "local key carries no kconfig suffix");
}

// -- config-gate classification --

fn write_config(lines: &[&str]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    for l in lines {
        writeln!(f, "{l}").unwrap();
    }
    f.flush().unwrap();
    f
}

/// [`ExtractedKernel`] shell for driving [`gate_config`] with a chosen
/// config / module-tree presence; the label surfaces as
/// `kernel_release` in gate errors.
fn extracted(label: &str, config: Option<&Path>, modules_dir: Option<&Path>) -> ExtractedKernel {
    ExtractedKernel {
        kernel_release: label.to_string(),
        image: PathBuf::from("/nonexistent/bzImage"),
        config: config.map(Path::to_path_buf),
        system_map: None,
        modules_dir: modules_dir.map(Path::to_path_buf),
        vmlinux: None,
    }
}

#[test]
fn gate_all_builtin_selects_no_modules() {
    let cfg = write_config(&[
        "CONFIG_VIRTIO=y",
        "CONFIG_VIRTIO_MMIO=y",
        "CONFIG_VIRTIO_CONSOLE=y",
    ]);
    let gate = gate_config(&extracted("test", Some(cfg.path()), None)).unwrap();
    assert!(
        gate.critical.is_empty() && gate.feature.is_empty(),
        "all builtin ⇒ no modules to load: {:?}/{:?}",
        gate.critical,
        gate.feature
    );
}

#[test]
fn gate_modular_boot_critical_selected() {
    let cfg = write_config(&[
        "CONFIG_VIRTIO=y",
        "CONFIG_VIRTIO_MMIO=m",
        "CONFIG_VIRTIO_CONSOLE=m",
    ]);
    let gate = gate_config(&extracted("test", Some(cfg.path()), None)).unwrap();
    assert!(gate.critical.contains(&"virtio_mmio"));
    assert!(gate.critical.contains(&"virtio_console"));
    assert!(
        !gate.critical.contains(&"virtio"),
        "=y virtio must not be requested as a module"
    );
}

#[test]
fn gate_missing_boot_critical_hard_fails() {
    // VIRTIO_MMIO absent entirely (not even `is not set`).
    let cfg = write_config(&["CONFIG_VIRTIO=y", "CONFIG_VIRTIO_CONSOLE=y"]);
    let err = gate_config(&extracted("badkernel", Some(cfg.path()), None)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("CONFIG_VIRTIO_MMIO"),
        "message must name the option: {msg}"
    );
    assert!(
        msg.contains("badkernel"),
        "message must name the kernel: {msg}"
    );
}

#[test]
fn gate_disabled_boot_critical_hard_fails() {
    let cfg = write_config(&[
        "CONFIG_VIRTIO=y",
        "CONFIG_VIRTIO_MMIO=y",
        "# CONFIG_VIRTIO_CONSOLE is not set",
    ]);
    let err = gate_config(&extracted("k", Some(cfg.path()), None)).unwrap_err();
    assert!(format!("{err}").contains("CONFIG_VIRTIO_CONSOLE"));
}

#[test]
fn gate_modular_feature_selected_absent_feature_warns_not_fails() {
    let cfg = write_config(&[
        "CONFIG_VIRTIO=y",
        "CONFIG_VIRTIO_MMIO=y",
        "CONFIG_VIRTIO_CONSOLE=y",
        "CONFIG_VIRTIO_BLK=m",
        // VIRTIO_NET / BTRFS_FS absent → warn, not fail.
    ]);
    let gate = gate_config(&extracted("test", Some(cfg.path()), None)).unwrap();
    assert!(
        gate.feature.contains(&"virtio_blk"),
        "modular feature must be bundled: {:?}",
        gate.feature
    );
}

#[test]
fn gate_no_config_and_no_tree_skips_and_selects_nothing() {
    let gate = gate_config(&extracted("test", None, None)).unwrap();
    assert!(gate.critical.is_empty() && gate.feature.is_empty());
}

/// No `.config` but a module tree present (pacman kernels): the gate
/// lists EVERY boot-critical and feature driver so the tree decides —
/// `select_boot_modules`' closure run skips builtins per
/// `modules.builtin`, loads shipped `.ko`s, and hard-errors on a
/// boot-critical driver found in neither.
#[test]
fn gate_no_config_with_tree_defers_to_module_tree() {
    let dir = tempfile::TempDir::new().unwrap();
    let gate = gate_config(&extracted("test", None, Some(dir.path()))).unwrap();
    assert_eq!(
        gate.critical,
        vec!["virtio", "virtio_mmio", "virtio_console"]
    );
    assert!(gate.feature.contains(&"virtio_net"));
    assert!(gate.feature.contains(&"btrfs"));
}

/// End-to-end tree gate over a synthetic module tree shaped like
/// linux-neptune: virtio + virtio_console builtin, virtio_mmio a
/// shipped module — the critical closure must resolve to exactly the
/// virtio_mmio `.ko` and skip the builtins.
#[test]
fn tree_gate_resolves_neptune_shaped_module_split() {
    let dir = tempfile::TempDir::new().unwrap();
    let rel = "6.18.33-valve2";
    let base = dir.path().join(format!("lib/modules/{rel}"));
    fs::create_dir_all(base.join("kernel/drivers/virtio")).unwrap();
    fs::write(
        base.join("kernel/drivers/virtio/virtio_mmio.ko"),
        b"fake-ko",
    )
    .unwrap();
    fs::write(
        base.join("modules.dep"),
        "kernel/drivers/virtio/virtio_mmio.ko:
",
    )
    .unwrap();
    fs::write(
        base.join("modules.builtin"),
        "kernel/drivers/virtio/virtio.ko
         kernel/drivers/virtio/virtio_ring.ko
         kernel/drivers/char/virtio_console.ko
         kernel/drivers/block/virtio_blk.ko
",
    )
    .unwrap();

    let mut ext = extracted(rel, None, Some(dir.path()));
    ext.kernel_release = rel.to_string();
    let gate = gate_config(&ext).unwrap();
    let mods = select_boot_modules(&ext, &gate).unwrap();
    let names: Vec<String> = mods
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.contains(&"virtio_mmio.ko".to_string()),
        "shipped =m driver must be selected: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains("virtio_console")),
        "builtin drivers must not be selected: {names:?}"
    );
}

// -- boot-module selection --

#[test]
fn select_boot_modules_empty_wanted_is_empty() {
    let extracted = ExtractedKernel {
        kernel_release: "6.1.0".to_string(),
        image: PathBuf::from("/nonexistent/bzImage"),
        config: None,
        system_map: None,
        modules_dir: None,
        vmlinux: None,
    };
    // No modules wanted must not require a module tree.
    let gate = GateResult {
        critical: Vec::new(),
        feature: Vec::new(),
    };
    let out = select_boot_modules(&extracted, &gate).unwrap();
    assert!(out.is_empty());
}

#[test]
fn select_boot_modules_without_tree_errors() {
    let extracted = ExtractedKernel {
        kernel_release: "6.1.0".to_string(),
        image: PathBuf::from("/nonexistent/bzImage"),
        config: None,
        system_map: None,
        modules_dir: None,
        vmlinux: None,
    };
    let gate = GateResult {
        critical: vec!["virtio_mmio"],
        feature: Vec::new(),
    };
    let err = select_boot_modules(&extracted, &gate).unwrap_err();
    assert!(format!("{err}").contains("no module tree"));
}

// -- modules/-dir boot-wiring discovery (synthetic cache entry) --

#[test]
fn boot_modules_discovered_in_load_order() {
    let dir = tempfile::TempDir::new().unwrap();
    fs::write(dir.path().join("bzImage"), b"img").unwrap();
    let mods = dir.path().join("modules");
    fs::create_dir_all(&mods).unwrap();
    // Written out of order; discovery must sort by the NNN- prefix.
    fs::write(mods.join("002-virtio_console.ko"), b"c").unwrap();
    fs::write(mods.join("000-virtio.ko"), b"v").unwrap();
    fs::write(mods.join("001-virtio_mmio.ko"), b"m").unwrap();

    let found = boot_modules_for_image(&dir.path().join("bzImage"));
    let names: Vec<String> = found
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![
            "000-virtio.ko".to_string(),
            "001-virtio_mmio.ko".to_string(),
            "002-virtio_console.ko".to_string(),
        ]
    );
}

#[test]
fn boot_modules_absent_dir_is_empty() {
    let dir = tempfile::TempDir::new().unwrap();
    fs::write(dir.path().join("bzImage"), b"img").unwrap();
    // No modules/ sibling — a built/tarball kernel.
    assert!(boot_modules_for_image(&dir.path().join("bzImage")).is_empty());
    assert!(ordered_boot_modules_in(&dir.path().join("modules")).is_empty());
}

// -- synthetic local-package boot fixture (CI-driven, #[ignore]) --

/// Pack the already-built CI kernel (the tarball cache entry for
/// `KTSTR_E2E_KERNEL_VERSION`) into a minimal synthetic `.rpm` at
/// `KTSTR_E2E_RPM_OUT`, so the CI `local-package-boot` recipe can drive
/// the local-package acquire+boot path end to end with ZERO network:
/// `shell --kernel <that.rpm> --exec 'uname -r'` then extracts, gates,
/// caches, and boots it.
///
/// The rpm carries the real boot image at `/boot/vmlinuz-<rel>` plus a
/// minimal `/boot/config-<rel>` declaring the three boot-critical virtio
/// options `=y`. A ktstr-built kernel always compiles those in (the
/// build path requires them), so the config gate sees them builtin and
/// the entry needs no module tree — the config-gate happy path. Reuses
/// the same `rpm::PackageBuilder` fixture path the extract unit tests
/// use.
///
/// `#[ignore]` because it reaches into the on-disk kernel cache and only
/// makes sense under CI (which runs `kernel-build` first); the recipe
/// invokes it by exact name with the two env vars set.
#[test]
#[ignore = "CI-only: packs the built kernel cache entry into an rpm for the boot e2e"]
fn pack_built_kernel_into_synthetic_rpm() {
    use crate::cache::KernelSource;

    let version =
        std::env::var("KTSTR_E2E_KERNEL_VERSION").expect("KTSTR_E2E_KERNEL_VERSION must be set");
    let out =
        PathBuf::from(std::env::var("KTSTR_E2E_RPM_OUT").expect("KTSTR_E2E_RPM_OUT must be set"));

    // Newest-first list; take the first tarball build for this version.
    let cache = CacheDir::new().expect("open kernel cache");
    let entries = cache.list().expect("list cache");
    let entry = entries
        .iter()
        .filter_map(|e| e.as_valid())
        .find(|e| {
            matches!(e.metadata.source, KernelSource::Tarball)
                && e.metadata
                    .version
                    .as_deref()
                    .is_some_and(|v| v.starts_with(&version))
        })
        .unwrap_or_else(|| {
            panic!("no built (tarball) cache entry for kernel {version}; run kernel-build first")
        });

    let rel = entry
        .metadata
        .version
        .clone()
        .unwrap_or_else(|| version.clone());
    let image = fs::read(entry.image_path()).expect("read cached boot image");
    let image_mib = image.len() / (1024 * 1024);
    let (arch, _) = crate::fetch::arch_info();

    let mut builder = rpm::PackageBuilder::new(
        "ktstr-synthetic-kernel",
        "1.0.0",
        "GPL-2.0-only",
        arch,
        "ktstr local-package boot e2e fixture",
    );
    builder.using_config(rpm::BuildConfig::default().compression(rpm::CompressionType::Zstd));
    let files: [(String, Vec<u8>); 2] = [
        (format!("/boot/vmlinuz-{rel}"), image),
        (
            format!("/boot/config-{rel}"),
            b"CONFIG_VIRTIO=y\nCONFIG_VIRTIO_MMIO=y\nCONFIG_VIRTIO_CONSOLE=y\n".to_vec(),
        ),
    ];
    for (path, body) in files {
        builder
            .with_file_contents(
                body,
                rpm::FileOptions::new(path.as_str()).permissions(0o644),
            )
            .expect("add rpm file");
    }
    let pkg = builder.build().expect("build synthetic rpm");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    pkg.write_file(&out).expect("write synthetic rpm");

    // The produced rpm must be a well-formed, host-arch package so the
    // CLI boot step's acquire doesn't trip on a malformed fixture.
    let produced_arch = package_arch(&out).expect("read produced rpm arch");
    ensure_arch_matches_host(&produced_arch, &out).expect("synthetic rpm targets host arch");
    eprintln!(
        "packed kernel {rel} ({image_mib} MiB image) -> {}",
        out.display()
    );
}
