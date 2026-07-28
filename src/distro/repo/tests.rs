//! Unit tests over the pure parsing/selection functions (small inline
//! fixtures, no network) plus one `#[ignore]` live smoke test per
//! distro that resolves against the real repos.

use super::*;
use std::cmp::Ordering;

#[test]
fn ubuntu_package_indexes_use_published_repository_roots() {
    assert_eq!(
        deb_packages_url(UBUNTU_ARCHIVE, "noble", "amd64"),
        "https://archive.ubuntu.com/ubuntu/dists/noble-updates/main/binary-amd64/Packages.gz",
    );
    assert_eq!(
        deb_packages_url(UBUNTU_PORTS, "noble", "arm64"),
        "https://ports.ubuntu.com/ubuntu-ports/dists/noble-updates/main/binary-arm64/Packages.gz",
    );
    assert_eq!(
        deb_packages_url(UBUNTU_DDEBS, "noble", "arm64"),
        "https://ddebs.ubuntu.com/dists/noble-updates/main/binary-arm64/Packages.gz",
    );
}

#[test]
fn ubuntu_ddeb_ref_has_independent_launchpad_download_origin() {
    let packages = parse_deb_packages(
        "Package: linux-image-unsigned-6.17.0-38-generic-dbgsym\n\
         Version: 6.17.0-38.42~24.04.1\n\
         Filename: pool/main/l/linux-hwe-6.17/linux-image-dbgsym.ddeb\n\
         SHA256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n\
         Size: 4096\n",
    );
    let package = ubuntu_ddeb_package_ref(&packages[0]).unwrap();

    assert_eq!(
        package.url,
        "https://ddebs.ubuntu.com/pool/main/l/linux-hwe-6.17/linux-image-dbgsym.ddeb"
    );
    assert_eq!(
        package.alternate_urls,
        ["https://launchpad.net/ubuntu/+archive/primary/+files/linux-image-dbgsym.ddeb"]
    );
    assert_eq!(package.size, Some(4096));
}

#[test]
fn launchpad_dbgsym_query_constrains_every_identity_axis() {
    let url = Url::parse(
        &launchpad_ddeb_query_url(
            "noble",
            "arm64",
            "linux-image-unsigned-6.8.0-101-generic-dbgsym",
            "6.8.0-101.101",
        )
        .unwrap(),
    )
    .unwrap();
    let query: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();

    assert_eq!(
        url.as_str().split('?').next().unwrap(),
        LAUNCHPAD_PRIMARY_ARCHIVE_API
    );
    assert_eq!(query["ws.op"], "getPublishedBinaries");
    assert_eq!(
        query["binary_name"],
        "linux-image-unsigned-6.8.0-101-generic-dbgsym"
    );
    assert_eq!(query["exact_match"], "true");
    assert_eq!(query["version"], "6.8.0-101.101");
    assert_eq!(
        query["distro_arch_series"],
        "https://api.launchpad.net/devel/ubuntu/noble/arm64"
    );
    assert_eq!(query["pocket"], "Updates");
    assert_eq!(query["status"], "Published");
}

#[test]
fn launchpad_dbgsym_responses_preserve_exact_verified_artifact() {
    let name = "linux-image-unsigned-6.8.0-101-generic-dbgsym";
    let version = "6.8.0-101.101";
    let publication = serde_json::to_vec(&serde_json::json!({
        "total_size": 1,
        "entries": [{
            "self_link": "https://api.launchpad.net/devel/ubuntu/+archive/primary/+binarypub/239341554",
            "binary_package_name": name,
            "binary_package_version": version,
            "distro_arch_series_link": "https://api.launchpad.net/devel/ubuntu/noble/amd64",
            "status": "Published",
            "pocket": "Updates",
            "is_debug": true
        }]
    }))
    .unwrap();
    let publication = exact_launchpad_publication(&publication, "noble", "amd64", name, version)
        .unwrap()
        .unwrap();
    assert_eq!(
        launchpad_binary_files_url(&publication.self_link).unwrap(),
        "https://api.launchpad.net/devel/ubuntu/+archive/primary/+binarypub/239341554?ws.op=binaryFileUrls&include_meta=true"
    );

    let files = serde_json::to_vec(&serde_json::json!([{
        "url": format!(
            "https://launchpad.net/ubuntu/+archive/primary/+files/{name}_{version}_amd64.ddeb"
        ),
        "size": 1_741_755_652_u64,
        "sha1": "ignored-by-ktstr",
        "sha256": TEST_SHA256
    }]))
    .unwrap();
    let package = launchpad_ddeb_package_ref(&files, name, version, "amd64").unwrap();
    assert_eq!(package.name, name);
    assert_eq!(package.version, version);
    assert_eq!(package.sha256, TEST_SHA256);
    assert_eq!(package.size, Some(1_741_755_652));
}

#[test]
fn launchpad_dbgsym_responses_reject_identity_or_file_mismatch() {
    let name = "linux-image-unsigned-6.8.0-101-generic-dbgsym";
    let version = "6.8.0-101.101";
    let wrong_publication = serde_json::to_vec(&serde_json::json!({
        "total_size": 1,
        "entries": [{
            "self_link": "https://api.launchpad.net/devel/ubuntu/+archive/primary/+binarypub/239341554",
            "binary_package_name": name,
            "binary_package_version": version,
            "distro_arch_series_link": "https://api.launchpad.net/devel/ubuntu/noble/arm64",
            "status": "Published",
            "pocket": "Updates",
            "is_debug": true
        }]
    }))
    .unwrap();
    let error = exact_launchpad_publication(&wrong_publication, "noble", "amd64", name, version)
        .expect_err("mismatched architecture must be rejected");
    assert!(error.to_string().contains("identity mismatch"));

    let wrong_file = serde_json::to_vec(&serde_json::json!([{
        "url": format!(
            "https://launchpad.net/ubuntu/+archive/primary/+files/{name}_{version}_arm64.ddeb"
        ),
        "size": 4096,
        "sha256": TEST_SHA256
    }]))
    .unwrap();
    let error = launchpad_ddeb_package_ref(&wrong_file, name, version, "amd64")
        .expect_err("mismatched file architecture must be rejected");
    assert!(error.to_string().contains("invalid primary-archive"));
}

// ---- Complete-resolution last-known-good cache -------------------

const TEST_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn cached_package(name: &str) -> PackageRef {
    PackageRef {
        name: name.to_string(),
        version: "1.2.3-4".to_string(),
        url: format!("https://packages.example.test/{name}"),
        alternate_urls: Vec::new(),
        sha256: TEST_SHA256.to_string(),
        size: Some(4096),
    }
}

fn cached_resolution(
    kind: DistroKind,
    release: Option<&str>,
    host_arch: &str,
) -> ResolvedDistroKernel {
    let (distro, arch) = match kind {
        DistroKind::Fedora => (
            format!("fedora{}", release.unwrap_or("44")),
            host_arch.to_string(),
        ),
        DistroKind::Ubuntu => (
            format!("ubuntu{}-hwe", release.unwrap_or("24.04")),
            match host_arch {
                "x86_64" => "amd64",
                "aarch64" => "arm64",
                other => other,
            }
            .to_string(),
        ),
        DistroKind::AmazonLinux => ("al2023".to_string(), host_arch.to_string()),
        DistroKind::SteamOs => (
            format!("steamos{}", release.unwrap_or("3.8")),
            host_arch.to_string(),
        ),
        DistroKind::Gke => ("gke129".to_string(), host_arch.to_string()),
    };
    ResolvedDistroKernel {
        distro,
        kernel_release: "6.12.34-1".to_string(),
        arch,
        packages: vec![cached_package("kernel")],
        debuginfo: if kind == DistroKind::SteamOs {
            Vec::new()
        } else {
            vec![cached_package("kernel-debuginfo")]
        },
    }
}

fn assert_cached_resolution_invalid(
    kind: DistroKind,
    release: Option<&str>,
    host_arch: &str,
    resolved: &ResolvedDistroKernel,
    message: &str,
) {
    let error = validate_cached_resolution(kind, release, host_arch, resolved)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(message),
        "expected {message:?} in validation error, got {error:?}"
    );
}

#[test]
fn distro_resolution_cache_key_is_versioned_and_covers_every_input_axis() {
    assert_eq!(
        distro_resolution_cache_key(DistroKind::Fedora, None, "x86_64"),
        "distro-resolution-v1/fedora/latest/x86_64"
    );
    assert_eq!(
        distro_resolution_cache_key(DistroKind::Fedora, Some("44"), "x86_64"),
        "distro-resolution-v1/fedora/release=44/x86_64"
    );

    let baseline = distro_resolution_cache_key(DistroKind::Fedora, None, "x86_64");
    assert_ne!(
        baseline,
        distro_resolution_cache_key(DistroKind::Ubuntu, None, "x86_64")
    );
    assert_ne!(
        baseline,
        distro_resolution_cache_key(DistroKind::Fedora, Some("44"), "x86_64")
    );
    assert_ne!(
        baseline,
        distro_resolution_cache_key(DistroKind::Fedora, None, "aarch64")
    );

    let amazonlinux = distro_resolution_cache_key(DistroKind::AmazonLinux, None, "x86_64");
    assert_eq!(
        amazonlinux,
        "distro-resolution-v1/amazonlinux/rolling-al2023/x86_64"
    );
    assert_eq!(
        amazonlinux,
        distro_resolution_cache_key(DistroKind::AmazonLinux, Some("2023"), "x86_64"),
        "equivalent Amazon Linux selectors must share one rolling AL2023 fallback",
    );
}

#[test]
fn cached_resolution_validator_accepts_every_repository_distro_shape() {
    for (kind, release, arch) in [
        (DistroKind::Fedora, None, "x86_64"),
        (DistroKind::Fedora, Some("44"), "aarch64"),
        (DistroKind::Ubuntu, None, "aarch64"),
        (DistroKind::Ubuntu, Some("24.04"), "x86_64"),
        (DistroKind::AmazonLinux, None, "aarch64"),
        (DistroKind::AmazonLinux, Some("2023"), "x86_64"),
        (DistroKind::SteamOs, None, "x86_64"),
        (DistroKind::SteamOs, Some("3.8"), "x86_64"),
    ] {
        let resolved = cached_resolution(kind, release, arch);
        validate_cached_resolution(kind, release, arch, &resolved)
            .unwrap_or_else(|error| panic!("{kind:?}/{release:?}/{arch}: {error:#}"));
    }
}

#[test]
fn cached_resolution_validator_rejects_request_identity_mismatches() {
    let mut resolved = cached_resolution(DistroKind::Fedora, Some("44"), "x86_64");
    resolved.distro = "fedora43".to_string();
    assert_cached_resolution_invalid(
        DistroKind::Fedora,
        Some("44"),
        "x86_64",
        &resolved,
        "does not match requested release",
    );

    let mut resolved = cached_resolution(DistroKind::Ubuntu, None, "aarch64");
    resolved.arch = "amd64".to_string();
    assert_cached_resolution_invalid(
        DistroKind::Ubuntu,
        None,
        "aarch64",
        &resolved,
        "expected \"arm64\"",
    );

    let resolved = cached_resolution(DistroKind::SteamOs, None, "x86_64");
    assert_cached_resolution_invalid(
        DistroKind::SteamOs,
        None,
        "aarch64",
        &resolved,
        "requires x86_64",
    );

    let resolved = cached_resolution(DistroKind::Gke, None, "x86_64");
    assert_cached_resolution_invalid(
        DistroKind::Gke,
        None,
        "x86_64",
        &resolved,
        "does not match requested release",
    );
}

#[test]
fn cached_resolution_validator_rejects_incomplete_payloads_and_debug_policy() {
    let mut resolved = cached_resolution(DistroKind::Fedora, None, "x86_64");
    resolved.kernel_release.clear();
    assert_cached_resolution_invalid(
        DistroKind::Fedora,
        None,
        "x86_64",
        &resolved,
        "kernel release is empty",
    );

    let mut resolved = cached_resolution(DistroKind::Fedora, None, "x86_64");
    resolved.packages.clear();
    assert_cached_resolution_invalid(
        DistroKind::Fedora,
        None,
        "x86_64",
        &resolved,
        "no kernel packages",
    );

    let mut resolved = cached_resolution(DistroKind::Ubuntu, None, "x86_64");
    resolved.debuginfo.clear();
    assert_cached_resolution_invalid(
        DistroKind::Ubuntu,
        None,
        "x86_64",
        &resolved,
        "no mandatory debuginfo",
    );

    let mut resolved = cached_resolution(DistroKind::SteamOs, None, "x86_64");
    resolved
        .debuginfo
        .push(cached_package("unexpected-debuginfo"));
    assert_cached_resolution_invalid(
        DistroKind::SteamOs,
        None,
        "x86_64",
        &resolved,
        "unexpectedly contains debuginfo",
    );
}

#[test]
fn cached_resolution_validator_rejects_malformed_package_fields() {
    let valid = cached_resolution(DistroKind::Fedora, None, "x86_64");

    let mut resolved = valid.clone();
    resolved.packages[0].name = " ".to_string();
    assert_cached_resolution_invalid(DistroKind::Fedora, None, "x86_64", &resolved, "empty name");

    let mut resolved = valid.clone();
    resolved.packages[0].version.clear();
    assert_cached_resolution_invalid(
        DistroKind::Fedora,
        None,
        "x86_64",
        &resolved,
        "empty version",
    );

    for url in [
        "ftp://packages.example.test/kernel",
        "https://",
        "not a URL",
    ] {
        let mut resolved = valid.clone();
        resolved.packages[0].url = url.to_string();
        assert_cached_resolution_invalid(DistroKind::Fedora, None, "x86_64", &resolved, "URL");
    }

    for sha256 in [
        "abc",
        "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
        let mut resolved = valid.clone();
        resolved.packages[0].sha256 = sha256.to_string();
        assert_cached_resolution_invalid(
            DistroKind::Fedora,
            None,
            "x86_64",
            &resolved,
            "invalid sha256",
        );
    }

    let mut resolved = valid;
    resolved.packages[0].size = Some(0);
    assert_cached_resolution_invalid(DistroKind::Fedora, None, "x86_64", &resolved, "zero size");
}

// ---- rpmvercmp / EVR ----------------------------------------------

#[test]
fn rpmvercmp_numeric_and_release() {
    assert_eq!(rpmvercmp("1.0", "1.0"), Ordering::Equal);
    assert_eq!(rpmvercmp("1.0", "1.1"), Ordering::Less);
    // Segment-wise numeric compare: 109 > 97 even though "97" > "109"
    // lexically — the exact trap naive string sort falls into.
    assert_eq!(rpmvercmp("6.1.109", "6.1.97"), Ordering::Greater);
    assert_eq!(rpmvercmp("6.1.10", "6.1.9"), Ordering::Greater);
    // Leading zeros ignored inside a numeric segment.
    assert_eq!(rpmvercmp("1.007", "1.7"), Ordering::Equal);
    // Longer version (more segments) wins when the prefix is equal.
    assert_eq!(rpmvercmp("1.0.1", "1.0"), Ordering::Greater);
}

#[test]
fn rpmvercmp_tilde_and_caret() {
    // `~` sorts before everything, including the empty string.
    assert_eq!(rpmvercmp("1.0~rc1", "1.0"), Ordering::Less);
    assert_eq!(rpmvercmp("1.0~rc1", "1.0~rc2"), Ordering::Less);
    // `^` sorts after a real segment but before an ended string.
    assert_eq!(rpmvercmp("1.0^20240101", "1.0"), Ordering::Greater);
}

#[test]
fn evr_selects_newest_al2023_kernel() {
    // Mirrors the live AL2023 spread: naive string sort mis-orders
    // these; rpmvercmp EVR ordering must pick 6.1.176-220.360.
    let mk = |ver: &str, rel: &str| Evr {
        epoch: "0".into(),
        ver: ver.into(),
        rel: rel.into(),
    };
    let cands: Vec<RpmCand> = [
        ("6.1.10", "15.42.amzn2023"),
        ("6.1.97", "104.177.amzn2023"),
        ("6.1.109", "118.189.amzn2023"),
        ("6.1.176", "220.360.amzn2023"),
    ]
    .iter()
    .map(|(v, r)| RpmCand {
        name: "kernel".into(),
        arch: "x86_64".into(),
        evr: mk(v, r),
        sha256: "deadbeef".into(),
        href: format!("../../blobstore/x/kernel-{v}-{r}.x86_64.rpm"),
        size: Some(1),
    })
    .collect();
    let newest = newest_evr(&cands, "kernel").unwrap();
    assert_eq!(newest.display(), "6.1.176-220.360.amzn2023");
}

// ---- AL2023 kernel streams ----------------------------------------

#[test]
fn al2023_stream_classification() {
    // Bare stream packages and their modules-extra map to the stream.
    assert_eq!(al2023_stream("kernel"), Some("kernel"));
    assert_eq!(al2023_stream("kernel6.12"), Some("kernel6.12"));
    assert_eq!(al2023_stream("kernel6.18"), Some("kernel6.18"));
    assert_eq!(al2023_stream("kernel-modules-extra"), Some("kernel"));
    assert_eq!(
        al2023_stream("kernel6.18-modules-extra"),
        Some("kernel6.18")
    );
    // Everything else kernel-adjacent is not a stream member.
    for name in [
        "kernel-devel",
        "kernel-headers",
        "kernel-tools",
        "kernel-libbpf",
        "kernel-livepatch-6.1.176-220.360",
        "kernel-modules-extra-common",
        "kernel6.18-devel",
        "kernel6.18-modules-extra-common",
        "kernel6.18-tools-devel",
        "kernel6-modules-extra",
        "bash",
    ] {
        assert_eq!(al2023_stream(name), None, "{name:?} must not classify");
    }
}

#[test]
fn al2023_picks_newest_stream_and_pairs_modules_extra() {
    // Mirrors the live AL2023 spread: the default 6.1 stream has a far
    // HIGHER release number — and here a higher EPOCH — than the 6.18
    // stream, so a whole-EVR max across package names (epoch-first)
    // would pick the wrong stream; the kernel version must dominate
    // across streams.
    let mk = |name: &str, epoch: &str, ver: &str, rel: &str| RpmCand {
        name: name.into(),
        arch: "x86_64".into(),
        evr: Evr {
            epoch: epoch.into(),
            ver: ver.into(),
            rel: format!("{rel}.amzn2023"),
        },
        sha256: format!("sha-{name}-{ver}"),
        href: format!("../../blobstore/x/{name}-{ver}-{rel}.amzn2023.x86_64.rpm"),
        size: Some(1),
    };
    let cands = vec![
        mk("kernel", "1", "6.1.176", "220.360"),
        mk("kernel-modules-extra", "1", "6.1.176", "220.360"),
        mk("kernel6.12", "1", "6.12.94", "123.176"),
        mk("kernel6.12-modules-extra", "1", "6.12.94", "123.176"),
        // Stale 6.18 build: must lose to 69.136 within the stream.
        mk("kernel6.18", "0", "6.18.35", "68.129"),
        mk("kernel6.18-modules-extra", "0", "6.18.35", "68.129"),
        mk("kernel6.18", "0", "6.18.36", "69.136"),
        mk("kernel6.18-modules-extra", "0", "6.18.36", "69.136"),
    ];
    let (stream, target) = al2023_pick_stream(&cands).unwrap();
    assert_eq!(stream, "kernel6.18");
    assert_eq!(target.display(), "6.18.36-69.136.amzn2023");

    let extra = format!("{stream}-modules-extra");
    let names = [stream, extra.as_str()];
    let refs = build_package_set(&cands, "https://example/repo/", &names, &target, true).unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].name, "kernel6.18");
    assert_eq!(refs[1].name, "kernel6.18-modules-extra");
    assert_eq!(refs[1].version, "6.18.36-69.136.amzn2023");

    // A stream whose modules-extra is absent at the kernel's EVR is a
    // hard error (virtio_console lives there).
    let missing: Vec<RpmCand> = cands
        .iter()
        .filter(|c| c.name != "kernel6.18-modules-extra")
        .cloned()
        .collect();
    let err = build_package_set(&missing, "https://example/repo/", &names, &target, true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("kernel6.18-modules-extra"), "{err}");
}

#[test]
fn evr_epoch_dominates() {
    let a = Evr {
        epoch: "1".into(),
        ver: "1.0".into(),
        rel: "1".into(),
    };
    let b = Evr {
        epoch: "0".into(),
        ver: "9.9".into(),
        rel: "9".into(),
    };
    assert_eq!(a.cmp(&b), Ordering::Greater);
    assert_eq!(a.display(), "1:1.0-1");
    assert_eq!(b.display(), "9.9-9");
}

// ---- Debian version compare ---------------------------------------

#[test]
fn deb_version_cmp_basics() {
    assert_eq!(deb_version_cmp("1.0", "1.0"), Ordering::Equal);
    // `~` sorts before the empty string (pre-releases before release).
    assert_eq!(deb_version_cmp("1.0~beta1", "1.0"), Ordering::Less);
    // Epoch dominates.
    assert_eq!(deb_version_cmp("1:1.0", "2.0"), Ordering::Greater);
    // Numeric segments compared numerically.
    assert_eq!(
        deb_version_cmp("6.17.0-35.35~24.04.1", "6.17.0-9.9~24.04.1"),
        Ordering::Greater
    );
    // Revision compared after upstream.
    assert_eq!(
        deb_version_cmp("6.8.0-31.31", "6.8.0-45.45"),
        Ordering::Less
    );
}

// ---- repomd + primary (RPM) ---------------------------------------

const REPOMD_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<repomd xmlns="http://linux.duke.edu/metadata/repo" xmlns:rpm="http://linux.duke.edu/metadata/rpm">
  <revision>1</revision>
  <data type="primary">
    <checksum type="sha256">abc123</checksum>
    <open-checksum type="sha256">def456</open-checksum>
    <location href="repodata/abc123-primary.xml.zst"/>
    <size>10</size>
  </data>
  <data type="filelists">
    <checksum type="sha256">999</checksum>
    <location href="repodata/999-filelists.xml.zst"/>
  </data>
</repomd>
"#;

#[test]
fn parse_repomd_finds_primary() {
    let datas = parse_repomd(REPOMD_XML.as_bytes()).unwrap();
    let primary = datas.iter().find(|d| d.data_type == "primary").unwrap();
    assert_eq!(primary.href, "repodata/abc123-primary.xml.zst");
    assert_eq!(primary.checksum, "abc123");
    assert_eq!(primary.checksum_type, "sha256");
}

// Two kernel-core versions (exercise rpmvercmp selection), the
// matching kernel-modules-core, and an unrelated package that must be
// filtered out. The `<format>` block carries `rpm:entry` children whose
// `name`/`ver` attribute spellings must NOT leak into the parsed
// package (first-wins guard).
const PRIMARY_XML: &str = r#"<?xml version="1.0"?>
<metadata xmlns="http://linux.duke.edu/metadata/common" xmlns:rpm="http://linux.duke.edu/metadata/rpm" packages="4">
<package type="rpm">
  <name>kernel-core</name>
  <arch>x86_64</arch>
  <version epoch="0" ver="7.1.3" rel="200.fc44"/>
  <checksum type="sha256" pkgid="YES">aaaa</checksum>
  <size package="100" installed="200" archive="300"/>
  <location href="Packages/k/kernel-core-7.1.3-200.fc44.x86_64.rpm"/>
  <format>
    <rpm:provides>
      <rpm:entry name="kernel-core" ver="7.1.3" rel="200.fc44"/>
    </rpm:provides>
  </format>
</package>
<package type="rpm">
  <name>kernel-core</name>
  <arch>x86_64</arch>
  <version epoch="0" ver="7.1.10" rel="200.fc44"/>
  <checksum type="sha256" pkgid="YES">bbbb</checksum>
  <size package="101"/>
  <location href="Packages/k/kernel-core-7.1.10-200.fc44.x86_64.rpm"/>
</package>
<package type="rpm">
  <name>kernel-modules-core</name>
  <arch>x86_64</arch>
  <version epoch="0" ver="7.1.10" rel="200.fc44"/>
  <checksum type="sha256" pkgid="YES">cccc</checksum>
  <size package="102"/>
  <location href="Packages/k/kernel-modules-core-7.1.10-200.fc44.x86_64.rpm"/>
</package>
<package type="rpm">
  <name>bash</name>
  <arch>x86_64</arch>
  <version epoch="0" ver="5.2" rel="1.fc44"/>
  <checksum type="sha256" pkgid="YES">dddd</checksum>
  <location href="Packages/b/bash-5.2-1.fc44.x86_64.rpm"/>
</package>
</metadata>
"#;

#[test]
fn parse_primary_filters_and_selects_newest() {
    let wanted = ["kernel-core", "kernel-modules-core"];
    let cands =
        parse_primary(Cursor::new(PRIMARY_XML.as_bytes()), |n| wanted.contains(&n)).unwrap();
    // bash filtered out; two kernel-core + one kernel-modules-core.
    assert_eq!(cands.len(), 3);
    assert!(cands.iter().all(|c| c.name != "bash"));

    let newest = newest_evr(&cands, "kernel-core").unwrap();
    assert_eq!(newest.display(), "7.1.10-200.fc44");

    let names = ["kernel-core", "kernel-modules-core"];
    let refs = build_package_set(&cands, "https://example/repo/", &names, &newest, false).unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].name, "kernel-core");
    assert_eq!(refs[0].sha256, "bbbb");
    assert_eq!(refs[0].size, Some(101));
    assert_eq!(
        refs[0].url,
        "https://example/repo/Packages/k/kernel-core-7.1.10-200.fc44.x86_64.rpm"
    );
    // First-wins: the 7.1.3 kernel-core's rpm:provides entry did not
    // clobber its own version.
    assert!(cands.iter().any(|c| c.evr.display() == "7.1.3-200.fc44"));
}

#[test]
fn build_package_set_strict_errors_on_missing() {
    let target = Evr {
        epoch: "0".into(),
        ver: "7.1.10".into(),
        rel: "200.fc44".into(),
    };
    let err = build_package_set(&[], "https://x/", &["kernel-debuginfo"], &target, true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("kernel-debuginfo"));
    assert!(err.contains("debuginfo"));
}

// ---- URL joining ---------------------------------------------------

#[test]
fn join_url_resolves_dotdot_traversal() {
    // AL2023 blobstore hrefs escape the repodata dir with `../`.
    let base = "https://cdn.amazonlinux.com/al2023/core/guids/HASH/x86_64/";
    let href = "../../../../blobstore/DEADBEEF/kernel-6.1.176-220.360.amzn2023.x86_64.rpm";
    assert_eq!(
        join_url(base, href).unwrap(),
        "https://cdn.amazonlinux.com/al2023/blobstore/DEADBEEF/kernel-6.1.176-220.360.amzn2023.x86_64.rpm"
    );
}

// ---- Debian Packages ----------------------------------------------

const DEB_PACKAGES: &str = "\
Package: linux-image-generic-hwe-24.04
Version: 6.17.0-35.35~24.04.1
Depends: linux-image-6.17.0-35-generic, linux-modules-extra-6.17.0-35-generic, linux-firmware
Filename: pool/main/l/linux-meta-hwe/linux-image-generic-hwe-24.04_6.17.0-35.35~24.04.1_amd64.deb
SHA256: 1111
Size: 1

Package: linux-image-6.17.0-9-generic
Version: 6.17.0-9.9~24.04.1
Filename: pool/main/l/linux-signed-hwe-6.17/linux-image-6.17.0-9-generic_6.17.0-9.9~24.04.1_amd64.deb
SHA256: 2222
Size: 16000000

Package: linux-image-6.17.0-35-generic
Version: 6.17.0-35.35~24.04.1
Filename: pool/main/l/linux-signed-hwe-6.17/linux-image-6.17.0-35-generic_6.17.0-35.35~24.04.1_amd64.deb
SHA256: 3333
Size: 16040642

Package: linux-modules-6.17.0-35-generic
Version: 6.17.0-35.35~24.04.1
Filename: pool/main/l/linux-hwe-6.17/linux-modules-6.17.0-35-generic_6.17.0-35.35~24.04.1_amd64.deb
SHA256: 4444
Size: 42342592
";

#[test]
fn newest_kernel_with_debuginfo_falls_back_when_newest_ddeb_lags() {
    // Two published kernels; the NEWER one's image exists but its dbgsym
    // ddeb has not landed yet (the hours-long lag). The resolver must
    // fall back to the older, fully-published kernel rather than fail.
    let debs = parse_deb_packages(
        "Package: linux-image-6.17.0-40-generic\nVersion: 6.17.0-40.44~24.04.1\n\n\
         Package: linux-image-6.17.0-38-generic\nVersion: 6.17.0-38.42~24.04.1\n",
    );
    let ddebs = parse_deb_packages(
        // Only the older kernel has a dbgsym ddeb published.
        "Package: linux-image-unsigned-6.17.0-38-generic-dbgsym\n\
         Version: 6.17.0-38.42~24.04.1\n",
    );
    let selected = newest_kernel_with_debuginfo(&debs, &ddebs).unwrap();
    assert_eq!(selected.kver, "6.17.0-38");
    assert_eq!(
        selected.dbgsym.package, "linux-image-unsigned-6.17.0-38-generic-dbgsym",
        "must carry the exact older ddeb stanza forward with the fallback",
    );

    // Once the newer ddeb lands, the newest is selected.
    let ddebs_caught_up = parse_deb_packages(
        "Package: linux-image-6.17.0-40-generic-dbgsym\nVersion: 6.17.0-40.44~24.04.1\n\n\
         Package: linux-image-unsigned-6.17.0-38-generic-dbgsym\n\
         Version: 6.17.0-38.42~24.04.1\n",
    );
    let selected = newest_kernel_with_debuginfo(&debs, &ddebs_caught_up).unwrap();
    assert_eq!(
        selected.kver, "6.17.0-40",
        "signed-dbgsym fallback counts; newest complete kernel wins",
    );
    assert_eq!(
        selected.dbgsym.package,
        "linux-image-6.17.0-40-generic-dbgsym",
    );

    // No kernel has debuginfo → None (a genuinely broken archive, which
    // the caller turns into the mandatory-debuginfo hard error).
    assert!(newest_kernel_with_debuginfo(&debs, &[]).is_none());
}

#[test]
fn exact_dbgsym_provider_checks_candidates_newest_first() {
    let debs = parse_deb_packages(
        "Package: linux-image-6.17.0-40-generic\nVersion: 6.17.0-40.44~24.04.1\n\n\
         Package: linux-image-6.17.0-38-generic\nVersion: 6.17.0-38.42~24.04.1\n",
    );
    let expected = "linux-image-unsigned-6.17.0-38-generic-dbgsym";
    let mut requests = Vec::new();
    let selected = newest_kernel_with_exact_dbgsym_lookup(&debs, |name, version| {
        requests.push((name.to_string(), version.to_string()));
        if name != expected {
            return Ok(None);
        }
        Ok(Some(PackageRef {
            name: name.to_string(),
            version: version.to_string(),
            url: format!("https://launchpad.net/ubuntu/+archive/primary/+files/{name}.ddeb"),
            alternate_urls: Vec::new(),
            sha256: TEST_SHA256.to_string(),
            size: Some(4096),
        }))
    })
    .unwrap()
    .unwrap();

    assert_eq!(selected.kver, "6.17.0-38");
    assert_eq!(selected.debuginfo.name, expected);
    assert_eq!(
        requests,
        [
            (
                "linux-image-unsigned-6.17.0-40-generic-dbgsym".to_string(),
                "6.17.0-40.44~24.04.1".to_string(),
            ),
            (
                "linux-image-6.17.0-40-generic-dbgsym".to_string(),
                "6.17.0-40.44~24.04.1".to_string(),
            ),
            (expected.to_string(), "6.17.0-38.42~24.04.1".to_string(),),
        ]
    );
}

#[test]
fn parse_deb_packages_and_hwe_chain() {
    let debs = parse_deb_packages(DEB_PACKAGES);
    assert_eq!(debs.len(), 4);

    let meta = newest_deb(&debs, "linux-image-generic-hwe-24.04").unwrap();
    let kver = hwe_kver(meta).unwrap();
    assert_eq!(kver, "6.17.0-35");

    // Depends dropped the `(>= x)` / version constraints to bare names.
    assert!(meta.depends.contains(&"linux-firmware".to_string()));

    let names = [
        format!("linux-image-{kver}-generic"),
        format!("linux-modules-{kver}-generic"),
    ];
    let refs =
        ubuntu_package_refs(&debs, "https://archive.ubuntu.com/ubuntu/", &names, false).unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].sha256, "3333");
    assert_eq!(
        refs[0].url,
        "https://archive.ubuntu.com/ubuntu/pool/main/l/linux-signed-hwe-6.17/linux-image-6.17.0-35-generic_6.17.0-35.35~24.04.1_amd64.deb"
    );
    assert_eq!(refs[1].name, "linux-modules-6.17.0-35-generic");
}

// ---- meta-release-lts ---------------------------------------------

const META_RELEASE_LTS: &str = "\
Dist: focal
Name: Focal Fossa
Version: 20.04.6 LTS
Supported: 1
Description: This is the 20.04 LTS release

Dist: jammy
Name: Jammy Jellyfish
Version: 22.04.5 LTS
Supported: 1
Description: This is the 22.04 LTS release

Dist: noble
Name: Noble Numbat
Version: 24.04.1 LTS
Supported: 1
Description: This is the 24.04 LTS release

Dist: dapper
Name: Dapper Drake
Version: 6.06 LTS
Supported: 0
Description: EOL
";

#[test]
fn latest_lts_and_codename_lookup() {
    let rels = parse_meta_release_lts(META_RELEASE_LTS);
    let (version, codename) = latest_lts(&rels).unwrap();
    assert_eq!(version, "24.04");
    assert_eq!(codename, "noble");
    assert_eq!(
        codename_for_version(&rels, "22.04").as_deref(),
        Some("jammy")
    );
    // EOL (Supported: 0) release never chosen as latest.
    assert_ne!(codename, "dapper");
}

#[test]
fn fallback_codename_offline() {
    assert_eq!(fallback_codename("24.04"), Some("noble"));
    assert_eq!(fallback_codename("18.04"), None);
}

// ---- Amazon Linux mirror.list -------------------------------------

#[test]
fn parse_mirror_list_normalizes_trailing_slash() {
    let with_slash = "https://cdn.amazonlinux.com/al2023/core/guids/HASH/x86_64/\n";
    assert_eq!(
        parse_mirror_list(with_slash).unwrap(),
        "https://cdn.amazonlinux.com/al2023/core/guids/HASH/x86_64/"
    );
    let no_slash = "https://cdn.amazonlinux.com/al2023/core/guids/HASH/x86_64";
    assert_eq!(
        parse_mirror_list(no_slash).unwrap(),
        "https://cdn.amazonlinux.com/al2023/core/guids/HASH/x86_64/"
    );
    assert_eq!(parse_mirror_list("\n\n"), None);
}

// ---- SteamOS pacman channels ---------------------------------------

const NEPTUNE_DESC: &str = "\
%FILENAME%
linux-neptune-618-6.18.33.valve2-1.1-x86_64.pkg.tar.zst

%NAME%
linux-neptune-618

%VERSION%
6.18.33.valve2-1.1

%CSIZE%
148103715

%SHA256SUM%
982329544c326401344d0787aa40b651d4fbcd620fc69b18c42bd2ebaced1180

%ARCH%
x86_64
";

#[test]
fn pacman_desc_parses_resolver_fields() {
    let pkg = parse_pacman_desc(NEPTUNE_DESC);
    assert_eq!(pkg.name, "linux-neptune-618");
    assert_eq!(pkg.version, "6.18.33.valve2-1.1");
    assert_eq!(
        pkg.filename,
        "linux-neptune-618-6.18.33.valve2-1.1-x86_64.pkg.tar.zst"
    );
    assert_eq!(pkg.sha256.len(), 64);
    assert_eq!(pkg.csize, Some(148103715));
    assert_eq!(pkg.arch, "x86_64");
}

#[test]
fn pacman_db_round_trips_through_tar_and_codecs() {
    // Build a two-package db tar at test time; `desc` must be matched
    // by basename inside the `<name>-<version>/` entry dir.
    let mut builder = tar::Builder::new(Vec::new());
    for (dir, desc) in [
        ("linux-neptune-618-6.18.33.valve2-1.1", NEPTUNE_DESC),
        (
            "bash-5.2-1",
            "%NAME%\nbash\n\n%VERSION%\n5.2-1\n\n%ARCH%\nx86_64\n",
        ),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(desc.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, format!("{dir}/desc"), desc.as_bytes())
            .unwrap();
    }
    let tar_bytes = builder.into_inner().unwrap();

    // The live jupiter dbs are xz; gz/zst/plain accepted for robustness.
    let mut xz = xz2::write::XzEncoder::new(Vec::new(), 6);
    std::io::Write::write_all(&mut xz, &tar_bytes).unwrap();
    let xz_bytes = xz.finish().unwrap();
    for bytes in [&tar_bytes, &xz_bytes] {
        let pkgs = parse_pacman_db(decompress_pacman_db(bytes).unwrap()).unwrap();
        assert_eq!(pkgs.len(), 2);
        assert_eq!(pkgs[0].name, "linux-neptune-618");
        assert_eq!(pkgs[1].name, "bash");
    }
}

#[test]
fn steamos_kernel_series_classification() {
    // The legacy base package and versioned series classify.
    assert_eq!(
        steamos_kernel_series("linux-neptune"),
        Some("linux-neptune")
    );
    assert_eq!(
        steamos_kernel_series("linux-neptune-618"),
        Some("linux-neptune-618")
    );
    // Variant packages never classify (from the live jupiter-3.8 db).
    for name in [
        "linux-neptune-618-headers",
        "linux-neptune-618-devel",
        "linux-neptune-618-drm-exec",
        "linux-neptune-65-kasan",
        "linux-neptune-rtw-debug",
        "linux-neptune-headers",
        "linux-firmware",
        "bash",
    ] {
        assert_eq!(steamos_kernel_series(name), None, "{name:?}");
    }
}

#[test]
fn steamos_picks_newest_kernel_series_by_version() {
    let mk = |name: &str, version: &str| PacmanPkg {
        name: name.into(),
        version: version.into(),
        filename: format!("{name}-{version}-x86_64.pkg.tar.zst"),
        sha256: "deadbeef".into(),
        csize: Some(1),
        arch: "x86_64".into(),
    };
    // Mirrors the live jupiter-3.8 spread: the older 6.16 series has a
    // far higher valve build number and pkgrel than 6.18, so only the
    // kernel-version compare picks the right series.
    let pkgs = vec![
        mk("linux-neptune", "5.13.0.valve37-1"),
        mk("linux-neptune-616", "6.16.12.valve24.2-2.1"),
        mk("linux-neptune-618", "6.18.33.valve2-1.1"),
        mk("linux-neptune-618-headers", "6.18.99.valve9-1"),
    ];
    let best = steamos_pick_kernel(&pkgs).unwrap();
    assert_eq!(best.name, "linux-neptune-618");
    assert_eq!(best.version, "6.18.33.valve2-1.1");
}

#[test]
fn steamos_latest_channel_skips_non_numeric() {
    // Shape of the live mirror listing: numbered trains, point/suffix
    // channels, and the rel/main/staging/ci churn.
    let html = r#"<a href="../">../</a>
<a href="holo-3.8/">holo-3.8/</a>
<a href="jupiter-3.3.1/">jupiter-3.3.1/</a>
<a href="jupiter-3.7/">jupiter-3.7/</a>
<a href="jupiter-3.8/">jupiter-3.8/</a>
<a href="jupiter-3.8.1x/">jupiter-3.8.1x/</a>
<a href="jupiter-ci-test/">jupiter-ci-test/</a>
<a href="jupiter-main/">jupiter-main/</a>
<a href="jupiter-rel/">jupiter-rel/</a>
<a href="jupiter-staging/">jupiter-staging/</a>
"#;
    assert_eq!(steamos_latest_channel(html).as_deref(), Some("3.8"));
    assert_eq!(steamos_latest_channel("<html></html>"), None);
}

// ---- Fedora releases.json -----------------------------------------

#[test]
fn fedora_latest_from_json_picks_max_numeric() {
    let json = r#"[
      {"version": "42", "arch": "x86_64"},
      {"version": "44", "arch": "aarch64"},
      {"version": "43", "arch": "x86_64"},
      {"version": "Rawhide", "arch": "x86_64"}
    ]"#;
    assert_eq!(fedora_latest_from_json(json).unwrap(), "44");
}

// ---- Live smoke tests (network; run with --ignored) ---------------

/// Shape checks every resolved package must satisfy: non-empty
/// name/version, an `http(s)` URL, and a 64-hex sha256.
fn assert_pkg_shape(p: &PackageRef) {
    assert!(!p.name.is_empty(), "empty package name");
    assert!(!p.version.is_empty(), "empty version for {}", p.name);
    for url in std::iter::once(&p.url).chain(&p.alternate_urls) {
        assert!(url.starts_with("http"), "bad url: {url}");
    }
    assert_eq!(
        p.sha256.len(),
        64,
        "sha256 not 64 hex: {} ({})",
        p.sha256,
        p.name
    );
    assert!(p.sha256.bytes().all(|c| c.is_ascii_hexdigit()));
}

/// Shape of a fully-resolved distro kernel that publishes debuginfo
/// (every distro except SteamOS): non-empty tag/release, at least one
/// kernel package, at least one debuginfo package, all well-shaped.
fn assert_resolved_shape(r: &ResolvedDistroKernel) {
    assert!(!r.distro.is_empty());
    assert!(!r.kernel_release.is_empty());
    assert!(!r.packages.is_empty(), "no kernel packages resolved");
    assert!(!r.debuginfo.is_empty(), "no debuginfo resolved");
    for p in r.packages.iter().chain(r.debuginfo.iter()) {
        assert_pkg_shape(p);
    }
}

/// Verify a resolved package URL exists upstream WITHOUT downloading its
/// body: a ranged GET (`Range: bytes=0-0`) whose status we inspect and
/// whose body we drop unread. Ranged GET rather than HEAD so the probe is
/// robust against a mirror that rejects HEAD; a server that ignores the
/// `Range` header still only ships headers here — the blocking body is
/// lazily read and we never read it, so no full package is pulled.
/// Ubuntu dbgsym packages may resolve through the DDEB archive or Launchpad's
/// independent primary-archive file store. The fetch helper bounds response
/// stalls and retries transient transport/
/// gateway failures while preserving this ranged request on every attempt.
/// A non-success status (notably 404) is a hard failure so an upstream
/// URL/layout change trips CI in advance of any real download.
fn assert_url_exists(url: &str) {
    let status = crate::fetch::probe_url_status(url)
        .unwrap_or_else(|e| panic!("existence probe for {url} failed: {e}"));
    assert!(
        status.is_success(),
        "package URL does not exist upstream ({status}): {url}",
    );
}

/// Probe every resolved kernel + debuginfo URL for upstream existence.
fn assert_urls_exist(r: &ResolvedDistroKernel) {
    for p in r.packages.iter().chain(r.debuginfo.iter()) {
        assert_url_exists(&p.url);
    }
}

#[test]
#[ignore = "hits the live Fedora mirror"]
fn live_fedora() {
    for arch in ["x86_64", "aarch64"] {
        let r = resolve_for_arch(DistroKind::Fedora, None, arch).unwrap();
        eprintln!(
            "fedora/{arch}: {} {} pkgs={:?} dbg={:?}",
            r.distro,
            r.kernel_release,
            r.packages.iter().map(|p| &p.name).collect::<Vec<_>>(),
            r.debuginfo.iter().map(|p| &p.name).collect::<Vec<_>>(),
        );
        assert_resolved_shape(&r);
        assert_urls_exist(&r);
    }
}

#[test]
#[ignore = "hits the live Ubuntu archive + ddebs"]
fn live_ubuntu() {
    for arch in ["x86_64", "aarch64"] {
        let r = resolve_for_arch(DistroKind::Ubuntu, None, arch).unwrap();
        eprintln!(
            "ubuntu/{arch}: {} {} pkgs={:?} dbg={:?}",
            r.distro,
            r.kernel_release,
            r.packages.iter().map(|p| &p.name).collect::<Vec<_>>(),
            r.debuginfo.iter().map(|p| &p.name).collect::<Vec<_>>(),
        );
        assert_resolved_shape(&r);
        assert_urls_exist(&r);
    }
}

#[test]
fn steamos_rejects_non_x86_64_before_any_fetch() {
    // The arch guard fires before any network access, so this runs in
    // the offline suite.
    let err = resolve_for_arch(DistroKind::SteamOs, None, "aarch64")
        .unwrap_err()
        .to_string();
    assert!(err.contains("x86_64 only"), "{err}");
}

#[test]
#[ignore = "hits the live SteamOS mirror"]
fn live_steamos() {
    // x86_64 only — Valve publishes no other architecture.
    let r = resolve_for_arch(DistroKind::SteamOs, None, "x86_64").unwrap();
    eprintln!(
        "steamos/x86_64: {} {} pkgs={:?}",
        r.distro,
        r.kernel_release,
        r.packages.iter().map(|p| &p.name).collect::<Vec<_>>(),
    );
    assert!(!r.packages.is_empty());
    assert!(r.packages[0].name.starts_with("linux-neptune"));
    for p in &r.packages {
        assert_pkg_shape(p);
    }
    // No debug packages exist for linux-neptune — the documented
    // mandatory-debuginfo exemption.
    assert!(r.debuginfo.is_empty());
    assert_urls_exist(&r);
}

#[test]
#[ignore = "hits the live Amazon Linux CDN"]
fn live_amazonlinux() {
    for arch in ["x86_64", "aarch64"] {
        let r = resolve_for_arch(DistroKind::AmazonLinux, Some("2023"), arch).unwrap();
        eprintln!(
            "al2023/{arch}: {} {} pkgs={:?} dbg={:?}",
            r.distro,
            r.kernel_release,
            r.packages.iter().map(|p| &p.name).collect::<Vec<_>>(),
            r.debuginfo.iter().map(|p| &p.name).collect::<Vec<_>>(),
        );
        assert_resolved_shape(&r);
        // The kernel stream must be paired with its modules-extra
        // subpackage (virtio_console.ko lives there).
        assert_eq!(r.packages.len(), 2, "expected kernel + modules-extra");
        assert!(r.packages[1].name.ends_with("-modules-extra"));
        assert_urls_exist(&r);
    }
}
