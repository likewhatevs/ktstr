//! Unit tests over the pure parsing/selection functions (small inline
//! fixtures, no network) plus one `#[ignore]` live smoke test per
//! distro that resolves against the real repos.

use super::*;
use std::cmp::Ordering;

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

fn assert_resolved_shape(r: &ResolvedDistroKernel) {
    assert!(!r.distro.is_empty());
    assert!(!r.kernel_release.is_empty());
    assert!(!r.packages.is_empty(), "no kernel packages resolved");
    assert!(!r.debuginfo.is_empty(), "no debuginfo resolved");
    for p in r.packages.iter().chain(r.debuginfo.iter()) {
        assert!(p.url.starts_with("http"), "bad url: {}", p.url);
        assert_eq!(
            p.sha256.len(),
            64,
            "sha256 not 64 hex: {} ({})",
            p.sha256,
            p.name
        );
        assert!(p.sha256.bytes().all(|c| c.is_ascii_hexdigit()));
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
    }
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
    }
}
