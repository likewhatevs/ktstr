//! Repo-metadata resolution: from a distro spec to concrete kernel
//! package URLs + sha256 checksums, pulled from each distro's official
//! repository metadata.
//!
//! This module resolves metadata ONLY — it never downloads the kernel
//! packages themselves. Extraction of the resolved packages and the
//! download/cache integration are handled elsewhere; the seam between
//! them is the [`ResolvedDistroKernel`] this module returns.
//!
//! # Endpoint choices
//!
//! - **Fedora** — latest stable release number from
//!   `https://fedoraproject.org/releases.json` (official, stable JSON,
//!   served by fedoraproject.org). It lists only shipped GA releases
//!   with numeric `version` fields, so filtering to all-digit versions
//!   and taking the max yields the current stable release without the
//!   Rawhide/Branched noise a mirror directory listing would carry.
//!   Kernel packages come from the `updates` repo (falling back to the
//!   GA `releases/.../os` repo), debuginfo from the matching `debug`
//!   repo. Package selection is by RPM EVR (`rpmvercmp`).
//! - **Ubuntu** — latest LTS from
//!   `https://changelogs.ubuntu.com/meta-release-lts` (official, plain
//!   text, carries the codename). Kernel = the HWE stack: the
//!   `linux-image-generic-hwe-{YY.MM}` meta package's `Depends` pins
//!   the concrete `linux-image-{kver}-generic`, whose modules live in
//!   `linux-modules{,-extra}-{kver}-generic`. debuginfo is the
//!   `-dbgsym` ddeb from `ddebs.ubuntu.com`.
//! - **Amazon Linux 2023** — repo base from
//!   `https://cdn.amazonlinux.com/al2023/core/mirrors/latest/{arch}/mirror.list`.
//!   AL2023 carries parallel kernel STREAMS as differently-named
//!   packages — the default `kernel` (6.1) plus versioned
//!   `kernel6.12` / `kernel6.18` — with no Fedora-style
//!   kernel-core/modules split; each stream splits a handful of extra
//!   drivers, `virtio_console` among them, into a
//!   `<stream>-modules-extra` subpackage (verified against the live
//!   filelists metadata). The resolver picks the stream with the
//!   newest kernel by RPM EVR and pairs it with its modules-extra at
//!   the same EVR; debuginfo (`<stream>-debuginfo`, which carries the
//!   full vmlinux on its own) comes from the sibling `debuginfo`
//!   mirror.
//!
//! All metadata fetches ride the shared HTTP client and transient-retry
//! seam via `crate::fetch::fetch_metadata_bytes` /
//! `crate::fetch::fetch_metadata_text`. Parsing/selection is factored
//! into pure functions over `&[u8]` / `&str` so the unit tests exercise
//! them on small fixtures without touching the network.

use std::cmp::Ordering;
use std::io::{BufReader, Cursor, Read};

use anyhow::{Context, Result, anyhow, bail};
use quick_xml::events::BytesStart;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use reqwest::Url;
use sha2::{Digest, Sha256};

pub use crate::kernel_path::DistroKind;

/// A single resolved package: name, version, download URL, and the
/// sha256 the repo metadata declares for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageRef {
    pub name: String,
    pub version: String,
    pub url: String,
    pub sha256: String,
    pub size: Option<u64>,
}

/// The resolved kernel for a distro spec: the packages that carry the
/// kernel image + modules, plus the vmlinux-bearing debuginfo packages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedDistroKernel {
    /// Short distro tag, e.g. `"fedora44"`, `"ubuntu24.04-hwe"`, `"al2023"`.
    pub distro: String,
    /// Package version-release, best effort (e.g. `"7.1.3-200.fc44"`).
    pub kernel_release: String,
    /// Distro package architecture (`x86_64`/`aarch64` for Fedora/AL,
    /// `amd64`/`arm64` for Ubuntu).
    pub arch: String,
    /// Kernel image + modules packages, in download order.
    pub packages: Vec<PackageRef>,
    /// vmlinux-bearing debuginfo packages.
    pub debuginfo: Vec<PackageRef>,
}

/// Resolve a distro spec to concrete kernel package URLs + checksums
/// for the host architecture.
///
/// `release` pins a specific stream (`Some("44")`, `Some("24.04")`);
/// `None` resolves the distro's latest. For Amazon Linux the release
/// selects the AL2023 stream (always latest-dated) regardless of value.
pub fn resolve_distro_kernel(
    kind: DistroKind,
    release: Option<&str>,
) -> Result<ResolvedDistroKernel> {
    let (arch, _) = crate::fetch::arch_info();
    resolve_for_arch(kind, release, arch)
}

/// Architecture-injectable core of [`resolve_distro_kernel`]. `arch` is
/// the host arch string from [`crate::fetch::arch_info`]
/// (`x86_64`/`aarch64`); the resolvers map it to each distro's naming.
/// Exposed to the in-module tests so they can resolve `aarch64`
/// metadata from an `x86_64` host (metadata-only, no download).
pub(crate) fn resolve_for_arch(
    kind: DistroKind,
    release: Option<&str>,
    arch: &str,
) -> Result<ResolvedDistroKernel> {
    match kind {
        DistroKind::Fedora => resolve_fedora(release, arch),
        DistroKind::Ubuntu => resolve_ubuntu(release, arch),
        DistroKind::AmazonLinux => resolve_amazonlinux(release, arch),
    }
}

// ------------------------------------------------------------------
// RPM EVR comparison (rpmvercmp)
// ------------------------------------------------------------------

/// An RPM epoch:version-release, kept as strings so comparison uses the
/// documented `rpmvercmp` segment algorithm rather than lossy parsing.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Evr {
    epoch: String,
    ver: String,
    rel: String,
}

impl Evr {
    /// `version-release`, prefixed with `epoch:` only when the epoch is
    /// not the default `0`.
    fn display(&self) -> String {
        if self.epoch.is_empty() || self.epoch == "0" {
            format!("{}-{}", self.ver, self.rel)
        } else {
            format!("{}:{}-{}", self.epoch, self.ver, self.rel)
        }
    }

    fn cmp(&self, other: &Evr) -> Ordering {
        let ea: i64 = self.epoch.parse().unwrap_or(0);
        let eb: i64 = other.epoch.parse().unwrap_or(0);
        ea.cmp(&eb)
            .then_with(|| rpmvercmp(&self.ver, &other.ver))
            .then_with(|| rpmvercmp(&self.rel, &other.rel))
    }
}

/// Compare two RPM version (or release) strings per the `rpmvercmp`
/// algorithm from `rpm`'s `lib/rpmvercmp.c`: split into maximal
/// all-alpha / all-numeric segments across separator characters,
/// compare numeric segments numerically (ignoring leading zeros) and
/// alpha segments lexically, with `~` sorting before everything (even
/// the empty string) and `^` sorting before an ended string but after
/// any real segment.
fn rpmvercmp(one: &str, two: &str) -> Ordering {
    if one == two {
        return Ordering::Equal;
    }
    let a = one.as_bytes();
    let b = two.as_bytes();
    let is_sep = |c: u8| !c.is_ascii_alphanumeric() && c != b'~' && c != b'^';
    let (mut i, mut j) = (0usize, 0usize);
    loop {
        while i < a.len() && is_sep(a[i]) {
            i += 1;
        }
        while j < b.len() && is_sep(b[j]) {
            j += 1;
        }

        // `~` sorts before everything, including the end of the string.
        let at = i < a.len() && a[i] == b'~';
        let bt = j < b.len() && b[j] == b'~';
        if at || bt {
            if !at {
                return Ordering::Greater;
            }
            if !bt {
                return Ordering::Less;
            }
            i += 1;
            j += 1;
            continue;
        }

        // `^` sorts before an ended string but after any real segment.
        let ac = i < a.len() && a[i] == b'^';
        let bc = j < b.len() && b[j] == b'^';
        if ac || bc {
            if i >= a.len() {
                return Ordering::Less;
            }
            if j >= b.len() {
                return Ordering::Greater;
            }
            if !ac {
                return Ordering::Greater;
            }
            if !bc {
                return Ordering::Less;
            }
            i += 1;
            j += 1;
            continue;
        }

        if i >= a.len() || j >= b.len() {
            break;
        }

        let isnum = a[i].is_ascii_digit();
        let s1 = take_segment(a, &mut i, isnum);
        let s2 = take_segment(b, &mut j, isnum);
        // A numeric segment always outranks an alpha (or empty) one.
        if s2.is_empty() {
            return if isnum {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        let cmp = if isnum {
            let n1 = trim_leading_zeros(s1);
            let n2 = trim_leading_zeros(s2);
            n1.len().cmp(&n2.len()).then_with(|| n1.cmp(n2))
        } else {
            s1.cmp(s2)
        };
        if cmp != Ordering::Equal {
            return cmp;
        }
    }

    // All compared segments were equal; whichever string still has
    // characters left is the newer one.
    match (i >= a.len(), j >= b.len()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => Ordering::Equal,
    }
}

/// Consume the maximal all-numeric (or all-alpha) run at `*idx` and
/// return it, advancing `idx` past it.
fn take_segment<'a>(s: &'a [u8], idx: &mut usize, numeric: bool) -> &'a [u8] {
    let start = *idx;
    while *idx < s.len()
        && ((numeric && s[*idx].is_ascii_digit()) || (!numeric && s[*idx].is_ascii_alphabetic()))
    {
        *idx += 1;
    }
    &s[start..*idx]
}

fn trim_leading_zeros(s: &[u8]) -> &[u8] {
    let mut k = 0;
    while k < s.len() && s[k] == b'0' {
        k += 1;
    }
    &s[k..]
}

// ------------------------------------------------------------------
// Debian version comparison (dpkg verrevcmp)
// ------------------------------------------------------------------

/// Compare two full Debian version strings (`[epoch:]upstream[-revision]`)
/// per dpkg's algorithm: compare epoch numerically, then the upstream
/// and revision parts with [`deb_verrevcmp`].
fn deb_version_cmp(a: &str, b: &str) -> Ordering {
    let (ea, ua, ra) = split_deb_version(a);
    let (eb, ub, rb) = split_deb_version(b);
    ea.cmp(&eb)
        .then_with(|| deb_verrevcmp(ua, ub))
        .then_with(|| deb_verrevcmp(ra, rb))
}

/// Split `[epoch:]upstream[-revision]` into `(epoch, upstream, revision)`.
fn split_deb_version(v: &str) -> (u64, &str, &str) {
    let (epoch, rest) = match v.split_once(':') {
        Some((e, r)) if !e.is_empty() && e.bytes().all(|c| c.is_ascii_digit()) => {
            (e.parse().unwrap_or(0), r)
        }
        _ => (0, v),
    };
    let (upstream, revision) = match rest.rsplit_once('-') {
        Some((u, r)) => (u, r),
        None => (rest, ""),
    };
    (epoch, upstream, revision)
}

/// Port of dpkg's `verrevcmp`: alternating non-digit / digit passes,
/// with `~` sorting before everything (even the empty string) and
/// letters sorting before non-letter punctuation.
fn deb_verrevcmp(a: &str, b: &str) -> Ordering {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let (mut i, mut j) = (0usize, 0usize);
    // dpkg's `order()`: digits sort as 0, letters as their ASCII value,
    // `~` below everything, other punctuation above letters.
    let order = |c: Option<u8>| -> i32 {
        match c {
            Some(c) if c.is_ascii_digit() => 0,
            Some(c) if c.is_ascii_alphabetic() => c as i32,
            Some(b'~') => -1,
            Some(c) => c as i32 + 256,
            None => 0,
        }
    };
    while i < a.len() || j < b.len() {
        // Non-digit prefix comparison.
        while (i < a.len() && !a[i].is_ascii_digit()) || (j < b.len() && !b[j].is_ascii_digit()) {
            let ac = order(a.get(i).copied());
            let bc = order(b.get(j).copied());
            if ac != bc {
                return ac.cmp(&bc);
            }
            i += 1;
            j += 1;
        }
        while i < a.len() && a[i] == b'0' {
            i += 1;
        }
        while j < b.len() && b[j] == b'0' {
            j += 1;
        }
        let mut first_diff = 0i32;
        while i < a.len() && a[i].is_ascii_digit() && j < b.len() && b[j].is_ascii_digit() {
            if first_diff == 0 {
                first_diff = a[i] as i32 - b[j] as i32;
            }
            i += 1;
            j += 1;
        }
        if i < a.len() && a[i].is_ascii_digit() {
            return Ordering::Greater;
        }
        if j < b.len() && b[j].is_ascii_digit() {
            return Ordering::Less;
        }
        if first_diff != 0 {
            return first_diff.cmp(&0);
        }
    }
    Ordering::Equal
}

// ------------------------------------------------------------------
// repomd.xml + primary.xml (rpm) parsing
// ------------------------------------------------------------------

/// One `<data type=...>` entry from a `repomd.xml`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RepoData {
    data_type: String,
    href: String,
    checksum: String,
    checksum_type: String,
}

/// Parse `repomd.xml`, returning every `<data>` entry with its
/// location href and (sha-typed) checksum.
fn parse_repomd(xml: &[u8]) -> Result<Vec<RepoData>> {
    let mut reader = Reader::from_reader(BufReader::new(Cursor::new(xml)));
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut out = Vec::new();
    let mut cur: Option<RepoData> = None;
    let mut in_checksum = false;
    loop {
        match reader
            .read_event_into(&mut buf)
            .with_context(|| "parse repomd.xml")?
        {
            Event::Start(e) => match e.local_name().as_ref() {
                b"data" => {
                    cur = Some(RepoData {
                        data_type: attr(&e, b"type").unwrap_or_default(),
                        href: String::new(),
                        checksum: String::new(),
                        checksum_type: String::new(),
                    });
                }
                b"checksum" => {
                    if let Some(c) = cur.as_mut() {
                        c.checksum_type = attr(&e, b"type").unwrap_or_default();
                        in_checksum = true;
                    }
                }
                _ => {}
            },
            Event::Empty(e) => {
                if e.local_name().as_ref() == b"location"
                    && let Some(c) = cur.as_mut()
                {
                    c.href = attr(&e, b"href").unwrap_or_default();
                }
            }
            Event::Text(t) if in_checksum => {
                if let Some(c) = cur.as_mut() {
                    c.checksum = t.decode().map(|s| s.into_owned()).unwrap_or_default();
                }
                in_checksum = false;
            }
            Event::End(e) => {
                if e.local_name().as_ref() == b"data"
                    && let Some(c) = cur.take()
                {
                    out.push(c);
                }
                in_checksum = false;
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

/// A candidate RPM parsed out of a `primary.xml`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RpmCand {
    name: String,
    arch: String,
    evr: Evr,
    sha256: String,
    href: String,
    size: Option<u64>,
}

/// Stream a (decompressed) `primary.xml`, collecting every package
/// whose `<name>` satisfies `wanted`. Streaming keeps the multi-MB
/// metadata out of memory: each `<package>` is filtered as its
/// `</package>` closes rather than materializing a DOM.
fn parse_primary<R: std::io::BufRead, F: Fn(&str) -> bool>(
    reader: R,
    wanted: F,
) -> Result<Vec<RpmCand>> {
    let mut reader = Reader::from_reader(reader);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut out = Vec::new();
    let mut cur: Option<RpmCand> = None;
    // Which text-bearing field the next `Event::Text` fills in.
    let mut pending: Option<Field> = None;
    loop {
        match reader
            .read_event_into(&mut buf)
            .with_context(|| "parse primary.xml")?
        {
            Event::Start(e) => match e.local_name().as_ref() {
                b"package" => cur = Some(RpmCand::default_cand()),
                b"name" if cur.is_some() => pending = Some(Field::Name),
                b"arch" if cur.is_some() => pending = Some(Field::Arch),
                b"checksum" if cur.is_some() => {
                    let c = cur.as_ref().unwrap();
                    if c.sha256.is_empty() && attr(&e, b"type").as_deref() == Some("sha256") {
                        pending = Some(Field::Checksum);
                    }
                }
                _ => apply_attr_element(cur.as_mut(), e.local_name().as_ref(), &e),
            },
            Event::Empty(e) => apply_attr_element(cur.as_mut(), e.local_name().as_ref(), &e),
            Event::Text(t) if pending.is_some() && cur.is_some() => {
                let val = t.decode().map(|s| s.into_owned()).unwrap_or_default();
                let c = cur.as_mut().unwrap();
                match pending.take().unwrap() {
                    Field::Name if c.name.is_empty() => c.name = val,
                    Field::Arch if c.arch.is_empty() => c.arch = val,
                    Field::Checksum if c.sha256.is_empty() => c.sha256 = val,
                    _ => {}
                }
            }
            Event::End(e) => {
                if e.local_name().as_ref() == b"package"
                    && let Some(c) = cur.take()
                    && wanted(&c.name)
                {
                    out.push(c);
                }
                pending = None;
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

enum Field {
    Name,
    Arch,
    Checksum,
}

impl RpmCand {
    fn default_cand() -> Self {
        RpmCand {
            name: String::new(),
            arch: String::new(),
            evr: Evr {
                epoch: String::new(),
                ver: String::new(),
                rel: String::new(),
            },
            sha256: String::new(),
            href: String::new(),
            size: None,
        }
    }
}

/// Fill the attribute-only `<version>`, `<location>`, and `<size>`
/// elements (self-closing in primary.xml, so they arrive as `Start`
/// with no children or `Empty`). First-wins: the top-level elements
/// precede the `<format>` block whose `rpm:entry` children reuse
/// `ver`/`name` attribute spellings under different element names.
fn apply_attr_element(cur: Option<&mut RpmCand>, local: &[u8], e: &BytesStart) {
    let Some(c) = cur else { return };
    match local {
        b"version" if c.evr.ver.is_empty() => {
            c.evr.epoch = attr(e, b"epoch").unwrap_or_else(|| "0".into());
            c.evr.ver = attr(e, b"ver").unwrap_or_default();
            c.evr.rel = attr(e, b"rel").unwrap_or_default();
        }
        b"location" if c.href.is_empty() => {
            c.href = attr(e, b"href").unwrap_or_default();
        }
        b"size" if c.size.is_none() => {
            c.size = attr(e, b"package").and_then(|s| s.parse().ok());
        }
        _ => {}
    }
}

/// Read attribute `key` off `e` as an owned, entity-unescaped `String`.
fn attr(e: &BytesStart, key: &[u8]) -> Option<String> {
    e.try_get_attribute(key)
        .ok()
        .flatten()
        .and_then(|a| a.unescape_value().ok().map(|c| c.into_owned()))
}

// ------------------------------------------------------------------
// Debian Packages + text-metadata parsing
// ------------------------------------------------------------------

/// One binary-package stanza from a Debian `Packages` file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DebPkg {
    package: String,
    version: String,
    filename: String,
    sha256: String,
    size: Option<u64>,
    depends: Vec<String>,
}

/// Parse a decompressed Debian `Packages` file into stanzas. Only the
/// fields the resolver consumes are extracted; folded continuation
/// lines (never used by these fields) are ignored.
fn parse_deb_packages(text: &str) -> Vec<DebPkg> {
    let mut out = Vec::new();
    for stanza in text.split("\n\n") {
        if stanza.trim().is_empty() {
            continue;
        }
        let mut p = DebPkg {
            package: String::new(),
            version: String::new(),
            filename: String::new(),
            sha256: String::new(),
            size: None,
            depends: Vec::new(),
        };
        for line in stanza.lines() {
            let Some((key, val)) = line.split_once(": ") else {
                continue;
            };
            match key {
                "Package" => p.package = val.trim().to_string(),
                "Version" => p.version = val.trim().to_string(),
                "Filename" => p.filename = val.trim().to_string(),
                "SHA256" => p.sha256 = val.trim().to_string(),
                "Size" => p.size = val.trim().parse().ok(),
                "Depends" => {
                    p.depends = val
                        .split(',')
                        // Take the bare package name of each dependency,
                        // dropping any `(>= x)` version constraint and
                        // `pkg:arch` qualifier.
                        .filter_map(|d| d.split_whitespace().next())
                        .map(|d| d.split(':').next().unwrap_or(d).to_string())
                        .collect();
                }
                _ => {}
            }
        }
        if !p.package.is_empty() {
            out.push(p);
        }
    }
    out
}

/// Newest stanza named `name` by Debian version order, if any.
fn newest_deb<'a>(pkgs: &'a [DebPkg], name: &str) -> Option<&'a DebPkg> {
    pkgs.iter()
        .filter(|p| p.package == name)
        .max_by(|a, b| deb_version_cmp(&a.version, &b.version))
}

/// One release row from `meta-release-lts`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct UbuntuRelease {
    dist: String,
    version: String,
    supported: bool,
}

/// Parse `meta-release-lts` (blank-line-separated stanzas of
/// `Key: value` lines) into release rows.
fn parse_meta_release_lts(text: &str) -> Vec<UbuntuRelease> {
    let mut out = Vec::new();
    for stanza in text.split("\n\n") {
        let mut dist = String::new();
        let mut version = String::new();
        let mut supported = false;
        for line in stanza.lines() {
            let Some((key, val)) = line.split_once(": ") else {
                continue;
            };
            match key {
                "Dist" => dist = val.trim().to_string(),
                "Version" => version = val.trim().to_string(),
                "Supported" => supported = val.trim() == "1",
                _ => {}
            }
        }
        if !dist.is_empty() {
            out.push(UbuntuRelease {
                dist,
                version,
                supported,
            });
        }
    }
    out
}

/// `major.minor` of a `meta-release-lts` `Version` string
/// (e.g. `"24.04.1 LTS"` -> `(24, 4)`), for LTS ordering.
fn lts_version_key(version: &str) -> Option<(u32, u32)> {
    let head = version.split_whitespace().next()?;
    let mut it = head.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

/// Newest supported LTS `(version, codename)` from the parsed rows,
/// where `version` is normalized to `major.minor` (e.g. `"24.04"`).
fn latest_lts(rels: &[UbuntuRelease]) -> Option<(String, String)> {
    rels.iter()
        .filter(|r| r.supported)
        .filter_map(|r| lts_version_key(&r.version).map(|k| (k, r)))
        .max_by_key(|(k, _)| *k)
        .map(|((major, minor), r)| (format!("{major}.{minor:02}"), r.dist.clone()))
}

/// Codename for a pinned `major.minor` version (e.g. `"24.04"`).
fn codename_for_version(rels: &[UbuntuRelease], version: &str) -> Option<String> {
    rels.iter()
        .find(|r| lts_version_key(&r.version) == lts_version_key(version))
        .map(|r| r.dist.clone())
}

/// Offline fallback: version -> codename for the LTS releases the live
/// `meta-release-lts` map should also carry. Used only when the network
/// map lacks the pinned version (and by the offline unit tests).
fn fallback_codename(version: &str) -> Option<&'static str> {
    match version {
        "24.04" => Some("noble"),
        "22.04" => Some("jammy"),
        "20.04" => Some("focal"),
        _ => None,
    }
}

/// First non-empty line of an Amazon Linux `mirror.list`, normalized to
/// end in `/` so it composes as a repo base URL.
fn parse_mirror_list(text: &str) -> Option<String> {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    Some(if line.ends_with('/') {
        line.to_string()
    } else {
        format!("{line}/")
    })
}

// ------------------------------------------------------------------
// URL joining + metadata decompression + checksum
// ------------------------------------------------------------------

/// Resolve `href` (which may be relative, including `../` traversal, as
/// AL2023's blobstore hrefs are) against `base` per RFC 3986.
fn join_url(base: &str, href: &str) -> Result<String> {
    Url::parse(base)
        .with_context(|| format!("parse repo base url {base}"))?
        .join(href)
        .with_context(|| format!("join {href} onto {base}"))
        .map(|u| u.to_string())
}

/// Wrap `bytes` in the decompressor its metadata filename extension
/// implies (`.gz` / `.zst` / `.xz`, or plain for `.xml`).
fn decompress_metadata<'a>(href: &str, bytes: &'a [u8]) -> Result<Box<dyn Read + 'a>> {
    let cursor = Cursor::new(bytes);
    if href.ends_with(".gz") {
        Ok(Box::new(flate2::read::GzDecoder::new(cursor)))
    } else if href.ends_with(".zst") {
        Ok(Box::new(
            zstd::Decoder::new(cursor).with_context(|| "init zstd decoder")?,
        ))
    } else if href.ends_with(".xz") {
        Ok(Box::new(xz2::read::XzDecoder::new(cursor)))
    } else if href.ends_with(".xml") {
        Ok(Box::new(cursor))
    } else {
        bail!("unsupported metadata compression for {href}")
    }
}

/// Verify `bytes` hash against a repomd-declared sha256 (case-insensitive).
fn verify_metadata_sha256(bytes: &[u8], expected: &str, href: &str) -> Result<()> {
    let mut h = Sha256::new();
    h.update(bytes);
    let actual = hex::encode(h.finalize());
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        bail!("sha256 mismatch for {href}: expected {expected}, got {actual}")
    }
}

/// Fetch `repodata/repomd.xml` under `base`, then the `type="primary"`
/// metadata it references, verify the compressed file's declared
/// sha256, and stream-parse it for the `wanted` package names.
fn fetch_repo_candidates(base: &str, wanted: &[&str]) -> Result<Vec<RpmCand>> {
    fetch_repo_candidates_where(base, |name| wanted.contains(&name))
}

/// Predicate-driven form of [`fetch_repo_candidates`], for resolvers
/// whose package names are not known up front (AL2023's dynamic
/// `kernel<X>.<Y>` stream names).
fn fetch_repo_candidates_where<F: Fn(&str) -> bool>(base: &str, wanted: F) -> Result<Vec<RpmCand>> {
    let repomd_url = join_url(base, "repodata/repomd.xml")?;
    let repomd = crate::fetch::fetch_metadata_bytes(&repomd_url, "fetch")?;
    let datas = parse_repomd(&repomd)?;
    let primary = datas
        .iter()
        .find(|d| d.data_type == "primary")
        .ok_or_else(|| anyhow!("repomd.xml at {repomd_url} has no primary metadata"))?;
    let primary_url = join_url(base, &primary.href)?;
    let compressed = crate::fetch::fetch_metadata_bytes(&primary_url, "fetch")?;
    if primary.checksum_type == "sha256" && !primary.checksum.is_empty() {
        verify_metadata_sha256(&compressed, &primary.checksum, &primary.href)?;
    }
    let reader = BufReader::new(decompress_metadata(&primary.href, &compressed)?);
    parse_primary(reader, wanted)
}

/// Newest EVR among `cands` named `name`.
fn newest_evr(cands: &[RpmCand], name: &str) -> Option<Evr> {
    cands
        .iter()
        .filter(|c| c.name == name)
        .max_by(|a, b| a.evr.cmp(&b.evr))
        .map(|c| c.evr.clone())
}

/// Build [`PackageRef`]s for `names` (in order) at exactly `target`
/// EVR. Missing names are skipped when `strict` is false; when `strict`
/// (debuginfo — mandatory in this design — and package sets where every
/// member is load-bearing, like AL2023's kernel + modules-extra pair) a
/// missing name is a hard error.
fn build_package_set(
    cands: &[RpmCand],
    base: &str,
    names: &[&str],
    target: &Evr,
    strict: bool,
) -> Result<Vec<PackageRef>> {
    let mut out = Vec::new();
    for name in names {
        match cands.iter().find(|c| c.name == *name && c.evr == *target) {
            Some(c) => out.push(PackageRef {
                name: c.name.clone(),
                version: c.evr.display(),
                url: join_url(base, &c.href)?,
                sha256: c.sha256.clone(),
                size: c.size,
            }),
            None if strict => bail!(
                "{name}-{} not found in repo metadata at {base}",
                target.display()
            ),
            None => {}
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------
// Per-distro resolvers
// ------------------------------------------------------------------

const FEDORA_RELEASES_JSON: &str = "https://fedoraproject.org/releases.json";
const FEDORA_MIRROR_BASE: &str = "https://dl.fedoraproject.org/pub/fedora/linux";

/// Latest stable Fedora release number from `releases.json`: the max
/// all-numeric `version` field (GA releases only; Rawhide/Branched carry
/// non-numeric or absent versions and are filtered out).
fn fedora_latest_release() -> Result<String> {
    let body = crate::fetch::fetch_metadata_text(FEDORA_RELEASES_JSON, "fetch")?;
    let latest = fedora_latest_from_json(&body)?;
    Ok(latest)
}

fn fedora_latest_from_json(body: &str) -> Result<String> {
    let json: serde_json::Value =
        serde_json::from_str(body).with_context(|| "parse fedora releases.json")?;
    json.as_array()
        .ok_or_else(|| anyhow!("fedora releases.json: expected top-level array"))?
        .iter()
        .filter_map(|r| r.get("version").and_then(|v| v.as_str()))
        .filter_map(|v| v.parse::<u32>().ok())
        .max()
        .map(|n| n.to_string())
        .ok_or_else(|| anyhow!("fedora releases.json: no numeric release version found"))
}

fn resolve_fedora(release: Option<&str>, arch: &str) -> Result<ResolvedDistroKernel> {
    let rel = match release {
        Some(r) => r.to_string(),
        None => fedora_latest_release()?,
    };
    let common = format!("kernel-debuginfo-common-{arch}");
    let pkg_names = ["kernel-core", "kernel-modules-core", "kernel-modules"];

    // Prefer the updates repo; fall back to the GA releases repo if it
    // carries no kernel-core (fresh release before its first update).
    let updates_base = format!("{FEDORA_MIRROR_BASE}/updates/{rel}/Everything/{arch}/");
    let mut base = updates_base.clone();
    let mut cands = fetch_repo_candidates(&base, &pkg_names)?;
    let mut debug_base = format!("{FEDORA_MIRROR_BASE}/updates/{rel}/Everything/{arch}/debug/");
    if newest_evr(&cands, "kernel-core").is_none() {
        base = format!("{FEDORA_MIRROR_BASE}/releases/{rel}/Everything/{arch}/os/");
        cands = fetch_repo_candidates(&base, &pkg_names)?;
        debug_base = format!("{FEDORA_MIRROR_BASE}/releases/{rel}/Everything/{arch}/debug/tree/");
    }

    let target = newest_evr(&cands, "kernel-core")
        .ok_or_else(|| anyhow!("no kernel-core found in Fedora {rel} repos for {arch}"))?;
    let packages = build_package_set(&cands, &base, &pkg_names, &target, false)?;

    let debug_names = ["kernel-debuginfo", common.as_str()];
    let debug_cands = fetch_repo_candidates(&debug_base, &debug_names)?;
    let debuginfo = build_package_set(&debug_cands, &debug_base, &debug_names, &target, true)?;

    Ok(ResolvedDistroKernel {
        distro: format!("fedora{rel}"),
        kernel_release: target.display(),
        arch: arch.to_string(),
        packages,
        debuginfo,
    })
}

const UBUNTU_META_RELEASE_LTS: &str = "https://changelogs.ubuntu.com/meta-release-lts";
const UBUNTU_ARCHIVE: &str = "https://archive.ubuntu.com/ubuntu/";
const UBUNTU_PORTS: &str = "https://ports.ubuntu.com/ubuntu-ports/";
const UBUNTU_DDEBS: &str = "http://ddebs.ubuntu.com/";

/// Map the host arch string to Ubuntu's package architecture.
fn ubuntu_arch(arch: &str) -> Result<&'static str> {
    match arch {
        "x86_64" => Ok("amd64"),
        "aarch64" => Ok("arm64"),
        other => bail!("unsupported arch for Ubuntu: {other}"),
    }
}

fn resolve_ubuntu(release: Option<&str>, arch: &str) -> Result<ResolvedDistroKernel> {
    let deb_arch = ubuntu_arch(arch)?;
    // amd64 lives on archive.ubuntu.com; every other port on ports.ubuntu.com.
    let archive_base = if deb_arch == "amd64" {
        UBUNTU_ARCHIVE
    } else {
        UBUNTU_PORTS
    };

    let meta = crate::fetch::fetch_metadata_text(UBUNTU_META_RELEASE_LTS, "fetch")?;
    let rels = parse_meta_release_lts(&meta);
    let (version, codename) = match release {
        Some(v) => {
            let codename = codename_for_version(&rels, v)
                .or_else(|| fallback_codename(v).map(str::to_string))
                .ok_or_else(|| anyhow!("no Ubuntu LTS codename for {v}"))?;
            (v.to_string(), codename)
        }
        None => latest_lts(&rels)
            .ok_or_else(|| anyhow!("meta-release-lts: no supported LTS release found"))?,
    };

    let debs = fetch_deb_packages(archive_base, &codename, deb_arch)?;
    let meta_name = format!("linux-image-generic-hwe-{version}");
    let meta_pkg = newest_deb(&debs, &meta_name).ok_or_else(|| {
        anyhow!("HWE meta {meta_name} not found in {codename}-updates/{deb_arch}")
    })?;
    let kver = hwe_kver(meta_pkg)
        .ok_or_else(|| anyhow!("{meta_name} Depends has no linux-image-*-generic"))?;

    // The HWE meta's Depends chain: linux-image (vmlinuz) ->
    // linux-modules, plus linux-modules-extra (the meta depends on it
    // directly). Include all three so no in-tree module the guest might
    // need is left out; the extraction stage decides what to unpack.
    let pkg_names = [
        format!("linux-image-{kver}-generic"),
        format!("linux-modules-{kver}-generic"),
        format!("linux-modules-extra-{kver}-generic"),
    ];
    let packages = ubuntu_package_refs(&debs, archive_base, &pkg_names, false)?;

    // Debuginfo: the -dbgsym ddeb carries vmlinux. The unsigned image's
    // dbgsym is the canonical vmlinux carrier; fall back to the signed
    // one. ddebs can lag a kernel publish by hours — a miss is a hard
    // error since debuginfo is mandatory here.
    let ddebs = fetch_deb_packages(UBUNTU_DDEBS, &codename, deb_arch)?;
    let dbg_names = [
        format!("linux-image-unsigned-{kver}-generic-dbgsym"),
        format!("linux-image-{kver}-generic-dbgsym"),
    ];
    let debuginfo = ubuntu_dbgsym_refs(&ddebs, &dbg_names, &kver, &codename, deb_arch)?;

    Ok(ResolvedDistroKernel {
        distro: format!("ubuntu{version}-hwe"),
        kernel_release: meta_pkg.version.clone(),
        arch: deb_arch.to_string(),
        packages,
        debuginfo,
    })
}

/// Fetch and parse `{codename}-updates` `Packages.gz` for `deb_arch`.
fn fetch_deb_packages(base: &str, codename: &str, deb_arch: &str) -> Result<Vec<DebPkg>> {
    let url = format!("{base}dists/{codename}-updates/main/binary-{deb_arch}/Packages.gz");
    let gz = crate::fetch::fetch_metadata_bytes(&url, "fetch")?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(Cursor::new(&gz))
        .read_to_string(&mut text)
        .with_context(|| format!("gunzip {url}"))?;
    Ok(parse_deb_packages(&text))
}

/// Extract the `{kver}` (e.g. `6.17.0-35`) from an HWE meta's
/// `linux-image-{kver}-generic` dependency.
fn hwe_kver(meta: &DebPkg) -> Option<String> {
    meta.depends
        .iter()
        .find(|d| d.starts_with("linux-image-") && d.ends_with("-generic"))
        .and_then(|d| d.strip_prefix("linux-image-"))
        .and_then(|d| d.strip_suffix("-generic"))
        .map(str::to_string)
}

/// Resolve concrete Ubuntu package stanzas to [`PackageRef`]s.
fn ubuntu_package_refs(
    debs: &[DebPkg],
    base: &str,
    names: &[String],
    strict: bool,
) -> Result<Vec<PackageRef>> {
    let mut out = Vec::new();
    for name in names {
        match newest_deb(debs, name) {
            Some(p) => out.push(deb_package_ref(p, base)?),
            None if strict => bail!("{name} not found in repo metadata"),
            None => {}
        }
    }
    Ok(out)
}

/// Pick the first available dbgsym name (unsigned preferred), erroring
/// with a clear message when none is present.
fn ubuntu_dbgsym_refs(
    ddebs: &[DebPkg],
    names: &[String],
    kver: &str,
    codename: &str,
    deb_arch: &str,
) -> Result<Vec<PackageRef>> {
    for name in names {
        if let Some(p) = newest_deb(ddebs, name) {
            return Ok(vec![deb_package_ref(p, UBUNTU_DDEBS)?]);
        }
    }
    bail!(
        "no dbgsym ddeb for linux-image-{kver}-generic in \
         {codename}-updates/{deb_arch} on ddebs.ubuntu.com — debuginfo \
         is mandatory; ddebs may lag a kernel publish by a few hours"
    )
}

fn deb_package_ref(p: &DebPkg, base: &str) -> Result<PackageRef> {
    Ok(PackageRef {
        name: p.package.clone(),
        version: p.version.clone(),
        url: join_url(base, &p.filename)?,
        sha256: p.sha256.clone(),
        size: p.size,
    })
}

const AL2023_MIRROR: &str = "https://cdn.amazonlinux.com/al2023/core/mirrors/latest";

/// The AL2023 kernel-stream family a package name belongs to: the
/// default `kernel` (6.1) stream or a versioned `kernel<MAJ>.<MIN>`
/// stream (`kernel6.12`, `kernel6.18`), each with a `-modules-extra`
/// subpackage. Returns `Some(stream)` for the bare stream package and
/// its modules-extra, `None` for every other kernel-adjacent name
/// (`-devel`, `-headers`, `-tools`, `-libbpf*`, `-livepatch-*`,
/// `-modules-extra-common`, …).
fn al2023_stream(name: &str) -> Option<&str> {
    let stream = name.strip_suffix("-modules-extra").unwrap_or(name);
    let rest = stream.strip_prefix("kernel")?;
    if rest.is_empty() {
        return Some(stream);
    }
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let (major, minor) = rest.split_once('.')?;
    (digits(major) && digits(minor)).then_some(stream)
}

/// Pick the AL2023 stream with the newest kernel, as `(stream, evr)`.
/// Selection is two-level: WITHIN a stream, the newest build by full
/// RPM EVR (epoch included — the live repo carries epoch 0 and 1
/// builds under one stream name); ACROSS streams, by the kernel
/// version alone (`rpmvercmp` on `ver`: `6.18.36` > `6.1.176`).
/// Epoch and release only order builds of the SAME package name, so
/// letting a whole-EVR compare span streams would let an epoch bump
/// in an old stream outrank a newer stream's kernel.
fn al2023_pick_stream(cands: &[RpmCand]) -> Option<(&str, Evr)> {
    let streams: std::collections::BTreeSet<&str> = cands
        .iter()
        .filter(|c| al2023_stream(&c.name) == Some(c.name.as_str()))
        .map(|c| c.name.as_str())
        .collect();
    streams
        .into_iter()
        .filter_map(|s| newest_evr(cands, s).map(|evr| (s, evr)))
        .max_by(|(_, a), (_, b)| rpmvercmp(&a.ver, &b.ver).then_with(|| a.cmp(b)))
}

fn resolve_amazonlinux(_release: Option<&str>, arch: &str) -> Result<ResolvedDistroKernel> {
    // AL2023 is a single rolling stream; `release` (Some("2023")/None)
    // always resolves to the latest dated build.
    let core_base = al2023_repo_base(&format!("{AL2023_MIRROR}/{arch}/mirror.list"))?;
    // Resolve every kernel stream (`kernel`, `kernel6.12`, `kernel6.18`,
    // …) plus each stream's `-modules-extra` subpackage — which carries
    // `virtio_console.ko`, without which ktstr's console cannot come up
    // — and pick the stream with the newest kernel. Both packages are
    // strict: a stream missing its modules-extra at the kernel's exact
    // EVR cannot boot under ktstr.
    let cands = fetch_repo_candidates_where(&core_base, |n| al2023_stream(n).is_some())?;
    let (stream, target) = al2023_pick_stream(&cands)
        .ok_or_else(|| anyhow!("no kernel stream package in AL2023 {arch} repo"))?;
    let extra = format!("{stream}-modules-extra");
    let pkg_names = [stream, extra.as_str()];
    let packages = build_package_set(&cands, &core_base, &pkg_names, &target, true)?;

    let debug_base = al2023_repo_base(&format!("{AL2023_MIRROR}/debuginfo/{arch}/mirror.list"))?;
    // `<stream>-debuginfo` alone carries the complete vmlinux
    // (`/usr/lib/debug/lib/modules/<rel>/vmlinux`). The sibling
    // `-debuginfo-common-{arch}` is deliberately NOT fetched: it only
    // adds `/usr/src/debug` sources, plus a second
    // `<rel>.microvm/vmlinux.debug` flavor tree (verified against the
    // live filelists) that would make the extracted packages disagree
    // on the kernel release.
    let dbg = format!("{stream}-debuginfo");
    let debug_names = [dbg.as_str()];
    let debug_cands = fetch_repo_candidates(&debug_base, &debug_names)?;
    let debuginfo = build_package_set(&debug_cands, &debug_base, &debug_names, &target, true)?;

    Ok(ResolvedDistroKernel {
        distro: "al2023".to_string(),
        kernel_release: target.display(),
        arch: arch.to_string(),
        packages,
        debuginfo,
    })
}

fn al2023_repo_base(mirror_url: &str) -> Result<String> {
    let list = crate::fetch::fetch_metadata_text(mirror_url, "fetch")?;
    parse_mirror_list(&list).ok_or_else(|| anyhow!("empty mirror.list at {mirror_url}"))
}

#[cfg(test)]
mod tests;
