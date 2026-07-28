//! Official Google Kubernetes Engine COS kernel acquisition.
//!
//! GKE publishes its node-image revisions in the official GKE release
//! notes. Each promoted `cos-MILESTONE-BUILD-PATCH-REVISION` maps to an
//! official `cos-tools` board directory containing a bootable `vmlinux`,
//! prepared kernel headers, and the exact kernel source archive.
//!
//! The GKE kernel deliberately builds a few drivers ktstr needs as
//! modules or leaves them disabled. ktstr keeps the Google-published
//! kernel byte-for-byte and builds only those missing external modules
//! against the matching Google headers/source:
//!
//! - `virtio_mmio` (with command-line MMIO-device registration enabled)
//! - `virtio_console`
//! - `virtio_blk`
//! - `raid6_pq`
//! - `btrfs`
//!
//! The resulting cache entry uses the same atomic cache install and
//! ordered initramfs module loading as other prebuilt distro kernels.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::cache::KernelSource;
use crate::distro::acquire::{install_extracted, scratch_dir};
use crate::distro::extract::{ExtractedKernel, extract_tar, extract_tar_matching};

const GKE_RELEASE_NOTES_URL: &str =
    "https://docs.cloud.google.com/kubernetes-engine/docs/release-notes";
const COS_TOOLS_BUCKET: &str = "cos-tools";
const GKE_RESOLUTION_CACHE_VERSION: &str = "v1";
const GKE_ARTIFACT_BASENAMES: [&str; 3] = ["vmlinux", "kernel-headers.tgz", "kernel-src.tar.gz"];

const MODULE_MAKEFILE: &str = "\
ccflags-y += -DCONFIG_VIRTIO_MMIO_CMDLINE_DEVICES

obj-m += drivers/char/virtio_console.o
obj-m += drivers/virtio/virtio_mmio.o
obj-m += drivers/block/virtio_blk.o
obj-m += lib/raid6/
obj-m += fs/btrfs/
";

const SOURCE_FILES: &[&str] = &[
    "drivers/char/virtio_console.c",
    "drivers/virtio/virtio_mmio.c",
    "drivers/block/virtio_blk.c",
    "drivers/tty/hvc/hvc_console.h",
];

const SOURCE_DIRS: &[&str] = &["lib/raid6", "fs/btrfs"];

const MODULE_OUTPUTS: &[(&str, &str)] = &[
    ("CONFIG_VIRTIO_MMIO", "drivers/virtio/virtio_mmio.ko"),
    ("CONFIG_VIRTIO_CONSOLE", "drivers/char/virtio_console.ko"),
    ("CONFIG_VIRTIO_BLK", "drivers/block/virtio_blk.ko"),
    ("CONFIG_RAID6_PQ", "lib/raid6/raid6_pq.ko"),
    ("CONFIG_BTRFS_FS", "fs/btrfs/btrfs.ko"),
];

/// Capabilities supplied by the Google kernel itself. A future COS
/// config that drops one fails acquisition explicitly rather than
/// silently caching a reduced-capability kernel.
const REQUIRED_BUILTINS: &[&str] = &[
    "CONFIG_MODULES",
    "CONFIG_VIRTIO",
    "CONFIG_VIRTIO_PCI",
    "CONFIG_VIRTIO_NET",
    "CONFIG_XOR_BLOCKS",
    "CONFIG_SCHED_CLASS_EXT",
    "CONFIG_BPF_SYSCALL",
    "CONFIG_BPF_JIT",
    "CONFIG_DEBUG_INFO_BTF",
    "CONFIG_KALLSYMS_ALL",
    "CONFIG_IKCONFIG",
    "CONFIG_IKCONFIG_PROC",
    "CONFIG_SERIAL_8250_CONSOLE",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct GkeImage {
    milestone: u32,
    build: u32,
    patch: u32,
    revision: u32,
}

impl GkeImage {
    fn image_name(self) -> String {
        format!(
            "cos-{}-{}-{}-{}",
            self.milestone, self.build, self.patch, self.revision
        )
    }

    fn tools_version(self) -> String {
        format!("{}.{}.{}", self.build, self.patch, self.revision)
    }

    fn board_prefix(self) -> String {
        format!("{}/gke-amd64-gcp", self.tools_version())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct GkeArtifact {
    name: String,
    generation: String,
    size: u64,
    md5_base64: String,
    download_url: String,
}

/// One indivisible, validated GKE metadata lookup result.
///
/// Persisting the image and all three generation-pinned objects
/// together prevents a fallback from combining an older promoted image
/// with any freshly resolved artifact identity (or vice versa).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct GkeResolution {
    image: GkeImage,
    artifacts: [GkeArtifact; 3],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GcsObjectMetadata {
    name: String,
    generation: String,
    size: String,
    md5_hash: String,
}

fn parse_release_notes(notes: &str, milestone: Option<u32>) -> Result<GkeImage> {
    // The delimiter clauses reject derivative names such as
    // `cos-beta-*` and `cos-...-beta`; only complete promoted COS image
    // identifiers participate.
    let re = Regex::new(r"(?:^|[^A-Za-z0-9_-])cos-(\d{2,3})-(\d+)-(\d+)-(\d+)(?:[^A-Za-z0-9_-]|$)")
        .expect("static GKE image regex");
    let mut images = Vec::new();
    for caps in re.captures_iter(notes) {
        let image = GkeImage {
            milestone: caps[1].parse()?,
            build: caps[2].parse()?,
            patch: caps[3].parse()?,
            revision: caps[4].parse()?,
        };
        if milestone.is_none_or(|wanted| wanted == image.milestone) {
            images.push(image);
        }
    }
    images.into_iter().max().ok_or_else(|| {
        if let Some(milestone) = milestone {
            anyhow!("official GKE release notes contain no promoted cos-{milestone} node image")
        } else {
            anyhow!("official GKE release notes contain no promoted COS node image")
        }
    })
}

fn parse_milestone_selector(milestone: Option<&str>) -> Result<Option<u32>> {
    milestone
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("parse GKE COS milestone {value:?}"))
        })
        .transpose()
}

fn resolve_promoted_image(milestone: Option<u32>) -> Result<GkeImage> {
    let bytes =
        crate::fetch::fetch_metadata_bytes(GKE_RELEASE_NOTES_URL, "fetch GKE release notes")
            .with_context(|| "download official GKE release notes")?;
    let notes = String::from_utf8(bytes).context("GKE release notes were not UTF-8")?;
    parse_release_notes(&notes, milestone)
}

fn encode_gcs_object_name(name: &str) -> String {
    let mut encoded = String::with_capacity(name.len() + 16);
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 0xf) as usize]));
        }
    }
    encoded
}

fn gcs_download_url(object_name: &str, generation: &str) -> String {
    let encoded = encode_gcs_object_name(object_name);
    format!(
        "https://storage.googleapis.com/download/storage/v1/b/{COS_TOOLS_BUCKET}/o/{encoded}?generation={generation}&alt=media"
    )
}

fn parse_gcs_metadata(bytes: &[u8], expected_name: &str) -> Result<GkeArtifact> {
    let metadata: GcsObjectMetadata =
        serde_json::from_slice(bytes).context("parse GCS object metadata")?;
    ensure!(
        metadata.name == expected_name,
        "GCS metadata object mismatch: requested {expected_name:?}, got {:?}",
        metadata.name
    );
    let size = metadata
        .size
        .parse::<u64>()
        .context("parse GCS object size")?;
    ensure!(size > 0, "GCS object {expected_name} is empty");
    ensure!(
        !metadata.generation.is_empty() && metadata.generation.bytes().all(|b| b.is_ascii_digit()),
        "GCS object {expected_name} has invalid generation {:?}",
        metadata.generation
    );
    let md5 = BASE64_STANDARD
        .decode(&metadata.md5_hash)
        .with_context(|| format!("decode GCS md5Hash for {expected_name}"))?;
    ensure!(
        md5.len() == 16,
        "GCS object {expected_name} md5Hash decoded to {} bytes, expected 16",
        md5.len()
    );
    let download_url = gcs_download_url(expected_name, &metadata.generation);
    Ok(GkeArtifact {
        name: metadata.name,
        generation: metadata.generation,
        size,
        md5_base64: metadata.md5_hash,
        download_url,
    })
}

fn resolve_artifact(object_name: &str) -> Result<GkeArtifact> {
    let encoded = encode_gcs_object_name(object_name);
    let metadata_url =
        format!("https://storage.googleapis.com/storage/v1/b/{COS_TOOLS_BUCKET}/o/{encoded}");
    let bytes = crate::fetch::fetch_metadata_bytes(&metadata_url, "fetch COS artifact metadata")
        .with_context(|| format!("download metadata for {object_name}"))?;
    parse_gcs_metadata(&bytes, object_name)
}

fn artifact_object_names(image: GkeImage) -> [String; 3] {
    let prefix = image.board_prefix();
    GKE_ARTIFACT_BASENAMES.map(|basename| format!("{prefix}/{basename}"))
}

fn resolve_artifacts(image: GkeImage) -> Result<[GkeArtifact; 3]> {
    let [vmlinux, headers, source] = artifact_object_names(image);
    Ok([
        resolve_artifact(&vmlinux)?,
        resolve_artifact(&headers)?,
        resolve_artifact(&source)?,
    ])
}

fn resolution_lookup_key(milestone: Option<u32>) -> String {
    let selector = milestone
        .map(|value| format!("milestone-{value}"))
        .unwrap_or_else(|| "latest".to_string());
    format!("gke-resolution/{GKE_RESOLUTION_CACHE_VERSION}/{selector}/x86_64")
}

fn validate_resolution(resolution: &GkeResolution, requested_milestone: Option<u32>) -> Result<()> {
    let image = resolution.image;
    ensure!(
        (10..=999).contains(&image.milestone),
        "cached GKE image has invalid COS milestone {}",
        image.milestone
    );
    if let Some(requested) = requested_milestone {
        ensure!(
            image.milestone == requested,
            "cached GKE image {} does not match requested COS milestone {requested}",
            image.image_name()
        );
    }

    for (artifact, expected_name) in resolution
        .artifacts
        .iter()
        .zip(artifact_object_names(image))
    {
        ensure!(
            artifact.name == expected_name,
            "cached GKE artifact order/name mismatch: expected {expected_name:?}, got {:?}",
            artifact.name
        );
        ensure!(
            !artifact.generation.is_empty()
                && artifact
                    .generation
                    .bytes()
                    .all(|byte| byte.is_ascii_digit()),
            "cached GKE artifact {expected_name} has invalid generation {:?}",
            artifact.generation
        );
        ensure!(
            artifact.size > 0,
            "cached GKE artifact {expected_name} is empty"
        );
        let md5 = BASE64_STANDARD
            .decode(&artifact.md5_base64)
            .with_context(|| format!("decode cached GCS md5Hash for {expected_name}"))?;
        ensure!(
            md5.len() == 16,
            "cached GKE artifact {expected_name} md5Hash decoded to {} bytes, expected 16",
            md5.len()
        );
        let expected_url = gcs_download_url(&expected_name, &artifact.generation);
        ensure!(
            artifact.download_url == expected_url,
            "cached GKE artifact {expected_name} download URL is not its exact HTTPS \
             generation-pinned GCS URL"
        );
    }
    Ok(())
}

fn cache_key(image: GkeImage, artifacts: &[GkeArtifact]) -> String {
    // The boot ELF, headers, and source form one indivisible input set.
    // Key all three object identities so a Google-side replacement of
    // headers/source under the same board version cannot reuse modules
    // built from an older generation.
    let mut hasher = Sha256::new();
    for artifact in artifacts {
        hasher.update(artifact.name.as_bytes());
        hasher.update([0]);
        hasher.update(artifact.generation.as_bytes());
        hasher.update([0]);
        hasher.update(artifact.size.to_le_bytes());
        hasher.update([0]);
        hasher.update(artifact.md5_base64.as_bytes());
        hasher.update([0xff]);
    }
    let digest = hex::encode(hasher.finalize());
    format!(
        "distro-gke{}-{}-x86_64-artifacts{}",
        image.milestone,
        image.tools_version(),
        &digest[..12],
    )
}

fn source_path_selected(path: &Path) -> bool {
    SOURCE_FILES.iter().any(|file| path == Path::new(file))
        || SOURCE_DIRS.iter().any(|dir| {
            let dir = Path::new(dir);
            path == dir || path.starts_with(dir)
        })
}

fn find_header_tree(root: &Path) -> Result<(PathBuf, String)> {
    let usr_src = root.join("usr/src");
    let mut candidates = fs::read_dir(&usr_src)
        .with_context(|| format!("read {}", usr_src.display()))?
        .flatten()
        .filter(|entry| entry.file_type().map(|ty| ty.is_dir()).unwrap_or(false))
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_prefix("linux-headers-")
                .map(|release| (entry.path(), release.to_string()))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    ensure!(
        candidates.len() == 1,
        "expected exactly one linux-headers-* tree in {}, found {}",
        usr_src.display(),
        candidates.len()
    );
    Ok(candidates.remove(0))
}

fn config_value<'a>(config: &'a str, option: &str) -> Option<&'a str> {
    config.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key == option).then_some(value)
    })
}

fn validate_gke_config(config: &str, release: &str) -> Result<()> {
    let missing = REQUIRED_BUILTINS
        .iter()
        .copied()
        .filter(|option| config_value(config, option) != Some("y"))
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "official GKE kernel {release} dropped ktstr-required built-in capabilities {missing:?}; \
         refusing a reduced-capability fallback"
    );
    ensure!(
        config_value(config, "CONFIG_MODVERSIONS").is_none(),
        "official GKE kernel {release} enables CONFIG_MODVERSIONS; exact external-module symbol \
         version support is not implemented"
    );
    ensure!(
        config_value(config, "CONFIG_MODULE_SIG_FORCE").is_none(),
        "official GKE kernel {release} enforces signed modules; ktstr cannot load its exact-source \
         transport modules without Google's private signing key"
    );
    Ok(())
}

fn command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn module_make_args(source_dir: &Path, jobs: usize, lld_available: bool) -> Vec<String> {
    let mut args = vec![format!("M={}", source_dir.display()), "LLVM=1".to_string()];
    // LLVM=1 selects Clang plus the LLVM binutils. Some distributions
    // package Clang without LLD; Linux Kbuild explicitly supports mixing
    // Clang with GNU ld via LD=ld. External modules retain the exact same
    // Google headers, source, config, compiler mode, and module ABI.
    if !lld_available {
        args.push("LD=ld".to_string());
    }
    args.extend([format!("-j{jobs}"), "modules".to_string()]);
    args
}

fn build_modules(
    headers: &Path,
    source_dir: &Path,
    config: &str,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<Vec<PathBuf>> {
    fs::write(source_dir.join("Makefile"), MODULE_MAKEFILE)
        .with_context(|| format!("write {}", source_dir.join("Makefile").display()))?;
    let jobs = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let lld_available = command_available("ld.lld");
    if !lld_available {
        let status = format!("{cli_label}: ld.lld not found; linking GKE modules with GNU ld");
        match mp {
            Some(progress) => progress.println(&status),
            None => eprintln!("{status}"),
        }
    }
    let args = module_make_args(source_dir, jobs, lld_available);
    let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let bar = mp.map(|progress| progress.step_bar(&format!("{cli_label}: building GKE modules")));
    let result = crate::cli::run_make_with_output(headers, &arg_refs, mp)
        .with_context(|| "build external modules against official GKE headers");
    if let Some(bar) = &bar {
        bar.finish();
    }
    result?;

    let mut modules = Vec::new();
    for (config_option, relative) in MODULE_OUTPUTS {
        let module = source_dir.join(relative);
        ensure!(
            module.is_file(),
            "GKE external-module build did not produce {}",
            module.display()
        );
        // Do not load a duplicate if a later GKE kernel builds the
        // driver in. We still build the complete known set in one
        // invocation so source/header drift is detected up front.
        if config_value(config, config_option) != Some("y") {
            modules.push(module);
        }
    }
    Ok(modules)
}

fn artifact_provenance(artifact: &GkeArtifact) -> String {
    format!(
        "gs://{COS_TOOLS_BUCKET}/{}#generation={}#size={}",
        artifact.name, artifact.generation, artifact.size
    )
}

/// Acquire the latest GKE-promoted COS kernel, optionally pinned to a
/// COS milestone (`gke-129`), and return its local cache entry.
pub(crate) fn acquire_gke_kernel(
    milestone: Option<&str>,
    cli_label: &str,
    mp: Option<&crate::cli::FetchProgress>,
) -> Result<PathBuf> {
    let (arch, _) = crate::fetch::arch_info();
    if arch != "x86_64" {
        bail!(
            "--kernel gke currently supports x86_64 only: the official artifact mapping uses \
             Google's gke-amd64-gcp COS board, not a substituted generic kernel"
        );
    }

    let requested_milestone = parse_milestone_selector(milestone)?;
    let resolution_key = resolution_lookup_key(requested_milestone);
    let resolution = crate::lookup_cache::last_known_good(
        &resolution_key,
        || {
            let image = resolve_promoted_image(requested_milestone)?;
            let artifacts = resolve_artifacts(image)?;
            Ok(GkeResolution { image, artifacts })
        },
        |resolution| validate_resolution(resolution, requested_milestone),
    )
    .with_context(|| {
        let selector = requested_milestone
            .map(|value| format!("COS milestone {value}"))
            .unwrap_or_else(|| "latest promoted COS image".to_string());
        format!("resolve official GKE metadata for {selector}")
    })?;
    let GkeResolution { image, artifacts } = resolution;
    let source = &artifacts[2];
    let key = cache_key(image, &artifacts);
    let cache = crate::cache::CacheDir::new()?;
    if let Some(entry) = cache.lookup(&key) {
        tracing::info!(cache_key = %key, image = %image.image_name(), "GKE kernel cache hit");
        return Ok(entry.path);
    }

    let status = format!(
        "{cli_label}: resolving official GKE image {} (Google kernel + exact-source modules)",
        image.image_name()
    );
    match mp {
        Some(progress) => progress.println(&status),
        None => eprintln!("{status}"),
    }

    let scratch = scratch_dir()?;
    let downloads = scratch.path().join("artifacts");
    fs::create_dir_all(&downloads)?;
    let destinations = [
        downloads.join("vmlinux"),
        downloads.join("kernel-headers.tgz"),
        downloads.join("kernel-src.tar.gz"),
    ];
    for (artifact, destination) in artifacts.iter().zip(destinations.iter()) {
        let label = artifact.name.rsplit('/').next().unwrap_or(&artifact.name);
        crate::fetch::download_verified_md5_file(
            &artifact.download_url,
            destination,
            &artifact.md5_base64,
            label,
            cli_label,
            mp,
        )
        .with_context(|| format!("download official GKE artifact {}", artifact.name))?;
    }

    let header_root = scratch.path().join("headers");
    fs::create_dir_all(&header_root)?;
    let header_file =
        fs::File::open(&destinations[1]).context("open downloaded GKE kernel headers")?;
    extract_tar(flate2::read::GzDecoder::new(header_file), &header_root)
        .context("extract official GKE kernel headers")?;
    let (header_tree, kernel_release) = find_header_tree(&header_root)?;
    let config_path = header_tree.join(".config");
    let config = fs::read_to_string(&config_path)
        .with_context(|| format!("read GKE kernel config {}", config_path.display()))?;
    validate_gke_config(&config, &kernel_release)?;

    let module_source = scratch.path().join("module-source");
    fs::create_dir_all(&module_source)?;
    let source_file =
        fs::File::open(&destinations[2]).context("open downloaded GKE kernel source")?;
    extract_tar_matching(
        flate2::read::GzDecoder::new(source_file),
        &module_source,
        source_path_selected,
    )
    .context("extract GKE external-module source subset")?;
    for required in SOURCE_FILES {
        ensure!(
            module_source.join(required).is_file(),
            "official GKE source archive omitted required module source {required}"
        );
    }
    for required in SOURCE_DIRS {
        ensure!(
            module_source.join(required).is_dir(),
            "official GKE source archive omitted required module directory {required}"
        );
    }

    let modules = build_modules(&header_tree, &module_source, &config, cli_label, mp)?;
    let extracted = ExtractedKernel {
        kernel_release: kernel_release.clone(),
        image: destinations[0].clone(),
        config: Some(config_path),
        system_map: None,
        modules_dir: None,
        // Google's published vmlinux is both the bootable image and
        // the exact debuginfo-bearing ELF.
        vmlinux: Some(destinations[0].clone()),
    };
    let mut provenance = vec![image.image_name()];
    provenance.extend(artifacts.iter().map(artifact_provenance));
    provenance.push(format!(
        "external-modules-built-from={}",
        artifact_provenance(source)
    ));
    install_extracted(
        &key,
        &extracted,
        &modules,
        KernelSource::DistroPackage {
            distro: format!("gke{}", image.milestone),
            packages: provenance,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolution_fixture() -> GkeResolution {
        let image = GkeImage {
            milestone: 129,
            build: 19506,
            patch: 224,
            revision: 80,
        };
        let prefix = image.board_prefix();
        let artifact = |basename: &str, generation: &str, size| {
            let name = format!("{prefix}/{basename}");
            GkeArtifact {
                download_url: gcs_download_url(&name, generation),
                name,
                generation: generation.to_string(),
                size,
                md5_base64: "EFRgfQGcG6lmUe4w6b9y1A==".to_string(),
            }
        };
        GkeResolution {
            image,
            artifacts: [
                artifact("vmlinux", "1782777210762594", 378_632_832),
                artifact("kernel-headers.tgz", "1782777210762595", 24_000_000),
                artifact("kernel-src.tar.gz", "1782777210762596", 220_000_000),
            ],
        }
    }

    #[test]
    fn release_notes_choose_latest_promoted_image() {
        let notes = r#"
            <li>COS_CONTAINERD: cos-128-19000-1-9</li>
            <li>COS_CONTAINERD: cos-beta-999-99999-9-9</li>
            <li>COS_CONTAINERD: cos-129-19506-224-80</li>
            <li>not promoted: cos-130-20000-1-1-beta</li>
            <li>COS_CONTAINERD: cos-129-19506-224-79</li>
        "#;
        assert_eq!(
            parse_release_notes(notes, None).unwrap(),
            GkeImage {
                milestone: 129,
                build: 19506,
                patch: 224,
                revision: 80,
            }
        );
    }

    #[test]
    fn release_notes_milestone_pin_stays_within_milestone() {
        let notes = " cos-128-19000-2-1  cos-129-19506-224-80  cos-128-19001-1-3 ";
        assert_eq!(
            parse_release_notes(notes, Some(128)).unwrap(),
            GkeImage {
                milestone: 128,
                build: 19001,
                patch: 1,
                revision: 3,
            }
        );
        assert!(parse_release_notes(notes, Some(127)).is_err());
    }

    #[test]
    fn gcs_metadata_is_exact_generation_pinned_and_md5_checked() {
        let name = "19506.224.80/gke-amd64-gcp/vmlinux";
        let json = format!(
            r#"{{"name":"{name}","generation":"1782777210762594","size":"378632832","md5Hash":"EFRgfQGcG6lmUe4w6b9y1A=="}}"#
        );
        let artifact = parse_gcs_metadata(json.as_bytes(), name).unwrap();
        assert_eq!(artifact.size, 378_632_832);
        assert!(
            artifact
                .download_url
                .contains("generation=1782777210762594")
        );
        assert!(
            artifact
                .download_url
                .contains("19506.224.80%2Fgke-amd64-gcp%2Fvmlinux")
        );

        let bad = json.replace("EFRgfQGcG6lmUe4w6b9y1A==", "AA==");
        assert!(parse_gcs_metadata(bad.as_bytes(), name).is_err());
        assert!(parse_gcs_metadata(json.as_bytes(), "other/object").is_err());
    }

    #[test]
    fn resolution_lookup_key_separates_channel_and_architecture() {
        assert_eq!(
            resolution_lookup_key(None),
            "gke-resolution/v1/latest/x86_64"
        );
        assert_eq!(
            resolution_lookup_key(Some(129)),
            "gke-resolution/v1/milestone-129/x86_64"
        );
        assert_ne!(
            resolution_lookup_key(None),
            resolution_lookup_key(Some(129))
        );
        assert_ne!(
            resolution_lookup_key(Some(128)),
            resolution_lookup_key(Some(129))
        );
        assert_eq!(
            resolution_lookup_key(parse_milestone_selector(Some("0129")).unwrap()),
            resolution_lookup_key(Some(129)),
            "equivalent numeric selectors must share one stable lookup key"
        );
        assert!(parse_milestone_selector(Some("latest")).is_err());
    }

    #[test]
    fn resolution_validator_pins_the_complete_generation_plan() {
        let baseline = resolution_fixture();
        validate_resolution(&baseline, None).unwrap();
        validate_resolution(&baseline, Some(129)).unwrap();
        assert!(validate_resolution(&baseline, Some(128)).is_err());

        let mut invalid_milestone = baseline.clone();
        invalid_milestone.image.milestone = 9;
        assert!(validate_resolution(&invalid_milestone, None).is_err());

        let image_mutations: [fn(&mut GkeImage); 3] = [
            |image: &mut GkeImage| image.build += 1,
            |image: &mut GkeImage| image.patch += 1,
            |image: &mut GkeImage| image.revision += 1,
        ];
        for mutate in image_mutations {
            let mut changed = baseline.clone();
            mutate(&mut changed.image);
            assert!(
                validate_resolution(&changed, None).is_err(),
                "every image version field must agree with the artifact plan"
            );
        }

        let mut wrong_name = baseline.clone();
        wrong_name.artifacts[0].name.push_str(".old");
        assert!(validate_resolution(&wrong_name, None).is_err());

        let mut wrong_order = baseline.clone();
        wrong_order.artifacts.swap(0, 1);
        assert!(validate_resolution(&wrong_order, None).is_err());

        for generation in ["", "not-numeric"] {
            let mut bad_generation = baseline.clone();
            bad_generation.artifacts[0].generation = generation.to_string();
            assert!(validate_resolution(&bad_generation, None).is_err());
        }

        let mut empty = baseline.clone();
        empty.artifacts[0].size = 0;
        assert!(validate_resolution(&empty, None).is_err());

        for md5 in ["not-base64", "AA=="] {
            let mut bad_md5 = baseline.clone();
            bad_md5.artifacts[0].md5_base64 = md5.to_string();
            assert!(validate_resolution(&bad_md5, None).is_err());
        }

        let mut non_https = baseline.clone();
        non_https.artifacts[0].download_url = non_https.artifacts[0]
            .download_url
            .replacen("https:", "http:", 1);
        assert!(validate_resolution(&non_https, None).is_err());

        let mut mismatched_generation = baseline;
        mismatched_generation.artifacts[0].generation = "1782777210762597".to_string();
        assert!(validate_resolution(&mismatched_generation, None).is_err());
    }

    #[test]
    fn cache_key_covers_every_artifact_generation() {
        let image = GkeImage {
            milestone: 129,
            build: 19506,
            patch: 224,
            revision: 80,
        };
        let artifact = |name: &str, generation: &str| GkeArtifact {
            name: name.to_string(),
            generation: generation.to_string(),
            size: 123,
            md5_base64: "EFRgfQGcG6lmUe4w6b9y1A==".to_string(),
            download_url: String::new(),
        };
        let artifacts = vec![
            artifact("board/vmlinux", "1"),
            artifact("board/kernel-headers.tgz", "2"),
            artifact("board/kernel-src.tar.gz", "3"),
        ];
        let baseline = cache_key(image, &artifacts);
        for index in 0..artifacts.len() {
            let mut changed = artifacts.clone();
            changed[index].generation.push('9');
            assert_ne!(
                cache_key(image, &changed),
                baseline,
                "artifact {index} generation must participate in the cache key"
            );
        }
    }

    #[test]
    fn source_selection_is_exact_and_does_not_match_prefix_siblings() {
        assert!(source_path_selected(Path::new(
            "drivers/virtio/virtio_mmio.c"
        )));
        assert!(source_path_selected(Path::new("fs/btrfs/inode.c")));
        assert!(source_path_selected(Path::new("lib/raid6/algos.c")));
        assert!(!source_path_selected(Path::new("fs/btrfsx/inode.c")));
        assert!(!source_path_selected(Path::new(
            "drivers/virtio/virtio_balloon.c"
        )));
    }

    #[test]
    fn config_gate_refuses_capability_or_module_abi_compromise() {
        let mut config = REQUIRED_BUILTINS
            .iter()
            .map(|option| format!("{option}=y\n"))
            .collect::<String>();
        config.push_str("# CONFIG_MODVERSIONS is not set\n");
        config.push_str("# CONFIG_MODULE_SIG_FORCE is not set\n");
        validate_gke_config(&config, "test").unwrap();

        assert!(validate_gke_config(&config.replace("CONFIG_VIRTIO_NET=y\n", ""), "test").is_err());
        assert!(
            validate_gke_config(&format!("{config}CONFIG_MODULE_SIG_FORCE=y\n"), "test").is_err()
        );
    }

    #[test]
    fn module_build_uses_gnu_ld_when_lld_is_unavailable() {
        let with_lld = module_make_args(Path::new("/tmp/modules"), 8, true);
        assert!(!with_lld.iter().any(|arg| arg.starts_with("LD=")));

        let without_lld = module_make_args(Path::new("/tmp/modules"), 8, false);
        assert!(without_lld.iter().any(|arg| arg == "LD=ld"));
        assert!(without_lld.iter().any(|arg| arg == "LLVM=1"));
    }

    /// Cheap live metadata gate used by `just test-distro-resolve`.
    /// It proves GKE release-note parsing and all three official GCS
    /// object metadata lookups without downloading the artifact bodies.
    #[test]
    #[ignore = "requires network access to official Google metadata"]
    fn live_gke_official_metadata_resolves() {
        let image = resolve_promoted_image(None).expect("resolve latest GKE-promoted COS image");
        let artifacts = resolve_artifacts(image).expect("resolve official cos-tools artifacts");
        assert_eq!(artifacts.len(), 3);
        assert!(artifacts.iter().all(|artifact| artifact.size > 0));
        assert!(
            artifacts
                .iter()
                .all(|artifact| artifact.download_url.contains("generation="))
        );
    }
}
