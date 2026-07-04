//! Unit tests for HZ resolution: `parse_config_hz`,
//! `vcpu_preemption_threshold_ns`, `guest_kernel_hz`, and the
//! IKCONFIG extractor (`read_hz_from_ikconfig`).
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::*;

#[test]
fn parse_config_hz_standard() {
    let config = "# comment\nCONFIG_HZ_1000=y\nCONFIG_HZ=1000\n";
    assert_eq!(parse_config_hz(config), Some(1000));
}

#[test]
fn parse_config_hz_250() {
    let config = "CONFIG_HZ=250\n";
    assert_eq!(parse_config_hz(config), Some(250));
}

#[test]
fn parse_config_hz_100() {
    let config = "CONFIG_HZ=100\n";
    assert_eq!(parse_config_hz(config), Some(100));
}

#[test]
fn parse_config_hz_missing() {
    let config = "CONFIG_PREEMPT=y\nCONFIG_HZ_1000=y\n";
    assert_eq!(parse_config_hz(config), None);
}

#[test]
fn parse_config_hz_garbage_value() {
    let config = "CONFIG_HZ=abc\n";
    assert_eq!(parse_config_hz(config), None);
}

#[test]
fn parse_config_hz_whitespace() {
    let config = "  CONFIG_HZ=1000  \n";
    assert_eq!(parse_config_hz(config), Some(1000));
}

#[test]
fn parse_config_hz_commented_out() {
    let config = "# CONFIG_HZ=1000\nCONFIG_HZ_1000=y\n";
    assert_eq!(parse_config_hz(config), None);
}

#[test]
fn vcpu_threshold_reasonable_range() {
    // With no kernel path, falls back to host config or DEFAULT_HZ=250.
    // Threshold should be between 10ms (HZ=1000) and 100ms (HZ=100).
    let t = vcpu_preemption_threshold_ns(None);
    assert!(
        (10_000_000..=100_000_000).contains(&t),
        "threshold {t} ns outside expected range 10ms-100ms"
    );
}

#[test]
fn vcpu_threshold_default_hz_fallback() {
    // Nonexistent kernel path with Some(...): IKCONFIG and .config
    // lookups both fail, so guest_kernel_hz returns DEFAULT_HZ without
    // consulting host /boot/config.
    let t = vcpu_preemption_threshold_ns(Some(std::path::Path::new("/nonexistent/bzImage")));
    assert!(
        (10_000_000..=100_000_000).contains(&t),
        "fallback threshold {t} ns outside expected range"
    );
}

/// Regression for the "host config leaks into guest HZ" bug:
/// when `kernel_path` is `Some`, `guest_kernel_hz` must not fall
/// back to `/boot/config-$(uname -r)`. A cached/built guest
/// kernel's HZ is independent of the host's HZ, so silently
/// picking up host HZ would yield wrong tick-dependent thresholds
/// on any mismatch.
///
/// This test points `kernel_path` at a nonexistent file. The
/// IKCONFIG and `.config` lookups both fail, and the function
/// must return exactly [`DEFAULT_HZ`] — NOT whatever the host's
/// `/boot/config` happens to contain.
#[test]
fn guest_kernel_hz_gated_on_kernel_path() {
    let bogus = std::path::Path::new("/nonexistent/ktstr-kernel/bzImage");
    let hz = guest_kernel_hz(Some(bogus));
    assert_eq!(
        hz, DEFAULT_HZ,
        "kernel_path=Some with no IKCONFIG/.config must fall back \
         to DEFAULT_HZ, not host /boot/config; got {hz}"
    );
}

/// Complement: with `kernel_path=None` (virtme-style run), the
/// host config IS authoritative and may legitimately override
/// `DEFAULT_HZ`. Check the returned value is a plausible HZ
/// value — i.e., the code path still works when we explicitly
/// want host fallback.
#[test]
fn guest_kernel_hz_none_consults_host_config() {
    let hz = guest_kernel_hz(None);
    // Accept any known Linux HZ value (DEFAULT_HZ=250 is in this set).
    assert!(
        matches!(hz, 100 | 250 | 300 | 1000),
        "guest_kernel_hz(None) = {hz} outside plausible HZ set"
    );
}

// -- IKCONFIG extraction tests --

/// Build a synthetic blob: padding + IKCFG_ST marker + gzip(config_text) + IKCFG_ED marker.
fn make_ikconfig_blob(config_text: &str) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    let mut blob = vec![0u8; 64]; // padding
    blob.extend_from_slice(IKCONFIG_MAGIC);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(config_text.as_bytes()).unwrap();
    blob.extend(encoder.finish().unwrap());
    blob.extend_from_slice(b"IKCFG_ED");
    blob
}

#[test]
fn ikconfig_extracts_hz_1000() {
    let blob = make_ikconfig_blob("CONFIG_HZ=1000\nCONFIG_PREEMPT=y\n");
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-1000");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, &blob).unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), Some(1000));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ikconfig_extracts_hz_250() {
    let blob = make_ikconfig_blob("CONFIG_HZ=250\n");
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-250");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, &blob).unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), Some(250));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ikconfig_no_marker_returns_none() {
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-none");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, b"no marker here").unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ikconfig_missing_config_hz_returns_none() {
    let blob = make_ikconfig_blob("CONFIG_PREEMPT=y\n");
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-nohz");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, &blob).unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), None);
    let _ = std::fs::remove_dir_all(&dir);
}
