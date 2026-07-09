//! Distro prebuilt-kernel support: resolve a distro spec
//! (`fedora` / `ubuntu` / `amazonlinux`) to concrete package URLs +
//! sha256 checksums pulled from the distro's official repo metadata.

pub mod repo;
