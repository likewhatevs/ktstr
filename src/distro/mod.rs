//! Prebuilt distro-kernel support for the `--kernel` flag.
//!
//! [`repo`] resolves a distro spec (`fedora` / `ubuntu` / `amazonlinux`)
//! to concrete package URLs + sha256 checksums pulled from the distro's
//! official repo metadata. [`extract`] turns downloaded or local
//! `.rpm` / `.deb` kernel packages into the artifacts ktstr needs to
//! boot a VM (a raw bootable image plus the matching module tree,
//! config, `System.map`, and optional `vmlinux`).

pub mod acquire;
pub mod extract;
pub mod repo;
