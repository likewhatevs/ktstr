//! Prebuilt distro-kernel support for the `--kernel` flag.
//!
//! [`repo`] resolves package-backed distro specs to concrete URLs and
//! checksums from official repository metadata. [`gke`] resolves the
//! official GKE-promoted COS boot ELF, headers, and matching source.
//! [`extract`] turns package/archive contents into the artifacts ktstr
//! needs to boot a VM; [`acquire`] shares capability gates and atomic
//! cache installation across the artifact shapes.

pub mod acquire;
pub mod extract;
pub mod gke;
pub mod repo;
