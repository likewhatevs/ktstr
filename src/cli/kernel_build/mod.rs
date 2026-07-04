//! Kernel build pipeline: configure, validate, build, cache-store.
//!
//! Split into three submodules:
//! - [`make`] — `make` subprocess wrappers ([`run_make`],
//!   [`run_make_with_output`], [`make_kernel_with_output`]) plus
//!   the byte-oriented line drain and timeout-poll loop they share.
//! - [`kconfig`] — fragment merging ([`kconfig::configure_kernel`]),
//!   `--extra-kconfig` parsing ([`read_extra_kconfig`],
//!   [`append_extra_kconfig_suffix`]), pre/post warning passes
//!   over user fragments, and the post-build critical-options
//!   check ([`validate_kernel_config`]).
//! - `build` — top-level orchestrator ([`kernel_build_pipeline`]),
//!   its two-phase reservation acquisition
//!   (`acquire_build_reservation`: LLC flock + cgroup sandbox), and
//!   the source-tree flock helper (`acquire_source_tree_lock`) that
//!   serializes parallel builds against the same on-disk source tree.

mod build;
mod kconfig;
mod make;

pub use build::{KernelBuildResult, kernel_build_pipeline};
pub use kconfig::{append_extra_kconfig_suffix, read_extra_kconfig, validate_kernel_config};
pub use make::{make_kernel_with_output, run_make, run_make_with_output};
