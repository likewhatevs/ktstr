//! Smaller `cargo ktstr` subcommand dispatchers.
//!
//! One submodule per subcommand whose implementation is too small
//! to warrant its own top-level module file, plus [`probe`], shared
//! probe-loop scaffolding consumed by [`export`] and `shell` rather
//! than a subcommand of its own. Each subcommand submodule contains
//! exactly the dispatcher(s) for one logical subcommand; the
//! unit-of-edit is "the work behind one CLI verb."
//!
//! - [`shell`]       — `cargo ktstr shell` (KVM VM + busybox).
//! - [`completions`] — `cargo ktstr completions` (clap_complete dump).
//! - [`export`]      — `cargo ktstr export` (`.run` self-extracting
//!   reproducer for a registered test).
//! - [`probe`]       — shared probe-loop scaffolding
//!   (`probe_first`/`probe_collect`) used by `export` and `shell` to
//!   locate a `#[ktstr_test]` registration; not a subcommand.
//!
//! The `--kernel` resolution shim used by `shell` (and re-used by
//! the verifier subcommand) lives in [`super::kernel`].

mod completions;
mod export;
mod probe;
mod shell;

pub(crate) use completions::run_completions;
pub(crate) use export::run_export;
pub(crate) use probe::{ProbeError, probe_collect, probe_collect_from_bins};
pub(crate) use shell::run_shell;
