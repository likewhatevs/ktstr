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
//! - [`funify`]      — `cargo ktstr funify` (JSON value funification).
//! - [`model`]       — `cargo ktstr model {fetch,status,clean}` (LLM
//!   model cache management).
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
mod funify;
#[cfg(feature = "llm")]
mod model;
mod probe;
mod shell;

pub(crate) use completions::run_completions;
pub(crate) use export::run_export;
pub(crate) use funify::run_funify;
#[cfg(feature = "llm")]
pub(crate) use model::{run_model_clean, run_model_fetch, run_model_status};
pub(crate) use shell::run_shell;
