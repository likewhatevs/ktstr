//! wprof tracer configuration shipped into auto-repro VMs.
//!
//! [wprof](https://github.com/anakryiko/wprof) is a BSD-3-Clause
//! BPF-based system-wide tracer/profiler that emits Perfetto-format
//! `.pb` traces. ktstr ships the binary at `/bin/wprof` in the guest
//! initramfs when a test enables wprof capture, and the guest init
//! invokes it during auto-repro to produce a trace alongside the
//! standard failure-dump output.
//!
//! The wprof bytes live in the `cargo-ktstr` binary (built by
//! `build.rs`, embedded via `include_bytes!` in
//! `src/bin/cargo_ktstr/blobs.rs`). At cargo-ktstr startup the
//! `install_env` hook extracts the bytes to a tempfile and exports
//! `KTSTR_WPROF_PATH=<path>`; [`WprofConfig::from_env`] reads that
//! env var and loads the file for the builder.

use anyhow::Result;

/// Minimum guest memory (MiB) for test entries that enable wprof.
///
/// Derivation: `WprofConfig::default_args` requests a single 16 MiB
/// BPF ring buffer (`--ringbuf-size=16384 --ringbuf-cnt=1`). wprof
/// faults the whole arena in on capture, so guest RAM must cover it.
/// The floor is computed from the default ringbuf constants at
/// `crate::test_support::runtime::WPROF_MIN_MEMORY_MIB` (16 MiB), so
/// editing `--ringbuf-size`/`--ringbuf-cnt` moves the floor with them.
///
/// Applied as a floor by
/// `crate::test_support::runtime::derive_test_memory_mib` —
/// the single derivation site shared by the entry-derived
/// topology path AND every dispatch site that constructs a
/// `crate::test_support::topo::TopoOverride` from CLI / preset
/// topology (so e.g. `cargo ktstr test --ktstr-topo NnNlNcNt` and
/// gauntlet preset runs honor the floor identically). Both
/// primary and auto-repro VMs route through the same derivation
/// so the floor propagates to both; identical topology between
/// the two VMs is also required for stall reproducibility.
///
/// The floor applies when the derived memory
/// `max(cpus*64, 256, entry.memory_mib)` falls below it AND
/// `entry.wprof` is true. At the minimal default the 16 MiB arena
/// sits below the universal 256 MiB memory floor, so this floor is
/// subsumed and never bumps a real VM — the point is that the
/// default no longer oversizes; it re-engages only if the default
/// ringbuf sizing grows the arena past the universal floor. A test
/// that overrides to a larger arena via
/// `#[ktstr_test(wprof_args = "...")]` must raise its own
/// `memory_mib` if that arena exceeds the guest's normal memory. An
/// operator-supplied `crate::test_support::topo::TopoOverride` with
/// explicit `memory_mib` is honored verbatim per the
/// override-is-verbatim contract. Shell-mode VMs
/// (`cargo ktstr shell --kernel ...`) bypass this floor entirely;
/// the operator sets memory size via shell-mode CLI args.
pub const WPROF_MIN_MEMORY_MIB: u32 = crate::test_support::runtime::WPROF_MIN_MEMORY_MIB;

/// Apply the wprof memory floor to a raw memory size.
///
/// Returns `WPROF_MIN_MEMORY_MIB` when `wprof` is true and `raw_mib`
/// falls below the floor; otherwise returns `raw_mib` unchanged.
/// Pure function — no logging, no side effects — so it can be
/// called from multiple resolution paths without duplicating the
/// floor formula.
///
/// Single source of truth shared by
/// `crate::test_support::runtime::derive_test_memory_mib` (the
/// test-launch path used by `cargo ktstr test`) AND by
/// `cargo ktstr shell --test <NAME>`'s router (which constructs a
/// shell VM matching the named test's topology). A future change
/// to the floor formula updates here once; both call sites pick
/// it up. Inline copies of the `raw < WPROF_MIN_MEMORY_MIB`
/// conditional are a regression per the
/// derive_test_memory_mib/attach_wprof_if_requested precedent.
pub fn apply_wprof_memory_floor(raw_mib: u32, wprof: bool) -> u32 {
    crate::test_support::runtime::apply_wprof_memory_floor(raw_mib, wprof)
}

/// wprof invocation args + binary path. Passed to
/// [`crate::vmm::KtstrVmBuilder::wprof`] when a test (or
/// `cargo ktstr shell`) wants the wprof tracer available inside
/// the guest VM.
///
/// The binary is sourced from a host path (typically the tempfile
/// path that `cargo-ktstr` exports via `KTSTR_WPROF_PATH`). The
/// library packs it at `bin/wprof` in the initramfs using the
/// existing include_files mechanism, which performs `DT_NEEDED`
/// resolution and pulls every shared library wprof links against
/// (libelf, libz, etc.) so the binary actually runs inside the
/// guest.
///
/// The args render to the kernel cmdline as
/// `KTSTR_WPROF_ARGS=<args joined with ASCII Unit Separator `\x1F`>`
/// (see [`WprofConfig::args_cmdline`]); guest init parses them and
/// invokes `/bin/wprof` with those args during auto-repro.
#[derive(Debug, Clone)]
pub struct WprofConfig {
    /// Host filesystem path to the wprof binary. The library opens
    /// this path to read the ELF (for `DT_NEEDED` resolution) and
    /// to copy the bytes into the initramfs. Usually populated from
    /// [`WprofConfig::from_env`] which reads `KTSTR_WPROF_PATH`.
    pub host_path: std::path::PathBuf,
    /// CLI args appended after `/bin/wprof` when guest init spawns
    /// the tracer. Defaults to [`WprofConfig::default_args`].
    pub args: Vec<String>,
}

impl WprofConfig {
    /// Construct a [`WprofConfig`] using the env-var-published
    /// wprof binary (set by `cargo-ktstr` at startup) and the
    /// project default args. Returns an error if `KTSTR_WPROF_PATH`
    /// is unset.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            host_path: crate::vmm::blobs::load_wprof_path()?,
            args: Self::default_args(),
        })
    }

    /// Default wprof args used by the auto-repro pipeline:
    /// `-d 500 -e sched --ringbuf-size=16384 --ringbuf-cnt=1`.
    ///
    /// Capture window: 500 ms of sched-event tracing into a single
    /// 16 MiB BPF ring buffer — wprof's own per-buffer default
    /// (`DEFAULT_RINGBUF_SZ`), the minimal arena that still captures
    /// ktstr's short sched traces without dropping. The ringbuf
    /// sizing comes from
    /// [`crate::test_support::runtime::WPROF_DEFAULT_RINGBUF_SIZE_KB`]
    /// and [`crate::test_support::runtime::WPROF_DEFAULT_RINGBUF_CNT`]
    /// so the flags and the derived [`WPROF_MIN_MEMORY_MIB`] floor
    /// stay in lockstep. Override per-test by constructing a
    /// [`WprofConfig`] with a custom `args` vec (or, from a test,
    /// `#[ktstr_test(wprof_args = "...")]`).
    pub fn default_args() -> Vec<String> {
        use crate::test_support::runtime::{
            WPROF_DEFAULT_RINGBUF_CNT, WPROF_DEFAULT_RINGBUF_SIZE_KB,
        };
        vec![
            "-d".to_string(),
            "500".to_string(),
            "-e".to_string(),
            "sched".to_string(),
            format!("--ringbuf-size={WPROF_DEFAULT_RINGBUF_SIZE_KB}"),
            format!("--ringbuf-cnt={WPROF_DEFAULT_RINGBUF_CNT}"),
        ]
    }

    /// Render the args as a delimited string suitable for passing on
    /// the kernel cmdline (`KTSTR_WPROF_ARGS=...`). The kernel
    /// cmdline tokenizer splits on whitespace and `/proc/cmdline`
    /// parsers (including the guest's `cmdline_val` via
    /// `split_whitespace`) cannot recover spaces inside a value, so
    /// joining args with a literal space would truncate everything
    /// after the first arg. Use ASCII Unit Separator (`\x1F`, the
    /// delimiter character codepoint reserved by ANSI X3.4 for this
    /// exact purpose) — non-whitespace, won't appear in real wprof
    /// args (which are flags, paths, integers, comma-separated
    /// feature lists), and the kernel passes it through to
    /// `/proc/cmdline` byte-for-byte.
    pub fn args_cmdline(&self) -> String {
        self.args.join("\x1f")
    }
}
