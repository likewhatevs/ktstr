// ctor 1.0's `#[ctor::ctor(...)]` macro expansion is a deep
// TT-muncher whose recursion depth on this crate's ctor sites
// exceeds Rust's default 128-frame macro-expansion budget. 256 is
// what the rustc lint's own help message recommends; ctor itself
// declares the same bump at the top of its lib.rs.
#![recursion_limit = "256"]

//! VM-based test framework for Linux kernel subsystems, with a focus on sched_ext.
//!
//! ktstr boots lightweight KVM virtual machines with controlled CPU topologies,
//! runs scheduler test scenarios inside them, and evaluates results from the
//! host via guest memory introspection. Each test creates cgroups, spawns
//! worker processes, and checks that the scheduler handled the workload
//! correctly. Also tests under the kernel's default EEVDF scheduler.
//!
//! # Quick start
//!
//! Declare cgroups and workloads as data, let the framework handle
//! lifecycle and checking:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
//! fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
//!     execute_defs(ctx, vec![
//!         CgroupDef::named("cg_0").workers(2),
//!         CgroupDef::named("cg_1").workers(2),
//!     ])
//! }
//! ```
//!
//! Requires a kernel image; see [`find_kernel()`] for the resolution chain.
//!
//! For multi-phase scenarios with dynamic topology changes:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
//! fn my_dynamic_test(ctx: &Ctx) -> Result<AssertResult> {
//!     let steps = vec![
//!         Step::with_defs(
//!             vec![CgroupDef::named("cg_0").workers(4)],
//!             HoldSpec::frac(0.5),
//!         ),
//!         Step::new(
//!             vec![Op::stop_cgroup("cg_0"), Op::remove_cgroup("cg_0")],
//!             HoldSpec::frac(0.5),
//!         ),
//!     ];
//!     execute_steps(ctx, steps)
//! }
//! ```
//!
//! # Scheduler definition
//!
//! Tests work with just topology parameters (as above). When multiple
//! tests share a scheduler, use `declare_scheduler!` to declare it
//! once with a binary, default topology, and any always-on args. Tests
//! reference the generated const and inherit its configuration:
//!
//! ```rust,no_run
//! use ktstr::prelude::*;
//!
//! declare_scheduler!(MY_SCHED, {
//!     name = "my_sched",
//!     binary = "scx_my_sched",
//! });
//!
//! #[ktstr_test(scheduler = MY_SCHED)]
//! fn basic(ctx: &Ctx) -> Result<AssertResult> {
//!     execute_defs(ctx, vec![
//!         CgroupDef::named("cg_0").workers(2),
//!         CgroupDef::named("cg_1").workers(2),
//!     ])
//! }
//! ```
//!
//! For full control over cgroup setup, worker spawning, and assertion
//! you can use the low-level API directly:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
//! fn my_low_level_test(ctx: &Ctx) -> Result<AssertResult> {
//!     let mut group = CgroupGroup::new(ctx.cgroups);
//!     group.add_cgroup_no_cpuset("workers")?;
//!     let cpus = ctx.topo.all_cpuset();
//!     ctx.cgroups.set_cpuset("workers", &cpus)?;
//!
//!     let cfg = WorkloadConfig {
//!         num_workers: 2,
//!         work_type: WorkType::SpinWait,
//!         ..Default::default()
//!     };
//!     let mut handle = WorkloadHandle::spawn(&cfg)?;
//!     ctx.cgroups.move_tasks("workers", &handle.worker_pids_for_cgroup_procs()?)?;
//!     handle.start();
//!
//!     std::thread::sleep(ctx.duration);
//!     let reports = handle.stop_and_collect();
//!
//!     let a = Assert::default_checks();
//!     Ok(a.assert_cgroup(&reports, None))
//! }
//! ```
//!
//! For pointwise assertions against captured stats — the most direct
//! way to express "this counter is at least N", "this rate is between
//! A and B", "this metric is finite" — use `Verdict` +
//! `#[derive(Claim)]` accessors and the [`claim!`] macro:
//!
//! ```rust
//! use ktstr::prelude::*;
//! use ktstr::workload::WorkerReport;
//! use std::collections::{BTreeMap, BTreeSet};
//!
//! // A test author would obtain `cg` and `report` from `ctx`-driven
//! // execution; the literal here just illustrates the assertion shape.
//! let cg = CgroupStats {
//!     num_workers: 2,
//!     num_cpus: 2,
//!     max_gap_ms: 50,
//!     p99_wake_latency_us: 25.0,
//!     median_wake_latency_us: 10.0,
//!     total_iterations: 5_000,
//!     ..Default::default()
//! };
//! let work_units = 10_000u64;
//! let throughput = work_units as f64 / 5.0;
//!
//! let mut v = Assert::default_checks().verdict();
//! cg.claim_max_gap_ms(&mut v).at_most(100);          // typed CgroupStats accessor
//! cg.claim_p99_wake_latency_us(&mut v).at_most(50.0);
//! cg.claim_total_iterations(&mut v).at_least(1_000);
//! claim!(v, work_units).at_least(5_000);             // local-binding label
//! claim!(v, throughput).is_finite();                  // expression label
//! claim!(v, cg.wake_latency_tail_ratio()).between(1.0, 5.0);
//! let r = v.into_result();
//! assert!(r.passed);
//! ```
//!
//! Every claim is labeled by `stringify!` on either a struct field name
//! (via the derive) or an identifier/expression (via the macro), so a
//! rename or refactor updates the failure-message label automatically
//! and a stale call site fails to compile. There is no manual-string
//! escape hatch — by design, every label is source-text-grounded.
//!
//! Run with `cargo nextest run` (requires `/dev/kvm`).
//!
//! See the [`prelude`] module for the full set of re-exports.
//!
//! # Library usage
//!
//! ```toml
//! [dev-dependencies]
//! ktstr = { version = "0.5" }
//! ```
//!
//! The only feature flag is `integration`, which gates
//! `resolve_func_ip` visibility for integration tests.
//!
//! # Crate organization
//!
//! - [`cache`] -- kernel image cache (XDG directories, metadata, atomic writes)
//! - [`cgroup`] -- cgroup v2 filesystem operations
//! - [`cli`] -- shared helpers backing the `ktstr` and `cargo-ktstr` binaries
//! - [`fetch`] -- kernel tarball and git source acquisition
//! - [`flock`] -- advisory file-locking primitives used by cache + LLC reservations
//! - [`kernel_path`] -- kernel ID parsing and filesystem image discovery
//! - [`remote_cache`] -- GitHub Actions cache integration
//! - [`scenario`] -- declarative ops API (`CgroupDef`, `Step`, `Op`, `Backdrop`, `execute_defs`, `execute_steps`, `execute_scenario`)
//! - [`scenario::scenarios`] -- curated canned scenarios for common patterns
//! - [`mod@assert`] -- pass/fail assertions (starvation, isolation, fairness)
//! - [`test_support`] -- `#[ktstr_test]` runtime and registration
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`verifier`] -- BPF verifier log parsing, cycle detection, and output formatting
//! - [`worker_ready`] / [`worker_ready_wait`] -- pid-scoped marker file the alloc/test workers write before the parent samples them
//! - [`workload`] -- worker process types and telemetry collection
//!
//! ## ctprof subsystem
//!
//! Per-thread + per-process runtime profile, captured via
//! `ktstr ctprof capture` and compared via
//! `ktstr ctprof compare`:
//!
//! - [`host_context`] -- one-shot host snapshot (kernel, CPU, memory, tunables)
//! - [`host_heap`] -- jemalloc global heap counters (mallctl)
//! - [`ctprof`] -- per-thread procfs walk + cumulative scheduling, I/O, page-fault, jemalloc TSD counters
//! - [`ctprof_compare`] -- two-snapshot diff engine (group-by + delta tables)
//!
//! `host_thread_probe` (the ELF/DWARF + ptrace + `process_vm_readv`
//! engine that pulls per-thread jemalloc TSD counters) is
//! `pub(crate)`-only and consumed exclusively by `ctprof` plus
//! the source-shared standalone `ktstr-jemalloc-probe` binary.
//! Direct probe access from downstream is intentionally not part
//! of the surface — scheduler authors get the captured counters
//! through `ctprof::ThreadState`.
//!
//! Internal modules (not re-exported): `host_thread_probe` reads
//! per-thread jemalloc TSD counters via ptrace, `monitor` reads
//! live guest state, `probe` attaches BPF probes to traced
//! functions, `vmm` owns the KVM VM lifecycle, and `timeline`
//! correlates stimulus events with monitor samples for
//! phase-aligned reporting.

// `#[derive(Payload)]` expands into `::ktstr::test_support::...`
// paths so downstream crates can use it without a `use` import.
// This alias lets the same derive be used inside the ktstr crate
// itself — for example by doctests and by integration-test modules
// under `tests/common/` that pull the derive through the same
// public path downstream authors take. No runtime cost:
// `extern crate self as ktstr` is a pure name-binding.
extern crate self as ktstr;

#[allow(
    clippy::all,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod bpf_skel;

#[cfg(test)]
#[macro_use]
mod test_macros;

/// Shared guidance for every `#[non_exhaustive]` type in this
/// crate. Individual types link here instead of repeating the
/// same migration rules in every doc block.
///
/// # `#[non_exhaustive]` conventions in ktstr
///
/// Most of ktstr's public structs and enums carry `#[non_exhaustive]`
/// so that adding a field or variant is not a breaking change for
/// downstream crates. The attribute has two consequences downstream
/// consumers must account for:
///
/// ## Pattern matching
///
/// Matches on a `#[non_exhaustive]` struct or enum from outside this
/// crate must end with a wildcard `..` (for structs) or `_ =>` arm
/// (for enums). Without it, a future addition to the type forces
/// every matcher into a compile break even when the new field or
/// variant is irrelevant to the caller.
///
/// ```ignore
/// // Good: `..` absorbs future fields.
/// if let MyStruct { name, .. } = value { /* ... */ }
/// match my_enum {
///     MyEnum::A => {}
///     MyEnum::B => {}
///     _ => {}          // absorbs future variants
/// }
/// ```
///
/// ## Construction
///
/// Cross-crate consumers **cannot** use any struct-expression form
/// for a `#[non_exhaustive]` struct — bare literals
/// (`MyStruct { name: "x", .. }`) and functional-update spreads
/// (`MyStruct { name: "x", ..Default::default() }`) are both
/// rejected by the compiler (E0639). Construction must go through
/// one of:
///
/// 1. A dedicated constructor (`MyStruct::new(...)`,
///    `MyStruct::from_*(...)`) exposed by the defining crate.
/// 2. A [`Default`] instance followed by field mutation, when the
///    type derives `Default`.
/// 3. A named `test_fixture` or equivalent associated function for
///    types that expose a populated baseline instead of the
///    all-default minimum.
///
/// The per-type doc picks whichever of these the type actually
/// supports; see [`host_context::HostContext`],
/// [`host_heap::HostHeapState`], and the Op/CpusetSpec docs in
/// [`scenario::ops`] for worked examples across the different
/// shapes.
///
/// ## Pattern matching inside this crate
///
/// `#[non_exhaustive]` is enforced only across crate boundaries.
/// In-crate matchers can remain exhaustive (and should, so the
/// compiler flags forgotten variants at the definition site), and
/// in-crate struct-literal construction still works for the tests
/// and fixtures that live alongside the type.
#[doc(hidden)]
pub mod non_exhaustive {}

pub mod cache;
pub mod cgroup;
pub mod flock;

/// Map a raw errno value to its C constant name.
///
/// Returns `None` for unrecognized values. [`nix::errno::Errno`] has
/// `#[derive(Debug)]`, but `format!("{:?}", e)` allocates a fresh
/// `String` on every call; the hand-rolled match below returns a
/// `&'static str` pointing at a literal instead. [`nix::errno::Errno`]
/// is used here to gate unknown errnos via
/// `matches!(e, UnknownErrno)`. Adding a new errno means extending
/// both nix's port-constants table (for the UnknownErrno gate) and
/// this match; the test suite pins a representative subset so a
/// stale arm surfaces at build time.
pub(crate) fn errno_name(errno: i32) -> Option<&'static str> {
    let e = nix::errno::Errno::from_raw(errno);
    if matches!(e, nix::errno::Errno::UnknownErrno) {
        return None;
    }
    // Hand-rolled match: returns a `&'static str` pointing at a
    // literal, avoiding the allocation that `format!("{:?}", e)` would
    // incur. Callers that compare these against string literals in
    // error formatting paths rely on the stable symbolic names below.
    Some(match e {
        nix::errno::Errno::EPERM => "EPERM",
        nix::errno::Errno::ENOENT => "ENOENT",
        nix::errno::Errno::ESRCH => "ESRCH",
        nix::errno::Errno::EINTR => "EINTR",
        nix::errno::Errno::EIO => "EIO",
        nix::errno::Errno::ENXIO => "ENXIO",
        nix::errno::Errno::E2BIG => "E2BIG",
        nix::errno::Errno::ENOEXEC => "ENOEXEC",
        nix::errno::Errno::EBADF => "EBADF",
        nix::errno::Errno::ECHILD => "ECHILD",
        nix::errno::Errno::EAGAIN => "EAGAIN",
        nix::errno::Errno::ENOMEM => "ENOMEM",
        nix::errno::Errno::EACCES => "EACCES",
        nix::errno::Errno::EFAULT => "EFAULT",
        nix::errno::Errno::EBUSY => "EBUSY",
        nix::errno::Errno::EEXIST => "EEXIST",
        nix::errno::Errno::ENODEV => "ENODEV",
        nix::errno::Errno::ENOTDIR => "ENOTDIR",
        nix::errno::Errno::EISDIR => "EISDIR",
        nix::errno::Errno::EINVAL => "EINVAL",
        nix::errno::Errno::ENFILE => "ENFILE",
        nix::errno::Errno::EMFILE => "EMFILE",
        nix::errno::Errno::ENOSPC => "ENOSPC",
        nix::errno::Errno::ESPIPE => "ESPIPE",
        nix::errno::Errno::EROFS => "EROFS",
        nix::errno::Errno::EPIPE => "EPIPE",
        nix::errno::Errno::EDOM => "EDOM",
        nix::errno::Errno::ERANGE => "ERANGE",
        nix::errno::Errno::EDEADLK => "EDEADLK",
        nix::errno::Errno::ENAMETOOLONG => "ENAMETOOLONG",
        nix::errno::Errno::ENOSYS => "ENOSYS",
        nix::errno::Errno::ENOTEMPTY => "ENOTEMPTY",
        nix::errno::Errno::ELOOP => "ELOOP",
        nix::errno::Errno::ENOTSUP => "ENOTSUP",
        nix::errno::Errno::EADDRINUSE => "EADDRINUSE",
        nix::errno::Errno::ECONNREFUSED => "ECONNREFUSED",
        nix::errno::Errno::ETIMEDOUT => "ETIMEDOUT",
        // Other well-defined constants exist on nix::errno::Errno
        // but were not in the previous curated list. Return None for
        // them to preserve the prior contract — callers that want
        // more coverage can extend this match explicitly.
        _ => return None,
    })
}

/// Read the kernel ring buffer (equivalent to `dmesg --notime`).
/// Exposed as `pub` so scenario tests that need to assert on
/// kernel-log content (e.g. the sched_ext stall duration emitted
/// by `scx_exit(SCX_EXIT_ERROR_STALL)` in `kernel/sched/ext.c`)
/// can read the same buffer the framework captures into
/// `AssertResult::details` on scheduler-died failures.
pub fn read_kmsg() -> String {
    match rmesg::log_entries(rmesg::Backend::Default, false) {
        Ok(entries) => entries
            .iter()
            .map(|e| e.message.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        Err(_) => String::new(),
    }
}

pub mod assert;
pub(crate) mod budget;
pub(crate) mod cargo_test_mode;
pub mod cli;
pub mod cpu_util;
pub mod ctprof;
pub mod ctprof_compare;
pub(crate) mod elf_strip;
pub mod export;
pub mod fetch;
pub mod fun;
pub mod host_context;
pub mod host_heap;
pub(crate) mod host_thread_probe;
pub mod kernel_path;
pub mod metric_types;
pub(crate) mod monitor;
pub(crate) mod probe;
pub(crate) mod report;
pub mod scenario;
pub(crate) mod stats;
pub(crate) mod taskstats;
pub mod test_support;
pub(crate) mod timeline;
pub mod topology;

/// Public surface for the live-host introspection pipeline.
///
/// Re-exports from the otherwise-internal `monitor` module so the
/// live-host capture binary, integration tests, and downstream
/// consumers can invoke the bpf()-syscall data path, kernel
/// auto-discovery, kallsyms parser, dmesg-scx parser, and the
/// reproducer-generator translation layer without the `monitor`
/// module's frozen-VM internals leaking into the public API.
///
/// This module is the entry point for binaries and tests that
/// consume the live-host capture pipeline.
pub mod live_host {
    pub use crate::monitor::bpf_map::{
        BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_ARRAY,
        BpfMapAccessor, BpfMapInfo,
    };
    pub use crate::monitor::bpf_syscall::BpfSyscallAccessor;
    pub use crate::monitor::debug_capture::{
        AffinityHint, CgroupHint, CtprofSampleRef, DEBUG_CAPTURE_SCHEMA, DebugCapture,
        SchedPolicyHint, WorkTypeHint, WorkloadFingerprint, WorkloadGroupHint, project_fingerprint,
    };
    pub use crate::monitor::dmesg_scx::{
        ScxExitEvent, ScxExitKind, StackSymbol, extract_stack_symbols, parse_kmsg_window,
    };
    pub use crate::monitor::live_host_kernel::{KallsymsTable, LiveHostKernelEnv, uname_release};
    pub use crate::monitor::reproducer_gen::{
        ReproducerNote, ReproducerSpec, generate_spec, render_ktstr_test_source,
        render_run_file_source,
    };
    pub use crate::monitor::timeline::{
        DEFAULT_SNAPSHOT_RING_DEPTH, IncrementalCapture, IncrementalSnapshot, SnapshotRing,
        TimelineCapture, TimelineEvent, TimelineEventRaw, parse_timeline_buf,
        parse_timeline_record, tl_evt,
    };
}

pub mod remote_cache;
pub(crate) mod sync;
pub(crate) mod tar_util;
pub mod verifier;
pub mod vm;
pub(crate) mod vmm;
pub mod worker_ready;

/// Test-only seams for the freeze coordinator's exit_kind gate.
/// Integration tests under `tests/` flip these to force the rare
/// silent-drop-fix branches (KVA translate failure, BPF latch
/// rescue) that real workloads only hit during teardown races. Both
/// statics default to `false`; production code paths read them on
/// every gate decision via two relaxed atomic loads — one per
/// static — that sit immediately before the KVA-translate and
/// `.bss` reads they gate, work that dwarfs the load cost. Relaxed
/// is sufficient because the bool is the entire signal: no other
/// memory's visibility depends on the flag value, and cross-thread
/// happens-before between the test setter and the freeze
/// coordinator is established by the surrounding rendezvous
/// mechanism (KVM signals, eventfds, mutexes), which is far stronger
/// than any single-bool ordering. The gate itself fires at most
/// once per error-class exit (not per task-switch), so even the
/// relaxed load is negligible against the surrounding rendezvous
/// work. `#[doc(hidden)]` keeps the symbols out of the published
/// rustdoc surface — `pub` is the only path that survives the
/// `tests/` integration-test boundary (Rust `#[cfg(test)]`-gated
/// items are invisible to integration tests, which compile as
/// separate crates linking against the library's public surface).
#[doc(hidden)]
pub use vmm::freeze_coord::{
    FREEZE_COORD_TEST_FORCE_BSS_TRIGGERED, FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE,
};

/// Pre-populate the on-disk cast analysis cache for a scheduler binary.
///
/// Called by cargo-ktstr before spawning nextest so test processes
/// find a warm cache instead of each independently running the 30s
/// analysis. Safe to call from a background thread — the function
/// is idempotent (content-hash-keyed) and writes atomically.
pub fn precompute_cast_analysis(path: &std::path::Path) {
    vmm::cast_analysis_load::cached_cast_analysis_for_scheduler(path);
}
pub mod worker_ready_wait;
pub mod workload;

/// Static busybox binary compiled in build.rs for guest shell mode.
pub(crate) const BUSYBOX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

/// Contents of `ktstr.kconfig` (the kernel-config fragment that
/// enables sched_ext, BPF, kprobes, cgroups, and the other options
/// ktstr requires) baked into the binary at build time via
/// `include_str!`. Consumed by the kernel build pipeline to
/// `olddefconfig` a kernel source tree, and used to derive the
/// cache key suffix so a kconfig change produces a fresh cache
/// entry.
pub const EMBEDDED_KCONFIG: &str = include_str!("../ktstr.kconfig");

/// CRC32 hash of the embedded kconfig fragment (8 hex chars).
pub fn kconfig_hash() -> String {
    format!("{:08x}", crc32fast::hash(EMBEDDED_KCONFIG.as_bytes()))
}

/// CRC32 hash (8 hex chars) of a user-supplied `--extra-kconfig`
/// fragment, hashed verbatim.
///
/// Hashes raw bytes — no comment stripping, no CRLF
/// canonicalization. Two semantically-equivalent inputs with
/// different comments or line endings produce different hashes and
/// therefore land at distinct cache entries — accept the disk waste
/// in exchange for byte-deterministic cache discrimination.
pub fn extra_kconfig_hash(extra: &str) -> String {
    format!("{:08x}", crc32fast::hash(extra.as_bytes()))
}

/// Cache key suffix derived from the embedded kconfig fragment.
/// Used in kernel cache keys so a kconfig change produces a distinct
/// cache entry. The kernel binary is independent of ktstr userspace
/// source, so no ktstr or consumer build identity feeds this suffix.
pub fn cache_key_suffix() -> String {
    kconfig_hash()
}

/// Two-segment cache key suffix accounting for an optional
/// `--extra-kconfig` fragment.
///
/// The suffix uses TWO segments instead of folding both inputs into
/// one hash:
///
/// - `extra = None` → `kconfig_hash()` only — byte-identical to
///   [`cache_key_suffix`], so paths that don't expose
///   `--extra-kconfig` (test / coverage / shell / verifier) keep
///   resolving the existing keyspace and pre-1.0 cached kernels are
///   not orphaned.
/// - `extra = Some(content)` → `{kconfig_hash()}-xkc{extra_hash}`,
///   making `kernel list` self-describing: a reader can see at a
///   glance which entries carry user extras and which are pure
///   baked-in builds. Different extra content yields different
///   `xkc{...}` segments, so cache discrimination across distinct
///   `--extra-kconfig` invocations is structural rather than
///   collapsed into a single opaque hash.
pub fn cache_key_suffix_with_extra(extra: Option<&str>) -> String {
    match extra {
        None => kconfig_hash(),
        Some(content) => format!("{}-xkc{}", kconfig_hash(), extra_kconfig_hash(content)),
    }
}

/// Merge the user-supplied `--extra-kconfig` fragment on top of
/// [`EMBEDDED_KCONFIG`] for the configure pass. Returns a
/// [`std::borrow::Cow`] so the no-extras branch borrows `baked`
/// without allocating; only the `Some` branch heaps the merged
/// String.
///
/// The user fragment is appended AFTER the baked-in fragment so
/// kbuild's last-wins rule
/// (`scripts/kconfig/confdata.c::conf_read_simple` —
/// "If conflicting CONFIG options are given from an input file,
/// the last one wins.") makes user values override baked-in ones
/// on conflict.
///
/// A single `\n` separator is interleaved between the two
/// fragments. EMBEDDED_KCONFIG ends in a newline today, so the
/// interleaved `\n` produces a blank line between the segments —
/// kbuild's `.config` parser ignores blank lines (every
/// `if (!line[0])` short-circuit in `conf_read_simple`), so the
/// blank line is harmless. The separator is mandatory for the
/// adversarial case where the operator hand-crafts an
/// EMBEDDED_KCONFIG without a trailing newline AND a user
/// fragment that starts with `CONFIG_X` — without the
/// interleaved `\n`, the two would concatenate into a single
/// malformed line. Always emit the separator so the merge is
/// safe regardless of either side's terminator.
///
/// The production configure path in
/// [`crate::cli::kernel_build_pipeline`] calls this helper to build
/// the bytes handed to `configure_kernel`. Tests that assert
/// merge-ordering invariants call it directly so the production
/// byte sequence is what kbuild's last-wins rule operates on.
/// (Note: [`cache_key_suffix_with_extra`] hashes `extra` ALONE for
/// its `xkc{...}` segment — it doesn't pass through this helper —
/// so the cache-key suffix and the merged-fragment content evolve
/// independently. The cache-key segment exists to discriminate
/// extras-vs-no-extras at the cache layer; the merge ordering
/// exists to give kbuild the right final value.)
pub fn merge_kconfig_fragments<'a>(
    baked: &'a str,
    extra: Option<&str>,
) -> std::borrow::Cow<'a, str> {
    match extra {
        None => std::borrow::Cow::Borrowed(baked),
        Some(content) => std::borrow::Cow::Owned(format!("{baked}\n{content}")),
    }
}

// Derive macros. `Payload` here is the `#[derive(Payload)]` proc
// macro; the same-named `Payload` struct (to which the derive
// applies) lives at `crate::test_support::Payload`. Rust's
// macro-vs-type namespace separation lets both coexist under the
// identifier `Payload` in `use ktstr::prelude::*;` — the derive
// position resolves to the macro, type position resolves to the
// struct.
pub use ktstr_macros::Claim;
pub use ktstr_macros::Payload;
pub use ktstr_macros::declare_scheduler;
pub use ktstr_macros::json;
pub use ktstr_macros::ktstr_test;

/// Internal re-exports for proc-macro-generated code. Not public API.
///
/// Grouped into a single hidden module so that `use ktstr::*;` pulls
/// in one module name instead of two leading-underscore items.
/// Consumers of `#[ktstr_test]` should not reference anything under
/// this path — the macro expansion names these crates via
/// `::ktstr::__private::ctor` / `serde_json` and the set may change
/// without notice. (`linkme` lives at the public crate root —
/// [`ktstr::linkme`](crate::linkme) — since the macro now emits the
/// public path.)
#[doc(hidden)]
pub mod __private {
    pub use ctor;
    pub use serde_json;
}

#[cfg(feature = "integration")]
pub use crate::probe::process::resolve_func_ip;

/// The `linkme` crate, re-exported as part of ktstr's public surface
/// so downstream code can reference it via [`ktstr::linkme`](crate::linkme)
/// in the `#[linkme(crate = ...)]` annotation that
/// [`distributed_slice`](crate::distributed_slice) registrations
/// require — without having to add `linkme` as a direct Cargo
/// dependency. See [`distributed_slice`](crate::distributed_slice)
/// for the usage pattern.
pub use ::linkme;

/// `linkme::distributed_slice` re-exported as part of ktstr's public
/// surface. Combined with [`crate::linkme`] for the
/// `#[linkme(crate = ...)]` annotation, this lets a downstream crate
/// register entries into
/// [`KTSTR_TESTS`](crate::test_support::KTSTR_TESTS) or
/// [`KTSTR_SCHEDULERS`](crate::test_support::KTSTR_SCHEDULERS)
/// without adding `linkme` as a direct Cargo dependency:
///
/// ```ignore
/// use ktstr::prelude::*;
///
/// fn my_test_fn(_ctx: &Ctx) -> Result<AssertResult> {
///     Ok(AssertResult::pass())
/// }
///
/// #[distributed_slice(KTSTR_TESTS)]
/// #[linkme(crate = ktstr::linkme)]
/// static MY_ENTRY: KtstrTestEntry = KtstrTestEntry {
///     name: "my_test",
///     func: my_test_fn,
///     ..KtstrTestEntry::DEFAULT
/// };
/// ```
///
/// The `#[linkme(crate = ...)]` annotation is REQUIRED because the
/// `linkme` proc-macro expansion hardcodes `::linkme::DistributedSlice`
/// — without the annotation, downstream crates without `linkme` in
/// their `Cargo.toml` get an unresolved-import error.
/// The annotation tells the macro to resolve type references through
/// `ktstr::linkme` instead, which IS reachable from downstream by
/// transitive dependency.
///
/// Downstream crates that already depend on `linkme = "0.3"` directly
/// can omit the annotation. The `#[ktstr_test]` proc macro emits both
/// attributes internally so test authors using the standard macro
/// surface never have to spell either out.
pub use linkme::distributed_slice;

/// Re-exports for writing `#[ktstr_test]` functions.
///
/// ```rust
/// use ktstr::prelude::*;
///
/// #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
/// fn my_test(ctx: &Ctx) -> Result<AssertResult> {
///     Ok(AssertResult::pass())
/// }
/// ```
///
/// For curated canned scenarios, see [`scenario::scenarios`].
pub mod prelude {
    pub use anyhow::Result;

    // `Scheduler` is the `test_support::Scheduler` struct — the
    // scheduler-definition record test authors build via the
    // `declare_scheduler!` macro.
    pub use crate::assert::{
        Assert, AssertDetail, AssertResult, COMPARATOR_VOCABULARY, ClaimBuilder, DetailKind,
        EachClaim, InfoNote, MAX_RECORDED_PASSES, NoteValue, Outcome, OutcomeRef,
        PASSES_TRUNCATION_SENTINEL_COMPARATOR, PASSES_TRUNCATION_SENTINEL_NAME, PassDetail,
        SchedulerBaseline, SeqClaim, SeriesField, SetClaim, Verdict, assert_baseline,
        assert_scx_events_clean,
    };
    pub use crate::cgroup::CgroupManager;
    pub use crate::claim;
    pub use crate::declare_scheduler;
    pub use crate::distributed_slice;
    pub use crate::host_context::HostContext;
    pub use crate::host_heap::HostHeapState;
    pub use crate::ktstr_test;
    pub use crate::scenario::backdrop::Backdrop;
    pub use crate::scenario::ops::{
        CgroupDef, CpusetSpec, HoldSpec, KernelTarget, KernelValue, KernelValueWidth, Op, Setup,
        Step, execute_defs, execute_scenario, execute_scenario_with, execute_steps,
        execute_steps_with,
    };
    pub use crate::scenario::payload_run::{PayloadHandle, PayloadRun};
    pub use crate::scenario::scenarios;
    // Snapshot accessor surface and the underlying report shapes
    // a test author needs to inspect the captured BTF-rendered
    // bytes. The renderer types come from monitor::btf_render and
    // monitor::dump (otherwise crate-private modules); re-exported
    // here so an out-of-crate caller can build synthetic
    // FailureDumpReports for unit-testing their assertions
    // against the snapshot accessor without booting a VM.
    //
    // Re-export of the `Payload` derive macro from the crate root.
    // The same identifier names the `Payload` struct re-exported a
    // few lines below from `crate::test_support`; the two live in
    // separate Rust namespaces (macro vs type) so they coexist in
    // `use ktstr::prelude::*;` without conflict.
    pub use crate::Payload;
    pub use crate::monitor::arena::{ArenaPage, ArenaSnapshot};
    pub use crate::monitor::bpf_prog::ProgRuntimeStats;
    pub use crate::monitor::btf_render::{RenderedMember, RenderedValue};
    pub use crate::monitor::dump::{
        DegradedFailureDumpReport, DualFailureDumpReport, EventCounterSample, FailureDumpEntry,
        FailureDumpFdArray, FailureDumpMap, FailureDumpPercpuEntry, FailureDumpPercpuHashEntry,
        FailureDumpReport, FailureDumpReportAny, FailureDumpRingbuf, FailureDumpStackTrace,
        FailureDumpStackTraceEntry, PerCpuTimeStats, PerNodeNumaStats, ProbeBssCounters,
        REASON_DEGRADED_RENDEZVOUS_TIMEOUT, SCHEMA_DEGRADED, SCHEMA_DUAL, SCHEMA_SINGLE,
        SNAPSHOT_TAG_EARLY_DEGRADED, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED,
        SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED, SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
    };
    pub use crate::monitor::scx_walker::{DsqState, RqScxState, ScxSchedState};
    pub use crate::monitor::task_enrichment::TaskEnrichment;
    pub use crate::scenario::sample::{
        BpfMapProjector, Sample, SampleSeries, StatsPathProjector, StatsValue,
    };
    pub use crate::scenario::snapshot::{
        BridgeGuard, CaptureCallback, JsonField, MAX_WATCH_SNAPSHOTS, Snapshot, SnapshotBridge,
        SnapshotEntry, SnapshotError, SnapshotField, SnapshotMap, SnapshotResult,
        WatchRegisterCallback, stats_path,
    };
    pub use crate::scenario::{CgroupGroup, Ctx, collect_all, spawn_diverse};
    // `Payload` in this group is the struct on which
    // `#[derive(Payload)]` is applied; it occupies the type
    // namespace, distinct from the derive macro re-exported above.
    pub use crate::test_support::{
        BpfMapWrite, CgroupPath, KTSTR_SCHEDULERS, KTSTR_TESTS, KtstrTestEntry, MemSideCache,
        Metric, MetricBounds, MetricCheck, MetricHint, MetricSource, MetricStream, NumaDistance,
        NumaNode, OutputFormat, Payload, PayloadKind, PayloadMetrics, Polarity, Scheduler,
        SchedulerSpec, SidecarResult, Sysctl, Topology, TopologyConstraints, extract_metrics,
        find_scheduler, find_test, sidecar_dir,
    };
    // The following items are intentionally NOT in the prelude. They
    // are binary-entry helpers (the `ktstr` / `cargo-ktstr` bins) or
    // macro-generated glue the `#[ktstr_test]` expansion consumes —
    // audiences distinct from the test-author surface this module
    // provides. Import directly from `ktstr::test_support::<item>`
    // when needed:
    // `newest_run_dir`, `runs_root`, `analyze_sidecars`, `ktstr_main`,
    // `ktstr_test_early_dispatch`, `run_ktstr_test`,
    // `resolve_scheduler`, `resolve_test_kernel`.
    //
    // `build_nodemask` (the low-level `set_mempolicy(2)` / `mbind(2)`
    // bitmask builder) is also excluded: test authors express NUMA
    // placement through the `MemPolicy` enum, not raw nodemask
    // construction. The helper itself lives in the crate-private
    // `workload::spawn` submodule with a `pub(crate)` re-export at
    // `crate::workload::build_nodemask` for `vmm::host_topology`
    // internal use.
    pub use crate::topology::{LlcInfo, NodeMemInfo, TestTopology};
    pub use crate::vmm::{VirtioBlkCountersSnapshot, VirtioNetCountersSnapshot};
    // `VmResult` is the host-side return value from booting a VM.
    // Surfaced for `#[ktstr_test(post_vm = ...)]` callbacks: the
    // hook signature is `fn(&VmResult) -> anyhow::Result<()>`, and
    // a test author writing the callback needs the type in scope
    // to declare the parameter.
    pub use crate::vmm::VmResult;
    pub use crate::vmm::disk_config::{
        DiskConfig, DiskThrottle, DiskThrottleValidationError, Filesystem, ThrottleDimension,
    };
    pub use crate::vmm::net_config::NetConfig;
    pub use crate::workload::{
        AffinityIntent, AluWidth, CloneMode, MemPolicy, Migration, MpolFlags, Phase,
        ResolvedAffinity, SchedPolicy, WorkSpec, WorkType, WorkTypeValidationError, WorkerReport,
        WorkloadConfig, WorkloadHandle,
    };
}

/// Name of the environment variable that selects a kernel for every
/// ktstr entry point (`ktstr run`, `ktstr shell`, `cargo ktstr test`,
/// in-process tests, post-run analysis). Single source of truth so
/// the name is not spelled by hand at each reader; if the name ever
/// changes, the change lands in one place instead of fanning out to
/// every call site.
pub const KTSTR_KERNEL_ENV: &str = "KTSTR_KERNEL";

/// Name of the environment variable that carries the multi-kernel
/// fan-out list across the `cargo ktstr` → `cargo nextest` → test-
/// binary boundary. Format: `label1=path1;label2=path2;…` (semicolon
/// entry separator, `=` separates label from absolute kernel-dir
/// path). Empty / unset means "single-kernel mode" — the test binary
/// honours `KTSTR_KERNEL_ENV` directly.
///
/// Set by `cargo ktstr test --kernel A --kernel B` (or any
/// `--kernel` value that expands to ≥ 2 entries — repeated
/// `--kernel` flags, or a single `--kernel START..END` range that
/// expands to multiple stable releases via
/// [`crate::kernel_path::KernelId::Range`]) before the `exec` into
/// `cargo nextest`. Read by the test binary's `--list` /
/// `--exact` handlers in `crate::test_support::dispatch` to fan
/// the gauntlet across kernels: each (test × scenario × topology ×
/// kernel) tuple becomes a distinct nextest test case so
/// nextest's parallelism, retries, and `-E` filtering work
/// natively. Per-variant subprocesses re-export `KTSTR_KERNEL` to
/// the kernel directory selected by the test name's `kernel_…`
/// suffix.
///
/// `KTSTR_KERNEL_ENV` is always set in tandem (to the first entry's
/// path) so downstream code that reads `KTSTR_KERNEL` directly —
/// budget-listing's vmlinux probe in `dispatch.rs` for example —
/// still observes a valid kernel even when running under multi-
/// kernel mode.
///
/// Single source of truth so the name is not spelled by hand at
/// each reader; if the name ever changes, the change lands in one
/// place instead of fanning out to every call site.
pub const KTSTR_KERNEL_LIST_ENV: &str = "KTSTR_KERNEL_LIST";

/// Name of the environment variable that overrides the rayon
/// pool width used by `cargo ktstr`'s `resolve_kernel_set` to
/// fan out per-spec kernel resolves (download / git-clone /
/// build) in parallel. Default cap is `available_parallelism()`
/// — the host's logical CPU count — chosen so download streams
/// do not outnumber threads the host can drive without
/// thrashing a contended local network (kernel.org CDN
/// per-IP throttle, developer ISP, CI shared NIC).
///
/// Operators override when the default is wrong for their
/// environment: a fast NIC + slow CPU benefits from raising
/// the cap above logical-CPU count to keep more downloads
/// in flight; a contended CI runner with concurrent jobs
/// benefits from lowering it to 1 or 2 to leave bandwidth
/// for siblings; a multi-version `--kernel A..Z` resolve on
/// a workstation may want a hand-tuned middle value to
/// balance throughput against background load.
///
/// Parsed as `usize`; 0 and unparseable values fall through
/// to the default cap so a typoed export does not silently
/// disable parallelism. Leading/trailing whitespace is trimmed
/// before parsing so a shell-quoted `=" 8 "` behaves the same
/// as the unquoted form. Read by
/// [`crate::cli::resolve_kernel_parallelism`] (the helper
/// that combines this env value with the
/// `available_parallelism()` fallback) so the parsing rules
/// live in one place.
///
/// Single source of truth so the name is not spelled by hand at
/// each reader; if the name ever changes, the change lands in one
/// place instead of fanning out to every call site.
pub const KTSTR_KERNEL_PARALLELISM_ENV: &str = "KTSTR_KERNEL_PARALLELISM";

/// Name of the environment variable that switches the `cargo ktstr
/// verifier` per-cell handler from the cycle-collapsed default
/// rendering to a raw scheduler-log dump. Set to any value (the
/// presence of the variable is what matters; the value is ignored)
/// by the dispatcher in `src/bin/cargo_ktstr/verifier.rs` when the
/// operator passes `--raw`, and read by
/// `crate::test_support::dispatch::run_verifier_cell` before
/// formatting via [`crate::verifier::format_verifier_output`].
///
/// Single source of truth so the name is not spelled by hand at
/// each reader; if the name ever changes, the change lands in one
/// place instead of fanning out to every call site.
pub const KTSTR_VERIFIER_RAW_ENV: &str = "KTSTR_VERIFIER_RAW";

/// Shared skip / error hint for call sites that cannot proceed
/// without a resolvable kernel. Phrased so the user sees the same
/// wording regardless of which layer surfaced the failure — tests,
/// CLI, monitor probes, and sidecar writers all point the operator
/// at the same remediation. Referenced by the non-VM-boot skip
/// paths in `cache.rs`, `probe/btf.rs`, `monitor/mod.rs`,
/// `test_support/eval.rs`, and `test_support/mod.rs`.
///
/// Format: caller prefixes the actionable first clause (e.g.
/// "no vmlinux found") and appends this constant as the
/// remediation tail. Keeping the prefix per-caller lets each site
/// name the specific artifact it needs while the `KTSTR_KERNEL`
/// wording stays consistent.
// NOTE: the "accepted forms" enumeration here mirrors
// [`kernel_path::KERNEL_ID_GRAMMAR`] verbatim — keep in sync when
// either changes. (Composition at const time needs `concat!`-of-
// literals, and `KERNEL_ID_GRAMMAR` is a `const &str` not a literal.)
pub const KTSTR_KERNEL_HINT: &str = "set KTSTR_KERNEL to one of: \
    exact version (`6.14`), inclusive range (`6.14..7.0` or \
    `6.14..=7.0`), git source (`git+URL#REF`), absolute or \
    `~`-prefixed path, or cache key. List cached keys with \
    `cargo ktstr kernel list`; build new ones with \
    `cargo ktstr kernel build`";

/// Read [`KTSTR_KERNEL_ENV`] once, normalizing the raw value:
/// missing / empty / whitespace-only reads collapse to `None`, and
/// a surrounding-whitespace trim is applied so a shell-quoted
/// `KTSTR_KERNEL=" ../linux"` behaves the same as the unquoted
/// form. Every caller that reads the env var should route through
/// this helper so the normalization rules live in one place; a
/// future change to the rules (e.g. accepting a trailing slash)
/// propagates to every site automatically.
///
/// Returns the raw string; callers that need a structured
/// identifier parse with [`kernel_path::KernelId::parse`].
pub fn ktstr_kernel_env() -> Option<String> {
    std::env::var(KTSTR_KERNEL_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Find a bootable kernel image on the host.
///
/// Resolution chain:
/// 1. `KTSTR_KERNEL` env var, parsed via `KernelId`:
///    - Path: search that directory for an arch-specific image
///    - Version/CacheKey: require cache access (error if cache
///      directory cannot be opened); on cache miss, skip the
///      general cache scan (step 2) and fall to filesystem
/// 2. XDG cache: most recent cached image (newest first)
/// 3. Local build trees (`./linux`, `../linux`,
///    `/lib/modules/{release}/build`)
/// 4. Host paths (`/lib/modules/{release}/vmlinuz`,
///    `/boot/vmlinuz-{release}`, `/boot/vmlinuz`)
///
/// Returns `Err` when `KTSTR_KERNEL` is a path that does not contain
/// a kernel image, or when it is a version/cache key and the cache
/// directory cannot be opened. Returns `Ok(None)` when no kernel is
/// found.
pub fn find_kernel() -> anyhow::Result<Option<std::path::PathBuf>> {
    use kernel_path::KernelId;

    let release = rustix::system::uname()
        .release()
        .to_str()
        .ok()
        .map(str::to_owned);
    let release_ref = release.as_deref();

    // Track whether KTSTR_KERNEL was set with a non-path value.
    // When the user explicitly requests a version or cache key that
    // misses cache, the general cache scan (step 2) must be skipped
    // to avoid silently returning a different kernel.
    let mut skip_cache_scan = false;

    // 1. KTSTR_KERNEL env var with KernelId parsing. Route through
    // `ktstr_kernel_env()` so the empty/whitespace normalization
    // matches every other reader in the crate.
    if let Some(val) = ktstr_kernel_env() {
        match KernelId::parse(&val) {
            KernelId::Path(ref p) => {
                // `KernelId::parse` already routed `val` through
                // `expand_tilde`, producing the resolved `PathBuf`
                // here. Pass that — not the raw `val` — into
                // `find_image` so a `~/...` env value resolves
                // against `$HOME`. Lossy `to_str` would silently
                // mishandle non-UTF-8 paths; bail explicitly with
                // the same hint shape as the not-found arm.
                let Some(s) = p.to_str() else {
                    anyhow::bail!(
                        "KTSTR_KERNEL={val} expands to a non-UTF-8 path. \
                         {KTSTR_KERNEL_HINT}"
                    );
                };
                match kernel_path::find_image(Some(s), release_ref) {
                    Some(found) => return Ok(Some(found)),
                    None => anyhow::bail!(
                        "KTSTR_KERNEL={val} does not contain a kernel image. {KTSTR_KERNEL_HINT}"
                    ),
                }
            }
            KernelId::Version(ref ver) => {
                // Only tarball keys use the {ver}-tarball-{arch}-kc{suffix} pattern.
                // Git keys are {ref}-git-{hash}-{arch}-kc{suffix} and local keys
                // are local-{hash}-{arch}-kc{suffix} — neither contains the
                // version as a prefix, so only tarball lookup is valid here.
                let cache = cache::CacheDir::new().map_err(|e| {
                    anyhow::anyhow!(
                        "KTSTR_KERNEL={val} requires cache access, \
                         but cache directory could not be opened: {e}"
                    )
                })?;
                let arch = std::env::consts::ARCH;
                let key = format!("{ver}-tarball-{arch}-kc{}", cache_key_suffix());
                if let Some(entry) = cache.lookup(&key) {
                    return Ok(Some(entry.image_path()));
                }
                // Version not in cache — skip general cache scan to
                // avoid returning a different kernel version.
                skip_cache_scan = true;
            }
            KernelId::CacheKey(ref key) => {
                let cache = cache::CacheDir::new().map_err(|e| {
                    anyhow::anyhow!(
                        "KTSTR_KERNEL={val} requires cache access, \
                         but cache directory could not be opened: {e}"
                    )
                })?;
                if let Some(entry) = cache.lookup(key) {
                    return Ok(Some(entry.image_path()));
                }
                // Explicit cache key not found — skip general cache scan.
                skip_cache_scan = true;
            }
            // Multi-kernel specs (`A..B` ranges, `git+URL#REF` sources)
            // are only meaningful at the test/coverage/verifier
            // subcommand entry points where the runner can fan out
            // across kernels. The KTSTR_KERNEL env reader resolves a
            // single kernel image for in-process use (BTF lookup,
            // direct boot path) and has no dispatch loop, so a range
            // or git spec here cannot be expanded.
            //
            // Run `validate()` first so an inverted range surfaces
            // the specific "swap the endpoints" diagnostic instead
            // of getting masked by the generic "not supported in
            // env-var form" bail below — operators with a typo see
            // the actionable fix; valid-but-unsupported specs get
            // the generic redirect.
            id @ (KernelId::Range { .. } | KernelId::Git { .. }) => {
                if let Err(e) = id.validate() {
                    anyhow::bail!("KTSTR_KERNEL={val}: {e}");
                }
                anyhow::bail!(
                    "KTSTR_KERNEL={val}: multi-kernel specs (ranges, \
                     git sources) are not supported in env-var form. \
                     Use --kernel on the test/coverage/verifier \
                     subcommands, or set KTSTR_KERNEL to a single \
                     version, cache key, or path."
                );
            }
        }
    }

    // 2. XDG cache: most recent cached image.
    // Skipped when KTSTR_KERNEL was an explicit version or cache key
    // that missed — returning a different kernel would be surprising.
    if !skip_cache_scan
        && let Ok(cache) = cache::CacheDir::new()
        && let Ok(entries) = cache.list()
    {
        let kc_hash = kconfig_hash();
        for listed in &entries {
            let cache::ListedEntry::Valid(entry) = listed else {
                continue;
            };
            // Skip entries built with a different kconfig. Untracked
            // (pre-kconfig-tracking) entries are reused — their image
            // could still boot correctly, and skipping them would
            // permanently orphan legacy cache entries.
            if entry.kconfig_status(&kc_hash).is_stale() {
                continue;
            }
            let image = entry.image_path();
            // TOCTOU guard: list() guarantees image existence at scan time,
            // but a concurrent cache-clean could delete between scan and use.
            if !image.exists() {
                continue;
            }
            // Guard: if a cached vmlinux is present but is missing
            // the symbols monitor code requires, skip the entry so
            // the caller falls through to a source tree. Older
            // caches built by a strip pipeline that dropped data
            // sections would pass the image-exists check but fail
            // downstream when the monitor initializes.
            if let Some(vmlinux) = entry.vmlinux_path()
                && let Err(e) = monitor::symbols::KernelSymbols::from_vmlinux(&vmlinux)
            {
                tracing::warn!(
                    entry = %entry.path.display(),
                    error = %e,
                    "skipping cached kernel with unusable vmlinux"
                );
                continue;
            }
            return Ok(Some(image));
        }
    }

    // 3-4. Filesystem fallbacks (local build trees, host paths).
    Ok(kernel_path::find_image(None, release_ref))
}

/// Build a cargo binary package and return its output path.
///
/// Runs from the ktstr crate's manifest directory (which is also the
/// workspace root in this repo) so that workspace-level feature
/// unification (e.g. vendored libbpf-sys) is always in effect,
/// regardless of the calling process's working directory.
pub fn build_and_find_binary(package: &str) -> anyhow::Result<std::path::PathBuf> {
    let output = std::process::Command::new("cargo")
        .args(["build", "-p", package, "--message-format=json"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| anyhow::anyhow!("cargo build: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("cargo build -p {package} failed:\n{stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line)
            && msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-artifact")
            && msg
                .get("profile")
                .and_then(|p| p.get("test"))
                .and_then(|t| t.as_bool())
                == Some(false)
            && msg
                .get("target")
                .and_then(|t| t.get("kind"))
                .and_then(|k| k.as_array())
                .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("bin")))
            && let Some(filenames) = msg.get("filenames").and_then(|f| f.as_array())
            && let Some(path) = filenames.first().and_then(|f| f.as_str())
        {
            return Ok(std::path::PathBuf::from(path));
        }
    }
    anyhow::bail!(
        "no binary artifact found for package '{package}' — cargo build \
         succeeded but no compiler-artifact JSON line declared a [[bin]] \
         target. Two common causes: (1) the package has no [[bin]] target \
         (library-only, or only [[example]] / [[bench]] targets); (2) the \
         cargo --message-format=json output shape changed and the \
         artifact walker missed the matching line. Run `cargo build -p \
         {package} --message-format=json` and check for a `compiler-artifact` \
         line with `\"target\":{{\"kind\":[\"bin\"],...}}` to confirm."
    )
}

/// Resolve the current executable path, falling back to `/proc/self/exe`
/// when the binary has been deleted (e.g. by `cargo llvm-cov`).
///
/// On Linux, `std::env::current_exe()` reads `/proc/self/exe`.  When the
/// binary is unlinked while running, the kernel appends ` (deleted)` to
/// the readlink target, producing a path that does not exist on disk.
/// `/proc/self/exe` itself remains usable as a file path because the
/// kernel keeps the inode alive, so we fall back to it.
pub(crate) fn resolve_current_exe() -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context;
    let exe = std::env::current_exe().context("resolve current exe")?;
    if exe.exists() {
        return Ok(exe);
    }
    let proc_exe = std::path::PathBuf::from("/proc/self/exe");
    anyhow::ensure!(
        proc_exe.exists(),
        "current exe not found: {}",
        exe.display()
    );
    Ok(proc_exe)
}

/// Boot a KVM VM in interactive shell mode.
///
/// Builds an initramfs with busybox and optional include files, then
/// launches a VM with bidirectional stdin/stdout forwarding. The guest
/// runs a shell via busybox; user-provided files are available at
/// `/include-files/<name>`.
///
/// `kernel`: path to the kernel image (bzImage/Image).
/// `numa_nodes`, `llcs`, `cores`, `threads`: guest CPU topology.
/// `include_files`: `(archive_path, host_path)` pairs for files to
///   include in the guest.
/// `memory_mib`: explicit guest memory override in MiB; conversion
///   at VM-launch is `value << 20` bytes. When `None`, memory is
///   computed from actual initramfs size after build.
/// `disk`: optional virtio-blk device backing for `/dev/vda`. When
///   `Some`, the framework calls
///   `vmm::KtstrVm::builder`'s `.disk(..)` so the guest probes a
///   raw block device sized per `disk.capacity_mib`.
#[allow(clippy::too_many_arguments)]
pub fn run_shell(
    kernel: std::path::PathBuf,
    numa_nodes: u32,
    llcs: u32,
    cores: u32,
    threads: u32,
    include_files: &[(&str, &std::path::Path)],
    memory_mib: Option<u32>,
    dmesg: bool,
    exec: Option<&str>,
    disk: Option<vmm::disk_config::DiskConfig>,
) -> anyhow::Result<()> {
    let payload = resolve_current_exe()?;

    let owned_includes: Vec<(String, std::path::PathBuf)> = include_files
        .iter()
        .map(|(a, p)| (a.to_string(), p.to_path_buf()))
        .collect();

    let mut cmdline = format!("KTSTR_MODE=shell KTSTR_TOPO={numa_nodes},{llcs},{cores},{threads}");
    if dmesg {
        cmdline.push_str(" loglevel=7");
    }
    if let Ok(val) = std::env::var("RUST_LOG") {
        cmdline.push_str(&format!(" RUST_LOG={val}"));
    }

    // Pass host terminal environment to guest.
    if let Ok(term) = std::env::var("TERM") {
        cmdline.push_str(&format!(" KTSTR_TERM={term}"));
    }
    if let Ok(ct) = std::env::var("COLORTERM") {
        cmdline.push_str(&format!(" KTSTR_COLORTERM={ct}"));
    }

    // Pass host terminal dimensions to guest for correct line wrapping.
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDIN_FILENO, libc::TIOCGWINSZ, &mut ws) == 0
            && ws.ws_col > 0
            && ws.ws_row > 0
        {
            cmdline.push_str(&format!(
                " KTSTR_COLS={} KTSTR_ROWS={}",
                ws.ws_col, ws.ws_row
            ));
        }
    }

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let mut builder = vmm::KtstrVm::builder()
        .kernel(&kernel)
        .init_binary(&payload)
        .topology(vmm::Topology::new(numa_nodes, llcs, cores, threads))
        .cmdline(&cmdline)
        .include_files(owned_includes)
        .busybox(true)
        .dmesg(dmesg)
        .no_perf_mode(no_perf_mode);

    if let Some(cmd) = exec {
        builder = builder.exec_cmd(cmd);
    }

    if let Some(d) = disk {
        builder = builder.disk(d);
    }

    builder = match memory_mib {
        Some(mib) => builder.memory_mib(mib),
        None => builder.memory_deferred(),
    };

    let vm = builder.build()?;

    vm.run_interactive()
}

#[cfg(test)]
mod tests;
