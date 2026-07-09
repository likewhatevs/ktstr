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
//! correctly. The same scenarios also run under the kernel's default
//! EEVDF scheduler, so a test can baseline sched_ext behavior against
//! stock scheduling.
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
//! assert!(r.is_pass());
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
//! Default install (full feature set — includes the installed
//! `ktstr` / `cargo-ktstr` bins' deps):
//!
//! ```toml
//! [dev-dependencies]
//! ktstr = "0.26.0"
//! ```
//!
//! Lean dev-dep (drops the host-tooling crates: tikv-jemallocator,
//! clap_complete, tree-sitter, tree-sitter-c, base64):
//!
//! ```toml
//! [dev-dependencies]
//! ktstr = { version = "0.26.0", default-features = false }
//! ```
//!
//! # Feature flags
//!
//! - **`cli-bins`** (default) — umbrella for deps used only by the
//!   four `src/bin/*.rs` entry points and the matching test-binary
//!   dispatch hooks. Pulls in `tikv-jemallocator`, `clap_complete`,
//!   `tree-sitter`, `tree-sitter-c`, and the `export` feature.
//! - **`export`** (pulled in by `cli-bins`) — gates
//!   [`mod@export`] and the `cargo ktstr export` dispatch path in
//!   the test binary. Drops `base64` from the manifest when off.
//! - **`wprof`** — embed the wprof BPF tracer in shell-mode VMs.
//!   First build clones `github.com/anakryiko/wprof` (requires git,
//!   make, gcc, clang, elfutils-devel, zlib-devel).
//! - **`pretty-labels`** — grex-based regex synthesis for
//!   `ctprof_compare` display labels. With the feature off,
//!   labels fall back to the deterministic join key.
//! - **`remote-cache`** — GitHub Actions cache backend for blob
//!   storage. CI-only; off-by-default. Pulls in `opendal` + minimal
//!   `tokio` runtime.
//! - **`integration`** — gates `resolve_func_ip` visibility for
//!   integration tests.
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
//! - [`mod@assert`] -- pass/fail assertions (worker progress, isolation, fairness)
//! - [`test_support`] -- `#[ktstr_test]` runtime and registration
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`verifier`] -- BPF verifier log parsing, cycle detection, and output formatting
//! - [`worker_ready`] / [`worker_ready_wait`] -- pid-scoped ready-marker the alloc/test worker writes once it is work-ready, polled (`worker_ready_wait`) before the probe is launched against it
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
//! functions, and `vmm` owns the KVM VM lifecycle.
//!
//! [`timeline`] is a public module (its `StimulusEvent` appears in
//! `assert::build_phase_buckets_with_stimulus`'s signature and is
//! re-exported in the [`prelude`]); it correlates stimulus events
//! with monitor samples for phase-aligned reporting.

// `#[derive(Payload)]` expands into `::ktstr::test_support::...`
// paths so downstream crates can use it without a `use` import.
// This alias lets the same derive be used inside the ktstr crate
// itself — for example by doctests and by integration-test modules
// under `tests/common/` that pull the derive through the same
// public path downstream authors take. No runtime cost:
// `extern crate self as ktstr` is a pure name-binding.
extern crate self as ktstr;

// Global allocator for every binary linking this crate — the shipped
// bins (ktstr, cargo-ktstr, the jemalloc fixtures; all carry
// `required-features = ["cli-bins"]`) and, since `integration` pulls
// `cli-bins` (Cargo.toml), every `#[ktstr_test]` integration-test binary
// (the framework packs it as the guest `/init` — vmm::initramfs
// strip+packs the test binary as rdinit). The guest-`/init` use is
// incidental, NOT a fix: a memory-constrained guest `/init` is kept alive
// by `vm.overcommit_memory=1` on the guest cmdline
// (vmm::setup::base_guest_cmdline), not by this allocator — the System
// allocator boots the guest fine under that sysctl (verified by running
// the shell-lifecycle suite with jemalloc disabled: 12/12 pass). Gated on
// `cli-bins`, which provides `tikv-jemallocator`; lean
// `default-features = false` library consumers (which never boot guests)
// keep the System allocator. The bins/probe that previously declared
// their own jemalloc now inherit this one; `jemalloc_alloc_worker`
// keeps its own because it does not link this crate (pure `#[path]`).
#[cfg(feature = "cli-bins")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Defense-in-depth for the e2e jemalloc-fixture invariant: `integration`
// pulls `cli-bins` (Cargo.toml), so the jemalloc introspection fixtures
// (`ktstr-jemalloc-probe` / `-alloc-worker`, gated on cli-bins +
// integration) build when their e2es do — those e2es
// `env!(CARGO_BIN_EXE_…)` the fixture bins. If the implication is ever
// severed the fixtures don't build and their e2es fail to compile. Fail
// the build loudly with the cause rather than ship the foot-gun.
#[cfg(all(feature = "integration", not(feature = "cli-bins")))]
compile_error!(
    "feature `integration` requires `cli-bins`: the jemalloc \
     introspection fixtures (ktstr-jemalloc-probe / -alloc-worker) are \
     gated on cli-bins and their e2es env!(CARGO_BIN_EXE_…) them. Build \
     with `--features integration` (which pulls cli-bins) or add cli-bins."
);

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

/// Forward the guest's `/dev/kmsg` ring buffer to the host over the
/// bulk port, so a host-side post_vm callback can read it via
/// `VmResult::guest_kmsg`. Mirrors `read_kmsg`: that reads the ring
/// buffer in the guest process; this forwards the bytes (typically
/// `read_kmsg().as_bytes()`) to the host. The scx_exit
/// SCX_EXIT_ERROR_STALL printk lands in /dev/kmsg but is suppressed
/// from the COM1 console at the default `loglevel=0`, so it never
/// reaches `VmResult::stderr`; this forward is the only host-visible
/// path to it.
pub fn send_kmsg(buf: &[u8]) {
    crate::vmm::guest_comms::send_dmesg(buf);
}

/// The host watchdog-override readback, surfaced via
/// `VmResult::watchdog_observation`. Hidden from rustdoc at its
/// definition, so this re-export adds no public doc surface.
pub use crate::monitor::WatchdogObservation;

pub mod assert;
pub(crate) mod budget;
pub(crate) mod cargo_test_mode;
pub mod cli;
pub mod cpu_util;
pub mod ctprof;
pub mod ctprof_compare;
pub mod distro;
pub(crate) mod elf_strip;
#[cfg(feature = "export")]
pub mod export;
pub mod fetch;
pub mod fun;
pub mod host_context;
pub mod host_heap;
pub(crate) mod host_thread_probe;
pub mod kernel_path;
// build_helpers.rs is `include!`d into build.rs at build-script
// compile time. Mounting it as a `#[cfg(test)]` mod here lets its
// unit tests run under `cargo nextest` / `cargo ktstr test`
// without exposing the helper on the public API (the `mod` is
// test-only; `cargo build` doesn't compile it as a lib module).
#[cfg(test)]
mod build_helpers;
pub mod metric_types;
pub(crate) mod monitor;
pub(crate) mod probe;
pub(crate) mod report;
pub mod scenario;
pub(crate) mod stats;
pub(crate) mod taskstats;
pub mod test_support;
// `pub` (not `pub(crate)`): `assert::build_phase_buckets_with_stimulus`
// takes `timeline::StimulusEvent` in its public signature, and result
// analyzers (post_vm callbacks folding `VmResult::stimulus_timeline()`
// through that fn) need to name the type.
pub mod timeline;
pub mod topology;

/// Public surface for the live-host introspection pipeline.
///
/// Re-exports from the otherwise-internal `monitor` module so the
/// live-host capture binary, integration tests, and downstream
/// consumers can invoke the bpf()-syscall data path, kernel
/// auto-discovery, kallsyms parser, and dmesg-scx parser without the
/// `monitor` module's frozen-VM internals leaking into the public API.
///
/// This module is the entry point for binaries and tests that
/// consume the live-host capture pipeline.
pub mod live_host {
    pub use crate::monitor::bpf_map::{
        BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_ARRAY,
        BpfMapAccessor, BpfMapInfo,
    };
    pub use crate::monitor::bpf_syscall::BpfSyscallAccessor;
    pub use crate::monitor::dmesg_scx::{
        ScxExitEvent, ScxExitKind, StackSymbol, extract_stack_symbols, parse_kmsg_window,
    };
    pub use crate::monitor::live_host_kernel::{KallsymsTable, LiveHostKernelEnv, uname_release};
    pub use crate::monitor::timeline::{
        DEFAULT_SNAPSHOT_RING_DEPTH, IncrementalCapture, IncrementalSnapshot, SnapshotRing,
        TimelineCapture, TimelineEvent, TimelineEventRaw, parse_timeline_buf,
        parse_timeline_record, tl_evt,
    };
}

#[cfg(feature = "remote-cache")]
pub mod remote_cache;

/// No-op stub for the `remote_cache` module when the `remote-cache`
/// feature is disabled. Exposes the same three entry points consumed
/// elsewhere in the crate — `is_enabled()`, `remote_lookup()`,
/// `remote_store()` — so call sites in `cli/resolve.rs` and
/// `cli/kernel_build/build.rs` compile and behave correctly without
/// any `#[cfg]` gating at the call site: `is_enabled()` always
/// returns false, the lookup/store entry points are unreachable in
/// practice, and the stubs satisfy the trait surface so type-checking
/// stays uniform across feature configurations.
#[cfg(not(feature = "remote-cache"))]
pub mod remote_cache {
    use crate::cache::{CacheDir, CacheEntry};

    pub fn is_enabled() -> bool {
        false
    }

    pub fn remote_lookup(
        _cache: &CacheDir,
        _cache_key: &str,
        _cli_label: &str,
    ) -> Option<CacheEntry> {
        None
    }

    pub fn remote_store(_entry: &CacheEntry, _cli_label: &str) {}
}
pub mod gauntlet;
pub(crate) mod reflink;
pub(crate) mod sync;
#[cfg(any(feature = "export", feature = "remote-cache"))]
pub(crate) mod tar_util;
pub mod verifier;
pub(crate) mod vmm;
pub mod worker_ready;

#[cfg(feature = "wprof")]
pub use vmm::wprof::{WPROF_MIN_MEMORY_MIB, apply_wprof_memory_floor};

/// Re-export of [`test_support::runtime::bypass_llc_locks_active`]
/// so the bin/cargo_ktstr + bin/ktstr CLI surfaces (separate
/// crates linking against this lib) can apply the canonical
/// empty-string-aware bypass check at their parse-time
/// `--cpu-cap` conflict guards. Mirrors the in-crate readers at
/// vmm/builder.rs + cli/kernel_build/build.rs.
pub use test_support::runtime::bypass_llc_locks_active;

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
/// this path — the `#[ktstr_test]` macro registers via
/// `::ktstr::distributed_slice` / `::ktstr::linkme`; the `ctor` /
/// `serde_json` re-exports here serve downstream test-author code
/// (pre-`main()` setup and sidecar parsing) and ktstr's own
/// declarative-ctor sites, and the set may change without notice.
/// (`linkme` lives at the public crate root —
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
/// [`distributed_slice`] registrations
/// require — without having to add `linkme` as a direct Cargo
/// dependency. See [`distributed_slice`]
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
    // The `#[derive(Claim)]`-generated `<Type>Claim` extension traits carry the
    // typed `claim_<field>(&mut verdict)` accessors. Every derive(Claim) stats
    // type is preluded, so its Claim trait is preluded alongside it and
    // `use ktstr::prelude::*` makes the accessors callable without a per-type
    // trait import: the four assert-module traits below, plus `WorkerReportClaim`
    // in the workload export block.
    pub use crate::assert::{
        AbsoluteThresholds, Assert, AssertDetail, AssertResult, COMPARATOR_VOCABULARY, CgroupStats,
        CgroupStatsClaim, ClaimBuilder, DetailKind, EachClaim, FracPair, InfoNote,
        MAX_RECORDED_PASSES, NoteValue, Outcome, OutcomeRef, PASSES_TRUNCATION_SENTINEL_COMPARATOR,
        PASSES_TRUNCATION_SENTINEL_NAME, PassDetail, PhaseBucket, PhaseBucketClaim,
        PhaseCgroupStats, PhaseCgroupStatsClaim, PhaseMapExt, ScenarioStats, ScenarioStatsClaim,
        SeqClaim, SeriesField, SetClaim, Verdict, assert_scx_events_clean, assert_thresholds,
        build_phase_buckets_with_stimulus,
    };
    // Per-phase-metric building blocks for `post_vm` callbacks doing
    // custom per-phase assertions: `StimulusEvent` is the timeline event
    // type `build_phase_buckets_with_stimulus` consumes, and
    // `VmResult::stimulus_timeline()` returns a `Vec<StimulusEvent>`
    // (step frames + scenario-end terminal) ready to fold through it.
    // The non-stimulus sibling `assert::build_phase_buckets` is
    // INTENTIONALLY not preluded: it groups by the raw bridge-stamped
    // step_index, which can collapse under a deferred-fire burst (see
    // `SampleSeries::by_stamped_phase`); the stimulus-aware variant above
    // is the collapse-immune common path, so the prelude surfaces only
    // it. The plain variant remains reachable by full path for the rare
    // no-stimulus-timeline case.
    pub use crate::cgroup::CgroupManager;
    pub use crate::claim;
    pub use crate::claim_present;
    pub use crate::declare_scheduler;
    pub use crate::distributed_slice;
    pub use crate::host_context::HostContext;
    pub use crate::host_heap::HostHeapState;
    pub use crate::ktstr_test;
    pub use crate::scenario::backdrop::Backdrop;
    pub use crate::scenario::ops::{
        CgroupDef, CpusetSpec, HoldSpec, IrqSelector, KernelTarget, KernelValue, KernelValueWidth,
        Op, Setup, SpawnPlacement, Step, execute_defs, execute_scenario, execute_scenario_with,
        execute_steps, execute_steps_with,
    };
    pub use crate::scenario::payload_run::{PayloadHandle, PayloadRun};
    pub use crate::scenario::scenarios;
    pub use crate::test_support::post_vm_skip;
    pub use crate::timeline::StimulusEvent;
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
        DegradedFailureDumpReport, DualFailureDumpReport, EventCounterSample,
        FailureDumpArrayEntry, FailureDumpEntry, FailureDumpFdArray, FailureDumpMap,
        FailureDumpPercpuEntry, FailureDumpPercpuHashEntry, FailureDumpReport,
        FailureDumpReportAny, FailureDumpRingbuf, FailureDumpStackTrace,
        FailureDumpStackTraceEntry, PerCpuTimeStats, PerNodeNumaStats, ProbeBssCounters,
        REASON_DEGRADED_RENDEZVOUS_TIMEOUT, SCHEMA_DEGRADED, SCHEMA_DUAL, SCHEMA_SINGLE,
        SNAPSHOT_TAG_EARLY_DEGRADED, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED,
        SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED, SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
    };
    pub use crate::monitor::scx_walker::{DsqState, RqScxState, ScxSchedState};
    pub use crate::monitor::task_enrichment::TaskEnrichment;
    pub use crate::scenario::sample::{
        BpfMapCpuProjector, BpfMapProjector, Sample, SampleSeries, StatsPathProjector, StatsValue,
    };
    pub use crate::scenario::snapshot::{
        BridgeGuard, CaptureCallback, CgroupProcsSnapshot, JsonField, MAX_WATCH_SNAPSHOTS,
        Snapshot, SnapshotBridge, SnapshotEntry, SnapshotError, SnapshotField, SnapshotMap,
        SnapshotResult, WatchRegisterCallback, pickers, stats_path,
    };
    pub use crate::scenario::{CgroupGroup, Ctx, collect_all, spawn_diverse};
    // `Payload` in this group is the struct on which
    // `#[derive(Payload)]` is applied; it occupies the type
    // namespace, distinct from the derive macro re-exported above.
    pub use crate::test_support::{
        BpfMapAgg, BpfMapWrite, CgroupPath, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_PASS,
        KTSTR_SCHEDULERS, KTSTR_TESTS, KtstrTestEntry, MemSideCache, Metric, MetricCheck,
        MetricHint, MetricStream, NumaDistance, NumaNode, OutputFormat, Payload, PayloadKind,
        PayloadMetrics, PerfDeltaAssertion, Polarity, Scheduler, SchedulerSpec, SidecarResult,
        Sysctl, Topology, TopologyConstraints, WatchBpfMap, extract_metrics, find_scheduler,
        find_test, sidecar_dir,
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
    // Surfaced for `post_vm` callbacks that drain the snapshot
    // bridge's per-tag kernel-op reply log via
    // `VmResult::snapshot_bridge::drain_kernel_ops`: the returned
    // `Vec<(String, KernelOpReplyPayload)>` carries `read_values`
    // of `KernelOpValue` variants the callback pattern-matches to
    // assert on a read-back u32 / u64 / Bytes payload from
    // `Op::ReadKernel{Hot,Cold}`. Mirrors the existing exports for
    // `VirtioBlkCountersSnapshot` etc. — observability types the
    // post_vm contract requires in scope.
    pub use crate::scenario::host_stuck::{StuckDiagnostic, StuckReport};
    pub use crate::vmm::wire::{KernelOpReplyPayload, KernelOpValue};
    pub use crate::workload::{
        AffinityIntent, AluWidth, CloneMode, CustomCfg, CustomFn, FutexLockMode, MemPolicy,
        Migration, MpolFlags, ReapMode, ResolvedAffinity, SchbenchConfig, SchedClass, SchedPolicy,
        TaobenchConfig, TaobenchStats, WakeMechanism, WorkPhase, WorkSpec, WorkType,
        WorkTypeValidationError, WorkerCtx, WorkerReport, WorkerReportClaim, WorkloadConfig,
        WorkloadHandle,
    };
    // Surface `Phase` from the assert module (the scenario-step
    // bucket) so test authors can write `Phase::step(0)` /
    // `Phase::baseline()` without disambiguating against the
    // formerly-named workload variant. The workload's compound-
    // pattern enum is now `WorkPhase` (above) so `Phase` alone
    // unambiguously means the scenario-phase bucket type users
    // reach for in `field.value_at_phase(Phase::step(0))` style.
    pub use crate::assert::Phase;
    // Typed built-in metric ids (the discoverable, typo-proof catalog) + the
    // `MetricId` hybrid that ALSO accepts a scheduler-runtime string — both flow
    // through every metric accessor via `impl Into<MetricId>`. A misspelled
    // built-in is a compile error, not a silent `None`.
    pub use crate::stats::{BuiltinMetric, MetricId};
}

/// # KTSTR_* env-var empty-string contract
///
/// Default policy across `KTSTR_*` env vars: **empty string is
/// treated as unset** (falls back to the same default the var
/// would use if absent). This prevents the "stale shell export"
/// footgun where an operator's previous `KTSTR_FOO=...` export
/// gets cleared by the new shell (`KTSTR_FOO=`) but the empty
/// value is still observed by child processes — without the
/// empty-as-unset rule, child code would see "set" via
/// `env::var(...).is_ok()` and try to use the empty value as
/// data, producing confusing failures far from the export site.
///
/// Per-const docs flag deviations from this default explicitly:
/// presence-only markers (e.g. [`KTSTR_ORCHESTRATED_ENV`]) treat
/// empty as set per documented contract; value-typed vars (e.g.
/// path overrides like [`KTSTR_HOST_CGROUP_PARENT_ENV`]) follow
/// the empty-as-unset default and surface the fallback at the
/// resolver site.
///
/// New `KTSTR_*` env vars must pick a policy at the const-decl
/// doc and the reader site must honor it; mixed empty-treatment
/// within one var is a footgun.
///
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

/// Name of the environment variable cargo-ktstr sets to a
/// `dir=commit;dir=commit;...` map of each resolved SOURCE kernel's
/// short commit hash (with a `-dirty` suffix when the tree is dirty),
/// keyed by the same directory string exported in [`KTSTR_KERNEL_ENV`]
/// / [`KTSTR_KERNEL_LIST_ENV`].
///
/// cargo-ktstr probes each kernel's git HEAD ONCE in the orchestrator
/// and records it here. The sidecar writer reads this map and looks
/// itself up (by its own `KTSTR_KERNEL` dir) instead of re-running a
/// gix HEAD read + dirty-walk over the kernel tree in every per-test
/// nextest process. That walk is memoized per process but NOT across
/// processes, so without this map each of N test processes re-pays the
/// full dirty-walk (seconds on a large kernel checkout).
///
/// Optimization only: a missing entry, absent env, or decode failure
/// falls back to the in-process resolve-and-walk, which is always
/// correct. Kernels with no recoverable source tree (transient
/// Range/Git specs, or a Version/CacheKey cache miss) are absent from
/// the map, and the fallback yields the same `None` for them.
pub const KTSTR_KERNEL_COMMIT_ENV: &str = "KTSTR_KERNEL_COMMIT";

/// Name of the environment variable cargo-ktstr's perf-delta sets to the
/// PROJECT tree's short commit hash (with a `-dirty` suffix when the tree
/// is dirty) that the child's sidecars must record as their
/// `project_commit`. Unlike [`KTSTR_KERNEL_COMMIT_ENV`] this is a single
/// value, not a `dir=commit` map — a child has exactly one project tree.
///
/// perf-delta computes the A/B commit labels ONCE in the orchestrator
/// (`short_hash`) and both the pool FILTER and the two run children read
/// the same value: the filter partitions on it, and each child records it
/// verbatim (the sidecar writer's `detect_project_commit` returns this
/// value when the env is set), so the recorded `project_commit` and the
/// filter can never diverge on the `-dirty` suffix. It also lets the
/// baseline run, whose tree is a plain gix checkout with no `.git`, skip a
/// `gix::discover` that would otherwise resolve to the wrong repo (or none).
///
/// Override only: an absent or empty env falls back to the in-process
/// `gix::discover` + dirty-walk, which is correct for a normal (non
/// perf-delta) test run.
pub const KTSTR_PROJECT_COMMIT_ENV: &str = "KTSTR_PROJECT_COMMIT";

/// Name of the environment variable cargo-ktstr sets to signal
/// "this test process was launched by a cargo-ktstr orchestration
/// path, not raw `cargo nextest`". cargo-ktstr's `test` and
/// `verifier` subcommands set it to `"1"` before spawning the
/// nextest child; the value content does not matter, only the
/// presence — `std::env::var(KTSTR_ORCHESTRATED_ENV).is_ok()`.
///
/// Tests that boot real KVM VMs (`src/vmm/*` integration tests)
/// use this signal to skip when an operator runs the test binary
/// directly via `cargo nextest run --lib`. Raw nextest fans
/// 7000+ tests at full host parallelism, which starves the
/// per-VM resource budgets these tests depend on (KVM page
/// allocation, vCPU thread scheduling, freeze rendezvous timing).
/// Failure shape is `kill set by AP` + watchdog-deadline timeout
/// shortly after VM start. cargo-ktstr's orchestrator constrains
/// the VM-test concurrency so the budgets hold; raw nextest
/// doesn't, so the skip surfaces operator-error (wrong runner)
/// rather than dismissing a real bug.
///
/// `KTSTR_KERNEL_ENV` alone is not sufficient: an operator may
/// have it set in their shell from a prior cargo-ktstr session
/// and then invoke raw nextest. The dedicated orchestration
/// marker discriminates the two cases.
pub const KTSTR_ORCHESTRATED_ENV: &str = "KTSTR_ORCHESTRATED";

/// Name of the environment variable carrying the `cargo ktstr test`
/// SESSION EPOCH: nanoseconds since the Unix epoch, stamped ONCE by
/// the orchestrator (`cargo_ktstr::run_cargo`) before it spawns
/// nextest, inherited by every per-test child process.
///
/// `test_support::sidecar::pre_clear_run_dir_once` uses it as an
/// opaque per-invocation session token: nextest is process-per-test
/// and every test sharing one `{kernel}-{project_commit}` run
/// directory would otherwise have a later process's pre-clear delete
/// an earlier peer's freshly-written `{test}-{hash}.ktstr.json`
/// (silent stats loss). The first process to clear a dir records
/// this token in a `.ktstr_run_epoch` sentinel; a later peer whose
/// token matches skips its wipe, sparing the peers' sidecars. Unset
/// under raw `cargo nextest run` (no orchestrator) — pre-clear then
/// falls back to its per-process wipe-everything behavior.
pub const KTSTR_RUN_EPOCH_ENV: &str = "KTSTR_RUN_EPOCH";

/// Name of the environment variable that pins the sidecar runs-root
/// to an ABSOLUTE path, overriding the CWD-relative
/// `{CARGO_TARGET_DIR or "target"}/ktstr` default of
/// [`test_support::runs_root`].
///
/// The `cargo ktstr` orchestrator (`cargo-ktstr` main) stamps this
/// once at startup to the cargo target dir's `ktstr` subdir (resolved
/// via `cargo metadata`), so its post-run footer / `stats` / `replay`
/// reads AND the child test processes' sidecar writes resolve the
/// SAME directory regardless of CWD. Without it, in a Cargo workspace
/// the test binaries (CWD = package dir, set by nextest) write to
/// `{package}/target/ktstr` while the orchestrator (CWD = invocation
/// dir) scans elsewhere, so the post-run footer finds nothing.
///
/// Set ONCE by the parent and inherited by every child test process —
/// children never re-run `cargo metadata` (it would be one subprocess
/// spawn per test process on the hot path). Unset under raw `cargo
/// nextest run` (no orchestrator): [`test_support::runs_root`] falls
/// back to its CWD-relative default, which is fine because raw nextest
/// has no footer to mismatch.
pub const KTSTR_RUNS_ROOT_ENV: &str = "KTSTR_RUNS_ROOT";

/// Name of the environment variable that overrides the default
/// host-mode cgroup parent (where `host_only` tests' workload
/// cgroups land). Empty / unset falls back to the canonical
/// default; a non-empty value must be rooted under
/// `/sys/fs/cgroup` and name a non-root subdirectory.
///
/// Single source of truth so the name is not spelled by hand at
/// each reader. Mirrors the sibling `KTSTR_KERNEL_ENV` /
/// `KTSTR_KERNEL_LIST_ENV` / `KTSTR_KERNEL_PARALLELISM_ENV` /
/// `KTSTR_VERIFIER_RAW_ENV` / `KTSTR_ORCHESTRATED_ENV`
/// constant-defined naming convention; a single grep across
/// `KTSTR_*_ENV` consts gives the operator the complete env-var
/// inventory.
///
/// Read by `crate::test_support::dispatch::resolve_host_cgroup_parent`.
pub const KTSTR_HOST_CGROUP_PARENT_ENV: &str = "KTSTR_HOST_CGROUP_PARENT";

/// Name of the environment variable that overrides the cgroup-fs
/// root [`crate::cgroup::CgroupManager::setup`] walks down from when
/// enabling controllers in every ancestor's `cgroup.subtree_control`.
/// Empty / unset falls back to `/sys/fs/cgroup` (the canonical
/// cgroup-v2 mount). Non-empty value must be a prefix of
/// [`KTSTR_HOST_CGROUP_PARENT_ENV`]'s configured parent so the walk
/// stays inside the directory the operator owns; values that do not
/// satisfy the prefix invariant are rejected upfront by
/// [`crate::cgroup::CgroupManager::with_walk_root`].
///
/// Exists for cgroup-v2 user delegation (Mode B/C: systemd
/// `Delegate=yes`, container `nsdelegate`): the operator owns
/// `subtree_control` writes only inside the delegated subtree, and a
/// blind walk from `/sys/fs/cgroup` down would EACCES at the
/// `user.slice` / container-root boundary. Setting the walk root to
/// the delegation boundary makes the setup-time controller-enable
/// walk stop there.
///
/// # Empty-string contract
///
/// Empty string is observationally identical to unset: both fall
/// back to `/sys/fs/cgroup` (the [`crate::cgroup::CgroupManager::new`]
/// default). This mirrors the sibling env vars; a shell that exports
/// `KTSTR_CGROUP_WALK_ROOT=` (without a value) explicitly opts back
/// into the default rather than passing an empty path down to
/// [`crate::cgroup::CgroupManager::with_walk_root`] (which would
/// always fail the prefix invariant).
///
/// Single source of truth so the name is not spelled by hand at each
/// reader. Mirrors the sibling [`KTSTR_HOST_CGROUP_PARENT_ENV`]
/// constant-defined naming convention; a single grep across
/// `KTSTR_*_ENV` consts gives the operator the complete env-var
/// inventory.
///
/// Read by `crate::test_support::dispatch::resolve_host_cgroup_parent`.
pub const KTSTR_CGROUP_WALK_ROOT_ENV: &str = "KTSTR_CGROUP_WALK_ROOT";

/// Name of the environment variable that overrides the poll cadence
/// (in milliseconds) of the host-mode stuck monitor in
/// [`crate::scenario::host_stuck`].
///
/// The monitor runs in a background thread and samples
/// `/proc/<pid>/sched` every N ms for every worker pid the scenario
/// spawned; W consecutive samples with `Δnr_switches == 0` AND
/// `Δsum_exec_runtime == 0` flip the stuck predicate. Default
/// cadence is 500 ms × W=4 = 2 s detection latency.
///
/// # **Empty = unset** (also: `0` / unparseable)
///
/// Empty / unset / `0` / unparseable falls back to the default
/// ([`crate::scenario::host_stuck::DEFAULT_POLL_INTERVAL_MS`]).
/// Mirrors the empty-as-unset contract documented on the sibling
/// `KTSTR_*_ENV` constants so a shell `KTSTR_STUCK_POLL_MS=` quirk
/// silently degrades to default behavior rather than poisoning the
/// poller with a zero interval (which would either busy-loop or be
/// no-op-rejected).
///
/// Read once at `crate::scenario::host_stuck::spawn_monitor` when the
/// scenario engine spawns the monitor; mid-scenario env mutations
/// are NOT observed by the running thread.
///
/// Single source of truth so the name is not spelled by hand at
/// each reader. Mirrors the sibling [`KTSTR_HOST_CGROUP_PARENT_ENV`]
/// constant-defined naming convention; a single grep across
/// `KTSTR_*_ENV` consts gives the operator the complete env-var
/// inventory.
pub const KTSTR_STUCK_POLL_MS_ENV: &str = "KTSTR_STUCK_POLL_MS";

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

/// Name of the environment variable carrying the directory that each
/// `cargo ktstr verifier` cell writes its per-cell PASS/FAIL record to.
/// The `cargo ktstr verifier` dispatcher creates the dir, exports this
/// var (inherited by the spawned `cargo nextest run` and thus by every
/// cell process), and after nextest returns reads the records back to
/// render the per-(topology × scheduler) summary grid. Unset
/// when a verifier cell runs outside the dispatcher (a hand-driven
/// `--exact verifier/...`): the cell then simply skips the record write.
/// Single source of truth so the name is not spelled by hand at the
/// writer (cell) and reader (dispatcher) ends.
pub const KTSTR_VERIFIER_RESULT_DIR_ENV: &str = "KTSTR_VERIFIER_RESULT_DIR";

/// Name of the environment variable carrying the operator's
/// `cargo ktstr verifier --scheduler <NAME>` filter. Set by the
/// dispatcher in `src/bin/cargo_ktstr/verifier.rs`; read by
/// `crate::test_support::dispatch`'s verifier cell emission, which
/// skips every declared scheduler whose `name` does not equal the value
/// so the sweep runs one scheduler across topologies instead of the
/// full declared-scheduler matrix. Unset, every declared scheduler is
/// swept. Single source of truth so the writer (dispatcher) and reader
/// (emission) do not spell the name by hand.
pub const KTSTR_VERIFIER_SCHEDULER_ENV: &str = "KTSTR_VERIFIER_SCHEDULER";

/// Name of the environment variable that forces ktstr to skip the
/// `perf_event_open` access check + the
/// `perf_event_paranoid`-relaxation gate. Read at scenario-engine
/// startup ([`crate::test_support::runtime`]) and by the
/// `cargo ktstr shell` / verifier dispatch sites that disable
/// perf collection when the operator passes `--no-perf-mode`.
///
/// **Empty = unset** per the default contract — empty value is
/// treated as not set. The canonical reader at
/// `crate::test_support::runtime::no_perf_mode_active` uses
/// `.map(|v| !v.is_empty()).unwrap_or(false)` after a prior
/// regression where CI shells exporting `KTSTR_NO_PERF_MODE=`
/// silently disabled perf mode for every `performance_mode` test.
/// Any non-empty value (`"1"`, `"yes"`, `"0"`, `"true"`) enables
/// no-perf-mode. All readers (shell-mode VM builder in
/// `lib.rs`, verifier dispatch in `verifier.rs`, dispatch
/// gauntlet + eval entry in `test_support/dispatch.rs` and
/// `test_support/eval/mod.rs`)
/// route through the canonical helper so the empty-string
/// contract holds uniformly.
pub const KTSTR_NO_PERF_MODE_ENV: &str = "KTSTR_NO_PERF_MODE";

/// Name of the environment variable that restricts a run to ONLY
/// `performance_mode` tests: when set to a non-empty value, every
/// test whose entry does not have `performance_mode` is skipped
/// (skip sidecar recorded, libtest sees pass) before any VM boot.
/// The mergebase perf-delta subcommand sets this so a regression run
/// measures only the tests configured for clean performance numbers;
/// an explicit nextest `-E` filter narrows further within the
/// perf-mode set.
///
/// **Empty = unset** per the default contract, matching
/// [`KTSTR_NO_PERF_MODE_ENV`]. The canonical reader
/// `test_support::runtime::perf_only_active` (pub(crate)) uses
/// `.map(|v| !v.is_empty()).unwrap_or(false)` so a stray
/// `KTSTR_PERF_ONLY=` pass-through does not silently skip every
/// non-perf test. Readers route through the helper (dispatch
/// gauntlet + named routes in `test_support/dispatch.rs` and the
/// eval entry in `test_support/eval/mod.rs`).
pub const KTSTR_PERF_ONLY_ENV: &str = "KTSTR_PERF_ONLY";

/// Name of the environment variable that enables the GitHub Actions
/// remote-cache backend in [`crate::remote_cache`]. Read at cache-
/// init time; value-typed — only the exact string `"1"` enables;
/// unset / empty / any other value (including `"true"`, `"yes"`,
/// `"0"`, `"false"`) is disabled. Set explicitly by GitHub Actions
/// workflows when the runner has cache-API credentials; absent in
/// dev environments where local-only caching is the right default.
pub const KTSTR_GHA_CACHE_ENV: &str = "KTSTR_GHA_CACHE";

/// Name of the environment variable that signals ktstr is running
/// in "cargo test" mode (raw test binary launched by cargo's test
/// harness, no orchestrator). Distinct from
/// [`KTSTR_ORCHESTRATED_ENV`] which marks cargo-ktstr orchestration;
/// `KTSTR_CARGO_TEST_MODE` is for narrower cases like in-process
/// VMM tests that adapt their resource budgets when run via
/// `cargo test` / `cargo nextest`. Read via
/// `crate::cargo_test_mode::cargo_test_mode_active`: treats
/// unset and empty as disabled; ANY non-empty value enables —
/// no trim, no special-case strings (`"0"` and `"false"` ENABLE
/// because they're non-empty).
pub const KTSTR_CARGO_TEST_MODE_ENV: &str = "KTSTR_CARGO_TEST_MODE";

/// Name of the environment variable that overrides ktstr's cache
/// root directory (kernel-build cache, btf-anchor cache, blob
/// cache, etc.). Empty / unset falls back to the per-user default
/// (typically `$HOME/.cache/ktstr`). Heavy test usage —
/// `crate::test_support::test_helpers::IsolatedCacheDir` sets it
/// to a temp dir per-test so cache reads don't leak host state
/// into the test, and post-test the original value is restored
/// via `crate::test_support::test_helpers::EnvVarGuard`.
pub const KTSTR_CACHE_DIR_ENV: &str = "KTSTR_CACHE_DIR";

/// Name of the environment variable that overrides ktstr's flock
/// directory for inter-process resource locking (cpuset / LLC
/// reservation locks). Empty / unset falls back to the hardcoded
/// `/tmp` default at `crate::cache::resolve::resolve_lock_dir`
/// (the literal string, not `std::env::temp_dir()` / `TMPDIR`
/// resolution — historical default kept for stability). Used by
/// tests + CI environments that need isolated lock-dirs.
pub const KTSTR_LOCK_DIR_ENV: &str = "KTSTR_LOCK_DIR";

/// Name of the environment variable that triggers verbose logging
/// in the VMM setup phase. Strict `v == "1"` semantics (only the
/// literal `"1"` enables; unset / empty / any other value —
/// including `"true"`, `"yes"`, `"0"` — is disabled). Read in
/// `crate::vmm::setup` at the two cmdline-assembly sites (one per
/// arch: x86_64 and aarch64); both readers identical.
pub const KTSTR_VERBOSE_ENV: &str = "KTSTR_VERBOSE";

/// Name of the environment variable that bypasses LLC resource
/// locks at scenario setup (test_support::dispatch / cargo-ktstr
/// shell). Set by the `--bypass-llc-locks` CLI flag.
///
/// Reader sites use the canonical
/// [`crate::bypass_llc_locks_active`] helper (re-export of
/// [`test_support::runtime::bypass_llc_locks_active`]) which
/// applies the empty-string-as-unset contract uniformly across
/// all 7 callers (vmm/builder.rs, cli/kernel_build/build.rs ×2,
/// bin/ktstr.rs ×2, bin/cargo_ktstr/{kernel/mod, misc/shell}).
pub const KTSTR_BYPASS_LLC_LOCKS_ENV: &str = "KTSTR_BYPASS_LLC_LOCKS";

/// Name of the environment variable that caps the host CPU count
/// the scenario engine sees, for testing scaling logic without a
/// real CPU narrowing. Read at
/// `crate::vmm::host_topology` and set by the `--cpu-cap` CLI
/// flag in the bin entry points. Empty falls back to the
/// host's actual CPU count; non-empty numeric value caps the
/// observed count.
pub const KTSTR_CPU_CAP_ENV: &str = "KTSTR_CPU_CAP";

/// Name of the environment variable that bypasses the contention
/// guard at scenario setup. Strict `v == "1"` semantics (only
/// the literal `"1"` enables; everything else disables). Used
/// by tests that need to provoke contention scenarios without
/// the production guard kicking in.
pub const KTSTR_CONTENTION_BYPASS_ENV: &str = "KTSTR_CONTENTION_BYPASS";

/// Name of the environment variable cargo-ktstr's test
/// dispatcher sets to disable the skip-on-contention test
/// behavior. Presence check via `var_os(...).is_some()` — set
/// to "1" by `cargo ktstr test --no-skip-mode`, absent
/// otherwise.
///
/// **Deviates from the contract-default empty-as-unset rule**:
/// `var_os` does not distinguish empty from non-empty, so
/// `KTSTR_NO_SKIP_MODE=` (empty) ENABLES the bypass. Same
/// shape as [`KTSTR_GUEST_INIT_ENV`].
pub const KTSTR_NO_SKIP_MODE_ENV: &str = "KTSTR_NO_SKIP_MODE";

/// Name of the environment variable that overrides the per-test
/// budget in seconds for VM-boot dispatch. Empty / unset falls
/// back to the dispatcher's default. Parsed as f64 seconds at
/// `crate::test_support::dispatch` (accepts fractional values
/// like `2.5`); invalid or non-positive values surface a warn
/// and the default applies.
pub const KTSTR_BUDGET_SECS_ENV: &str = "KTSTR_BUDGET_SECS";

/// Name of the environment variable that overrides the sidecar
/// output directory (the per-test `*.ktstr.json` write target).
/// Empty / unset falls back to the per-test
/// `target/ktstr/<run-id>` location. Read at
/// `crate::test_support::sidecar` +
/// `crate::cli::stats_cmds::dispatch` (the stats reader).
pub const KTSTR_SIDECAR_DIR_ENV: &str = "KTSTR_SIDECAR_DIR";

/// Name of the environment variable that overrides the scheduler
/// binary path test_support::eval uses for in-process scheduler
/// dispatch. Read at `crate::test_support::eval`. This is the COARSE
/// (global) override: it applies to EVERY `SchedulerSpec::Discover`
/// scheduler regardless of name, so a test declaring multiple distinct
/// schedulers can't point them at different binaries through it — use
/// the per-name variant ([`per_name_scheduler_env`]) for that.
///
/// Resolution precedence: the per-name override is checked FIRST; this
/// global var is the fallback when no per-name var is set; if neither
/// resolves, the cascade falls through to the workspace build.
pub const KTSTR_SCHEDULER_ENV: &str = "KTSTR_SCHEDULER";

/// Per-scheduler-NAME override environment variable for a
/// `SchedulerSpec::Discover(name)` scheduler:
/// `KTSTR_SCHEDULER_BIN_<NAME>`, where `<NAME>` is the discover name
/// uppercased with every non-alphanumeric character replaced by `_`
/// (e.g. `scx_layered` -> `KTSTR_SCHEDULER_BIN_SCX_LAYERED`,
/// `scx-ktstr` -> `KTSTR_SCHEDULER_BIN_SCX_KTSTR`).
///
/// The `BIN` infix keeps the per-name namespace disjoint from the
/// `KTSTR_SCHEDULER_*` meta-variables ([`KTSTR_SCHEDULER_ENV`] = the
/// global override, [`KTSTR_SCHEDULER_PROFILE_ENV`] = the build
/// profile): without it a scheduler named `profile` would derive
/// `KTSTR_SCHEDULER_PROFILE` and shadow the build-profile selector.
/// `-` and `_` both map to `_` (env-var names can't contain `-`), so
/// `scx-foo` and `scx_foo` derive the same var — not a practical
/// ambiguity, as a scheduler is referred to by one canonical spelling
/// per run.
///
/// Checked BEFORE the global [`KTSTR_SCHEDULER_ENV`] in the Discover
/// resolution cascade, so a test that declares several distinct
/// Discover schedulers (one `entry.scheduler` plus staged schedulers)
/// can point each at its own pre-built binary. The global var remains
/// the coarse fallback for the common single-scheduler case. A set
/// per-name var whose path does not exist falls through to the global
/// var and then the build cascade (lenient, matching the global var's
/// own missing-path behavior).
pub fn per_name_scheduler_env(name: &str) -> String {
    let suffix: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("KTSTR_SCHEDULER_BIN_{suffix}")
}

/// Name of the environment variable that overrides the kernel
/// path the eval dispatch reads (orthogonal to
/// [`KTSTR_KERNEL_ENV`] which the main entry points use). Read
/// at `crate::test_support::eval::resolve_test_kernel`: a
/// set-but-empty `KTSTR_TEST_KERNEL=` surfaces a
/// `KTSTR_TEST_KERNEL not found:` hard error (typo-loud per
/// reader comment); ONLY the unset / `Err(NotPresent)` case
/// falls through to `crate::find_kernel()` (cache + sysroot
/// probes), which themselves fall through to a
/// `KernelUnavailable` error hinting at both
/// `KTSTR_TEST_KERNEL` and `KTSTR_KERNEL`.
pub const KTSTR_TEST_KERNEL_ENV: &str = "KTSTR_TEST_KERNEL";

/// Name of the environment variable cargo-ktstr's spawn-pipeline
/// sets to "1" inside the guest-side init binary so the binary
/// detects it's running as the guest init. Presence check via
/// `var_os(...).is_none()` at `crate::workload::spawn` —
/// absent in host-side dispatch.
///
/// **Deviates from the contract-default empty-as-unset rule**:
/// `var_os` does not distinguish empty from non-empty, so
/// `KTSTR_GUEST_INIT=` (empty) is observed as SET and disables
/// the orphan-detection fast-path. Same shape as
/// [`KTSTR_NO_SKIP_MODE_ENV`].
pub const KTSTR_GUEST_INIT_ENV: &str = "KTSTR_GUEST_INIT";

/// Name of the environment variable that points at a probe binary
/// for jemalloc-feature detection. Empty / unset leaves the probe
/// binary unwired (the [`crate::test_support::runtime`]
/// builder-wiring site calls `.jemalloc_probe_binary()` only on
/// set+non-empty — there is no `which`-based fallback). Tests
/// that need the probe set this var via `#[ctor]` before the
/// harness runs (see `tests/jemalloc_probe_tests.rs`).
pub const KTSTR_JEMALLOC_PROBE_BINARY_ENV: &str = "KTSTR_JEMALLOC_PROBE_BINARY";

/// Name of the environment variable that points at a worker
/// binary for jemalloc allocation-probe runs. Empty / unset
/// leaves the worker binary unwired — same shape as
/// [`KTSTR_JEMALLOC_PROBE_BINARY_ENV`]; the
/// [`crate::test_support::runtime`] builder-wiring site calls
/// `.jemalloc_alloc_worker_binary()` only on set+non-empty, no
/// `which`-based fallback. Set alongside the probe via `#[ctor]`
/// in `tests/jemalloc_probe_tests.rs`.
pub const KTSTR_JEMALLOC_ALLOC_WORKER_BINARY_ENV: &str = "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY";

/// Name of the environment variable that opts into per-assertion
/// PASS logging in the verdict pipeline. Read once per call at
/// [`crate::assert::claim::Verdict::new`] via the
/// `log_passes_default` helper in src/assert/claim.rs:
/// the reader is `!(v.is_empty() || v == "0")`, so empty and
/// the literal `"0"` disable; any other value (`"1"`, `"true"`,
/// `"yes"`, even `"false"` because it isn't `"0"`) enables.
/// Unset → disabled. Default-off keeps the PASS path
/// unallocated under normal runs.
pub const KTSTR_LOG_PASSES_ENV: &str = "KTSTR_LOG_PASSES";

/// Name of the environment variable that points at the busybox
/// blob on-disk. Exported by `cargo-ktstr`'s startup
/// `install_env` (see `bin/cargo_ktstr/blobs.rs`) which extracts
/// the embedded `BUSYBOX_BYTES` to a tempfile and sets this var
/// to the absolute path; read by
/// `crate::vmm::blobs::load_busybox_bytes`. Both unset and
/// set-but-empty surface a hard error, BUT the diagnostic
/// differs: unset hits the `Err(_)` arm and surfaces the
/// "blob is provided by `cargo-ktstr` at startup" install-env
/// hint; empty hits the `Ok("")` arm and falls through to a
/// generic `fs::read("")` ENOENT — less actionable. Operators
/// invoking `cargo ktstr <SUB>` see neither case; raw
/// `cargo nextest run` reliably triggers the unset diagnostic.
/// Busybox is load-bearing for shell-mode VMs + disk-template
/// builds.
pub const KTSTR_BUSYBOX_PATH_ENV: &str = "KTSTR_BUSYBOX_PATH";

#[cfg(feature = "wprof")]
pub const KTSTR_WPROF_PATH_ENV: &str = "KTSTR_WPROF_PATH";

/// Shared skip / error hint for call sites that cannot proceed
/// without a resolvable kernel. Phrased so the user sees the same
/// wording regardless of which layer surfaced the failure — tests,
/// CLI, monitor probes, and sidecar writers all point the operator
/// at the same remediation. Referenced by the non-VM-boot skip
/// paths in `cache.rs`, `probe/btf.rs`, `monitor/mod.rs`,
/// `test_support/eval/mod.rs`, and `test_support/mod.rs`.
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
    `6.14..=7.0`), git source (`git+URL#tag=NAME`, \
    `git+URL#branch=NAME`, or `git+URL#sha=<40-hex>`), absolute or \
    `~`-prefixed path, local kernel package (`*.rpm` or `*.deb`), \
    distro kernel (`fedora`/`fedora-44`/`f44`, `ubuntu`/`ubuntu-24.04`, \
    `amazonlinux`/`amazonlinux-2023`/`al2023`), or cache key. List \
    cached keys with `cargo ktstr kernel list`; build new ones with \
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
            // Multi-kernel specs (`A..B` ranges; git sources like
            // `git+URL#branch=main` are single-kernel but share this arm)
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
            // Local packages and distro kernels aren't wired into the
            // env-var resolver yet. Validate first so a malformed distro
            // release surfaces its specific diagnostic before the
            // generic "not yet supported" bail.
            id @ (KernelId::Package { .. } | KernelId::Distro { .. }) => {
                if let Err(e) = id.validate() {
                    anyhow::bail!("KTSTR_KERNEL={val}: {e}");
                }
                anyhow::bail!(
                    "KTSTR_KERNEL={val}: local kernel packages and distro \
                     kernels are not yet supported — set KTSTR_KERNEL to a \
                     single version, cache key, or path."
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

/// Name of the environment variable selecting the cargo build profile
/// for a `SchedulerSpec::Discover` scheduler built on demand by
/// [`build_and_find_binary`]. Holds a cargo profile NAME; unset / empty
/// means the RELEASE default (see `build_and_find_binary` — a debug
/// sched_ext scheduler is never the intended thing to test). Set by
/// `cargo ktstr <cmd> --profile <NAME>` (on `test` / `coverage` /
/// `verifier` / `perf-delta` / `replay`), or exported directly to pick a
/// non-default profile. This is INDEPENDENT of the harness `--release`
/// (`--cargo-profile release` to nextest), which selects the harness/test
/// binary's compile profile and does NOT touch this var — DECOUPLING the
/// scheduler-under-test's profile from the harness profile: the scheduler
/// runs optimized by default while the harness keeps its dev-profile
/// assertion thresholds and `catch_unwind` behavior unless its own build
/// profile is set separately.
pub const KTSTR_SCHEDULER_PROFILE_ENV: &str = "KTSTR_SCHEDULER_PROFILE";

/// Name of the presence-only opt-out env var that re-enables the
/// pre-built-binary fallback after a FAILED orchestrated scheduler
/// build. When set to a NON-EMPTY value, a failed `cargo build -p
/// <sched>` in the non-cargo-test `Discover` path falls back to a
/// sibling / `target/{debug,release}/` binary AS-IS instead of failing
/// the test. Default (unset / empty) REFUSES the stale fallback so a
/// build that fails for a new reason cannot silently validate the test
/// against an old scheduler. Empty-string rejection mirrors
/// `KTSTR_CARGO_TEST_MODE` (`cargo_test_mode_active`) — NOT the
/// presence-only [`KTSTR_ORCHESTRATED_ENV`], which activates on an empty
/// value — so a stray `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK=` cannot
/// re-enable the hazard.
pub const KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK_ENV: &str = "KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK";

/// The cargo build profile NAME for the scheduler-under-test:
/// [`KTSTR_SCHEDULER_PROFILE_ENV`] when set non-empty, else the
/// `"release"` default. Single source of the default so
/// [`build_and_find_binary`] (which builds it) and the `Discover`
/// fallback probe (which locates a pre-built one) never disagree on
/// which `target/<dir>/` the scheduler lands in.
pub fn scheduler_profile_name() -> String {
    resolve_scheduler_profile(std::env::var(KTSTR_SCHEDULER_PROFILE_ENV).ok())
}

/// Pure resolution of the scheduler build-profile NAME from the raw
/// [`KTSTR_SCHEDULER_PROFILE_ENV`] value: a non-empty value verbatim,
/// else the `"release"` default. An empty string is treated as UNSET —
/// so a stray `--profile ""` / `KTSTR_SCHEDULER_PROFILE=` can never make
/// [`build_and_find_binary`] run `cargo build --profile ""` (which would
/// resolve the artifact under an empty-named profile dir). Split from
/// [`scheduler_profile_name`] so the empty / unset / named cases are
/// unit-testable without mutating process env.
fn resolve_scheduler_profile(env_val: Option<String>) -> String {
    env_val
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "release".to_string())
}

#[cfg(test)]
mod scheduler_profile_tests {
    use super::resolve_scheduler_profile;

    /// The scheduler build-profile default: unset AND empty both fall
    /// back to `release` (empty is treated as unset so a stray
    /// `--profile ""` / `KTSTR_SCHEDULER_PROFILE=` cannot build under an
    /// empty-named profile dir); a non-empty name passes through verbatim
    /// (`dev`, `release`, or any custom `[profile.<name>]`). Pins the
    /// exact regressions [`resolve_scheduler_profile`] guards — dropping
    /// the `.filter(!is_empty)` (empty → empty name) or flipping the
    /// default flips a case.
    #[test]
    fn resolve_scheduler_profile_defaults_and_passthrough() {
        assert_eq!(
            resolve_scheduler_profile(None),
            "release",
            "unset -> release default"
        );
        assert_eq!(
            resolve_scheduler_profile(Some(String::new())),
            "release",
            "empty string is treated as unset -> release, never an empty profile name",
        );
        assert_eq!(
            resolve_scheduler_profile(Some("dev".into())),
            "dev",
            "named profile passes through verbatim"
        );
        assert_eq!(resolve_scheduler_profile(Some("release".into())), "release");
        assert_eq!(
            resolve_scheduler_profile(Some("custom".into())),
            "custom",
            "custom [profile.<name>] passes through unchanged",
        );
    }
}

/// Build a cargo binary package and return its output path.
///
/// Runs from the ktstr crate's manifest directory (which is also the
/// workspace root in this repo) so that workspace-level feature
/// unification (e.g. vendored libbpf-sys) is always in effect,
/// regardless of the calling process's working directory.
///
/// The scheduler-under-test is built with the RELEASE profile by
/// DEFAULT: a debug sched_ext scheduler is far slower and its BPF
/// verifier instruction counts differ, so it is never the intended thing
/// to test. [`KTSTR_SCHEDULER_PROFILE_ENV`] overrides the profile name
/// (set by `cargo ktstr <cmd> --profile <NAME>`); an unset / empty value
/// keeps the release default. The build passes `cargo build --profile
/// <name>` verbatim (`dev` = the default profile, `release` == `--release`,
/// any custom `[profile.<name>]`), so the returned artifact path resolves
/// under the matching `target/<dir>/`.
pub fn build_and_find_binary(package: &str) -> anyhow::Result<std::path::PathBuf> {
    let profile = scheduler_profile_name();
    let build_args: Vec<String> = vec![
        "build".into(),
        "-p".into(),
        package.into(),
        "--message-format=json".into(),
        "--profile".into(),
        profile,
    ];
    let output = std::process::Command::new("cargo")
        .args(&build_args)
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
/// `wprof_args`: requires the `wprof` cargo feature. When the
///   feature is enabled and `Some`, replaces `WprofConfig::args`
///   with the tokenised value before booting; `None` keeps the
///   defaults. Without the feature, this parameter is ignored
///   (a warning is emitted to stderr if `Some`).
/// `performance_mode`: forwarded to
///   `vmm::KtstrVmBuilder::performance_mode`; when `true`, the
///   builder pins vCPU threads, applies hugepages, NUMA mbinds, and
///   promotes vCPU threads to SCHED_FIFO (host-side optimizations).
/// `sched_enable_cmds` / `sched_disable_cmds`: forwarded to
///   `vmm::KtstrVmBuilder::sched_enable_cmds` /
///   `vmm::KtstrVmBuilder::sched_disable_cmds`. Non-empty when the
///   shell is reproducing a test whose scheduler is a
///   [`test_support::SchedulerSpec::KernelBuiltin`] variant —
///   guest init runs the enable cmds before drop-to-busybox and the
///   disable cmds on shell exit, so the operator gets the same
///   scheduler-loaded environment the test would see. Empty slices
///   mean "no scheduler-lifecycle commands."
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
    exec_timeout: std::time::Duration,
    disk: Option<vmm::disk_config::DiskConfig>,
    wprof_args: Option<&str>,
    performance_mode: bool,
    sched_enable_cmds: &[&str],
    sched_disable_cmds: &[&str],
) -> anyhow::Result<Option<i32>> {
    // Re-ignore SIGPIPE for the lifetime of the shell-mode VM. The
    // `cargo-ktstr` main installs SIG_DFL on SIGPIPE (so streaming
    // subcommands like `ktstr kernel list | head` exit cleanly rather
    // than panicking inside `print!`), but shell-mode owns a multi-
    // thread VM whose stdout/stderr writers (virtio-console TX
    // forwarder, COM1 console dump, banner / cleanup `eprintln!`s)
    // each issue blocking writes against the caller's stdio. When a
    // caller spawns `cargo ktstr shell --exec ...` via
    // `Command::output()` and reads the captured streams only after
    // the child exits, intermediate write contention can surface as
    // EPIPE on the next byte — under SIG_DFL that races the
    // `write_all().is_err()` BrokenPipe-handling branches in the
    // forwarder threads and kills the process before they can flip
    // `kill` and exit cleanly. SIG_IGN lets every writer observe the
    // EPIPE return value and propagate it through normal Rust error
    // handling, which is what shell mode already handles.
    //
    // SAFETY: `libc::signal` is async-signal-safe and only updates a
    // process-wide table entry; SIG_IGN is a well-known constant.
    // The change is intentionally permanent for the rest of the
    // process — the only caller that left SIG_DFL active was the
    // streaming-subcommand path, and that path never reaches
    // `run_shell`.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }

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

    let no_perf_mode = crate::test_support::runtime::no_perf_mode_active();
    use anyhow::Context;
    let busybox_bytes =
        vmm::blobs::load_busybox_bytes().context("load busybox blob for shell-mode VM")?;
    #[cfg(feature = "wprof")]
    let wprof_config = {
        // A `wprof`-feature build provisions the blob via cargo-ktstr's
        // install_env (KTSTR_WPROF_PATH) UNLESS it was built with
        // KTSTR_SKIP_WPROF_BUILD=1 (the documented escape hatch, which
        // leaves the blob empty so install_env exports no path). A failed
        // resolve here therefore means that hatch is set or install_env
        // was bypassed — fail loud either way rather than silently
        // shipping a wprof-less shell VM, which surfaces downstream as a
        // confusing "/bin/wprof: No such file" inside the guest.
        let mut c = vmm::wprof::WprofConfig::from_env()
            .context("resolve wprof for shell mode (the `wprof` feature is enabled)")?;
        if let Some(args_str) = wprof_args {
            c.args = args_str.split_whitespace().map(String::from).collect();
        }
        Some(c)
    };
    #[cfg(not(feature = "wprof"))]
    if wprof_args.is_some() {
        eprintln!(
            "ktstr: wprof_args ignored — ktstr was built without the \
             `wprof` cargo feature; /bin/wprof will not be available \
             in the guest"
        );
    }
    let mut builder = vmm::KtstrVm::builder()
        .kernel(&kernel)
        .init_binary(&payload)
        .topology(vmm::Topology::new(numa_nodes, llcs, cores, threads))
        .cmdline(&cmdline)
        .include_files(owned_includes)
        .busybox(Some(busybox_bytes))
        .dmesg(dmesg)
        .no_perf_mode(no_perf_mode)
        .performance_mode(performance_mode)
        .sched_enable_cmds(sched_enable_cmds)
        .sched_disable_cmds(sched_disable_cmds);

    #[cfg(feature = "wprof")]
    {
        builder = builder.wprof(wprof_config);
    }

    if let Some(cmd) = exec {
        // exec_timeout bounds the payload's wall-clock (panic-less hang
        // guard); only meaningful in exec mode, so set it alongside.
        builder = builder.exec_cmd(cmd).exec_timeout(exec_timeout);
    }

    if let Some(d) = disk {
        builder = builder.disk(d);
    }

    // Shell-mode initramfs (busybox, operator includes, and wprof
    // when the `wprof` feature is enabled) can exceed a test's
    // declared `memory_mib`. Treat the caller's value as a FLOOR;
    // the deferred path takes the max of it and the actual
    // initramfs size.
    builder = match memory_mib {
        Some(mib) => builder.memory_deferred_min(mib),
        None => builder.memory_deferred(),
    };

    let vm = builder.build()?;

    vm.run_interactive()
}

#[cfg(test)]
mod tests;
