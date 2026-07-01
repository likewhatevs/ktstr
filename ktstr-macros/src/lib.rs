use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[allow(dead_code)]
mod kernel_path;

mod claim;
mod common;
mod json;
mod ktstr_test;
mod payload;
mod scheduler;

/// Attribute macro that registers a function as a ktstr integration test.
///
/// The annotated function must have signature `fn(&ktstr::scenario::Ctx) ->
/// anyhow::Result<ktstr::assert::AssertResult>`. The macro:
///
/// 1. Renames the original function to `__ktstr_inner_{name}`.
/// 2. Registers it in the `KTSTR_TESTS` distributed slice via linkme.
/// 3. Emits a `#[test]` wrapper that boots a VM and runs the function
///    inside it.
///
/// Every attribute is optional. Most take a `key = value` form; the
/// sixteen boolean attributes (`auto_repro`, `expect_auto_repro`,
/// `not_starved`, `isolation`, `performance_mode`, `pci`, `no_perf_mode`,
/// `requires_smt`, `expect_err`, `survives_storm`, `allow_inconclusive`,
/// `fail_on_stall`, `host_only`, `ignore`, `kaslr`, `wprof`) also accept a
/// bare form as shorthand for `= true` — e.g.
/// `#[ktstr_test(host_only)]` is equivalent to
/// `#[ktstr_test(host_only = true)]`. Of the sixteen, `auto_repro`
/// and `kaslr` are the two whose default is `true`, so the bare form
/// is a no-op; `auto_repro = false` / `kaslr = false` are the only
/// way to disable each. The other fourteen default to `false`, so the
/// bare form is the meaningful shorthand.
///
/// The accepted attributes and their defaults are the fields of
/// `ktstr::test_support::KtstrTestEntry` (runtime metadata) and
/// `ktstr::assert::Assert` (checking thresholds). A few are
/// worth calling out because their names differ from the underlying
/// field or because they have nontrivial defaults:
///
///   - `llcs = N` — number of LLCs (default: inherited from
///     scheduler, or 1).
///   - `cores = N` (default: inherited from scheduler, or 2)
///   - `threads = N` (default: inherited from scheduler, or 1)
///   - `numa_nodes = N` (default: inherited from scheduler, or 1)
///   - `memory_mib = N` — per-test minimum memory in MiB (default:
///     2048). The framework picks `max(total_cpus * 64, 256,
///     memory_mib)` MiB at VM-launch time, so for tests with more
///     than 32 vCPUs the cpu-based floor dominates the macro
///     default. Below ~4 vCPUs the absolute 256-MiB floor wins if
///     `memory_mib` is also below it. Setting `memory_mib` above
///     the cpu-based floor is only meaningful when the test needs
///     more headroom than the per-cpu budget. The unit is binary
///     mebibytes; the conversion at VM-launch is `value << 20`
///     bytes, not decimal megabytes.
///   - `duration_s = N` — scenario run duration in seconds; maps
///     onto `KtstrTestEntry::duration`
///   - `watchdog_timeout_s = N` — watchdog fire threshold in
///     seconds; maps onto `KtstrTestEntry::watchdog_timeout`
///   - `cleanup_budget_ms = N` — sub-watchdog cap on host-side VM
///     teardown wall time; maps onto `KtstrTestEntry::cleanup_budget`
///     as `Duration::from_millis(N)`. Default: `None` (unenforced).
///   - `num_snapshots = N` — fire `N` periodic
///     `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` boundaries inside the workload's
///     10 %–90 % window, stored on the host
///     `SnapshotBridge` under `periodic_NNN`. `0` (default)
///     disables periodic capture entirely. Maps onto
///     `KtstrTestEntry::num_snapshots`; runtime
///     `KtstrTestEntry::validate` rejects values past the bridge
///     cap (`MAX_STORED_SNAPSHOTS`), `host_only = true`, and
///     duration / `N` settings that would land boundaries closer
///     than 100 ms apart.
///   - `scheduler = PATH` — path to a `const Scheduler` (typically
///     produced by `declare_scheduler!(...)`). Maps onto
///     `KtstrTestEntry::scheduler`, which is typed
///     `&'static Scheduler`. Default: `&Scheduler::EEVDF`, the
///     no-scx placeholder that runs under the kernel's default
///     scheduler.
///   - `payload = PATH` — path to a `const Payload` used as the
///     primary binary workload (must be `PayloadKind::Binary`;
///     runtime-enforced). Default: `None` (scheduler-only test).
///     Coexists with `scheduler = PATH` — the payload runs *under*
///     the selected scheduler.
///   - `workloads = [PATH, PATH, ...]` — additional `const Payload`
///     references composed with the primary via `Ctx::payload` in
///     the test body. Default: `&[]`. Must not contain the same
///     path as `payload` — reject at expansion time to catch the
///     common "fio as primary AND workload" slip.
///   - `auto_repro = bool` (default: `true`)
///   - `wprof = bool` (default: `false`; requires the `wprof`
///     cargo feature on ktstr) — attach `/bin/wprof` to the
///     test's VM(s) and ship the Perfetto trace to the host.
///   - `wprof_args = "..."` (requires the `wprof` cargo feature)
///     — override `WprofConfig::default_args`. Only meaningful
///     with `wprof = true`. Parsed as space-separated tokens.
///   - `host_only = bool` (default: `false`) — run the test function
///     on the host instead of inside a VM
///   - `no_perf_mode = bool` (default: `false`) — decouple the
///     virtual topology from host hardware. The VM is built with
///     the declared `numa_nodes` / `llcs` / `cores` / `threads`
///     even on smaller hosts; vCPU pinning, hugepages, NUMA mbind,
///     RT scheduling, and KVM exit suppression are skipped, and
///     gauntlet preset filtering relaxes host-topology checks
///     to the single "host has enough total CPUs" inequality.
///     Mutually exclusive with `performance_mode = true`. Maps onto
///     `KtstrTestEntry::no_perf_mode`.
///   - `post_vm = PATH` — host-side callback invoked after
///     `vm.run()` returns, with access to the full `VmResult`.
///     Use for assertions that need host-side state — e.g.
///     draining `VmResult.snapshot_bridge` after a snapshot
///     capture pipeline fires inside the guest. The function
///     must have signature
///     `fn(&ktstr::vmm::VmResult) -> anyhow::Result<()>`. PATH
///     accepts any Rust path-expression that resolves to a value
///     of that fn type — both free-function refs
///     (`my_post_vm_check`) AND UFCS method refs
///     (`VmResult::assert_wprof_pb_landed`) work via Rust's
///     function-item-to-fn-pointer coercion, so a method that
///     already has the right `&self -> Result<()>` shape can be
///     pointed at directly without wrapping in a one-line
///     delegating free fn.
///     SUPPRESSED on guest-reported fail — see
///     `post_vm_unconditional` for the always-runs sibling.
///     Default: `None` (no callback).
///   - `post_vm_unconditional = PATH` — host-side callback that
///     always runs after `vm.run()` returns, bypassing the
///     guest-fail suppression that gates `post_vm`. Same
///     signature as `post_vm` and PATH accepts the same form:
///     any Rust path-expression resolving to a value of that fn
///     type — both free-function refs AND UFCS method refs
///     (`VmResult::assert_wprof_pb_landed`-style) work via
///     function-item-to-fn-pointer coercion. Use when the
///     callback must observe host-side state regardless of
///     guest-side outcome (e.g. verifying a sidecar artifact
///     landed even when the guest reported a deliberate fail).
///     The callback is responsible for guarding against missing
///     state when the scheduler crashed before producing it —
///     the canonical guard is
///     `if !result.success { return Ok(()); }` at the top of the
///     callback body. Setting `post_vm_unconditional` does NOT
///     invert the test verdict — a guest-reported fail still
///     fails the test even if the unconditional callback returns
///     Ok. Both attributes may be set on the same entry (both
///     errors surface via `combine_post_vm_errs` when both
///     fire). Default: `None` (no callback).
///   - `disk = PATH` — path to a `const DiskConfig` attached to the
///     VM as a virtio-blk device at `/dev/vda`. Construct via
///     `DiskConfig::DEFAULT.with_name("data")` or similar const-fn
///     chain (the `with_name` builder takes `&'static str` so the
///     full expression is const-evaluable). Maps onto
///     `KtstrTestEntry::disk`. Default: `None` (no disk).
///     Mutually exclusive with `host_only = true` —
///     `host_only` skips the VM boot that owns the device lifecycle,
///     so a `disk` attached under `host_only` would never bind;
///     `KtstrTestEntry::validate` rejects the pairing at runtime.
///   - `networks = [PATH, ...]` — array of `const NetConfig` paths, one
///     virtio-net device per element (in-VMM loopback backend). On x86_64
///     each lands on its own virtio-pci function (PCI slots 1..=N, one INTx
///     GSI apiece); aarch64 takes a single virtio-MMIO NIC (build() errors
///     on more than one). Construct
///     each via `NetConfig::DEFAULT.mac(...)` or `NetConfig::DEFAULT`
///     (const-fn chain). Maps onto `KtstrTestEntry::networks`. Default: `[]`
///     (no NIC). Like `disk`, mutually exclusive with `host_only`.
///   - `config = EXPR` — inline scheduler config content, written
///     into the guest at the path declared by the scheduler's
///     `config_file_def`. `EXPR` is either a string literal or a
///     path to a `const &'static str` (e.g. `LAYERED_CONFIG`).
///     Maps onto `KtstrTestEntry::config_content`. Required when
///     the scheduler declares `config_file_def`; rejected when the
///     scheduler does not. The pairing is enforced at compile time
///     via a `const` assertion against `Scheduler::config_file_def`,
///     and again at runtime by `KtstrTestEntry::validate` so direct
///     programmatic-entry construction sees the same gate.
///   - `expect_scx_bpf_error_contains = EXPR` — literal-substring
///     matcher applied to the captured `scx_bpf_error` text in
///     reproducer mode. `EXPR` is either a string literal or a path
///     to a `const &'static str`. Maps onto
///     `Assert::expect_scx_bpf_error_contains`. Requires
///     `expect_err = true` (rejected at construction otherwise by
///     `KtstrTestEntry::validate`). Empty strings panic at
///     construction. When both `_contains` and `_matches` are set,
///     the evaluator ANDs them — every set matcher must hit.
///   - `expect_scx_bpf_error_matches = EXPR` — regex matcher with
///     the same accepted forms and gating as `_contains`. Maps onto
///     `Assert::expect_scx_bpf_error_matches`. Validated at
///     construction: empty patterns, invalid regex syntax, and any
///     pattern satisfying `is_match("")` all panic immediately. The
///     `is_match("")` predicate catches two no-op classes with one
///     check: patterns that match every position (e.g. `a?`, `.*`,
///     `(?:)`) trivially pass against any corpus, and patterns that
///     match only the empty string (e.g. `^$`) trivially fail
///     against any non-empty corpus — both are equally useless pins.
///     Bare `\b` slips the gate (no word characters in `""`); see
///     `Assert::expect_scx_bpf_error_matches` for the operator
///     direction.
///   - `survives_storm` — assert the scx scheduler SURVIVES the run
///     (does not die or get ejected during any hold); the positive
///     inverse of `expect_err`. Requires an active scheduler and is
///     mutually exclusive with both `expect_err` and `expect_auto_repro`
///     (rejected at macro-parse and by `KtstrTestEntry::validate`).
///     Enforced on scenarios driven through `execute_defs` /
///     `execute_steps` / `execute_scenario` (which run the scheduler
///     liveness probe); a survival violation surfaces as a failing exit
///     with a survival-specific explainer.
///   - `extra_include_files = ["PATH", "PATH", ...]` — host-side
///     file paths to bundle into the guest initramfs beyond what
///     the entry's `scheduler` / `payload` / `workloads` already
///     declare via their own `include_files`. Use this for
///     test-level dependencies that don't belong on a specific
///     Payload: auxiliary data files, per-test helper scripts,
///     fixtures. Each element must be a string literal (no
///     expressions). Maps onto
///     `KtstrTestEntry::extra_include_files` and is unioned with
///     the per-payload specs at `run_ktstr_test` time via
///     `KtstrTestEntry::all_include_files`. Default: `[]`.
///     Path resolution: bare names (no `/`) search `PATH`; paths
///     containing `/` are absolute or relative to the test process
///     current directory; directories are walked recursively at
///     test-run time (rejected by `cargo ktstr export` since the
///     `.run` packager handles regular files only — recursive
///     directory packaging is a v2 enhancement); a missing file
///     fails loudly at setup with an actionable error naming the
///     missing path.
///
/// Duplicate keys: each attribute KEY may appear at most once per
/// `#[ktstr_test]` invocation; duplicate keys (whether the values
/// match or differ) fail at expansion rather than silently letting
/// the later value win. `#[ktstr_test(host_only = false,
/// host_only)]` and `#[ktstr_test(llcs = 4, llcs = 8)]` both fail.
/// The bare form (`host_only`) and explicit form (`host_only =
/// true`) of the same attribute collide as well — they refer to
/// the same slot. List values like `workloads = [FIO, FIO]` are
/// NOT affected by this rule; the duplicate check is on attribute
/// keys, not on values within an array. `payload = ...` and
/// `workloads = [..]` keep their tailored messages directing the
/// author to the right home for extras; `config = ...` and
/// `expect_scx_bpf_error_{contains,matches} = ...` likewise have
/// tailored wording; every other attribute uses a uniform
/// "duplicate attribute" diagnostic.
///
/// Path / list forms: `#[ktstr_test(crate::host_only)]` (a
/// multi-segment path, whether bare or as a key in
/// `crate::host_only = true`) is rejected with a targeted message
/// naming both valid forms with concrete examples — the macro only
/// accepts bare single-segment idents because routing dispatches on
/// the ident string against `BOOL_ATTR_NAMES` or the value-attr match
/// arms (enumerated in `VALUE_ATTR_NAMES`).
/// `#[ktstr_test(host_only(false))]` (parenthesised
/// arguments) is rejected with a separate targeted message naming
/// the attribute and the two valid forms (`= value` or bare); the
/// same diagnostic fires for `crate::host_only(false)` so the
/// operator sees one combined error rather than chasing two.
#[proc_macro_attribute]
pub fn ktstr_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    match ktstr_test::ktstr_test_impl(attr.into(), item.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Function-style macro that registers a `Scheduler` const.
///
/// # Syntax
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// declare_scheduler!(MITOSIS, {
///     name = "mitosis",
///     binary = "scx_mitosis",
///     cgroup_parent = "/ktstr",
///     sched_args = ["--exit-dump-len", "1048576"],
///     kernels = ["6.14", "7.0..=7.2"],
///     constraints = TopologyConstraints {
///         min_llcs: 1, max_llcs: Some(8), max_cpus: Some(64),
///         ..TopologyConstraints::DEFAULT
///     },
/// });
/// ```
///
/// # Generated items
///
/// Given `declare_scheduler!(MITOSIS, { ... })`:
///
/// - `pub static MITOSIS: ::ktstr::test_support::Scheduler` — the declared
///   scheduler value. No `_PAYLOAD` suffix; the const IS the
///   `Scheduler`.
/// - A hidden `static __KTSTR_SCHED_REG_MITOSIS: &'static Scheduler`
///   registered in `KTSTR_SCHEDULERS` (`ktstr::test_support::KTSTR_SCHEDULERS`)
///   via linkme so the verifier can discover the declaration by
///   spawning the test binary with `--ktstr-list-schedulers`.
///
/// # Visibility prefix
///
/// An optional Rust visibility prefix may precede the const name:
///
/// ```rust,ignore
/// declare_scheduler!(MY_SCHED, { ... });             // defaults to `pub`
/// declare_scheduler!(pub MY_SCHED, { ... });          // explicit `pub`
/// declare_scheduler!(pub(crate) MY_SCHED, { ... });   // crate-local
/// declare_scheduler!(pub(super) MY_SCHED, { ... });   // parent-module
/// declare_scheduler!(pub(in crate::test_support) MY_SCHED, { ... });
/// ```
///
/// Omitting the prefix defaults to `pub` — schedulers are normally
/// public so the verifier and other crates can reference them; an
/// explicit prefix is needed only when the declaration sits inside
/// a module that wants to narrow the exposed name. (Field content
/// shown above as `{ ... }` is elided; consult the Syntax example
/// for the required fields.) The hidden registry static (see
/// Generated items above) is always `static` (private) regardless
/// of the user-facing const's visibility — `linkme` gathers it via
/// link-section walking, not Rust name resolution, so the slice
/// mechanism works at every visibility level.
///
/// # Accepted fields
///
/// Exactly one scheduler-source must be declared: `binary`,
/// `binary_path`, or the `kernel_builtin_enable` + `kernel_builtin_disable`
/// pair. The three options select between the matching
/// `SchedulerSpec` variants. To run under the kernel default
/// instead, reference `ktstr::test_support::Scheduler::EEVDF`
/// directly rather than declaring a new scheduler.
///
/// | Field | Required | Description |
/// |---|---|---|
/// | `name = "..."` | yes | Scheduler name (sidecar / logs). |
/// | `binary = "..."` | one source | Binary name → `SchedulerSpec::Discover(...)`. Matched against `[[bin]]` names in `target/{debug,release}/`, the test binary's directory, or `KTSTR_SCHEDULER` env var. Often equal to the cargo package name but not required to be. |
/// | `binary_path = "/abs/path"` | one source | Absolute filesystem path → `SchedulerSpec::Path(...)`. The runtime does not auto-build this variant: the file must already exist at the path when the test runs. Use for prebuilt binaries that live outside the cargo discovery cascade. Macro-time validation rejects empty strings, relative paths, and `~`-prefixed paths (no compile-time tilde expansion); existence is the runtime's job. |
/// | `kernel_builtin_enable = [..]` + `kernel_builtin_disable = [..]` | one source | Two string-array literals that together select `SchedulerSpec::KernelBuiltin { enable: &[..], disable: &[..] }`. The framework writes the enable commands to the guest's `/sched_enable` and the disable commands to `/sched_disable` (see `src/vmm/initramfs.rs`), and the guest interpreter runs each entry once at scenario start / teardown. Both fields must be set together — setting only one is rejected. The interpreter (`src/vmm/rust_init/dump.rs`) accepts EXACTLY ONE shell-line shape: `echo VALUE > /path` (plus blank lines and `#` comments). Pipes, `>>`, `;`, variable expansion, and any other syntax silently no-ops at runtime, so the macro rejects entries that don't match `echo … > /…` at expand time. At least one of the two arrays must be non-empty: a pair that supplies neither enable nor disable commands is equivalent to the EEVDF baseline — reference `Scheduler::EEVDF` for that. Note: `cargo ktstr export` currently bails on KernelBuiltin schedulers (`src/export.rs`); declarations using this variant cannot be reproduced via the export-to-shar workflow until that limitation is lifted. |
/// | `topology = (numa, llcs, cores, threads)` | no | Default VM topology. Default: `(1, 1, 2, 1)` (from `Scheduler::named`). Validated at compile time: each value must be non-zero, and `llcs` must be a multiple of `numa`. |
/// | `cgroup_parent = "..."` | no | Cgroup parent path (must begin with `/`). |
/// | `sched_args = [..]` | no | Scheduler CLI args prepended before per-test `extra_sched_args`. |
/// | `sysctls = [Sysctl::new("k", "v"), ..]` | no | Guest sysctls. |
/// | `kargs = [..]` | no | Extra guest kernel cmdline args. |
/// | `kernels = ["6.14", "7.0..=7.2", ..]` | no | Kernel specs the verifier sweeps. Same parser as the `--kernel` CLI flag — accepts exact versions, ranges (`..` or `..=`, both inclusive), git refs (`git+URL#REF`), paths, and cache keys. Each entry is validated at macro-expand time via the same `KernelId::parse` + `validate` the verifier uses at runtime; empty entries, inverted ranges, and `..`-containing strings whose endpoints aren't version-shaped (e.g. `"abc..def"`) are rejected. |
/// | `constraints = TopologyConstraints { .. }` | no | Gauntlet preset constraints — maps directly onto `Scheduler::constraints`. Filters which gauntlet topology presets exercise this scheduler. When given as a struct literal, the macro additionally cross-checks each literal field against the effective topology (explicit `topology` field if present, otherwise the `(1, 1, 2, 1)` default from `Scheduler::named`) and rejects infeasible pairings; non-struct-literal forms (e.g. `OTHER::CONST_CONSTRAINTS`) skip that check. |
/// | `assert = Assert::NO_OVERRIDES.method().chain()` | no | Scheduler-wide assertion overrides — maps directly onto `Scheduler::assert`. Merged with `Assert::default_checks()` and the per-test `assert` at runtime (`default ← scheduler ← per-test`). Accepts any const-evaluable expression: a const path like `Assert::NO_OVERRIDES`, a const-fn call like `Assert::default_checks()`, or a chain of const-fn setters like `Assert::NO_OVERRIDES.check_not_starved().max_gap_ms(50)`. The macro accepts MethodCall chains and Path-rooted (type/module-prefixed) Calls — only bare single-segment lowercase Calls like `helper()` are rejected as non-const free-fn patterns; non-const methods on a Path receiver slip through and surface as a deep const-eval failure at the spread site. |
/// | `config_file = "..."` | no | Host-side config file path. |
/// | `config_file_def = ("--config {file}", "/include-files/cfg.json")` | no | Inline-config plumbing — maps directly onto `Scheduler::config_file_def`. 2-tuple of string literals: arg_template (CLI arg with `{file}` placeholder substituted at run time) and guest_path (absolute path where the framework writes the JSON inside the guest). Distinct from `config_file` (which references a pre-existing host file). The macro validates: tuple-arity = 2, both elements non-empty string literals, `{file}` placeholder present in arg_template, guest_path absolute. |
///
/// # Const naming rules
///
/// The first argument must be a SCREAMING_SNAKE_CASE identifier and
/// must NOT be one of the reserved built-in names (`EEVDF`,
/// `KERNEL_DEFAULT`). The macro emits a `compile_error!` if either rule
/// is violated.
#[proc_macro]
pub fn declare_scheduler(input: TokenStream) -> TokenStream {
    match scheduler::declare_scheduler_inner(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Derive macro that generates a `Payload` const from an annotated
/// struct for a userspace binary workload (stress-ng, fio, and
/// similar tools test authors compose under a scheduler).
///
/// # Required struct-level attributes (`#[payload(...)]`)
///
/// - `binary = "..."` — the binary name resolved by the guest's
///   include-files infrastructure (required). Becomes
///   `PayloadKind::Binary(name)` (`ktstr::test_support::PayloadKind::Binary`),
///   and is also auto-prepended to the emitted `include_files` slice
///   so the binary is packaged into the initramfs without needing a
///   separate `#[include_files("...")]` entry. Extra auxiliary files
///   (helpers, configs, fixtures) still go on `#[include_files(...)]`.
///
/// # Optional struct-level attributes
///
/// - `name = "..."` — short name used in logs and sidecar records.
///   Defaults to the binary name.
/// - `output = Json | ExitCode | LlmExtract("hint")` — how the
///   framework extracts metrics from the payload's stdout. The
///   variant names match the `OutputFormat` enum and the `Polarity`
///   kwarg grammar. Defaults to `ExitCode`. The `LlmExtract` form
///   accepts an optional string literal focus hint appended to the
///   default LLM prompt; bare `LlmExtract` with no parenthesized
///   argument is a shorthand for `LlmExtract()` (no hint).
///
/// # Optional outer attributes
///
/// - `#[default_args("--a", "--b", ...)]` — variadic string
///   literals appended to the binary's argv when the payload runs.
///   May repeat across multiple `#[default_args(...)]` attrs; entries
///   accumulate in source order.
/// - `#[default_check(...)]` — one `MetricCheck` (`ktstr::test_support::MetricCheck`)
///   construction expression (e.g. `min("iops", 1000.0)`,
///   `exit_code_eq(0)`). May repeat; entries accumulate in source
///   order. Both `min(...)` and `MetricCheck::min(...)` are accepted: the
///   macro prepends `::ktstr::test_support::MetricCheck::` when the
///   expression doesn't already spell `MetricCheck::` on its callee path,
///   so bare constructors work without an import and qualified
///   constructors read naturally in modules that already have
///   `MetricCheck` in scope.
/// - `#[metric(name = "...", polarity = ..., unit = "...")]` —
///   kwarg form. `polarity` is one of `HigherBetter`, `LowerBetter`,
///   `TargetValue(f64)`, `Unknown`. May repeat; entries accumulate.
/// - `#[include_files("helper", "config.json", ...)]` — variadic
///   string literals appended to the emitted `include_files` slice
///   after the auto-injected binary entry. Each entry passes through
///   the same resolver used by the CLI `-i` flag (bare names search
///   host `PATH`; explicit paths must exist; directories are walked).
///   The primary binary is already packaged automatically, so this
///   attribute is only needed for auxiliary files the payload
///   depends on.
///
/// # Const name derivation
///
/// Strip trailing `"Payload"` suffix (if present), then convert to
/// `SCREAMING_SNAKE_CASE`. `FioPayload` → `FIO`,
/// `StressNgPayload` → `STRESS_NG`, `Fio` (no suffix) → `FIO`.
///
/// # Example
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// #[derive(Payload)]
/// #[payload(binary = "fio", output = Json)]
/// #[default_args("--output-format=json", "--minimal")]
/// #[default_check(exit_code_eq(0))]
/// #[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
/// struct FioPayload;
/// ```
#[proc_macro_derive(
    Payload,
    attributes(payload, default_args, default_check, metric, include_files)
)]
pub fn derive_payload(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match payload::derive_payload_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Generate per-field claim accessors on a stats struct.
///
/// See the `claim` module docs for the dispatch rules and label
/// invariant. Reject non-struct inputs and tuple-struct inputs — the
/// claim API is keyed on field names, which tuple structs do not have.
#[proc_macro_derive(Claim, attributes(claim))]
pub fn derive_claim(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match claim::derive_claim_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Convert JSON-like Rust tokens into a `&'static str` at compile time.
///
/// Accepts a superset of JSON syntax using Rust token trees:
/// - Objects: `{ "key": value, ... }`
/// - Arrays: `[value, ...]`
/// - Strings: `"hello"`
/// - Numbers: `42`, `3.14`, `-1`
/// - Booleans: `true`, `false`
/// - Null: `null`
/// - Trailing commas are stripped
///
/// ```rust,ignore
/// const CFG: &str = ktstr::json!({
///     "layers": [{
///         "name": "batch",
///         "kind": { "Grouped": { "cpus_range": [0, 4] } },
///     }],
/// });
/// ```
#[proc_macro]
pub fn json(input: TokenStream) -> TokenStream {
    let mut out = String::new();
    json::tokens_to_json(&mut out, proc_macro2::TokenStream::from(input));
    let lit = syn::LitStr::new(&out, proc_macro2::Span::call_site());
    TokenStream::from(quote! { #lit })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn camel_to_screaming_snake_acronym_run() {
        assert_eq!(
            payload::camel_to_screaming_snake("HTTPServer"),
            "HTTP_SERVER"
        );
    }

    #[test]
    fn camel_to_screaming_snake_single_word() {
        assert_eq!(payload::camel_to_screaming_snake("Llc"), "LLC");
    }

    #[test]
    fn camel_to_screaming_snake_all_caps_passthrough() {
        assert_eq!(payload::camel_to_screaming_snake("LLC"), "LLC");
    }

    #[test]
    fn option_tokens_some_int() {
        let opt: Option<u32> = Some(42);
        let ts = ktstr_test::option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { Some(42u32) }.to_string());
    }

    #[test]
    fn option_tokens_none_int() {
        let opt: Option<u32> = None;
        let ts = ktstr_test::option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { None }.to_string());
    }

    #[test]
    fn option_tokens_some_bool() {
        let opt: Option<bool> = Some(true);
        let ts = ktstr_test::option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { Some(true) }.to_string());
    }

    /// Pins the two attribute-name registries the unknown-attribute diagnostic
    /// is derived from: they must be internally duplicate-free, disjoint (a
    /// name is a bool flag XOR a value attr, never both), and sum to the full
    /// accepted set. A maintainer adding an attribute to the wrong slice — or
    /// to both — silently shifts the user-facing "expected:" list; this catches
    /// it. (The complementary direction — a name in a slice with no handling
    /// match arm — is guarded at parse time by the unknown-attribute catch-all
    /// `assert!`.)
    #[test]
    fn attr_name_registries_disjoint_and_complete() {
        use std::collections::HashSet;
        let bool_names = ktstr_test::BOOL_ATTR_NAMES;
        let value_names = ktstr_test::VALUE_ATTR_NAMES;
        let bool_set: HashSet<&str> = bool_names.iter().copied().collect();
        let value_set: HashSet<&str> = value_names.iter().copied().collect();
        // No duplicates within either slice.
        assert_eq!(
            bool_set.len(),
            bool_names.len(),
            "BOOL_ATTR_NAMES has duplicate entries",
        );
        assert_eq!(
            value_set.len(),
            value_names.len(),
            "VALUE_ATTR_NAMES has duplicate entries",
        );
        // Disjoint: an attribute is a bool flag XOR a value attr.
        let overlap: Vec<&str> = bool_set.intersection(&value_set).copied().collect();
        assert!(
            overlap.is_empty(),
            "BOOL_ATTR_NAMES and VALUE_ATTR_NAMES overlap: {overlap:?}",
        );
        // Cardinality pin: 16 bool + 49 value = 65 accepted attributes.
        assert_eq!(bool_names.len(), 16, "bool attribute count changed");
        assert_eq!(value_names.len(), 49, "value attribute count changed");
    }

    /// Contract pin: `ktstr_test::AttrValues::default()` is the single source of
    /// truth for every `#[ktstr_test]` macro default since step 2/4 of
    /// the parse-loop refactor. Without a field-by-field positive
    /// assertion a maintainer editing the [`Default`] impl can shift
    /// any user-visible default (auto_repro, kaslr, memory_mib, the
    /// gauntlet caps, etc.) with zero test feedback. Same precedent
    /// as `resolve_host_cgroup_parent_env_unset_returns_default` in
    /// src/test_support/dispatch_tests.rs pinning a runtime const
    /// against production source.
    #[test]
    fn attr_values_default_matches_documented_macro_defaults() {
        let d = ktstr_test::AttrValues::default();

        // -- Topology --
        assert_eq!(d.llcs, ktstr_test::DEFAULT_LLCS);
        assert_eq!(d.cores, ktstr_test::DEFAULT_CORES);
        assert_eq!(d.threads, ktstr_test::DEFAULT_THREADS);
        assert_eq!(d.numa_nodes, 1);
        assert!(!d.llcs_set);
        assert!(!d.cores_set);
        assert!(!d.threads_set);
        assert!(!d.numa_nodes_set);

        // -- Memory + duration --
        assert_eq!(d.memory_mib, ktstr_test::DEFAULT_MEMORY_MIB);
        assert!(!d.memory_mib_set);
        assert_eq!(d.duration_s, 2);
        assert!(!d.duration_s_set);
        assert_eq!(d.cleanup_budget_ms, None);
        assert_eq!(d.watchdog_timeout_s, 4);
        assert!(!d.watchdog_timeout_s_set);

        // -- Scheduler refs --
        assert!(d.scheduler.is_none());
        assert!(d.payload.is_none());
        assert!(d.workloads.is_none());
        assert!(d.staged_schedulers.is_none());
        assert!(d.bpf_map_write.is_none());
        assert!(d.post_vm.is_none());
        assert!(d.post_vm_unconditional.is_none());
        assert!(d.disk.is_none());
        assert!(d.networks.is_none());

        // -- Assert overrides --
        assert_eq!(d.not_starved, None);
        assert_eq!(d.isolation, None);
        assert_eq!(d.max_gap_ms, None);
        assert_eq!(d.max_spread_pct, None);
        assert_eq!(d.max_imbalance_ratio, None);
        assert_eq!(d.max_local_dsq_depth, None);
        assert_eq!(d.fail_on_stall, None);
        assert_eq!(d.sustained_samples, None);
        assert_eq!(d.max_throughput_cv, None);
        assert_eq!(d.min_work_rate, None);
        assert_eq!(d.max_fallback_rate, None);
        assert_eq!(d.max_keep_last_rate, None);
        assert_eq!(d.max_p99_wake_latency_ns, None);
        assert_eq!(d.max_wake_latency_cv, None);
        assert_eq!(d.min_iteration_rate, None);
        assert_eq!(d.max_migration_ratio, None);
        assert_eq!(d.min_page_locality, None);
        assert_eq!(d.max_cross_node_migration_ratio, None);
        assert_eq!(d.max_slow_tier_ratio, None);

        // -- TopologyConstraints --
        assert_eq!(d.min_numa_nodes, 1);
        assert!(!d.min_numa_nodes_set);
        assert_eq!(d.min_llcs, 1);
        assert!(!d.min_llcs_set);
        assert!(!d.requires_smt);
        assert!(!d.requires_smt_set);
        assert_eq!(d.min_cpus, 1);
        assert!(!d.min_cpus_set);
        assert_eq!(d.max_llcs, Some(12));
        assert!(!d.max_llcs_set);
        assert_eq!(d.max_numa_nodes, Some(1));
        assert!(!d.max_numa_nodes_set);
        assert_eq!(d.max_cpus, Some(192));
        assert!(!d.max_cpus_set);
        assert_eq!(d.cpu_budget, None);

        // -- Bool attrs (auto_repro + kaslr default TRUE; others false) --
        assert!(d.auto_repro);
        assert!(!d.auto_repro_set);
        assert!(!d.expect_auto_repro);
        assert!(!d.expect_auto_repro_set);
        assert!(!d.performance_mode);
        assert!(!d.performance_mode_set);
        assert!(!d.no_perf_mode);
        assert!(!d.no_perf_mode_set);
        assert!(!d.expect_err);
        assert!(!d.expect_err_set);
        assert!(!d.allow_inconclusive);
        assert!(!d.allow_inconclusive_set);
        assert!(!d.host_only);
        assert!(!d.host_only_set);
        assert!(!d.ignore_test);
        assert!(d.kaslr);
        assert!(!d.kaslr_set);
        assert!(!d.wprof);
        assert!(!d.wprof_set);
        assert_eq!(d.num_snapshots, 0);
        assert!(!d.num_snapshots_set);

        // -- Strings + tokens --
        assert!(d.extra_sched_args.is_empty());
        assert!(d.extra_include_files.is_empty());
        assert_eq!(d.workload_root_cgroup, None);
        assert!(d.wprof_args.is_none());
        assert!(d.expect_scx_bpf_error_contains_tokens.is_none());
        assert!(d.expect_scx_bpf_error_matches_tokens.is_none());
        assert!(d.config_expr.is_none());
        assert!(!d.config_set);
    }

    // -- expect_auto_repro macro-parse positive tests --
    //
    // Synthesize each attribute spelling, invoke ktstr_test_impl
    // directly (bypassing proc_macro::TokenStream which panics outside
    // a procedural-macro invocation), parse the output back into a
    // syn AST, locate the `static __KTSTR_ENTRY_*: KtstrTestEntry =
    // KtstrTestEntry { ... };` registration emitted by the macro, and
    // assert the `expect_auto_repro` field is either:
    //   - absent (omitted spelling — DEFAULT spread carries false), or
    //   - present with a `Lit::Bool { value: true/false }` value.
    //
    // The AST round-trip (rather than substring matching on the
    // output's `.to_string()`) guards against two failure modes:
    //   1. proc_macro2 version drift in colon/whitespace formatting —
    //      a future proc_macro2 release that emits `field: true` (no
    //      space) vs `field : true` (current) would silently flip a
    //      substring-matching test from PASS to FAIL or vice versa.
    //   2. structural defects that a substring check cannot detect —
    //      wrong outer struct name, wrong field value type (e.g. a
    //      String literal where a bool is expected), or a phantom
    //      sub-literal elsewhere in the output that happens to
    //      contain the same substring.
    //
    // For spellings that set expect_auto_repro = true, the cross-
    // attribute validation pass (added at the same time as the field)
    // requires a scheduler attribute + wprof attribute to be present.
    // The fixture inputs satisfy those preconditions so the parser
    // reaches codegen without rejection.

    /// Type-erased extraction from a `syn::Expr` for the
    /// [`field_value_in_static_entry`] helper. Each impl panics with
    /// a descriptive error if the expression shape doesn't match the
    /// expected literal kind — same wrong-type-rejection contract as
    /// the single-purpose helper this generalization replaces.
    trait ExtractFromExpr: Sized {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self;
    }

    impl ExtractFromExpr for bool {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Bool(b),
                    ..
                }) => b.value,
                other => panic!("{field_name} field value must be a Lit::Bool; got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for u32 {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(i),
                    ..
                }) => i
                    .base10_parse::<u32>()
                    .unwrap_or_else(|e| panic!("{field_name} field value parse as u32: {e}")),
                other => panic!("{field_name} field value must be a Lit::Int (u32); got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for u64 {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(i),
                    ..
                }) => i
                    .base10_parse::<u64>()
                    .unwrap_or_else(|e| panic!("{field_name} field value parse as u64: {e}")),
                other => panic!("{field_name} field value must be a Lit::Int (u64); got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for String {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => s.value(),
                other => panic!("{field_name} field value must be a Lit::Str; got {other:?}"),
            }
        }
    }

    impl ExtractFromExpr for syn::Path {
        fn extract_or_panic(field_name: &str, expr: &syn::Expr) -> Self {
            match expr {
                syn::Expr::Path(p) => p.path.clone(),
                other => panic!("{field_name} field value must be an Expr::Path; got {other:?}"),
            }
        }
    }

    /// Locate the `__KTSTR_ENTRY_*` static the macro emits and return
    /// the value of its `field_name` field if explicitly set, or
    /// `None` if the field is absent from the struct literal
    /// (omitted spellings fall through to the
    /// `..KtstrTestEntry::DEFAULT` spread).
    ///
    /// Generalized from the single-field `expect_auto_repro` helper
    /// to read any field whose value-type implements
    /// [`ExtractFromExpr`] (bool, u32, u64, String, syn::Path).
    /// The wrong-value-type rejection moves into the per-type
    /// `extract_or_panic` impl — a field whose expression doesn't
    /// match the requested type panics with a message naming both
    /// the field and the expected type.
    ///
    /// Verifies along the way:
    /// - exactly one `__KTSTR_ENTRY_*` static is emitted (panics on
    ///   zero or multiple — a future codegen change that emitted
    ///   two prefixed statics would silently key off the first
    ///   without the count assertion),
    /// - its declared type's last path segment is `KtstrTestEntry`,
    /// - its initializer is a struct literal whose path's last
    ///   segment is `KtstrTestEntry` (catches a regression that
    ///   wrapped the literal in a different outer struct),
    /// - any present `field_name` field's value matches the
    ///   requested type (panics via [`ExtractFromExpr::extract_or_panic`]
    ///   otherwise).
    fn field_value_in_static_entry<T: ExtractFromExpr>(
        output: &proc_macro2::TokenStream,
        field_name: &str,
    ) -> Option<T> {
        let file: syn::File =
            syn::parse2(output.clone()).expect("macro output must parse as a syn::File");
        let static_candidates: Vec<&syn::ItemStatic> = file
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Static(s) if s.ident.to_string().starts_with("__KTSTR_ENTRY_") => {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            static_candidates.len(),
            1,
            "macro must emit exactly one __KTSTR_ENTRY_* static; found {}",
            static_candidates.len()
        );
        let static_item = static_candidates[0];
        let static_type_last = match static_item.ty.as_ref() {
            syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
            _ => None,
        };
        assert_eq!(
            static_type_last.as_deref(),
            Some("KtstrTestEntry"),
            "static type's last path segment must be KtstrTestEntry"
        );
        let expr_struct = match static_item.expr.as_ref() {
            syn::Expr::Struct(s) => s,
            other => panic!("static initializer must be a struct literal; got {other:?}"),
        };
        let struct_last = expr_struct
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string());
        assert_eq!(
            struct_last.as_deref(),
            Some("KtstrTestEntry"),
            "struct-literal path's last segment must be KtstrTestEntry"
        );
        for field in &expr_struct.fields {
            let ident_matches = matches!(
                &field.member,
                syn::Member::Named(ident) if ident == field_name
            );
            if !ident_matches {
                continue;
            }
            return Some(T::extract_or_panic(field_name, &field.expr));
        }
        None
    }

    /// `#[ktstr_test(expect_auto_repro)]` (bare form) emits
    /// `expect_auto_repro: true` as a field on the
    /// `KtstrTestEntry` struct literal. Pins the bare-flag arm of
    /// the macro's bool-slot parser. Gated on the `wprof` feature: the
    /// attr pairs `wprof` with `expect_auto_repro` (the latter requires the
    /// former), and the macro only accepts `wprof` when the feature is on —
    /// so this case can only parse successfully under `--features wprof`.
    #[cfg(feature = "wprof")]
    #[test]
    fn macro_parses_expect_auto_repro_bare_to_true() {
        let attr = quote! { scheduler = SCHED, wprof, expect_auto_repro };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test::ktstr_test_impl(attr, item)
            .expect("bare attribute must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            Some(true),
            "bare `expect_auto_repro` must emit a `expect_auto_repro: true` field"
        );
    }

    /// `#[ktstr_test(expect_auto_repro = true)]` emits
    /// `expect_auto_repro: true`. Pins the explicit-true arm. Gated on the
    /// `wprof` feature (the attr requires `wprof`, accepted only with the
    /// feature on).
    #[cfg(feature = "wprof")]
    #[test]
    fn macro_parses_expect_auto_repro_explicit_true() {
        let attr = quote! { scheduler = SCHED, wprof, expect_auto_repro = true };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test::ktstr_test_impl(attr, item)
            .expect("explicit-true attribute must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            Some(true),
            "explicit `expect_auto_repro = true` must emit a `expect_auto_repro: true` field"
        );
    }

    /// `#[ktstr_test(expect_auto_repro = false)]` emits
    /// `expect_auto_repro: false`. Pins the explicit-false arm
    /// against a regression that conflated explicit-false with
    /// omission (which would silently leave DEFAULT untouched and
    /// lose the user's negative declaration). No cross-attribute
    /// gates apply when expect_auto_repro is false — the only
    /// rejection arms trigger on the true value.
    #[test]
    fn macro_parses_expect_auto_repro_explicit_false() {
        let attr = quote! { expect_auto_repro = false };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test::ktstr_test_impl(attr, item)
            .expect("explicit-false attribute must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            Some(false),
            "explicit `expect_auto_repro = false` must emit a `expect_auto_repro: false` field"
        );
    }

    /// Omitting the attribute entirely emits NO
    /// `expect_auto_repro` field — the generated struct literal
    /// uses the `..KtstrTestEntry::DEFAULT` spread to inherit the
    /// false default. Pins backward-compat: an existing
    /// `#[ktstr_test(...)]` with no expect_auto_repro must not
    /// gain a phantom field that flips the entry's behavior.
    #[test]
    fn macro_parses_omitted_expect_auto_repro_leaves_field_unemitted() {
        let attr = quote! {};
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test::ktstr_test_impl(attr, item)
            .expect("attribute-less invocation must parse successfully");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "expect_auto_repro"),
            None,
            "omitted attribute must NOT emit any `expect_auto_repro` field — DEFAULT spread carries the false"
        );
    }

    /// `#[ktstr_test(scheduler = SCHED, survives_storm)]` emits
    /// `survives_storm: true`. Pins BOOL_ATTR_NAMES + assign_bool + codegen.
    /// A scheduler token is present so the mutex does not reject.
    #[test]
    fn macro_parses_survives_storm_bare_to_true() {
        let attr = quote! { scheduler = SCHED, survives_storm };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test::ktstr_test_impl(attr, item)
            .expect("bare survives_storm with a scheduler must parse");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "survives_storm"),
            Some(true),
            "bare `survives_storm` must emit `survives_storm: true`"
        );
    }

    /// `#[ktstr_test(survives_storm = false)]` emits `survives_storm: false`
    /// and needs no scheduler (the mutex only fires on the true value).
    #[test]
    fn macro_parses_survives_storm_explicit_false() {
        let attr = quote! { survives_storm = false };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let out = ktstr_test::ktstr_test_impl(attr, item)
            .expect("explicit-false survives_storm must parse");
        assert_eq!(
            field_value_in_static_entry::<bool>(&out, "survives_storm"),
            Some(false),
            "explicit `survives_storm = false` must emit `survives_storm: false`"
        );
    }

    /// `survives_storm` + `expect_err` is rejected at macro parse —
    /// contradictory polarity. Pins the `validate_survives_storm_mutex`
    /// expect_err arm.
    #[test]
    fn macro_rejects_survives_storm_with_expect_err() {
        let attr = quote! { scheduler = SCHED, survives_storm, expect_err = true };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let err = ktstr_test::ktstr_test_impl(attr, item).unwrap_err();
        assert!(
            err.to_string().contains("survives_storm") && err.to_string().contains("expect_err"),
            "diagnostic must name both survives_storm and expect_err: {err}"
        );
    }

    /// `survives_storm` without a scheduler is rejected at macro parse —
    /// the kernel default has no scx scheduler to die or be ejected.
    #[test]
    fn macro_rejects_survives_storm_without_scheduler() {
        let attr = quote! { survives_storm };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let err = ktstr_test::ktstr_test_impl(attr, item).unwrap_err();
        assert!(
            err.to_string().contains("survives_storm") && err.to_string().contains("scheduler"),
            "diagnostic must name survives_storm and the missing scheduler: {err}"
        );
    }

    /// `survives_storm` + `expect_auto_repro` is rejected at macro parse —
    /// both are inversion intents (survives_storm forces a death fail to
    /// EXIT_FAIL; expect_auto_repro inverts a crash-with-repro fail to
    /// PASS). Feature-agnostic: `validate_cross_attr` runs the
    /// survives_storm mutex (which fires on the `expect_auto_repro` arm)
    /// BEFORE the `#[cfg(not(feature = "wprof"))]` wprof-required rejection,
    /// so the diagnostic names both regardless of the `wprof` feature —
    /// unlike the positive parse tests, which need `--features wprof` to
    /// reach codegen successfully. The runtime twin is
    /// `validate_rejects_survives_storm_with_expect_auto_repro`.
    #[test]
    fn macro_rejects_survives_storm_with_expect_auto_repro() {
        let attr = quote! { scheduler = SCHED, wprof, survives_storm, expect_auto_repro };
        let item = quote! {
            fn t(_: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
                Ok(ktstr::assert::AssertResult::pass())
            }
        };
        let err = ktstr_test::ktstr_test_impl(attr, item).unwrap_err();
        assert!(
            err.to_string().contains("survives_storm")
                && err.to_string().contains("expect_auto_repro"),
            "diagnostic must name both survives_storm and expect_auto_repro: {err}"
        );
    }

    // -- check_visible_lit (commit f4018278) --------------------------

    #[test]
    fn check_visible_lit_visible_string_passes() {
        let expr: syn::Expr = syn::parse_quote!("hello");
        scheduler::check_visible_lit("hello", &expr, "name")
            .expect("non-empty visible string must pass");
    }

    #[test]
    fn check_visible_lit_empty_string_rejected() {
        let expr: syn::Expr = syn::parse_quote!("");
        let err = scheduler::check_visible_lit("", &expr, "name").unwrap_err();
        assert!(
            err.to_string()
                .contains("`name` must contain at least one visible character"),
            "expected `name` visible-empty diagnostic, got: {err}"
        );
    }

    #[test]
    fn check_visible_lit_whitespace_only_rejected() {
        let expr: syn::Expr = syn::parse_quote!("   ");
        let err = scheduler::check_visible_lit("   ", &expr, "binary").unwrap_err();
        assert!(
            err.to_string()
                .contains("`binary` must contain at least one visible character"),
            "expected `binary` visible-empty diagnostic, got: {err}"
        );
    }

    #[test]
    fn check_visible_lit_invisible_only_rejected() {
        let expr: syn::Expr = syn::parse_quote!("zwsp");
        let err = scheduler::check_visible_lit("\u{200B}", &expr, "binary_path").unwrap_err();
        assert!(
            err.to_string()
                .contains("`binary_path` must contain at least one visible character"),
            "expected `binary_path` visible-empty diagnostic, got: {err}"
        );
    }

    // -- validate_kernel_builtin_pair (commit 753ecf9e) ---------------

    #[test]
    fn validate_kernel_builtin_pair_both_set_passes() {
        let span = proc_macro2::Span::call_site();
        scheduler::validate_kernel_builtin_pair(Some(span), Some(span))
            .expect("both set is valid (KernelBuiltin)");
    }

    #[test]
    fn validate_kernel_builtin_pair_neither_set_passes() {
        scheduler::validate_kernel_builtin_pair(None, None)
            .expect("neither set is valid (not KernelBuiltin)");
    }

    #[test]
    fn validate_kernel_builtin_pair_enable_only_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = scheduler::validate_kernel_builtin_pair(Some(span), None).unwrap_err();
        assert!(
            err.to_string()
                .contains("`kernel_builtin_enable` set without `kernel_builtin_disable`"),
            "expected enable-without-disable diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_kernel_builtin_pair_disable_only_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = scheduler::validate_kernel_builtin_pair(None, Some(span)).unwrap_err();
        assert!(
            err.to_string()
                .contains("`kernel_builtin_disable` set without `kernel_builtin_enable`"),
            "expected disable-without-enable diagnostic, got: {err}"
        );
    }

    // -- validate_exactly_one_source (commit 7c796939) ----------------

    #[test]
    fn validate_exactly_one_source_none_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = scheduler::validate_exactly_one_source(false, false, false, span).unwrap_err();
        assert!(
            err.to_string().contains("no scheduler source declared"),
            "expected no-source diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_exactly_one_source_only_binary_passes() {
        let span = proc_macro2::Span::call_site();
        scheduler::validate_exactly_one_source(true, false, false, span)
            .expect("binary-only is valid");
    }

    #[test]
    fn validate_exactly_one_source_only_binary_path_passes() {
        let span = proc_macro2::Span::call_site();
        scheduler::validate_exactly_one_source(false, true, false, span)
            .expect("binary_path-only is valid");
    }

    #[test]
    fn validate_exactly_one_source_only_kernel_builtin_passes() {
        let span = proc_macro2::Span::call_site();
        scheduler::validate_exactly_one_source(false, false, true, span)
            .expect("kernel_builtin-only is valid");
    }

    #[test]
    fn validate_exactly_one_source_binary_and_path_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = scheduler::validate_exactly_one_source(true, true, false, span).unwrap_err();
        assert!(
            err.to_string()
                .contains("more than one scheduler source declared"),
            "expected multi-source diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_exactly_one_source_all_three_rejected() {
        let span = proc_macro2::Span::call_site();
        let err = scheduler::validate_exactly_one_source(true, true, true, span).unwrap_err();
        assert!(
            err.to_string()
                .contains("more than one scheduler source declared"),
            "expected multi-source diagnostic, got: {err}"
        );
    }

    // -- validate_kernel_name_collision (commit 7480df1c) -------------

    #[test]
    fn validate_kernel_name_collision_non_kernel_passes() {
        let expr: syn::Expr = syn::parse_quote!("scx_mitosis");
        scheduler::validate_kernel_name_collision(true, "scx_mitosis", Some(&expr))
            .expect("non-`kernel` name is valid");
    }

    #[test]
    fn validate_kernel_name_collision_not_kernel_builtin_passes() {
        let expr: syn::Expr = syn::parse_quote!("kernel");
        scheduler::validate_kernel_name_collision(false, "kernel", Some(&expr))
            .expect("`kernel` name is valid when not KernelBuiltin variant");
    }

    #[test]
    fn validate_kernel_name_collision_exact_kernel_rejected() {
        let expr: syn::Expr = syn::parse_quote!("kernel");
        let err =
            scheduler::validate_kernel_name_collision(true, "kernel", Some(&expr)).unwrap_err();
        assert!(
            err.to_string()
                .contains("collides with the KernelBuiltin variant's display_name"),
            "expected collision diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_kernel_name_collision_case_insensitive_rejected() {
        let expr: syn::Expr = syn::parse_quote!("Kernel");
        let err =
            scheduler::validate_kernel_name_collision(true, "Kernel", Some(&expr)).unwrap_err();
        assert!(
            err.to_string()
                .contains("collides with the KernelBuiltin variant's display_name"),
            "expected case-insensitive collision diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_kernel_name_collision_whitespace_padded_rejected() {
        let expr: syn::Expr = syn::parse_quote!("  Kernel  ");
        let err =
            scheduler::validate_kernel_name_collision(true, "  Kernel  ", Some(&expr)).unwrap_err();
        assert!(
            err.to_string()
                .contains("collides with the KernelBuiltin variant's display_name"),
            "expected whitespace-insensitive collision diagnostic, got: {err}"
        );
    }

    // -- validate_payload_workloads_dedup (commit 26352fd7) -----------

    #[test]
    fn validate_payload_workloads_dedup_empty_workloads_passes() {
        let payload: Option<syn::Path> = Some(syn::parse_quote!(FIO));
        ktstr_test::validate_payload_workloads_dedup(&payload, &[])
            .expect("empty workloads is valid");
    }

    #[test]
    fn validate_payload_workloads_dedup_disjoint_passes() {
        let payload: Option<syn::Path> = Some(syn::parse_quote!(FIO));
        let workloads: Vec<syn::Path> =
            vec![syn::parse_quote!(STRESS_NG), syn::parse_quote!(NETPERF)];
        ktstr_test::validate_payload_workloads_dedup(&payload, &workloads)
            .expect("disjoint workloads is valid");
    }

    #[test]
    fn validate_payload_workloads_dedup_primary_in_workloads_rejected() {
        let payload: Option<syn::Path> = Some(syn::parse_quote!(FIO));
        let workloads: Vec<syn::Path> = vec![syn::parse_quote!(STRESS_NG), syn::parse_quote!(FIO)];
        let err = ktstr_test::validate_payload_workloads_dedup(&payload, &workloads).unwrap_err();
        assert!(
            err.to_string().contains("appears in both"),
            "expected payload-in-workloads diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_payload_workloads_dedup_pairwise_duplicate_rejected() {
        let payload: Option<syn::Path> = None;
        let workloads: Vec<syn::Path> = vec![syn::parse_quote!(FIO), syn::parse_quote!(FIO)];
        let err = ktstr_test::validate_payload_workloads_dedup(&payload, &workloads).unwrap_err();
        assert!(
            err.to_string().contains("appears twice"),
            "expected pairwise-dupe diagnostic, got: {err}"
        );
    }
}
