//! Unit tests for [`super`] (the `host_context` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

use super::*;

/// Host-context reads are host-dependent: we assert the
/// collector returns SOMETHING, not specific values. On Linux
/// CI the uname fields at least should populate.
#[cfg(target_os = "linux")]
#[test]
fn collect_host_context_returns_populated_struct_on_linux() {
    let ctx = collect_host_context();
    // uname is always readable on Linux (it's a syscall, no
    // filesystem dependency), so these three must populate.
    assert_eq!(ctx.kernel_name.as_deref(), Some("Linux"));
    assert!(ctx.kernel_release.is_some(), "uname release present");
    assert!(ctx.arch.is_some(), "uname machine present");
}

/// `/proc/cmdline` is always readable on a running Linux system
/// (the kernel exposes it unconditionally). The capture is
/// verbatim — `read_trimmed_sysfs` trims leading/trailing
/// whitespace and returns `None` only when the read fails or
/// the file is empty after trim. No token filtering is applied.
/// Because the cmdline is always present on Linux, this test
/// asserts the field populates unconditionally; an if-let
/// version of this check would pass vacuously on a kernel that
/// accidentally dropped the capture.
#[cfg(target_os = "linux")]
#[test]
fn collect_host_context_captures_cmdline_on_linux() {
    let ctx = collect_host_context();
    let cmdline = ctx
        .kernel_cmdline
        .as_deref()
        .expect("/proc/cmdline is always readable on a running Linux system");
    assert!(
        !cmdline.is_empty(),
        "populated kernel_cmdline must not be empty"
    );
    assert_eq!(cmdline, cmdline.trim());
}

/// Stability regression for the STATIC subset: uname triple,
/// CPU identity, total_memory_kib, hugepages_size_kib,
/// online_cpus, numa_nodes, cpufreq_governor. These fields are
/// memoised in [`STATIC_HOST_INFO`] (or, for `cpufreq_governor`,
/// in [`CPUFREQ_GOVERNORS`]) and therefore return identical
/// values across back-to-back calls regardless of what other
/// tests run concurrently — they are safe to assert equality
/// on under nextest's parallel-test model.
#[cfg(target_os = "linux")]
#[test]
fn collect_host_context_static_subset_is_stable_across_calls() {
    let a = collect_host_context();
    let b = collect_host_context();
    assert_eq!(a.kernel_name, b.kernel_name);
    assert_eq!(a.kernel_release, b.kernel_release);
    assert_eq!(a.arch, b.arch);
    assert_eq!(a.cpu_model, b.cpu_model);
    assert_eq!(a.cpu_vendor, b.cpu_vendor);
    assert_eq!(a.total_memory_kib, b.total_memory_kib);
    assert_eq!(a.hugepages_size_kib, b.hugepages_size_kib);
    assert_eq!(a.online_cpus, b.online_cpus);
    assert_eq!(a.numa_nodes, b.numa_nodes);
    assert_eq!(a.cpufreq_governor, b.cpufreq_governor);
}

/// Stability regression for the DYNAMIC subset: kernel_cmdline,
/// hugepages_{total,free}, thp_enabled / thp_defrag, and
/// sched_tunables. These fields are re-read on every
/// [`collect_host_context`] call by design — a concurrent test
/// that reserves hugepages, flips a THP policy, or writes a
/// `/proc/sys/kernel/sched_*` tunable would cause back-to-back
/// reads to diverge under nextest's parallel-test model. The
/// in-tree tests do not touch these knobs, so on a quiescent
/// host the fields match; the assertion is relaxed to "both
/// Some or both None" rather than full equality so a concurrent
/// hugepage reservation in a theoretical future test does not
/// flake this regression guard. `kernel_cmdline` is effectively
/// static (changes only across reboot), so it asserts equality.
#[cfg(target_os = "linux")]
#[test]
fn collect_host_context_dynamic_subset_is_stable_across_calls() {
    let a = collect_host_context();
    let b = collect_host_context();
    // kernel_cmdline changes only across reboot — safe to pin.
    assert_eq!(a.kernel_cmdline, b.kernel_cmdline);
    // For the remaining dynamic fields, assert presence parity
    // only: a concurrent sysctl/THP/hugepage twiddle would
    // break equality but must not break the "collector keeps
    // producing readable values" contract.
    assert_eq!(a.hugepages_total.is_some(), b.hugepages_total.is_some());
    assert_eq!(a.hugepages_free.is_some(), b.hugepages_free.is_some());
    assert_eq!(a.thp_enabled.is_some(), b.thp_enabled.is_some());
    assert_eq!(a.thp_defrag.is_some(), b.thp_defrag.is_some());
    assert_eq!(a.sched_tunables.is_some(), b.sched_tunables.is_some());
}

/// Direct OnceLock caching test for `STATIC_HOST_INFO`. The
/// sibling `collect_host_context_static_subset_is_stable_across_calls`
/// proves static fields match between calls but does not
/// verify the cache mechanism itself — the two reads could
/// both hit
/// `compute_static_host_info` and still match on a quiescent
/// host. This test pins the caching contract directly: after
/// the first call populates `STATIC_HOST_INFO`, the stored
/// reference survives the second call unchanged (same allocation
/// address AND same field values), proving `get_or_init` hit the
/// cached branch instead of re-running the init closure.
///
/// Uses `OnceLock::get` (non-init probe) to observe cache state
/// without touching it.
///
/// Robust to test ordering: if another test populated
/// `STATIC_HOST_INFO` first, `collect_host_context()` here hits
/// the cache and the pointer comparison still passes because
/// `OnceLock` permits no re-init.
#[cfg(target_os = "linux")]
#[test]
fn static_host_info_is_cached_after_first_call() {
    let _ = collect_host_context();
    let first = STATIC_HOST_INFO
        .get()
        .expect("STATIC_HOST_INFO must be populated after collect_host_context");
    let first_ptr = first as *const StaticHostInfo;

    let _ = collect_host_context();
    let second = STATIC_HOST_INFO
        .get()
        .expect("STATIC_HOST_INFO must still be populated on second call");
    let second_ptr = second as *const StaticHostInfo;

    assert_eq!(
        first_ptr, second_ptr,
        "OnceLock must return the same allocation across calls — \
             a pointer mismatch means the cache re-initialized, \
             defeating the get_or_init contract",
    );
    // Cross-check field-level equality. Redundant with the pointer
    // check but serves as a second anchor so a future replacement
    // of `OnceLock` with something that clones on access still
    // fails loudly rather than silently weakening the cache.
    assert_eq!(first.cpu_model, second.cpu_model);
    assert_eq!(first.kernel_release, second.kernel_release);
    assert_eq!(first.total_memory_kib, second.total_memory_kib);
}

/// Host context round-trips through JSON — every field uses
/// `#[serde(default, skip_serializing_if)]` so absent Options
/// do not appear in the output and empty output parses back to
/// `HostContext::default()`.
#[test]
fn host_context_empty_round_trips_via_json() {
    let empty = HostContext::default();
    let json = serde_json::to_string(&empty).expect("serialize empty");
    assert_eq!(
        json, "{}",
        "default host context must serialize to empty object"
    );
    let decoded: HostContext = serde_json::from_str(&json).expect("deserialize empty");
    assert!(decoded.cpu_model.is_none());
    assert!(decoded.kernel_name.is_none());
    assert!(decoded.kernel_cmdline.is_none());
}

/// Populated host context round-trips — struct-level
/// `PartialEq` makes one `assert_eq!(decoded, ctx)` cover every
/// field. Any future field addition or serde-attr change that
/// breaks the round-trip for any single field is caught without
/// needing a per-field assertion.
#[test]
fn host_context_populated_round_trips_via_json() {
    let mut tunables = BTreeMap::new();
    tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
    let ctx = HostContext {
        cpu_model: Some("Example CPU".to_string()),
        cpu_vendor: Some("GenuineExample".to_string()),
        total_memory_kib: Some(16_384_000),
        hugepages_total: Some(0),
        hugepages_free: Some(0),
        hugepages_size_kib: Some(2048),
        thp_enabled: Some("always [madvise] never".to_string()),
        thp_defrag: Some("[always] defer defer+madvise madvise never".to_string()),
        sched_tunables: Some(tunables),
        online_cpus: Some(16),
        numa_nodes: Some(2),
        cpufreq_governor: BTreeMap::new(),
        kernel_name: Some("Linux".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        arch: Some("x86_64".to_string()),
        kernel_cmdline: Some("preempt=lazy transparent_hugepage=madvise".to_string()),
        task_delayacct: Some(DelayacctState::On),
        config_task_xacct: Some(XacctState::On),
        heap_state: Some(crate::host_heap::HostHeapState::test_fixture()),
    };
    let json = serde_json::to_string(&ctx).expect("serialize");
    let decoded: HostContext = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded, ctx);
}

/// Partial-None round-trip: mixed `Some`/`None` fields plus a
/// `Some(BTreeMap)` that is intentionally empty. Covers the gap
/// between the fully-None and fully-populated endpoints — a
/// regression that drops a specific `Some` into `None` (or
/// coerces `Some(empty map)` into `None` on deserialize) would
/// pass both existing tests while breaking real sidecars where
/// partial ctprof captures are the norm (first `/proc`
/// entry unreadable, sched_* dir readable but filtered to
/// empty, etc.). Struct-level `PartialEq` catches the whole
/// shape in one assertion.
#[test]
fn host_context_partial_none_round_trips_via_json() {
    let ctx = HostContext {
        // Identity captured on the production path.
        kernel_name: Some("Linux".to_string()),
        // Release read failed (e.g. uname syscall error on the
        // simulated failure path).
        kernel_release: None,
        arch: Some("x86_64".to_string()),
        // Map was captured but is empty — the `read_dir` of
        // /proc/sys/kernel succeeded, no entries matched the
        // `sched_*` filter (unusual but the code contract
        // explicitly distinguishes this from `None`).
        sched_tunables: Some(BTreeMap::new()),
        // Rest: None to exercise the omitted-key deserialize
        // path for every other Option field.
        cpu_model: None,
        cpu_vendor: None,
        total_memory_kib: None,
        hugepages_total: None,
        hugepages_free: None,
        hugepages_size_kib: None,
        thp_enabled: None,
        thp_defrag: None,
        online_cpus: None,
        numa_nodes: None,
        cpufreq_governor: BTreeMap::new(),
        kernel_cmdline: None,
        task_delayacct: None,
        config_task_xacct: None,
        heap_state: None,
    };
    let json = serde_json::to_string(&ctx).expect("serialize");
    let decoded: HostContext = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded, ctx);
}

#[test]
fn parse_cpuinfo_identity_happy_path() {
    let text = "\
processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 85
model name\t: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
stepping\t: 4
";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert_eq!(
        model.as_deref(),
        Some("Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz"),
    );
    assert_eq!(vendor.as_deref(), Some("GenuineIntel"));
}

#[test]
fn parse_cpuinfo_identity_empty_input() {
    let (model, vendor) = parse_cpuinfo_identity("");
    assert!(model.is_none());
    assert!(vendor.is_none());
}

#[test]
fn parse_cpuinfo_identity_arm64_no_model_or_vendor() {
    // ARM64 /proc/cpuinfo has neither `model name` nor
    // `vendor_id` — it uses `CPU implementer`, `CPU part`, etc.
    let text = "\
processor\t: 0
BogoMIPS\t: 50.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x3
CPU part\t: 0xd0c
CPU revision\t: 1
";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert!(model.is_none(), "no 'model name' line on ARM64");
    assert!(vendor.is_none(), "no 'vendor_id' line on ARM64");
}

#[test]
fn parse_cpuinfo_identity_malformed_lines_are_skipped() {
    // Lines without ':' are skipped; lines with empty value
    // after trim are skipped.
    let text = "\
nonsense line with no colon
vendor_id\t:
model name\t:    Actual Model Name
vendor_id\t: ActualVendor
";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert_eq!(model.as_deref(), Some("Actual Model Name"));
    assert_eq!(
        vendor.as_deref(),
        Some("ActualVendor"),
        "empty vendor line must not poison — next real value wins",
    );
}

#[test]
fn parse_cpuinfo_identity_crlf_line_endings() {
    // `str::lines()` accepts both \n and \r\n — the \r in \r\n
    // is stripped by str::lines() itself; the trim handles any
    // residual whitespace.
    let text = "vendor_id\t: GenuineIntel\r\nmodel name\t: Some CPU\r\n";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert_eq!(model.as_deref(), Some("Some CPU"));
    assert_eq!(vendor.as_deref(), Some("GenuineIntel"));
}

#[test]
fn parse_cpuinfo_identity_first_processor_only() {
    // Multi-processor /proc/cpuinfo — blank line separates
    // processor blocks. Only the first block's values must
    // surface; later blocks with different values are ignored.
    let text = "\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: First CPU

processor\t: 1
vendor_id\t: DifferentVendor
model name\t: Second CPU
";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert_eq!(model.as_deref(), Some("First CPU"));
    assert_eq!(vendor.as_deref(), Some("GenuineIntel"));
}

#[test]
fn parse_meminfo_happy_path() {
    let text = "\
MemTotal:       16384000 kB
MemFree:         8000000 kB
HugePages_Total:      42
HugePages_Free:       40
Hugepagesize:       2048 kB
";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(16_384_000));
    assert_eq!(out.hugepages_total, Some(42));
    assert_eq!(out.hugepages_free, Some(40));
    assert_eq!(out.hugepages_size_kib, Some(2048));
}

#[test]
fn parse_meminfo_empty_input() {
    let out = parse_meminfo("");
    assert!(out.mem_total_kib.is_none());
    assert!(out.hugepages_total.is_none());
    assert!(out.hugepages_free.is_none());
    assert!(out.hugepages_size_kib.is_none());
}

#[test]
fn parse_meminfo_missing_fields_stay_none() {
    // Only MemTotal is present — the other three fields must
    // remain None so callers can distinguish "zero" from
    // "absent."
    let text = "MemTotal:       1024 kB\nMemFree:         512 kB\n";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(1024));
    assert!(out.hugepages_total.is_none());
    assert!(out.hugepages_free.is_none());
    assert!(out.hugepages_size_kib.is_none());
}

#[test]
fn parse_meminfo_non_numeric_value_skipped() {
    // A future kernel flags-style line ("SomeFlags: abc def")
    // must not poison the struct — its non-numeric first token
    // causes the line to be skipped silently.
    let text = "\
MemTotal:       2048 kB
SomeFlags:      abc def ghi
Hugepagesize:      2048 kB
";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(2048));
    assert_eq!(out.hugepages_size_kib, Some(2048));
}

#[test]
fn parse_meminfo_unknown_fields_tolerated() {
    // Unknown keys must be ignored without affecting known
    // fields — adding new /proc/meminfo lines upstream is a
    // no-op here.
    let text = "\
MemTotal:       100 kB
Unknown_Field:  999
HugePages_Total:   3
Another_Unknown: 77 kB
";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(100));
    assert_eq!(out.hugepages_total, Some(3));
    assert!(out.hugepages_free.is_none());
}

#[test]
fn parse_meminfo_crlf_line_endings() {
    let text = "MemTotal:       512 kB\r\nHugePages_Total:    2\r\nHugepagesize:   2048 kB\r\n";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(512));
    assert_eq!(out.hugepages_total, Some(2));
    assert_eq!(out.hugepages_size_kib, Some(2048));
}

#[test]
fn parse_cpuinfo_identity_duplicate_key_first_wins() {
    // Two `model name` / `vendor_id` lines in the first
    // processor block. The match guard is `if model.is_none()`,
    // so the first occurrence must win; the second is ignored.
    let text = "\
vendor_id\t: FirstVendor
model name\t: First Model
vendor_id\t: SecondVendor
model name\t: Second Model
";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert_eq!(model.as_deref(), Some("First Model"));
    assert_eq!(vendor.as_deref(), Some("FirstVendor"));
}

#[test]
fn parse_cpuinfo_identity_value_with_internal_colon() {
    // `str::split_once(':')` splits on the first colon only,
    // so any ':' inside the value survives verbatim. Real
    // /proc/cpuinfo model names rarely contain ':' but the
    // parser must preserve them.
    let text = "model name\t: Intel(R): Xeon(R) CPU @ 2.00GHz\n";
    let (model, _vendor) = parse_cpuinfo_identity(text);
    assert_eq!(
        model.as_deref(),
        Some("Intel(R): Xeon(R) CPU @ 2.00GHz"),
        "internal ':' must be preserved in the value",
    );
}

#[test]
fn parse_cpuinfo_identity_leading_blank_line() {
    // The loop breaks on the first empty line (processor-block
    // boundary). A leading blank line therefore terminates
    // before any field is read — result is (None, None).
    let text = "\nvendor_id\t: GenuineIntel\nmodel name\t: Some CPU\n";
    let (model, vendor) = parse_cpuinfo_identity(text);
    assert!(model.is_none(), "leading blank line must short-circuit");
    assert!(vendor.is_none(), "leading blank line must short-circuit");
}

#[test]
fn parse_meminfo_duplicate_key_last_wins() {
    // Unlike parse_cpuinfo_identity, parse_meminfo's match
    // arms assign unconditionally — the last occurrence of a
    // key overrides earlier ones. Documented here so a future
    // change to this behavior (e.g. adding a first-wins guard)
    // is caught by this test.
    let text = "MemTotal:       100 kB\nMemTotal:       200 kB\n";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(200));
}

#[test]
fn parse_meminfo_line_without_colon() {
    // Lines without ':' are skipped via `split_once(':')`
    // returning None. Real /proc/meminfo never emits such
    // lines but the parser must tolerate them without
    // dropping the surrounding valid content.
    let text = "\
garbage line without any colon
MemTotal:       100 kB
another garbage line
HugePages_Total:   3
";
    let out = parse_meminfo(text);
    assert_eq!(out.mem_total_kib, Some(100));
    assert_eq!(out.hugepages_total, Some(3));
}

#[test]
fn parse_meminfo_empty_value_after_colon() {
    // A key with an empty value after the colon: rest is "",
    // split_whitespace().next() returns None, token becomes
    // the empty string, parse::<u64>() fails, the line is
    // skipped. The target field stays None so the absence is
    // visible to callers.
    let text = "MemTotal:\nHugePages_Total:  5\n";
    let out = parse_meminfo(text);
    assert!(
        out.mem_total_kib.is_none(),
        "empty value after ':' must leave the field None",
    );
    assert_eq!(
        out.hugepages_total,
        Some(5),
        "subsequent valid lines must still parse",
    );
}

#[test]
fn parse_meminfo_negative_and_overflow_value_skipped() {
    // u64 parsing rejects both negative values and values
    // exceeding u64::MAX. Both failure modes must skip the
    // line silently; later valid lines still parse.
    let text = "\
MemTotal:       -1 kB
HugePages_Total:   99999999999999999999999
Hugepagesize:       2048 kB
";
    let out = parse_meminfo(text);
    assert!(
        out.mem_total_kib.is_none(),
        "negative value must fail u64 parse and skip",
    );
    assert!(
        out.hugepages_total.is_none(),
        "overflow value must fail u64 parse and skip",
    );
    assert_eq!(
        out.hugepages_size_kib,
        Some(2048),
        "later valid line must still parse",
    );
}

#[test]
fn parse_trimmed_empty_is_none() {
    assert!(parse_trimmed("").is_none());
}

#[test]
fn parse_trimmed_whitespace_only_is_none() {
    // Spaces, tabs, and newlines all count as whitespace for
    // `str::trim`; a file containing only those characters
    // carries no signal and must map to None.
    assert!(parse_trimmed("   \n\t  \r\n").is_none());
}

#[test]
fn parse_trimmed_strips_trailing_newline() {
    // sysfs leaves typically end with a single trailing '\n';
    // the parser must strip it so downstream comparisons do
    // not carry stray whitespace.
    assert_eq!(parse_trimmed("content\n").as_deref(), Some("content"));
}

#[test]
fn parse_trimmed_preserves_bracketed_thp() {
    // THP policy files read like `"always [madvise] never\n"`;
    // the bracket indicating the active selection must survive
    // the trim verbatim because `str::trim` only touches the
    // edges.
    assert_eq!(
        parse_trimmed("always [madvise] never\n").as_deref(),
        Some("always [madvise] never"),
    );
}

// -- format_human / diff --

/// Canonical list of every `HostContext` field name that
/// [`HostContext::format_human`] and [`HostContext::diff`] must
/// render. Used to pin both surfaces against the same enumeration
/// so a new field landing on the struct is caught in the
/// render-check tests even if the author remembered to extend
/// the destructure bindings but forgot the corresponding `row()`
/// call.
///
/// The destructuring binds in `format_human` / `diff` already
/// force every struct field to appear by NAME (exhaustive
/// pattern — a new field without a binding fails to compile).
/// What destructure-binding does NOT catch is the follow-on
/// "added binding, forgot `row()` call" drift: an unused
/// destructure binding is a warning, not an error, under the
/// default lint profile, so the renderer can silently drop a
/// field. This constant + the paired enumeration-coverage tests
/// below close that gap — the test iterates every name here
/// against the actual render output.
///
/// **When adding a `HostContext` field:** extend this list AND
/// both render functions. A missing entry here surfaces as a
/// test failure in `format_human_renders_every_documented_field`
/// / `diff_renders_every_documented_field`; a missing
/// `row()`/destructure binding surfaces as a compile error in
/// the render function itself; a struct-field-count / list-
/// cardinality mismatch surfaces as a compile error in
/// [`struct_field_array`] below (see `_HOST_CONTEXT_FIELD_COUNT_PIN`).
const HOST_CONTEXT_FIELDS: &[&str] = &[
    "kernel_name",
    "kernel_release",
    "arch",
    "cpu_model",
    "cpu_vendor",
    "total_memory_kib",
    "hugepages_total",
    "hugepages_free",
    "hugepages_size_kib",
    "online_cpus",
    "numa_nodes",
    "thp_enabled",
    "thp_defrag",
    "kernel_cmdline",
    "cpufreq_governor",
    "sched_tunables",
    "heap_state",
    "task_delayacct",
    "config_task_xacct",
];

/// Consume any value, returning `()`. Test-only helper used
/// by [`struct_field_array`] to turn each destructured
/// [`HostContext`] field into a `()` slot so the resulting
/// fixed-size array's length IS the field count.
fn drop_to_unit<T>(_: T) {}

/// Exhaustive destructure of an owned [`HostContext`] into a
/// fixed-size array whose length is statically typed as
/// [`HOST_CONTEXT_FIELDS.len()`]. Never called at runtime
/// (marked dead_code) — exists purely to cross-enforce the
/// three cardinalities that must agree whenever
/// [`HostContext`] grows a field.
///
/// Compile-time cross-check:
///
///   1. Adding a struct field WITHOUT updating the destructure
///      pattern here triggers `missing fields in pattern` —
///      the destructure uses no `..` rest, so exhaustiveness
///      is enforced by the compiler.
///   2. Adding a destructure binding WITHOUT extending the
///      array initializer below triggers an unused-variable
///      warning AND a length mismatch against the return
///      type `[(); HOST_CONTEXT_FIELDS.len()]`.
///   3. Extending the array initializer WITHOUT growing
///      [`HOST_CONTEXT_FIELDS`] fails at the return-type
///      check: the literal has N+1 elements, the return type
///      demands N.
///
/// Dropped by value (non-const fn) — [`HostContext`] owns
/// `String`/`Option<String>` which is not const-droppable, so
/// this cannot be a `const fn`. The compile-time value is not
/// in the call but in the TYPE-CHECKED destructure: the
/// function's body is still type-checked by the compiler even
/// though no call site exists, which is all this pin needs.
#[allow(dead_code)]
fn struct_field_array(ctx: HostContext) -> [(); HOST_CONTEXT_FIELDS.len()] {
    let HostContext {
        cpu_model,
        cpu_vendor,
        total_memory_kib,
        hugepages_total,
        hugepages_free,
        hugepages_size_kib,
        thp_enabled,
        thp_defrag,
        sched_tunables,
        online_cpus,
        numa_nodes,
        cpufreq_governor,
        kernel_name,
        kernel_release,
        arch,
        kernel_cmdline,
        task_delayacct,
        config_task_xacct,
        heap_state,
    } = ctx;
    [
        drop_to_unit(cpu_model),
        drop_to_unit(cpu_vendor),
        drop_to_unit(total_memory_kib),
        drop_to_unit(hugepages_total),
        drop_to_unit(hugepages_free),
        drop_to_unit(hugepages_size_kib),
        drop_to_unit(thp_enabled),
        drop_to_unit(thp_defrag),
        drop_to_unit(sched_tunables),
        drop_to_unit(online_cpus),
        drop_to_unit(numa_nodes),
        drop_to_unit(cpufreq_governor),
        drop_to_unit(kernel_name),
        drop_to_unit(kernel_release),
        drop_to_unit(arch),
        drop_to_unit(kernel_cmdline),
        drop_to_unit(task_delayacct),
        drop_to_unit(config_task_xacct),
        drop_to_unit(heap_state),
    ]
}

/// Compile-time cardinality pin. The three surfaces that
/// must stay in lock-step when [`HostContext`] grows a
/// field:
///
///   - struct field count (source of truth),
///   - [`struct_field_array`] destructure + initializer
///     (three-way compile-time cross-check per its doc),
///   - [`HOST_CONTEXT_FIELDS`] name list (runtime
///     enumeration-coverage tests consume this).
///
/// The struct ↔ destructure link is compile-enforced by the
/// exhaustive pattern; the destructure ↔ array link is
/// compile-enforced by the return-type literal length. This
/// `const {}` block closes the remaining link by asserting
/// the name list length equals the array length. A mismatch
/// on any of the three surfaces aborts the build with a
/// named diagnostic.
#[allow(dead_code)]
const _HOST_CONTEXT_FIELD_COUNT_PIN: () = {
    assert!(
        HOST_CONTEXT_FIELDS.len() == 19,
        "HOST_CONTEXT_FIELDS cardinality drifted from the \
             HostContext struct — if a field was added, extend \
             HOST_CONTEXT_FIELDS, struct_field_array's destructure, \
             and struct_field_array's initializer together; then \
             bump this literal to the new field count",
    );
};

/// `format_human` must emit a row for every name in
/// [`HOST_CONTEXT_FIELDS`]. Runs against the default context
/// because every row renders regardless of value — the
/// enumeration check is about which NAMES land in the output,
/// not what values sit on the right of each colon.
///
/// Catches the "added a struct field, extended the
/// destructure, forgot the `row(&mut out, "foo", ...)` call"
/// regression that the compile-time destructure check does not
/// catch.
#[test]
fn format_human_renders_every_documented_field() {
    let out = HostContext::default().format_human();
    for key in HOST_CONTEXT_FIELDS {
        assert!(
            out.contains(&format!("{key}:")),
            "field '{key}' is declared in HOST_CONTEXT_FIELDS but does \
                 not appear in format_human output — either the `row()` \
                 call was forgotten or the field name drifted:\n{out}",
        );
    }
}

/// `diff` must emit a row for every name in
/// [`HOST_CONTEXT_FIELDS`] when the two contexts differ on
/// every field. Mirror of
/// `format_human_renders_every_documented_field` for the diff
/// surface — a field that is destructured on both halves but
/// never reaches a `row()` / per-key diff loop silently
/// disappears from `show-host` diff output.
///
/// Construction: fixture A is the default (every Option
/// `None`); fixture B flips each field to a distinct
/// populated value. The expected diff therefore names every
/// field.
#[test]
fn diff_renders_every_documented_field() {
    let a = HostContext::default();
    let heap = crate::host_heap::HostHeapState {
        active_bytes: Some(1),
        allocated_bytes: Some(2),
        resident_bytes: Some(3),
        mapped_bytes: Some(4),
        narenas: Some(1),
    };
    let mut tunables = BTreeMap::new();
    tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
    let mut b = HostContext {
        kernel_name: Some("Linux".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        arch: Some("x86_64".to_string()),
        cpu_model: Some("Example CPU".to_string()),
        cpu_vendor: Some("GenuineIntel".to_string()),
        total_memory_kib: Some(16_384_000),
        hugepages_total: Some(0),
        hugepages_free: Some(0),
        hugepages_size_kib: Some(2048),
        online_cpus: Some(8),
        numa_nodes: Some(1),
        thp_enabled: Some("always [madvise] never".to_string()),
        thp_defrag: Some("always [madvise] never".to_string()),
        kernel_cmdline: Some("preempt=lazy".to_string()),
        sched_tunables: Some(tunables),
        heap_state: Some(heap),
        task_delayacct: Some(DelayacctState::RuntimeOff),
        config_task_xacct: Some(XacctState::Off),
        ..Default::default()
    };
    b.cpufreq_governor.insert(0, "performance".to_string());

    let out = a.diff(&b);
    for key in HOST_CONTEXT_FIELDS {
        // Accept both forms that the diff renderer uses:
        //   `{key}:` — scalar/Option fields emitted by the
        //       shared `row()` helper;
        //   `{key}.` — structured/map fields (`cpufreq_governor`,
        //       `sched_tunables`) emitted as dotted per-key rows.
        let direct = format!("{key}:");
        let dotted = format!("{key}.");
        assert!(
            out.contains(&direct) || out.contains(&dotted),
            "field '{key}' is declared in HOST_CONTEXT_FIELDS but does \
                 not appear (as '{direct}' or '{dotted}') in diff output \
                 against a fully-populated partner — either the per-field \
                 row was forgotten or the field name drifted:\n{out}",
        );
    }
}

/// Snapshot-style pin of the label sequence `format_human`
/// emits. The order is load-bearing — downstream diff tools and
/// operator-eye scanning depend on a stable top-to-bottom field
/// ordering (uname → CPU → memory → hugepages → online_cpus →
/// NUMA → THP → kernel_cmdline → cpufreq_governor →
/// sched_tunables → heap_state). A silent reorder from a future
/// edit that shuffles the `row(...)` calls would slip past the
/// existing `.contains(...)` checks, which are order-blind.
/// This test fails the moment the sequence drifts; updating it
/// forces the author to acknowledge the reorder and
/// double-check that downstream consumers can absorb it.
#[test]
fn format_human_field_order_is_stable() {
    let out = HostContext::default().format_human();
    let labels: Vec<&str> = out
        .lines()
        .filter_map(|l| l.split(':').next())
        .filter(|s| !s.starts_with(' '))
        .collect();
    assert_eq!(
        labels,
        vec![
            "kernel_name",
            "kernel_release",
            "arch",
            "cpu_model",
            "cpu_vendor",
            "total_memory_kib",
            "hugepages_total",
            "hugepages_free",
            "hugepages_size_kib",
            "online_cpus",
            "numa_nodes",
            "thp_enabled",
            "thp_defrag",
            "kernel_cmdline",
            "task_delayacct",
            "config_task_xacct",
            "cpufreq_governor",
            "sched_tunables",
            "heap_state",
        ],
        "format_human field order drifted — if intentional, update \
             the expected vector and audit downstream diff/scan consumers",
    );
}

/// `format_human` on a default context must render every
/// field visibly — scalar/Option fields as `(unknown)`, and
/// collection-typed fields (cpufreq_governor, sched_tunables)
/// as `(empty)` when their default-construct is a zero-length
/// map. Silently suppressing absent or empty fields would
/// hide collection failures from the operator running
/// `cargo ktstr show-host` on a degraded host.
#[test]
fn format_human_default_renders_unknown_everywhere() {
    let out = HostContext::default().format_human();
    // Scalar / Option fields render as `(unknown)`.
    for key in [
        "kernel_name",
        "kernel_release",
        "arch",
        "cpu_model",
        "cpu_vendor",
        "total_memory_kib",
        "hugepages_total",
        "hugepages_free",
        "hugepages_size_kib",
        "online_cpus",
        "numa_nodes",
        "thp_enabled",
        "thp_defrag",
        "kernel_cmdline",
        "task_delayacct",
        "config_task_xacct",
        "sched_tunables",
        "heap_state",
    ] {
        assert!(
            out.contains(&format!("{key}: (unknown)")),
            "key '{key}' must render as (unknown) on a default context, got:\n{out}",
        );
    }
    // Collection-typed fields whose default is an EMPTY map
    // (not `None` — the struct field type is `BTreeMap`, not
    // `Option<BTreeMap>`). They render as `(empty)` to
    // distinguish "collected an empty set" from "not
    // collected". cpufreq_governor's type is `BTreeMap`,
    // so `Default::default()` gives an empty map.
    assert!(
        out.contains("cpufreq_governor: (empty)"),
        "cpufreq_governor must render as (empty) on default context, got:\n{out}",
    );
    assert!(
        out.ends_with('\n'),
        "format_human must end with a newline for direct print!() use",
    );
}

/// Populated fields render verbatim and `sched_tunables`
/// expands per-entry under the parent key.
#[test]
fn format_human_populated_shows_values_and_tunables() {
    let mut tunables = BTreeMap::new();
    tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
    tunables.insert("sched_min_granularity_ns".to_string(), "750000".to_string());
    let ctx = HostContext {
        kernel_name: Some("Linux".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        arch: Some("x86_64".to_string()),
        cpu_model: Some("Example CPU".to_string()),
        total_memory_kib: Some(16_384_000),
        sched_tunables: Some(tunables),
        kernel_cmdline: Some("preempt=lazy".to_string()),
        ..HostContext::default()
    };
    let out = ctx.format_human();
    assert!(out.contains("kernel_name: Linux"), "{out}");
    assert!(out.contains("kernel_release: 6.11.0"), "{out}");
    assert!(out.contains("cpu_model: Example CPU"), "{out}");
    assert!(out.contains("total_memory_kib: 16384000"), "{out}");
    assert!(out.contains("kernel_cmdline: preempt=lazy"), "{out}");
    assert!(out.contains("sched_tunables:\n"), "{out}");
    assert!(out.contains("  sched_migration_cost_ns = 500000"), "{out}");
    assert!(out.contains("  sched_min_granularity_ns = 750000"), "{out}");
    // Non-populated fields still render as (unknown) — show-host
    // never silently hides a field.
    assert!(out.contains("cpu_vendor: (unknown)"), "{out}");
    assert!(
        out.ends_with('\n'),
        "format_human output must terminate with a newline so the \
             next line the operator sees sits on its own row: {out:?}",
    );
}

/// `sched_tunables: Some(empty)` must not render as the generic
/// `(unknown)` — an empty map is a valid result (kernel with
/// no `sched_*` entries readable) and is distinguishable from
/// `None` (read_dir failure).
#[test]
fn format_human_sched_tunables_empty_vs_none() {
    let mut ctx = HostContext {
        sched_tunables: Some(BTreeMap::new()),
        ..Default::default()
    };
    let out_empty = ctx.format_human();
    assert!(
        out_empty.contains("sched_tunables: (empty)"),
        "empty map must render distinctly from None: {out_empty}",
    );
    assert!(
        out_empty.ends_with('\n'),
        "format_human with empty tunables must still end with a \
             newline: {out_empty:?}",
    );
    ctx.sched_tunables = None;
    let out_none = ctx.format_human();
    assert!(
        out_none.contains("sched_tunables: (unknown)"),
        "None map must render as (unknown): {out_none}",
    );
    assert!(
        out_none.ends_with('\n'),
        "format_human with no tunables must still end with a \
             newline: {out_none:?}",
    );
}

/// Two identical contexts diff to an empty string. This is the
/// signal `compare_partitions` uses to print `host: identical
/// between a and b` instead of an empty delta section.
#[test]
fn diff_identical_is_empty() {
    let ctx = HostContext {
        kernel_name: Some("Linux".to_string()),
        cpu_model: Some("Example CPU".to_string()),
        ..HostContext::default()
    };
    assert_eq!(ctx.diff(&ctx), "");
}

/// A single changed field produces a single `key: before →
/// after` line; unchanged fields are omitted so the operator
/// sees only what shifted.
#[test]
fn diff_single_field_surfaces_only_that_field() {
    let a = HostContext {
        kernel_cmdline: Some("preempt=lazy".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        ..HostContext::default()
    };
    let b = HostContext {
        kernel_cmdline: Some("preempt=full".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        ..HostContext::default()
    };
    let out = a.diff(&b);
    assert!(
        out.contains("kernel_cmdline: preempt=lazy → preempt=full"),
        "kernel_cmdline change must appear: {out}",
    );
    assert!(
        !out.contains("kernel_release"),
        "unchanged kernel_release must not appear: {out}",
    );
}

/// Per-CPU cpufreq_governor diff: unchanged CPUs omitted,
/// a CPU present in one side only renders as `(absent)`, a
/// value change renders as `old → new`. Mirrors the
/// `sched_tunables.<key>` per-key pattern so operators
/// running `stats compare` see governor churn per-CPU rather
/// than a collapsed "N entries changed" summary.
#[test]
fn diff_cpufreq_governor_both_empty_produces_no_lines() {
    let a = HostContext::default();
    let b = HostContext::default();
    let out = a.diff(&b);
    assert!(
        !out.contains("cpufreq_governor"),
        "two empty cpufreq_governor maps must not emit any \
             diff lines: {out}",
    );
}

#[test]
fn diff_cpufreq_governor_cpu_only_in_a_shows_absent() {
    let mut a_gov = BTreeMap::new();
    a_gov.insert(0, "performance".to_string());
    let a = HostContext {
        cpufreq_governor: a_gov,
        ..HostContext::default()
    };
    let b = HostContext::default();
    let out = a.diff(&b);
    assert!(
        out.contains("cpufreq_governor.cpu0: performance → (absent)"),
        "cpu0 removed must render as <value> → (absent): {out}",
    );
}

#[test]
fn diff_cpufreq_governor_value_change_shows_old_arrow_new() {
    let mut a_gov = BTreeMap::new();
    a_gov.insert(0, "performance".to_string());
    a_gov.insert(1, "powersave".to_string());
    let mut b_gov = BTreeMap::new();
    b_gov.insert(0, "schedutil".to_string());
    b_gov.insert(1, "powersave".to_string());
    let a = HostContext {
        cpufreq_governor: a_gov,
        ..HostContext::default()
    };
    let b = HostContext {
        cpufreq_governor: b_gov,
        ..HostContext::default()
    };
    let out = a.diff(&b);
    assert!(
        out.contains("cpufreq_governor.cpu0: performance → schedutil"),
        "cpu0 change must appear as old → new: {out}",
    );
    assert!(
        !out.contains("cpufreq_governor.cpu1"),
        "unchanged cpu1 (both powersave) must not appear: {out}",
    );
}

/// When the host exposes `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`,
/// `read_cpufreq_governors` must return a non-empty map with
/// non-empty trimmed governor values. Kernels compiled
/// without `CONFIG_CPU_FREQ` — most VMs, many containers —
/// have no `cpufreq/` directory per-CPU; treat that as a
/// skip rather than a failure.
#[cfg(target_os = "linux")]
#[test]
fn read_cpufreq_governors_returns_populated_map_when_sysfs_exposes_it() {
    use std::path::Path;
    let cpu0_gov = Path::new("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    if !cpu0_gov.exists() {
        eprintln!(
            "skipping: /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
                 absent (kernel without CONFIG_CPU_FREQ or VM without passthrough)"
        );
        return;
    }
    let m = read_cpufreq_governors();
    assert!(
        !m.is_empty(),
        "cpu0 scaling_governor is present on-disk — map must be \
             non-empty; got {m:?}"
    );
    for (cpu, gov) in &m {
        assert!(
            !gov.is_empty(),
            "cpu{cpu} governor string is empty after trim; sysfs \
                 usually writes non-empty content",
        );
    }
}

/// `None → Some(..)` renders as `(unknown) → <value>` so a
/// field that starts appearing in a newer run is not confused
/// with a field that was already present.
#[test]
fn diff_none_to_some_shows_unknown_arrow() {
    let a = HostContext::default();
    let b = HostContext {
        kernel_name: Some("Linux".to_string()),
        ..HostContext::default()
    };
    let out = a.diff(&b);
    assert!(
        out.contains("kernel_name: (unknown) → Linux"),
        "(unknown) → Linux must appear: {out}",
    );
}

/// Per-key `sched_tunables` diff: identical keys are omitted,
/// changed keys show old → new, and keys present on only one
/// side render as `(absent)`.
#[test]
fn diff_sched_tunables_per_key() {
    let mut am = BTreeMap::new();
    am.insert("sched_a".to_string(), "1".to_string());
    am.insert("sched_b".to_string(), "old".to_string());
    let mut bm = BTreeMap::new();
    bm.insert("sched_a".to_string(), "1".to_string());
    bm.insert("sched_b".to_string(), "new".to_string());
    bm.insert("sched_c".to_string(), "3".to_string());
    let a = HostContext {
        sched_tunables: Some(am),
        ..HostContext::default()
    };
    let b = HostContext {
        sched_tunables: Some(bm),
        ..HostContext::default()
    };
    let out = a.diff(&b);
    assert!(
        !out.contains("sched_tunables.sched_a"),
        "unchanged sched_a must not appear: {out}",
    );
    assert!(
        out.contains("sched_tunables.sched_b: old → new"),
        "changed sched_b must appear: {out}",
    );
    assert!(
        out.contains("sched_tunables.sched_c: (absent) → 3"),
        "new key sched_c must appear as (absent) → 3: {out}",
    );
}

/// `None vs Some(map)` at the outer `sched_tunables` level
/// still surfaces a line — otherwise a read_dir regression
/// would silently suppress the tunables section in compare
/// output. The Some side carries a cardinality sentinel so
/// the reader knows how much new data appeared.
#[test]
fn diff_sched_tunables_none_vs_some() {
    let mut m = BTreeMap::new();
    m.insert("sched_x".to_string(), "1".to_string());
    let a = HostContext::default();
    let b = HostContext {
        sched_tunables: Some(m),
        ..HostContext::default()
    };
    let out = a.diff(&b);
    assert!(
        out.contains("sched_tunables: (unknown) → (1 entry)"),
        "None → Some(1 entry) must surface cardinality: {out}",
    );
}

/// A field that transitions from `Some(value)` → `None`
/// (for example `kernel_cmdline` becoming unreadable in a
/// later run — `/proc/cmdline` normally always readable, but
/// a restricted procfs mount could hide it) must surface as
/// `<old> → (unknown)` so an
/// operator running `stats compare` sees the disappearance
/// explicitly.
#[test]
fn diff_some_to_none_shows_arrow_unknown() {
    let a = HostContext {
        kernel_release: Some("6.11.0".to_string()),
        ..HostContext::default()
    };
    let b = HostContext::default();
    let out = a.diff(&b);
    assert!(
        out.contains("kernel_release: 6.11.0 → (unknown)"),
        "Some → None must surface as <value> → (unknown): {out}",
    );
}

/// A per-key `sched_tunables` entry that exists in `a` but
/// not in `b` renders as `<value> → (absent)`, the mirror of
/// the `(absent) → <value>` case. Without this, a tunable
/// that was being overridden in the older run and reverted to
/// default in the newer run would silently disappear from the
/// diff.
#[test]
fn diff_sched_tunables_key_removed() {
    let mut am = BTreeMap::new();
    am.insert("sched_a".to_string(), "1".to_string());
    am.insert("sched_b".to_string(), "2".to_string());
    let mut bm = BTreeMap::new();
    bm.insert("sched_a".to_string(), "1".to_string());
    let a = HostContext {
        sched_tunables: Some(am),
        ..HostContext::default()
    };
    let b = HostContext {
        sched_tunables: Some(bm),
        ..HostContext::default()
    };
    let out = a.diff(&b);
    assert!(
        !out.contains("sched_tunables.sched_a"),
        "unchanged sched_a must not appear: {out}",
    );
    assert!(
        out.contains("sched_tunables.sched_b: 2 → (absent)"),
        "removed sched_b must surface as <value> → (absent): {out}",
    );
}

// ------------------------------------------------------------
// read_trimmed_sysfs — IO-wrapper edge cases. `parse_trimmed`
// is tested separately; these tests exercise the `read_to_string
// + parse_trimmed` chain end-to-end against real files via
// `tempfile::NamedTempFile`.
// ------------------------------------------------------------

/// Nonexistent path → `None`. `read_to_string` returns `ENOENT`;
/// `.ok()` converts to `None`; the `and_then` short-circuits.
/// Guards against a regression that re-introduces `unwrap()`
/// on the read result.
///
/// The "nonexistent" path is constructed under a fresh
/// `TempDir` (unique per invocation, auto-cleaned on drop)
/// rather than a fixed name under `std::env::temp_dir()` —
/// the latter would race with a concurrent run of the same
/// test from a parallel test runner or cargo-watch session.
#[test]
fn read_trimmed_sysfs_missing_file_returns_none() {
    let scratch = tempfile::TempDir::new().expect("create scratch temp dir");
    let missing = scratch.path().join("nonexistent-target");
    assert!(read_trimmed_sysfs(&missing).is_none());
}

/// Whitespace-only file → `None`. `str::trim` leaves the empty
/// string; `parse_trimmed` catches that and returns `None`.
/// A kernel sysfs file that transiently reads as just `"\n"`
/// must map to `None` rather than `Some("")`.
#[test]
fn read_trimmed_sysfs_whitespace_only_returns_none() {
    let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
    std::io::Write::write_all(&mut f, b"  \n\t \r\n  ").expect("write whitespace");
    assert!(read_trimmed_sysfs(f.path()).is_none());
}

/// Populated file → `Some(trimmed)`. Exercises the full IO +
/// trim chain against a realistic sysfs shape (`value\n`).
#[test]
fn read_trimmed_sysfs_populated_file_returns_trimmed_content() {
    let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
    std::io::Write::write_all(&mut f, b"madvise\n").expect("write content");
    assert_eq!(read_trimmed_sysfs(f.path()).as_deref(), Some("madvise"));
}

/// Bracketed-selection THP shape round-trips through the IO
/// wrapper. `parse_trimmed_preserves_bracketed_thp` already pins
/// the pure trim-preservation; this test walks the whole IO +
/// trim chain so a regression that double-trims or parses the
/// brackets is caught at the wrapper boundary.
#[test]
fn read_trimmed_sysfs_preserves_thp_bracket_selection() {
    let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
    std::io::Write::write_all(&mut f, b"always [madvise] never\n").expect("write");
    assert_eq!(
        read_trimmed_sysfs(f.path()).as_deref(),
        Some("always [madvise] never"),
    );
}

/// `read_sched_tunables_from` happy path: only regular files whose
/// names start with `sched_` are included, non-prefix files are
/// ignored, subdirectories are filtered by the `is_file` guard,
/// and each value is trimmed by the existing `read_trimmed_sysfs`
/// hop. Drives the path-parameterized seam against a controlled
/// tempdir so the walk + filter + read pipeline is exercised end
/// to end without touching `/proc`.
#[test]
fn read_sched_tunables_from_filters_and_trims() {
    let tmp = tempfile::TempDir::new().expect("create tempdir");
    let dir = tmp.path();
    std::fs::write(dir.join("sched_foo"), b"42\n").expect("write sched_foo");
    std::fs::write(dir.join("sched_bar"), b"1\n").expect("write sched_bar");
    // Non-`sched_` prefix — filtered out by the name check.
    std::fs::write(dir.join("not_sched_baz"), b"99\n").expect("write not_sched_baz");
    // Subdirectory whose name starts with `sched_` — filtered
    // out by the `is_file` guard.
    std::fs::create_dir(dir.join("sched_subdir")).expect("create sched_subdir");

    let out = read_sched_tunables_from(dir).expect("walk must succeed on readable dir");
    assert_eq!(out.len(), 2, "expected only two sched_* files, got {out:?}");
    assert_eq!(out.get("sched_foo").map(String::as_str), Some("42"));
    assert_eq!(out.get("sched_bar").map(String::as_str), Some("1"));
    assert!(
        !out.contains_key("not_sched_baz"),
        "non-sched_ prefix must be filtered out"
    );
    assert!(
        !out.contains_key("sched_subdir"),
        "subdirectories must be filtered by is_file"
    );
}

// ------------------------------------------------------------
// count_numa_nodes_in_topology — UMA fallback + sparse / dense
// dedup paths. Pure logic; the IO-reading wrapper
// `count_numa_nodes_via_topology` is left untested here (that
// was the tradeoff in the seam extraction — the IO path just
// delegates to this helper after a sysfs probe).
// ------------------------------------------------------------

/// Empty `cpu_to_node` map → `1`. This is the UMA fallback
/// branch: every Linux system has at least one NUMA node, so
/// returning zero would misrepresent the topology. Guarded
/// against a refactor that removes the `is_empty` check and
/// lets `BTreeSet::len()` return 0.
#[test]
fn count_numa_nodes_in_topology_empty_returns_one() {
    let topo = crate::vmm::host_topology::HostTopology {
        llc_groups: Vec::new(),
        online_cpus: Vec::new(),
        cpu_to_node: std::collections::HashMap::new(),
        host_node_llcs: std::collections::BTreeMap::new(),
    };
    assert_eq!(count_numa_nodes_in_topology(&topo), 1);
}

/// Single-node: every CPU maps to node 0. Dedup produces a
/// set with one entry. Pinned separately from the empty-map
/// case because the code path is different — `is_empty` is
/// false here, so the `BTreeSet` branch runs and must still
/// return 1.
#[test]
fn count_numa_nodes_in_topology_single_node() {
    let mut cpu_to_node = std::collections::HashMap::new();
    for cpu in 0..8 {
        cpu_to_node.insert(cpu, 0);
    }
    let topo = crate::vmm::host_topology::HostTopology {
        llc_groups: Vec::new(),
        online_cpus: (0..8).collect(),
        cpu_to_node,
        host_node_llcs: std::collections::BTreeMap::new(),
    };
    assert_eq!(count_numa_nodes_in_topology(&topo), 1);
}

/// Two-node split (CPUs 0-3 → node 0, CPUs 4-7 → node 1).
/// The common post-fix case a sidecar host-context snapshot
/// needs to report correctly.
#[test]
fn count_numa_nodes_in_topology_two_nodes() {
    let mut cpu_to_node = std::collections::HashMap::new();
    for cpu in 0..4 {
        cpu_to_node.insert(cpu, 0);
    }
    for cpu in 4..8 {
        cpu_to_node.insert(cpu, 1);
    }
    let topo = crate::vmm::host_topology::HostTopology {
        llc_groups: Vec::new(),
        online_cpus: (0..8).collect(),
        cpu_to_node,
        host_node_llcs: std::collections::BTreeMap::new(),
    };
    assert_eq!(count_numa_nodes_in_topology(&topo), 2);
}

/// Sparse node IDs — `{0, 2, 5}` with non-contiguous numbering
/// (e.g. a CXL-host topology where some nodes are memory-only).
/// `BTreeSet::from_iter` dedups on insert, so the count is the
/// number of distinct IDs, NOT `max_id + 1`.
#[test]
fn count_numa_nodes_in_topology_sparse_ids() {
    let mut cpu_to_node = std::collections::HashMap::new();
    cpu_to_node.insert(0, 0);
    cpu_to_node.insert(1, 2);
    cpu_to_node.insert(2, 5);
    cpu_to_node.insert(3, 0); // duplicate of cpu 0's node
    let topo = crate::vmm::host_topology::HostTopology {
        llc_groups: Vec::new(),
        online_cpus: vec![0, 1, 2, 3],
        cpu_to_node,
        host_node_llcs: std::collections::BTreeMap::new(),
    };
    assert_eq!(
        count_numa_nodes_in_topology(&topo),
        3,
        "sparse IDs {{0, 2, 5}} must count as 3, not max_id+1",
    );
}

/// Pin all three caching invariants with a direct call-count
/// probe:
///
/// 1. `compute_static_host_info` runs at MOST once per process
///    — the `OnceLock::get_or_init` contract. Across N repeated
///    `collect_host_context()` calls, the delta must stay ≤ 1
///    (the first call from-cold executes the closure; every
///    subsequent call hits the cache).
/// 2. `read_meminfo` runs EXACTLY N times across N calls — one
///    read per `collect_host_context` invocation, regardless of
///    cache state. The cold path no longer double-reads
///    meminfo (the dedup shares the parsed struct between the
///    init closure and the per-call path); this test pins the
///    dedup so a regression that re-adds a second read inside
///    `compute_static_host_info` trips the assertion.
/// 3. `read_cpufreq_governors` runs at MOST once per process —
///    the [`CPUFREQ_GOVERNORS`] `OnceLock::get_or_init`
///    contract. Across N repeated `collect_host_context()`
///    calls, the delta must stay ≤ 1. On a 256-CPU host this
///    collapses up to N × 256 sysfs reads into 256.
/// 4. Cold-init anchors: if a cache was not yet populated when
///    the test started, exactly one underlying read must run
///    during this test (one for `compute_static_host_info`, one
///    for `read_cpufreq_governors`).
///
/// Deltas (`load() - before-snapshot`) absorb pre-population
/// from sibling tests: the test is robust to execution order.
///
/// # Nextest subprocess-isolation assumption
///
/// The before-snapshot / after-delta arithmetic assumes no
/// **other** concurrent test inside the same process mutates
/// the counters mid-run. ktstr's test suite is driven by
/// `cargo nextest run`, which spawns a fresh subprocess per
/// test by default — so each test sees a freshly-initialized
/// process with its own counters, and the only writers to
/// `STATIC_INIT_CALLS` / `MEMINFO_READ_CALLS` /
/// `CPUFREQ_GOVERNORS_READ_CALLS` during this test's window
/// are its own five `collect_host_context()` calls. Under
/// `cargo test` (shared-process, thread-parallel) a sibling
/// test calling `collect_host_context()` in parallel would
/// skew the deltas. The project rule "always use
/// `cargo nextest run`, never `cargo test`" is what keeps this
/// assumption load-bearing; a future migration away from
/// nextest would need to re-assess this test's atomic-delta
/// scheme (likely via per-test-thread counters or a mutex
/// around the whole call window).
#[cfg(target_os = "linux")]
#[test]
fn collect_host_context_call_counts_match_caching_invariants() {
    use std::sync::atomic::Ordering;
    const N: usize = 5;

    let static_was_populated_pre = STATIC_HOST_INFO.get().is_some();
    let cpufreq_was_populated_pre = CPUFREQ_GOVERNORS.get().is_some();
    let init_before = STATIC_INIT_CALLS.load(Ordering::Relaxed);
    let meminfo_before = MEMINFO_READ_CALLS.load(Ordering::Relaxed);
    let cpufreq_before = CPUFREQ_GOVERNORS_READ_CALLS.load(Ordering::Relaxed);

    for _ in 0..N {
        let _ = collect_host_context();
    }

    let init_delta = STATIC_INIT_CALLS.load(Ordering::Relaxed) - init_before;
    let meminfo_delta = MEMINFO_READ_CALLS.load(Ordering::Relaxed) - meminfo_before;
    let cpufreq_delta = CPUFREQ_GOVERNORS_READ_CALLS.load(Ordering::Relaxed) - cpufreq_before;

    assert!(
        init_delta <= 1,
        "compute_static_host_info must run at most once across {N} collect_host_context calls, ran {init_delta}",
    );
    assert_eq!(
        meminfo_delta, N,
        "read_meminfo must run exactly {N} times across {N} collect_host_context calls, ran {meminfo_delta} — the dedup would regress if this trips",
    );
    assert!(
        cpufreq_delta <= 1,
        "read_cpufreq_governors must run at most once across {N} collect_host_context calls, ran {cpufreq_delta} — a regression that bypassed the CPUFREQ_GOVERNORS cache would trip this",
    );

    if !static_was_populated_pre {
        assert_eq!(
            init_delta, 1,
            "cold-init anchor: compute_static_host_info must run exactly once on the populate path, not {init_delta}",
        );
    }
    if !cpufreq_was_populated_pre {
        assert_eq!(
            cpufreq_delta, 1,
            "cold-init anchor: read_cpufreq_governors must run exactly once on the populate path, not {cpufreq_delta}",
        );
    }

    assert!(
        STATIC_HOST_INFO.get().is_some(),
        "STATIC_HOST_INFO must be populated after at least one collect_host_context call",
    );
    assert!(
        CPUFREQ_GOVERNORS.get().is_some(),
        "CPUFREQ_GOVERNORS must be populated after at least one collect_host_context call",
    );
}

/// `count_numa_nodes_in_topology` counts the cardinality of
/// distinct values in [`HostTopology::cpu_to_node`] — the
/// "CPU-bearing nodes" count, and nothing else. Memory-only
/// NUMA nodes (CXL / Intel Optane / persistent memory tiers)
/// have no CPUs by definition and are structurally
/// unrepresentable in the current [`HostTopology`]: the struct
/// has no "all nodes" field populated from
/// `/sys/devices/system/node/*` independently of the CPU
/// mapping. From the counter's perspective a memory-only node
/// and a non-existent node are indistinguishable — both are
/// simply missing from `cpu_to_node`.
///
/// **What this test pins is narrow**: the counter's only
/// source is `cpu_to_node`. A regression that added a parallel
/// source (e.g. an `all_nodes: Vec<u32>` field fed from
/// `/sys/...`) and summed it into the count would inflate the
/// "CPUs per node" denominator for every downstream consumer —
/// cgroup cpuset assignments, scheduler placement, and the
/// NUMA memory-policy validator in
/// [`ops::validate_mempolicy_cpuset`] — all of which are
/// CPU-keyed and would quietly break under an inflated count.
/// The exclusion is therefore by construction (the parallel
/// field doesn't exist), not by active filtering.
///
/// Fixture: 4 CPUs mapped across nodes 0 and 1, so
/// `cpu_to_node.values()` has 2 distinct entries. The assertion
/// demands `count == 2`. A future impl that introduced a second
/// source must either (a) audit all CPU-keyed consumers at the
/// same time and update this doc to match, or (b) leave this
/// counter cpu_to_node-driven and add a separate
/// `count_all_nodes_including_memory_only` helper with its own
/// coverage. The inline comment at the "absent node id" line
/// carries the same contract for readers browsing the test
/// body.
#[test]
fn count_numa_nodes_in_topology_excludes_memory_only_nodes() {
    let mut cpu_to_node = std::collections::HashMap::new();
    cpu_to_node.insert(0, 0);
    cpu_to_node.insert(1, 0);
    cpu_to_node.insert(2, 1);
    cpu_to_node.insert(3, 1);
    // Node id 2 intentionally absent from cpu_to_node — it is
    // the memory-only tier under test. The function has no
    // other channel to learn about node 2, so a future change
    // that adds awareness of memory-only nodes (via a separate
    // field) would need to opt-in explicitly — this test pins
    // the current silent-exclusion contract.
    let topo = crate::vmm::host_topology::HostTopology {
        llc_groups: Vec::new(),
        online_cpus: vec![0, 1, 2, 3],
        cpu_to_node,
        host_node_llcs: std::collections::BTreeMap::new(),
    };
    assert_eq!(
        count_numa_nodes_in_topology(&topo),
        2,
        "memory-only nodes must not inflate the CPU-bearing node count",
    );
}

/// `parse_bracketed_active_policy` extracts the content between
/// the first `[` and subsequent `]`. Covers the canonical THP-
/// enabled menu shape `"always [madvise] never"`.
#[test]
fn parse_bracketed_active_policy_middle_selection() {
    assert_eq!(
        parse_bracketed_active_policy("always [madvise] never"),
        Some("madvise"),
    );
}

/// Leading-slot selection: the bracket is at the front of the
/// menu, and the extracted token must not be empty.
#[test]
fn parse_bracketed_active_policy_leading_selection() {
    assert_eq!(
        parse_bracketed_active_policy("[always] madvise never"),
        Some("always"),
    );
}

/// Trailing-slot selection covers the other edge.
#[test]
fn parse_bracketed_active_policy_trailing_selection() {
    assert_eq!(
        parse_bracketed_active_policy("always madvise [never]"),
        Some("never"),
    );
}

/// THP-defrag menu format is longer but uses the same bracket
/// convention. Pins that the multi-word hyphenated option
/// `defer+madvise` round-trips correctly — the parser doesn't
/// split on `+` or whitespace inside the brackets.
#[test]
fn parse_bracketed_active_policy_thp_defrag_hyphenated() {
    assert_eq!(
        parse_bracketed_active_policy("always defer [defer+madvise] madvise never",),
        Some("defer+madvise"),
    );
}

/// No brackets at all → None. Guards against a kernel whose
/// THP-enabled output lost the brackets entirely; downstream
/// tooling sees the raw menu via `thp_enabled` and this helper
/// returns None rather than inventing a fake active value.
#[test]
fn parse_bracketed_active_policy_no_brackets_is_none() {
    assert_eq!(parse_bracketed_active_policy("always madvise never"), None);
}

/// Empty string → None. Boundary.
#[test]
fn parse_bracketed_active_policy_empty_is_none() {
    assert_eq!(parse_bracketed_active_policy(""), None);
}

/// Unbalanced `[` with no `]` → None. A malformed sysfs read
/// (truncated by a concurrent write) must not panic or return
/// a half-parsed substring.
#[test]
fn parse_bracketed_active_policy_unclosed_bracket_is_none() {
    assert_eq!(parse_bracketed_active_policy("always [madvise never"), None);
}

/// Unopened `]` with no preceding `[` → None. The scanner
/// requires a leading `[` before it looks for `]`; a stray
/// closing bracket mid-string (e.g. a malformed menu written
/// as `"always madvise] never"`) must not be misread as a
/// zero-length active token.
#[test]
fn parse_bracketed_active_policy_unopened_bracket_is_none() {
    assert_eq!(parse_bracketed_active_policy("always madvise] never"), None);
}

/// Multiple `[..]` pairs → the FIRST pair wins. Pins the
/// first-bracket-wins invariant documented on the parser so a
/// future refactor that switched to "last wins" or merged the
/// tokens would trip this test. The kernel only emits one pair
/// in practice; this test exists to lock the degenerate-input
/// behavior, not to describe reality.
#[test]
fn parse_bracketed_active_policy_multiple_pairs_first_wins() {
    assert_eq!(
        parse_bracketed_active_policy("[always] [never]"),
        Some("always"),
    );
}

/// Nested / doubled brackets truncate at the FIRST `]` after the
/// first `[`. The scanner does not balance brackets — it's a
/// two-step `find('[')` → `find(']')` on the remaining slice.
/// For a fixture like `"[a[b]c]"` the scan opens at index 0,
/// the remainder is `"a[b]c]"`, and the first `]` in that
/// remainder sits at index 3, so the returned slice is `"a[b"`.
/// Kernel-emitted menus never produce nested brackets; this
/// test pins the degenerate-input behavior so a future refactor
/// to bracket-balancing (or an off-by-one on the inner search)
/// cannot silently change the output for malformed fixtures or
/// hand-written test menus.
#[test]
fn parse_bracketed_active_policy_nested_brackets_truncate_at_inner_close() {
    // Inner pair wholly inside the outer pair — the scan stops
    // at the inner `]` and returns the partial token.
    assert_eq!(
        parse_bracketed_active_policy("[a[b]c]"),
        Some("a[b"),
        "nested-bracket fixture must truncate at the first inner `]`",
    );
    // Unpaired nest: `[` appears twice, only one `]` follows.
    // Same truncation rule applies — the first `]` closes the
    // scan, regardless of how many `[` it crossed.
    assert_eq!(
        parse_bracketed_active_policy("[a[b] c"),
        Some("a[b"),
        "unpaired nest must still close at the first inner `]`",
    );
    // Nested pair in the prose prefix preceding the real active
    // token: because the scanner picks the FIRST `[`, the
    // bracketed token in the prefix wins — even if it's the
    // literal text the menu is commenting on rather than the
    // kernel's own selection. Documents the "first-bracket-wins"
    // rule's interaction with prefix text.
    assert_eq!(
        parse_bracketed_active_policy("prefix [lit] then [active] tail"),
        Some("lit"),
        "first-bracket-wins overrides any later 'real' active token",
    );
}

/// `HostContext::thp_enabled_active` routes through the parser
/// and returns `None` when the field is absent. Pins the
/// method-level contract alongside the parser-level tests.
#[test]
fn host_context_thp_active_methods_extract_bracketed_choice() {
    let mut ctx = HostContext::test_fixture();
    // Fixture defaults: "always [madvise] never" / "... [madvise] ...".
    assert_eq!(ctx.thp_enabled_active(), Some("madvise"));
    assert_eq!(ctx.thp_defrag_active(), Some("madvise"));
    ctx.thp_enabled = None;
    assert_eq!(ctx.thp_enabled_active(), None);
    ctx.thp_defrag = Some("no brackets here".to_string());
    assert_eq!(ctx.thp_defrag_active(), None);
}

/// `parse_kconfig_xacct` classifies the three CONFIG_TASK_XACCT line
/// states from decompressed kconfig text (the gz I/O is a thin wrapper
/// around this pure parser).
#[test]
fn parse_kconfig_xacct_classifies_the_three_states() {
    assert_eq!(
        parse_kconfig_xacct("CONFIG_FOO=y\nCONFIG_TASK_XACCT=y\nCONFIG_BAR=m\n"),
        XacctState::On,
    );
    assert_eq!(
        parse_kconfig_xacct("CONFIG_FOO=y\n# CONFIG_TASK_XACCT is not set\n"),
        XacctState::Off,
    );
    // Symbol absent entirely (a partial/foreign config) -> Unknown,
    // which gates as measured so no false "absent" verdict is produced.
    assert_eq!(
        parse_kconfig_xacct("CONFIG_FOO=y\nCONFIG_BAR=m\n"),
        XacctState::Unknown,
    );
    // Whole-line match: a substring inside another symbol must not
    // false-match the `=y` case.
    assert_eq!(
        parse_kconfig_xacct("CONFIG_TASK_XACCT_EXTRA=y\n"),
        XacctState::Unknown,
    );
}

/// `read_task_delayacct_from` maps the sysctl file to the three
/// delay-accounting states: absent file -> ConfigOff (the sysctl is
/// registered only under CONFIG_TASK_DELAY_ACCT), "0" -> RuntimeOff,
/// "1" -> On. Drives the production seam against a tempdir.
#[test]
fn read_task_delayacct_from_maps_the_three_states() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("task_delayacct");

    // Absent file -> ConfigOff.
    assert_eq!(read_task_delayacct_from(&path), DelayacctState::ConfigOff);

    // "0" -> RuntimeOff (built in, runtime toggle off).
    std::fs::write(&path, "0\n").expect("write");
    assert_eq!(read_task_delayacct_from(&path), DelayacctState::RuntimeOff);

    // "1" -> On.
    std::fs::write(&path, "1\n").expect("write");
    assert_eq!(read_task_delayacct_from(&path), DelayacctState::On);
}

/// `taskstats_active_from` reduces the two probe enums to the per-sub-family
/// active flags — the counter-semantics crux. cpu_delay is active unless
/// ConfigOff (it SURVIVES RuntimeOff — the gate is `!= ConfigOff`, NOT `== On`);
/// delay_block needs On; xacct is active unless Off (Unknown -> active to avoid a
/// false absent). All 3x3 combinations pinned.
#[test]
fn taskstats_active_from_reduces_every_combination() {
    use DelayacctState::{ConfigOff, On as DOn, RuntimeOff};
    use XacctState::{Off as XOff, On as XOn, Unknown};
    // (delayacct, xacct) -> (cpu_delay, delay_block, xacct)
    let cases = [
        (ConfigOff, XOff, (false, false, false)),
        (ConfigOff, XOn, (false, false, true)),
        (ConfigOff, Unknown, (false, false, true)),
        (RuntimeOff, XOff, (true, false, false)),
        (RuntimeOff, XOn, (true, false, true)),
        (RuntimeOff, Unknown, (true, false, true)),
        (DOn, XOff, (true, true, false)),
        (DOn, XOn, (true, true, true)),
        (DOn, Unknown, (true, true, true)),
    ];
    for (d, x, (cpu, blk, xa)) in cases {
        let a = taskstats_active_from(d, x);
        assert_eq!(
            (a.cpu_delay, a.delay_block, a.xacct),
            (cpu, blk, xa),
            "taskstats_active_from({d:?}, {x:?})",
        );
    }
    // The crux, called out explicitly: RuntimeOff keeps cpu_delay measurable (the
    // sched_info CPU block survives the toggle) but kills the resource-wait block.
    let r = taskstats_active_from(RuntimeOff, XOn);
    assert!(
        r.cpu_delay && !r.delay_block,
        "cpu_delay must survive RuntimeOff while delay_block must not",
    );
}

/// `read_config_task_xacct_from` folds BOTH failure branches — an unreadable /
/// absent file, and present-but-non-gzip (or truncated) content the GzDecoder
/// rejects — to `XacctState::Unknown`, the no-config-exposed fallback that gates
/// as active to avoid a false absent.
#[test]
fn read_config_task_xacct_from_unreadable_or_non_gz_is_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Absent file -> read error -> Unknown.
    let absent = dir.path().join("config.gz");
    assert_eq!(read_config_task_xacct_from(&absent), XacctState::Unknown);
    // Present but not gzip (raw text, no gzip magic) -> GzDecoder error -> Unknown.
    let raw = dir.path().join("config-raw.gz");
    std::fs::write(&raw, b"CONFIG_TASK_XACCT=y\nnot actually gzip\n").expect("write");
    assert_eq!(read_config_task_xacct_from(&raw), XacctState::Unknown);
}
