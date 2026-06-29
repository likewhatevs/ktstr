//! Host runtime state captured at sidecar-write time.
//!
//! [`HostContext`] is a snapshot of the host running the tool:
//! kernel release, CPU identity, memory size, hugepages config,
//! transparent-hugepage policy, kernel scheduler tunables, NUMA
//! node count, and kernel cmdline. Static fields (CPU identity,
//! total memory, hugepage size, NUMA count, uname triple,
//! per-CPU cpufreq governor) are memoized in [`OnceLock`] across
//! the process; dynamic fields (sched tunables, hugepages totals,
//! THP policy, cmdline) are re-read on every call so run-time
//! sysctl changes or hugepage reservations between tests are not
//! hidden by the cache.
//!
//! ## Static-cache staleness under hotplug
//!
//! The static-field cache pins the first snapshot it observes for
//! the life of the process. This is OUR invariant, not the
//! kernel's: `/proc/meminfo`'s `MemTotal`,
//! `/sys/devices/system/node/*`, and the `uname()` return all
//! update live when memory or NUMA hotplug fires, and a freshly-
//! started process would pick up the new values on its next
//! collect call. It is `STATIC_HOST_INFO`'s `OnceLock` that
//! binds a single read for the process lifetime — not any
//! kernel-side caching.
//!
//! So on a host where CPU / NUMA / memory hotplug fires between
//! two collect calls in the same process, `HostContext` continues
//! to report the pre-hotplug values — `total_memory_kib` stays at
//! the original snapshot, `numa_nodes` does not reflect an
//! added/removed node. `arch` is the only field genuinely immune
//! (a reboot is required to change architecture).
//!
//! `cpufreq_governor` is similarly pinned: the per-CPU
//! `scaling_governor` map is read once on first
//! [`collect_host_context`] call and reused thereafter. A test
//! that writes to `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
//! mid-process will not see the post-write value reflected in
//! later snapshots. Governor changes are rare (they typically
//! happen at boot via `cpupower`, systemd unit, or kernel default)
//! and the cache trades that rare-mutation visibility for
//! eliminating up to N × M sysfs reads per process (N = online
//! CPUs, M = `collect_host_context` invocations).
//!
//! Tests that need live-updated values must either (a) avoid
//! reading HostContext after the hotplug event, or (b) restart
//! the process to force a fresh `OnceLock` population. No
//! `reset` hook is exposed in production; the `#[cfg(test)]`-only
//! reset machinery is for unit tests, not runtime recapture.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Host-level runtime state snapshot attached to each
/// [`SidecarResult`](crate::test_support::SidecarResult). Every
/// field is optional so a partial read (missing /proc entry,
/// permission denied, parse failure) still records the fields that
/// did succeed instead of dropping the whole snapshot.
///
/// # Constructing instances in tests
///
/// `HostContext` is `#[non_exhaustive]` — see
/// [`crate::non_exhaustive`] for the cross-crate construction and
/// pattern-match rules shared by every such type in the crate. The
/// concrete pattern for `HostContext` is to start from a [`Default`]
/// instance and mutate fields:
///
/// ```
/// use ktstr::prelude::HostContext;
/// let mut ctx = HostContext::default();
/// ctx.cpu_model = Some("Test CPU".to_string());
/// ctx.numa_nodes = Some(2);
/// ```
///
/// For tests that want a populated baseline (non-trivial defaults
/// for every field) instead of `Default`'s all-`None` minimum, start
/// from [`HostContext::test_fixture`] and mutate from there.
///
/// # Partial-read round-trip
///
/// Fields representing producer-time partial-read outcomes use
/// `serde(default, skip_serializing_if = ...)` so the absent
/// state round-trips through the sidecar JSON — `None` for the
/// `Option<T>` fields paired with `Option::is_none`, empty for
/// the `cpufreq_governor` `BTreeMap<usize, String>` paired with
/// `BTreeMap::is_empty`. A producer-time partial read (missing
/// `/proc` entry, permission denied, parse failure) lands at the
/// absent state, gets omitted on serialize, and deserializes back
/// to the same absent state. The pattern exists for that
/// producer-side partial-population path, not for cross-binary-
/// version compatibility. Per the pre-1.0 sidecar-disposable
/// rule, a sidecar written by a different binary version may
/// fail to deserialize when the schema has diverged — re-run the
/// test to regenerate it with the current schema.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HostContext {
    /// CPU model string — the `model name` line of `/proc/cpuinfo`.
    /// Single value (first processor entry) since heterogeneous
    /// CPU models on a single host are rare enough that the
    /// extra complexity is not worth carrying.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    /// CPU vendor ID — the `vendor_id` line of `/proc/cpuinfo`
    /// (e.g. `GenuineIntel`, `AuthenticAMD`). On ARM64,
    /// `/proc/cpuinfo` uses `CPU implementer` instead of
    /// `vendor_id`, so this field is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_vendor: Option<String>,
    /// Total physical memory in KiB — `MemTotal:` from
    /// `/proc/meminfo`. The kernel labels the value `kB` but the
    /// scale is 1024 bytes (KiB); the field name uses the
    /// unambiguous IEC binary unit so the sidecar reader does not
    /// need to guess the scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_memory_kib: Option<u64>,
    /// Configured huge pages — `HugePages_Total` from `/proc/meminfo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hugepages_total: Option<u64>,
    /// Free huge pages — `HugePages_Free` from `/proc/meminfo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hugepages_free: Option<u64>,
    /// Hugepage size in KiB — `Hugepagesize:` from `/proc/meminfo`
    /// (labeled `kB` in the file; the scale is 1024 bytes / KiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hugepages_size_kib: Option<u64>,
    /// Active THP policy — content of
    /// `/sys/kernel/mm/transparent_hugepage/enabled` with the
    /// bracketed selection preserved verbatim (e.g.
    /// `"always [madvise] never"`). Trimmed of leading and
    /// trailing whitespace by `read_trimmed_sysfs`, so the trailing
    /// newline that sysfs appends does not appear in the captured
    /// value. Stored as-read rather than parsed because the bracket
    /// is the meaningful part and downstream tooling may want the
    /// full menu too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thp_enabled: Option<String>,
    /// Active THP defrag policy — content of
    /// `/sys/kernel/mm/transparent_hugepage/defrag`, bracket
    /// preserved. Trimmed of leading and trailing whitespace by
    /// `read_trimmed_sysfs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thp_defrag: Option<String>,
    /// `/proc/sys/kernel/sched_*` tunables. Keys are the leaf
    /// basename (e.g. `sched_migration_cost_ns`); values are the
    /// file content trimmed of leading and trailing whitespace
    /// (internal whitespace preserved — `read_trimmed_sysfs` uses
    /// `str::trim`, which only strips edges). Every current
    /// `sched_*` tunable is a scalar, but a future kernel that
    /// exposes a multi-line tunable would land here as a
    /// multi-line `String`. `None` when the `read_dir` of
    /// `/proc/sys/kernel` fails; empty map when the directory is
    /// readable but contains no entries starting with `sched_`
    /// (or all such entries fail the per-file read or trim to
    /// empty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_tunables: Option<BTreeMap<String, String>>,
    /// Number of online host CPUs — `HostTopology::online_cpus.len()`
    /// from the same `from_sysfs` probe that drives `numa_nodes`.
    /// `None` when the topology probe fails. Captured as a discrete
    /// field so downstream consumers (sidecar readers, scheduler
    /// regression dashboards) don't need to reconstruct a
    /// HostTopology just to learn the CPU count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub online_cpus: Option<usize>,
    /// Count of NUMA nodes — derived from
    /// `HostTopology::from_sysfs` (the `cpu_to_node` map's distinct
    /// value count). `None` when the topology probe itself fails so
    /// "unknown" is distinguishable from a populated result. A probe
    /// that succeeds but reports no CPU→node entries defaults to
    /// `Some(1)` because every Linux system has at least one NUMA
    /// node — see `count_numa_nodes_in_topology` for the full
    /// rationale (in production, empty `cpu_to_node` from a
    /// successful probe cannot happen because `TestTopology::from_system`
    /// bails on zero online CPUs; the `.max(1)` floor is a guard
    /// for synthetic/test topologies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa_nodes: Option<usize>,
    /// Per-CPU scaling_governor string, keyed by CPU id. Read
    /// from `/sys/devices/system/cpu/cpu{N}/cpufreq/scaling_governor`
    /// for every online CPU. Value is the trimmed governor name
    /// as written by the kernel (e.g. `"performance"`,
    /// `"powersave"`, `"schedutil"`, `"ondemand"`).
    ///
    /// Per-CPU granularity matters: heterogeneous hosts (big.LITTLE,
    /// P/E cores) can carry different governors on different CPUs,
    /// and a scheduler micro-benchmark landing on a `powersave`
    /// CPU sees 2× the latency of one landing on a `performance`
    /// CPU. A run-level single-governor field would average this
    /// out and hide the variance.
    ///
    /// Empty map when `/sys/devices/system/cpu/online` is
    /// unreadable (sysfs absent, container without it mounted)
    /// or when every per-CPU read fails. `skip_serializing_if`
    /// keeps the sidecar compact on hosts without the data.
    ///
    /// Cached: the first [`collect_host_context`] call populates a
    /// process-wide [`OnceLock`] with one read per online CPU;
    /// subsequent calls clone the cached map. Governor changes
    /// after first capture are not reflected — see the
    /// "Static-cache staleness under hotplug" section in the
    /// module-level docs for the full contract.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cpufreq_governor: BTreeMap<usize, String>,
    /// Kernel name — `uname.sysname` (typically `"Linux"`).
    /// The nodename field is intentionally dropped; it's a local
    /// hostname and has no place in a published sidecar.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_name: Option<String>,
    /// Kernel release — `uname.release` (e.g. `"6.11.0-rc3"`).
    /// The full `/proc/version` banner is NOT captured because it
    /// embeds the build host + gcc version string, which is
    /// environment leakage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_release: Option<String>,
    /// Machine architecture — `uname.machine` (e.g. `"x86_64"`,
    /// `"aarch64"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// `/proc/cmdline` verbatim (trimmed of leading and trailing
    /// whitespace). Captures boot-time parameters that materially
    /// affect scheduler behavior — `preempt=`, `isolcpus=`,
    /// `nohz_full=`, `mitigations=`, hugepage reservations,
    /// `transparent_hugepage=`, and others. Stored as a single
    /// string because any split-into-pairs parser loses the
    /// quoted-value and flag-only variants the kernel accepts.
    ///
    /// Named `kernel_cmdline` rather than `cmdline` to disambiguate
    /// from [`SidecarResult::kargs`](crate::test_support::SidecarResult):
    /// that field carries the extra kargs the ktstr VMM appended
    /// when booting the guest, NOT the running host's boot line.
    /// Both are cmdline-shaped strings but describe different
    /// systems.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_cmdline: Option<String>,
    /// Kernel delay-accounting state probed from
    /// `/proc/sys/kernel/task_delayacct`. `None` only in synthetic
    /// contexts that did not probe; [`collect_host_context`] always
    /// populates it. Gates which taskstats delay-family fields are
    /// genuinely measured — see [`DelayacctState`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_delayacct: Option<DelayacctState>,
    /// CONFIG_TASK_XACCT build state probed from `/proc/config.gz`.
    /// Gates the taskstats memory-watermark fields — see
    /// [`XacctState`]. `None` only in synthetic contexts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_task_xacct: Option<XacctState>,
    /// Running process's jemalloc heap state — active / allocated /
    /// resident / mapped bytes and arena count. Populated on
    /// jemalloc-linked builds (every ktstr binary), `None` on
    /// downstream consumers that use the library without
    /// installing `tikv_jemallocator` as `#[global_allocator]`. See
    /// [`HostHeapState`](crate::host_heap::HostHeapState) for the
    /// field-level documentation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heap_state: Option<crate::host_heap::HostHeapState>,
}

/// Extract the bracketed active policy from a kernel mm
/// menu-style string such as `"always [madvise] never"` (THP
/// enabled) or `"always defer defer+madvise [madvise] never"`
/// (THP defrag). Returns the content between the first `[` and
/// first subsequent `]`, or `None` if either bracket is missing.
///
/// **First-bracket-wins**: if the string contains multiple `[..]`
/// pairs (e.g. a hand-written test fixture or a malformed sysfs
/// read), only the FIRST pair is returned; later pairs are
/// ignored. The kernel emits exactly one bracketed token in
/// practice — this scanner exists to decode that canonical shape,
/// not to validate arbitrary input.
///
/// Exposed as a pure helper so downstream tooling that wants the
/// active policy (not the full menu) does not have to re-implement
/// the bracket scan. The raw field is kept on [`HostContext`] for
/// consumers that want the menu; [`HostContext::thp_enabled_active`]
/// and [`HostContext::thp_defrag_active`] route through this
/// helper.
pub fn parse_bracketed_active_policy(s: &str) -> Option<&str> {
    let open = s.find('[')?;
    let rest = &s[open + 1..];
    let close = rest.find(']')?;
    Some(&rest[..close])
}

fn fmt_opt<T: std::fmt::Display>(v: Option<&T>) -> String {
    match v {
        Some(v) => v.to_string(),
        None => "(unknown)".to_string(),
    }
}

fn diff_row<T: std::fmt::Display + PartialEq>(
    out: &mut String,
    key: &str,
    a: Option<&T>,
    b: Option<&T>,
) {
    use std::fmt::Write;
    if a == b {
        return;
    }
    let _ = writeln!(out, "  {key}: {} → {}", fmt_opt(a), fmt_opt(b));
}

fn summarize_tunables(m: Option<&BTreeMap<String, String>>) -> String {
    match m {
        None => "(unknown)".to_string(),
        Some(map) if map.is_empty() => "(empty)".to_string(),
        Some(map) if map.len() == 1 => "(1 entry)".to_string(),
        Some(map) => format!("({} entries)", map.len()),
    }
}

/// Append the per-CPU `cpufreq_governor` section of a
/// [`HostContext::diff`]. Iterates the union of CPU ids from both
/// maps in `BTreeSet` (ascending CPU id) order and emits one
/// `cpufreq_governor.cpuN: before → after` line per CPU whose
/// governor differs; a CPU present on only one side renders
/// `(absent)` on the other.
fn diff_cpufreq_governor(
    out: &mut String,
    a_cpufreq_governor: &BTreeMap<usize, String>,
    b_cpufreq_governor: &BTreeMap<usize, String>,
) {
    use std::fmt::Write;
    let mut cpus: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    cpus.extend(a_cpufreq_governor.keys().copied());
    cpus.extend(b_cpufreq_governor.keys().copied());
    for cpu in cpus {
        let av = a_cpufreq_governor.get(&cpu);
        let bv = b_cpufreq_governor.get(&cpu);
        if av != bv {
            let _ = writeln!(
                out,
                "  cpufreq_governor.cpu{cpu}: {} → {}",
                av.map(String::as_str).unwrap_or("(absent)"),
                bv.map(String::as_str).unwrap_or("(absent)"),
            );
        }
    }
}

/// Append the `sched_tunables` section of a [`HostContext::diff`].
/// When both sides carry a map, emits one `sched_tunables.KEY:
/// before → after` line per differing key in `BTreeSet`
/// (ascending key) order, rendering `(absent)` for a key present on
/// only one side. When the maps' `Option` presence or cardinality
/// differs but a per-key diff is not applicable (one side `None`),
/// emits a single summarized `sched_tunables: before → after` line
/// via [`summarize_tunables`].
fn diff_sched_tunables(
    out: &mut String,
    a_sched_tunables: Option<&BTreeMap<String, String>>,
    b_sched_tunables: Option<&BTreeMap<String, String>>,
) {
    use std::fmt::Write;
    match (a_sched_tunables, b_sched_tunables) {
        (Some(am), Some(bm)) => {
            let mut keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            keys.extend(am.keys().map(String::as_str));
            keys.extend(bm.keys().map(String::as_str));
            for k in keys {
                let av = am.get(k);
                let bv = bm.get(k);
                if av != bv {
                    let _ = writeln!(
                        out,
                        "  sched_tunables.{k}: {} → {}",
                        av.map(String::as_str).unwrap_or("(absent)"),
                        bv.map(String::as_str).unwrap_or("(absent)"),
                    );
                }
            }
        }
        (am, bm) if am != bm => {
            let _ = writeln!(
                out,
                "  sched_tunables: {} → {}",
                summarize_tunables(am),
                summarize_tunables(bm),
            );
        }
        _ => {}
    }
}

/// Append the `heap_state` section of a [`HostContext::diff`]. When
/// both sides carry a [`HostHeapState`](crate::host_heap::HostHeapState),
/// delegates to its own `diff` and, if non-empty, emits a
/// `heap_state:` header followed by the nested diff indented two
/// extra spaces. When only one side is present, emits a single
/// `heap_state: (present) → (unknown)` (or reverse) line.
fn diff_heap_state(
    out: &mut String,
    a_heap_state: Option<&crate::host_heap::HostHeapState>,
    b_heap_state: Option<&crate::host_heap::HostHeapState>,
) {
    use std::fmt::Write;
    match (a_heap_state, b_heap_state) {
        (Some(ah), Some(bh)) => {
            let inner = ah.diff(bh);
            if !inner.is_empty() {
                out.push_str("  heap_state:\n");
                for line in inner.lines() {
                    let _ = writeln!(out, "    {line}");
                }
            }
        }
        (a, b) if a != b => {
            let _ = writeln!(
                out,
                "  heap_state: {} → {}",
                if a.is_some() {
                    "(present)"
                } else {
                    "(unknown)"
                },
                if b.is_some() {
                    "(present)"
                } else {
                    "(unknown)"
                },
            );
        }
        _ => {}
    }
}

impl HostContext {
    /// Populated [`HostContext`] for unit tests. Every field carries
    /// a reasonable non-trivial value so call sites only spell out
    /// what they want to vary via post-hoc field assignment
    /// (`#[non_exhaustive]` rejects all StructExpression forms
    /// cross-crate, including functional update):
    ///
    /// ```
    /// use ktstr::prelude::HostContext;
    /// let mut ctx = HostContext::test_fixture();
    /// ctx.numa_nodes = Some(4);
    /// ```
    ///
    /// Defaults model a plausible 2-node x86_64 Linux host: Intel
    /// CPU identity, 64 GiB memory, 2 NUMA nodes, default THP
    /// policies, a minimal `sched_*` tunable map, and a populated
    /// uname triple. Parity with
    /// `SidecarResult::test_fixture`
    /// — both fixtures exist so tests don't re-derive an
    /// "everything populated" baseline in every call site.
    ///
    /// # Usage guidance
    ///
    /// Prefer this fixture over local "populated default" helpers
    /// — a local closure duplicates the default set and drifts the
    /// moment [`HostContext`] grows a field. This is the single
    /// place those defaults live. Hash-stability and
    /// serialization-pin tests are the one exception: they must
    /// NOT rely on these defaults, because any future change to
    /// the fixture would silently shift the pinned value. Spell
    /// every participating field out explicitly in such tests so
    /// the pin is robust against fixture evolution.
    pub fn test_fixture() -> HostContext {
        let mut sched_tunables = BTreeMap::new();
        sched_tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
        sched_tunables.insert("sched_latency_ns".to_string(), "24000000".to_string());
        HostContext {
            cpu_model: Some("Intel(R) Xeon(R) Test CPU".to_string()),
            cpu_vendor: Some("GenuineIntel".to_string()),
            total_memory_kib: Some(64 * 1024 * 1024),
            hugepages_total: Some(0),
            hugepages_free: Some(0),
            hugepages_size_kib: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("always defer defer+madvise [madvise] never".to_string()),
            sched_tunables: Some(sched_tunables),
            online_cpus: Some(16),
            numa_nodes: Some(2),
            cpufreq_governor: {
                let mut m = BTreeMap::new();
                for cpu in 0..16 {
                    m.insert(cpu, "performance".to_string());
                }
                m
            },
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.16.0-test".to_string()),
            arch: Some("x86_64".to_string()),
            kernel_cmdline: Some("BOOT_IMAGE=/boot/vmlinuz-test root=/dev/sda1".to_string()),
            task_delayacct: Some(DelayacctState::On),
            config_task_xacct: Some(XacctState::On),
            heap_state: Some(crate::host_heap::HostHeapState::test_fixture()),
        }
    }

    /// Render as a human-readable multi-line report. Each field
    /// occupies one line as `key: value`. Absent fields render as
    /// `(unknown)` rather than being dropped, so operators see
    /// which fields failed to populate. The `sched_tunables` map
    /// is expanded one entry per line under the parent key; an
    /// empty map renders as `(empty)` and a `None` map as
    /// `(unknown)`. The output ends with a newline.
    ///
    /// This output is for human inspection only. For programmatic
    /// access, parse the sidecar JSON directly or drive `serde_json`
    /// against the [`HostContext`] struct — the text format here is
    /// not a stable serialization contract and may be retuned for
    /// readability without notice.
    ///
    /// Naming: the name pair (`format_human` with no
    /// `format_machine`) is intentional rather than accidental
    /// asymmetry. The "machine" surface is serde JSON — callers
    /// that want a machine-readable rendering use
    /// `serde_json::to_string(ctx)` directly. A dedicated
    /// `format_machine` wrapper around that one line would add no
    /// value. `format_human` stays named as it is (not as
    /// `impl Display`) because it prints a multi-line block with
    /// its own newline, which clashes with `Display`'s implicit
    /// one-value-per-formatter convention; embedding this in
    /// `format!("{ctx}")` would surprise callers used to single-
    /// line Display output.
    pub fn format_human(&self) -> String {
        use std::fmt::Write;
        // Destructuring bind forces every field of HostContext to
        // appear by name here. Adding a new field to the struct
        // will fail compilation until this function handles it —
        // that is the intent, it prevents `show-host` from
        // silently dropping a freshly-captured dimension.
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
        } = self;
        fn row<T: std::fmt::Display>(out: &mut String, key: &str, value: Option<&T>) {
            match value {
                Some(v) => {
                    let _ = writeln!(out, "{key}: {v}");
                }
                None => {
                    let _ = writeln!(out, "{key}: (unknown)");
                }
            }
        }
        let mut out = String::new();
        row(&mut out, "kernel_name", kernel_name.as_ref());
        row(&mut out, "kernel_release", kernel_release.as_ref());
        row(&mut out, "arch", arch.as_ref());
        row(&mut out, "cpu_model", cpu_model.as_ref());
        row(&mut out, "cpu_vendor", cpu_vendor.as_ref());
        row(&mut out, "total_memory_kib", total_memory_kib.as_ref());
        row(&mut out, "hugepages_total", hugepages_total.as_ref());
        row(&mut out, "hugepages_free", hugepages_free.as_ref());
        row(&mut out, "hugepages_size_kib", hugepages_size_kib.as_ref());
        row(&mut out, "online_cpus", online_cpus.as_ref());
        row(&mut out, "numa_nodes", numa_nodes.as_ref());
        row(&mut out, "thp_enabled", thp_enabled.as_ref());
        row(&mut out, "thp_defrag", thp_defrag.as_ref());
        row(&mut out, "kernel_cmdline", kernel_cmdline.as_ref());
        row(&mut out, "task_delayacct", task_delayacct.as_ref());
        row(&mut out, "config_task_xacct", config_task_xacct.as_ref());
        if cpufreq_governor.is_empty() {
            out.push_str("cpufreq_governor: (empty)\n");
        } else {
            out.push_str("cpufreq_governor:\n");
            for (cpu, gov) in cpufreq_governor {
                let _ = writeln!(&mut out, "  cpu{cpu} = {gov}");
            }
        }
        match sched_tunables {
            Some(map) if !map.is_empty() => {
                out.push_str("sched_tunables:\n");
                for (k, v) in map {
                    let _ = writeln!(&mut out, "  {k} = {v}");
                }
            }
            Some(_) => out.push_str("sched_tunables: (empty)\n"),
            None => out.push_str("sched_tunables: (unknown)\n"),
        }
        match heap_state {
            Some(h) => {
                out.push_str("heap_state:\n");
                for line in h.format_human().lines() {
                    let _ = writeln!(&mut out, "  {line}");
                }
            }
            None => out.push_str("heap_state: (unknown)\n"),
        }
        out
    }

    /// Active THP-enabled policy, extracted from the bracketed
    /// `[...]` token inside [`Self::thp_enabled`]. Returns the
    /// content between the first `[` and subsequent `]` (e.g.
    /// `"madvise"` from `"always [madvise] never"`). `None` when
    /// `thp_enabled` is `None`, empty, or carries no bracketed
    /// token (kernels that reshape the menu format).
    ///
    /// Provided so downstream tooling (`cargo ktstr stats`, CI
    /// regression gates, custom dashboards) can consume the active
    /// policy as a bare token without re-implementing the bracket
    /// scan in every caller.
    pub fn thp_enabled_active(&self) -> Option<&str> {
        self.thp_enabled
            .as_deref()
            .and_then(parse_bracketed_active_policy)
    }

    /// Active THP-defrag policy, extracted the same way as
    /// [`Self::thp_enabled_active`]. Returns e.g. `"madvise"` from
    /// `"always defer defer+madvise [madvise] never"`.
    pub fn thp_defrag_active(&self) -> Option<&str> {
        self.thp_defrag
            .as_deref()
            .and_then(parse_bracketed_active_policy)
    }

    /// Render the differences between two host contexts as
    /// indented `key: before → after` lines. Fields that compare
    /// equal are omitted; an empty return value means the two
    /// contexts are field-for-field identical (including
    /// `sched_tunables`). `None` values render as `(unknown)` and
    /// map entries present in one side only render as `(absent)`
    /// so a `None → Some(..)` transition does not silently look
    /// the same as an unchanged absent field. When only one side
    /// has a `sched_tunables` map, the other side renders
    /// `(unknown)`; the Some side renders as `(empty)` for an
    /// empty map or `(N entries)` for a populated one so the
    /// cardinality of the new data is visible at a glance.
    pub fn diff(&self, other: &HostContext) -> String {
        // Symmetric destructuring bind of both sides: forces every
        // field to appear by name here, same reason as
        // `format_human` — a new HostContext field must be
        // explicitly classified as hash-participating, scalar, or
        // structured before diff will compile.
        let HostContext {
            cpu_model: a_cpu_model,
            cpu_vendor: a_cpu_vendor,
            total_memory_kib: a_total_memory_kib,
            hugepages_total: a_hugepages_total,
            hugepages_free: a_hugepages_free,
            hugepages_size_kib: a_hugepages_size_kib,
            thp_enabled: a_thp_enabled,
            thp_defrag: a_thp_defrag,
            sched_tunables: a_sched_tunables,
            online_cpus: a_online_cpus,
            numa_nodes: a_numa_nodes,
            cpufreq_governor: a_cpufreq_governor,
            kernel_name: a_kernel_name,
            kernel_release: a_kernel_release,
            arch: a_arch,
            kernel_cmdline: a_kernel_cmdline,
            task_delayacct: a_task_delayacct,
            config_task_xacct: a_config_task_xacct,
            heap_state: a_heap_state,
        } = self;
        let HostContext {
            cpu_model: b_cpu_model,
            cpu_vendor: b_cpu_vendor,
            total_memory_kib: b_total_memory_kib,
            hugepages_total: b_hugepages_total,
            hugepages_free: b_hugepages_free,
            hugepages_size_kib: b_hugepages_size_kib,
            thp_enabled: b_thp_enabled,
            thp_defrag: b_thp_defrag,
            sched_tunables: b_sched_tunables,
            online_cpus: b_online_cpus,
            numa_nodes: b_numa_nodes,
            cpufreq_governor: b_cpufreq_governor,
            kernel_name: b_kernel_name,
            kernel_release: b_kernel_release,
            arch: b_arch,
            kernel_cmdline: b_kernel_cmdline,
            task_delayacct: b_task_delayacct,
            config_task_xacct: b_config_task_xacct,
            heap_state: b_heap_state,
        } = other;
        let mut out = String::new();
        diff_row(
            &mut out,
            "kernel_name",
            a_kernel_name.as_ref(),
            b_kernel_name.as_ref(),
        );
        diff_row(
            &mut out,
            "kernel_release",
            a_kernel_release.as_ref(),
            b_kernel_release.as_ref(),
        );
        diff_row(&mut out, "arch", a_arch.as_ref(), b_arch.as_ref());
        diff_row(
            &mut out,
            "cpu_model",
            a_cpu_model.as_ref(),
            b_cpu_model.as_ref(),
        );
        diff_row(
            &mut out,
            "cpu_vendor",
            a_cpu_vendor.as_ref(),
            b_cpu_vendor.as_ref(),
        );
        diff_row(
            &mut out,
            "total_memory_kib",
            a_total_memory_kib.as_ref(),
            b_total_memory_kib.as_ref(),
        );
        diff_row(
            &mut out,
            "hugepages_total",
            a_hugepages_total.as_ref(),
            b_hugepages_total.as_ref(),
        );
        diff_row(
            &mut out,
            "hugepages_free",
            a_hugepages_free.as_ref(),
            b_hugepages_free.as_ref(),
        );
        diff_row(
            &mut out,
            "hugepages_size_kib",
            a_hugepages_size_kib.as_ref(),
            b_hugepages_size_kib.as_ref(),
        );
        diff_row(
            &mut out,
            "online_cpus",
            a_online_cpus.as_ref(),
            b_online_cpus.as_ref(),
        );
        diff_row(
            &mut out,
            "numa_nodes",
            a_numa_nodes.as_ref(),
            b_numa_nodes.as_ref(),
        );
        diff_row(
            &mut out,
            "thp_enabled",
            a_thp_enabled.as_ref(),
            b_thp_enabled.as_ref(),
        );
        diff_row(
            &mut out,
            "thp_defrag",
            a_thp_defrag.as_ref(),
            b_thp_defrag.as_ref(),
        );
        diff_row(
            &mut out,
            "kernel_cmdline",
            a_kernel_cmdline.as_ref(),
            b_kernel_cmdline.as_ref(),
        );
        diff_row(
            &mut out,
            "task_delayacct",
            a_task_delayacct.as_ref(),
            b_task_delayacct.as_ref(),
        );
        diff_row(
            &mut out,
            "config_task_xacct",
            a_config_task_xacct.as_ref(),
            b_config_task_xacct.as_ref(),
        );
        diff_cpufreq_governor(&mut out, a_cpufreq_governor, b_cpufreq_governor);
        diff_sched_tunables(
            &mut out,
            a_sched_tunables.as_ref(),
            b_sched_tunables.as_ref(),
        );
        diff_heap_state(&mut out, a_heap_state.as_ref(), b_heap_state.as_ref());
        out
    }
}

/// Static-fields cache. These values do not change for the lifetime
/// of the process (CPU identity, total installed memory, hugepage
/// size chosen at boot, NUMA count, uname triple), so walking
/// `/proc` and `/sys` for them once and reusing the result avoids
/// repeated syscalls on every sidecar write. Dynamic fields
/// (sched_tunables, hugepages_total, hugepages_free, thp_enabled,
/// thp_defrag, kernel_cmdline) are NOT cached — they can shift
/// between tests via sysctl, hugepage reservation, THP policy flip,
/// or live kexec, and a cached snapshot would hide that change.
///
/// Per-CPU `cpufreq_governor` is cached separately in
/// [`CPUFREQ_GOVERNORS`] rather than embedded here so the cache
/// hit on the per-call path does not clone a `BTreeMap<usize, String>`
/// of up to `online_cpus` entries through the `StaticHostInfo`
/// clone — `StaticHostInfo` carries only primitive `Option<…>`
/// fields and stays cheap to clone, while `CPUFREQ_GOVERNORS`
/// owns the heavyweight collection and is cloned on its own
/// hit-path.
#[derive(Clone)]
struct StaticHostInfo {
    cpu_model: Option<String>,
    cpu_vendor: Option<String>,
    total_memory_kib: Option<u64>,
    hugepages_size_kib: Option<u64>,
    online_cpus: Option<usize>,
    numa_nodes: Option<usize>,
    kernel_name: Option<String>,
    kernel_release: Option<String>,
    arch: Option<String>,
}

static STATIC_HOST_INFO: OnceLock<StaticHostInfo> = OnceLock::new();

/// Process-wide cache for the per-CPU `scaling_governor` map. The
/// first [`collect_host_context`] call populates this lock by
/// invoking [`read_cpufreq_governors`]; every later call clones
/// the cached `BTreeMap` instead of re-reading
/// `/sys/devices/system/cpu/cpu{N}/cpufreq/scaling_governor` for
/// every online CPU. With N online CPUs and M sidecar writes per
/// process, this collapses up to N × M sysfs reads (a 256-CPU
/// host running a 1000-test session = 256 000 reads) to N. See
/// the module-level "Static-cache staleness under hotplug"
/// section for the consequences of pinning the first observed
/// snapshot — runtime governor changes after first capture are
/// not reflected.
static CPUFREQ_GOVERNORS: OnceLock<BTreeMap<usize, String>> = OnceLock::new();

/// Test-only call counter for [`compute_static_host_info`]. Pinned
/// by `call_counts_*` tests to prove the OnceLock is exercised at
/// most once per process, independent of how many
/// `collect_host_context` calls happen. Production builds do not
/// carry the counter.
#[cfg(test)]
static STATIC_INIT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Test-only call counter for [`read_meminfo`]. Pinned by
/// `call_counts_*` tests to prove the `/proc/meminfo` dedup holds
/// — exactly one read per `collect_host_context` call, not the
/// pre-dedup two reads on the cold path. Production builds do not
/// carry the counter.
#[cfg(test)]
static MEMINFO_READ_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Test-only call counter for [`read_cpufreq_governors`]. Pinned
/// by `call_counts_*` tests to prove the [`CPUFREQ_GOVERNORS`]
/// cache exercises the underlying sysfs walk at most once per
/// process. Production builds do not carry the counter.
#[cfg(test)]
static CPUFREQ_GOVERNORS_READ_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Capture the host context. Static fields are collected once
/// and cached; dynamic fields are re-read on every call so
/// intra-run sysctl / hugepage / THP changes are reflected.
///
/// Every sub-read is fallible; individual failures leave the
/// corresponding field `None` and the rest of the context
/// proceeds. Even on a host where every `/proc` and `/sys` read
/// fails, the three uname-derived fields (`kernel_name`,
/// `kernel_release`, `arch`) still populate because they come from
/// the `uname()` syscall — filesystem-independent. An
/// otherwise-empty `HostContext` serializes to a near-empty JSON
/// object and distinguishes "collection attempted, nothing known"
/// from "collection not attempted" (represented at the enclosing
/// `Option<HostContext>` layer on
/// [`SidecarResult`](crate::test_support::SidecarResult)).
///
/// # Timing: post-run snapshot
///
/// Production call sites invoke this at sidecar-write time (see
/// `test_support::sidecar::write_sidecar` and `write_skip_sidecar`),
/// which runs AFTER the VM finishes. The returned snapshot
/// therefore reflects post-run host state, not the pre-run
/// environment the scheduler booted into.
///
/// Fields fall into two groups by how they are read:
///
/// Static subset (memoised in `STATIC_HOST_INFO` —
/// or, for `cpufreq_governor`, the parallel
/// `CPUFREQ_GOVERNORS` cache — identical across every call in
/// the process, shift only under CPU / memory / NUMA hotplug or
/// runtime governor change): the uname triple, CPU identity
/// (`cpu_model` + `cpu_vendor`), `total_memory_kib`,
/// `hugepages_size_kib`, `online_cpus`, `numa_nodes`, and
/// `cpufreq_governor`.
///
/// Dynamic subset (re-read on every call): `kernel_cmdline`,
/// `hugepages_total`, `hugepages_free`, `thp_enabled`,
/// `thp_defrag`, `sched_tunables`. `kernel_cmdline` is
/// mechanically dynamic (re-read each call) but effectively
/// static for the process (changes only across reboot). The
/// others can genuinely drift between pre-run and post-run:
///
/// - `sched_tunables`: a test that writes to `/proc/sys/kernel/sched_*`
///   and does not restore the previous value will be observed
///   with the test-mutated value.
/// - `hugepages_total` / `hugepages_free`: a test that reserves
///   or releases hugepages shifts the counts.
/// - `thp_enabled` / `thp_defrag`: a test that flips THP policy
///   is captured with the flipped policy.
///
/// Dashboards and regression tooling that need the environment
/// the scheduler actually saw (not the post-run state) should
/// treat the three drift-prone fields as "post-run snapshot" and
/// either (a) disable them in the comparison, or (b) capture a
/// pre-run snapshot via [`collect_host_context_pre_run`] and
/// travel the pair via [`HostContextSnapshots`].
pub fn collect_host_context() -> HostContext {
    // Read `/proc/meminfo` exactly once per call and share the
    // parsed fields with `compute_static_host_info` (for `mem_total_kib`
    // / `hugepages_size_kib` on cold init) and with the per-call
    // hugepage counters. The prior formulation read `/proc/meminfo`
    // twice on the cold path — once here for the dynamic counters
    // and once inside the `OnceLock` init for the static fields —
    // which is wasted syscall + parse work.
    let meminfo = read_meminfo();
    let static_info = STATIC_HOST_INFO
        .get_or_init(|| compute_static_host_info(&meminfo))
        .clone();
    HostContext {
        cpu_model: static_info.cpu_model,
        cpu_vendor: static_info.cpu_vendor,
        total_memory_kib: static_info.total_memory_kib,
        hugepages_total: meminfo.hugepages_total,
        hugepages_free: meminfo.hugepages_free,
        hugepages_size_kib: static_info.hugepages_size_kib,
        thp_enabled: read_trimmed_sysfs("/sys/kernel/mm/transparent_hugepage/enabled"),
        thp_defrag: read_trimmed_sysfs("/sys/kernel/mm/transparent_hugepage/defrag"),
        sched_tunables: read_sched_tunables(),
        online_cpus: static_info.online_cpus,
        numa_nodes: static_info.numa_nodes,
        cpufreq_governor: cached_cpufreq_governors(),
        kernel_name: static_info.kernel_name,
        kernel_release: static_info.kernel_release,
        arch: static_info.arch,
        kernel_cmdline: read_trimmed_sysfs("/proc/cmdline"),
        task_delayacct: Some(read_task_delayacct()),
        config_task_xacct: Some(read_config_task_xacct()),
        // `heap_state` is a post-run snapshot of the running ktstr
        // process's jemalloc footprint. Captured here alongside the
        // other dynamic fields so sidecar consumers can correlate
        // test outcomes with runner memory pressure. libjemalloc is
        // linked into every binary in this workspace (hard dep of
        // `tikv-jemalloc-ctl`), so `collect()` always returns a
        // populated struct when `#[global_allocator]` is jemalloc.
        // Downstream consumers using ktstr without jemallocator
        // installed see `allocated_bytes == Some(0)` and
        // `active_bytes == Some(0)` because libjemalloc is linked
        // but unused — collapse that shape to `None` so the sidecar
        // does not carry a misleading empty row. `arenas.narenas` is
        // still populated in the collapsed shape but alone carries
        // no runner-pressure information, so it travels with the
        // stats that give it meaning.
        heap_state: {
            let h = crate::host_heap::collect();
            if h.allocated_bytes == Some(0) && h.active_bytes == Some(0) {
                None
            } else {
                Some(h)
            }
        },
    }
}

/// Capture the host context at the start of a run, before the VM
/// boots or the test body mutates any sysctl / hugepage / THP
/// setting. Semantic alias for [`collect_host_context`] — the
/// collection mechanism is identical (same static-cache + dynamic
/// re-read policy) and callers remain free to call either function
/// on either side of the run, but the name pins intent:
/// `collect_host_context_pre_run` documents that the returned
/// snapshot is the authoritative view of the drift-prone dynamic
/// fields (`sched_tunables`, `hugepages_total` / `hugepages_free`,
/// `thp_enabled` / `thp_defrag`) as the scheduler saw them.
///
/// Pair the pre-run snapshot with the post-run snapshot produced by
/// [`collect_host_context`] via [`HostContextSnapshots`] so
/// downstream consumers can diff the two and surface environment
/// mutations attributable to the test body (e.g. "scheduler config
/// reservoir bumped `/proc/sys/kernel/sched_migration_cost_ns` mid-run")
/// rather than silently folding them into a single ambiguous
/// "post-run" record.
///
/// Static fields (uname triple, CPU identity, total memory,
/// hugepage size, online CPU count, NUMA node count) are
/// memoised across every call in the process via
/// `STATIC_HOST_INFO`, so `collect_host_context_pre_run` and
/// `collect_host_context` observing different values for a static
/// field implies CPU/memory/NUMA hotplug between the two calls —
/// see the module-level "Static-cache staleness under hotplug"
/// section for the hotplug contract.
pub fn collect_host_context_pre_run() -> HostContext {
    // Intentional delegation rather than code duplication: the
    // pre/post distinction is purely about WHEN the caller fires
    // the snapshot, not HOW the fields are read. Forking the
    // implementation would open the door to the two paths drifting
    // apart (a fix to dynamic-field parsing landing in one but not
    // the other), which is exactly the kind of bug the pair is
    // meant to expose.
    collect_host_context()
}

/// Paired pre-run / post-run [`HostContext`] snapshots captured
/// from a single test run, intended for sidecar persistence so
/// downstream analysis can diff the drift-prone dynamic fields
/// (`sched_tunables`, `hugepages_*`, `thp_*`) between the two
/// endpoints.
///
/// The struct deliberately carries both snapshots in full —
/// including the static fields (uname triple, CPU identity, total
/// memory) that are OnceLock-cached and therefore guaranteed equal
/// across a single process. Duplicating them on the wire (a few
/// hundred bytes of JSON per sidecar) keeps each snapshot
/// self-describing so a consumer that only cares about the
/// post-run state can read
/// [`HostContextSnapshots::post`] in isolation without reassembling
/// fields from [`HostContextSnapshots::pre`], and a consumer that
/// diffs the pair does not have to special-case "which field is
/// cached and which is dynamic".
///
/// Serde shape: both fields serialize as a full `HostContext`
/// object under their own keys. The per-field
/// `#[serde(default, skip_serializing_if = ...)]` policy on
/// `HostContext` carries through, so populated snapshots stay
/// compact. The whole struct is `#[non_exhaustive]` — see
/// [`crate::non_exhaustive`] for construction and pattern-match
/// rules.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HostContextSnapshots {
    /// Captured before the test body runs — typically via
    /// [`collect_host_context_pre_run`] at the start of sidecar
    /// setup.
    pub pre: HostContext,
    /// Captured after the test body finishes — typically via
    /// [`collect_host_context`] at sidecar-write time.
    pub post: HostContext,
}

impl HostContextSnapshots {
    /// Construct a pair from explicit pre/post snapshots. Prefer
    /// this constructor over a (forbidden cross-crate) struct
    /// literal so future fields can land on
    /// [`HostContextSnapshots`] without breaking callers.
    pub fn new(pre: HostContext, post: HostContext) -> Self {
        Self { pre, post }
    }

    /// Capture both endpoints in a single call. Useful for tests
    /// and callers that don't observe a test body between the two
    /// snapshots and only want to stamp the pair structurally (both
    /// endpoints will reflect the same dynamic state because no
    /// mutation happened in between).
    ///
    /// `#[cfg(test)]`-gated so production sidecar writers cannot
    /// reach it by accident — they need
    /// [`collect_host_context_pre_run`] before the run and
    /// [`collect_host_context`] after, which
    /// [`HostContextSnapshots::new`] then pairs. The compile-time
    /// gate replaces the earlier doc-only warning.
    #[cfg(test)]
    pub fn capture_same_instant() -> Self {
        let snap = collect_host_context();
        Self {
            pre: snap.clone(),
            post: snap,
        }
    }
}

/// Return the per-CPU `scaling_governor` map, populating the
/// process-wide [`CPUFREQ_GOVERNORS`] cache on first call and
/// cloning the cached value on every subsequent call. A clone of a
/// `BTreeMap<usize, String>` of even a few hundred entries is
/// orders of magnitude cheaper than the up to 256 sysfs `read`
/// syscalls the underlying [`read_cpufreq_governors`] performs on
/// a 256-CPU host.
fn cached_cpufreq_governors() -> BTreeMap<usize, String> {
    CPUFREQ_GOVERNORS
        .get_or_init(read_cpufreq_governors)
        .clone()
}

/// Read `scaling_governor` for every online CPU, keyed by CPU
/// id. Reads `/sys/devices/system/cpu/cpu{N}/cpufreq/scaling_governor`
/// for each entry in `/sys/devices/system/cpu/online`. Returns an
/// empty map when `/sys/devices/system/cpu/online` is unreadable
/// (sysfs absent, constrained container) or when every per-CPU
/// read fails. A CPU with no `cpufreq/` directory (non-CPUFREQ
/// kernel, VM without passthrough) contributes no entry — the
/// missing-key shape is the "no governor reported" signal for
/// consumers.
///
/// Production callers reach this through
/// [`cached_cpufreq_governors`] which memoises the result in
/// [`CPUFREQ_GOVERNORS`]; a transient sysfs failure on the very
/// first call therefore pins an empty map for the remainder of
/// the process — see the module-level "Static-cache staleness"
/// section for the contract.
fn read_cpufreq_governors() -> BTreeMap<usize, String> {
    #[cfg(test)]
    CPUFREQ_GOVERNORS_READ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let Ok(online_raw) = std::fs::read_to_string("/sys/devices/system/cpu/online") else {
        return BTreeMap::new();
    };
    let Ok(cpus) = crate::topology::parse_cpu_list(&online_raw) else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    for cpu in cpus {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor");
        if let Some(gov) = read_trimmed_sysfs(&path) {
            out.insert(cpu, gov);
        }
    }
    out
}

/// Populate the static-fields cache on first access. Takes the
/// already-parsed `/proc/meminfo` from the caller so the cold path
/// does not re-read the file. Reads `/proc/cpuinfo` (CPU identity),
/// the host NUMA topology, and a single `uname()` call.
fn compute_static_host_info(meminfo: &MeminfoFields) -> StaticHostInfo {
    #[cfg(test)]
    STATIC_INIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (cpu_model, cpu_vendor) = read_cpuinfo_identity();
    // `uname(2)` is unit-tested only through
    // `collect_host_context_returns_populated_struct_on_linux`
    // (integration-style — runs the real syscall and asserts the
    // sysname field populates). No injection seam exists by design:
    // the only post-syscall logic here is `.to_str().ok().map(...)`,
    // which is three method calls on `rustix::system::UtsName`'s
    // already-null-terminated-`CStr` accessors. Extracting that into
    // a pure parser would test `CStr::to_str` — std's invariant, not
    // ours — and the real fragility (syscall return, encoding on
    // non-Linux hosts) is untestable without a kernel mock, which
    // is outside ktstr's scope. Marking this not-unit-tested by
    // design.
    let u = rustix::system::uname();
    let (online_cpus, numa_nodes) = probe_host_topology_counts();
    StaticHostInfo {
        cpu_model,
        cpu_vendor,
        total_memory_kib: meminfo.mem_total_kib,
        hugepages_size_kib: meminfo.hugepages_size_kib,
        online_cpus,
        numa_nodes,
        kernel_name: u.sysname().to_str().ok().map(|s| s.to_string()),
        kernel_release: u.release().to_str().ok().map(|s| s.to_string()),
        arch: u.machine().to_str().ok().map(|s| s.to_string()),
    }
}

/// One `HostTopology::from_sysfs` probe → both the online-CPU
/// count and the NUMA-node count. Returning a tuple keeps the
/// two derived values bound to the same probe, so a hotplug
/// event between reads cannot make them disagree. Both values
/// are `None` when the probe errors.
fn probe_host_topology_counts() -> (Option<usize>, Option<usize>) {
    match crate::vmm::host_topology::HostTopology::from_sysfs() {
        Ok(topo) => (
            Some(topo.online_cpus.len()),
            Some(count_numa_nodes_in_topology(&topo)),
        ),
        Err(_) => (None, None),
    }
}

/// Read `/proc/cpuinfo` and extract the first processor's
/// `vendor_id` and `model name` lines. Thin I/O wrapper; the
/// parsing logic lives in [`parse_cpuinfo_identity`] so it can
/// be unit-tested with synthetic fixtures.
fn read_cpuinfo_identity() -> (Option<String>, Option<String>) {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return (None, None);
    };
    parse_cpuinfo_identity(&text)
}

/// Pure parser split from `read_cpuinfo_identity` for unit
/// testability. Parses the first processor's `vendor_id` and
/// `model name` lines from `/proc/cpuinfo` content. Returning
/// after the first blank line (processor boundary) keeps the
/// scan O(one processor) on big machines where `/proc/cpuinfo`
/// can span many MiB.
fn parse_cpuinfo_identity(text: &str) -> (Option<String>, Option<String>) {
    let mut model: Option<String> = None;
    let mut vendor: Option<String> = None;
    for line in text.lines() {
        if line.is_empty() {
            // End of the first processor block — both fields we want
            // are per-processor and appear before the first blank
            // line.
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match key {
                "model name" if model.is_none() => model = Some(value.to_string()),
                "vendor_id" if vendor.is_none() => vendor = Some(value.to_string()),
                _ => {}
            }
        }
    }
    (model, vendor)
}

/// The `/proc/meminfo` fields the host-context snapshot consumes. A
/// purpose-built struct avoids the BTreeMap lookup/clone dance and
/// makes the set of captured fields explicit at the type level.
#[derive(Default)]
struct MeminfoFields {
    mem_total_kib: Option<u64>,
    hugepages_total: Option<u64>,
    hugepages_free: Option<u64>,
    hugepages_size_kib: Option<u64>,
}

/// Read `/proc/meminfo` and extract the four fields the host
/// context needs. Thin I/O wrapper; parsing lives in
/// [`parse_meminfo`] so it can be unit-tested with synthetic
/// fixtures.
fn read_meminfo() -> MeminfoFields {
    #[cfg(test)]
    MEMINFO_READ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return MeminfoFields::default();
    };
    parse_meminfo(&text)
}

/// Pure parser split from `read_meminfo` for unit testability.
/// Parses the four `/proc/meminfo` fields the host context needs
/// from already-read content. Lines without a numeric first token
/// are silently skipped so a kernel that introduces a new
/// non-numeric line (e.g. a future flags field) does not poison
/// the struct.
fn parse_meminfo(text: &str) -> MeminfoFields {
    let mut out = MeminfoFields::default();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let token = rest.split_whitespace().next().unwrap_or("");
        let Ok(n) = token.parse::<u64>() else {
            continue;
        };
        match key {
            "MemTotal" => out.mem_total_kib = Some(n),
            "HugePages_Total" => out.hugepages_total = Some(n),
            "HugePages_Free" => out.hugepages_free = Some(n),
            "Hugepagesize" => out.hugepages_size_kib = Some(n),
            _ => {}
        }
    }
    out
}

/// Read a sysfs leaf (or `/proc` pseudofile) and return its
/// trimmed content. Thin I/O wrapper; parsing lives in
/// [`parse_trimmed`] so it can be unit-tested with synthetic
/// fixtures. Returns `None` on any read error (ENOENT, EACCES,
/// EIO) so the caller records the field as absent without
/// treating it as a fatal context-collection failure.
fn read_trimmed_sysfs(path: impl AsRef<std::path::Path>) -> Option<String> {
    std::fs::read_to_string(path.as_ref())
        .ok()
        .and_then(|s| parse_trimmed(&s))
}

/// Pure parser split from `read_trimmed_sysfs` for unit
/// testability. Trims leading and trailing whitespace; returns
/// `None` when the result is empty — an empty cmdline or thp
/// file is not useful to record. Bracketed content inside the
/// value (e.g. `"always [madvise] never"` from THP) is preserved
/// verbatim because `str::trim` only affects the edges.
fn parse_trimmed(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Walk `/proc/sys/kernel` for entries whose name starts with
/// `sched_` and record each as `basename → content`. Skips any
/// entry that is not a regular file — directories, symlinks,
/// sockets, fifos, and block/char devices all fall through the
/// `file_type.is_file()` guard. The kernel exposes no non-file
/// `sched_*` entries today but guarding keeps behavior defined if
/// that changes. Also skips entries whose name is not valid UTF-8
/// and entries whose contents cannot be read or trim to empty.
///
/// Returns `None` only when the directory listing itself fails
/// (unreadable `/proc/sys/kernel`); an empty map is a valid result
/// — it means the directory was readable but had no entries
/// starting with `sched_`, or every such entry failed the
/// per-file read or trim to empty.
fn read_sched_tunables() -> Option<BTreeMap<String, String>> {
    read_sched_tunables_from(std::path::Path::new("/proc/sys/kernel"))
}

/// Path-parameterized walk used by [`read_sched_tunables`]. Seam for
/// unit tests that drive the walk with a tempdir full of `sched_*`
/// fixture files — everything the production caller does is mirrored
/// here except the hardcoded sysfs path, so a future test can
/// exercise the real walk + filter + read pipeline against a
/// controlled directory rather than against `/proc`.
fn read_sched_tunables_from(dir: &std::path::Path) -> Option<BTreeMap<String, String>> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut out = BTreeMap::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("sched_") {
            continue;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        if let Some(content) = read_trimmed_sysfs(&path) {
            out.insert(name.to_string(), content);
        }
    }
    Some(out)
}

/// Build + runtime state of kernel delay accounting, probed from
/// `/proc/sys/kernel/task_delayacct`. Determines which taskstats
/// delay-family fields are actually being populated:
/// - `cpu_delay_*` come from `tsk->sched_info` and are filled
///   UNCONDITIONALLY by `delayacct_add_tsk` (kernel/delayacct.c)
///   whenever CONFIG_TASK_DELAY_ACCT is built in — they survive the
///   runtime toggle, so they read real values in both [`Self::On`]
///   and [`Self::RuntimeOff`].
/// - the per-resource lock categories (`blkio`, `swapin`,
///   `freepages`, `thrashing`, `compact`, `wpcopy`, `irq`) are gated
///   by `tsk->delays`, allocated at fork only when `delayacct_on`, so
///   they read genuine values only in [`Self::On`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DelayacctState {
    /// `/proc/sys/kernel/task_delayacct` absent. The sysctl is
    /// registered only under CONFIG_TASK_DELAY_ACCT
    /// (kernel/delayacct.c), so an absent file means the option is not
    /// built in and NO delay-family field is populated. (Under ktstr's
    /// root execution a present file is always readable, so absent is
    /// the only not-`On`/not-`RuntimeOff` outcome.)
    ConfigOff,
    /// File present, reads `0` — built in but the runtime toggle is
    /// off. `cpu_delay_*` still populate; the lock categories read zero
    /// for tasks forked while off.
    RuntimeOff,
    /// File present, reads `1` — delay accounting fully active.
    On,
}

impl std::fmt::Display for DelayacctState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DelayacctState::ConfigOff => "config-off",
            DelayacctState::RuntimeOff => "runtime-off",
            DelayacctState::On => "on",
        };
        f.write_str(s)
    }
}

/// Build state of extended task accounting (CONFIG_TASK_XACCT),
/// probed from `/proc/config.gz`. Gates the taskstats memory
/// watermark fields (`hiwater_rss_bytes`, `hiwater_vm_bytes`), which
/// `xacct_add_tsk` (kernel/tsacct.c) fills whenever CONFIG_TASK_XACCT
/// is built in — there is NO runtime toggle, unlike delay accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum XacctState {
    /// `/proc/config.gz` shows `CONFIG_TASK_XACCT=y` — watermarks
    /// populate (for tasks with an address space; a kernel thread
    /// reads zero).
    On,
    /// `/proc/config.gz` shows `# CONFIG_TASK_XACCT is not set` — not
    /// built in, watermarks read zero.
    Off,
    /// `/proc/config.gz` is unreadable (no CONFIG_IKCONFIG_PROC, or the
    /// gz read/parse failed) or the symbol line is absent. Treated as
    /// measured (like [`Self::On`]) for gating so a host that does not
    /// expose its config never produces a false "absent" verdict.
    Unknown,
}

impl std::fmt::Display for XacctState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            XacctState::On => "on",
            XacctState::Off => "off",
            XacctState::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// Probe the delay-accounting state from `/proc/sys/kernel/task_delayacct`.
fn read_task_delayacct() -> DelayacctState {
    read_task_delayacct_from(std::path::Path::new("/proc/sys/kernel/task_delayacct"))
}

/// Path-parameterized seam for [`read_task_delayacct`] — unit tests
/// drive it with a tempdir holding a `task_delayacct` fixture (or no
/// such file, for the [`DelayacctState::ConfigOff`] case). A
/// present-but-empty or unreadable file also maps to `ConfigOff`
/// (`read_trimmed_sysfs` returns `None`); the real `proc_dointvec_minmax`
/// sysctl always emits "0"/"1", so that conflation is unreachable under
/// ktstr's root execution where a present file is readable.
fn read_task_delayacct_from(path: &std::path::Path) -> DelayacctState {
    match read_trimmed_sysfs(path) {
        None => DelayacctState::ConfigOff,
        Some(s) if s == "1" => DelayacctState::On,
        // Any present-but-not-"1" value (canonically "0") is runtime-off.
        Some(_) => DelayacctState::RuntimeOff,
    }
}

/// Probe CONFIG_TASK_XACCT from `/proc/config.gz`.
fn read_config_task_xacct() -> XacctState {
    read_config_task_xacct_from(std::path::Path::new("/proc/config.gz"))
}

/// Gz-decode `path` and classify CONFIG_TASK_XACCT. Unreadable file or
/// decode failure → [`XacctState::Unknown`]. The decompressed-text
/// classification lives in [`parse_kconfig_xacct`] so it is unit-testable
/// without writing a gz fixture.
fn read_config_task_xacct_from(path: &std::path::Path) -> XacctState {
    let Ok(bytes) = std::fs::read(path) else {
        return XacctState::Unknown;
    };
    use std::io::Read as _;
    let mut gz = flate2::read::GzDecoder::new(&bytes[..]);
    let mut text = String::new();
    if gz.read_to_string(&mut text).is_err() {
        return XacctState::Unknown;
    }
    parse_kconfig_xacct(&text)
}

/// Classify CONFIG_TASK_XACCT from decompressed kconfig text. `=y` →
/// [`XacctState::On`]; `# CONFIG_TASK_XACCT is not set` →
/// [`XacctState::Off`]; the symbol absent entirely →
/// [`XacctState::Unknown`] (a partial/foreign config, treated as
/// measured downstream). Matches the whole trimmed line so a substring
/// in an unrelated symbol cannot false-match.
fn parse_kconfig_xacct(text: &str) -> XacctState {
    if text.lines().any(|l| l.trim() == "CONFIG_TASK_XACCT=y") {
        XacctState::On
    } else if text.lines().any(|l| l.trim() == "# CONFIG_TASK_XACCT is not set") {
        XacctState::Off
    } else {
        XacctState::Unknown
    }
}

/// Per-sub-family "is this taskstats family genuinely populated host-wide"
/// state, reduced from the two kernel-accounting probes. Computed once per
/// ctprof snapshot and baked into each captured thread (AND-ed with the
/// per-thread query-Ok), so the group aggregation gates each sub-family
/// independently rather than treating the whole taskstats payload as one flag.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskstatsActive {
    /// `cpu_delay_*` survive the runtime toggle (sched_info-sourced, filled
    /// unconditionally) — active unless CONFIG_TASK_DELAY_ACCT is off entirely.
    pub cpu_delay: bool,
    /// The delayacct resource-wait categories need delayacct runtime-on
    /// (`tsk->delays` is allocated at fork only when `delayacct_on`).
    pub delay_block: bool,
    /// The xacct watermarks need CONFIG_TASK_XACCT; [`XacctState::Unknown`]
    /// (config not exposed) is treated as active to avoid a false absent.
    pub xacct: bool,
}

/// Probe both kernel-accounting states and reduce to the per-sub-family active
/// flags. Two small `/proc` reads; called once per ctprof snapshot.
pub(crate) fn probe_taskstats_active() -> TaskstatsActive {
    taskstats_active_from(read_task_delayacct(), read_config_task_xacct())
}

/// Pure reduction of the two probe enums to the per-sub-family active flags —
/// the counter-semantics crux, split out so it is unit-testable without `/proc`.
/// `cpu_delay` survives `RuntimeOff` (gate is `!= ConfigOff`, NOT `== On`,
/// because `delayacct_add_tsk` fills the sched_info CPU block BEFORE the
/// `if (!tsk->delays)` gate); the resource-wait categories need `== On`
/// (`tsk->delays` is allocated at fork only when delayacct is on); xacct keys off
/// CONFIG only, with `Unknown` treated active to avoid a false absent.
fn taskstats_active_from(delayacct: DelayacctState, xacct: XacctState) -> TaskstatsActive {
    TaskstatsActive {
        cpu_delay: delayacct != DelayacctState::ConfigOff,
        delay_block: delayacct == DelayacctState::On,
        xacct: xacct != XacctState::Off,
    }
}

/// Pure-function seam used by [`probe_host_topology_counts`]
/// (which itself wraps
/// [`HostTopology::from_sysfs`](crate::vmm::host_topology::HostTopology::from_sysfs),
/// which in turn wraps
/// [`TestTopology::from_system`](crate::topology::TestTopology::from_system)):
/// given a [`HostTopology`](crate::vmm::host_topology::HostTopology),
/// return the number of distinct NUMA nodes it claims. An empty
/// `cpu_to_node` map maps to `1` because every Linux system has
/// at least one NUMA node — returning zero would misrepresent the
/// topology. Sparse / non-contiguous node IDs are counted
/// correctly because `BTreeSet::from_iter` deduplicates on
/// insert.
///
/// # Empty `cpu_to_node`: UMA or broken probe?
///
/// In production the answer is: empty cannot occur from a
/// successful probe.
/// [`TestTopology::from_system`](crate::topology::TestTopology::from_system)
/// bails on `online_cpus.is_empty()`, and every online CPU
/// whose `/sys/devices/system/cpu/cpuN/` directory exists falls
/// through to at least `llc_id=0, node_id=0` when the per-CPU
/// reads inside that directory fail. CPUs listed in
/// `/sys/devices/system/cpu/online` whose sysfs directory is
/// absent are dropped with a `tracing::warn!` rather than
/// fallen-through — so on a host where every listed CPU lacks
/// its sysfs dir, `llc_groups` would be empty and
/// `cpu_to_node` would be empty too. That failure mode is
/// degenerate (a listed-but-absent CPU is itself a kernel/sysfs
/// bug) and not the common case. The `.max(1)` floor is
/// therefore a guard for synthetic topologies (unit-test
/// callers of this pure function) and for the degenerate
/// "all-dropped" probe — treating "no entries, but probe said
/// OK" as UMA is the conservative interpretation.
///
/// Keeping the I/O (sysfs probe) separate from the pure counting
/// logic lets unit tests exercise the fallback branch and the
/// dedup path without standing up a real /sys layout.
pub(crate) fn count_numa_nodes_in_topology(
    topo: &crate::vmm::host_topology::HostTopology,
) -> usize {
    topo.cpu_to_node
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<usize>>()
        .len()
        .max(1)
}

// Most tests in this module are pure parsers / formatters / diff
// helpers that compile and pass on any target. The handful that
// actually read `/proc`, `/sys`, or assert `kernel_name == "Linux"`
// are individually gated with `#[cfg(target_os = "linux")]` at the
// test-fn level so non-Linux contributors still get coverage of the
// portable surface.

#[cfg(test)]
#[path = "host_context_tests.rs"]
mod tests;
