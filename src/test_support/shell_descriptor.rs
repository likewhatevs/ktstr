//! Shared `ShellTestDescriptor` for the `--ktstr-shell-test=<NAME>`
//! probe wire format.
//!
//! Producer: `dispatch::maybe_dispatch_shell_test` serializes a
//! fully-populated descriptor to stdout when a test binary is probed.
//!
//! Consumer: `cargo_ktstr::misc::shell::resolve_shell_from_test_entry`
//! deserializes the stdout, then `run_shell` passes the descriptor's
//! fields to [`crate::run_shell`] so the shell VM mirrors the test's
//! topology, scheduler, wprof, and performance settings.
//!
//! Older `cargo-ktstr` binaries can deserialize JSON emitted by a
//! newer test binary that adds a field because the derived
//! `Deserialize` impl ignores unknown fields (no
//! `#[serde(deny_unknown_fields)]`): the field the old struct doesn't
//! know about is dropped, and existing fields remain populated. The
//! reverse direction (newer `cargo-ktstr` reading older JSON that
//! lacks a field the new code added) works because every field carries
//! `#[serde(default)]`, which supplies a default for each missing
//! field.

use serde::{Deserialize, Serialize};

/// Discriminator for the test's scheduler-spec shape on the
/// `--ktstr-shell-test=<NAME>` wire format. Wire-byte-compatible with
/// the prior stringly-typed `"eevdf" | "discover" | "path" |
/// "kernel_builtin"` values via `#[serde(rename_all = "snake_case")]`;
/// the typed enum replaces the stringly-typed boundary so a rename on
/// either side (producer in `dispatch::maybe_dispatch_shell_test`,
/// consumer in `cargo_ktstr::misc::shell`) is a compile error instead
/// of a silent banner-emit-gate regression.
///
/// 1:1 with [`crate::test_support::SchedulerSpec`]'s 4 variants; the
/// payload data (scheduler binary name, KernelBuiltin enable/disable
/// commands) rides separately on the descriptor's `scheduler_name` /
/// `scheduler_enable_cmds` / `scheduler_disable_cmds` fields so this
/// type carries only the discriminator.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerKind {
    /// Kernel-default scheduling — no userspace binary, no
    /// kernel-builtin lifecycle. The no-scx control.
    #[default]
    Eevdf,
    /// Userspace scx binary located via `resolve_scheduler` cascade
    /// (name-only, path is discovered at run time).
    Discover,
    /// Userspace scx binary at a fully-qualified path.
    Path,
    /// In-kernel scheduling class activated via `scheduler_enable_cmds`
    /// before workload start, torn down via `scheduler_disable_cmds`
    /// on shell exit / test teardown.
    KernelBuiltin,
}

impl std::fmt::Display for SchedulerKind {
    /// Renders the snake_case wire form (`"eevdf"`, `"discover"`,
    /// `"path"`, `"kernel_builtin"`) so banner format strings + log
    /// lines stay byte-compatible with the prior stringly-typed
    /// behavior.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SchedulerKind::Eevdf => "eevdf",
            SchedulerKind::Discover => "discover",
            SchedulerKind::Path => "path",
            SchedulerKind::KernelBuiltin => "kernel_builtin",
        };
        f.write_str(s)
    }
}

impl From<&crate::test_support::SchedulerSpec> for SchedulerKind {
    /// Single source of truth for the SchedulerSpec → SchedulerKind
    /// mapping. Exhaustive match so adding a 5th SchedulerSpec
    /// variant triggers a compile error here, forcing the
    /// discriminator to grow in lockstep.
    fn from(spec: &crate::test_support::SchedulerSpec) -> Self {
        match spec {
            crate::test_support::SchedulerSpec::Eevdf => SchedulerKind::Eevdf,
            crate::test_support::SchedulerSpec::Discover(_) => SchedulerKind::Discover,
            crate::test_support::SchedulerSpec::Path(_) => SchedulerKind::Path,
            crate::test_support::SchedulerSpec::KernelBuiltin { .. } => {
                SchedulerKind::KernelBuiltin
            }
        }
    }
}

/// Wire-format descriptor exchanged between a test binary and
/// `cargo ktstr shell --test <NAME>` to let the shell VM mirror the
/// named `#[ktstr_test]`'s topology, scheduler, wprof config, and
/// performance mode.
///
/// `scheduler_kind` is a typed [`SchedulerKind`] discriminator so the
/// banner can hint at how to repro the scheduler (Discover/Path =
/// userspace binary at `/bin/<n>`; KernelBuiltin = no binary, runs
/// `scheduler_enable_cmds` before drop-to-shell and
/// `scheduler_disable_cmds` on shell exit; Eevdf = no setup needed).
///
/// `scheduler_enable_cmds` and `scheduler_disable_cmds` are extracted
/// from the [`crate::test_support::SchedulerSpec::KernelBuiltin`]
/// variant's `enable` and `disable` slices respectively; the other
/// three variants emit empty vecs (no kernel-builtin shell ops to
/// invoke).
///
/// `wprof_args`: requires the `wprof` cargo feature. When the
/// feature is enabled and `Some`, replaces `WprofConfig::args`;
/// without the feature, the value is ignored by `run_shell`.
/// `None` means "use the default wprof args."
///
/// `performance_mode` mirrors the test's
/// `#[ktstr_test(performance_mode)]` attribute so the shell VM
/// reproduces vCPU pinning, hugepages, NUMA mbind, and SCHED_FIFO
/// promotion when the test requested them.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellTestDescriptor {
    #[serde(default)]
    pub numa_nodes: u32,
    #[serde(default)]
    pub llcs: u32,
    #[serde(default)]
    pub cores: u32,
    #[serde(default)]
    pub threads: u32,
    #[serde(default)]
    pub memory_mib: u32,
    #[serde(default)]
    pub wprof: bool,
    #[serde(default)]
    pub extra_include_files: Vec<String>,
    #[serde(default)]
    pub scheduler_name: String,
    #[serde(default)]
    pub scheduler_kind: SchedulerKind,
    /// Custom wprof CLI args (requires the `wprof` cargo feature).
    /// When `Some` and the feature is enabled, the shell VM
    /// overrides `WprofConfig::args` with the space-tokenised
    /// value before booting.
    #[serde(default)]
    pub wprof_args: Option<String>,
    /// Mirrors `KtstrTestEntry::performance_mode`. The shell VM
    /// forwards this to
    /// `crate::vmm::KtstrVmBuilder::performance_mode`.
    #[serde(default)]
    pub performance_mode: bool,
    /// Shell commands invoked before drop-to-busybox when the test's
    /// scheduler is a
    /// [`crate::test_support::SchedulerSpec::KernelBuiltin`] variant —
    /// empty vec for the other three variants. Populated from the
    /// variant's `enable` slice.
    #[serde(default)]
    pub scheduler_enable_cmds: Vec<String>,
    /// Shell commands invoked on shell exit when the test's scheduler
    /// is a [`crate::test_support::SchedulerSpec::KernelBuiltin`]
    /// variant — empty vec for the other three variants. Populated
    /// from the variant's `disable` slice.
    #[serde(default)]
    pub scheduler_disable_cmds: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fully_populated() -> ShellTestDescriptor {
        ShellTestDescriptor {
            numa_nodes: 2,
            llcs: 4,
            cores: 6,
            threads: 2,
            memory_mib: 4096,
            wprof: true,
            extra_include_files: vec!["a:/x".to_string(), "b:/y".to_string()],
            scheduler_name: "scx_test".to_string(),
            scheduler_kind: SchedulerKind::KernelBuiltin,
            wprof_args: Some("-d 2000 -e sched,irq".to_string()),
            performance_mode: true,
            scheduler_enable_cmds: vec!["echo on > /sys/kernel/debug/foo".to_string()],
            scheduler_disable_cmds: vec!["echo off > /sys/kernel/debug/foo".to_string()],
        }
    }

    #[test]
    fn roundtrip_preserves_every_field() {
        let original = fully_populated();
        let json = serde_json::to_string(&original).expect("serialize fully-populated descriptor");
        let parsed: ShellTestDescriptor =
            serde_json::from_str(&json).expect("deserialize fully-populated descriptor");
        assert_eq!(parsed, original);
    }

    #[test]
    fn missing_new_fields_default_to_empty() {
        // JSON shape emitted by an older test binary that doesn't
        // know about wprof_args, performance_mode,
        // scheduler_enable_cmds, or scheduler_disable_cmds. The
        // four new fields are absent — each must take its serde
        // default (None, false, empty vec) rather than failing the
        // deserialize.
        let legacy_json = r#"{
            "numa_nodes": 1,
            "llcs": 1,
            "cores": 2,
            "threads": 1,
            "memory_mib": 1024,
            "wprof": false,
            "extra_include_files": [],
            "scheduler_name": "scx_legacy",
            "scheduler_kind": "discover"
        }"#;
        let parsed: ShellTestDescriptor =
            serde_json::from_str(legacy_json).expect("legacy JSON missing new fields must parse");
        assert_eq!(parsed.wprof_args, None);
        assert!(!parsed.performance_mode);
        assert!(parsed.scheduler_enable_cmds.is_empty());
        assert!(parsed.scheduler_disable_cmds.is_empty());
        // Existing fields still populate correctly.
        assert_eq!(parsed.numa_nodes, 1);
        assert_eq!(parsed.scheduler_name, "scx_legacy");
        // Pin the snake_case-string → enum conversion at the
        // legacy-JSON forward-compat boundary: a regression that
        // dropped the rename-all gate would deserialize Discover
        // as Eevdf default and break the consumer's banner branch.
        assert_eq!(parsed.scheduler_kind, SchedulerKind::Discover);
    }

    /// Per-variant wire-roundtrip pin. Catches regressions that
    /// rename a variant's snake_case form (e.g. accidentally
    /// dropping the `_` in `kernel_builtin`) or break a single
    /// variant's serde codepath while leaving the others working.
    #[test]
    fn scheduler_kind_serde_each_variant_roundtrips_snake_case() {
        for (variant, wire) in [
            (SchedulerKind::Eevdf, "\"eevdf\""),
            (SchedulerKind::Discover, "\"discover\""),
            (SchedulerKind::Path, "\"path\""),
            (SchedulerKind::KernelBuiltin, "\"kernel_builtin\""),
        ] {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, wire, "serialize mismatch for {variant:?}");
            let back: SchedulerKind = serde_json::from_str(wire).expect("deserialize");
            assert_eq!(back, variant, "roundtrip mismatch for {variant:?}");
        }
    }

    /// Display impl must produce the same lowercase strings the
    /// wire format does. Banner formatting depends on this
    /// equivalence — a drift between Display and serde would
    /// silently print a string the operator can't grep-correlate
    /// with the JSON shape they'd see in any tooling output.
    #[test]
    fn scheduler_kind_display_matches_serde_snake_case() {
        for variant in [
            SchedulerKind::Eevdf,
            SchedulerKind::Discover,
            SchedulerKind::Path,
            SchedulerKind::KernelBuiltin,
        ] {
            let json = serde_json::to_string(&variant).expect("serialize");
            let unquoted = json.trim_matches('"');
            assert_eq!(
                variant.to_string(),
                unquoted,
                "Display must equal serde wire form for {variant:?}",
            );
        }
    }

    /// From<&SchedulerSpec> exhaustive coverage — pins each
    /// variant's mapping. Catches typo regressions like
    /// `SchedulerSpec::Path → SchedulerKind::Discover` that the
    /// compile-time exhaustive match wouldn't catch (both arms
    /// would still type-check; the wrong-mapping is silent).
    #[test]
    fn scheduler_kind_from_scheduler_spec_per_variant() {
        use crate::test_support::SchedulerSpec;
        assert_eq!(
            SchedulerKind::from(&SchedulerSpec::Eevdf),
            SchedulerKind::Eevdf,
        );
        assert_eq!(
            SchedulerKind::from(&SchedulerSpec::Discover("scx_test")),
            SchedulerKind::Discover,
        );
        assert_eq!(
            SchedulerKind::from(&SchedulerSpec::Path("/bin/scx_test")),
            SchedulerKind::Path,
        );
        assert_eq!(
            SchedulerKind::from(&SchedulerSpec::KernelBuiltin {
                enable: &[],
                disable: &[],
            }),
            SchedulerKind::KernelBuiltin,
        );
    }

    /// Unknown wire values MUST fail deserialize rather than
    /// silently fall back to a default. Pins the typed-enum
    /// strictness: a regression to `#[serde(other)]` catchall or
    /// re-introduction of a String field would silently accept the
    /// junk value and break consumer banner gates.
    #[test]
    fn scheduler_kind_unknown_variant_fails_deserialize() {
        let r: Result<SchedulerKind, _> = serde_json::from_str("\"rust\"");
        assert!(r.is_err(), "unknown variant 'rust' must reject; got {r:?}",);
    }

    #[test]
    fn missing_every_field_yields_full_defaults() {
        // Pathological case: an empty object. Every field is
        // #[serde(default)] so deserialize succeeds and every
        // field becomes its type default. Forward-compat applies to
        // existing fields too — a future cargo-ktstr that adds yet
        // another field shouldn't choke on an old JSON missing the
        // new field, and the same logic protects existing fields
        // against a JSON shape they happen to be missing from.
        let empty_json = "{}";
        let parsed: ShellTestDescriptor =
            serde_json::from_str(empty_json).expect("empty JSON must parse with all defaults");
        assert_eq!(parsed.numa_nodes, 0);
        assert_eq!(parsed.scheduler_name, "");
        // SchedulerKind::Default = Eevdf, the no-scx control — the
        // safest fallback for an empty object: the consumer's
        // KernelBuiltin/non-Eevdf banner gates won't fire, matching
        // the prior empty-string behavior where the !="kernel_builtin"
        // and !="eevdf" string compares fell through.
        assert_eq!(parsed.scheduler_kind, SchedulerKind::Eevdf);
        assert!(parsed.extra_include_files.is_empty());
        assert_eq!(parsed.wprof_args, None);
        assert!(!parsed.performance_mode);
        assert!(parsed.scheduler_enable_cmds.is_empty());
        assert!(parsed.scheduler_disable_cmds.is_empty());
    }
}
