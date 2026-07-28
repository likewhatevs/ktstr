//! `cargo ktstr shell` — boot a KVM VM and drop into a busybox shell.
//!
//! Houses [`run_shell`], the dispatcher behind the `Shell` enum
//! variant in [`crate::cli::KtstrCommand`]. Kernel-spec resolution
//! delegates to [`crate::kernel::resolve_kernel_image`]; topology
//! parsing, include-file resolution, and KVM probe go through
//! [`ktstr::cli`]; the actual VM boot is [`ktstr::run_shell`].

use std::path::{Path, PathBuf};
use std::process::Command;

use ktstr::cli;
use ktstr::test_support::ShellTestDescriptor;

use super::probe::{ProbeError, probe_collect};
use crate::kernel::resolve_kernel_image;

/// Render the primary `ktstr shell --test <NAME>` banner string. Kept
/// pure (returns a `String`, no I/O) so unit tests can pin the
/// format. The caller writes the result to stderr immediately before
/// VM boot.
///
/// `operator_include_count` is the length of the operator's `-i /
/// --include-files` flag vec. `desc.extra_include_files.len()` is the test's own
/// include count. Both are UNIONed before being passed to the VM,
/// but they're surfaced separately in the banner so an operator can
/// see whether they accidentally dropped a `-i` flag.
fn format_test_banner(
    name: &str,
    desc: &ShellTestDescriptor,
    memory_mib: u32,
    operator_include_count: usize,
) -> String {
    let wprof_args_display = desc
        .wprof_args
        .as_deref()
        .map(|s| if s.is_empty() { "<empty>" } else { s }.to_string())
        .unwrap_or_else(|| "default".to_string());
    format!(
        "ktstr shell: test={} scheduler={} ({}) memory_mib={} \
         topology={}n{}l{}c{}t includes=test:{}+cli:{} \
         perf={} wprof_args={}",
        name,
        desc.scheduler_name,
        desc.scheduler_kind,
        memory_mib,
        desc.numa_nodes,
        desc.llcs,
        desc.cores,
        desc.threads,
        desc.extra_include_files.len(),
        operator_include_count,
        if desc.performance_mode { "on" } else { "off" },
        wprof_args_display,
    )
}

/// Probe each workspace test binary with `--ktstr-shell-test=<NAME>`
/// and resolve the named test's shell-relevant fields. Mirrors
/// `run_export`'s exit-code disambiguation: exit 0 = match + JSON;
/// exit 1 = not-registered-here (try next); exit 2 = registered
/// but rejected (`host_only`); other = error from a candidate.
///
/// Unlike `run_export`'s first-match-wins, this probe walks ALL
/// binaries and FAILS LOUDLY on ambiguous names: if `NAME` is
/// registered in two binaries, the operator gets a bail listing
/// every matching binary so they can rename or disambiguate via
/// `--package` (when that flag lands). Silent first-match-wins
/// would give the operator a different descriptor on different
/// days as `cargo metadata`'s iteration order shifts — a
/// no-silent-drops concern.
fn resolve_shell_from_test_entry(test: &str) -> Result<ShellTestDescriptor, String> {
    let test_flag = format!("--ktstr-shell-test={test}");

    let configure_cmd = |bin: &std::path::Path| {
        let mut cmd = Command::new(bin);
        cmd.arg(&test_flag)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        cmd
    };

    // Walk-all + ambiguity-bail — shell mode rejects rather than
    // picks one when multiple binaries claim the name, since
    // cargo metadata's iteration order is unstable across runs
    // (no-silent-drops + deterministic-resolution concern).
    let on_success = |bin: &std::path::Path,
                      out: &std::process::Output|
     -> Result<(PathBuf, ShellTestDescriptor), String> {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let desc = serde_json::from_str::<ShellTestDescriptor>(stdout.trim()).map_err(|e| {
            format!(
                "shell-test descriptor from {}: invalid JSON ({e}); \
                 candidate stdout: {stdout:?}",
                bin.display(),
            )
        })?;
        Ok((bin.to_path_buf(), desc))
    };

    // Match cargo-ktstr's own build profile so the fixture-resolution
    // `cargo build --tests` is a fingerprint hit against the test
    // artifacts the operator already built — a cold mismatch recompiles
    // the whole test set. cargo exposes no profile NAME at runtime;
    // debug_assertions tracks the release axis (on for dev/test, off for
    // release) and the workspace defines no custom profiles, so
    // dev/test↔release is the full space. (The e2e path sets
    // KTSTR_TEST_BINARY and skips this build entirely.)
    let release = !cfg!(debug_assertions);
    match probe_collect(None, release, configure_cmd, on_success) {
        Ok(matches) => {
            if matches.len() > 1 {
                let names: Vec<String> = matches
                    .iter()
                    .map(|(p, _)| p.display().to_string())
                    .collect();
                return Err(format!(
                    "test '{test}' is ambiguous — registered in {} workspace \
                     test binaries: [{}]. Rename one of the registrations \
                     or specify a single binary explicitly (--package not \
                     yet supported on shell mode).",
                    matches.len(),
                    names.join(", "),
                ));
            }
            let (_, desc) = matches
                .into_iter()
                .next()
                .expect("probe_collect Ok is non-empty per its contract");
            Ok(desc)
        }
        Err(ProbeError::Setup(msg)) => Err(msg),
        Err(ProbeError::Miss(miss)) => Err(miss.render(test, "cannot be used for shell mode")),
    }
}

/// Dispatch the `cargo ktstr shell` subcommand: launch a KVM VM and
/// drop into a busybox shell inside the guest.
///
/// `--cpu-cap` requiring `--no-perf-mode` is enforced at clap parse
/// time via `requires = "no_perf_mode"` on the Shell variant in
/// [`crate::cli`]; this dispatcher trusts the parser's gate and only
/// re-validates the env-level conflict (`KTSTR_BYPASS_LLC_LOCKS=1`)
/// that clap cannot see.
///
/// Both `unsafe std::env::set_var` calls below run on the call chain
/// `main` → match arm → `run_shell` with no thread spawn anywhere —
/// the binary's `tokio` features (`rt` only, no `rt-multi-thread`)
/// rule out runtime-spawned worker threads, and no helper here calls
/// `thread::spawn` before the env write. The single-threaded
/// invariant the SAFETY comments rely on holds for the lifetime of
/// these writes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_shell(
    kernel: Option<String>,
    test: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
    memory_mib: Option<u32>,
    dmesg: bool,
    exec: Option<String>,
    exec_timeout: std::time::Duration,
    no_perf_mode: bool,
    cpu_cap: Option<usize>,
    disk: Option<String>,
) -> Result<Option<i32>, String> {
    if no_perf_mode {
        // SAFETY: single-threaded at this point — main → dispatch →
        // run_shell, no prior thread spawn (ktstr's tokio feature set
        // is `rt` only, no `rt-multi-thread`; no helper above this
        // point calls `thread::spawn`). No concurrent env readers
        // exist, so set_var is sound.
        unsafe { std::env::set_var(ktstr::KTSTR_NO_PERF_MODE_ENV, "1") };
    }
    if let Some(cap) = cpu_cap {
        // Env-level conflict with KTSTR_BYPASS_LLC_LOCKS=1. The
        // CLI-level `--cpu-cap` requires `--no-perf-mode` rule is
        // enforced at clap parse time via `requires = "no_perf_mode"`
        // on the Shell variant in crate::cli, not here.
        if ktstr::bypass_llc_locks_active() {
            return Err(
                "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                 --cpu-cap is a resource contract; bypass disables the contract entirely."
                    .to_string(),
            );
        }
        // Validate early so a bad cap surfaces at CLI-parse time.
        cli::CpuCap::new(cap).map_err(|e| format!("{e:#}"))?;
        // SAFETY: single-threaded at this point per the chain
        // documented on the function — no concurrent env readers.
        unsafe { std::env::set_var(ktstr::KTSTR_CPU_CAP_ENV, cap.to_string()) };
    }
    // Validate every directly supplied VM-shape argument before probing the
    // host. A malformed topology or disk must remain the primary diagnostic
    // even on a machine without /dev/kvm.
    let parsed_topology = if test.is_none() {
        Some(cli::parse_topology_string(&topology).map_err(|e| format!("{e:#}"))?)
    } else {
        None
    };
    // Parse the human-readable disk size into a DiskConfig before the KVM
    // probe. `parse_disk_arg` returns `Ok(None)` when the attribute is absent
    // and applies `DiskConfig::default()` for every knob except
    // `capacity_mib` when present.
    let disk_cfg = cli::parse_disk_arg(disk.as_deref()).map_err(|e| format!("{e:#}"))?;
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    // No `--kernel`: default to the cwd when it is a kernel source
    // tree so `cargo ktstr shell` run from inside a tree builds and
    // boots that kernel without an explicit `--kernel .`. Scoped to
    // `shell`; other flows keep their cache/auto-download defaults.
    let kernel = cli::shell_kernel_or_cwd(kernel, "cargo ktstr");
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    // `--test <NAME>`: probe each workspace test binary for a
    // registered `#[ktstr_test]` entry named NAME and apply its
    // topology / memory_mib / extra_include_files / wprof_args /
    // performance_mode / scheduler_enable_cmds /
    // scheduler_disable_cmds to the shell VM. Print a one-line
    // banner to stderr BEFORE the VM boots so the operator sees
    // test-context up front (PS1-in-guest is a follow-up; this v1
    // keeps the wire-format simple). Mutex with `--topology` /
    // `--memory-mib` is enforced at clap parse time.
    // `--include-files -i` is ADDITIVE (concat with the test's
    // extra_include_files).
    //
    // When `--test` is absent the operator supplies the topology and
    // there is no test entry to derive scheduler-lifecycle commands,
    // wprof args, or performance mode from — each defaults to a
    // no-op: `None` wprof_args (use WprofConfig defaults if wprof
    // engages), `false` performance_mode, and empty scheduler
    // enable/disable command vecs.
    let (
        numa_nodes,
        llcs,
        cores,
        threads,
        resolved_memory,
        mut extra_includes,
        wprof_args,
        performance_mode,
        sched_enable_cmds,
        sched_disable_cmds,
    ) = if let Some(name) = test.as_deref() {
        let desc = resolve_shell_from_test_entry(name)?;
        // When the `wprof` feature is enabled, shell mode packs
        // the wprof binary and needs enough guest memory for BPF
        // ringbufs. Without the feature, `/bin/wprof` is not
        // available in the guest.
        let mem = desc.memory_mib;
        #[cfg(feature = "wprof")]
        let mem = ktstr::apply_wprof_memory_floor(mem, true);
        // Banner: print to stderr BEFORE VM boot. `--exec` consumers
        // see this on stderr (separate from the exec command's
        // stdout); interactive shell consumers see it as a header.
        // `includes=test:M+cli:N` is the count of include files the
        // VM will see — `M` from the test's `extra_include_files`,
        // `N` from the operator's `-i` flags. Both UNION into the
        // VM. Always emitted (even when both are 0) so the operator
        // immediately sees an accidentally-dropped include.
        ktstr::cli::print_status_line(&format_test_banner(name, &desc, mem, include_files.len()));
        if desc.scheduler_kind == ktstr::test_support::SchedulerKind::KernelBuiltin {
            if desc.scheduler_enable_cmds.is_empty() {
                ktstr::ktstr_status!(
                    "ktstr shell: scheduler '{}' is KernelBuiltin with no enable cmds \
                     declared — drop-to-shell will run under the kernel default; \
                     refer to the test's #[ktstr_test(...)] attributes for sysctl \
                     to repro the workload.",
                    desc.scheduler_name,
                );
            } else {
                ktstr::ktstr_status!(
                    "ktstr shell: scheduler '{}' is KernelBuiltin — running {} enable \
                     cmd(s) before drop-to-shell and {} disable cmd(s) on shell exit. \
                     You can manually re-disable inside busybox if you want to inspect \
                     the kernel default mid-session.",
                    desc.scheduler_name,
                    desc.scheduler_enable_cmds.len(),
                    desc.scheduler_disable_cmds.len(),
                );
            }
        } else if desc.scheduler_kind != ktstr::test_support::SchedulerKind::Eevdf {
            ktstr::ktstr_status!(
                "ktstr shell: repro the workload by invoking the scheduler binary \
                 inside the guest (e.g. /bin/{}) — its sched_args are encoded in \
                 the test source; this v1 doesn't stage the scheduler binary \
                 into the guest, so you may need to copy it via additional `-i`.",
                desc.scheduler_name,
            );
        }
        (
            desc.numa_nodes,
            desc.llcs,
            desc.cores,
            desc.threads,
            Some(mem),
            desc.extra_include_files
                .into_iter()
                .map(PathBuf::from)
                .collect::<Vec<_>>(),
            desc.wprof_args,
            desc.performance_mode,
            desc.scheduler_enable_cmds,
            desc.scheduler_disable_cmds,
        )
    } else {
        let (n, l, c, t) = parsed_topology
            .expect("a shell invocation without --test validates its explicit topology");
        (
            n,
            l,
            c,
            t,
            memory_mib,
            Vec::new(),
            None,
            false,
            Vec::new(),
            Vec::new(),
        )
    };

    // Operator's `-i` includes UNION the test's extra_include_files.
    extra_includes.extend(include_files.iter().cloned());

    let resolved_includes =
        cli::resolve_include_files(&extra_includes).map_err(|e| format!("{e:#}"))?;

    let include_refs: Vec<(&str, &Path)> = resolved_includes
        .iter()
        .map(|(a, p)| (a.as_str(), p.as_path()))
        .collect();

    // Borrow the owned String vecs as &[&str] for the lib-side call.
    let sched_enable_refs: Vec<&str> = sched_enable_cmds.iter().map(String::as_str).collect();
    let sched_disable_refs: Vec<&str> = sched_disable_cmds.iter().map(String::as_str).collect();

    ktstr::run_shell(
        kernel_path,
        numa_nodes,
        llcs,
        cores,
        threads,
        &include_refs,
        resolved_memory,
        dmesg,
        exec.as_deref(),
        exec_timeout,
        disk_cfg,
        wprof_args.as_deref(),
        performance_mode,
        &sched_enable_refs,
        &sched_disable_refs,
    )
    .map_err(|e| format!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_desc(extras: Vec<&'static str>) -> ShellTestDescriptor {
        ShellTestDescriptor {
            numa_nodes: 1,
            llcs: 1,
            cores: 2,
            threads: 1,
            memory_mib: 1024,
            wprof: false,
            extra_include_files: extras.into_iter().map(String::from).collect(),
            scheduler_name: "scx_rusty".to_string(),
            scheduler_kind: ktstr::test_support::SchedulerKind::Discover,
            wprof_args: None,
            performance_mode: false,
            scheduler_enable_cmds: Vec::new(),
            scheduler_disable_cmds: Vec::new(),
        }
    }

    #[test]
    fn banner_zero_includes_emits_test_0_cli_0() {
        let desc = sample_desc(vec![]);
        let line = format_test_banner("my_test", &desc, 2048, 0);
        assert!(
            line.contains("includes=test:0+cli:0"),
            "banner must surface zero counts so a dropped -i is obvious; got: {line}",
        );
    }

    #[test]
    fn banner_operator_include_count_propagates() {
        let desc = sample_desc(vec![]);
        let line = format_test_banner("my_test", &desc, 2048, 3);
        assert!(
            line.contains("includes=test:0+cli:3"),
            "banner must echo operator -i count (3); got: {line}",
        );
    }

    #[test]
    fn banner_unions_test_and_cli_include_counts() {
        let desc = sample_desc(vec!["a:/x", "b:/y"]);
        let line = format_test_banner("my_test", &desc, 2048, 4);
        assert!(
            line.contains("includes=test:2+cli:4"),
            "banner must show both test (2) and cli (4) counts \
             separately so the operator can verify each source; got: {line}",
        );
    }

    #[test]
    fn banner_preserves_topology_and_memory_fields() {
        let mut desc = sample_desc(vec![]);
        desc.numa_nodes = 2;
        desc.llcs = 4;
        desc.cores = 6;
        desc.threads = 2;
        let line = format_test_banner("topo_test", &desc, 4096, 0);
        assert!(line.contains("topology=2n4l6c2t"), "topology axes: {line}");
        assert!(line.contains("memory_mib=4096"), "memory: {line}");
        assert!(line.contains("test=topo_test"), "test name: {line}");
    }

    #[test]
    fn banner_defaults_show_perf_off_and_wprof_default() {
        let desc = sample_desc(vec![]);
        let line = format_test_banner("plain", &desc, 2048, 0);
        assert!(
            line.contains("perf=off"),
            "banner must surface performance_mode=false as perf=off so \
             operator sees the absence of vCPU pinning / SCHED_FIFO / \
             hugepages; got: {line}",
        );
        assert!(
            line.contains("wprof_args=default"),
            "banner must surface wprof_args=None as wprof_args=default \
             so operator distinguishes 'no override' from 'override with \
             empty args' (which renders as wprof_args=<empty>); got: {line}",
        );
    }

    #[test]
    fn banner_surfaces_perf_on_and_custom_wprof_args() {
        let mut desc = sample_desc(vec![]);
        desc.performance_mode = true;
        desc.wprof_args = Some("--sched-events --kstack".to_string());
        let line = format_test_banner("perf_test", &desc, 2048, 0);
        assert!(
            line.contains("perf=on"),
            "banner must surface performance_mode=true as perf=on so \
             the operator can correlate observed timings with the \
             host-side optimization stack; got: {line}",
        );
        assert!(
            line.contains("wprof_args=--sched-events --kstack"),
            "banner must echo the wprof_args override verbatim so \
             reproducing the same flags by hand is one copy-paste; \
             got: {line}",
        );
    }

    #[test]
    fn banner_distinguishes_empty_wprof_override_from_default() {
        let mut desc = sample_desc(vec![]);
        desc.wprof_args = Some(String::new());
        let line = format_test_banner("empty_args", &desc, 2048, 0);
        assert!(
            line.contains("wprof_args=<empty>"),
            "Some(\"\") MUST render as <empty> to disambiguate from \
             None (which renders as default) — the two have different \
             semantics per the run_shell contract; got: {line}",
        );
    }
}
