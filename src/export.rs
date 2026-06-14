//! `cargo ktstr export` — package a registered test as a self-extracting
//! `.run` file that reproduces the scenario on bare metal without a VM.
//!
//! [`export_test`] is the entry point invoked from the test binary's
//! `#[ctor]` dispatch (see
//! `crate::test_support::dispatch::maybe_dispatch_export`) when
//! `cargo ktstr export` exec's the binary with
//! `--ktstr-export-test=NAME`. The export pipeline locates the named
//! test in the `KTSTR_TESTS` distributed slice, gathers the
//! binaries it needs (the running test binary itself via
//! `current_exe()`, the scheduler binary, and per-test include
//! files), tarballs them with gzip, and emits a single shell script:
//!
//! ```text
//! #!/bin/bash
//! ... preamble: root check, prereq check, sched_ext conflict check,
//!     topology check, arg parsing, mktemp+trap, archive extract,
//!     scheduler launch, test run ...
//! __ARCHIVE__
//! <base64-encoded gzipped tarball>
//! ```
//!
//! The result is `chmod +x` so the operator can `./repro.run`
//! directly on a target host. `ktstr run --ktstr-test-fn <name>` is
//! the same dispatch the in-guest test harness already uses
//! (`test_support::eval` invokes it after VM boot), so the bare-metal
//! path reuses every existing test entry — no separate registry, no
//! rebuilt scenarios.
//!
//! # Why bare-metal repro?
//!
//! The framework's primary execution path runs every test inside a
//! KVM VM. That gets us deterministic topology, fast spin-up, and
//! kernel/scheduler isolation; it also abstracts away from real
//! hardware. When a test fails on bare metal but passes in the VM
//! (or vice versa) the operator wants to bisect. A self-contained
//! `.run` file means they can hand the failing test to any host with
//! a compatible kernel and topology, run it without re-building the
//! workspace, and capture the output through ordinary stdout/stderr
//! channels.
//!
//! # Out of scope
//!
//! - `host_only` tests: they orchestrate cargo invocations and nested
//!   VMs themselves; running them outside the framework's harness
//!   isn't useful.
//! - `bpf_map_write` tests: they need the framework's runtime
//!   probe-based map-write surface, not yet replicated outside the
//!   VM dispatch.
//! - `KernelBuiltin` schedulers: they activate via shell commands
//!   (`enable` / `disable` slots on the spec) rather than launching a
//!   userspace binary. The preamble doesn't generate those commands
//!   in v1; export rejects the variant with an actionable error.
//!
//! # Include-file directories
//!
//! The framework's full include-file resolver (re-exported as
//! [`crate::cli::resolve_include_files`]) walks directories
//! recursively and produces an `archive_path/host_path` map that
//! preserves directory structure. Export uses a simpler subset:
//! every included entry must be a regular file, and the archive
//! layout flattens by basename to `include/<basename>`. Directory
//! specs error with EISDIR. Recursive directory packaging is a v2
//! enhancement.
//!
//! # Bash-only
//!
//! The preamble's heredoc shebang names `/bin/bash` and uses
//! features bash carries — indexed-array syntax
//! (`RUN_ARGS=(...)`, `${RUN_ARGS[@]}` expansion) and
//! `set -o pipefail` (the `o`-form syntax). Bourne / dash /
//! busybox sh would mis-parse the script; the operator must run
//! on a host with bash installed.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use flate2::Compression;
use flate2::write::GzEncoder;

use crate::test_support::{
    KtstrTestEntry, SchedulerSpec, content_hash, find_test, resolve_scheduler, scratch_dir,
};

/// Build a self-extracting `.run` file for the given test.
///
/// `test_name` must match a `#[ktstr_test]` registration's `name`
/// field exactly (case-sensitive). Use `cargo nextest list` to
/// enumerate names; strip the `<binary>::` prefix the way
/// `cargo ktstr show-thresholds` does.
///
/// `output` is the destination path. `None` defaults to
/// `<test_name>.run` in the current directory. The output file is
/// written with mode 0o755 so the operator can invoke it directly.
pub fn export_test(test_name: &str, output: Option<PathBuf>) -> Result<()> {
    let entry = find_test(test_name)
        .ok_or_else(|| anyhow::anyhow!("no registered test named '{test_name}'"))?;

    if entry.host_only {
        bail!(
            "test '{test_name}' is host_only — it orchestrates cargo / nested VMs \
             from inside the test body and cannot be reproduced outside the \
             framework harness. host_only tests are out of scope for export."
        );
    }
    if !entry.bpf_map_write.is_empty() {
        bail!(
            "test '{test_name}' uses bpf_map_write — runtime BPF map writes are \
             driven by the framework's host-side probe machinery, which is not \
             reproduced bare-metal. bpf_map_write tests are out of scope for v1 \
             export."
        );
    }
    // KernelBuiltin schedulers don't ship a userspace binary; they
    // activate via shell commands stored on the spec's `enable` /
    // `disable` slots. The framework runs those commands in the VM
    // around the scheduler binary launch (eval.rs builds
    // sched_enable_cmds / sched_disable_cmds on the VmBuilder). The
    // preamble in v1 does not generate equivalent shell commands —
    // running the .run file on a host without applying those
    // settings would silently mis-launch the scheduler. Reject with
    // an actionable diagnostic.
    if let SchedulerSpec::KernelBuiltin { .. } = &entry.scheduler.binary {
        bail!(
            "test '{test_name}' uses a KernelBuiltin scheduler — it activates via \
             host-side shell commands (`enable` / `disable` slots) rather than a \
             userspace binary. The export preamble does not yet emit those \
             commands; KernelBuiltin export is out of scope for v1."
        );
    }

    let test_binary =
        std::env::current_exe().context("locate the current test binary via /proc/self/exe")?;

    let scheduler_path = resolve_scheduler_for_export(entry)?;
    let mut include_files = resolve_include_files(entry)?;
    let config_additions = compute_config_export_additions(entry)
        .context("resolve scheduler config file for export")?;
    for addition in &config_additions {
        include_files.push(addition.host_path.clone());
    }

    let output_path = output.unwrap_or_else(|| PathBuf::from(format!("{test_name}.run")));

    let archive = build_archive(&test_binary, scheduler_path.as_deref(), &include_files)
        .context("build embedded gzip tarball")?;

    let preamble = generate_preamble(entry, scheduler_path.is_some(), &config_additions);

    write_runfile(&output_path, &preamble, &archive)
        .with_context(|| format!("write runfile to {}", output_path.display()))?;

    eprintln!(
        "wrote {} ({} bytes archive, {} include files)",
        output_path.display(),
        archive.len(),
        include_files.len()
    );
    Ok(())
}

/// Resolve the scheduler binary for an entry, returning `None` for
/// EEVDF / kernel-builtin payloads (which don't ship a binary).
///
/// Reuses `crate::test_support::eval::resolve_scheduler` so the
/// resolution matches the in-guest path: `KTSTR_SCHEDULER` env wins;
/// then, outside cargo-test-mode, a fresh `cargo build -p {name}` so an
/// edited scheduler is never exported stale, falling back to a
/// pre-built sibling-exe / target/debug / target/release binary when
/// the build cannot run. cargo-test-mode resolves via `$PATH` first.
fn resolve_scheduler_for_export(entry: &KtstrTestEntry) -> Result<Option<PathBuf>> {
    let (path, _source) = resolve_scheduler(&entry.scheduler.binary)
        .with_context(|| format!("resolve scheduler binary for test '{}'", entry.name))?;
    Ok(path)
}

/// A scheduler config file or inline content contributes one
/// host-side file plus one set of CLI arguments to the export.
///
/// Mirrors the in-VM path at
/// `crate::test_support::eval::run_ktstr_test_inner` which calls
/// `crate::test_support::runtime::config_file_parts` +
/// `crate::test_support::runtime::config_content_parts` to
/// resolve the same two slots.
#[derive(Debug)]
struct ConfigExportAddition {
    /// Host-filesystem path of the config file. For
    /// `scheduler.config_file` (on-disk file), this is the file
    /// itself. For `entry.config_content` (inline content), this
    /// is a temp file written under `$TMPDIR` containing the
    /// content bytes; the export-side caller doesn't differentiate
    /// — both go through the same `build_archive` path.
    host_path: PathBuf,
    /// Shell-ready CLI argument string PREPENDED to the launched
    /// scheduler's argv at preamble-render time so the rendered
    /// argv matches the in-VM eval.rs:1112-1125 ordering
    /// (`--config FIRST, append_base_sched_args LAST`). Uses
    /// `"$DIR/include/<basename>"` (with the `$DIR` shell variable
    /// preserved for runtime expansion by the .run extractor) so
    /// the path resolves to the operator's extracted .run tree on
    /// the target host. No leading space — the caller manages
    /// spacing between additions and the base args.
    args_shell_prefix: String,
}

/// Compute the scheduler-config export additions for an entry.
///
/// Returns 0, 1, or 2 additions matching the in-VM path's
/// dual-slot handling:
///   - `entry.scheduler.config_file` (`Option<host path>`)
///   - `entry.config_content` (`Option<inline content>`) paired
///     with `entry.scheduler.config_file_def`
///     (`Option<(arg_template, guest_path)>`)
///
/// Both slots are processed independently; in practice
/// `crate::test_support::KtstrTestEntry::validate` gates
/// `config_content` to require a matching `config_file_def` and
/// rejects an unpaired `config_content`, so the inline path emits
/// at most one addition. The on-disk path is orthogonal and could
/// in theory co-exist with the inline path — handled in lockstep
/// with the in-VM eval.rs behavior so a single .run runs the
/// scheduler with the same argv as a normal test invocation.
fn compute_config_export_additions(entry: &KtstrTestEntry) -> Result<Vec<ConfigExportAddition>> {
    let mut out = Vec::new();
    if let Some(addition) = config_file_addition(entry)? {
        out.push(addition);
    }
    if let Some(addition) = config_content_addition(entry)? {
        out.push(addition);
    }
    Ok(out)
}

/// Translate `entry.scheduler.config_file` (`Option<host path>`)
/// into a [`ConfigExportAddition`]. Hardcoded `--config` arg
/// matches the in-VM behavior at
/// `crate::test_support::runtime::config_file_parts` and the
/// surrounding push at `eval.rs`.
fn config_file_addition(entry: &KtstrTestEntry) -> Result<Option<ConfigExportAddition>> {
    let Some(config_path) = entry.scheduler.config_file else {
        return Ok(None);
    };
    let host_path = PathBuf::from(config_path);
    if !host_path.exists() {
        bail!(
            "scheduler '{}' declares config_file {} but the file is not present on the host",
            entry.scheduler.name,
            host_path.display()
        );
    }
    // Mirror resolve_include_files's directory-reject (export.rs near
    // is_dir() rejection in that helper): export packs regular files,
    // so a directory-shaped config_file would silently fail later in
    // build_archive's `std::fs::read` with a less-actionable EISDIR.
    if host_path.is_dir() {
        bail!(
            "scheduler '{}' declares config_file {} but the path is a directory — \
             config_file must point at a regular file. Recursive directory packaging \
             is a v2 enhancement; for now, list a single file or split the directory \
             contents across `include_files` declarations.",
            entry.scheduler.name,
            host_path.display()
        );
    }
    let basename = host_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "scheduler config_file {} has no valid basename",
                host_path.display()
            )
        })?
        .to_string();
    reject_shell_metacharacters_in_basename(&basename, &host_path.display().to_string())?;
    let args_shell_prefix = format!("--config \"$DIR/include/{basename}\"");
    Ok(Some(ConfigExportAddition {
        host_path,
        args_shell_prefix,
    }))
}

/// Translate `entry.config_content` (`Option<inline content>`) +
/// `entry.scheduler.config_file_def`
/// (`Option<(arg_template, guest_path)>`) into a
/// [`ConfigExportAddition`] by writing the
/// content bytes to a temp file under `$TMPDIR` and substituting
/// `{file}` in the arg template with the export-side runtime path
/// `"$DIR/include/<basename>"`.
///
/// The basename derives from the scheduler's declared guest path
/// (e.g. `/include-files/layers.json` → `layers.json`) so a
/// scheduler family that declares a stable guest_path naming
/// convention sees the same basename in the .run archive that it
/// would see in the in-VM /include-files mount.
fn config_content_addition(entry: &KtstrTestEntry) -> Result<Option<ConfigExportAddition>> {
    let Some(content) = entry.config_content else {
        return Ok(None);
    };
    let Some((arg_template, guest_path)) = entry.scheduler.config_file_def else {
        return Ok(None);
    };
    let basename = std::path::Path::new(guest_path)
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "scheduler '{}' config_file_def guest_path '{}' has no valid basename",
                entry.scheduler.name,
                guest_path
            )
        })?
        .to_string();
    reject_shell_metacharacters_in_basename(&basename, guest_path)?;
    let hash = content_hash(content);
    // Write to a uniquely-named scratch file inside the shared
    // process-owned 0o700 scratch directory, then atomic-rename to
    // the canonical content-addressed path. See
    // [`crate::test_support::runtime::scratch_dir`] for the
    // symlink-defense + leak-bound rationale — that helper is
    // shared with the in-VM `config_content_parts` path so both
    // sites get the same atexit cleanup + per-process 0o700
    // directory without divergent maintenance.
    let dir = scratch_dir();
    let canonical = dir.join(format!("ktstr-export-config-{hash:016x}-{basename}"));
    let mut scratch = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| "create ktstr export-config scratch file")?;
    scratch
        .as_file_mut()
        .write_all(content.as_bytes())
        .with_context(|| "write inline config_content to scratch")?;
    scratch.persist(&canonical).with_context(|| {
        format!(
            "atomic-rename export-config scratch to {}",
            canonical.display()
        )
    })?;
    let runtime_path = format!("\"$DIR/include/{basename}\"");
    let expanded = arg_template.replace("{file}", &runtime_path);
    Ok(Some(ConfigExportAddition {
        host_path: canonical,
        args_shell_prefix: expanded,
    }))
}

/// Reject basenames containing shell-metacharacters that would
/// break the double-quote-interpolation context the addition's
/// `args_shell_prefix` ends up in (e.g.
/// `--config "$DIR/include/<basename>"`). Test-author input is
/// trusted (static string slots), but defense-in-depth catches
/// any future regression that would land a basename with `"`,
/// `\`, `$`, or `` ` `` and silently produce a broken .run script.
fn reject_shell_metacharacters_in_basename(basename: &str, source: &str) -> Result<()> {
    for c in basename.chars() {
        if c == '"' || c == '\\' || c == '$' || c == '`' {
            bail!(
                "scheduler config file basename {basename:?} (from {source}) contains shell-metacharacter {c:?}; \
                 this would break the double-quoted .run preamble interpolation. \
                 Rename the file to use only ASCII letters, digits, `_`, `-`, and `.`."
            );
        }
    }
    Ok(())
}

/// Resolve every `all_include_files()` spec to a host-side path.
///
/// The framework's full PATH / directory-walking resolver lives at
/// [`crate::cli::resolve_include_files`] and returns an
/// `archive_path/host_path` map that preserves recursive directory
/// structure. Export uses a deliberately simpler subset:
///   - explicit absolute paths → use as-is when they exist
///   - explicit relative paths (containing `/` or starting with `.`)
///     → relative to current dir
///   - bare names → search `PATH`
///   - directories → reject with an actionable "is a directory" error
///     (export packs regular files only; recursive directory
///     packaging is a v2 enhancement)
///
/// The simpler layout (flat `include/<basename>`) keeps the
/// extracted .run tree predictable for the operator, at the cost of
/// not handling tests whose include specs name directories.
///
/// Missing files are surfaced as a hard error so the operator can
/// fix the include spec rather than discovering the gap on the
/// target host.
fn resolve_include_files(entry: &KtstrTestEntry) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for spec in entry.all_include_files() {
        let path = if spec.starts_with('/')
            || spec.starts_with("./")
            || spec.starts_with("../")
            || spec.contains('/')
        {
            PathBuf::from(spec)
        } else {
            // Bare name — search PATH.
            search_path_for(spec).ok_or_else(|| {
                anyhow::anyhow!(
                    "include file '{spec}' not found in PATH (test \
                     declared it but the host doesn't have it; install or \
                     supply an absolute path)"
                )
            })?
        };
        if !path.exists() {
            bail!("include file does not exist on host: {}", path.display());
        }
        // Reject directories explicitly: export packs files only,
        // and a directory spec would silently fail later inside
        // `append_file`'s `std::fs::read` with a less-actionable
        // error message.
        if path.is_dir() {
            bail!(
                "include file '{}' is a directory — export packs regular files \
                 only. Recursive directory packaging is a v2 enhancement; for \
                 now, list each file individually in the test's \
                 `include_files` slot.",
                path.display()
            );
        }
        out.push(path);
    }
    Ok(out)
}

/// Search `PATH` for an executable named `name`. Returns the first
/// match. Mirrors the simplest case of the framework's PATH resolver
/// — sufficient for export's needs since tests typically declare
/// either bare standard tools (stress-ng, schbench) or paths to
/// build artifacts.
///
/// A "match" requires the candidate to be (a) a regular file and
/// (b) executable (any of the user/group/other execute bits set).
/// Without the executable check, a non-binary file with a colliding
/// name (e.g. a `stress-ng` documentation file in a PATH dir) would
/// be picked up first and silently fail at .run time when the guest
/// tries to exec it.
fn search_path_for(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if !candidate.is_file() {
            continue;
        }
        let executable = candidate
            .metadata()
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false);
        if executable {
            return Some(candidate);
        }
    }
    None
}

/// Tar+gzip the binaries into an in-memory blob.
///
/// Layout inside the archive:
///   - `ktstr` — the runner binary (the one calling
///     [`export_test`])
///   - `scheduler` — the scheduler binary (when present)
///   - `include/<basename>` — every include file, flattened by
///     basename. Collisions on basename are not allowed.
///
/// Permissions: every entry is chmod 0755. The `.run` extractor
/// preserves these, so the operator can invoke them directly under
/// `$DIR/ktstr` / `$DIR/scheduler` without re-chmod.
fn build_archive(ktstr: &Path, scheduler: Option<&Path>, includes: &[PathBuf]) -> Result<Vec<u8>> {
    let buf: Vec<u8> = Vec::new();
    let gz = GzEncoder::new(buf, Compression::default());
    let mut tar = tar::Builder::new(gz);

    append_file(&mut tar, ktstr, "ktstr")?;
    if let Some(s) = scheduler {
        append_file(&mut tar, s, "scheduler")?;
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for inc in includes {
        let name = inc
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("include file has no basename: {}", inc.display()))?
            .to_string_lossy()
            .into_owned();
        if !seen.insert(name.clone()) {
            bail!(
                "include-file basename collision: two specs both flatten to \
                 'include/{name}'. Rename one or use distinct paths."
            );
        }
        let archive_name = format!("include/{name}");
        append_file(&mut tar, inc, &archive_name)?;
    }

    let gz = tar.into_inner().context("finalise tar stream")?;
    let blob = gz.finish().context("finalise gzip stream")?;
    Ok(blob)
}

/// Append one host file at `host_path` into `tar` under `archive_name`.
/// Forces mode 0o755 so the extracted entry is executable on the
/// target host. Regenerates the tar header rather than reusing the
/// host's path metadata (which could leak environment-specific
/// information into the published artifact).
fn append_file<W: Write>(
    tar: &mut tar::Builder<W>,
    host_path: &Path,
    archive_name: &str,
) -> Result<()> {
    let bytes = std::fs::read(host_path)
        .with_context(|| format!("read {} for archive", host_path.display()))?;
    crate::tar_util::pack_tar_entry(
        tar,
        archive_name,
        0o755,
        bytes.len() as u64,
        bytes.as_slice(),
    )
    .with_context(|| format!("append {archive_name} to tar"))?;
    Ok(())
}

/// Generate the bash preamble. The output is a complete shell script
/// up to (but not including) the `__ARCHIVE__` marker; [`write_runfile`]
/// concatenates the preamble, the marker line, and the base64 archive.
///
/// The preamble is verbose by default: a banner identifying test +
/// scheduler + git provenance, every prereq/conflict check spelled
/// out with actionable error text, and a security-posture line
/// warning the operator to inspect the script (everything before
/// `__ARCHIVE__`) before running on a system they do not control.
/// `--quiet` suppresses the banner only — error paths still print
/// so a failing repro is never silent.
fn generate_preamble(
    entry: &KtstrTestEntry,
    has_scheduler: bool,
    config_additions: &[ConfigExportAddition],
) -> String {
    let topology = entry.topology;
    let need_llcs = topology.llcs;
    let need_cores = topology.cores_per_llc;
    let need_threads = topology.threads_per_core;
    let need_numa = topology.numa_nodes;

    // Compose scheduler args via the same host-side builder the
    // in-VM test path uses to assemble the scheduler argv
    // (`append_base_sched_args`, invoked from both `eval.rs` and
    // `probe.rs`): cgroup-parent auto-inject from
    // `entry.scheduler.cgroup_parent`, then the scheduler def's own
    // `sched_args`, then per-test `extra_sched_args`. Reusing the
    // helper keeps the bare-metal repro aligned with a normal test
    // run for these three sources — without it, the export would
    // silently drop `--cell-parent-cgroup` and any sched-def
    // baseline args, and the reproduced scheduler would land in the
    // wrong cgroup tree.
    //
    // Config-driven schedulers (declared `config_file` and/or
    // `config_content`) are handled via the `config_additions`
    // parameter: each addition contributes a shell-ready prefix
    // that lands BEFORE the joined base args. The ordering matches
    // the in-VM path at `eval.rs:1112-1125` which pushes
    // `--config <path>` (and any `config_file_def`-templated arg)
    // first and then appends `append_base_sched_args` last.
    // Keeping in-VM and export argv-order in lockstep means an
    // exported reproducer's scheduler-launch line is byte-similar
    // (modulo path expansion) to the live test's, so clap parsers
    // with order-sensitive semantics (e.g. trailing-args,
    // override-on-conflict) behave identically across both paths.
    //
    // Each prefix uses `"$DIR/include/<basename>"` so the path
    // resolves to the operator's extracted .run tree at script-run
    // time — the matching include-file is packed into the archive
    // at `include/<basename>` by [`compute_config_export_additions`]
    // and [`build_archive`].
    let mut sched_arg_tokens_raw: Vec<String> = Vec::new();
    crate::test_support::append_base_sched_args(entry, &mut sched_arg_tokens_raw);
    let base_joined: String = sched_arg_tokens_raw
        .iter()
        .map(|a| shell_quote(a))
        .collect::<Vec<_>>()
        .join(" ");
    let mut sched_args_joined = String::new();
    for addition in config_additions {
        if !sched_args_joined.is_empty() {
            sched_args_joined.push(' ');
        }
        sched_args_joined.push_str(&addition.args_shell_prefix);
    }
    if !base_joined.is_empty() {
        if !sched_args_joined.is_empty() {
            sched_args_joined.push(' ');
        }
        sched_args_joined.push_str(&base_joined);
    }

    // Defensive shell-quoting on every interpolated runtime value.
    // The names come from compile-time const slots that are
    // `[A-Za-z0-9_-]+` in practice today, but interpolating
    // unquoted lets a future producer regression land an
    // unescaped value with shell metacharacters in the preamble.
    // Quoting at the producer is cheap and matches the same defense
    // applied to extra_sched_args above.
    let test_name = shell_quote(entry.name);
    let scheduler_name = shell_quote(entry.scheduler.name);
    let git_hash = shell_quote(&git_provenance());

    let duration_secs = entry.duration.as_secs();
    let watchdog_secs = entry.watchdog_timeout.as_secs();

    let scheduler_launch = if has_scheduler {
        format!(
            r#"
# --- scheduler launch ---
echo "ktstr export: launching scheduler $KTSTR_SCHED_NAME"
"$DIR/scheduler" {sched_args_joined} &
SCHED_PID=$!

# Wait up to 10s for the scheduler to attach. The kernel's sysfs
# layout exposes attach state under two files; both are accepted
# so the wait loop works on every kernel that ships sched_ext:
#   - `/sys/kernel/sched_ext/root/ops` — non-empty when a scheduler
#     is currently attached. Present on every kernel revision that
#     has sched_ext, but the path moved structurally between early
#     6.x revisions and the upstream-stabilized layout. Treat the
#     file's absence as "no scheduler attached" rather than an
#     error; the secondary check below catches stabilized kernels.
#   - `/sys/kernel/sched_ext/state` (introduced upstream in 6.12)
#     reads `enabled` once a scheduler attaches, `disabled`
#     otherwise. Use as the primary signal where available; it has
#     a stable wire format across kernel versions.
# Bail if the scheduler exits before attaching, or if the timeout
# elapses while the scheduler is still alive but unattached.
ATTACHED=""
for _ in $(seq 1 100); do
    if ! kill -0 "$SCHED_PID" 2>/dev/null; then
        echo "error: scheduler $KTSTR_SCHED_NAME exited before attaching" >&2
        wait "$SCHED_PID" || true
        exit 1
    fi
    if [ -r /sys/kernel/sched_ext/state ]; then
        STATE=$(cat /sys/kernel/sched_ext/state 2>/dev/null || true)
        if [ "$STATE" = "enabled" ]; then
            ATTACHED="$STATE"
            break
        fi
    fi
    if [ -f /sys/kernel/sched_ext/root/ops ]; then
        OPS=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || true)
        if [ -n "$OPS" ]; then
            ATTACHED="$OPS"
            break
        fi
    fi
    sleep 0.1
done
if [ -z "$ATTACHED" ]; then
    echo "error: scheduler $KTSTR_SCHED_NAME launched but did not attach within 10s" >&2
    echo "       (process is still alive; check kernel log for BPF verifier or load errors)" >&2
    exit 1
fi
"#
        )
    } else {
        // Binary-kind / EEVDF payload — no scheduler.
        String::new()
    };

    format!(
        r#"#!/bin/bash
# Generated by `cargo ktstr export`. Do not edit; regenerate to update.
set -euo pipefail

# --- frozen test specification ---
KTSTR_TEST_NAME={test_name}
KTSTR_SCHED_NAME={scheduler_name}
KTSTR_GIT_HASH={git_hash}
NEED_LLCS={need_llcs}
NEED_CORES_PER_LLC={need_cores}
NEED_THREADS_PER_CORE={need_threads}
NEED_NUMA_NODES={need_numa}
TEST_DURATION_SECS={duration_secs}
TEST_WATCHDOG_SECS={watchdog_secs}

QUIET=0
DURATION_OVERRIDE=""
WATCHDOG_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quiet) QUIET=1; shift ;;
        --duration) DURATION_OVERRIDE="$2"; shift 2 ;;
        --watchdog-timeout) WATCHDOG_OVERRIDE="$2"; shift 2 ;;
        --cpus|--topology|--affinity)
            echo "error: --$1 is frozen for repro fidelity. Re-export to change." >&2
            exit 1 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--quiet] [--duration SECS] [--watchdog-timeout SECS]

Reproduces ktstr test '$KTSTR_TEST_NAME' on bare metal. The script
extracts an embedded gzip tarball containing the ktstr binary and
the scheduler binary, then dispatches the test directly without
booting a VM.

Frozen (cannot be overridden):
  scheduler         $KTSTR_SCHED_NAME
  topology          $NEED_NUMA_NODES NUMA / $NEED_LLCS LLCs / $NEED_CORES_PER_LLC cores/LLC / $NEED_THREADS_PER_CORE threads/core
  scheduler args    (compiled into the script)
  --cpus, --topology, --affinity reject any override

Overridable:
  --duration SECS         workload duration (default $TEST_DURATION_SECS)
  --watchdog-timeout SECS scheduler watchdog (default $TEST_WATCHDOG_SECS)
  --quiet                 suppress the banner (errors still print)

Requirements:
  Run as root. The script attaches a kernel BPF scheduler and sets
  up cgroup v2 subgroups; both need CAP_SYS_ADMIN.

  Host must satisfy the frozen topology (LLCs, cores per LLC,
  threads per core, NUMA nodes); the script's topology check bails
  with a specific "host has X, test needs Y" message if not.

  /sys/kernel/sched_ext must exist (kernel built with
  CONFIG_SCHED_CLASS_EXT) and no other sched_ext scheduler may be
  attached.

Exit codes:
  0   test passed
  1   prerequisite or topology check failed, scheduler attach
      failed, or test failed
EOF
            exit 0 ;;
        *) echo "error: unknown arg '$1' (use --help)" >&2; exit 1 ;;
    esac
done

if [ "$QUIET" != "1" ]; then
    cat <<EOF
ktstr export: test=$KTSTR_TEST_NAME scheduler=$KTSTR_SCHED_NAME git=$KTSTR_GIT_HASH
Generated by cargo ktstr export. This script attaches a kernel BPF scheduler
and runs as root. Inspect this script (everything before __ARCHIVE__) before
running on a system you do not control.
EOF
fi

# --- root check ---
if [ "$(id -u)" != "0" ]; then
    echo "error: must run as root (need CAP_SYS_ADMIN for sched_ext + cgroup ops)" >&2
    exit 1
fi

# --- prereq checks ---
if [ ! -d /sys/kernel/sched_ext ]; then
    echo "error: kernel lacks sched_ext support (no /sys/kernel/sched_ext)" >&2
    exit 1
fi
if [ ! -d /sys/fs/cgroup ]; then
    echo "error: cgroup2 not mounted at /sys/fs/cgroup" >&2
    exit 1
fi
if ! grep -q '^cgroup2 /sys/fs/cgroup ' /proc/mounts; then
    echo "error: /sys/fs/cgroup is not a cgroup2 mount" >&2
    exit 1
fi

# --- sched_ext conflict check ---
# Mirror the attach-detection logic below: prefer
# /sys/kernel/sched_ext/state (stabilized in 6.12) when readable,
# fall back to /sys/kernel/sched_ext/root/ops otherwise. Either
# file reporting an attached scheduler aborts here so we don't
# silently displace someone else's running scheduler.
if [ -r /sys/kernel/sched_ext/state ]; then
    CURRENT_STATE=$(cat /sys/kernel/sched_ext/state 2>/dev/null || true)
    if [ "$CURRENT_STATE" = "enabled" ]; then
        CURRENT_OPS=""
        if [ -f /sys/kernel/sched_ext/root/ops ]; then
            CURRENT_OPS=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || true)
        fi
        echo "error: another sched_ext scheduler is already attached (state=enabled, ops=${{CURRENT_OPS:-unknown}})." >&2
        echo "       Detach it before running this repro (e.g. kill its supervisor)." >&2
        exit 1
    fi
elif [ -f /sys/kernel/sched_ext/root/ops ]; then
    CURRENT=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || true)
    if [ -n "$CURRENT" ]; then
        echo "error: another sched_ext scheduler '$CURRENT' is already attached." >&2
        echo "       Detach it before running this repro (e.g. kill its supervisor)." >&2
        exit 1
    fi
fi

# --- topology check ---
# LLC count: find the highest cache-index level under cpu0 (index3
# on most x86, but skylake-x has a dedicated L4 at index4 and ARM
# machines vary). Sum distinct shared_cpu_lists at that level.
HIGHEST_INDEX=$(ls -d /sys/devices/system/cpu/cpu0/cache/index* 2>/dev/null \
    | sort -V | tail -n1 || true)
if [ -n "$HIGHEST_INDEX" ]; then
    HIGHEST_LEVEL=$(basename "$HIGHEST_INDEX")
    HOST_LLCS=$(ls -d /sys/devices/system/cpu/cpu*/cache/$HIGHEST_LEVEL 2>/dev/null \
        | xargs -I{{}} cat {{}}/shared_cpu_list 2>/dev/null \
        | sort -u | wc -l)
else
    HOST_LLCS=0
fi
HOST_NUMA=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l || echo 0)
[ "$HOST_NUMA" -lt 1 ] && HOST_NUMA=1

# Cores per LLC: count distinct core_id values among cpus that share
# the highest-level cache with cpu0. threads per core: count cpus
# that share the same core_id within one LLC.
if [ -n "$HIGHEST_INDEX" ]; then
    CPU0_LLC=$(cat "$HIGHEST_INDEX/shared_cpu_list" 2>/dev/null || echo "")
else
    CPU0_LLC=""
fi
HOST_CORES_PER_LLC=0
HOST_THREADS_PER_CORE=0
if [ -n "$CPU0_LLC" ]; then
    # Expand cpu list ranges (e.g. "0-3,8-11") into individual ids.
    CPU_IDS=$(echo "$CPU0_LLC" | tr ',' '\n' | while read range; do
        if [ -z "$range" ]; then continue; fi
        if echo "$range" | grep -q '-'; then
            start=$(echo "$range" | cut -d- -f1)
            end=$(echo "$range" | cut -d- -f2)
            seq "$start" "$end"
        else
            echo "$range"
        fi
    done)
    HOST_CORES_PER_LLC=$(for id in $CPU_IDS; do
        cat "/sys/devices/system/cpu/cpu$id/topology/core_id" 2>/dev/null || echo
    done | sort -u | wc -l)
    CPU0_CORE=$(cat /sys/devices/system/cpu/cpu0/topology/core_id 2>/dev/null || echo)
    if [ -n "$CPU0_CORE" ]; then
        HOST_THREADS_PER_CORE=$(for id in $CPU_IDS; do
            this_core=$(cat "/sys/devices/system/cpu/cpu$id/topology/core_id" 2>/dev/null || echo)
            if [ "$this_core" = "$CPU0_CORE" ]; then echo "$id"; fi
        done | wc -l)
    fi
fi

if [ "$HOST_LLCS" = "0" ]; then
    echo "warning: could not detect host LLC count from sysfs (no cache/index* found for cpu0); the topology check below will fail" >&2
fi
if [ "$HOST_LLCS" -lt "$NEED_LLCS" ]; then
    echo "error: host has $HOST_LLCS LLCs, test needs $NEED_LLCS" >&2
    exit 1
fi
if [ "$HOST_NUMA" -lt "$NEED_NUMA_NODES" ]; then
    echo "error: host has $HOST_NUMA NUMA nodes, test needs $NEED_NUMA_NODES" >&2
    exit 1
fi
if [ "$HOST_CORES_PER_LLC" -gt 0 ] && [ "$HOST_CORES_PER_LLC" -lt "$NEED_CORES_PER_LLC" ]; then
    echo "error: host has $HOST_CORES_PER_LLC cores per LLC, test needs $NEED_CORES_PER_LLC" >&2
    exit 1
fi
if [ "$HOST_THREADS_PER_CORE" -gt 0 ] && [ "$HOST_THREADS_PER_CORE" -lt "$NEED_THREADS_PER_CORE" ]; then
    echo "error: host has $HOST_THREADS_PER_CORE threads per core, test needs $NEED_THREADS_PER_CORE" >&2
    exit 1
fi

# --- extract embedded archive ---
DIR=$(mktemp -d -t ktstr-export-XXXXXX)
chmod 700 "$DIR"
# The ktstr in-process dispatch creates its cgroup tree under
# /sys/fs/cgroup/ktstr — the export-relevant path goes through the
# ctor early-dispatch into `test_support::probe::build_dispatch_ctx_parts`
# which calls `test_support::args::resolve_cgroup_root` (args.rs:111
# fallback), and the in-VM init follows the same convention.
# Capture the path here so the trap teardown can clean any subgroups
# the dispatch created. The rmdir must walk depth-first because
# cgroup v2 forbids rmdir on a subtree that still contains child
# groups.
#
# WARNING: this cleanup removes ALL subgroups under
# /sys/fs/cgroup/ktstr, including those created by concurrent
# ktstr processes. Do not run multiple ktstr workloads on the same
# host simultaneously.
KTSTR_CGROUP_PARENT="/sys/fs/cgroup/ktstr"
SCHED_PID=""
cleanup() {{
    if [ -n "$SCHED_PID" ]; then
        kill "$SCHED_PID" 2>/dev/null || true
        wait "$SCHED_PID" 2>/dev/null || true
    fi
    rm -rf "$DIR"
    # Cgroup teardown: depth-first rmdir over every subgroup the
    # test created. cgroup v2's interface files (cgroup.procs,
    # cgroup.controllers, ...) are auto-removed when their parent
    # directory rmdirs, so a recursive `rm -rf` is wrong (would
    # ENOTEMPTY on every interior node). `find -depth` visits
    # leaves before parents; rmdir succeeds at each step because
    # children are gone. Errors swallowed via `2>/dev/null` so a
    # cleanup race with another tool doesn't bleed into the test
    # exit status.
    if [ -d "$KTSTR_CGROUP_PARENT" ]; then
        find "$KTSTR_CGROUP_PARENT" -mindepth 1 -depth -type d \
            -exec rmdir {{}} + 2>/dev/null || true
        rmdir "$KTSTR_CGROUP_PARENT" 2>/dev/null || true
    fi
}}
trap cleanup EXIT

# Decode embedded base64 archive (everything after __ARCHIVE__).
sed -n '/^__ARCHIVE__$/,$p' "$0" | tail -n+2 | base64 -d | tar xz -C "$DIR"

if [ ! -x "$DIR/ktstr" ]; then
    echo "error: extracted ktstr binary missing or not executable" >&2
    exit 1
fi
{scheduler_launch}
# --- run the test ---
# `--ktstr-test-fn $KTSTR_TEST_NAME` is intercepted by the ktstr
# binary's `#[ctor::ctor] ktstr_test_early_dispatch` (in
# `src/test_support/dispatch.rs`), which fires from `.init_array`
# BEFORE `main()` runs. The ctor reads the argv directly via
# `extract_test_fn_arg` and dispatches via
# `maybe_dispatch_vm_test_with_args` (in
# `src/test_support/probe.rs`) which calls `(entry.func)(&ctx)`
# directly, then exits the process on completion. The leading
# `"run"` token is cosmetic — it's never parsed because the ctor
# exits before clap sees it. This early-dispatch path is the
# contract for in-process repro and is load-bearing: a future
# refactor that moves dispatch out of the ctor must keep an
# equivalent argv-intercept path in place, or this preamble must
# change to match the new dispatch shape.
#
# IMPORTANT: do NOT use `exec` here. `exec` replaces the bash
# shell with the ktstr binary and DESTROYS the EXIT trap before
# it can fire — leaking the scheduler PID, the tempdir, and the
# cgroup tree. Run as a child and forward the exit code so the
# trap fires on bash exit.
RUN_ARGS=("run" "--ktstr-test-fn" "$KTSTR_TEST_NAME")
if [ -n "$DURATION_OVERRIDE" ]; then
    RUN_ARGS+=("--duration" "$DURATION_OVERRIDE")
fi
if [ -n "$WATCHDOG_OVERRIDE" ]; then
    RUN_ARGS+=("--watchdog-timeout" "$WATCHDOG_OVERRIDE")
fi
# Disable errexit just for the ktstr invocation so a non-zero
# exit from the test (the legitimate "test failed" outcome)
# propagates as our exit code instead of triggering set -e and
# bypassing the cleanup. The `|| true` would also keep going,
# but `set +e` makes the intent explicit.
set +e
"$DIR/ktstr" "${{RUN_ARGS[@]}}"
EXIT_CODE=$?
set -e
exit $EXIT_CODE
"#
    )
}

/// Best-effort git provenance: the project HEAD short hex, or
/// `"unknown"` when not in a git checkout. Stamped into the
/// preamble's banner so an operator running an old `.run` can tell
/// what code was packaged.
///
/// Uses gix in-process rather than shelling out to `git rev-parse`.
/// Same shape as [`crate::fetch::inspect_local_source_state`]: walk
/// up from the current directory with `gix::discover`, read the head
/// id, format and truncate. No process fork, no PATH dependency —
/// the export pipeline never depends on a `git` binary being
/// installed on the host running `cargo ktstr export`.
fn git_provenance() -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| gix::discover(&cwd).ok())
        .and_then(|repo| {
            // `head_id()` returns an Id<'_> borrowing `repo`, so format
            // and truncate to an owned String inside the same scope as
            // `repo` to satisfy the borrow checker. Mirrors the
            // pattern at fetch.rs:1016-1017.
            repo.head_id()
                .ok()
                .map(|id| format!("{id}").chars().take(7).collect::<String>())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Single-quote a shell argument. Embedded single quotes are
/// terminated, escaped via `'\''`, and re-opened. Sufficient for
/// passing arbitrary `extra_sched_args` strings through the
/// preamble's word-split positional context.
///
/// Empty input is quoted to `''` rather than left as the empty
/// string. An unquoted empty arg word-splits to nothing in bash,
/// silently dropping the slot — quoting preserves the empty
/// positional argument so the scheduler's argv index is preserved.
fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if !s.contains('\'')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-+=/:".contains(c))
    {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Write the final `.run` file: preamble bytes, the `__ARCHIVE__`
/// marker line, then the base64-encoded archive split into 76-column
/// lines (POSIX-friendly width). Sets executable mode 0o755.
fn write_runfile(path: &Path, preamble: &str, archive: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o755)
        .open(path)
        .with_context(|| format!("open {} for write", path.display()))?;

    f.write_all(preamble.as_bytes()).context("write preamble")?;
    f.write_all(b"__ARCHIVE__\n")
        .context("write archive marker")?;

    let encoded = BASE64.encode(archive);
    // Split into 76-char lines so the file works through legacy
    // text-only transports (email MIME, some line editors).
    for chunk in encoded.as_bytes().chunks(76) {
        f.write_all(chunk).context("write base64 chunk")?;
        f.write_all(b"\n").context("write newline")?;
    }
    f.sync_all().context("fsync runfile")?;
    drop(f);
    let mut perms = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)
        .with_context(|| format!("chmod 755 {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
#[path = "export_tests.rs"]
mod tests;
