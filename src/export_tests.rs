//! Unit tests for [`super`] (the `export` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

use super::*;
use crate::test_support::TopologyConstraints;

#[test]
fn shell_quote_preserves_safe_strings() {
    assert_eq!(shell_quote("simple"), "simple");
    assert_eq!(shell_quote("--foo=bar"), "--foo=bar");
    assert_eq!(shell_quote("/usr/bin/foo"), "/usr/bin/foo");
    assert_eq!(shell_quote("a.b-c_d"), "a.b-c_d");
}

#[test]
fn shell_quote_wraps_special_chars() {
    // Spaces, quotes, semicolons all force quoting.
    assert_eq!(shell_quote("with space"), "'with space'");
    assert_eq!(shell_quote("a;b"), "'a;b'");
    assert_eq!(shell_quote("$VAR"), "'$VAR'");
}

#[test]
fn shell_quote_escapes_embedded_single_quotes() {
    // POSIX-standard quote-escape pattern: terminate, escape,
    // re-open.
    assert_eq!(shell_quote("don't"), "'don'\\''t'");
}

#[test]
fn shell_quote_empty_string_yields_quoted_empty() {
    // Unquoted empty arg word-splits to nothing in bash, dropping
    // the positional slot. Quoting preserves it as an empty
    // argument.
    assert_eq!(shell_quote(""), "''");
}

/// Tab (`\t`) is whitespace under bash IFS — unquoted, the
/// arg would split on the tab. Quoting preserves the literal
/// tab byte inside the single-quoted form (single quotes are
/// fully literal in POSIX shell, no escape interpretation).
#[test]
fn shell_quote_tab() {
    assert_eq!(shell_quote("a\tb"), "'a\tb'");
}

/// Newline must be quoted: an unquoted newline ends the
/// command and starts a new one. Inside single quotes the
/// literal `\n` byte is preserved verbatim.
#[test]
fn shell_quote_newline() {
    assert_eq!(shell_quote("a\nb"), "'a\nb'");
}

/// Backslash is a non-special byte INSIDE POSIX single quotes
/// (only `'` ends the quoted region). The shell_quote
/// implementation must preserve the backslash verbatim, NOT
/// double it (a common bug from "escape everything" thinking).
#[test]
fn shell_quote_backslash() {
    assert_eq!(shell_quote(r"a\b"), r"'a\b'");
    // Trailing backslash is also fine — the closing quote
    // terminates the literal cleanly.
    assert_eq!(shell_quote(r"trail\"), r"'trail\'");
}

/// Unicode is preserved byte-for-byte: emoji + CJK both
/// roundtrip through the single-quoted form. POSIX single
/// quotes are byte-literal so multi-byte UTF-8 sequences are
/// safe; we still wrap because the `is_ascii_alphanumeric`
/// gate rejects non-ASCII and falls into the quoted path.
#[test]
fn shell_quote_unicode_emoji_and_cjk() {
    assert_eq!(shell_quote("test ✅"), "'test ✅'");
    assert_eq!(shell_quote("日本語"), "'日本語'");
    assert_eq!(shell_quote("héllo"), "'héllo'");
}

/// NUL byte: bash refuses NUL bytes in argv, but `shell_quote`
/// itself must not panic — emit a quoted form and let the
/// downstream shell reject it. `\0` inside single quotes is
/// byte-literal, same as any other non-quote byte.
#[test]
fn shell_quote_null_byte() {
    let s = "a\0b";
    let q = shell_quote(s);
    assert_eq!(q, "'a\0b'");
}

/// Mixed quote types (`'` and `"`): single quotes are escaped
/// via the `'\''` close-escape-reopen pattern; double quotes
/// are byte-literal inside single quotes and pass through
/// untouched.
#[test]
fn shell_quote_mixed_quote_types() {
    // Single inside, double outside: single becomes the
    // escape sequence, double is literal.
    assert_eq!(shell_quote(r#"he said "don't""#), r#"'he said "don'\''t"'"#);
}

/// String that already looks like a single-quoted literal
/// must be re-quoted: the existing surrounding `'` are bytes,
/// not metacharacters, so the wrapping pattern still applies
/// and the embedded `'` get the close-escape-reopen treatment.
#[test]
fn shell_quote_already_single_quoted() {
    assert_eq!(shell_quote("'pre-quoted'"), r"''\''pre-quoted'\'''");
}

/// A bare single quote: edge of the escape pattern. The output
/// must still be a valid single-quoted shell word.
#[test]
fn shell_quote_only_single_quote() {
    let q = shell_quote("'");
    // Must be syntactically valid POSIX: `'\''` is the canonical
    // pattern. Wrapped in outer quotes -> `''\'''`.
    assert_eq!(q, r"''\'''");
}

/// Carriage return is non-printable and would terminate a
/// shell line on systems that treat `\r` as a newline (CRLF
/// terminals). Single-quoted form preserves the byte verbatim.
#[test]
fn shell_quote_carriage_return() {
    assert_eq!(shell_quote("a\rb"), "'a\rb'");
    assert_eq!(shell_quote("a\r\nb"), "'a\r\nb'");
}

/// Consecutive embedded single quotes: each one independently
/// triggers the close-escape-reopen pattern. Tests that the
/// per-char loop doesn't collapse runs of `'` into a single
/// escape.
#[test]
fn shell_quote_consecutive_single_quotes() {
    assert_eq!(shell_quote("a''b"), r"'a'\'''\''b'");
    assert_eq!(shell_quote("'''"), r"''\'''\'''\'''");
}

/// Combination: tab byte AND embedded single quote in the same
/// input. The tab passes through verbatim (single quotes are
/// byte-literal) and the apostrophe takes the escape path.
#[test]
fn shell_quote_tab_with_single_quote() {
    assert_eq!(shell_quote("a\t'b"), "'a\t'\\''b'");
}

/// Other low-byte control characters (vertical tab, form feed,
/// bell, ESC). All non-printable, all byte-literal inside POSIX
/// single quotes.
#[test]
fn shell_quote_low_control_bytes() {
    assert_eq!(shell_quote("\x07"), "'\x07'"); // bell
    assert_eq!(shell_quote("\x08"), "'\x08'"); // backspace
    assert_eq!(shell_quote("\x0b"), "'\x0b'"); // vertical tab
    assert_eq!(shell_quote("\x0c"), "'\x0c'"); // form feed
    assert_eq!(shell_quote("\x1b[31mred\x1b[0m"), "'\x1b[31mred\x1b[0m'");
}

/// Safe-set chars (`+`, `=`, `:`, `/`, `.`, `_`, `-`) at
/// boundaries: this pins which characters bypass the quote
/// wrap. The safe set is a positive list; any change to it
/// flips this test.
#[test]
fn shell_quote_safe_set_unquoted() {
    // Each of these should pass through verbatim.
    for raw in [
        "+",
        "=",
        ":",
        "/",
        ".",
        "_",
        "-",
        "abc+def",
        "key=value",
        "ns:resource",
        "/usr/local/bin",
        "v1.0.0",
        "file_name-1.txt",
    ] {
        assert_eq!(
            shell_quote(raw),
            raw,
            "safe-set input must remain unquoted: {raw:?}"
        );
    }
}

/// Long string with no special chars stays unquoted; long
/// string with a single special char anywhere triggers the
/// full-string wrap. Pins behavior for size-conscious callers.
#[test]
fn shell_quote_long_strings() {
    let safe = "a".repeat(1024);
    assert_eq!(
        shell_quote(&safe),
        safe,
        "long safe-set string passes through"
    );

    let with_space = format!("{}{}", "a".repeat(512), " end");
    let q = shell_quote(&with_space);
    assert!(q.starts_with('\'') && q.ends_with('\''));
    assert_eq!(&q[1..q.len() - 1], &with_space);
}

/// Shell metacharacters that would otherwise be interpreted
/// by the parser must all roundtrip through the quoted form
/// without forking a subshell, expanding a glob, or
/// dereferencing a variable.
#[test]
fn shell_quote_shell_metacharacters() {
    // Each of these would, unquoted, do something dangerous.
    for raw in [
        "a&b", "a|b", "a`b`c", "a$b", "a*b", "a?b", "a[b]c", "a{b}c", "a(b)c", "a~b", "a#b", "a!b",
    ] {
        let q = shell_quote(raw);
        assert!(
            q.starts_with('\'') && q.ends_with('\''),
            "metachar input must be wrapped: input={raw:?} output={q:?}"
        );
        // The byte content (between the wrapping quotes) must
        // equal the original — single quotes are byte-literal,
        // no escape interpretation, no `'` to escape in these
        // inputs.
        let inner = &q[1..q.len() - 1];
        assert_eq!(
            inner, raw,
            "metachar input must be byte-preserved inside the wrap"
        );
    }
}

#[test]
fn search_path_for_finds_existing_executable() {
    // /bin/sh is executable on every supported host; PATH lookup
    // must resolve it.
    let found = search_path_for("sh");
    assert!(found.is_some(), "PATH search for `sh` returned None");
    let path = found.unwrap();
    assert!(path.is_file(), "resolved path is not a file: {path:?}");
    let mode = path.metadata().unwrap().permissions().mode();
    assert!(
        mode & 0o111 != 0,
        "resolved path is not executable: {path:?} mode={mode:o}"
    );
}

#[test]
fn search_path_for_returns_none_on_missing() {
    // A name that cannot exist as a binary.
    let found = search_path_for("definitely-not-a-real-binary-xyzzy-987");
    assert!(found.is_none());
}

#[test]
fn search_path_for_skips_non_executable_files() {
    // Plant a non-executable file in a temp dir, prepend that
    // dir to PATH, search for the name. The lookup must skip
    // it (mode lacks any execute bit) rather than return the
    // path.
    let tmp = tempfile::TempDir::new().expect("create temp dir");
    let dummy = tmp.path().join("dummy_non_exec");
    std::fs::write(&dummy, b"#!/bin/sh\necho hi\n").expect("write dummy");
    let mut perms = std::fs::metadata(&dummy).unwrap().permissions();
    perms.set_mode(0o644);
    std::fs::set_permissions(&dummy, perms).expect("set non-exec perms");

    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let new_path = {
        let mut paths = vec![tmp.path().to_path_buf()];
        paths.extend(std::env::split_paths(&original_path));
        std::env::join_paths(paths).expect("join paths")
    };
    // SAFETY: this test is a unit test that does not spawn
    // threads concurrently and the env var is restored before
    // exit. Other concurrent tests under nextest run in
    // separate processes.
    unsafe { std::env::set_var("PATH", &new_path) };
    let found = search_path_for("dummy_non_exec");
    unsafe { std::env::set_var("PATH", &original_path) };

    assert!(
        found.is_none(),
        "non-executable file must NOT match PATH lookup, got: {found:?}",
    );
}

/// Read every entry from a gzip-compressed tar blob. Used by the
/// archive-shape tests below — keeps the read path consolidated
/// so a future tar-or-gzip change forces only one update.
fn read_archive_entries(blob: &[u8]) -> Vec<(String, u32, Vec<u8>)> {
    use flate2::read::GzDecoder;
    use std::io::Read as _;
    let gz = GzDecoder::new(blob);
    let mut archive = tar::Archive::new(gz);
    let mut out = Vec::new();
    for entry in archive.entries().expect("read tar entries") {
        let mut e = entry.expect("entry");
        let name = e.path().expect("entry path").to_string_lossy().into_owned();
        let mode = e.header().mode().expect("entry mode");
        let mut data = Vec::new();
        e.read_to_end(&mut data).expect("read entry body");
        out.push((name, mode, data));
    }
    out
}

/// `build_archive` with no scheduler and no includes packs ONLY
/// the ktstr binary under the name `ktstr`. Pins the
/// scheduler-less code path: the EEVDF/binary-payload export
/// must NOT silently embed extra entries when the test has no
/// scheduler binary to ship.
#[test]
fn build_archive_no_scheduler_packs_only_ktstr() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let ktstr_path = tmp.path().join("fake-ktstr");
    std::fs::write(&ktstr_path, b"#!/bin/sh\necho ktstr-stub\n").expect("write fake ktstr");

    let blob = build_archive(&ktstr_path, None, &[]).expect("build_archive");
    let entries = read_archive_entries(&blob);

    assert_eq!(entries.len(), 1, "expected 1 entry, got: {entries:?}");
    let (name, mode, data) = &entries[0];
    assert_eq!(name, "ktstr", "entry must be named 'ktstr'");
    assert_eq!(*mode, 0o755, "entry must be mode 0o755 (executable)");
    assert_eq!(
        data.as_slice(),
        b"#!/bin/sh\necho ktstr-stub\n",
        "entry payload must roundtrip the input file bytes",
    );
}

/// `build_archive` with a scheduler and N include files emits
/// the canonical layout: `ktstr`, `scheduler`, then
/// `include/<basename>` per include. Mode 0o755 on every entry
/// so the .run extractor preserves executable bits. Pins the
/// archive-layout contract the preamble's
/// `sed | base64 -d | tar xz` extraction depends on.
#[test]
fn build_archive_packs_ktstr_scheduler_and_includes() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let ktstr_path = tmp.path().join("fake-ktstr");
    let sched_path = tmp.path().join("fake-sched");
    let inc_a = tmp.path().join("inc_a.txt");
    let inc_b = tmp.path().join("inc_b.txt");
    std::fs::write(&ktstr_path, b"K").expect("write ktstr");
    std::fs::write(&sched_path, b"S").expect("write scheduler");
    std::fs::write(&inc_a, b"A").expect("write inc_a");
    std::fs::write(&inc_b, b"B").expect("write inc_b");

    let includes = vec![inc_a.clone(), inc_b.clone()];
    let blob = build_archive(&ktstr_path, Some(&sched_path), &includes).expect("build_archive");
    let entries = read_archive_entries(&blob);

    let names: Vec<&str> = entries.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "ktstr",
            "scheduler",
            "include/inc_a.txt",
            "include/inc_b.txt"
        ],
        "entry names and order must match the documented layout",
    );
    for (name, mode, _) in &entries {
        assert_eq!(*mode, 0o755, "entry {name} must be mode 0o755");
    }
}

/// `build_archive` rejects two include specs whose basenames
/// collide. The error message must name the colliding basename
/// so the operator can find and rename one. The flat
/// `include/<basename>` layout is documented as the contract;
/// silently dropping a collision would corrupt the archive.
#[test]
fn build_archive_rejects_basename_collision() {
    let tmp_a = tempfile::TempDir::new().expect("temp dir a");
    let tmp_b = tempfile::TempDir::new().expect("temp dir b");
    let inc_1 = tmp_a.path().join("dup.txt");
    let inc_2 = tmp_b.path().join("dup.txt");
    std::fs::write(&inc_1, b"first").expect("write inc_1");
    std::fs::write(&inc_2, b"second").expect("write inc_2");

    let ktstr_path = tmp_a.path().join("fake-ktstr");
    std::fs::write(&ktstr_path, b"K").expect("write ktstr");

    let err = build_archive(&ktstr_path, None, &[inc_1.clone(), inc_2.clone()])
        .expect_err("colliding basenames must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("dup.txt"),
        "error must name the colliding basename: '{msg}'",
    );
    assert!(
        msg.contains("collision") || msg.contains("collide"),
        "error must describe the failure as a collision: '{msg}'",
    );
}

/// `generate_preamble` emits a syntactically valid bash script
/// that `bash -n` parses without error. This is the load-bearing
/// invariant of the export pipeline — a malformed preamble means
/// every `.run` file the operator generates is a dud, regardless
/// of whether the embedded archive is correct.
///
/// Skipped (not failed) when `bash` is unavailable on the host
/// running the unit tests; bash is universal on linux dev hosts
/// but a CI image without it shouldn't make the rest of the
/// suite red.
#[test]
fn generate_preamble_parses_under_bash_n() {
    if which_bash().is_none() {
        crate::report::test_skip("no bash on PATH");
        return;
    }

    let entry = KtstrTestEntry {
        name: "test_preamble_smoke",
        extra_sched_args: &["--foo", "bar baz"],
        ..KtstrTestEntry::DEFAULT
    };

    for has_scheduler in [true, false] {
        let preamble = generate_preamble(&entry, has_scheduler, &[]);
        assert_bash_n_accepts(&preamble, has_scheduler);
    }
}

/// `generate_preamble` interpolates the test name, scheduler
/// name, topology, duration, and watchdog into the script in
/// shape that an operator can grep. Pins the variable names
/// (KTSTR_TEST_NAME, KTSTR_SCHED_NAME, NEED_LLCS,
/// TEST_DURATION_SECS, TEST_WATCHDOG_SECS) and their values
/// against the entry — a future format change must update this
/// test in lockstep with the preamble.
#[test]
fn generate_preamble_interpolates_entry_fields() {
    let entry = KtstrTestEntry {
        name: "interp_smoke",
        duration: std::time::Duration::from_secs(7),
        watchdog_timeout: std::time::Duration::from_secs(13),
        topology: crate::vmm::topology::Topology {
            llcs: 3,
            cores_per_llc: 5,
            threads_per_core: 2,
            numa_nodes: 4,
            nodes: None,
            distances: None,
        },
        ..KtstrTestEntry::DEFAULT
    };
    let preamble = generate_preamble(&entry, true, &[]);

    assert!(
        preamble.contains("KTSTR_TEST_NAME=interp_smoke"),
        "preamble must set KTSTR_TEST_NAME from entry.name",
    );
    assert!(
        preamble.contains("NEED_LLCS=3"),
        "preamble must set NEED_LLCS from entry.topology.llcs",
    );
    assert!(
        preamble.contains("NEED_CORES_PER_LLC=5"),
        "preamble must set NEED_CORES_PER_LLC",
    );
    assert!(
        preamble.contains("NEED_THREADS_PER_CORE=2"),
        "preamble must set NEED_THREADS_PER_CORE",
    );
    assert!(
        preamble.contains("NEED_NUMA_NODES=4"),
        "preamble must set NEED_NUMA_NODES",
    );
    assert!(
        preamble.contains("TEST_DURATION_SECS=7"),
        "preamble must set TEST_DURATION_SECS from entry.duration",
    );
    assert!(
        preamble.contains("TEST_WATCHDOG_SECS=13"),
        "preamble must set TEST_WATCHDOG_SECS from entry.watchdog_timeout",
    );
}

/// `generate_preamble` does NOT auto-inject `--cell-parent-cgroup`
/// from the scheduler's `cgroup_parent` slot. The two concerns are
/// decoupled: `cgroup_parent` controls the framework's cgroup root
/// (where test cgroups live), while `--cell-parent-cgroup` is a
/// scheduler-specific argv flag that cell-aware schedulers
/// (scx_mitosis et al.) interpret by enabling
/// userspace_managed_cell_mode and starting an inotify-driven
/// CellManager — a mode that interferes with the host-side
/// periodic-capture pipeline. Schedulers that genuinely need
/// `--cell-parent-cgroup` must opt in by including it in the
/// declaration's `sched_args` or the per-test `extra_sched_args`.
#[test]
fn generate_preamble_does_not_auto_inject_cell_parent_cgroup_from_cgroup_parent() {
    use crate::test_support::{CgroupPath, Scheduler, SchedulerSpec};
    static SCHED_WITH_PARENT: Scheduler = Scheduler {
        name: "sched_with_parent",
        binary: SchedulerSpec::Discover("sched_with_parent_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: Some(CgroupPath::new("/ktstr_export_test")),
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: None,
        kernels: &[],
    };
    let entry = KtstrTestEntry {
        name: "no_auto_inject_smoke",
        scheduler: &SCHED_WITH_PARENT,
        ..KtstrTestEntry::DEFAULT
    };
    let preamble = generate_preamble(&entry, true, &[]);
    assert!(
        !preamble.contains("--cell-parent-cgroup"),
        "preamble must NOT auto-inject --cell-parent-cgroup from \
             entry.scheduler.cgroup_parent (the two concerns are \
             decoupled); got: {preamble}"
    );
}

/// `generate_preamble` passes a user-supplied `--cell-parent-cgroup`
/// through `extra_sched_args` unchanged. The scheduler's
/// `cgroup_parent` slot has NO bearing on the scheduler's argv
/// (the two concerns are decoupled); only the explicit
/// user-supplied flag reaches the scheduler.
#[test]
fn generate_preamble_passes_user_supplied_cell_parent_cgroup_through() {
    use crate::test_support::{CgroupPath, Scheduler, SchedulerSpec};
    static SCHED_WITH_PARENT: Scheduler = Scheduler {
        name: "sched_with_parent_user_supplied",
        binary: SchedulerSpec::Discover("sched_with_parent_user_supplied_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: Some(CgroupPath::new("/auto_inject_path")),
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: None,
        kernels: &[],
    };
    let entry = KtstrTestEntry {
        name: "user_supplied_smoke",
        scheduler: &SCHED_WITH_PARENT,
        extra_sched_args: &["--cell-parent-cgroup", "/user_supplied_path"],
        ..KtstrTestEntry::DEFAULT
    };
    let preamble = generate_preamble(&entry, true, &[]);
    assert!(
        preamble.contains("/user_supplied_path"),
        "preamble must carry the user-supplied path; got: {preamble}"
    );
    assert!(
        !preamble.contains("/auto_inject_path"),
        "preamble must NOT inject the scheduler's cgroup_parent \
             into argv (decoupled); got: {preamble}"
    );
}

/// `compute_config_export_additions` returns an empty vec when
/// neither slot is set. Baseline against false-positive: a
/// scheduler without `config_file` / `config_file_def` and a
/// test entry without `config_content` must not contribute any
/// additions and the export pipeline must not touch any
/// `--config` arg.
#[test]
fn compute_config_export_additions_returns_empty_when_no_config() {
    let entry = KtstrTestEntry {
        name: "no_config_smoke",
        ..KtstrTestEntry::DEFAULT
    };
    let additions =
        compute_config_export_additions(&entry).expect("compute additions must not error");
    assert!(
        additions.is_empty(),
        "no config slots set must yield zero additions; got {} addition(s)",
        additions.len()
    );
}

/// `config_file_addition` mirrors the in-VM behavior at
/// [`crate::test_support::runtime::config_file_parts`] +
/// the surrounding push in `crate::test_support::eval`: a scheduler with
/// `config_file = Some(<host_path>)` causes a hardcoded
/// `--config` arg to be emitted. The export-side path uses
/// `"$DIR/include/<basename>"` so the .run extractor resolves
/// the path at script-run time on the target host. Pins the
/// in-VM-vs-export argv parity so a config-driven scheduler's
/// exported reproducer launches with the same flag as a normal
/// test run.
#[test]
fn config_file_addition_emits_hardcoded_config_arg_with_dir_expansion() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let host_cfg = tmp.path().join("scheduler-config.json");
    std::fs::write(&host_cfg, b"{\"layer_count\": 3}\n").expect("write fixture cfg");
    let host_cfg_str: &'static str =
        Box::leak(host_cfg.to_string_lossy().into_owned().into_boxed_str());
    let sched: &'static Scheduler = Box::leak(Box::new(Scheduler {
        name: "config_file_export_test",
        binary: SchedulerSpec::Discover("config_file_export_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: Some(host_cfg_str),
        config_file_def: None,
        kernels: &[],
    }));
    let entry = KtstrTestEntry {
        name: "config_file_export_smoke",
        scheduler: sched,
        ..KtstrTestEntry::DEFAULT
    };
    let addition = config_file_addition(&entry)
        .expect("config_file_addition must not error")
        .expect("config_file set must yield Some");
    assert_eq!(
        addition.host_path, host_cfg,
        "host_path must be the configured scheduler.config_file path verbatim"
    );
    assert_eq!(
        addition.args_shell_prefix, "--config \"$DIR/include/scheduler-config.json\"",
        "args_shell_prefix must use the hardcoded --config flag with \
             `$DIR/include/<basename>` path expansion and NO leading space \
             (the caller manages spacing); got: {:?}",
        addition.args_shell_prefix
    );
}

/// `config_content_addition` mirrors the in-VM behavior at
/// [`crate::test_support::runtime::config_content_parts`]: an
/// entry's `config_content = Some(<bytes>)` paired with a
/// scheduler's `config_file_def = Some((arg_template,
/// guest_path))` causes the content to be written to a temp
/// file on the host AND the scheduler arg template to be
/// substituted with the `"$DIR/include/<basename>"` runtime
/// path. The basename derives from the scheduler's declared
/// guest_path so a scheduler family with a stable naming
/// convention sees the same basename in the .run archive and
/// in the live /include-files mount.
#[test]
fn config_content_addition_writes_temp_file_and_substitutes_template() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    static SCHED_CONFIG_CONTENT: Scheduler = Scheduler {
        name: "config_content_export_test",
        binary: SchedulerSpec::Discover("config_content_export_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: Some(("--layered-config {file}", "/include-files/layers.json")),
        kernels: &[],
    };
    const CONTENT: &str = "{\"layers\": [\"foo\", \"bar\"]}\n";
    let entry = KtstrTestEntry {
        name: "config_content_export_smoke",
        scheduler: &SCHED_CONFIG_CONTENT,
        config_content: Some(CONTENT),
        ..KtstrTestEntry::DEFAULT
    };
    let addition = config_content_addition(&entry)
        .expect("config_content_addition must not error")
        .expect("config_content + config_file_def set must yield Some");
    let written =
        std::fs::read_to_string(&addition.host_path).expect("temp file must exist on disk");
    assert_eq!(
        written,
        CONTENT,
        "temp file at {} must contain the inline config_content bytes verbatim",
        addition.host_path.display()
    );
    assert_eq!(
        addition.args_shell_prefix, "--layered-config \"$DIR/include/layers.json\"",
        "args_shell_prefix must substitute `{{file}}` with `\"$DIR/include/<basename>\"` \
             where basename is derived from the scheduler's config_file_def guest_path, \
             with NO leading space (the caller manages spacing); got: {:?}",
        addition.args_shell_prefix
    );
}

/// Pin the load-bearing security property of the
/// [`scratch_dir`]-based atomic-rename pattern: the host_path
/// produced by `config_content_addition` lives INSIDE the
/// process-owned 0o700 scratch directory, never bare
/// `std::env::temp_dir()`. A regression that reverts to a bare
/// `std::env::temp_dir().join(...)` path would silently restore
/// the symlink-attack surface — every other functional
/// assertion in this file's tests would still pass. This test
/// is the dedicated guard against that revert path.
#[test]
fn config_content_addition_writes_inside_process_scratch_dir() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    static SCHED_SCRATCH_DIR_PIN: Scheduler = Scheduler {
        name: "scratch_dir_pin",
        binary: SchedulerSpec::Discover("scratch_dir_pin_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: Some(("--config={file}", "/include-files/dirpin.json")),
        kernels: &[],
    };
    let entry = KtstrTestEntry {
        name: "scratch_dir_pin_smoke",
        scheduler: &SCHED_SCRATCH_DIR_PIN,
        config_content: Some("{\"pin\": true}\n"),
        ..KtstrTestEntry::DEFAULT
    };
    let addition = config_content_addition(&entry)
        .expect("config_content_addition must not error")
        .expect("config_content + config_file_def set must yield Some");
    let dir = scratch_dir();
    assert!(
        addition.host_path.starts_with(dir),
        "host_path {} must live inside the process scratch_dir {} — a regression \
             to bare std::env::temp_dir() would silently restore the symlink-attack \
             surface this test guards against",
        addition.host_path.display(),
        dir.display(),
    );
}

/// Pin the content-addressed naming idempotence: calling
/// `config_content_addition` twice with the same
/// `config_content` produces the same canonical host_path.
/// Verifies the atomic-rename collision-handling described in
/// the production-code comment plus the content-hash filename
/// template `ktstr-export-config-{hash:016x}-{basename}`.
#[test]
fn config_content_addition_same_content_same_canonical_path() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    static SCHED_IDEMPOTENT: Scheduler = Scheduler {
        name: "idempotent_pin",
        binary: SchedulerSpec::Discover("idempotent_pin_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: Some(("--config={file}", "/include-files/idem.json")),
        kernels: &[],
    };
    const CONTENT: &str = "{\"idem\": 42}\n";
    let entry = KtstrTestEntry {
        name: "idempotent_pin_smoke",
        scheduler: &SCHED_IDEMPOTENT,
        config_content: Some(CONTENT),
        ..KtstrTestEntry::DEFAULT
    };
    let a = config_content_addition(&entry).unwrap().unwrap();
    let b = config_content_addition(&entry).unwrap().unwrap();
    assert_eq!(
        a.host_path,
        b.host_path,
        "same content must produce same canonical host_path; got {} vs {}",
        a.host_path.display(),
        b.host_path.display(),
    );
    let basename = a
        .host_path
        .file_name()
        .and_then(|s| s.to_str())
        .expect("host_path must have a UTF-8 basename");
    assert!(
        basename.starts_with("ktstr-export-config-") && basename.ends_with("idem.json"),
        "basename must match the `ktstr-export-config-{{hash:016x}}-<basename>` \
             template (with the scheduler's config_file_def basename suffix); got {basename}",
    );
}

/// Pin the `reject_shell_metacharacters_in_basename` gate on the
/// inline-content path. Parallel to the existing
/// `config_file_addition_rejects_basename_with_shell_metacharacters`
/// test for the config_file path — the metacharacter rejection
/// must fire BEFORE any scratch-file write touches the disk.
#[test]
fn config_content_addition_rejects_basename_with_shell_metacharacters() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    static SCHED_METACHAR: Scheduler = Scheduler {
        name: "metachar_pin",
        binary: SchedulerSpec::Discover("metachar_pin_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        // Basename "$evil.json" derives from this guest_path —
        // `$` is a shell-metacharacter that would expand
        // unpredictably in the args_shell_prefix's double-quoted
        // context. The metacharacter check at the top of
        // config_content_addition must reject it before any
        // scratch write fires.
        config_file_def: Some(("--config={file}", "/include-files/$evil.json")),
        kernels: &[],
    };
    let entry = KtstrTestEntry {
        name: "metachar_pin_smoke",
        scheduler: &SCHED_METACHAR,
        config_content: Some("ignored\n"),
        ..KtstrTestEntry::DEFAULT
    };
    let err = config_content_addition(&entry)
        .expect_err("basename with shell metacharacter must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("shell-metacharacter") && msg.contains('$'),
        "rejection diagnostic must name the failure mode and the offending \
             character; got {msg}",
    );
}

/// End-to-end: `generate_preamble` consumes the config
/// additions and prepends every `args_shell_prefix` ahead of
/// the base sched-args. The .run script must contain the
/// config flag so the scheduler launches with the expected
/// argv when the operator runs the exported reproducer. Pins
/// the wiring between [`compute_config_export_additions`] and
/// [`generate_preamble`].
#[test]
fn generate_preamble_prepends_config_addition_prefix() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    static SCHED: Scheduler = Scheduler {
        name: "preamble_config_smoke",
        binary: SchedulerSpec::Discover("preamble_config_smoke_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: None,
        kernels: &[],
    };
    let entry = KtstrTestEntry {
        name: "preamble_config_smoke",
        scheduler: &SCHED,
        ..KtstrTestEntry::DEFAULT
    };
    let additions = vec![ConfigExportAddition {
        host_path: PathBuf::from("/tmp/ktstr-export-config-test.json"),
        args_shell_prefix: "--config \"$DIR/include/test.json\"".to_string(),
    }];
    let preamble = generate_preamble(&entry, true, &additions);
    assert!(
        preamble.contains("--config \"$DIR/include/test.json\""),
        "preamble must contain the verbatim args_shell_prefix from the \
             config addition; got: {preamble}"
    );
}

/// Order parity with the in-VM path in
/// `crate::test_support::eval::run_ktstr_test_inner_impl`:
/// `--config <path>` (and any `config_file_def`-templated arg)
/// is pushed FIRST, then `append_base_sched_args` (which
/// includes `--cell-parent-cgroup` auto-inject) is appended
/// LAST. The export must match this ordering so clap parsers
/// with order-sensitive semantics behave identically across
/// both paths. Scheduler with `cgroup_parent =
/// Some("/order_pin_parent")` exercises the auto-inject so
/// the test has a concrete base-args anchor to compare
/// against. Regression here means a future change reverted to
/// suffix-position config additions and broke argv ordering
/// parity.
#[test]
fn generate_preamble_emits_config_addition_before_base_sched_args() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    static SCHED: Scheduler = Scheduler {
        name: "order_pin_test",
        binary: SchedulerSpec::Discover("order_pin_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        // `cgroup_parent` no longer auto-injects --cell-parent-cgroup
        // (decoupled); pin the scheduler-side flag in
        // `sched_args` explicitly so this ordering test still
        // exercises the config-vs-base-sched-args anchor.
        cgroup_parent: None,
        sched_args: &["--cell-parent-cgroup", "/order_pin_parent"],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
        config_file_def: None,
        kernels: &[],
    };
    let entry = KtstrTestEntry {
        name: "order_pin_test",
        scheduler: &SCHED,
        ..KtstrTestEntry::DEFAULT
    };
    let additions = vec![ConfigExportAddition {
        host_path: PathBuf::from("/tmp/order-pin-test.json"),
        args_shell_prefix: "--config \"$DIR/include/order-pin-test.json\"".to_string(),
    }];
    let preamble = generate_preamble(&entry, true, &additions);
    let config_pos = preamble
        .find("--config \"$DIR/include/order-pin-test.json\"")
        .expect("preamble must contain the --config arg from the addition");
    let cgroup_pos = preamble
        .find("--cell-parent-cgroup")
        .expect("preamble must contain --cell-parent-cgroup from the explicit sched_args entry");
    assert!(
        config_pos < cgroup_pos,
        "argv ordering parity with run_ktstr_test_inner_impl: `--config` must \
             appear BEFORE `--cell-parent-cgroup`. \
             config_pos={config_pos}, cgroup_pos={cgroup_pos}, preamble:\n{preamble}"
    );
}

/// Pin the intentional dual-fire when BOTH
/// `scheduler.config_file` AND `entry.config_content` (paired
/// with `scheduler.config_file_def`) are set. The two slots
/// are orthogonal — a scheduler that declares both legitimately
/// gets BOTH `--config` and the templated arg in its argv,
/// matching the in-VM behavior in
/// `crate::test_support::eval::run_ktstr_test_inner_impl` which
/// pushes each slot independently. A regression that introduced
/// a mutual-exclusion check (or coalesced the two slots) would
/// silently drop one source.
#[test]
fn compute_config_export_additions_dual_fire_when_file_and_content_set() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let host_cfg = tmp.path().join("static-cfg.json");
    std::fs::write(&host_cfg, b"{\"static\": true}\n").expect("write fixture cfg");
    let host_cfg_str: &'static str =
        Box::leak(host_cfg.to_string_lossy().into_owned().into_boxed_str());
    let sched: &'static Scheduler = Box::leak(Box::new(Scheduler {
        name: "dual_fire_test",
        binary: SchedulerSpec::Discover("dual_fire_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: Some(host_cfg_str),
        config_file_def: Some(("--layered-config {file}", "/include-files/layers.json")),
        kernels: &[],
    }));
    let entry = KtstrTestEntry {
        name: "dual_fire_smoke",
        scheduler: sched,
        config_content: Some("{\"layers\": []}\n"),
        ..KtstrTestEntry::DEFAULT
    };
    let additions =
        compute_config_export_additions(&entry).expect("compute additions must not error");
    assert_eq!(
        additions.len(),
        2,
        "both config_file and config_content slots set must yield 2 additions \
             (orthogonal — each contributes its own scheduler arg), got {} addition(s)",
        additions.len()
    );
    // Order is documented as file-first, content-second by the
    // helper at compute_config_export_additions (push order).
    assert_eq!(
        additions[0].args_shell_prefix, "--config \"$DIR/include/static-cfg.json\"",
        "first addition must be the config_file source"
    );
    assert_eq!(
        additions[1].args_shell_prefix, "--layered-config \"$DIR/include/layers.json\"",
        "second addition must be the config_content source"
    );
}

/// `config_file_addition` mirrors `resolve_include_files`'s
/// directory-reject: a `scheduler.config_file` pointing at a
/// directory must fail with an actionable error at export
/// resolution time, not late during `build_archive`'s
/// `std::fs::read` (which would surface a less-actionable
/// EISDIR). Recursive directory packaging is a v2 enhancement;
/// the user-facing error names the constraint explicitly.
#[test]
fn config_file_addition_rejects_directory_with_actionable_error() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let dir_path = tmp.path().join("config-as-dir");
    std::fs::create_dir(&dir_path).expect("create directory fixture");
    let dir_path_str: &'static str =
        Box::leak(dir_path.to_string_lossy().into_owned().into_boxed_str());
    let sched: &'static Scheduler = Box::leak(Box::new(Scheduler {
        name: "directory_reject_test",
        binary: SchedulerSpec::Discover("directory_reject_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: Some(dir_path_str),
        config_file_def: None,
        kernels: &[],
    }));
    let entry = KtstrTestEntry {
        name: "directory_reject_smoke",
        scheduler: sched,
        ..KtstrTestEntry::DEFAULT
    };
    let err =
        config_file_addition(&entry).expect_err("config_file pointing at a directory must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("is a directory"),
        "error must name the directory failure mode; got: {msg}"
    );
    assert!(
        msg.contains("config_file must point at a regular file"),
        "error must name the v1 constraint actionably; got: {msg}"
    );
}

/// `config_file_addition` rejects basenames that contain shell
/// metacharacters which would break the
/// `"$DIR/include/<basename>"` double-quote interpolation in
/// the .run script (`"`, `\`, `$`, `` ` ``). Defense-in-depth
/// — test-author input is trusted (static string slots), but a
/// regression that landed a scheduler with a metacharacter-laden
/// basename would silently produce a broken preamble.
#[test]
fn config_file_addition_rejects_basename_with_shell_metacharacters() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    let tmp = tempfile::TempDir::new().expect("temp dir");
    // Use `$` which is both a shell metacharacter AND a valid
    // POSIX filename character on most filesystems; if the host
    // FS rejects the filename outright, the test is documented
    // as host-dependent below.
    let evil_path = tmp.path().join("$evil.json");
    if std::fs::write(&evil_path, b"{}\n").is_err() {
        crate::report::test_skip("filesystem rejected fixture with $ in basename");
        return;
    }
    let evil_path_str: &'static str =
        Box::leak(evil_path.to_string_lossy().into_owned().into_boxed_str());
    let sched: &'static Scheduler = Box::leak(Box::new(Scheduler {
        name: "metachar_reject_test",
        binary: SchedulerSpec::Discover("metachar_reject_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: Some(evil_path_str),
        config_file_def: None,
        kernels: &[],
    }));
    let entry = KtstrTestEntry {
        name: "metachar_reject_smoke",
        scheduler: sched,
        ..KtstrTestEntry::DEFAULT
    };
    let err =
        config_file_addition(&entry).expect_err("basename with shell metacharacter must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("shell-metacharacter"),
        "error must name the shell-metacharacter constraint; got: {msg}"
    );
    assert!(
        msg.contains('$') || msg.contains("\"$\""),
        "error must name the offending character; got: {msg}"
    );
}

/// Archive-layout-vs-args-prefix basename parity: the basename
/// embedded in `args_shell_prefix` MUST match the basename
/// `build_archive` flattens to `include/<basename>`. A
/// regression that derived basename differently between the
/// two sites would extract the file to one path but launch
/// the scheduler pointing at another. Roundtrips a config-bearing
/// entry through `compute_config_export_additions` +
/// `build_archive` + `read_archive_entries` to verify the two
/// sides agree.
#[test]
fn archive_basename_matches_args_shell_prefix_for_config_file() {
    use crate::test_support::{Scheduler, SchedulerSpec};
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let ktstr_path = tmp.path().join("fake-ktstr");
    std::fs::write(&ktstr_path, b"FAKE_KTSTR").expect("write ktstr stub");
    let host_cfg = tmp.path().join("parity-config.json");
    std::fs::write(&host_cfg, b"{\"parity\": true}\n").expect("write fixture cfg");
    let host_cfg_str: &'static str =
        Box::leak(host_cfg.to_string_lossy().into_owned().into_boxed_str());
    let sched: &'static Scheduler = Box::leak(Box::new(Scheduler {
        name: "basename_parity_test",
        binary: SchedulerSpec::Discover("basename_parity_test_bin"),
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NO_OVERRIDES,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: Some(host_cfg_str),
        config_file_def: None,
        kernels: &[],
    }));
    let entry = KtstrTestEntry {
        name: "basename_parity_smoke",
        scheduler: sched,
        ..KtstrTestEntry::DEFAULT
    };
    let additions =
        compute_config_export_additions(&entry).expect("compute additions must not error");
    assert_eq!(additions.len(), 1, "expected exactly 1 addition");
    let archive =
        build_archive(&ktstr_path, None, &[additions[0].host_path.clone()]).expect("build archive");
    let entries = read_archive_entries(&archive);
    let archive_names: Vec<&str> = entries.iter().map(|(n, _, _)| n.as_str()).collect();
    assert!(
        archive_names.contains(&"include/parity-config.json"),
        "archive must contain include/parity-config.json (basename derived from \
             host_path.file_name()); got: {archive_names:?}"
    );
    assert!(
        additions[0]
            .args_shell_prefix
            .contains("\"$DIR/include/parity-config.json\""),
        "args_shell_prefix must reference the same basename build_archive flattens to; \
             got: {:?}",
        additions[0].args_shell_prefix
    );
}

/// `write_runfile` produces the exact on-disk shape the
/// preamble's extraction step depends on:
/// 1. The preamble bytes verbatim
/// 2. A literal `__ARCHIVE__\n` marker line
/// 3. Base64-encoded archive bytes split into lines of <= 76
///    characters each
///
/// The runfile must also be mode 0o755 so the operator can run
/// `./repro.run` directly. Decoding the base64 chunk back must
/// reproduce the original archive bytes exactly.
#[test]
fn runfile_layout_and_archive_roundtrip() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let out = tmp.path().join("smoke.run");
    let preamble = "#!/bin/bash\necho hello\n";
    // Use bytes that look nothing like ASCII so we know we're
    // verifying the binary roundtrip and not picking up a
    // text-mode coincidence.
    let archive: Vec<u8> = (0u8..=255).chain(0u8..=128).collect();

    write_runfile(&out, preamble, &archive).expect("write_runfile");

    // Mode 0o755 on the file itself.
    let mode = std::fs::metadata(&out).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755, "runfile mode must be 0o755");

    let raw = std::fs::read_to_string(&out).expect("read runfile");
    let marker = "\n__ARCHIVE__\n";
    let split_at = raw
        .find(marker)
        .expect("runfile must contain a __ARCHIVE__ marker line");
    // The preamble portion is everything up to and including the
    // newline that immediately precedes __ARCHIVE__. The preamble
    // string already terminates with `\n`, so the first split
    // point includes that trailing newline.
    assert_eq!(
        &raw[..split_at + 1],
        preamble,
        "preamble must be written verbatim before the marker",
    );

    let after = &raw[split_at + marker.len()..];
    for line in after.lines() {
        assert!(
            line.len() <= 76,
            "base64 line must be <= 76 cols (POSIX MIME width), got {}: {line:?}",
            line.len(),
        );
    }
    let joined: String = after.lines().collect();
    let decoded = BASE64
        .decode(joined.as_bytes())
        .expect("base64 decode of runfile tail must succeed");
    assert_eq!(
        decoded, archive,
        "base64 roundtrip must reproduce the input archive bytes",
    );
}

/// End-to-end smoke: build_archive + generate_preamble +
/// write_runfile compose into a runfile whose extracted archive
/// contains the expected entries AND whose preamble parses
/// under `bash -n`. Validates that the three pipeline stages
/// glue together without truncation, escaping bugs, or shape
/// drift between layers.
#[test]
fn export_pipeline_round_trip_for_eevdf_entry() {
    if which_bash().is_none() {
        crate::report::test_skip("no bash on PATH");
        return;
    }
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let ktstr_path = tmp.path().join("fake-ktstr");
    std::fs::write(&ktstr_path, b"FAKE_KTSTR").expect("write ktstr stub");
    let inc = tmp.path().join("topology.yaml");
    std::fs::write(&inc, b"some: yaml").expect("write include");
    let out = tmp.path().join("e2e.run");

    let entry = KtstrTestEntry {
        name: "export_smoke",
        ..KtstrTestEntry::DEFAULT
    };
    let archive =
        build_archive(&ktstr_path, None, std::slice::from_ref(&inc)).expect("build archive");
    let preamble = generate_preamble(&entry, false, &[]);
    write_runfile(&out, &preamble, &archive).expect("write runfile");

    // Preamble half: split at marker, run bash -n on the
    // preamble portion.
    let raw = std::fs::read_to_string(&out).expect("read runfile");
    let split_at = raw.find("\n__ARCHIVE__\n").expect("marker present");
    assert_bash_n_accepts(&raw[..split_at + 1], false);

    // Archive half: decode + verify both expected entries.
    let entries = read_archive_entries(&archive);
    let names: Vec<&str> = entries.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec!["ktstr", "include/topology.yaml"],
        "archive must contain ktstr and the single include entry",
    );

    // Preamble half (positive content checks): operator-visible
    // identifiers must reflect the entry name and the test's
    // duration default (DEFAULT.duration is 12s).
    assert!(
        raw[..split_at].contains("KTSTR_TEST_NAME=export_smoke"),
        "preamble must name the entry",
    );
    assert!(
        raw[..split_at].contains("TEST_DURATION_SECS=12"),
        "preamble must reflect the entry's duration",
    );
}

/// Pipe `script` through `bash -n` and assert exit status 0.
/// `has_scheduler` is included only to make failure messages
/// distinguish the two preamble shapes.
fn assert_bash_n_accepts(script: &str, has_scheduler: bool) {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let bash = which_bash().expect("bash should have been checked by caller");
    let mut child = Command::new(&bash)
        .arg("-n")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bash -n");
    child
        .stdin
        .as_mut()
        .expect("bash stdin")
        .write_all(script.as_bytes())
        .expect("pipe script to bash");
    let output = child.wait_with_output().expect("bash -n wait");
    assert!(
        output.status.success(),
        "bash -n rejected the preamble (has_scheduler={has_scheduler}); \
             stderr:\n{}\nscript:\n{script}",
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Locate `bash` on the host running the unit tests. Returns
/// `None` rather than panicking when bash is absent; callers
/// use the `None` path to skip the bash-dependent checks.
fn which_bash() -> Option<PathBuf> {
    search_path_for("bash")
}
