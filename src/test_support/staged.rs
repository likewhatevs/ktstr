//! Staging-path helpers for the scheduler-lifecycle Ops (#7).
//!
//! Holds the small reserved-name table that prevents a staged
//! scheduler's `Scheduler::name` from colliding with the framework's
//! own boot-time initramfs entries, plus the canonical name →
//! guest-side staging path mapping used by the
//! [`KtstrTestEntry::staged_schedulers`](crate::test_support::KtstrTestEntry::staged_schedulers)
//! pipeline.
//!
//! The pure-helper split keeps this module unit-testable: every
//! function takes a name and returns a `Result` or `String` — no
//! filesystem I/O, no initramfs construction.

/// Names the framework reserves for boot-time initramfs entries.
/// A staged [`Scheduler`](crate::test_support::Scheduler) whose
/// `name` matches any entry here would land at a guest path that
/// shadows a framework slot (e.g. the boot scheduler's `/scheduler`,
/// the kernel-builtin enable/disable shell scripts, the per-test
/// arg files); [`validate_staged_scheduler_name`] rejects such
/// names at validate time so the collision never reaches the
/// initramfs builder.
///
/// The list is intentionally narrow — only names the guest init
/// actually reads (see `src/vmm/rust_init/scheduler.rs` for the
/// `/scheduler` binary spawn in `spawn_scheduler_from_paths`,
/// `build_suffix` in `src/vmm/initramfs.rs` (~1150-1191) for the
/// suffix file shapes).
/// Adding a name here is a behavior change for every existing
/// scheduler whose `name` happens to match; expand cautiously.
pub(crate) const RESERVED_SCHEDULER_NAMES: &[&str] = &[
    "scheduler",
    "sched_args",
    "init",
    "args",
    "exec_cmd",
    "sched_enable",
    "sched_disable",
];

/// Maximum byte length for a staged scheduler `name`. Caps the
/// composed cpio entry path `staging/schedulers/<name>/sched_args`
/// well under the kernel's PATH_MAX (4096) so the kernel cpio
/// extractor's over-length-name skip at `init/initramfs.c:296`
/// (`if (name_len > PATH_MAX) return 0;` — a silent no-extract,
/// not an error) never fires — instead
/// [`validate_staged_scheduler_name`] rejects the over-length name
/// with a sharp error. 128 chosen to comfortably exceed every
/// real-world scheduler name (typically 10-30 bytes, e.g.
/// `scx_mitosis_args_a`) while leaving headroom for descriptive
/// variant suffixes.
pub(crate) const MAX_STAGED_SCHEDULER_NAME_LEN: usize = 128;

/// Reject `name` if it is empty, contains a path separator
/// (`/` or `\`), contains a `\0` byte, starts with `.`, or matches
/// a [`RESERVED_SCHEDULER_NAMES`] entry. These rules guarantee that
/// a future `staged_scheduler_path` call against a validated name
/// produces a path that (a) is unique within the staging tree,
/// (b) cannot escape the staging directory via path-traversal
/// (`../`), and (c) does not collide with the framework's own
/// boot-time initramfs entries.
///
/// `who` is included in the error message so the operator can
/// trace which `KtstrTestEntry` field surfaced the name — typically
/// the entry's `name` field plus the staged scheduler's own
/// `name`, in the form `"<entry_name>.staged_schedulers"`.
pub(crate) fn validate_staged_scheduler_name(who: &str, sched_name: &str) -> anyhow::Result<()> {
    if sched_name.is_empty() {
        anyhow::bail!(
            "{who}: staged Scheduler.name must be non-empty (empty names \
             would collapse to the staging root directory)",
        );
    }
    if sched_name.len() > MAX_STAGED_SCHEDULER_NAME_LEN {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' is {len} bytes; \
             the cap is {MAX_STAGED_SCHEDULER_NAME_LEN} bytes so the \
             composed `staging/schedulers/<name>/sched_args` path stays \
             well under the kernel's PATH_MAX cpio limit \
             (init/initramfs.c silently skips names > PATH_MAX \
             (do_header return 0, no extract); the cap surfaces a \
             cleaner host-side error here)",
            len = sched_name.len(),
        );
    }
    if sched_name.contains('/') || sched_name.contains('\\') {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' must not contain \
             path separators ('/' or '\\') — they would let the staging \
             path escape its scoped directory or land at an unintended \
             guest location. Use '_' or '-' as a separator instead \
             (e.g. 'scx_mitosis_variant_a').",
        );
    }
    if sched_name.contains('\0') {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' must not contain \
             a NUL byte — POSIX filesystem path strings are NUL-terminated",
        );
    }
    if sched_name.starts_with('.') {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' must not start \
             with '.' (would produce a hidden directory in the staging \
             tree that breaks recursive listing for debugging). Use a \
             name starting with a letter or digit.",
        );
    }
    if RESERVED_SCHEDULER_NAMES.contains(&sched_name) {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' is reserved \
             (reserved names {RESERVED_SCHEDULER_NAMES:?} collide with \
             framework boot-time initramfs slots and would be silently \
             overwritten when the suffix archive lands). Pick a name \
             that does not match any of the reserved entries above.",
        );
    }
    Ok(())
}

/// Compute the guest-side directory path under which a staged
/// scheduler's binary + per-scheduler arg file will be packed.
/// Nested layout (`/staging/schedulers/<name>/`) so each scheduler
/// gets its own scope and a future per-scheduler companion file
/// (config, RUST_LOG override, sentinel) can land under the same
/// directory without polluting the archive root.
///
/// Caller must pre-validate via [`validate_staged_scheduler_name`];
/// the helper trusts the input on the hot path. Production code
/// reaches this through the
/// [`KtstrTestEntry::validate`](crate::test_support::KtstrTestEntry::validate)
/// gate which calls the validator on every staged entry before any
/// path expansion fires.
//
// Reached from production via [`staged_scheduler_binary_path`] and
// [`staged_scheduler_args_path`], which the Op::AttachScheduler /
// Op::ReplaceScheduler dispatch calls
// (`src/scenario/ops/dispatch.rs` dispatch_attach_scheduler /
// dispatch_replace_scheduler) before handing the paths to
// `spawn_scheduler_for_op` -> `try_spawn_scheduler`.
pub(crate) fn staged_scheduler_dir(sched_name: &str) -> String {
    format!("/staging/schedulers/{sched_name}")
}

/// Guest path of the staged scheduler binary itself.
/// `<dir>/scheduler` mirrors the boot-time `/scheduler` shape so
/// the dispatch code path reuses the existing
/// `Path::new(binary_path).exists()` pattern against the staged
/// path with no shape divergence (`try_spawn_scheduler`,
/// `src/vmm/rust_init/scheduler.rs:551`).
pub(crate) fn staged_scheduler_binary_path(sched_name: &str) -> String {
    format!("{}/scheduler", staged_scheduler_dir(sched_name))
}

/// Guest path of the staged scheduler's args file. `<dir>/sched_args`
/// mirrors the boot-time `/sched_args` shape — the spawn code
/// (`try_spawn_scheduler`) reads CLI args from this file with the
/// same parser as the boot scheduler launch
/// (`spawn_scheduler_from_paths` also routes through
/// `try_spawn_scheduler`).
pub(crate) fn staged_scheduler_args_path(sched_name: &str) -> String {
    format!("{}/sched_args", staged_scheduler_dir(sched_name))
}

/// Archive-relative directory (no leading `/`) under which a
/// staged scheduler's binary + args file are packed in the cpio
/// initramfs. Mirrors [`staged_scheduler_dir`] with the leading
/// slash stripped so the result can pass directly as a cpio entry
/// name (cpio entries are archive-relative; a leading `/` would
/// create an absolute-path entry the kernel extractor refuses).
///
/// Co-located with the guest-path helpers so a future layout
/// change (e.g. renaming `staging` → `dynsched`) updates a
/// single module instead of drifting between the packer and the
/// runtime resolver.
pub(crate) fn staged_scheduler_archive_dir(sched_name: &str) -> String {
    format!("staging/schedulers/{sched_name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_staged_scheduler_name_accepts_well_formed_identifier() {
        assert!(
            validate_staged_scheduler_name("entry.staged_schedulers", "scx_mitosis_args_a").is_ok()
        );
    }

    #[test]
    fn validate_staged_scheduler_name_rejects_empty() {
        let err = validate_staged_scheduler_name("e.staged_schedulers", "").unwrap_err();
        assert!(format!("{err:#}").contains("must be non-empty"));
    }

    #[test]
    fn validate_staged_scheduler_name_rejects_path_separators() {
        let err = validate_staged_scheduler_name("e.staged_schedulers", "a/b").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("path separators"));
        // Actionable hint — operator must know WHAT to use, not just
        // what NOT to use.
        assert!(msg.contains("'_' or '-'"), "hint missing: {msg}");
        let err = validate_staged_scheduler_name("e.staged_schedulers", "a\\b").unwrap_err();
        assert!(format!("{err:#}").contains("path separators"));
    }

    #[test]
    fn validate_staged_scheduler_name_dot_prefix_message_carries_hint() {
        let err = validate_staged_scheduler_name("e.staged_schedulers", ".hidden").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("letter or digit"),
            "hint missing on dot-prefix error: {msg}"
        );
    }

    #[test]
    fn validate_staged_scheduler_name_reserved_message_lists_reserved_names() {
        let err = validate_staged_scheduler_name("e.staged_schedulers", "scheduler").unwrap_err();
        let msg = format!("{err:#}");
        // Both the reserved-list dump AND an actionable suggestion
        // must surface so the operator can resolve without guessing.
        assert!(msg.contains("sched_args"), "reserved list missing: {msg}");
        assert!(
            msg.contains("does not match"),
            "hint missing on reserved error: {msg}"
        );
    }

    #[test]
    fn validate_staged_scheduler_name_rejects_overlong_name() {
        let long_name = "a".repeat(MAX_STAGED_SCHEDULER_NAME_LEN + 1);
        let err = validate_staged_scheduler_name("e.staged_schedulers", &long_name).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("bytes"),
            "length-cap error missing 'bytes': {msg}"
        );
        assert!(
            msg.contains(&format!("{}", MAX_STAGED_SCHEDULER_NAME_LEN)),
            "length-cap error must mention the cap: {msg}"
        );
    }

    #[test]
    fn validate_staged_scheduler_name_accepts_name_exactly_at_max_length() {
        let exact = "a".repeat(MAX_STAGED_SCHEDULER_NAME_LEN);
        assert!(
            validate_staged_scheduler_name("e.staged_schedulers", &exact).is_ok(),
            "name exactly at MAX_STAGED_SCHEDULER_NAME_LEN must be accepted"
        );
    }

    #[test]
    fn validate_staged_scheduler_name_rejects_null_byte() {
        let err = validate_staged_scheduler_name("e.staged_schedulers", "a\0b").unwrap_err();
        assert!(format!("{err:#}").contains("NUL byte"));
    }

    #[test]
    fn validate_staged_scheduler_name_rejects_dot_prefix() {
        let err = validate_staged_scheduler_name("e.staged_schedulers", ".hidden").unwrap_err();
        assert!(format!("{err:#}").contains("must not start"));
    }

    #[test]
    fn validate_staged_scheduler_name_rejects_each_reserved_name() {
        for &reserved in RESERVED_SCHEDULER_NAMES {
            let err = validate_staged_scheduler_name("e.staged_schedulers", reserved).unwrap_err();
            assert!(
                format!("{err:#}").contains("reserved"),
                "reserved name {reserved:?} must surface 'reserved' in error message: {err:#}"
            );
        }
    }

    #[test]
    fn staged_scheduler_dir_uses_nested_layout() {
        assert_eq!(
            staged_scheduler_dir("scx_mitosis_args_a"),
            "/staging/schedulers/scx_mitosis_args_a"
        );
    }

    #[test]
    fn staged_scheduler_binary_path_appends_scheduler_to_dir() {
        assert_eq!(
            staged_scheduler_binary_path("scx_mitosis"),
            "/staging/schedulers/scx_mitosis/scheduler"
        );
    }

    #[test]
    fn staged_scheduler_args_path_appends_sched_args_to_dir() {
        assert_eq!(
            staged_scheduler_args_path("scx_mitosis"),
            "/staging/schedulers/scx_mitosis/sched_args"
        );
    }

    #[test]
    fn staged_scheduler_archive_dir_strips_leading_slash() {
        assert_eq!(
            staged_scheduler_archive_dir("scx_mitosis_args_a"),
            "staging/schedulers/scx_mitosis_args_a"
        );
    }

    /// Archive form must be exactly the guest form minus the leading
    /// `/` — verifies the two helpers stay in lockstep so a future
    /// layout rename doesn't accidentally diverge the cpio entry
    /// path from the runtime resolver path.
    #[test]
    fn staged_scheduler_archive_dir_matches_guest_dir_without_leading_slash() {
        let name = "scx_mitosis";
        let guest = staged_scheduler_dir(name);
        let archive = staged_scheduler_archive_dir(name);
        assert_eq!(guest.strip_prefix('/').unwrap(), archive);
    }
}
