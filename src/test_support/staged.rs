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
/// actually reads (see `src/vmm/rust_init.rs:2618` for the
/// `Path::new("/scheduler").exists()` check + L2653 `Command::new("/scheduler")`,
/// `src/vmm/initramfs.rs:1007-1016` for the suffix file shapes).
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

/// Reject `name` if it is empty, contains a path separator
/// (`/` or `\`), contains a `\0` byte, starts with `.`, or matches
/// a [`RESERVED_SCHEDULER_NAMES`] entry. These rules guarantee that
/// a future [`staged_scheduler_path`] call against a validated name
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
    if sched_name.contains('/') || sched_name.contains('\\') {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' must not contain \
             path separators ('/' or '\\') — they would let the staging \
             path escape its scoped directory or land at an unintended \
             guest location",
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
             tree that breaks recursive listing for debugging)",
        );
    }
    if RESERVED_SCHEDULER_NAMES.contains(&sched_name) {
        anyhow::bail!(
            "{who}: staged Scheduler.name '{sched_name}' is reserved \
             (reserved names {RESERVED_SCHEDULER_NAMES:?} collide with \
             framework boot-time initramfs slots and would be silently \
             overwritten when the suffix archive lands)",
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
// `#[allow(dead_code)]` because no production caller exists today
// — the initramfs packing pipeline and Op dispatch that will
// consume the path helpers land in follow-up work. Tests in this
// module exercise the helpers as they land; the allow becomes a
// no-op the moment the first production caller wires up.
#[allow(dead_code)]
pub(crate) fn staged_scheduler_dir(sched_name: &str) -> String {
    format!("/staging/schedulers/{sched_name}")
}

/// Guest path of the staged scheduler binary itself.
/// `<dir>/scheduler` mirrors the boot-time `/scheduler` shape so
/// the future dispatch code path can reuse the existing
/// [`Path::new("/scheduler").exists()`] pattern against the staged
/// path with no shape divergence.
#[allow(dead_code)] // see staged_scheduler_dir
pub(crate) fn staged_scheduler_binary_path(sched_name: &str) -> String {
    format!("{}/scheduler", staged_scheduler_dir(sched_name))
}

/// Guest path of the staged scheduler's args file. `<dir>/sched_args`
/// mirrors the boot-time `/sched_args` shape — the future spawn
/// code reads CLI args from this file with the same parser as the
/// boot scheduler launch.
#[allow(dead_code)] // see staged_scheduler_dir
pub(crate) fn staged_scheduler_args_path(sched_name: &str) -> String {
    format!("{}/sched_args", staged_scheduler_dir(sched_name))
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
        assert!(format!("{err:#}").contains("path separators"));
        let err = validate_staged_scheduler_name("e.staged_schedulers", "a\\b").unwrap_err();
        assert!(format!("{err:#}").contains("path separators"));
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
}
