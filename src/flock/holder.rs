//! [`super::HolderInfo`] construction + rendering for diagnostics.
//!
//! Two roles:
//!
//!  - [`holder_info_for_pid`] — read `/proc/{pid}/cmdline`, shape it
//!    for a [`super::HolderInfo`] (lossy UTF-8, `\0 → space`, truncate
//!    to [`CMDLINE_MAX_CHARS`] with `…` suffix). Missing /
//!    permission-denied / racing /proc entries — and an empty
//!    cmdline (kernel threads) — fall back to
//!    `"<cmdline unavailable>"` so the PID still surfaces.
//!  - [`format_holder_list`] — render a `&[HolderInfo]` for inclusion
//!    in an operator-facing error string. Empty list yields the
//!    [`NO_HOLDERS_RECORDED`] sentinel; non-empty renders one
//!    `pid={pid} cmd={cmdline}` line per holder, newline-separated
//!    and two-space-indented.

use super::HolderInfo;

/// Cmdline truncation limit: 100 chars, so a rendered holder line
/// stays single-line. Referenced by `HolderInfo`'s doc in
/// [`super::HolderInfo`].
const CMDLINE_MAX_CHARS: usize = 100;

/// Diagnostic text for lock-holder error messages when /proc/locks
/// lists no PID against the lockfile inode. Centralized so every
/// caller renders the empty-holders case with the same string.
/// Non-empty so log-scrapers can key on it without accidentally
/// matching a blank field.
pub(crate) const NO_HOLDERS_RECORDED: &str = "<none recorded>";

/// Read and shape `/proc/{pid}/cmdline` for a [`HolderInfo`].
/// `\0` → ` `, lossy UTF-8, truncated to [`CMDLINE_MAX_CHARS`] with
/// `…` suffix on overflow. Missing / racing / permission-denied on
/// `/proc/{pid}/cmdline` — and a successfully-read but empty cmdline
/// (kernel threads) — produce `"<cmdline unavailable>"`; the pid
/// still carries diagnostic value even without the command.
pub(super) fn holder_info_for_pid(pid: u32) -> HolderInfo {
    let raw = match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(bytes) => bytes,
        Err(_) => {
            return HolderInfo {
                pid,
                cmdline: "<cmdline unavailable>".to_string(),
            };
        }
    };
    // Kernel writes argv joined with \0 and terminated by \0. Lossy
    // decode handles non-UTF-8 argv bytes (rare — most binaries use
    // UTF-8 args, but the kernel does not enforce it).
    let text: String = String::from_utf8_lossy(&raw)
        .chars()
        .map(|c| if c == '\0' { ' ' } else { c })
        .collect::<String>()
        .trim_end()
        .to_string();
    let truncated = if text.chars().count() > CMDLINE_MAX_CHARS {
        let head: String = text.chars().take(CMDLINE_MAX_CHARS).collect();
        format!("{head}…")
    } else if text.is_empty() {
        "<cmdline unavailable>".to_string()
    } else {
        text
    };
    HolderInfo {
        pid,
        cmdline: truncated,
    }
}

/// Format a [`HolderInfo`] slice for inclusion in user-facing error
/// strings. Empty slice yields the `NO_HOLDERS_RECORDED` sentinel so the
/// diagnostic is unambiguous — a stale lockfile whose holder has
/// exited presents as empty, and the error should say so rather than
/// print a misleading blank. Non-empty renders one
/// `pid={pid} cmd={cmdline}` line per holder, newline-separated and
/// indented two spaces, so a multi-holder error stays readable when
/// embedded in a wrapping anyhow chain; the prior comma-joined form
/// ran every holder into a single wide line that terminals wrapped
/// arbitrarily mid-cmdline.
pub fn format_holder_list(holders: &[HolderInfo]) -> String {
    if holders.is_empty() {
        NO_HOLDERS_RECORDED.to_string()
    } else {
        holders
            .iter()
            .map(|h| format!("  pid={} cmd={}", h.pid, h.cmdline))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // format_holder_list — rendering contract
    // ---------------------------------------------------------------

    /// Empty slice yields the sentinel [`NO_HOLDERS_RECORDED`] so
    /// log-scrapers have a stable key.
    #[test]
    fn format_holder_list_empty_yields_sentinel() {
        assert_eq!(format_holder_list(&[]), NO_HOLDERS_RECORDED);
    }

    /// Single holder renders with the `  pid={pid} cmd={cmdline}`
    /// shape. Two-space indent is load-bearing — a future revert
    /// to comma-join would break terminal rendering on
    /// multi-holder lockfiles.
    #[test]
    fn format_holder_list_single_holder() {
        let holders = [HolderInfo {
            pid: 12345,
            cmdline: "cargo build".to_string(),
        }];
        assert_eq!(format_holder_list(&holders), "  pid=12345 cmd=cargo build");
    }

    /// Multiple holders newline-separated (not comma-joined). The
    /// previous shape was `", "` — this test pins the newline.
    #[test]
    fn format_holder_list_multiple_newline_separated() {
        let holders = [
            HolderInfo {
                pid: 1,
                cmdline: "a".to_string(),
            },
            HolderInfo {
                pid: 2,
                cmdline: "b".to_string(),
            },
        ];
        let out = format_holder_list(&holders);
        assert!(out.contains("\n"), "must contain newline: {out}");
        assert!(!out.contains(", "), "must NOT contain comma-space: {out}");
        assert_eq!(out, "  pid=1 cmd=a\n  pid=2 cmd=b");
    }

    /// [`HolderInfo`] serializes with `pid` and `cmdline` as
    /// snake_case keys — stable JSON contract for `ktstr locks
    /// --json` downstream consumers. A future refactor that
    /// rename_all = "camelCase" (or drops the derive) would
    /// silently break shell-script consumers that `jq .[].pid`;
    /// this test pins the key names so that regression fails the
    /// build.
    #[test]
    fn holder_info_json_keys_are_snake_case() {
        let holder = HolderInfo {
            pid: 123,
            cmdline: "bash".to_string(),
        };
        let val = serde_json::to_value(&holder).expect("serialize");
        // Pin both keys exist + have expected types.
        assert_eq!(val["pid"], serde_json::json!(123));
        assert_eq!(val["cmdline"], serde_json::json!("bash"));
        // Negative: no camelCase variants slipped in.
        assert!(
            val.get("cmdLine").is_none(),
            "camelCase cmdLine must not appear: {val}",
        );
    }
}
