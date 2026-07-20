//! Poll-with-timeout `flock(2)` acquire — the high-level entry point
//! that wraps [`super::primitives::try_flock`] in a deadline loop and
//! decorates timeout errors with the holder list from
//! [`super::proc_locks`] / [`super::holder`].
//!
//! Used by every ktstr surface that wants bounded-wait flock semantics:
//!
//!  - [`crate::cache`]'s per-cache-entry coordination locks (cache key
//!    derives the lockfile path; the helper handles open + flock +
//!    holder-enumeration on timeout).
//!  - The per-run-key sidecar-write locks in
//!    `crate::test_support::sidecar` (run-dir parent + .locks/{leaf}
//!    convention).
//!  - [`crate::vmm::disk_template`]'s per-template-key cache-entry
//!    lock (`acquire_template_lock`, `LOCK_EX` with a remediation
//!    hint).
//!
//! These share the timeout-with-holder-list error shape that
//! `crate::test_support::eval::kernel::is_flock_timeout_message`
//! keys on to classify a flock timeout as a typed, retryable
//! `ResourceContention` rather than an unclassified hard FAIL.

use anyhow::Result;
use std::os::fd::OwnedFd;
use std::path::Path;

use super::FlockMode;
use super::holder::format_holder_list;
use super::primitives::try_flock;
use super::proc_locks::read_holders;

/// Poll interval for [`acquire_flock_with_timeout`].
/// `flock(2)` has no native timed-wait variant; the helper emulates
/// one by retrying the non-blocking form with a short sleep between
/// attempts. 100 ms balances responsiveness — contention typically
/// clears in ≤1 poll under normal load — against CPU burn (at most
/// 10 wakes/s per waiter). Both prior call sites (the cache
/// module's per-cache-entry locks and the run-dir flock surface in
/// `crate::test_support::sidecar`) converged on the same 100 ms
/// value before consolidation; hoisting the constant here makes
/// that consistency explicit.
const FLOCK_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Poll-acquire an advisory `flock(2)` at `lock_path` with a
/// wall-clock `timeout`.
///
/// Generic over caller context — both [`crate::cache`]'s
/// per-cache-entry coordination locks and the per-run-key
/// sidecar-write locks in `crate::test_support::sidecar`
/// delegate to this helper. Callers derive the lockfile path
/// themselves (the cache module keys off `cache_key`, sidecar keys
/// off `dir.parent()/.locks/{leaf}.lock`) and pass it in alongside
/// a `context` string that names the resource being locked. The
/// timeout error renders as `"flock {kind_str} on {context} timed
/// out after {timeout:?} (lockfile {lock_path}, holders:
/// {holders})."` plus an optional remediation tail.
///
/// `flock(2)` has no cancellable blocking variant, so the helper
/// loops [`try_flock`]'s non-blocking form and sleeps
/// [`FLOCK_POLL_INTERVAL`] between attempts. Each `Ok(None)`
/// iteration emits a `tracing::debug!` event keyed to the lockfile
/// path so an operator who installs a `tracing-subscriber` filter
/// on `target = "ktstr::flock::acquire"` (the `module_path!()`
/// default target) can see contention progress in real time
/// without rebuilding.
///
/// The lockfile's parent directory is created lazily via
/// [`std::fs::create_dir_all`] before the first acquire attempt —
/// `create_dir_all` is idempotent, so callers that already
/// materialize the parent (e.g. via [`super::materialize`]) do not
/// need to gate the helper. Callers that want to fail fast on a
/// missing parent should `materialize` directly instead of relying
/// on this helper.
///
/// `context` should be a short noun phrase ("cache entry
/// {key:?}", "run-dir {dir}") so the rendered error reads
/// naturally. `remediation`, when `Some`, appends an operator-
/// facing recovery hint ("A peer cargo ktstr test process is
/// writing sidecars …; wait for it to finish or kill it, then
/// retry."). Use `None` when the timeout itself is the only
/// signal (the cache shared-lock surface `acquire_shared_lock`
/// today — operators triage via the holder PID list and decide
/// whether to wait or kill; the cache store-lock and
/// disk_template both pass `Some`).
///
/// Returns:
///  - `Ok(OwnedFd)` on successful acquire. Caller drops the fd to
///    release the kernel-side flock; OFD-bound semantics mean no
///    explicit unlock call is needed.
///  - `Err(_)` on (a) parent-directory creation failure, (b) any
///    [`try_flock`] error other than `EWOULDBLOCK` (the helper
///    treats `EWOULDBLOCK` as "keep polling"), or (c) the
///    wall-clock `timeout` elapsing while a peer continues to
///    hold an incompatible lock. The timeout error names the
///    mode (`LOCK_EX` / `LOCK_SH`), the `context`, the `timeout`,
///    the lockfile path, the holder list parsed from
///    `/proc/locks` via [`read_holders`], and (when supplied) the
///    `remediation` string.
pub(crate) fn acquire_flock_with_timeout(
    lock_path: &Path,
    mode: FlockMode,
    timeout: std::time::Duration,
    context: &str,
    remediation: Option<&str>,
) -> Result<OwnedFd> {
    use anyhow::Context;

    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create lock subdirectory {}", parent.display()))?;
    }
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match try_flock(lock_path, mode)? {
            Some(fd) => return Ok(fd),
            None => {
                if std::time::Instant::now() >= deadline {
                    let holders = read_holders(lock_path).unwrap_or_default();
                    let kind_str = match mode {
                        FlockMode::Exclusive => "LOCK_EX",
                        FlockMode::Shared => "LOCK_SH",
                    };
                    let tail = remediation.map(|r| format!(" {r}")).unwrap_or_default();
                    anyhow::bail!(
                        "flock {kind_str} on {context} timed out after \
                         {timeout:?} (lockfile {lock_path}, holders: \
                         {holders}).{tail}",
                        lock_path = lock_path.display(),
                        holders = format_holder_list(&holders),
                    );
                }
                // Log AFTER the deadline check so the final iteration
                // (which bails) does not emit a misleading "waiting"
                // event. Operators tailing the debug stream see a
                // heartbeat for each iteration that actually goes
                // back around the loop, then either an acquire or a
                // bail — never a "waiting" event followed by an
                // immediate timeout.
                tracing::debug!("waiting on flock at {lock_path:?}");
                std::thread::sleep(FLOCK_POLL_INTERVAL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First acquire on a fresh lockfile path materializes the parent
    /// directory and returns an `OwnedFd` holding the requested mode.
    /// Pins the lazy-create contract so callers (cache + sidecar) can
    /// hand a `{some_root}/.locks/{key}.lock` path without pre-running
    /// `create_dir_all`.
    #[test]
    fn acquire_flock_creates_parent_lazily() {
        use tempfile::TempDir;
        let tmp = TempDir::new().expect("tempdir");
        let lockfile = tmp.path().join(".locks").join("fresh.lock");
        assert!(
            !tmp.path().join(".locks").exists(),
            ".locks/ must not exist before first acquire (sanity)",
        );
        let fd = acquire_flock_with_timeout(
            &lockfile,
            FlockMode::Exclusive,
            std::time::Duration::from_secs(1),
            "test",
            None,
        )
        .expect("first acquire on fresh path must succeed");
        assert!(
            tmp.path().join(".locks").is_dir(),
            "parent .locks/ must be created lazily on first acquire",
        );
        assert!(
            lockfile.exists(),
            "lockfile itself must materialize via try_flock's O_CREAT",
        );
        drop(fd);
    }

    /// While a peer fd holds `LOCK_EX` on the same lockfile, the
    /// helper polls until `timeout` elapses and then returns an error
    /// whose message names the mode (`LOCK_EX`), the `context`, and
    /// the `timed out` substring. Exercises the contention path that
    /// the `tracing::debug!` log decorates and that the wrapper-level
    /// tests (cache + sidecar) also cover end-to-end.
    #[test]
    fn acquire_flock_times_out_when_peer_holds_lock() {
        use tempfile::TempDir;
        let tmp = TempDir::new().expect("tempdir");
        let lockfile = tmp.path().join(".locks").join("contended.lock");
        std::fs::create_dir_all(lockfile.parent().unwrap()).unwrap();
        let _peer = try_flock(&lockfile, FlockMode::Exclusive)
            .expect("peer flock attempt")
            .expect("peer must acquire on a fresh lockfile");
        let start = std::time::Instant::now();
        let err = acquire_flock_with_timeout(
            &lockfile,
            FlockMode::Exclusive,
            std::time::Duration::from_millis(300),
            "contended-resource",
            None,
        )
        .expect_err("acquire must fail while peer holds LOCK_EX");
        let elapsed = start.elapsed();
        // The helper must wait roughly the requested timeout before
        // erroring — proves it polled rather than returning
        // EWOULDBLOCK on the first try.
        assert!(
            elapsed >= std::time::Duration::from_millis(250),
            "acquire must wait ~timeout before erroring; elapsed={elapsed:?}",
        );
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timed out"),
            "error must surface the timeout cause; got: {msg}",
        );
        assert!(
            msg.contains("LOCK_EX"),
            "error must name the flock mode for operator triage; got: {msg}",
        );
        assert!(
            msg.contains("contended-resource"),
            "error must include the caller-supplied context; got: {msg}",
        );
    }

    /// Mode-string asymmetry: the timeout error names `LOCK_SH` for
    /// shared-mode acquires and `LOCK_EX` for exclusive. Both call
    /// sites trigger the timeout error with mode-specific wording so
    /// operators reading the log can distinguish reader vs writer
    /// contention. The exclusive case is covered above; this test
    /// pins the shared variant.
    #[test]
    fn acquire_flock_shared_timeout_names_lock_sh() {
        use tempfile::TempDir;
        let tmp = TempDir::new().expect("tempdir");
        let lockfile = tmp.path().join(".locks").join("shared.lock");
        std::fs::create_dir_all(lockfile.parent().unwrap()).unwrap();
        // Peer holds LOCK_EX so the SHARED acquire below must wait.
        let _peer = try_flock(&lockfile, FlockMode::Exclusive)
            .expect("peer flock attempt")
            .expect("peer must acquire on a fresh lockfile");
        let err = acquire_flock_with_timeout(
            &lockfile,
            FlockMode::Shared,
            std::time::Duration::from_millis(150),
            "shared-test",
            None,
        )
        .expect_err("shared acquire must fail under LOCK_EX peer");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("LOCK_SH"),
            "shared-mode timeout must name LOCK_SH; got: {msg}",
        );
    }

    /// `remediation = Some(...)` appends the operator-facing recovery
    /// hint to the timeout error; `None` omits it. The sidecar
    /// surface, the cache store-lock, and disk_template use `Some`
    /// (peer-write / stale-lock recovery instructions); only the
    /// cache shared-lock uses `None` (operator triages via holder
    /// PID list). This test pins the contract end-to-end so a
    /// regression that ignores `remediation` (or always appends it
    /// even on `None`) surfaces here.
    #[test]
    fn acquire_flock_remediation_appended_when_some() {
        use tempfile::TempDir;
        let tmp = TempDir::new().expect("tempdir");
        let locks_dir = tmp.path().join(".locks");
        std::fs::create_dir_all(&locks_dir).unwrap();
        let hint = "Wait for peer or kill it, then retry.";

        // Some(hint) and None each run against their own lockfile +
        // peer fd. Two lockfiles instead of one + drop-and-recreate
        // removes the implicit dependency on Drop ordering: the
        // None case was previously gated on the LOCK_EX peer from
        // the Some case being dropped first, which made the test
        // order-sensitive and easy to break by reordering the arms.
        let lockfile_with = locks_dir.join("remediation_some.lock");
        let _peer_with = try_flock(&lockfile_with, FlockMode::Exclusive)
            .expect("peer (Some case) flock attempt")
            .expect("peer (Some case) must acquire on a fresh lockfile");
        let err_with = acquire_flock_with_timeout(
            &lockfile_with,
            FlockMode::Exclusive,
            std::time::Duration::from_millis(120),
            "rem-test",
            Some(hint),
        )
        .expect_err("acquire must fail under LOCK_EX peer (Some case)");
        let msg_with = format!("{err_with:#}");
        assert!(
            msg_with.contains(hint),
            "Some(hint) must append the remediation; got: {msg_with}",
        );

        let lockfile_without = locks_dir.join("remediation_none.lock");
        let _peer_without = try_flock(&lockfile_without, FlockMode::Exclusive)
            .expect("peer (None case) flock attempt")
            .expect("peer (None case) must acquire on a fresh lockfile");
        let err_without = acquire_flock_with_timeout(
            &lockfile_without,
            FlockMode::Exclusive,
            std::time::Duration::from_millis(120),
            "rem-test",
            None,
        )
        .expect_err("acquire must fail under LOCK_EX peer (None case)");
        let msg_without = format!("{err_without:#}");
        assert!(
            !msg_without.contains(hint),
            "None must NOT append the remediation; got: {msg_without}",
        );
    }

    /// Pin the substring contract that
    /// `crate::test_support::eval::is_flock_timeout_message` keys on
    /// to classify a flock timeout as
    /// [`crate::vmm::host_topology::ResourceContention`] (typed,
    /// retryable) vs a plain anyhow error (unclassified hard FAIL).
    /// The seam tests for BOTH
    /// `"flock LOCK_"` AND `"timed out after"` appearing in the
    /// rendered error; any wording change in
    /// [`acquire_flock_with_timeout`]'s bail format that drops
    /// either substring would silently regress the test-kernel-cache
    /// flock-timeout path from a typed contention verdict to an
    /// unclassified hard FAIL.
    ///
    /// Drives a real contended flock: peer holds `LOCK_EX`, this
    /// thread requests `LOCK_SH` with a short timeout, the helper
    /// times out and bails with the literal format produced at
    /// `acquire_flock_with_timeout`'s deadline arm. Both
    /// `LOCK_EX → LOCK_SH` (the cache shared-lock path) and
    /// `LOCK_EX → LOCK_EX` (the cache store-lock / sidecar epoch-reset /
    /// disk-template path) are exercised so
    /// a regression that affects only one mode label still trips
    /// the assertion.
    #[test]
    fn acquire_flock_timeout_message_pins_eval_seam_substrings() {
        use tempfile::TempDir;
        for mode in [FlockMode::Shared, FlockMode::Exclusive] {
            let tmp = TempDir::new().expect("tempdir");
            let lockfile = tmp.path().join(".locks").join("seam.lock");
            std::fs::create_dir_all(lockfile.parent().unwrap()).unwrap();
            // Peer holds LOCK_EX so the next acquire (regardless of
            // mode) must wait. Holding the OwnedFd in scope keeps
            // the kernel-side flock alive.
            let _peer = try_flock(&lockfile, FlockMode::Exclusive)
                .expect("peer flock attempt")
                .expect("peer must acquire on a fresh lockfile");
            let err = acquire_flock_with_timeout(
                &lockfile,
                mode,
                std::time::Duration::from_millis(120),
                "seam-test",
                None,
            )
            .expect_err("acquire must time out under LOCK_EX peer");
            let msg = format!("{err:#}");
            // BOTH substrings must appear together — the eval seam
            // ANDs them. A wording change that splits or removes
            // either is a contract violation.
            assert!(
                msg.contains("flock LOCK_"),
                "rendered error must include the literal `flock LOCK_` \
                 prefix that eval.rs::is_flock_timeout_message keys on; \
                 mode={mode:?} got: {msg}",
            );
            assert!(
                msg.contains("timed out after"),
                "rendered error must include the literal `timed out \
                 after` substring that eval.rs::is_flock_timeout_message \
                 keys on; mode={mode:?} got: {msg}",
            );
        }
    }
}
