//! `KTSTR_CARGO_TEST_MODE` env-var contract. Single read-helper
//! shared by `test_support` (gauntlet emission, scheduler PATH
//! lookup) and the vmm side (`host_topology`'s LLC / per-CPU flock
//! bypass, `builder`'s matching overcommit-warning suppression).
//! Living at the crate root keeps the contract in one place without
//! pulling a `test_support` dependency into vmm-side production code.

/// True when `KTSTR_CARGO_TEST_MODE` is set to a NON-EMPTY value.
///
/// Marks a test invocation that runs without the cargo-ktstr wrapper
/// (typically `KTSTR_KERNEL=... KTSTR_CARGO_TEST_MODE=1 cargo test
/// -- some_test`). When active, the harness:
///
/// 1. Skips host-topology LLC / per-CPU flock acquisition — tests
///    run on whatever CPUs the OS schedules them onto. No
///    peer-coordination flocks are taken; perf-mode behavior is
///    governed separately by
///    `crate::test_support::runtime::no_perf_mode_active` and is
///    NOT toggled by this flag. The builder's would-be-overcommit
///    warning is suppressed to match, since no budget is enforced.
/// 2. Skips gauntlet variant expansion in nextest discovery —
///    each `#[ktstr_test]` runs once with its declared topology.
///    No multi-kernel fan-out via `KTSTR_KERNEL_LIST`.
/// 3. Resolves `SchedulerSpec::Discover(name)` via a `$PATH` lookup
///    (before the workspace build) so a user can install scx_layered
///    on PATH and run their test without driving the cargo-ktstr
///    build pipeline.
/// 4. Suppresses the early-dispatch coverage-gap warnings that fire
///    under bare `cargo test` when a binary registers real
///    `#[ktstr_test]` entries OR `declare_scheduler!` declarations
///    (warning that gauntlet variants won't fan out and verifier
///    cells won't be emitted without nextest). The operator has
///    acknowledged both tradeoffs by setting the env var.
///
/// Empty-string rejection mirrors
/// `crate::test_support::runtime::no_perf_mode_active`: a stray
/// `KTSTR_CARGO_TEST_MODE=` from CI shell quirks must not silently
/// flip the harness into degraded coordination mode.
pub(crate) fn cargo_test_mode_active() -> bool {
    std::env::var(crate::KTSTR_CARGO_TEST_MODE_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

    /// `KTSTR_CARGO_TEST_MODE=1` (any non-empty value) flips the
    /// flag to true. Pins the activation contract against the
    /// dispatch sites that gate gauntlet emission, scheduler PATH
    /// lookup, and host-topology flock bypass on this exact
    /// predicate.
    #[test]
    fn cargo_test_mode_active_set_non_empty() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
        assert!(cargo_test_mode_active());
        let _env_word = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "yes");
        assert!(cargo_test_mode_active());
    }

    /// Empty-string rejection: `KTSTR_CARGO_TEST_MODE=` does NOT
    /// flip the flag. Mirrors `no_perf_mode_active`'s behavior so
    /// CI shells / docker pass-through that set the var without a
    /// value never accidentally degrade the harness.
    #[test]
    fn cargo_test_mode_active_empty_string_rejected() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "");
        assert!(
            !cargo_test_mode_active(),
            "empty-string KTSTR_CARGO_TEST_MODE must NOT activate; \
             treating it as truthy would silently degrade flock \
             coordination on a stray `--env` pass-through"
        );
    }

    /// Unset env var → false.
    #[test]
    fn cargo_test_mode_active_unset() {
        let _lock = lock_env();
        let _env = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);
        assert!(!cargo_test_mode_active());
    }

    /// `KTSTR_CARGO_TEST_MODE=0` ACTIVATES the bypass — the
    /// predicate is "non-empty", not "C-style truthy." Pins this
    /// surprise so a future contributor cannot silently flip to a
    /// `0=off` semantic without also updating the doc + the 4
    /// existing tests in lockstep.
    #[test]
    fn cargo_test_mode_active_zero_string_activates_per_non_empty_contract() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "0");
        assert!(
            cargo_test_mode_active(),
            "\"0\" is non-empty so the bypass activates per the \
             documented contract; if you expect the C-style \
             \"0=off\" semantic, the doc + predicate must change \
             together"
        );
    }
}
