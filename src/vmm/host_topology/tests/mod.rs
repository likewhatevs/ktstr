use super::*;

#[test]
fn reservation_wait_progress_hook_is_synchronous_and_scoped() {
    let ticks = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let callback_ticks = std::rc::Rc::clone(&ticks);
    assert!(reservation_wait_progress_poll().is_none());

    let value = with_reservation_wait_progress(
        move || callback_ticks.set(callback_ticks.get() + 1),
        || {
            assert_eq!(
                reservation_wait_progress_poll(),
                Some(RESERVATION_WAIT_PROGRESS_POLL),
            );
            tick_reservation_wait_progress();
            17
        },
    );

    assert_eq!(value, 17);
    assert_eq!(ticks.get(), 1);
    assert!(
        reservation_wait_progress_poll().is_none(),
        "the callback cannot leak into a later acquisition",
    );
}

// ─── SYNTHETIC-TOPOLOGY OFFSET CONVENTION ────────────────────
//
// Flock-path tests in this module install a per-test lockfile
// prefix guard — [`LlcLockPrefixGuard`], [`CpuLockPrefixGuard`],
// or the [`LockPrefixesGuard`] bundle — which redirects their
// lockfiles into a per-test `TempDir` via
// [`LLC_LOCK_PREFIX_OVERRIDE`] / [`CPU_LOCK_PREFIX_OVERRIDE`].
// `llc_lock_path` / `cpu_lock_path` (in the parent module) honor
// that override instead of the production `{lock_dir}/ktstr-llc-`
// / `{lock_dir}/ktstr-cpu-` prefix, so lockfiles land in the
// test's own tempdir. Cross-process collision on a real
// `/tmp/ktstr-llc-*.lock` is therefore impossible regardless of
// the index a test picks.
//
// The 9xxxx LLC/CPU indices that remain in some tests
// (locking.rs 90xxx; planning.rs retry seam 93500/93501/93600,
// cargo-test-mode bypass 95100/95200) are a legacy/organizational
// convention and double as synthetic CPU IDs / LLC indices —
// they are NOT a collision-avoidance requirement, since the
// prefix guards already isolate the lockfile pool. Newer
// acquire_llc_plan tests use small indices (0, 1, 2, …) under a
// prefix guard instead.
//
// When adding a new test that flocks, install a prefix guard
// (default to [`LockPrefixesGuard`] when in doubt) so the
// lockfiles land in a per-test tempdir; the specific index no
// longer needs to dodge a high range.
// ─────────────────────────────────────────────────────────────

/// Collect the distinct host NUMA node IDs the given CPUs belong
/// to. Tests that assert "these N CPUs all live on one NUMA node"
/// (or span two) route through this helper so the CPU → node
/// lookup and the single-CPU default stay in one place rather
/// than duplicating the same closure across every assertion
/// site.
fn numa_nodes_for_cpus(topo: &HostTopology, cpus: &[usize]) -> std::collections::BTreeSet<usize> {
    cpus.iter()
        .map(|c| topo.cpu_to_node.get(c).copied().unwrap_or(0))
        .collect()
}

// -- synthetic topology mapping tests --

/// Backwards-compat helper: builds a synthetic HostTopology from
/// LLC-group CPU lists, assigning each group to a NUMA node equal
/// to its positional index (LLC 0 → node 0, LLC 1 → node 1, …).
/// Delegates to [`HostTopology::new_for_tests`].
///
/// Kept as a thin wrapper so the many existing call sites that
/// pass only CPU lists (no explicit NUMA info) don't have to
/// thread node ids through their parameter lists.
fn synthetic_topo(groups: Vec<Vec<usize>>) -> HostTopology {
    let tagged: Vec<(Vec<usize>, usize)> = groups
        .into_iter()
        .enumerate()
        .map(|(node, cpus)| (cpus, node))
        .collect();
    HostTopology::new_for_tests(&tagged)
}

// -- NUMA-aware pinning tests --

/// Backwards-compat helper: builds a synthetic HostTopology from
/// `(numa_node, cpu_list)` pairs. Delegates to
/// [`HostTopology::new_for_tests`], flipping the tuple order to
/// `(cpus, node)` so the underlying constructor presents a
/// consistent `(cpus, node)` shape to callers that build pairs
/// directly.
fn synthetic_topo_numa(groups: Vec<(usize, Vec<usize>)>) -> HostTopology {
    let tagged: Vec<(Vec<usize>, usize)> = groups
        .into_iter()
        .map(|(node, cpus)| (cpus, node))
        .collect();
    HostTopology::new_for_tests(&tagged)
}

/// RAII guard for a per-test LLC-plan lockfile pool. Installs
/// `{tempdir}/llc-` and `{tempdir}/cpu-` prefixes on construction
/// and unsets both on Drop because an [`LlcPlan`] reserves shared CPU
/// locks as well as shared LLC locks. Two parallel tests
/// using this guard each get their own tempdir, so their
/// `acquire_llc_plan` lockfiles can't collide. Eliminates the
/// 90K+ empty `LlcGroup` padding that earlier tests used to
/// sidestep collision with real host LLC indices.
///
/// Uses [`tempfile::TempDir`] so cleanup runs via RAII on panic
/// — a panicking test can't leak `/tmp` lockfiles into other
/// test runs.
struct LlcLockPrefixGuard {
    _dir: tempfile::TempDir,
}

impl LlcLockPrefixGuard {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let llc_prefix = format!("{}/llc-", dir.path().display());
        let cpu_prefix = format!("{}/cpu-", dir.path().display());
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(llc_prefix));
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(cpu_prefix));
        LlcLockPrefixGuard { _dir: dir }
    }
}

impl Drop for LlcLockPrefixGuard {
    fn drop(&mut self) {
        LLC_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
    }
}

/// RAII guard for a per-test CPU lockfile path prefix. Mirrors
/// [`LlcLockPrefixGuard`] for the CPU-lock side of the
/// explicit two-resource-class admission path. See that struct's doc for the
/// per-test-tempdir + panic-safe-cleanup rationale.
struct CpuLockPrefixGuard {
    _dir: tempfile::TempDir,
}

impl CpuLockPrefixGuard {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let prefix = format!("{}/cpu-", dir.path().display());
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = Some(prefix));
        CpuLockPrefixGuard { _dir: dir }
    }
}

impl Drop for CpuLockPrefixGuard {
    fn drop(&mut self) {
        CPU_LOCK_PREFIX_OVERRIDE.with(|p| *p.borrow_mut() = None);
    }
}

/// RAII bundle that installs BOTH [`LlcLockPrefixGuard`]
/// AND [`CpuLockPrefixGuard`] in one call. Used by any
/// test that hits both LLC and CPU lockfile families —
/// topology admission (LLC + per-CPU), or any
/// future helper that composes the two. Each test gets
/// its own per-tempdir prefix for both lockfile families,
/// so cross-run / cross-process collisions on
/// `/tmp/ktstr-llc-*.lock` and `/tmp/ktstr-cpu-*.lock`
/// cannot occur. When in doubt about which guard to pick,
/// default to this bundle — over-provisioning a tempdir
/// is cheap and is always safe; under-provisioning leaks
/// production-path test collisions.
struct LockPrefixesGuard {
    _cpu: CpuLockPrefixGuard,
    _llc: LlcLockPrefixGuard,
    retry_wake_marker: Option<std::path::PathBuf>,
}

impl LockPrefixesGuard {
    fn new() -> Self {
        Self::new_with_retry_wake(true)
    }

    /// Keep the production real-inotify transport for end-to-end wake
    /// contract tests while retaining isolated lock paths.
    fn new_real_wake() -> Self {
        Self::new_with_retry_wake(false)
    }

    fn new_with_retry_wake(test_retry: bool) -> Self {
        let cpu = CpuLockPrefixGuard::new();
        let llc = LlcLockPrefixGuard::new();
        let retry_wake_marker = test_retry.then(|| {
            let marker = super::protocol::test_retry_wake_marker_path_for_tests();
            std::fs::write(&marker, b"test-retry")
                .expect("create test retry wake marker beside registry");
            marker
        });
        Self {
            _cpu: cpu,
            _llc: llc,
            retry_wake_marker,
        }
    }
}

impl Drop for LockPrefixesGuard {
    fn drop(&mut self) {
        if let Some(marker) = self.retry_wake_marker.take() {
            let _ = std::fs::remove_file(marker);
        }
    }
}

/// RAII guard for a per-test override of
/// [`host_allowed_cpus`]'s return value via
/// [`ALLOWED_CPUS_OVERRIDE`]. Lets tests pin the 30%-default and
/// allowed-cpu filtering math to a known input regardless of
/// what the CI runner's real sched_getaffinity returns. Unset on
/// Drop so a panicking test cannot leak state across the suite.
struct AllowedCpusGuard;

impl AllowedCpusGuard {
    fn new(cpus: Vec<usize>) -> Self {
        ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = Some(cpus));
        AllowedCpusGuard
    }
}

impl Drop for AllowedCpusGuard {
    fn drop(&mut self) {
        ALLOWED_CPUS_OVERRIDE.with(|p| *p.borrow_mut() = None);
    }
}

/// Destructure a `LockOutcome::Acquired { llc_offset, locks }` or
/// panic with a stable diagnostic on `Unavailable`. `ctx` is an
/// optional site-specific clause that the panic message inlines
/// after "expected Acquired" with a single leading space:
/// `None` produces `"expected Acquired, got Unavailable: ..."`,
/// `Some("in cargo-test mode")` produces
/// `"expected Acquired in cargo-test mode, got Unavailable: ..."`.
/// The helper owns the space-prefix so callers cannot accidentally
/// produce `"expected Acquiredfoo"` by forgetting it.
///
/// See [`expect_unavailable`] for tests that expect the
/// `Unavailable` branch instead.
fn unwrap_acquired(outcome: LockOutcome, ctx: Option<&str>) -> (usize, Vec<std::os::fd::OwnedFd>) {
    match outcome {
        LockOutcome::Acquired { llc_offset, locks } => (llc_offset, locks),
        LockOutcome::Unavailable(reason) => {
            let suffix = ctx.map(|c| format!(" {c}")).unwrap_or_default();
            panic!("expected Acquired{suffix}, got Unavailable: {reason}")
        }
    }
}

/// Destructure a `LockOutcome::Unavailable(reason)` for tests that
/// EXPECT the unavailable branch and assert on the reason string.
/// Panics on `Acquired`. `ctx` follows the same convention as
/// [`unwrap_acquired`]: `None` produces
/// `"expected Unavailable, got Acquired"`,
/// `Some("while lock is held")` produces
/// `"expected Unavailable while lock is held, got Acquired"`.
fn expect_unavailable(outcome: LockOutcome, ctx: Option<&str>) -> String {
    match outcome {
        LockOutcome::Unavailable(reason) => reason,
        LockOutcome::Acquired { .. } => {
            let suffix = ctx.map(|c| format!(" {c}")).unwrap_or_default();
            panic!("expected Unavailable{suffix}, got Acquired")
        }
    }
}

// ---------------------------------------------------------------
// CpuCap — construction, env resolution, acquire-time bounding
// ---------------------------------------------------------------

/// Serialize KTSTR_CPU_CAP env-var mutation across test threads.
/// std::env::set_var is process-wide (unsafe in edition 2024);
/// parallel tests would race if each mutated the same variable
/// without coordination. Delegates to the ONE crate-wide env mutex
/// ([`crate::test_support::test_helpers::lock_env`]) so KTSTR_CPU_CAP
/// mutation here serializes against EVERY env-touching test in the
/// lib-test binary — including the builder tests that read it via
/// CpuCap::resolve and the KTSTR_BYPASS_LLC_LOCKS tests. A
/// module-local mutex left those cross-module process-wide env reads
/// racing.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    // lock_env() already recovers from poison.
    crate::test_support::test_helpers::lock_env()
}

/// RAII guard for scoped `std::env::set_var` mutation inside a
/// test. On construction sets the variable to `value`; on Drop
/// removes it regardless of whether the test body panicked or
/// returned early. Pairs with [`env_lock`] — callers take the
/// mutex first, then mint the guard, so two env-touching tests
/// never observe each other's intermediate state.
///
/// Replaces the bare `unsafe { set_var(..) } ... unsafe {
/// remove_var(..) }` pairs that appeared in every env-set test:
/// an early return or panic between the set and the remove used
/// to leak the env var into subsequent tests serialized on the
/// same mutex. `Drop` closes that leak.
struct EnvGuard {
    name: &'static str,
}

impl EnvGuard {
    /// Set `name=value` under the assumed-held `env_lock` mutex.
    /// The caller must have taken `env_lock()` before calling
    /// this constructor — `EnvGuard` does NOT take the mutex
    /// itself because some tests need to interleave multiple
    /// guards (e.g. set, read, remove, re-set) within a single
    /// lock scope.
    fn set(name: &'static str, value: &str) -> Self {
        // SAFETY: caller holds the env_lock mutex; edition 2024
        // set_var is unsafe-marked because it races with reads
        // from other threads, but the mutex serializes every
        // env-touching test so no other test is reading
        // concurrently.
        unsafe {
            std::env::set_var(name, value);
        }
        EnvGuard { name }
    }

    /// Remove `name` under the assumed-held `env_lock` mutex.
    /// Symmetric helper for tests that want to start from a
    /// known-unset state without first creating a set-and-drop
    /// guard.
    fn remove(name: &'static str) -> Self {
        // SAFETY: caller holds the env_lock mutex; see set().
        unsafe {
            std::env::remove_var(name);
        }
        EnvGuard { name }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: guard lifetime is bounded by env_lock held by
        // the test that constructed it. Drop runs before the
        // mutex guard is released, so the remove_var happens
        // under the same mutex as the matching set_var.
        unsafe {
            std::env::remove_var(self.name);
        }
    }
}

// ---------------------------------------------------------------
// NUMA primitives — host_llcs_by_numa_node / with_capacity /
// sorted_by_distance
// ---------------------------------------------------------------

/// Backwards-compat helper: forwards to
/// [`HostTopology::new_for_tests`]. Kept so existing tests that
/// reference `synth_host_topo` don't need to be renamed in lock-
/// step with the consolidation — the single authoritative
/// constructor is `new_for_tests`, this and
/// [`synthetic_topo`] / [`synthetic_topo_numa`] are thin adapters
/// over it.
fn synth_host_topo(groups: &[(Vec<usize>, usize)]) -> HostTopology {
    HostTopology::new_for_tests(groups)
}

// Test groups extracted from the original flat tests.rs; the helper fns
// and RAII scaffolding structs above stay here so every group reaches
// them as a child module via `use super::*`.
mod locking;
mod pinning;
mod planning;
mod protocol;
