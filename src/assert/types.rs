use super::*;

thread_local! {
    /// Thread-local active phase label. Set by the [`PhaseGuard`]
    /// scope helper at scenario-driver `run_step` entry and read by
    /// [`AssertDetail::new`] / [`PassDetail::binary`] /
    /// [`PassDetail::unary`] / [`InfoNote::new`] producers so every
    /// detail constructed under a guarded scope auto-stamps its
    /// `phase` field with the active label without the producer
    /// having to thread context through every `with_phase` chain.
    /// `None` outside any guarded scope (boot, BASELINE settle,
    /// non-scenario test fixtures).
    static ACTIVE_PHASE: RefCell<Option<std::borrow::Cow<'static, str>>> =
        const { RefCell::new(None) };
}

/// Snapshot the active phase label installed by the most recent
/// [`PhaseGuard::install`] on this thread. `None` outside any
/// guarded scope. Construction sites for [`AssertDetail`] /
/// [`PassDetail`] / [`InfoNote`] call this to auto-stamp the
/// `phase` field; the test author can still override via the
/// builder `with_phase(...)` chain when an explicit value is
/// preferred.
pub fn current_phase_label() -> Option<std::borrow::Cow<'static, str>> {
    ACTIVE_PHASE.with(|p| p.borrow().clone())
}

/// RAII scope guard for the `ACTIVE_PHASE` thread-local. Install
/// at scenario-driver `run_step` entry; the guard's `Drop` restores
/// the prior phase label, supporting cleanly-nested scenario
/// dispatch (sub-scenarios layer over a parent's phase context
/// without leaking).
///
/// ```ignore
/// let _guard = PhaseGuard::install_step(0); // Step[0] → "Step[0]"
/// // ... apply_ops + hold, every assert constructed here stamps
/// //     phase = Some("Step[0]") automatically ...
/// // drop on scope exit restores the prior label (BASELINE outside
/// // any nested Step).
/// ```
#[must_use = "PhaseGuard restores the prior phase on Drop — bind it to a local"]
pub struct PhaseGuard {
    /// The phase label that was active before this guard installed.
    /// Restored on Drop so nested guards stack cleanly.
    previous: Option<std::borrow::Cow<'static, str>>,
}

impl PhaseGuard {
    /// Install `label` as the active phase. Captures the
    /// previously-active label for restoration on Drop. Use
    /// [`Self::install_step`] / [`Self::install_baseline`] for the
    /// scenario-driver call sites — they produce the standard
    /// `"Step[k]"` / `"BASELINE"` labels matching the rest of the
    /// pipeline.
    pub fn install(label: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        let previous = ACTIVE_PHASE.with(|p| p.replace(Some(label.into())));
        Self { previous }
    }

    /// Convenience: install the `"Step[k]"` label for the
    /// `zero_indexed`-th scenario Step. Matches the label
    /// [`PhaseBucket`] embeds + the [`Phase::step`] display
    /// (`Step[0]`, `Step[1]`, ...).
    pub fn install_step(zero_indexed: u16) -> Self {
        Self::install(format!("Step[{}]", zero_indexed))
    }

    /// Convenience: install the `"BASELINE"` label for the
    /// pre-first-Step settle window. Matches the label
    /// [`PhaseBucket`] uses for `step_index = 0`.
    pub fn install_baseline() -> Self {
        Self::install(std::borrow::Cow::Borrowed("BASELINE"))
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        ACTIVE_PHASE.with(|p| {
            *p.borrow_mut() = self.previous.take();
        });
    }
}

/// Per-VMA entry parsed from `/proc/self/numa_maps`.
#[derive(Debug, Clone, Default)]
pub struct NumaMapsEntry {
    /// Virtual address of the VMA.
    pub addr: u64,
    /// Per-node page counts (node_id -> page_count).
    pub node_pages: BTreeMap<usize, u64>,
}

/// Parse `/proc/self/numa_maps` content into per-VMA entries.
///
/// Each line has the format:
///   `<hex_addr> <policy> [key=val ...]`
/// where per-node page counts appear as `N<node>=<count>`.
pub fn parse_numa_maps(content: &str) -> Vec<NumaMapsEntry> {
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let addr = match parts.next().and_then(|s| u64::from_str_radix(s, 16).ok()) {
            Some(a) => a,
            None => continue,
        };
        // Skip policy field.
        let _ = parts.next();

        let mut entry = NumaMapsEntry {
            addr,
            ..Default::default()
        };

        for token in parts {
            if let Some(rest) = token.strip_prefix('N')
                && let Some((node_str, count_str)) = rest.split_once('=')
                && let (Ok(node), Ok(count)) = (node_str.parse::<usize>(), count_str.parse::<u64>())
            {
                *entry.node_pages.entry(node).or_insert(0) += count;
            }
        }

        if !entry.node_pages.is_empty() {
            entries.push(entry);
        }
    }
    entries
}

/// Compute page locality fraction from parsed numa_maps entries.
///
/// Returns the fraction of pages residing on any node in
/// `expected_nodes` (0.0-1.0). Returns 0.0 when no pages are observed
/// — a zero-allocation workload is not vacuously local; reporting 1.0
/// would let `min_page_locality` thresholds silently pass on broken
/// runs that produced no NUMA signal. The expected node set is the
/// cgroup's cpuset NUMA-node set (the nodes whose CPUs the worker is
/// confined to), supplied by the caller — see `assert_cgroup` in
/// `crate::assert::plan` / `crate::assert::reductions`, not the
/// worker's [`MemPolicy`](crate::workload::MemPolicy).
pub fn page_locality(entries: &[NumaMapsEntry], expected_nodes: &BTreeSet<usize>) -> f64 {
    let mut total: u64 = 0;
    let mut local: u64 = 0;
    for entry in entries {
        for (&node, &count) in &entry.node_pages {
            total += count;
            if expected_nodes.contains(&node) {
                local += count;
            }
        }
    }
    if total > 0 {
        local as f64 / total as f64
    } else {
        0.0
    }
}

/// Extract `numa_pages_migrated` from `/proc/vmstat` content.
///
/// Returns `None` if the counter is not present. The counter is
/// cumulative; callers diff pre- and post-workload snapshots to
/// get migration count during the test.
pub fn parse_vmstat_numa_pages_migrated(content: &str) -> Option<u64> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("numa_pages_migrated") {
            let rest = rest.trim();
            if let Ok(v) = rest.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

pub(crate) fn gap_threshold_ms() -> u64 {
    // Unoptimized debug builds have higher scheduling overhead.
    if cfg!(debug_assertions) { 3000 } else { 2000 }
}

pub(crate) fn spread_threshold_pct() -> f64 {
    // Debug builds in small VMs (especially under EEVDF) show higher
    // spread than optimized builds under sched_ext schedulers.
    if cfg!(debug_assertions) { 35.0 } else { 15.0 }
}
