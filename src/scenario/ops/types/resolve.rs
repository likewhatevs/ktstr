//! [`CpusetSpec`] → concrete [`BTreeSet<usize>`] resolution
//! (`validate` / `resolve` / `resolve_quiet` / `resolve_inner`).
//! Sibling to the construction impl block in [`super::op`]; this
//! file holds the topology-aware logic that maps a spec onto the
//! `Ctx`-observed usable-CPU set.

use std::collections::BTreeSet;

use crate::scenario::Ctx;

use super::CpusetSpec;

// ---------------------------------------------------------------------------
// CpusetSpec resolution
// ---------------------------------------------------------------------------

impl CpusetSpec {
    /// Check whether this spec can produce a non-empty cpuset for the
    /// given topology. Returns `Err` with a human-readable reason on
    /// failure.
    pub fn validate(&self, ctx: &Ctx) -> std::result::Result<(), String> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) if *idx >= ctx.topo.num_llcs() => Err(format!(
                "Llc({idx}) out of range: topology has {} LLCs",
                ctx.topo.num_llcs()
            )),
            CpusetSpec::Numa(node) if *node >= ctx.topo.num_numa_nodes() => Err(format!(
                "Numa({node}) out of range: topology has {} NUMA nodes",
                ctx.topo.num_numa_nodes()
            )),
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. } if *of == 0 => {
                Err("partition count (of) must be > 0".into())
            }
            CpusetSpec::Disjoint { index, of, .. } | CpusetSpec::Overlap { index, of, .. }
                if *index >= *of =>
            {
                Err(format!("index {index} >= partition count {of}"))
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if !start_frac.is_finite() || !end_frac.is_finite() => Err(format!(
                "Range start_frac ({start_frac}) or end_frac ({end_frac}) is not finite"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if *start_frac < 0.0 || *end_frac > 1.0 => Err(format!(
                "Range fracs must lie in [0.0, 1.0]: start_frac={start_frac}, end_frac={end_frac}"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if start_frac >= end_frac => Err(format!(
                "Range start_frac ({start_frac}) >= end_frac ({end_frac})"
            )),
            CpusetSpec::Overlap { frac, .. } if !frac.is_finite() => {
                Err(format!("Overlap frac ({frac}) is not finite"))
            }
            CpusetSpec::Overlap { frac, .. } if *frac < 0.0 || *frac > 1.0 => {
                Err(format!("Overlap frac ({frac}) must lie in [0.0, 1.0]"))
            }
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. }
                if usable.len() < *of =>
            {
                Err(format!(
                    "not enough usable CPUs ({}) for {} partitions",
                    usable.len(),
                    of
                ))
            }
            CpusetSpec::Exact(cpus) if cpus.is_empty() => {
                Err("CpusetSpec::Exact(empty) would assign no CPUs to the \
                 cgroup; cpuset.cpus rejects an empty mask and the \
                 cgroup would become unschedulable"
                    .into())
            }
            CpusetSpec::Exact(cpus) => {
                // Reject only CPUs the topology doesn't physically have
                // (`all_cpuset`), not the ones outside `usable_cpuset`.
                // A scheduler author may intentionally pin to an
                // isolated CPU (e.g. the root-reserved one) for
                // testing; writing it to cpuset.cpus is a legitimate
                // operation and the kernel is the final authority on
                // whether the write succeeds. Only truly-nonexistent
                // CPU indices are guaranteed to produce EINVAL.
                let all = ctx.topo.all_cpuset();
                let missing: Vec<usize> =
                    cpus.iter().copied().filter(|c| !all.contains(c)).collect();
                if !missing.is_empty() {
                    return Err(format!(
                        "CpusetSpec::Exact contains CPU(s) {missing:?} \
                         outside the topology's physical CPU set (max \
                         CPU index: {}); writing them to cpuset.cpus \
                         would fail with EINVAL",
                        all.iter().next_back().copied().unwrap_or(0),
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Resolve to a concrete CPU set given the topology.
    ///
    /// **Callers SHOULD run [`Self::validate`] first and propagate
    /// its error.** `apply_setup` and `apply_ops::SetCpuset` do so
    /// via `anyhow::bail!`, then call [`Self::resolve_quiet`] which
    /// skips the warns this method emits on degenerate inputs.
    ///
    /// Defense-in-depth: every malformed input that `validate`
    /// rejects (out-of-range `Llc`/`Numa`, partition `of == 0`,
    /// `index >= of`, inverted or non-finite `Range.start_frac` /
    /// `end_frac`, out-of-bounds `Overlap.frac`) also has a
    /// panic-free fallback here — out-of-range indices clamp to the
    /// last valid index with a `tracing::warn!`, `of == 0` returns
    /// an empty set with a warn, and inverted/non-finite fracs
    /// clamp to `[0, len]` so the resulting slice never inverts.
    /// Skipping `validate` therefore degrades into a usable
    /// (possibly empty) cpuset rather than crashing the caller, but
    /// the warns surface the silent-degradation case — a caller who
    /// computed a CPU count via [`crate::scenario::Ctx::cpuset_cpus`]
    /// (which doesn't validate) sees the warn instead of silently
    /// planning against the wrong denominator.
    pub fn resolve(&self, ctx: &Ctx) -> BTreeSet<usize> {
        self.resolve_inner(ctx, false)
    }

    /// Like [`Self::resolve`] but suppresses the degenerate-input
    /// `tracing::warn!`s. Use this from call sites that pair the
    /// resolution with a [`Self::validate`] call (either before or
    /// after this one) and bail on its error — validate is the
    /// canonical error channel for malformed specs, and a warn
    /// here would be redundant noise on a path already known-
    /// broken via the validate gate. `apply_setup` resolves first
    /// (to keep the workers_pct empty-cpuset diagnostic ahead of
    /// validate's generic empty-Exact rejection) and validates
    /// after; `Op::SetCpuset` validates first and resolves after.
    /// Both patterns satisfy the contract.
    pub fn resolve_quiet(&self, ctx: &Ctx) -> BTreeSet<usize> {
        self.resolve_inner(ctx, true)
    }

    fn resolve_inner(&self, ctx: &Ctx, quiet: bool) -> BTreeSet<usize> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) => {
                if *idx >= ctx.topo.num_llcs() {
                    // Graceful fallback: clamp to last LLC instead of panicking.
                    let clamped = ctx.topo.num_llcs().saturating_sub(1);
                    if !quiet {
                        tracing::warn!(
                            llc_idx = idx,
                            num_llcs = ctx.topo.num_llcs(),
                            clamped,
                            "CpusetSpec::Llc index out of range, clamping",
                        );
                    }
                    ctx.topo.llc_aligned_cpuset(clamped)
                } else {
                    ctx.topo.llc_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Numa(idx) => {
                if *idx >= ctx.topo.num_numa_nodes() {
                    let clamped = ctx.topo.num_numa_nodes().saturating_sub(1);
                    if !quiet {
                        tracing::warn!(
                            numa_node = idx,
                            num_numa_nodes = ctx.topo.num_numa_nodes(),
                            clamped,
                            "CpusetSpec::Numa index out of range, clamping",
                        );
                    }
                    ctx.topo.numa_aligned_cpuset(clamped)
                } else {
                    ctx.topo.numa_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } => {
                let len = usable.len();
                // Defense-in-depth: clamp non-finite fracs to 0 (NaN
                // would saturate to 0 via `as usize` anyway; explicit
                // check matches validate's rejection reason).
                let sf = if start_frac.is_finite() {
                    *start_frac
                } else {
                    0.0
                };
                let ef = if end_frac.is_finite() { *end_frac } else { 0.0 };
                let start = (len as f64 * sf) as usize;
                let end = (len as f64 * ef) as usize;
                // Guard against inverted Range (start_frac > end_frac)
                // — `&usable[start..end]` panics when start > end even
                // if both are clamped to `len`. `e = end.min(len).max(s)`
                // clamps `e` up to `s`, so when start > end the slice is
                // empty (`usable[s..s]`) instead of panicking.
                let s = start.min(len);
                let e = end.min(len).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Disjoint { index, of } => {
                if *of == 0 {
                    // Defense-in-depth: `validate` rejects of==0 with a
                    // clear error. If a caller reaches `resolve` with
                    // of==0 anyway (skipped validate, or used a
                    // malformed programmatic spec), returning an empty
                    // set is safer than the div-by-zero panic.
                    if !quiet {
                        tracing::warn!("CpusetSpec::Disjoint with of=0 — returning empty cpuset");
                    }
                    return BTreeSet::new();
                }
                let chunk = usable.len() / of;
                let start = index * chunk;
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    (index + 1) * chunk
                };
                let s = start.min(usable.len());
                let e = end.min(usable.len()).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Overlap { index, of, frac } => {
                if *of == 0 {
                    if !quiet {
                        tracing::warn!("CpusetSpec::Overlap with of=0 — returning empty cpuset");
                    }
                    return BTreeSet::new();
                }
                let chunk = usable.len() / of;
                // Clamp finite frac to [0.0, 1.0]; map non-finite frac
                // to 0.0, so the overlap computation stays bounded.
                let frac = if frac.is_finite() {
                    frac.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let overlap = (chunk as f64 * frac) as usize;
                let start = if *index == 0 {
                    0
                } else {
                    (index * chunk).saturating_sub(overlap)
                };
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    ((index + 1) * chunk + overlap).min(usable.len())
                };
                let s = start.min(usable.len());
                let e = end.min(usable.len()).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Exact(cpus) => cpus.clone(),
        }
    }
}
