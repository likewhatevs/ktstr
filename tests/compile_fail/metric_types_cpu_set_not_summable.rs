//! `CpuSet` is a `Vec<u32>` of CPU IDs — affinity, not a
//! cumulative counter. Summing two affinity sets is undefined
//! (there is no cross-thread sum; the affinity reduction is
//! `AffinitySummary` in `ctprof_compare`, which reports a
//! min/max num_cpus range plus a uniform-cpuset flag, not a
//! Summable trait method). Pin the type-system rejection: a
//! generic site bound on `T: Summable` must refuse `CpuSet`.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::CpuSet>();
}
