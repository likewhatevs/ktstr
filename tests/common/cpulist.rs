//! Linux cpulist parsing shared by the topology integration tests
//! (`topology_fat_llc`, `wide_smp_boot_e2e`). `#[path]`-included by each so
//! only this helper is pulled in, not the whole `common` tree.

/// Count the CPUs named by a Linux cpulist string ("0-255", "0,2,4", "0-3,8").
pub fn count_cpulist(s: &str) -> usize {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| match p.trim().split_once('-') {
            Some((a, b)) => {
                let a: usize = a.trim().parse().unwrap_or(0);
                let b: usize = b.trim().parse().unwrap_or(0);
                b.saturating_sub(a) + 1
            }
            None => 1,
        })
        .sum()
}
