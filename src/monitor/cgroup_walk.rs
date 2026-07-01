//! Host-side cgroup-hierarchy walk for the per-cgroup PSI-irq axis (Phase A).
//!
//! Locates the test's workload cgroups in guest memory and reads each one's
//! `struct psi_group` PSI_IRQ_FULL, host-side (no guest helper), at a freeze.
//! The walk:
//!
//! 1. `cgrp_dfl_root` (the cgroup2 default-hierarchy root, a `.data` global) →
//!    `+ offsetof(cgroup_root, cgrp)` = the root `struct cgroup`. The global is
//!    kernel-image-mapped, so the CALLER pre-translates the root's PA + runtime
//!    KVA (via `text_kva_to_pa` + the KASLR slide) and passes them in; EVERY hop
//!    below is a `kzalloc`'d slab object in the direct map, so [`kva_to_pa`].
//! 2. Descend the host-held workload-root PATH (default `/sys/fs/cgroup/ktstr`)
//!    by matching each child's `kn->name`, filtering the walk to that subtree.
//!    PRECONDITION: the scheduler cgroup must not be nested UNDER the workload
//!    root. By default it is not — the scheduler has no `cgroup_parent` and
//!    inherits the root cgroup (outside the workload-root subtree), so its
//!    IRQ-servicing pressure cannot confound the per-cgroup axis. A config that
//!    places the scheduler's cgroup strictly under the workload root would
//!    surface it as a spurious leaf; the freeze coordinator warns on that case.
//! 3. From the workload root, DFS the subtree and collect LEAF cgroups (empty
//!    `self.children`) ONLY. cgroup2's no-internal-process rule puts tasks in
//!    leaves, so leaf cgroups carry the DISJOINT per-cgroup IRQ stall — the
//!    sound basis for the cross-cgroup max/concentration fold. CONTRACT:
//!    only leaf DESCENDANTS are captured; tasks placed directly in the workload
//!    root (or any non-leaf) are not measured — ktstr always runs workers in
//!    `CgroupDef` leaf children. A non-root cgroup's `psi` pointer is NULL when
//!    `psi_cgroups_enabled` is off → that leaf is skipped (loud-absent).
//!
//! HOSTILE-INPUT / corrupt-memory defense (the [`super::scx_walker`] list-walk
//! contract): every `list_head` walk is bounded by a node cap AND a visited-set
//! cycle guard, the whole DFS by a global node cap, the `kn->name` read by a
//! length cap, and every guest-pointer deref tolerates a bad pointer (an OOB PA
//! reads 0 via [`GuestMem`], terminating the walk) rather than faulting or
//! looping. The freeze rendezvous parks all vCPUs, so the tree is not mutated
//! during the read; the guards are defense against a corrupt/hostile guest, not
//! a concurrency race.
//!
//! RAW values are captured ([`CgroupPsiStat`]); decode (`/2048` percent, `/1000`
//! µs) and the cross-cgroup fold happen at the metric layer, mirroring how
//! [`super::PsiIrqSample`] defers decode to the summary fold.

use std::collections::HashSet;

use super::btf_offsets::{CgroupWalkOffsets, PsiGroupOffsets};
use super::reader::GuestMem;
use super::symbols::kva_to_pa;

/// Global node-visit budget for the whole walk (descent + subtree DFS): a
/// single shared counter, decremented per child read in [`children_of`], that
/// bounds TOTAL guest reads AND the DFS worklist to O(this). This is NOT the
/// per-list cap [`super::scx_walker`]'s `MAX_NODES_PER_LIST` is — that caps one
/// flat list; a tree walk with an independent node-cap × per-list-cap would
/// MULTIPLY (a hostile wide+deep tree → caps² reads + a caps²-entry stack). One
/// shared budget collapses the worst case to O(this). A real ktstr test has a
/// handful of workload cgroups; 1024 is far above any real subtree and bounds a
/// corrupt/hostile/cyclic hierarchy. Exhaustion is logged (loud truncation,
/// never a silent drop).
const MAX_CGROUPS_WALKED: u32 = 1024;

/// Cap on the `kn->name` C-string read. kernfs directory names are short
/// (`NAME_MAX` is 255); bound the read against a corrupt `name` pointer.
const MAX_CGROUP_NAME_LEN: usize = 256;

/// The cgroup2 mount point the workload-root path is rooted at (`rust_init`
/// mounts cgroup2 here). Stripped from the host-held workload-root path to
/// derive the cgroup-relative segments matched against `kn->name`.
const CGROUP2_MOUNT_PREFIX: &str = "/sys/fs/cgroup";

/// One test cgroup's raw PSI-irq read at a freeze. RAW kernel values, decoded
/// at the metric-layer fold via [`super::btf_offsets::decode_avg10_percent`] /
/// [`super::btf_offsets::decode_total_us`] — same raw-then-decode split as
/// [`super::PsiIrqSample`]. Keyed by `cgroup_kva` (the cgroup's address, a
/// stable-for-lifetime identity) so the per-phase fold can correlate the
/// same cgroup across freezes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CgroupPsiStat {
    /// Kernel virtual address of this leaf's `struct cgroup`. Stable for a
    /// single cgroup's lifetime (not reallocated until its deferred post-rmdir
    /// CSS free), but NOT a sufficient cross-freeze identity ON ITS OWN: a freed
    /// `struct cgroup` is a plain `kzalloc` whose slab KVA a later cgroup can
    /// reuse, so the cross-freeze fold pairs this with `serial_nr` to reject a
    /// same-KVA-different-cgroup match. NOT a PA to re-read: the walk re-derives
    /// PAs each freeze (a cached PA could read a freed slab); this is only an
    /// identity for cross-freeze correlation.
    pub cgroup_kva: u64,
    /// Raw `cgroup->psi->total[PSI_AVGS][PSI_IRQ_FULL]` — cumulative IRQ stall
    /// ns (decode `/1000` = µs).
    pub total_ns: u64,
    /// Raw `cgroup->psi->avg[PSI_IRQ_FULL][0]` — the 10s EWMA in fixed-point
    /// (percent × 2048; decode `/2048`, clamp `[0,100]`).
    pub avg10_raw: u64,
    /// Raw `cgroup_subsys_state.serial_nr` — the monotonic per-creation serial
    /// (`css_serial_nr_next++` at `cgroup_create`). Paired with `cgroup_kva` as
    /// the cross-freeze identity: a freed cgroup's slab KVA can be reused by a
    /// later cgroup, so a KVA match with a DIFFERENT `serial_nr` is a different
    /// cgroup — the fold drops it rather than computing a bogus cross-cgroup
    /// counter delta.
    pub serial_nr: u64,
}

/// Split a host-held workload-root path into the cgroup-relative segments to
/// match against `kn->name`. Accepts BOTH shapes the two callers produce: the
/// default is the full mount-prefixed path (`/sys/fs/cgroup/ktstr`, from the
/// freeze coordinator) — [`CGROUP2_MOUNT_PREFIX`] is stripped; a user-set
/// `workload_root_cgroup` arrives already cgroup-relative (`/ktstr`) and is used
/// as-is (the strip is a no-op). Either way leading/trailing/internal empty
/// components are dropped, so `/sys/fs/cgroup/ktstr`, `/ktstr`, and `ktstr` all
/// → `["ktstr"]`. (The cgroup2 hierarchy root corresponds to the mount point,
/// whose `kn->name` is `/`, never a matched segment.)
fn workload_root_segments(path: &str) -> Vec<&str> {
    let rel = path.strip_prefix(CGROUP2_MOUNT_PREFIX).unwrap_or(path);
    rel.split('/').filter(|s| !s.is_empty()).collect()
}

/// Iterate a cgroup's children, returning each child cgroup's KVA. `parent_pa`
/// is the parent cgroup's already-translated PA (text-mapped for the root,
/// direct-mapped for any heap cgroup); `parent_kva` is its KVA (for the
/// terminate-at-anchor comparison). The children list is anchored at
/// `parent.self.children`; each child links via its `self.sibling`, so a child
/// cgroup is `sibling_node - offsetof(css, sibling) - offsetof(cgroup, self)`.
///
/// `budget` is the shared global node-visit budget ([`MAX_CGROUPS_WALKED`]):
/// each child read decrements it, and the walk stops when it reaches 0. Threaded
/// through every `children_of` call across the descent AND the DFS, it bounds
/// TOTAL guest reads — and therefore the caller's DFS stack — to O(budget)
/// across the whole walk, NOT O(nodes × children). A per-call visited-set guards
/// a cyclic sibling list; a 0/bad `next` pointer terminates (tolerate-bad-pointer).
fn children_of(
    budget: &mut u32,
    mem: &GuestMem,
    off: &CgroupWalkOffsets,
    page_offset: u64,
    parent_kva: u64,
    parent_pa: u64,
) -> Vec<u64> {
    // The children list_head sits at `self.children` within the cgroup.
    let children_field = off.cgroup_self + off.css_children;
    let anchor_kva = parent_kva.wrapping_add(children_field as u64);
    let mut out = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    // First node = anchor.next (list_head.next is the first field, offset 0).
    let mut node = mem.read_u64(parent_pa, children_field);
    while node != anchor_kva {
        if node == 0 || *budget == 0 || !seen.insert(node) {
            // 0 = unmapped/torn; budget = global node-visit budget exhausted
            // (the caller logs the truncation); revisit = cyclic sibling list.
            // Stop rather than fault/loop.
            break;
        }
        *budget -= 1;
        // child css = sibling_node - offsetof(css, sibling); child cgroup =
        // child css - offsetof(cgroup, self).
        let child_cgroup = node
            .wrapping_sub(off.css_sibling as u64)
            .wrapping_sub(off.cgroup_self as u64);
        out.push(child_cgroup);
        // Advance: next = *node (list_head.next at offset 0). The node is a
        // heap sibling list_head → direct-map.
        let node_pa = kva_to_pa(node, page_offset);
        node = mem.read_u64(node_pa, 0);
    }
    out
}

/// True iff `cgroup` is a leaf: its `self.children` list is empty, i.e. the
/// list_head's `next` points back at the anchor's own KVA.
fn is_leaf(off: &CgroupWalkOffsets, cgroup_kva: u64, cgroup_pa: u64, mem: &GuestMem) -> bool {
    let children_field = off.cgroup_self + off.css_children;
    let anchor_kva = cgroup_kva.wrapping_add(children_field as u64);
    mem.read_u64(cgroup_pa, children_field) == anchor_kva
}

/// Read a cgroup's directory name from `cgroup->kn->name` (bounded). `None`
/// when `kn` or `name` is NULL or the string is not valid UTF-8 — a cgroup
/// whose name can't be read can't match a path segment (it is skipped, never
/// faulted on).
fn read_cgroup_name(
    mem: &GuestMem,
    off: &CgroupWalkOffsets,
    page_offset: u64,
    cgroup_pa: u64,
) -> Option<String> {
    let kn_kva = mem.read_u64(cgroup_pa, off.cgroup_kn);
    if kn_kva == 0 {
        return None;
    }
    let kn_pa = kva_to_pa(kn_kva, page_offset);
    let name_kva = mem.read_u64(kn_pa, off.kernfs_node_name);
    if name_kva == 0 {
        return None;
    }
    let name_pa = kva_to_pa(name_kva, page_offset);
    let mut buf = [0u8; MAX_CGROUP_NAME_LEN];
    let n = mem.read_bytes(name_pa, &mut buf);
    if n == 0 {
        return None;
    }
    // NUL-terminated C string, bounded to the bytes actually read.
    let end = buf[..n].iter().position(|&b| b == 0).unwrap_or(n);
    std::str::from_utf8(&buf[..end]).ok().map(str::to_owned)
}

/// Read a non-root cgroup's PSI_IRQ_FULL via its `psi` pointer. `None` when
/// `cgroup->psi` is NULL (`psi_cgroups_enabled` off → this cgroup has no
/// per-cgroup PSI accounting; loud-absent) or the `PSI_IRQ_FULL` offsets are
/// config-absent.
fn read_cgroup_psi(
    mem: &GuestMem,
    off: &CgroupWalkOffsets,
    psi_off: &PsiGroupOffsets,
    page_offset: u64,
    cgroup_kva: u64,
    cgroup_pa: u64,
) -> Option<CgroupPsiStat> {
    let psi_kva = mem.read_u64(cgroup_pa, off.cgroup_psi);
    if psi_kva == 0 {
        return None;
    }
    let total_off = psi_off.total_irq_full_off()?;
    let avg10_off = psi_off.avg10_irq_full_off()?;
    let psi_pa = kva_to_pa(psi_kva, page_offset);
    // serial_nr lives in the embedded `self` css (at cgroup_self), read from
    // the cgroup object (not the psi_group): cgroup_pa + cgroup_self + css_serial_nr.
    Some(CgroupPsiStat {
        cgroup_kva,
        total_ns: mem.read_u64(psi_pa, total_off),
        avg10_raw: mem.read_u64(psi_pa, avg10_off),
        serial_nr: mem.read_u64(cgroup_pa, off.cgroup_self + off.css_serial_nr),
    })
}

/// Walk the test's workload cgroups and read each leaf's PSI_IRQ_FULL.
///
/// Returns the per-leaf raw PSI samples under the host-held `workload_root_path`
/// subtree (empty when the workload root isn't present yet — e.g. a freeze
/// before the scenario created its cgroups — or has no leaves, the loud-absent
/// path). The caller pre-translates the hierarchy root cgroup: `root_cgroup_kva`
/// is its RUNTIME KVA (carrying any virtual-KASLR slide, for the children-list
/// anchor compare) and `root_cgroup_pa` its guest physical address (the root is
/// embedded in the kernel-image-mapped `cgrp_dfl_root` global). Every descendant
/// is a direct-mapped slab object, translated via `kva_to_pa(_, page_offset)`.
#[allow(clippy::too_many_arguments)]
pub fn collect_workload_cgroup_psi(
    mem: &GuestMem,
    off: &CgroupWalkOffsets,
    psi_off: &PsiGroupOffsets,
    root_cgroup_kva: u64,
    root_cgroup_pa: u64,
    workload_root_path: &str,
    page_offset: u64,
) -> Vec<CgroupPsiStat> {
    // One shared global node-visit budget bounds TOTAL guest reads AND the DFS
    // worklist to O(MAX_CGROUPS_WALKED) across the descent AND the subtree walk,
    // so a hostile wide+deep tree cannot multiply (node-cap × child-cap) reads
    // or balloon the stack. children_of decrements it per child read.
    let mut budget: u32 = MAX_CGROUPS_WALKED;

    // Descend the workload-root path by name from the pre-translated root
    // cgroup. The root's own kn->name is "/" (the mount point), so matching
    // begins at its children.
    let mut cur_kva = root_cgroup_kva;
    let mut cur_pa = root_cgroup_pa;
    for segment in workload_root_segments(workload_root_path) {
        let Some((child_kva, child_pa)) =
            children_of(&mut budget, mem, off, page_offset, cur_kva, cur_pa)
                .into_iter()
                .map(|c| (c, kva_to_pa(c, page_offset)))
                .find(|&(_, pa)| {
                    read_cgroup_name(mem, off, page_offset, pa).as_deref() == Some(segment)
                })
        else {
            // Workload root (or an intermediate segment) not found — no
            // workload cgroups to attribute yet. Loud-absent.
            return Vec::new();
        };
        cur_kva = child_kva;
        cur_pa = child_pa;
    }

    // DFS the workload-root subtree, collecting leaf descendants' PSI. The
    // workload root itself is the container (not a leaf candidate) — start from
    // its children so an empty workload root yields no samples. `visited` dedups
    // a node reached via multiple parents (a corrupt DAG); the budget (threaded
    // through children_of) bounds both total reads and the stack size.
    let mut out = Vec::new();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut stack: Vec<(u64, u64)> =
        children_of(&mut budget, mem, off, page_offset, cur_kva, cur_pa)
            .into_iter()
            .map(|c| (c, kva_to_pa(c, page_offset)))
            .collect();
    while let Some((kva, pa)) = stack.pop() {
        if !visited.insert(kva) {
            continue; // already processed (cycle / DAG dedup)
        }
        // is_leaf is a SEPARATE single read, deliberately NOT folded into
        // children_of: an empty children_of result can mean "leaf" OR "budget
        // exhausted", and conflating them would misclassify a budget-truncated
        // non-leaf as a leaf and read that non-leaf container's OWN PSI (the DFS
        // reads the popped node's own psi pointer), leaving its real leaf
        // descendants uncaptured. is_leaf is bounded by
        // the budget-bounded stack (one read per popped node).
        if is_leaf(off, kva, pa, mem) {
            if let Some(stat) = read_cgroup_psi(mem, off, psi_off, page_offset, kva, pa) {
                out.push(stat);
            }
        } else {
            for (child_kva, child_pa) in children_of(&mut budget, mem, off, page_offset, kva, pa)
                .into_iter()
                .map(|c| (c, kva_to_pa(c, page_offset)))
            {
                stack.push((child_kva, child_pa));
            }
        }
    }
    if budget == 0 {
        // The node-visit budget cap was fully consumed. This never fires for a
        // real ktstr test (a handful of cgroups); it signals a corrupt/hostile
        // hierarchy or a subtree at/above the cap. If the subtree EXCEEDS the
        // cap, its tail (some leaves' per-cgroup PSI) was not captured — a loud
        // signal, never a silent drop. (A subtree of exactly the cap size is
        // walked in full; the budget alone can't distinguish that boundary, but
        // neither case occurs for a real test.)
        tracing::warn!(
            cap = MAX_CGROUPS_WALKED,
            "collect_workload_cgroup_psi: node-visit budget cap reached — if the \
             workload cgroup subtree exceeds the cap, its tail (some leaves' \
             per-cgroup PSI) was not captured"
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    // Offsets for a synthetic cgroup layout. `self` at 0 (css embedded first);
    // within the css, sibling then children list_heads then the serial_nr; kn
    // and psi pointers after the css. cgroup_root.cgrp embeds the root cgroup at
    // a nonzero offset. kernfs_node.name pointer at a fixed offset.
    const SELF_OFF: usize = 0;
    const CSS_SIBLING: usize = 16;
    const CSS_CHILDREN: usize = 32;
    const CSS_SERIAL_NR: usize = 48; // serial_nr within the css, after the list_heads
    const CGROUP_KN: usize = 64;
    const CGROUP_PSI: usize = 72;
    const ROOT_CGRP: usize = 256; // cgrp embedded in cgroup_root at +256
    const KN_NAME: usize = 8; // kernfs_node.name pointer offset

    fn offs() -> CgroupWalkOffsets {
        CgroupWalkOffsets {
            cgroup_self: SELF_OFF,
            cgroup_kn: CGROUP_KN,
            cgroup_psi: CGROUP_PSI,
            css_sibling: CSS_SIBLING,
            css_children: CSS_CHILDREN,
            css_serial_nr: CSS_SERIAL_NR,
            cgroup_root_cgrp: ROOT_CGRP,
            kernfs_node_name: KN_NAME,
        }
    }

    fn psi_offs() -> PsiGroupOffsets {
        // total at 0, avg at 512, PSI_IRQ_FULL idx 6 — derived offsets land
        // well inside the synthetic psi_group object below.
        PsiGroupOffsets {
            psi_group_total: 0,
            psi_group_avg: 512,
            psi_irq_full_idx: Some(6),
        }
    }

    /// A flat guest-memory image with a uniform translation: both the root
    /// (text) and heap resolve to `addr - DEFAULT_PAGE_OFFSET` here because the
    /// test passes `root_cgroup_pa = root` directly (the caller does the real
    /// text-vs-direct-map translation). This focuses the unit tests on the WALK
    /// LOGIC (linkage, leaf detection, name match, psi read, budget bound); the
    /// two real translation modes (text-mapped root vs direct-mapped
    /// descendants) are an e2e concern — the metric e2e boots a VM and will
    /// exercise them against a live kernel.
    struct Image {
        buf: Vec<u8>,
    }
    impl Image {
        fn new(size: usize) -> Self {
            Self {
                buf: vec![0u8; size],
            }
        }
        fn kva(pa: usize) -> u64 {
            DEFAULT_PAGE_OFFSET + pa as u64
        }
        fn w64(&mut self, pa: usize, v: u64) {
            self.buf[pa..pa + 8].copy_from_slice(&v.to_le_bytes());
        }
        fn wstr(&mut self, pa: usize, s: &str) {
            self.buf[pa..pa + s.len()].copy_from_slice(s.as_bytes());
            self.buf[pa + s.len()] = 0;
        }
        /// Initialize a cgroup at `pa` as an empty leaf (children list points
        /// at itself) with the given kn->name and psi pointer.
        fn init_cgroup(&mut self, pa: usize, kn_pa: usize, psi_kva: u64) {
            // empty children list: next = &children (self-pointing).
            let children_kva = Self::kva(pa + SELF_OFF + CSS_CHILDREN);
            self.w64(pa + SELF_OFF + CSS_CHILDREN, children_kva);
            self.w64(pa + CGROUP_KN, Self::kva(kn_pa));
            self.w64(pa + CGROUP_PSI, psi_kva);
        }
        /// Link `children` (cgroup PAs) into `parent`'s children list.
        fn link_children(&mut self, parent_pa: usize, children: &[usize]) {
            let anchor = Self::kva(parent_pa + SELF_OFF + CSS_CHILDREN);
            let mut prev_next_field = parent_pa + SELF_OFF + CSS_CHILDREN; // anchor.next
            for &c in children {
                let sibling_node = Self::kva(c + SELF_OFF + CSS_SIBLING);
                self.w64(prev_next_field, sibling_node);
                prev_next_field = c + SELF_OFF + CSS_SIBLING; // child.sibling.next
            }
            // Close the ring back to the anchor.
            self.w64(prev_next_field, anchor);
        }
    }

    /// Lay out root → "ktstr" → {cg_0, cg_1} leaves (each with a psi_group) +
    /// a sibling "sched" subtree the walk must EXCLUDE. Assert the walk returns
    /// exactly the two workload leaves' psi.
    #[test]
    fn walk_collects_workload_leaves_excludes_scheduler() {
        let mut img = Image::new(0x8000);
        // psi_group objects: total@0, avg@512, so size > 512 + 7*8.
        let psi_cg0 = 0x4000usize;
        let psi_cg1 = 0x4800usize;
        // total[PSI_AVGS][PSI_IRQ_FULL] at total + 6*8 = 48; avg[6][0] at avg + 6*3*8 = 512+144.
        img.w64(psi_cg0 + 6 * 8, 111_000); // cg0 total_ns
        img.w64(psi_cg0 + 512 + 6 * 3 * 8, 2048); // cg0 avg10_raw (= 1.0%)
        img.w64(psi_cg1 + 6 * 8, 222_000); // cg1 total_ns
        img.w64(psi_cg1 + 512 + 6 * 3 * 8, 4096); // cg1 avg10_raw

        // kernfs name strings.
        let kn_root = 0x100usize;
        let name_ktstr = 0x140usize;
        let kn_ktstr = 0x180usize;
        let name_sched = 0x1c0usize;
        let kn_sched = 0x200usize;
        let name_cg0 = 0x240usize;
        let kn_cg0 = 0x280usize;
        let name_cg1 = 0x2c0usize;
        let kn_cg1 = 0x300usize;
        img.wstr(name_ktstr, "ktstr");
        img.wstr(name_sched, "sched");
        img.wstr(name_cg0, "cg_0");
        img.wstr(name_cg1, "cg_1");
        // kernfs_node.name pointers.
        img.w64(kn_root + KN_NAME, Image::kva(0x130)); // root name "/" (unused)
        img.wstr(0x130, "/");
        img.w64(kn_ktstr + KN_NAME, Image::kva(name_ktstr));
        img.w64(kn_sched + KN_NAME, Image::kva(name_sched));
        img.w64(kn_cg0 + KN_NAME, Image::kva(name_cg0));
        img.w64(kn_cg1 + KN_NAME, Image::kva(name_cg1));

        // cgroups.
        let cgrp_dfl_root = 0x1000usize; // cgroup_root; cgrp embedded at +256.
        let root = cgrp_dfl_root + ROOT_CGRP; // root cgroup
        let ktstr = 0x2000usize;
        let sched = 0x2400usize;
        let cg0 = 0x2800usize;
        let cg1 = 0x2c00usize;
        // root + the two top-level subtrees are non-leaf; init as leaves first
        // then link children (link overwrites the children.next).
        for (cg, kn) in [(root, kn_root), (ktstr, kn_ktstr), (sched, kn_sched)] {
            img.init_cgroup(cg, kn, 0); // intermediate: psi unused
        }
        img.init_cgroup(cg0, kn_cg0, Image::kva(psi_cg0));
        img.init_cgroup(cg1, kn_cg1, Image::kva(psi_cg1));
        // Distinct per-leaf creation serials (read from the embedded css at
        // SELF_OFF + CSS_SERIAL_NR) — pins the walk reads serial_nr.
        img.w64(cg0 + SELF_OFF + CSS_SERIAL_NR, 7);
        img.w64(cg1 + SELF_OFF + CSS_SERIAL_NR, 9);
        img.link_children(root, &[ktstr, sched]);
        img.link_children(ktstr, &[cg0, cg1]);
        // sched has its own leaf the walk must NOT reach.
        let sched_leaf = 0x3000usize;
        let kn_sl = 0x340usize;
        let name_sl = 0x380usize;
        img.wstr(name_sl, "scx");
        img.w64(kn_sl + KN_NAME, Image::kva(name_sl));
        let psi_sl = 0x5000usize;
        img.w64(psi_sl + 6 * 8, 999_000);
        img.init_cgroup(sched_leaf, kn_sl, Image::kva(psi_sl));
        img.link_children(sched, &[sched_leaf]);

        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        let mut got = collect_workload_cgroup_psi(
            &mem,
            &offs(),
            &psi_offs(),
            Image::kva(root),
            root as u64, // uniform xlate in-test: kva_to_pa(Image::kva(root)) == root
            "/sys/fs/cgroup/ktstr",
            DEFAULT_PAGE_OFFSET,
        );
        got.sort_by_key(|s| s.total_ns);
        assert_eq!(
            got.len(),
            2,
            "exactly the two workload leaves (sched excluded)"
        );
        assert_eq!(got[0].total_ns, 111_000);
        assert_eq!(got[0].avg10_raw, 2048);
        assert_eq!(got[0].cgroup_kva, Image::kva(cg0));
        assert_eq!(
            got[0].serial_nr, 7,
            "cg0 serial_nr read from the embedded css"
        );
        assert_eq!(got[1].total_ns, 222_000);
        assert_eq!(got[1].cgroup_kva, Image::kva(cg1));
        assert_eq!(
            got[1].serial_nr, 9,
            "cg1 serial_nr read from the embedded css"
        );
    }

    /// A leaf whose cgroup->psi is NULL (psi_cgroups disabled) is skipped, not
    /// captured as a 0 — the loud-absent contract.
    #[test]
    fn walk_skips_null_psi_leaf() {
        let mut img = Image::new(0x4000);
        let kn_root = 0x100usize;
        let kn_ktstr = 0x180usize;
        let kn_cg0 = 0x280usize;
        img.wstr(0x140, "ktstr");
        img.wstr(0x240, "cg_0");
        img.w64(kn_root + KN_NAME, Image::kva(0x130));
        img.wstr(0x130, "/");
        img.w64(kn_ktstr + KN_NAME, Image::kva(0x140));
        img.w64(kn_cg0 + KN_NAME, Image::kva(0x240));
        let cgrp_dfl_root = 0x1000usize;
        let root = cgrp_dfl_root + ROOT_CGRP;
        let ktstr = 0x2000usize;
        let cg0 = 0x2800usize;
        img.init_cgroup(root, kn_root, 0);
        img.init_cgroup(ktstr, kn_ktstr, 0);
        img.init_cgroup(cg0, kn_cg0, 0); // psi = NULL
        img.link_children(root, &[ktstr]);
        img.link_children(ktstr, &[cg0]);
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        let got = collect_workload_cgroup_psi(
            &mem,
            &offs(),
            &psi_offs(),
            Image::kva(root),
            root as u64,
            "/sys/fs/cgroup/ktstr",
            DEFAULT_PAGE_OFFSET,
        );
        assert!(
            got.is_empty(),
            "NULL-psi leaf is skipped (loud-absent), not a 0"
        );
    }

    /// Workload root absent (not yet created) → empty, no fault.
    #[test]
    fn walk_absent_workload_root_is_empty() {
        let mut img = Image::new(0x4000);
        let kn_root = 0x100usize;
        img.w64(kn_root + KN_NAME, Image::kva(0x130));
        img.wstr(0x130, "/");
        let cgrp_dfl_root = 0x1000usize;
        let root = cgrp_dfl_root + ROOT_CGRP;
        img.init_cgroup(root, kn_root, 0); // root with no children
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        let got = collect_workload_cgroup_psi(
            &mem,
            &offs(),
            &psi_offs(),
            Image::kva(root),
            root as u64,
            "/sys/fs/cgroup/ktstr",
            DEFAULT_PAGE_OFFSET,
        );
        assert!(got.is_empty(), "absent workload root → loud-absent");
    }

    /// A cyclic children list is bounded by the visited-set guard (no hang).
    #[test]
    fn walk_tolerates_cyclic_children_list() {
        let mut img = Image::new(0x4000);
        let kn_root = 0x100usize;
        let kn_ktstr = 0x180usize;
        img.wstr(0x140, "ktstr");
        img.w64(kn_root + KN_NAME, Image::kva(0x130));
        img.wstr(0x130, "/");
        img.w64(kn_ktstr + KN_NAME, Image::kva(0x140));
        let cgrp_dfl_root = 0x1000usize;
        let root = cgrp_dfl_root + ROOT_CGRP;
        let ktstr = 0x2000usize;
        img.init_cgroup(root, kn_root, 0);
        img.init_cgroup(ktstr, kn_ktstr, 0);
        img.link_children(root, &[ktstr]);
        // Corrupt ktstr's children list into a self-cycle that never reaches
        // the anchor: children.next points at a node whose next points back.
        let bogus = 0x2800usize;
        img.w64(
            ktstr + SELF_OFF + CSS_CHILDREN,
            Image::kva(bogus + CSS_SIBLING),
        );
        img.w64(bogus + CSS_SIBLING, Image::kva(bogus + CSS_SIBLING)); // self-loop
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        // Must terminate (the test would hang on an unbounded walk).
        let got = collect_workload_cgroup_psi(
            &mem,
            &offs(),
            &psi_offs(),
            Image::kva(root),
            root as u64,
            "/sys/fs/cgroup/ktstr",
            DEFAULT_PAGE_OFFSET,
        );
        // The bogus node resolves to some cgroup that has no psi → skipped;
        // the key property is that the walk terminated.
        assert!(got.len() <= 1, "cyclic list bounded, no hang");
    }

    /// The DFS recurses past one level: root → ktstr → mid → leaf (a 3-level
    /// workload subtree). Pins that a deep leaf's PSI is captured (the happy-path
    /// test only nests 2 levels, so it would not catch a single-level DFS bug).
    #[test]
    fn walk_descends_deep_tree() {
        let mut img = Image::new(0x6000);
        let kn_root = 0x100usize;
        let kn_ktstr = 0x180usize;
        let kn_mid = 0x200usize;
        let kn_leaf = 0x280usize;
        img.wstr(0x140, "ktstr");
        img.wstr(0x1c0, "mid");
        img.wstr(0x240, "leaf");
        img.w64(kn_root + KN_NAME, Image::kva(0x130));
        img.wstr(0x130, "/");
        img.w64(kn_ktstr + KN_NAME, Image::kva(0x140));
        img.w64(kn_mid + KN_NAME, Image::kva(0x1c0));
        img.w64(kn_leaf + KN_NAME, Image::kva(0x240));
        let cgrp_dfl_root = 0x1000usize;
        let root = cgrp_dfl_root + ROOT_CGRP;
        let ktstr = 0x2000usize;
        let mid = 0x2400usize;
        let leaf = 0x2800usize;
        let psi_leaf = 0x4000usize;
        img.w64(psi_leaf + 6 * 8, 333_000); // leaf total_ns
        img.init_cgroup(root, kn_root, 0);
        img.init_cgroup(ktstr, kn_ktstr, 0);
        img.init_cgroup(mid, kn_mid, 0);
        img.init_cgroup(leaf, kn_leaf, Image::kva(psi_leaf));
        img.link_children(root, &[ktstr]);
        img.link_children(ktstr, &[mid]);
        img.link_children(mid, &[leaf]);
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        let got = collect_workload_cgroup_psi(
            &mem,
            &offs(),
            &psi_offs(),
            Image::kva(root),
            root as u64,
            "/sys/fs/cgroup/ktstr",
            DEFAULT_PAGE_OFFSET,
        );
        assert_eq!(
            got.len(),
            1,
            "the single deep leaf under ktstr/mid is captured"
        );
        assert_eq!(got[0].total_ns, 333_000);
        assert_eq!(got[0].cgroup_kva, Image::kva(leaf));
    }

    /// children_of stops at the shared global budget (the NUMERIC bound, not the
    /// cycle guard): a parent with 5 distinct children walked with budget=3
    /// returns exactly 3 and drains the budget to 0. This exercises the
    /// non-multiplicative bound that caps a hostile wide fan-out.
    #[test]
    fn children_of_respects_budget() {
        let mut img = Image::new(0x4000);
        let parent = 0x1000usize;
        let kids = [0x1400usize, 0x1800, 0x1c00, 0x2000, 0x2400];
        img.init_cgroup(parent, 0x100, 0);
        for &k in &kids {
            img.init_cgroup(k, 0x100, 0);
        }
        img.link_children(parent, &kids);
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        let mut budget: u32 = 3;
        let children = children_of(
            &mut budget,
            &mem,
            &offs(),
            DEFAULT_PAGE_OFFSET,
            Image::kva(parent),
            parent as u64,
        );
        assert_eq!(
            children.len(),
            3,
            "budget caps the per-list walk numerically"
        );
        assert_eq!(budget, 0, "budget fully consumed");
    }

    /// A multi-segment workload-root path (/sys/fs/cgroup/a/b) descends TWO named
    /// levels before the leaf DFS — pins the end-to-end multi-segment descent
    /// that `segments_strip_mount_prefix` only covers at the parse layer.
    #[test]
    fn walk_multi_segment_workload_root() {
        let mut img = Image::new(0x6000);
        let kn_root = 0x100usize;
        let kn_a = 0x180usize;
        let kn_b = 0x200usize;
        let kn_leaf = 0x280usize;
        img.wstr(0x140, "a");
        img.wstr(0x1c0, "b");
        img.wstr(0x240, "cg_0");
        img.w64(kn_root + KN_NAME, Image::kva(0x130));
        img.wstr(0x130, "/");
        img.w64(kn_a + KN_NAME, Image::kva(0x140));
        img.w64(kn_b + KN_NAME, Image::kva(0x1c0));
        img.w64(kn_leaf + KN_NAME, Image::kva(0x240));
        let cgrp_dfl_root = 0x1000usize;
        let root = cgrp_dfl_root + ROOT_CGRP;
        let a = 0x2000usize;
        let b = 0x2400usize;
        let leaf = 0x2800usize;
        let psi_leaf = 0x4000usize;
        img.w64(psi_leaf + 6 * 8, 444_000);
        img.init_cgroup(root, kn_root, 0);
        img.init_cgroup(a, kn_a, 0);
        img.init_cgroup(b, kn_b, 0);
        img.init_cgroup(leaf, kn_leaf, Image::kva(psi_leaf));
        img.link_children(root, &[a]);
        img.link_children(a, &[b]);
        img.link_children(b, &[leaf]);
        // SAFETY: buf outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(img.buf.as_mut_ptr(), img.buf.len() as u64) };
        let got = collect_workload_cgroup_psi(
            &mem,
            &offs(),
            &psi_offs(),
            Image::kva(root),
            root as u64,
            "/sys/fs/cgroup/a/b",
            DEFAULT_PAGE_OFFSET,
        );
        assert_eq!(
            got.len(),
            1,
            "leaf under the a/b multi-segment root is captured"
        );
        assert_eq!(got[0].total_ns, 444_000);
    }

    #[test]
    fn segments_strip_mount_prefix() {
        assert_eq!(
            workload_root_segments("/sys/fs/cgroup/ktstr"),
            vec!["ktstr"]
        );
        assert_eq!(workload_root_segments("/sys/fs/cgroup/a/b"), vec!["a", "b"]);
        assert_eq!(workload_root_segments("ktstr"), vec!["ktstr"]);
        assert_eq!(workload_root_segments("/sys/fs/cgroup"), Vec::<&str>::new());
    }
}
