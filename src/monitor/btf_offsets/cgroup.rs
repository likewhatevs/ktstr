//! BTF offsets for the per-cgroup PSI-irq host-walk (Phase A).
//!
//! Locates each test cgroup's `struct psi_group` by walking the guest's cgroup
//! hierarchy from the `cgrp_dfl_root` global (the cgroup2 default hierarchy
//! root) down the host-held workload-root path to its leaf descendants, then
//! reads each leaf's `cgroup->psi` pointer. The psi_group read + decode reuses
//! [`super::PsiGroupOffsets`]: a per-cgroup psi_group is the SAME `struct
//! psi_group` as the system-wide `psi_system` (`kernel/sched/psi.c`
//! `psi_cgroup_alloc` `kzalloc`s the identical type), so the total/avg offsets +
//! `PSI_IRQ_FULL` index apply verbatim to a per-cgroup instance. The walk itself
//! lives in [`super::super::cgroup_walk`].
//!
//! KERNEL LAYOUT (cited HEAD c80ba8d3):
//! - `struct cgroup` (`include/linux/cgroup-defs.h:474`): first member is
//!   `struct cgroup_subsys_state self` (`:476`, so `offsetof(cgroup, self) == 0`
//!   — `cgroup` addr == its `self` css addr), `struct kernfs_node *kn` (`:526`),
//!   `struct psi_group *psi` (`:621`, a POINTER to a separately-`kzalloc`'d
//!   psi_group, NULL when `psi_cgroups_enabled` is off → loud-absent).
//! - `struct cgroup_subsys_state` (`:181`): `struct list_head sibling` (`:213`,
//!   links this css into its parent's child list), `struct list_head children`
//!   (`:214`, the anchor of this css's children), `u64 serial_nr` (`:230`, the
//!   monotonic per-creation serial — a creation identity that survives a
//!   slab-KVA reuse, so the cross-freeze fold can reject a different cgroup that
//!   reused a freed KVA).
//! - `struct cgroup_root` (`:651`): embeds `struct cgroup cgrp` LAST (`:681`,
//!   after `release_agent_path[PATH_MAX]` + `name[]` char arrays — a large,
//!   layout-sensitive offset).
//! - `struct kernfs_node` (`include/linux/kernfs.h:194`): `const char __rcu
//!   *name` (`:207`) — the cgroup's directory name. Its offset SHIFTS with
//!   `CONFIG_DEBUG_LOCK_ALLOC` (a `#ifdef`'d `struct lockdep_map dep_map`
//!   precedes it).
//!
//! Every offset is BTF-resolved, never hardcoded: three are config/layout
//! sensitive (`kernfs_node.name` lockdep `#ifdef`, `cgroup_root.cgrp` last
//! member, `cgroup_subsys_state.{sibling,children}` percpu_ref/rstat sizing), so
//! a literal would silently mis-read on a kernel built differently.

use anyhow::Result;
use btf_rs::Btf;

use super::{find_struct, member_byte_offset};

/// BTF-resolved byte offsets for the cgroup-hierarchy walk that locates a
/// cgroup's `struct psi_group`. Resolved once at monitor-thread setup;
/// `Err` only when a required struct/field is absent (a kernel without
/// `CONFIG_CGROUPS`, or a stripped vmlinux), in which case the per-cgroup
/// axis reads loud-absent. `cgroup.psi` is an unconditional pointer member
/// (always BTF-resolvable); it reads NULL at runtime, not `Err`, when
/// per-cgroup PSI is off.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CgroupWalkOffsets {
    /// `offsetof(struct cgroup, self)` — the embedded `cgroup_subsys_state`.
    /// 0 on every current kernel (`self` is the first member), but resolved
    /// so a future leading-field addition is followed, not mis-assumed. The
    /// `self` css base is where `children`/`sibling` are read.
    pub cgroup_self: usize,
    /// `offsetof(struct cgroup, kn)` — the `struct kernfs_node *` whose
    /// `name` is the cgroup's directory name (used for the workload-root path
    /// match).
    pub cgroup_kn: usize,
    /// `offsetof(struct cgroup, psi)` — the `struct psi_group *` POINTER.
    /// Reading it yields the per-cgroup psi_group KVA, or 0/NULL when
    /// `psi_cgroups_enabled` is off (loud-absent: that cgroup contributes no
    /// per-cgroup PSI sample). The `psi` member is an unconditional pointer
    /// field of `struct cgroup` (not `#ifdef`-guarded), so BTF always emits it
    /// and this offset always resolves; under `CONFIG_PSI=n` only the pointee
    /// `struct psi_group` degrades to empty and the pointer reads NULL at
    /// runtime (loud-absent), not `Err`.
    pub cgroup_psi: usize,
    /// `offsetof(struct cgroup_subsys_state, sibling)` — the `list_head`
    /// linking a css into its parent's `children` list. A child cgroup is
    /// recovered as `sibling_node_kva - cgroup_self - css_sibling` (with
    /// `cgroup_self == 0`, just `- css_sibling`).
    pub css_sibling: usize,
    /// `offsetof(struct cgroup_subsys_state, children)` — the `list_head`
    /// anchor of a css's children. A cgroup is a LEAF iff this list is empty
    /// (`children.next` points back at the anchor's own KVA).
    pub css_children: usize,
    /// `offsetof(struct cgroup_subsys_state, serial_nr)` — the `u64` monotonic
    /// per-creation serial (`css->serial_nr = css_serial_nr_next++`,
    /// `kernel/cgroup/cgroup.c`). Read alongside the cgroup KVA so the
    /// cross-freeze fold can tell a cgroup apart from a DIFFERENT one that later
    /// reused the same slab KVA: a freed `struct cgroup` is a plain `kzalloc`
    /// eligible for slab reuse, so KVA alone is NOT a stable cross-lifetime
    /// identity — a KVA match with a different serial is a different cgroup. The
    /// serial lives in the embedded `self` css, so its address is
    /// `cgroup_kva + cgroup_self + css_serial_nr`.
    pub css_serial_nr: usize,
    /// `offsetof(struct cgroup_root, cgrp)` — the embedded root `struct
    /// cgroup`. `cgrp_dfl_root + this` is the hierarchy root cgroup (the
    /// cgroup2 mount point; its children are the top-level cgroups like the
    /// workload root). Last member after `PATH_MAX` arrays — never hardcode.
    pub cgroup_root_cgrp: usize,
    /// `offsetof(struct kernfs_node, name)` — the `const char *` directory
    /// name. Shifts with `CONFIG_DEBUG_LOCK_ALLOC` (a `#ifdef`'d
    /// `lockdep_map` precedes it) — never hardcode.
    pub kernfs_node_name: usize,
}

impl CgroupWalkOffsets {
    /// Resolve the cgroup-walk offsets from a pre-loaded BTF object. `Err`
    /// when any required struct/field is absent (no `CONFIG_CGROUPS`, or a
    /// stripped vmlinux; `cgroup.psi` is an unconditional pointer member that
    /// always resolves, reading NULL at runtime when per-cgroup PSI is off) —
    /// the freeze coordinator `.ok()`s it, so absence disables the per-cgroup
    /// capture (loud-absent), mirroring [`super::PsiGroupOffsets::from_btf`].
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (cgroup, _) = find_struct(btf, "cgroup")?;
        let cgroup_self = member_byte_offset(btf, &cgroup, "self")?;
        let cgroup_kn = member_byte_offset(btf, &cgroup, "kn")?;
        let cgroup_psi = member_byte_offset(btf, &cgroup, "psi")?;

        let (css, _) = find_struct(btf, "cgroup_subsys_state")?;
        let css_sibling = member_byte_offset(btf, &css, "sibling")?;
        let css_children = member_byte_offset(btf, &css, "children")?;
        let css_serial_nr = member_byte_offset(btf, &css, "serial_nr")?;

        let (cgroup_root, _) = find_struct(btf, "cgroup_root")?;
        let cgroup_root_cgrp = member_byte_offset(btf, &cgroup_root, "cgrp")?;

        let (kernfs_node, _) = find_struct(btf, "kernfs_node")?;
        let kernfs_node_name = member_byte_offset(btf, &kernfs_node, "name")?;

        Ok(Self {
            cgroup_self,
            cgroup_kn,
            cgroup_psi,
            css_sibling,
            css_children,
            css_serial_nr,
            cgroup_root_cgrp,
            kernfs_node_name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::load_btf_from_path;
    use super::*;

    /// Resolve [`CgroupWalkOffsets`] against the test vmlinux and pin the
    /// layout invariants the walk depends on: `self` at offset 0 (so a cgroup
    /// addr equals its css addr), the `sibling`/`children` list_heads at
    /// distinct offsets within the css, and `cgroup.psi` present (the test
    /// kernel has `CONFIG_PSI`).
    #[test]
    fn parse_cgroup_walk_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let btf = match load_btf_from_path(&path) {
            Ok(b) => b,
            Err(e) => skip!("vmlinux BTF load failed: {e}"),
        };
        let off = match CgroupWalkOffsets::from_btf(&btf) {
            Ok(o) => o,
            Err(e) => skip!("CgroupWalkOffsets::from_btf failed: {e}"),
        };
        // `struct cgroup`'s first member is `self` (the embedded css), so the
        // cgroup address coincides with its css address.
        assert_eq!(
            off.cgroup_self, 0,
            "offsetof(cgroup, self) must be 0 (self is the first member)"
        );
        // sibling and children are distinct list_heads within the css.
        assert_ne!(
            off.css_sibling, off.css_children,
            "css.sibling and css.children must be at distinct offsets"
        );
        // cgrp is the last member of cgroup_root, after the PATH_MAX char
        // arrays — a nonzero, large offset.
        assert!(
            off.cgroup_root_cgrp > 0,
            "offsetof(cgroup_root, cgrp) must be nonzero (cgrp is not the first member)"
        );
    }
}
