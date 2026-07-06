# CgroupGroup

`CgroupGroup` is an RAII guard that removes cgroups on drop. It
prevents cgroup leaks when workload spawning or any other operation
fails between cgroup creation and cleanup.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Track</strong><p>Remember every cgroup created during a scenario scope.</p></div>
<div class="kt-doc-card"><strong>Guard</strong><p>Drop cleanup runs on both happy paths and early errors.</p></div>
<div class="kt-doc-card"><strong>Warn</strong><p>Teardown errors are logged with context instead of panicking in <code>Drop</code>.</p></div>
</div>

```rust,ignore
#[must_use = "dropping a CgroupGroup immediately destroys the cgroups it manages"]
pub struct CgroupGroup<'a> { /* ... */ }
```

The `#[must_use]` is deliberate: binding the guard to `_` (rather
than `_guard`) drops it immediately and destroys the cgroups before
the workload runs.

## Methods

**`new(cgroups: &dyn CgroupOps)`** — creates an empty group bound to
any `CgroupOps` implementor ([`CgroupManager`](cgroup-manager.md) in
production, an in-memory fake in tests).

**`add_cgroup(name, cpuset)`** — creates a cgroup and sets its
cpuset. Auto-enables the `Cpuset` controller on the parent's
`cgroup.subtree_control` first — the difference that matters vs
`add_cgroup_no_cpuset`, which creates the cgroup without a cpuset and
without touching controllers. Both track the cgroup for removal on
drop.

**`names()`** — the names of all tracked cgroups.

## Drop behavior

On drop, the group calls `remove_cgroup()` on each tracked cgroup in
reverse insertion order, so nested children are removed before their
parents (a parent still holding child directories fails with
`ENOTEMPTY`).

`ENOENT` is the one errno the drop swallows silently: it means the
directory is already gone, so the post-condition already holds and no
cleanup is owed. (It can legitimately appear via a narrow race
between the existence check and `remove_dir`.) Every other error
surfaces as a `tracing::warn!` record carrying the cgroup name and
the full error chain — the drop never panics, but teardown failures
are visible in logs rather than silently swallowed. The record's
shape:

```text
CgroupGroup::drop: remove_cgroup returned non-ENOENT error
  cgroup=<name> err=<error chain>
  hint=EBUSY: cgroup still has live tasks — workloads were not drained before teardown
```

`EBUSY` at drop means exactly what the hint says: something is still
running in the cgroup — typically a `WorkloadHandle` that outlives
the guard, so its workers were never stopped before teardown. Drop
(or `stop_and_collect`) the handle before the guard goes out of
scope. `EACCES` gets its own hint pointing at cgroup ownership and
delegation.

## Usage

`CgroupGroup` is the standard cgroup-lifecycle pattern for custom
scenarios — [CgroupManager](cgroup-manager.md) shows the full worked
example. The shape in brief:

```rust,ignore
let mut guard = CgroupGroup::new(ctx.cgroups);
guard.add_cgroup("cg_0", &cpuset_a)?;
guard.add_cgroup("cg_1", &cpuset_b)?;
// If anything below fails, `guard` drops and removes both cgroups.
```

The helper `setup_cgroups(ctx, n, &wl)` bundles the pattern: it
creates `n` cgroups, spawns workers in each, and returns the handles
alongside the guard.

See also: [CgroupManager](cgroup-manager.md) for filesystem
operations, [Workers and Workloads](workers.md) for worker lifecycle.
