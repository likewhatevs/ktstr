# Run Modes

Every ktstr VM boots in one of three run modes, and the mode decides how
much of the host the run reserves, how its vCPU threads are placed, and
therefore how much a timing measurement can be trusted. The mode is not a
tuning knob you reach for mid-test — it is the contract the run makes with
the host and with every concurrent ktstr process sharing it.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Default</strong><p>Per-offset shared LLC lock plus an exclusive per-CPU lock, walked 1:1 against the host. Reserves without isolating; overcommits when the host is too small.</p></div>
<div class="kt-doc-card"><strong>Performance</strong><p>Whole LLCs held exclusively, vCPUs pinned 1:1 under SCHED_FIFO with a reserved service CPU, NUMA-bound hugetlb memory. Measurement fidelity at the cost of host sharing.</p></div>
<div class="kt-doc-card"><strong>No-perf</strong><p>A shared subset of LLCs sized to a CPU budget, vCPUs masked (not pinned) and spread across least-held LLCs. Coordination without the full isolation contract.</p></div>
</div>

Two of these modes have dedicated deep-dive pages —
[Performance Mode](performance-mode.md) for the isolation contract and
[Resource Budget](resource-budget.md) for the coordination mechanics.
This page is the concept that unifies all three: what each promises, how
the shared reliability machinery behaves under each, and how to choose.

## The three modes

| | Default | Performance | No-perf |
|---|---|---|---|
| **Selected by** | neither switch set | `performance_mode = true` | `--no-perf-mode` / `KTSTR_NO_PERF_MODE`, or a `cpu_budget` |
| **LLC lock** | shared (`LOCK_SH`), on the 1:1 plan's LLCs | exclusive (`LOCK_EX`), one per virtual LLC | shared (`LOCK_SH`), on a budgeted LLC subset |
| **Per-CPU lock** | exclusive (`LOCK_EX`), one per assigned host CPU | none — the exclusive LLC lock covers its CPUs | none — the cgroup cpuset is the enforcement layer |
| **vCPU placement** | pinned 1:1 (or masked, overcommitted, when the host is too small) | pinned 1:1 to reserved cores | masked to the reserved CPU pool, not pinned |
| **vCPU scheduling** | `SCHED_OTHER` | `SCHED_FIFO` priority 1 | `SCHED_OTHER` |
| **Memory** | anonymous, THP via `MADV_HUGEPAGE` | NUMA-bound 2 MB hugetlb (`MAP_HUGETLB`, strict `MPOL_BIND`) | anonymous, THP via `MADV_HUGEPAGE` |
| **What it promises** | a reservation, so peers do not time-slice the same CPU | host-noise-free timing — a gap in the guest is the scheduler's | bounded host footprint, not isolation |

The same contrast as a host sketch — how each mode's vCPU threads land on
host CPUs:

<div class="kt-figure"><svg width="700" height="268" viewBox="0 0 700 268" role="img" aria-label="How each mode places vCPU threads on host CPUs. Performance: three vCPUs each pinned 1:1 to a core inside an exclusively-locked host LLC, with a separate reserved service CPU hosting the pinned FIFO-2 sensing threads; hugetlb memory with strict mbind; dilation near 1.0, higher means the isolation was violated. Default: three vCPUs pinned 1:1 to cores inside a shared-locked LLC whose individual CPUs are exclusively locked; THP via MADV_HUGEPAGE; FIFO-2 sensing unpinned; dilation quantifies residual host delay. No-perf: three vCPU threads fan into a shared LOCK_SH pool of LLC CPUs enforced by a cpuset — no per-CPU ownership; THP via MADV_HUGEPAGE; FIFO-2 sensing unpinned; dilation quantifies the sharing the budget accepted.">
  <g>
    <text x="20" y="26" font-size="12" font-weight="700" fill="var(--kt-accent)">Performance</text>
    <text x="20" y="42" font-size="9" fill="var(--fg)" opacity=".75">pin 1:1 + service CPU</text>
    <rect x="32" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="47" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v0</text>
    <rect x="72" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="87" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v1</text>
    <rect x="112" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="127" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v2</text>
    <line x1="47" y1="82" x2="47" y2="124" stroke="var(--kt-accent)" stroke-width="1.3"/>
    <line x1="87" y1="82" x2="87" y2="124" stroke="var(--kt-accent)" stroke-width="1.3"/>
    <line x1="127" y1="82" x2="127" y2="124" stroke="var(--kt-accent)" stroke-width="1.3"/>
    <rect x="26" y="126" width="128" height="58" rx="8" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
    <rect x="32" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".8"/>
    <text x="47" y="148" text-anchor="middle" font-size="8" fill="var(--fg)">c0</text>
    <rect x="72" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".8"/>
    <text x="87" y="148" text-anchor="middle" font-size="8" fill="var(--fg)">c1</text>
    <rect x="112" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".8"/>
    <text x="127" y="148" text-anchor="middle" font-size="8" fill="var(--fg)">c2</text>
    <text x="90" y="176" text-anchor="middle" font-size="8.5" fill="var(--kt-accent)">host LLC — LOCK_EX</text>
    <rect x="162" y="126" width="52" height="58" rx="8" fill="none" stroke="var(--kt-accent)" stroke-width="1.3" stroke-dasharray="3 3"/>
    <text x="188" y="150" text-anchor="middle" font-size="8.5" font-weight="700" fill="var(--fg)">svc</text>
    <text x="188" y="164" text-anchor="middle" font-size="7.5" fill="var(--fg)" opacity=".8">FIFO-2</text>
    <text x="188" y="175" text-anchor="middle" font-size="7.5" fill="var(--fg)" opacity=".8">pinned</text>
    <text x="20" y="206" font-size="8.5" fill="var(--fg)" opacity=".75">vCPUs SCHED_FIFO-1</text>
    <text x="20" y="220" font-size="8.5" fill="var(--fg)" opacity=".75">hugetlb 2 MB · strict mbind</text>
    <text x="20" y="234" font-size="8.5" fill="var(--fg)" opacity=".75">D ≈ 1.0 — higher means the</text>
    <text x="20" y="248" font-size="8.5" fill="var(--fg)" opacity=".75">isolation was violated</text>
  </g>
  <g>
    <text x="250" y="26" font-size="12" font-weight="700" fill="var(--fg)">Default</text>
    <text x="250" y="42" font-size="9" fill="var(--fg)" opacity=".75">pin 1:1, reservation only</text>
    <rect x="262" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="277" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v0</text>
    <rect x="302" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="317" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v1</text>
    <rect x="342" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="357" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v2</text>
    <line x1="277" y1="82" x2="277" y2="124" stroke="var(--fg)" stroke-width="1.2" opacity=".7"/>
    <line x1="317" y1="82" x2="317" y2="124" stroke="var(--fg)" stroke-width="1.2" opacity=".7"/>
    <line x1="357" y1="82" x2="357" y2="124" stroke="var(--fg)" stroke-width="1.2" opacity=".7"/>
    <rect x="256" y="126" width="128" height="58" rx="8" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
    <rect x="262" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--kt-accent)" stroke-width="1.4"/>
    <text x="277" y="148" text-anchor="middle" font-size="8" fill="var(--fg)">c0</text>
    <rect x="302" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--kt-accent)" stroke-width="1.4"/>
    <text x="317" y="148" text-anchor="middle" font-size="8" fill="var(--fg)">c1</text>
    <rect x="342" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--kt-accent)" stroke-width="1.4"/>
    <text x="357" y="148" text-anchor="middle" font-size="8" fill="var(--fg)">c2</text>
    <text x="320" y="176" text-anchor="middle" font-size="8.5" fill="var(--fg)" opacity=".8">LLC LOCK_SH · CPUs LOCK_EX</text>
    <text x="250" y="206" font-size="8.5" fill="var(--fg)" opacity=".75">vCPUs SCHED_OTHER</text>
    <text x="250" y="220" font-size="8.5" fill="var(--fg)" opacity=".75">THP via MADV_HUGEPAGE</text>
    <text x="250" y="234" font-size="8.5" fill="var(--fg)" opacity=".75">FIFO-2 sensing, unpinned</text>
    <text x="250" y="248" font-size="8.5" fill="var(--fg)" opacity=".75">D quantifies residual host delay</text>
  </g>
  <g>
    <text x="480" y="26" font-size="12" font-weight="700" fill="var(--fg)">No-perf</text>
    <text x="480" y="42" font-size="9" fill="var(--fg)" opacity=".75">mask onto shared pool</text>
    <rect x="492" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="507" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v0</text>
    <rect x="532" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="547" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v1</text>
    <rect x="572" y="62" width="30" height="20" rx="5" fill="none" stroke="var(--kt-rule)" stroke-width="1.2"/>
    <text x="587" y="76" text-anchor="middle" font-size="8" fill="var(--fg)">v2</text>
    <line x1="507" y1="82" x2="546" y2="124" stroke="var(--fg)" stroke-width="1.2" opacity=".6"/>
    <line x1="547" y1="82" x2="554" y2="124" stroke="var(--fg)" stroke-width="1.2" opacity=".6"/>
    <line x1="587" y1="82" x2="562" y2="124" stroke="var(--fg)" stroke-width="1.2" opacity=".6"/>
    <rect x="486" y="126" width="168" height="58" rx="8" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.3"/>
    <rect x="492" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".5"/>
    <rect x="532" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".5"/>
    <rect x="572" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".5"/>
    <rect x="612" y="132" width="30" height="24" rx="4" fill="none" stroke="var(--fg)" stroke-width="1" opacity=".5"/>
    <text x="570" y="176" text-anchor="middle" font-size="8.5" fill="var(--fg)" opacity=".8">LLC subset — LOCK_SH pool · cpuset</text>
    <text x="480" y="206" font-size="8.5" fill="var(--fg)" opacity=".75">vCPUs SCHED_OTHER, no ownership</text>
    <text x="480" y="220" font-size="8.5" fill="var(--fg)" opacity=".75">THP via MADV_HUGEPAGE</text>
    <text x="480" y="234" font-size="8.5" fill="var(--fg)" opacity=".75">FIFO-2 sensing, unpinned</text>
    <text x="480" y="248" font-size="8.5" fill="var(--fg)" opacity=".75">D quantifies the accepted sharing</text>
  </g>
</svg></div>

### Default

The default path defers its plan to run time: it walks each LLC offset,
computing a 1:1 candidate placement and taking `LOCK_SH` on the LLC plus
`LOCK_EX` on each assigned host CPU until one offset's locks are all free
([`acquire_default_run_locks`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/mod.rs)).
The shared LLC lock coexists with other default and no-perf holders; the
exclusive per-CPU locks are what keep two default VMs from ever
time-slicing the same host CPU. If a 1:1 candidate *maps* but every offset
is busy, the run does not overcommit onto a peer's CPUs — it joins
the lock-dir acquisition queue and, as head, re-probes every offset
on each lock-dir wake (see the resource-budget page's queue section);
only a zero-progress patience window yields a retryable
`ResourceContention` failure that nextest re-runs.

When no 1:1 plan can exist because the host is simply too small,
the default path falls back to
[`build_overcommit_run_locks`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/mod.rs):
every vCPU thread is masked to the allowed CPU set and the run proceeds
*overcommitted*, warning when that genuinely oversubscribes the guest.
The overcommit-vs-contention decision is the host-gate policy — make the
run work, overcommit only when nothing else is possible — and the stamped
`cpu_budget` in the sidecar drops below the vCPU count so an A/B against an
overcommitted run is flagged, not silently confounded.

<div class="kt-figure"><svg width="700" height="452" viewBox="0 0 700 452" role="img" aria-label="Default-mode run-lock decision walk. From run()'s acquire_default_run_locks: if the host topology was not cached (sysfs unreadable) the run goes straight to the overcommit fallback. Otherwise a per-offset candidate walk starts at a pid-windowed offset and wraps over max_slots: compute_pinning produces a 1:1 candidate for the offset; if it maps, the run takes a shared LLC flock plus an exclusive per-CPU flock — acquired means a 1:1 pinned run with the module-default halt-poll; a busy lock or an unmappable offset moves to the next offset. When all offsets are tried: if any offset produced a candidate the run queues and re-probes every offset on each lock-dir wake, failing with a retryable transient ResourceContention only if a zero-progress patience window expires (nextest re-runs it); if none could map, the run overcommits — vCPUs masked to the allowed cpuset, budget rewritten with a warning, halt-poll zero.">
  <defs><marker id="rm-arrA" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <rect x="20" y="16" width="200" height="44" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
  <text x="36" y="34" font-size="11" font-weight="700" fill="var(--fg)">default mode</text>
  <text x="36" y="50" font-size="9.5" fill="var(--fg)" opacity=".75">run() → acquire_run_locks</text>
  <path d="M220 38 L 248 38" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrA)"/>
  <rect x="252" y="16" width="180" height="44" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="342" y="42" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--fg)">host topology cached?</text>
  <path d="M432 38 L 600 38 L 600 366" stroke="var(--fg)" stroke-width="1.3" fill="none" marker-end="url(#rm-arrA)"/>
  <text x="442" y="30" font-size="9" fill="var(--fg)" opacity=".65">no — sysfs unreadable</text>
  <path d="M342 60 L 342 84" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrA)"/>
  <text x="350" y="78" font-size="9" fill="var(--fg)" opacity=".65">yes</text>
  <rect x="40" y="88" width="430" height="184" rx="10" fill="none" stroke="var(--kt-rule)" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="56" y="108" font-size="9.5" font-weight="700" fill="var(--fg)" opacity=".8">per-offset candidate walk — pid-windowed start, wraps over max_slots</text>
  <rect x="60" y="124" width="190" height="48" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="155" y="143" text-anchor="middle" font-size="10" font-weight="700" fill="var(--fg)">compute_pinning(offset)</text>
  <text x="155" y="159" text-anchor="middle" font-size="9" fill="var(--fg)" opacity=".75">candidate maps 1:1?</text>
  <path d="M250 148 L 278 148" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrA)"/>
  <text x="256" y="141" font-size="8.5" fill="var(--fg)" opacity=".65">yes</text>
  <rect x="280" y="124" width="172" height="48" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="366" y="143" text-anchor="middle" font-size="10" font-weight="700" fill="var(--fg)">take LLC LOCK_SH</text>
  <text x="366" y="159" text-anchor="middle" font-size="9" fill="var(--fg)" opacity=".75">+ per-CPU LOCK_EX</text>
  <path d="M366 172 C 336 202, 196 202, 162 176" stroke="var(--fg)" stroke-width="1.2" fill="none" opacity=".7" marker-end="url(#rm-arrA)"/>
  <text x="264" y="212" text-anchor="middle" font-size="8.5" fill="var(--fg)" opacity=".65">lock busy or no 1:1 map → next offset</text>
  <path d="M430 172 C 420 240, 180 260, 122 366" stroke="var(--kt-accent)" stroke-width="1.4" fill="none" marker-end="url(#rm-arrA)"/>
  <text x="404" y="224" font-size="9" fill="var(--kt-accent)">acquired</text>
  <path d="M315 272 L 315 288" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrA)"/>
  <text x="323" y="286" font-size="9" fill="var(--fg)" opacity=".65">all offsets tried</text>
  <rect x="210" y="292" width="210" height="40" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="315" y="309" text-anchor="middle" font-size="10" font-weight="700" fill="var(--fg)">any offset produced</text>
  <text x="315" y="323" text-anchor="middle" font-size="10" font-weight="700" fill="var(--fg)">a 1:1 candidate?</text>
  <path d="M315 332 L 348 366" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrA)"/>
  <text x="232" y="356" font-size="9" fill="var(--fg)" opacity=".65">yes — every slot busy</text>
  <path d="M420 312 L 560 312 L 560 366" stroke="var(--fg)" stroke-width="1.3" fill="none" marker-end="url(#rm-arrA)"/>
  <text x="430" y="305" font-size="9" fill="var(--fg)" opacity=".65">no — cannot map</text>
  <rect x="20" y="370" width="200" height="64" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.6"/>
  <text x="36" y="390" font-size="11" font-weight="700" fill="var(--kt-accent)">1:1 pinned run</text>
  <text x="36" y="405" font-size="8.5" fill="var(--fg)" opacity=".8">each vCPU owns a host CPU</text>
  <text x="36" y="418" font-size="8.5" fill="var(--fg)" opacity=".8">halt-poll: module default</text>
  <rect x="250" y="370" width="200" height="64" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
  <text x="266" y="390" font-size="11" font-weight="700" fill="var(--fg)">ResourceContention</text>
  <text x="266" y="405" font-size="8.5" fill="var(--fg)" opacity=".8">queue + re-plan on wake; fail</text>
  <text x="266" y="418" font-size="8.5" fill="var(--fg)" opacity=".8">only on zero-progress patience</text>
  <rect x="480" y="370" width="200" height="64" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.4" stroke-dasharray="5 4"/>
  <text x="496" y="390" font-size="11" font-weight="700" fill="var(--fg)">overcommit fallback</text>
  <text x="496" y="404" font-size="8.5" fill="var(--fg)" opacity=".8">vCPUs masked to allowed cpuset</text>
  <text x="496" y="417" font-size="8.5" fill="var(--fg)" opacity=".8">budget rewritten + warning</text>
  <text x="496" y="430" font-size="8.5" fill="var(--fg)" opacity=".8">halt-poll: 0</text>
</svg></div>

### Performance

`performance_mode = true` takes a whole host LLC per virtual LLC under
`LOCK_EX`, pins each vCPU thread 1:1 to a core within it, and reserves one
extra host CPU that no vCPU shares for the monitor and watchdog. On top of
the pinning it runs vCPUs under `SCHED_FIFO`, binds guest memory to the
pinned vCPUs' NUMA nodes (strict `MPOL_BIND`, no silent remote fallback),
backs it with 2 MB hugetlb pages, and — on x86_64 — suppresses PAUSE/HLT
exits and hints `KVM_HINTS_REALTIME` so the guest drives its own
haltpoll instead of paying wakeup latency
([`validate_performance_mode`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/builder.rs),
[`vcpu.rs`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/vcpu.rs),
[`numa_mem.rs`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/numa_mem.rs)).
It never runs unisolated: if the host cannot honor the guarantee the build
fails before boot and the test skips visibly. The full contract, failure
modes, and host-sizing are in [Performance Mode](performance-mode.md).

### No-perf

`--no-perf-mode` (or a `cpu_budget`) forces `performance_mode = false` and
routes the run through the budgeted path: the planner reserves a subset of
host LLCs sized to a CPU budget, takes `LOCK_SH` on them, and masks every
vCPU thread onto the reserved pool via a cgroup v2 cpuset — no pinning, no
RT scheduling, no hugetlb. Placement is *Spread*: concurrent no-perf VMs
fan out across the least-held LLCs rather than stacking onto the same
low-LLC prefix (builds keep *Consolidate*). The build-time `LOCK_SH` fds
are held through setup so a concurrent peer's holder count reads true, then
`run()` re-plans against those now-truthful counts and adopts the fresh
plan's own fds (acquire-before-release, so retained LLCs never flicker
free — [`acquire_run_locks`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/mod.rs),
no-perf arm in
[`builder.rs`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/builder.rs)).
The CPU-budget resolution and the cpuset sandbox live in
[Resource Budget](resource-budget.md).

## Who selects what

Selection is by two switches, resolved at build time: `performance_mode`
on the test attribute or builder, and `--no-perf-mode` / `KTSTR_NO_PERF_MODE`
(any non-empty value). A per-test `cpu_budget` attribute or an operator
`--cpu-cap` implies the budgeted no-perf path. `--no-perf-mode` wins — it
forces `performance_mode = false` — so an operator can always strip
isolation off a test that asked for it.

The **verifier sweep hardcodes** `no_perf_mode(true)`, and the reason is
instructive
([`collect_verifier_output`](https://github.com/likewhatevs/ktstr/blob/main/src/verifier.rs)):
it only loads the scheduler's BPF and reads the kernel verifier's
load-time `verified_insns` counts, a value fixed at BPF load and wholly
independent of every perf-mode tuning (pinning, RT priority, hugepages,
NUMA mbind, exit suppression). It needs none of that. Disabling perf mode
also moves the run *off* the default run-lock path — whose per-offset
`LOCK_SH` search hard-fails `all N LLC slots busy` when no offset is free
— *onto* the no-perf plan, which reserves a shared `LOCK_SH` subset;
`LOCK_SH` holders are mutually compatible, so a 30-cell parallel sweep no
longer starves itself on the LLC lock. A `performance_mode` peer holding
`LOCK_EX` can still defer a verifier cell (nextest retries it), which is
correct: the verifier must not perturb an isolated peer's pinned CPUs.

<div class="kt-figure"><svg width="700" height="212" viewBox="0 0 700 212" role="img" aria-label="Mode-selection decision flow. Starting from a VM run: if performance_mode is true, the run is Performance mode (exclusive LLC locks, 1:1 FIFO pinning). Otherwise, if no-perf-mode or a cpu_budget or the verifier is set, the run is No-perf mode (shared LLC subset, masked vCPUs). Otherwise it is Default mode, which pins 1:1 when a plan fits and overcommits when the host is too small.">
  <defs><marker id="rm-arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <rect x="20" y="86" width="96" height="40" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.4"/>
  <text x="68" y="103" text-anchor="middle" font-size="11" font-weight="700" fill="var(--fg)">VM run</text>
  <text x="68" y="118" text-anchor="middle" font-size="9" fill="var(--fg)" opacity=".7">build time</text>
  <path d="M116 106 L 150 106" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arr)"/>
  <rect x="152" y="84" width="150" height="44" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="227" y="103" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--fg)">performance_mode</text>
  <text x="227" y="118" text-anchor="middle" font-size="9.5" fill="var(--fg)" opacity=".75">= true ?</text>
  <path d="M302 106 L 336 106" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arr)"/>
  <text x="316" y="99" font-size="9" fill="var(--fg)" opacity=".6">no</text>
  <rect x="338" y="84" width="150" height="44" rx="9" fill="none" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="413" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="var(--fg)">no-perf / cpu_budget</text>
  <text x="413" y="118" text-anchor="middle" font-size="9.5" fill="var(--fg)" opacity=".75">/ verifier ?</text>
  <path d="M488 106 L 522 106" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arr)"/>
  <text x="504" y="99" font-size="9" fill="var(--fg)" opacity=".6">no</text>
  <rect x="524" y="84" width="156" height="44" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-rule)" stroke-width="1.3"/>
  <text x="602" y="100" text-anchor="middle" font-size="11" font-weight="700" fill="var(--fg)">Default</text>
  <text x="602" y="118" text-anchor="middle" font-size="9" fill="var(--fg)" opacity=".75">1:1 pin · else overcommit</text>
  <path d="M227 84 L 227 46" stroke="var(--kt-accent)" stroke-width="1.3" marker-end="url(#rm-arr)"/>
  <text x="235" y="66" font-size="9" fill="var(--kt-accent)" opacity=".9">yes</text>
  <rect x="152" y="12" width="150" height="34" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.5"/>
  <text x="227" y="33" text-anchor="middle" font-size="11" font-weight="700" fill="var(--kt-accent)">Performance</text>
  <path d="M413 84 L 413 46" stroke="var(--kt-accent)" stroke-width="1.3" marker-end="url(#rm-arr)"/>
  <text x="421" y="66" font-size="9" fill="var(--kt-accent)" opacity=".9">yes</text>
  <rect x="338" y="12" width="150" height="34" rx="9" fill="var(--kt-accent-soft)" stroke="var(--kt-accent)" stroke-width="1.5"/>
  <text x="413" y="33" text-anchor="middle" font-size="11" font-weight="700" fill="var(--kt-accent)">No-perf</text>
  <text x="20" y="176" font-size="9.5" fill="var(--fg)" opacity=".7">Exclusive LLC locks · 1:1 FIFO pinning · hugetlb · NUMA mbind</text>
  <text x="20" y="192" font-size="9.5" fill="var(--fg)" opacity=".7">Shared LLC subset · masked vCPUs · Spread placement · cpuset budget</text>
  <text x="20" y="208" font-size="9.5" fill="var(--fg)" opacity=".7">Shared LLC + exclusive per-CPU · 1:1 reservation, overcommit fallback</text>
</svg></div>

## Shared reliability machinery

The safety and observability machinery runs in every mode; some of it
reads the mode to decide how hard to press.

### The progress watchdog is mode-independent

The three-tier progress watchdog
([`watchdog_step.rs`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/freeze_coord/watchdog_step.rs))
makes the same decisions regardless of mode — it reasons about a phase's
CPU burn and runnable demand, not about pinning:

- **Tier-1 (spinning wedge)** — the busiest single vCPU's in-phase CPU
  burn crossed the phase's *flat, width-independent* budget without
  reaching a milestone. The evidence is the max per-vCPU burn, so a wide
  idle guest's diffuse background CPU does not trip it; a lone spinner
  does. The budget is widened 3/2 under the pthread CPU currency to absorb
  VM-exit overhead.
- **Tier-2 (silent idle wedge)** — an INFRA phase sat past its wall
  backstop with live evidence channels and no runnable demand. The
  runnable conjunct is the load-bearing idea: a starved-but-alive cell
  *with work* always shows queued-or-running tasks in its own runqueue
  memory (readable regardless of host scheduling), so it is exempt at any
  dilation, while a cell with nothing runnable is not starved of anything
  — idle in an infrastructure phase past the backstop *is* the wedge.
  Deliberately no CPU term: a wide idle guest's housekeeping-CPU burn
  scales with vCPU count, so no width-stable CPU floor exists.
- **Tier-3 (deadman deferral)** — the guest-derived hard deadline fires at
  the wall only if the monitor is dead or the cell is inert (its busiest
  single vCPU's CPU trickle stalled below a currency-dependent floor for
  two consecutive 10 s windows, *and* no milestone within a 60 s grace). A
  merely-slow-but-alive cell outlives the wall deadline by design; its
  outer bound is the harness `terminate-after`, not this deadman.

Only the CPU *currency* differs by host, not by mode: the PMU task-clock
(guest-only time) uses the tight floors, and the pthread fallback (which
charges VM-exit overhead to an idle guest) uses widened floors so an idle
wedge stays killable without a starved cell being misjudged.

### FIFO-2 sensing threads

The watchdog and hang detector run at `SCHED_FIFO` priority 2
**unconditionally**, in every mode
([`freeze_coord/mod.rs`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/freeze_coord/mod.rs)).
The detector's own sensing must not dilate with the load it measures:
under extreme host dilation a `SCHED_OTHER` watchdog can miss its own
100 ms tick, latch `monitor_live = false`, and fire the deadman against a
cell that was actively progressing. FIFO-2 grants scheduling immunity at
~microseconds per tick. In perf mode the vCPUs sit at FIFO-1, below it; in
default and no-perf mode it outranks the `SCHED_OTHER` vCPUs — intended,
sensing must win. The *service-CPU pin* on those threads is the only
mode-gated part: it applies only in perf mode, where a service CPU was
reserved for them to pin to.

### Halt-poll policy is mode-aware

`KVM_CAP_HALT_POLL` is set per-VM from the resolved mode plus the run-time
overcommit outcome
([`halt_poll_policy`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/setup/mod.rs)):

| Mode / outcome | Halt-poll | Why |
|---|---|---|
| No-perf | `0` | the guest deliberately shares host CPUs; polling burns CPU that belongs to peers |
| Performance | module default | perf mode disables HLT exits and drives the guest's own haltpoll — host polling is redundant |
| Default, overcommit fallback | `0` | vCPUs exceed the acquired host CPUs; polling wastes contended time |
| Default, 1:1 pin | module default | each vCPU owns a host CPU; leave the stock 200 µs |

### Hugepages

Perf mode reserves explicit 2 MB **hugetlb** pages (`MAP_HUGETLB`) and
NUMA-binds them; `MADV_HUGEPAGE` is not applied there (and is rejected on
hugetlb mappings anyway). Default and no-perf mode use anonymous memory
with `MADV_HUGEPAGE`, so THP applies subject to the host's `THP=madvise`
policy — a hint, not a reservation
([`numa_mem.rs`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/numa_mem.rs)).

### Dilation reporting

Every run samples host-side vCPU scheduling dilation
`D = 1 + Σrun_delay / Σon_cpu` over its vCPU host threads
([`HostVcpuSchedstat::dilation`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/result.rs)),
measured host-side deliberately so it captures the CPU the vCPU thread was
starved of by the *host*, orthogonal to the guest scheduler under test.
Its meaning shifts by mode: in perf mode, where vCPUs are pinned 1:1 to
reserved cores, `D` should read ~1.0, so a departure is an
**isolation-violation indicator** — something got onto those cores. In
default and no-perf mode, where vCPUs share host CPUs, `D` **quantifies
the overcommit** the mode accepted.

### The contention witness and the wall-latency verdict

The Body (measurement) phase's dilation `D` — plus a variable-width series of
runner-cgroup CPU-pressure deltas during that phase — form the **contention
witness** the wall-latency ceilings consume. Full vCPU schedstat sweeps happen
only at lifecycle transitions; the Body hot path reads one PSI counter on the
monitor's already-existing wake, independent of vCPU count. A complete
task-specific schedstat delta over the same widened Body span caps the PSI
result, so another concurrently running cell in the same runner cgroup cannot
inflate `W` beyond this VM's own accumulated delay. From the series ktstr
derives `W(L)`, the worst host delay any interval of length `L` could have
absorbed; pairing that with a gate's measured latency turns
`max_p99_wake_latency_ns` from a plain threshold into the contention-aware
tri-state (pass always sound / fail only when refutation-proof /
indeterminate = a non-blocking annotated pass) described under
[Wall-latency ceilings under host contention](checking.md#wall-latency-ceilings-under-host-contention).

The same page covers the perf-mode counterpart to the isolation-violation
indicator above: when a `performance_mode` cell's Body `D` exceeds the
perf-isolation ceiling, that is not a scheduler failure but an infra
fault, and the run's timing verdicts are marked untrustworthy regardless
of any gate outcome.

## Lock interaction across concurrent modes

The flock semantics compose so the modes can share a host safely. Shared
holders coexist; an exclusive holder blocks every shared acquirer and vice
versa. The practical matrix, per resource:

| Concurrent runs | LLC lock | Per-CPU lock | Result |
|---|---|---|---|
| Default + Default | both `LOCK_SH` — coexist | `LOCK_EX` each — mutually exclusive | share LLCs, never the same host CPU |
| Default / No-perf + No-perf | all `LOCK_SH` — coexist | no per-CPU lock | share LLCs freely |
| Any shared holder + Performance | `LOCK_SH` vs `LOCK_EX` — block | — | perf waits for all shared holders to release |
| Performance + Performance | `LOCK_EX` each — mutually exclusive | — | serialized per LLC |

So any number of default and no-perf runs (and kernel builds, which take
the shared path) coexist on shared LLCs; a perf-mode run waits for them
all, and while it holds its LLCs nobody else touches those CPUs. The
per-resource lock table and the planning phases are in
[Resource Budget](resource-budget.md#the-three-coordination-modes).

## Tradeoffs

| | Default | Performance | No-perf |
|---|---|---|---|
| **Measurement fidelity** | reservation only — host noise remains | highest — host variance removed | lowest — vCPUs float on shared CPUs |
| **Host sharing** | shares LLCs, owns its CPUs | owns whole LLCs exclusively | shares a budgeted LLC subset |
| **Contention behaviour** | wait for holders, then retryable fail (or overcommit if the host is too small) | wait + retryable fail (`ResourceContention`) | shared pool — peers wait, not race |
| **Cost** | none beyond the reservation | needs `(llcs·cores·threads)+1` CPUs, free LLCs, hugepages, `CAP_SYS_NICE` | one CPU budget, no privileges |
| **Use when** | correctness tests where pass/fail is binary | timing thresholds — gaps, spreads, wake-latency, A/B against the same host | multi-tenant CI, kernel builds beside perf runs, deliberate oversubscription |

The tradeoff space at a glance — each mode trades measurement fidelity
against how densely the host can be shared, and each fails contention
differently. Positions are ordered by the documented mechanisms; the
dilation figures are the [measured reference points](#validation-evidence):

<div class="kt-figure"><svg width="700" height="290" viewBox="0 0 700 290" role="img" aria-label="The mode tradeoff space: measurement fidelity on the vertical axis against host sharing and parallel density on the horizontal axis. Performance mode sits top-left — highest fidelity, exclusive host use, dilation about 1.0 when pinned, and contention resolves as skip plus retry. Default mode sits mid-chart — a reservation without isolation, sharing LLCs while owning its CPUs, busy slots resolving as skip plus retry. No-perf mode sits lower right — a shared pool where peers wait rather than race, with the measured wide near-1:1 dilation about 1.13 and the +40 percent wakeup-p99 tail from FIFO-2 sensing. A dashed arrow from Default leads to its overcommit fallback at the bottom right: when the host is too small the run proceeds masked and oversubscribed, where timing metrics are host artifacts.">
  <defs><marker id="rm-arrC" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg)"/></marker></defs>
  <text x="48" y="20" font-size="10" font-weight="700" fill="var(--fg)" opacity=".8">measurement fidelity</text>
  <path d="M60 250 L 60 30" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrC)"/>
  <path d="M60 250 L 665 250" stroke="var(--fg)" stroke-width="1.3" marker-end="url(#rm-arrC)"/>
  <text x="665" y="268" text-anchor="end" font-size="10" font-weight="700" fill="var(--fg)" opacity=".8">host sharing / parallel density</text>
  <circle cx="150" cy="62" r="7" fill="var(--kt-accent)"/>
  <text x="166" y="58" font-size="10.5" font-weight="700" fill="var(--kt-accent)">Performance</text>
  <text x="166" y="72" font-size="9" fill="var(--fg)" opacity=".75">D ≈ 1.0 pinned · exclusive LLCs</text>
  <text x="166" y="85" font-size="9" fill="var(--fg)" opacity=".75">contention → skip + retry</text>
  <circle cx="320" cy="122" r="7" fill="var(--kt-accent)"/>
  <text x="336" y="118" font-size="10.5" font-weight="700" fill="var(--fg)">Default</text>
  <text x="336" y="132" font-size="9" fill="var(--fg)" opacity=".75">reservation only — shares LLCs, owns CPUs</text>
  <text x="336" y="145" font-size="9" fill="var(--fg)" opacity=".75">busy slots → skip + retry</text>
  <circle cx="470" cy="158" r="7" fill="var(--kt-accent)"/>
  <text x="486" y="153" font-size="10.5" font-weight="700" fill="var(--fg)">No-perf</text>
  <text x="486" y="167" font-size="9" fill="var(--fg)" opacity=".75">shared pool — peers wait, not race</text>
  <text x="486" y="180" font-size="9" fill="var(--fg)" opacity=".75">wide near-1:1: D ≈ 1.13,</text>
  <text x="486" y="193" font-size="9" fill="var(--fg)" opacity=".75">+40% wakeup-p99 tail</text>
  <path d="M328 130 C 360 215, 450 240, 598 237" stroke="var(--fg)" stroke-width="1.2" fill="none" stroke-dasharray="5 4" opacity=".7" marker-end="url(#rm-arrC)"/>
  <circle cx="610" cy="235" r="6" fill="none" stroke="var(--fg)" stroke-width="1.4" stroke-dasharray="3 3"/>
  <text x="690" y="210" text-anchor="end" font-size="9" font-weight="700" fill="var(--fg)" opacity=".8">default, host too small → overcommit</text>
  <text x="690" y="223" text-anchor="end" font-size="9" fill="var(--fg)" opacity=".7">timing metrics are host artifacts</text>
</svg></div>

Correctness tests (cpuset isolation, zero-work detection) do not need
perf mode — their verdict is binary and unaffected by host noise. Timing
tests (`max_gap_ms`, `max_spread_pct`, `max_p99_wake_latency_ns`) do, and
their results are comparable only against runs on the same host, since
guest-side jitter from shared caches and memory bandwidth remains even
under full isolation. See [Checking](checking.md) for which checks are
timing-sensitive.

## Validation evidence

The mode machinery is measured, not assumed. The committed evidence is
[`dilation_validation.md`](https://github.com/likewhatevs/ktstr/blob/main/src/vmm/freeze_coord/dilation_validation.md);
the raw numbers live there, summarized here.

**The reliability rework does not shift the measurements it annotates.**
The full matrix was re-run against the pre-branch baseline at the
branch's final code state (release builds, N=6 per side, one run at a
time on a load-gated host): across the perf-mode steady and
across-detach cells, the default-mode pipe, split, and iteration-rate
cells, and the narrow wakeup/request percentiles, every metric lands
inside the baseline's run-to-run envelope or within integer-µs bucket
noise, and negative-detection is 12/12 on both sides. The systematic
departures are *improvements*: the default-mode split request tail and
the wide request median are reproducibly faster — flagged for
re-baselining, not blunted thresholds.

**The dilation samples read as expected**, and are the reference points
for what each mode's placement produces on a quiet host:

- **~1.0x** — a perf-mode 1:1-pinned cell (near-ideal, the isolation
  contract holding).
- **~1.08x** — the steady perf schbench cell; **~1.13x** a near-1:1 wide
  no-perf cell; both undiluted.
- **~1.2x** — the per-phase-across-detach cell, where a mid-run scheduler
  swap widens the tail.

**The wide-topology finding, re-measured on the final code.** An interim
capture on wide (56 vCPU) near-1:1 no-perf cells showed the unconditional
FIFO-2 sensing threads pushing wakeup-p99 up ~40%. Re-measured at the
branch's final code state against the same baseline (release builds, N=6
per side), **that tail is gone**: wide wakeup-p99 lands at 0.983x the
baseline with a far tighter spread. What remains is a small, *constant*
cost in its place — wide wakeup-**p50** shifts from ~6.5 µs to ~15.8 µs
(+9.3 µs, spreads disjoint), the per-wakeup preemption charge of the RT
sensing threads spread uniformly instead of erratically. It surfaces only
at width (narrow wakeups are identical to the µs on both sides), only on
the wakeup median, and only in a mode whose contract already declines
timing fidelity: no-perf mode masks rather than pins, and its
oversubscription warning states plainly that under it *"timing metrics
are host artifacts."* Throughput is unaffected (rps +0.9%, loop count
+0.5%, in-envelope) and the wide request median is reproducibly ~8%
*faster*. A cross-commit absolute baseline on wide-cell wakeup-p50 should
be re-taken; a threshold test, which gates on metric presence, is
unaffected.

**The contention witness no longer scales its hot-path file reads with VM
width.** Earlier validation laddered the original per-tick O(vCPU) schedstat
sweep and found no measurable steady-schbench shift, but the shape was still
an avoidable observer cost on wide cells. The current path takes those sweeps
only at lifecycle transitions and uses one cumulative runner-cgroup
CPU-pressure read per Body monitor tick. That scope retains delayed-vCPU
coverage without charging pressure from unrelated host cgroups, and a
task-specific lifecycle schedstat cap removes noise from other cells sharing
the runner cgroup. If scoped PSI is unavailable, the same complete cap provides
a coarser whole-span fallback. The worker's per-checkpoint CPU-clock read
remains; its historical worst-case SpinWait cost and the original ladder are
retained in the validation record.

**The watchdog catches real wedges fast and leaves healthy cells alive.**
The injected-wedge fixtures
([`progress_watchdog_e2e.rs`](https://github.com/likewhatevs/ktstr/blob/main/tests/progress_watchdog_e2e.rs))
kill a spinning Teardown wedge on Tier-1 once its lone hot vCPU burns the
flat CPU budget (measured 16–35 s total, run-to-run wedge-start variance)
and a silent idle wedge on Tier-2 at the 15 s phase wall backstop
(measured ~19 s total) — both far below the ~150 s Tier-3 deadman, and at
1 vCPU and 64 vCPUs alike: the wide idle wedge, formerly bounded only by
the harness `terminate-after` because the old summed-CPU-trickle conjunct
stayed above any fixed floor at width, dies on the same backstop as the
narrow one (measured 19.2 s) now that Tier-2 carries no CPU term. No
watchdog tier fired on any of the healthy wide runs, and the
previously-false-killed 256-vCPU CI shapes survive their long idle
phases.

## Related

- [Performance Mode](performance-mode.md) — the full isolation contract,
  prerequisites, and failure modes.
- [Resource Budget](resource-budget.md) — the CPU budget, the cpuset
  sandbox, lock planning, and `ktstr locks`.
- [Checking](checking.md) — which checks are timing-sensitive and so want
  perf mode.
- [Monitor](../architecture/monitor.md) — the host-side observer the
  watchdog and dilation sampling build on.
