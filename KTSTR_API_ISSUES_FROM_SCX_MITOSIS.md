# ktstr API omissions and bugs found while writing scx_mitosis-steal-revamp reproducer

Authored by another claude (in scx repo) while debugging
`scheds/rust/scx_mitosis/tests/ktstr_mitosis_steal_tests.rs` against
ktstr 0.5.2 (crates.io published) plus `cargo-ktstr` / `ktstr` CLI
0.5.2. Running list — APPEND, don't rewrite. Each entry self-contained
with file:line citations into ktstr 0.5.2 source under
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/` and concrete
repro evidence from the scx_mitosis test runs.

## Bugs (correctness)

### B1 — `auto_repro` confusion on primary-VM init flake

**Symptom:** When the primary VM fails at init (e.g.
"send_sys_rdy retry budget exhausted (10 s)" from
ktstr/src/vmm/rust_init.rs), the auto-repro VM runs the SAME scenario
and reports `repro VM: scheduler ran normally (crash did not
reproduce)`. The test ouptut then suggests "the bug is gone" when in
reality the primary VM didn't get far enough to exercise the bug.

**Repro:** scheds/rust/scx_mitosis/tests/ktstr_mitosis_steal_tests.rs
with `topology = (1, 7, 9, 2)` (126 vCPUs). Three consecutive runs
(bgcu7mcax, bwoz9tajg, bjli04wz7) all hit "init script never started",
auto-repro all said "scheduler ran normally". Only when I reduced
topology to (1,7,2,2)=28 vCPUs did the primary VM boot and reveal that
the test scenario itself didn't fire the bug (b0belql6a).

**Fix idea:** When primary failure mode is `init script never started`,
auto-repro should label its "scheduler ran normally" verdict as
"PRIMARY DID NOT REACH WORKLOAD — auto-repro is not load-bearing" so
the user doesn't conclude the bug is absent.

### B2 — 10s `send_sys_rdy` budget is insufficient for >56 vCPU VMs under host contention

**Symptom:** With `topology = (1, 7, 9, 2)` (126 vCPUs) on a host
running other workloads (sccache, semcode-mcp, zellij — load avg ~10
on a 316-CPU box), the guest's ktstr-init never sends sys_rdy within
the 10s budget. Fails with `WARN ktstr::vmm::rust_init:
ktstr-init: send_sys_rdy retry budget exhausted (10 s)`. Reproduces
across multiple runs.

**Source:** ktstr/src/vmm/rust_init.rs send_sys_rdy retry budget. Hard-
coded 10s.

**Fix idea:** Scale with vCPU count (e.g., `max(10s, vcpus * 100ms)`),
or expose as a `ktstr_test` attribute. The implicit assumption that
10s suffices doesn't hold under no_perf_mode oversubscription.

### B3 — Existing test file ktstr_mitosis_tests.rs (from upstream PR #3569 WIP) does not compile against published ktstr 0.5.2

**Symptom:** PR #3569's test file uses `#[derive(ktstr::Scheduler)]`
and references `MITOSIS_PAYLOAD` const. Neither exists in ktstr 0.5.2
on crates.io. The derive lives only in ktstr-macros worktrees
(e.g. `~/opensource/stt/.claude/worktrees/agent-a153024e84ecaba8f/
ktstr-macros/src/lib.rs:1604`) but was apparently dropped before
0.5.2's release.

**Citation:** ktstr 0.5.2 publishes `pub use ktstr_macros::Payload;`
(src/lib.rs:590) but NO `Scheduler` re-export. ktstr-macros 0.5.2 has
no `proc_macro_derive(Scheduler` (grep is empty). worktree variants do
have it.

**Fix idea:** Either (a) re-publish the Scheduler derive in 0.5.x so
PR #3569's tests compile, or (b) add a clear migration note for users
who copy that pattern.

### B4 — `auto_repro` doesn't propagate the primary VM's failure-dump.json into the listed test stderr in a structured way

**Symptom:** When the bug fires (scx_bpf_error), the test panics with
"no test result received from guest". The actual scx_bpf_error string
is buried in the `--- sched_ext dump ---` section ~200 lines into
stderr (test run b0k5jog6x line 218). Finding the actionable line
requires scrolling through KVM masks, BSP run-loop traces, and sysctl
dumps.

**Fix idea:** Top-of-stderr "BUG SUMMARY:" line that extracts the
first `scx_bpf_error (path:line: msg)` from the SCX exit dump and
surfaces it before the noisy diagnostics.

### B5 — Test framework fails to surface "scheduler exited cleanly but produced no AssertResult" distinctly from "scheduler crashed"

**Symptom:** When scheduler exits early (e.g., apply_cell_config
returns -EINVAL → userspace bails cleanly), the test failure message
says "scheduler process died unexpectedly during workload" but exit
code is 1 (userspace bail) not a crash signal. Confusingly the run-loop
sentinel reports `code=-1` while the real exit code is 1.

**Fix idea:** Distinguish "scheduler clean-exit with non-zero code" vs
"scheduler signal/crash" in the failure summary.

## API omissions / pain points (request-for-comment)

### A1 — `--cell-parent-cgroup` is silently auto-injected based on `cgroup_parent` attribute

**Symptom:** I added `"--cell-parent-cgroup", "/ktstr"` to
`sched_args` (matching the documented `--cell-parent-cgroup` flag the
mitosis CLI exposes), but the test failed at scheduler launch with
`error: the argument '--cell-parent-cgroup <CELL_PARENT_CGROUP>'
cannot be used multiple times`.

**Source:** ktstr 0.5.2 src/test_support/runtime.rs:226 unconditionally
pushes `--cell-parent-cgroup {cgroup_parent}` if cgroup_parent is set.
Same path at runtime.rs:593. Plus the runtime.rs check at
src/test_support/args.rs:106 detects "passed via sched_args" but
doesn't dedup.

**Fix idea:** Either (a) skip the auto-inject if sched_args already
contains `--cell-parent-cgroup`, (b) make the auto-inject opt-in
(`#[scheduler(auto_cell_parent_cgroup = false)]`), or (c) document
the auto-inject in the declare_scheduler! docstring so users don't
copy it from the scheduler's --help output.

### A2 — `Op::RemoveCgroup` is rejected for Backdrop-owned cgroups with no documented workaround for "remove a long-lived cgroup mid-scenario"

**Symptom:** `Op::remove_cgroup("cg_b")` where cg_b was declared via
`Backdrop::new().with_cgroup(CgroupDef::named("cg_b")...)` fails with:
> step 2 failed: Op::RemoveCgroup targets Backdrop-owned cgroup
> 'cg_b' — Backdrop cgroups live for the full scenario and must not
> be removed from a Step; drop the op or move the cgroup declaration
> out of the Backdrop

**Root issue:** The framework's cgroup lifecycle has 2 categories
(Backdrop-persistent and step-local), neither of which supports
"create at scenario start with workers, then remove deterministically
at a specific step boundary". Step-local dies at step end
unconditionally; Backdrop can't be removed.

**Workaround I used:** Spawn a host thread that directly does
`std::fs::create_dir("/sys/fs/cgroup/ktstr/...")` and
`std::fs::remove_dir(...)` — bypasses the framework entirely. Works
but undermines the framework's encapsulation and races with its own
cgroup management.

**Fix idea:** Add an explicit "Removable persistent" cgroup category,
or document the host-thread workaround as a supported pattern, or
allow `Op::RemoveCgroup` to take a force flag for Backdrop cgroups
with a clear warning.

### A3 — No `Phase::AluHot` variant for `WorkType::Sequence`; can't compose AluHot with duty-cycle modulation

**Symptom:** Spec called for "AluHot at ~90% with modulation". The
intent is AluHot's widest-data-path SIMD arithmetic, modulated
between burst and sleep (e.g. for ~90% utilization with cadence).
Available primitives in ktstr 0.5.2:
- `WorkType::AluHot { width: AluWidth::Widest }` — pure compute, no
  modulation knob
- `WorkType::Sequence { first: Phase, rest: Vec<Phase> }` —
  Phase has `Spin`, `Sleep`, `Yield`, `Io`. NO `AluHot`.
- `WorkType::Bursty { burst_duration, sleep_duration }` — Spin-based,
  not AluHot

**Source:** src/workload/types/mod.rs Phase enum lines 21-36.

**Fix idea:** Add `Phase::AluHot { width: AluWidth, duration: Duration }`.
Or add a duty-cycle modifier to AluHot itself.

### A4 — `HoldSpec` does not derive `Copy`, causing E0382 in obvious-looking step-construction loops

**Symptom:** When constructing a `Vec<Step>` in a `for` loop reusing
the same `HoldSpec` value, `let hold = HoldSpec::Fixed(...);` and
then passing `hold` per iteration fails with:
> error[E0382]: use of moved value: `hold`
> = ... HoldSpec ... which does not implement the `Copy` trait

**Source:** src/scenario/ops/types.rs HoldSpec definition.

**Fix idea:** Derive `Copy` on HoldSpec (its variants are
`Fixed(Duration)`, `Frac(f64)`, `Loop { interval }` — all `Copy`).
Or alternatively give it a `.clone()`-free pattern via a const
constructor.

### A5 — Bool attributes on `#[ktstr_test(...)]` require explicit `= true`; bare attribute compiles to "expected `key = value`"

**Symptom:** Writing `#[ktstr_test(no_perf_mode, ...)]` (no value)
fails to compile:
> error: expected `key = value`
> --> tests/ktstr_mitosis_steal_tests.rs:60:5
> 60 |     no_perf_mode,
>    |     ^^^^^^^^^^^^

**Fix idea:** Either accept bare attribute as syntactic sugar for
`= true`, or update the docstring example in the attribute macro to
show `no_perf_mode = true` consistently.

### A6 — `#[derive(Scheduler)]` was removed without a deprecation period; published examples reference it

**Symptom:** ktstr 0.4.x had `pub use ktstr_macros::Scheduler;` (line
570 of ktstr 0.4.23 lib.rs) and a working `proc_macro_derive(Scheduler)`
in ktstr-macros 0.4.23. ktstr 0.5.2 published the macros crate
without the derive, but doesn't document the change. The scx_mitosis
PR #3569 test uses the derive and references `MITOSIS_PAYLOAD` which
the derive produced.

**Fix idea:** Either restore the derive or publish a migration note
showing the `declare_scheduler!` equivalent.

### A7 — No documented Backdrop cgroup-id ordering guarantee; figuring out which cgroup gets which cell_id required reading scheduler logs

**Symptom:** I needed to reason about cell_id allocation order to
construct a sparse cell_id range (cg_b in the middle so its removal
leaves a gap). The doc doesn't say "Backdrop creates cgroups in
declaration order" nor "cell_ids are sequential from the lowest
free". I had to read the run's scheduler log:
> Created cell 1 for cgroup /sys/fs/cgroup/ktstr/cg_a (cgid=56)
> Created cell 2 for cgroup /sys/fs/cgroup/ktstr/cg_b (cgid=82)
> ...
to deduce the ordering.

**Fix idea:** Document the Backdrop creation order and the
scheduler's cell_id allocation policy (lowest free) in the Backdrop
docstring.

### A8 — No `workers_pct(f64)` helper on `CgroupDef`

**Symptom:** To express "90% of cpuset capacity workers" I had to
manually compute `(36f64 * 0.9) as usize`. The cpuset size is
implicit in the topology + spec; the test author has to mirror the
framework's CPU-to-cpuset math.

**Fix idea:** `CgroupDef::workers_pct(0.9)` resolved at apply-setup
time against the resolved cpuset.

### A9 — `ctx` doesn't expose `cpuset_cpus(&CpusetSpec) -> usize`

**Symptom:** Same as A8 but more general. For `CpusetSpec::Llc(N)` or
`Disjoint{i,of}`, the count depends on the topology. Tests have to
hand-compute. A `ctx.cpuset_cpus(&spec)` would let tests express
worker count proportional to the cpuset without baking in topology
assumptions.

**Fix idea:** Add accessor to `Ctx`.

### A10 — Test name must contain `ktstr` to match the canonical `-E 'test(/ktstr/)'` filter — undocumented convention

**Symptom:** PR #3569's documented run command is
`cargo ktstr test --kernel ../linux --features ktstr-tests -E 'test(/ktstr/)'`.
The filter passes only tests whose name contains `ktstr`. My function
had to be named `ktstr_mitosis_steal_cpuset_churn` to match — but the
`#[ktstr_test]` attribute is what makes it a ktstr test, not the
name.

**Fix idea:** Either change the filter convention to match the
attribute (e.g., `cargo ktstr test --kind ktstr`) or document the
naming requirement loudly.

### A11 — `memory_mb` default of 2048 is too small for some valid topologies and isn't auto-scaled

**Symptom:** With topology=(1,7,9,2)=126 vCPUs + heavy workload, the
default 2048MB led to slow boot (contributed to init flake B2 above).
Bumping to 4096 helped marginally.

**Source:** ktstr-macros 0.5.2 src/lib.rs:127 `let mut memory_mb =
DEFAULT_MEMORY_MB;` and the default is 2048.

**Fix idea:** Scale memory_mb with vCPU count or document the
recommended floor for various topologies.

### A12 — `Op::add_cgroup` + `Op::set_cpuset` + worker spawn requires three separate ops; no `CgroupDef`-equivalent for "step-local with cpuset"

**Symptom:** To create a cgroup with a cpuset and workers mid-step,
you either:
- Use `Step::with_defs(vec![CgroupDef::named(...).with_cpuset(...)...])`
  — but that's the step's *setup*, not a mid-step op
- Use `Op::add_cgroup` then `Op::set_cpuset` then `Op::spawn` — three
  ops where one would do

**Fix idea:** Add `Op::add_cgroup_def(CgroupDef)` that accepts the
full builder.

### A13 — `Op::RemoveCgroup` error message could suggest the host-thread workaround

**Symptom:** See A2. The error message says "drop the op or move the
cgroup declaration out of the Backdrop" — but if the test author
actually NEEDS Backdrop-lifetime + mid-scenario removal, neither
option works. A note pointing to the host-thread mkdir/rmdir pattern
would have saved me hours.

### A14 — Failure-dump artifact path isn't surfaced in test stderr

**Symptom:** After a failure, I went looking for the
`failure-dump.json` artifact (per the tester's spec saying it lands
in `target/ktstr/<kernel>-<commit>/<test_name>.failure-dump.json`). I
ran `find ~/opensource/scx/target -name "*mitosis_steal*"` and got
nothing — the dump dir doesn't exist on my system. The dump content
IS available, but as part of the test stderr (the
`--- sched_ext dump ---` section), not as a separate file. Tester's
spec said files would be at that path; reality differs.

**Fix idea:** Either actually create the dump files at the documented
path, or update the docs to say "dump content is embedded in the
test stderr — no separate file is written".

### A15 — `cargo-ktstr` install pattern is documented but the dual-version mismatch (ktstr 0.4 on crates.io, 0.5 in local) is silent

**Symptom:** README.md says `[dev-dependencies] ktstr = { version = "0.4" }`.
But when local stt is at version 0.5.2 (unpublished at the time),
adding `ktstr = "0.4"` to scx_mitosis Cargo.toml resolves to 0.4.23
from crates.io — which has a DIFFERENT API than the local 0.5.2 you
might be referencing in worktree docs (e.g. the Scheduler derive). No
warning about the version skew.

**Fix idea:** Either (a) keep crates.io aligned with the local
workspace's version, (b) make API differences between 0.4 and 0.5
loud in the changelog, or (c) document which version to use against
which ktstr CLI version.

### A16 — No `Scheduler::with_assertion` / per-test failure-message override

**Symptom:** When the reproducer fires, the AssertResult chains
through 3 levels of `Caused by:` to surface the actual BPF error.
Test authors might want to attach a "expected error pattern" matcher
to confirm THIS is the bug being reproduced (vs some other bug). No
such matcher exists; the test author has to read stderr to confirm.

**Fix idea:** Add `Assert::expect_scx_bpf_error_matches(regex)` so
the reproducer can pin the exact bug.

---

## Next steps for the reading claude

Each entry above has a `Fix idea:` — pick one and implement. The Bugs
(B*) are correctness, fix them first. The API omissions (A*) are
ergonomics; pick by impact-to-fix ratio. Cross-reference with the
scx_mitosis fix PR (on `mitosis-steal-revamp` branch) for additional
context on the BUG 2 / BUG 3 scenarios the reproducer exercises.

If you find new issues while implementing, APPEND them here with a
new ID. Don't renumber existing entries.
