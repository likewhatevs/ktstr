# Troubleshooting

## Build errors

### clang not found

```text
error: failed to run custom build command for `ktstr`
  ...
  clang: No such file or directory
```

The BPF skeleton build (`libbpf-cargo`) invokes clang to compile
`.bpf.c` sources. Install clang:

- Debian/Ubuntu: `sudo apt install clang`
- Fedora: `sudo dnf install clang`

### pkg-config not found

```text
error: failed to run custom build command for `libbpf-sys`
  ...
  pkg-config: command not found
```

libbpf-sys uses pkg-config during its vendored build. Install it:

- Debian/Ubuntu: `sudo apt install pkg-config`
- Fedora: `sudo dnf install pkgconf`

### autotools errors (autoconf, autopoint, aclocal)

```text
autoreconf: command not found
aclocal: command not found
autopoint: command not found
```

The vendored libbpf-sys build compiles bundled libelf and zlib from
source using autotools. These libraries are not system dependencies
-- they ship with libbpf-sys -- but the autotools toolchain is
needed to build them. Install:

- Debian/Ubuntu: `sudo apt install autoconf autopoint flex bison gawk`
- Fedora: `sudo dnf install autoconf gettext-devel flex bison gawk`

### make or gcc not found

```text
busybox build requires 'make' — install build-essential (Debian/Ubuntu) or base-devel (Fedora/Arch)
busybox build requires 'gcc' — install build-essential (Debian/Ubuntu) or base-devel (Fedora/Arch)
```

The build script compiles busybox from source for guest shell mode.
This requires make and gcc.

- Debian/Ubuntu: `sudo apt install make gcc`
- Fedora: `sudo dnf install make gcc`

### BTF errors

```text
no BTF source found. Set KTSTR_KERNEL to a kernel build directory,
or ensure /sys/kernel/btf/vmlinux exists.
```

build.rs generates `vmlinux.h` from kernel BTF data. It searches
the kernel discovery chain (`KTSTR_KERNEL`, `./linux`, `../linux`,
installed kernel) for a `vmlinux` file, falling back to
`/sys/kernel/btf/vmlinux`. Most distros ship
`/sys/kernel/btf/vmlinux` with CONFIG_DEBUG_INFO_BTF enabled.

**Fixes:**

- Verify BTF is available: `ls /sys/kernel/btf/vmlinux`
- If missing, set `KTSTR_KERNEL` to a kernel build directory that
  contains a `vmlinux` with BTF:
  `export KTSTR_KERNEL=/path/to/linux`
- Build a kernel with `CONFIG_DEBUG_INFO_BTF=y`.
- Some minimal/cloud kernels strip BTF. Use a distro kernel or
  build your own.

### busybox download failure

```text
failed to obtain busybox source after 4 attempts.
  tarball (https://github.com/mirror/busybox/archive/refs/tags/1_36_1.tar.gz): ...
  Remediation:
    • Check network connectivity (the build script needs HTTPS access to github.com to fetch the upstream tarball).
    • If behind a proxy, ensure HTTP_PROXY/HTTPS_PROXY environment variables are set.
    • Or set KTSTR_BUSYBOX_TARBALL=<path> to point at a pre-fetched local copy.
    • Or set KTSTR_SKIP_BUSYBOX_BUILD=1 to skip the busybox compile entirely (shell mode will be unavailable).
```

build.rs downloads the busybox tarball on first build (4 attempts with
backoff). Subsequent builds use the cached binary in `$OUT_DIR`.

**Fixes:**

- Verify network connectivity to github.com.
- If behind a proxy, set `HTTP_PROXY` / `HTTPS_PROXY`.
- Set `KTSTR_BUSYBOX_TARBALL=<path>` to point at a pre-fetched local copy.
- Set `KTSTR_SKIP_BUSYBOX_BUILD=1` to skip the busybox compile entirely
  (shell mode will be unavailable).
- After a successful first build, no network access is needed
  unless `cargo clean` removes the cached binary.

## /dev/kvm not accessible

The host-side pre-flight emits one of the following, depending on
whether the device node is missing or merely unreadable:

```text
/dev/kvm not found. KVM requires:
  - Linux kernel with KVM support (CONFIG_KVM)
  - Access to /dev/kvm (check permissions or add user to 'kvm' group)
  - Hardware virtualization enabled in BIOS (VT-x/AMD-V)
```

```text
/dev/kvm: permission denied. Add your user to the 'kvm' group:
  sudo usermod -aG kvm $USER
  then log out and back in.
```

ktstr boots Linux kernels in KVM virtual machines. The host must have
KVM enabled and the user must have read+write access to `/dev/kvm`.

**Diagnose:**

- Check the device exists and inspect its permissions and owning group:
  `ls -l /dev/kvm`. Typical output: `crw-rw---- 1 root kvm 10, 232 ...`.
- Confirm the `kvm` group exists and see its members:
  `getent group kvm`.

**Fixes:**

- Load the KVM module: `modprobe kvm_intel` or `modprobe kvm_amd`.
- Follow the group-membership hint in the error text above (log out
  and back in afterward for the group change to take effect).
- On cloud VMs (GCP, AWS, Azure) or nested hypervisors, nested
  virtualization is typically off by default. Enable it per the
  provider's instructions (e.g. GCP `--enable-nested-virtualization`,
  AWS metal/`.metal` instance types, Azure Dv3/Ev3+ with nested virt).
- In CI, ensure the runner has KVM access (e.g. `runs-on: [self-hosted, kvm]`).

## No kernel found

```text
no kernel found — the test harness was likely invoked outside `cargo ktstr test` (which builds and injects a kernel automatically).
  hint: run `cargo ktstr test --kernel <path-or-version>` to drive this test, or set KTSTR_TEST_KERNEL=/path/to/{bzImage|Image} to point at a pre-built bootable image directly.
  hint: set KTSTR_KERNEL to one of: exact version (`6.14`), inclusive range (`6.14..7.0` or `6.14..=7.0`), git source (`git+URL#tag=NAME`, `git+URL#branch=NAME`, or `git+URL#sha=<40-hex>`), absolute or `~`-prefixed path, or cache key. List cached keys with `cargo ktstr kernel list`; build new ones with `cargo ktstr kernel build`
```

On aarch64 the first hint's image filename is `Image` instead of
`bzImage`.

`ktstr shell` and `cargo ktstr shell` auto-download the latest
stable kernel when no `--kernel` is specified and no kernel is found
via the discovery chain. See
[Kernel auto-download failures](#kernel-auto-download-failures) for
download-specific errors.

ktstr needs a bootable Linux kernel image (`bzImage` on x86_64,
`Image` on aarch64). See
[Kernel discovery](getting-started.md#kernel-discovery) for the
search order.

**Fixes:**

- Download and cache a kernel: `cargo ktstr kernel build`
- Build from a local tree: `cargo ktstr kernel build --kernel ../linux`
- Set `KTSTR_TEST_KERNEL` to an explicit image path.
- The host's installed kernel works for basic testing.

## Scheduler not found

```text
scheduler 'scx_mitosis' not found. Set KTSTR_SCHEDULER or
place it next to the test binary or in target/{debug,release}/
```

When using `SchedulerSpec::Discover`, ktstr searches for the scheduler
binary in:

1. `KTSTR_SCHEDULER` environment variable.
   - When the OPERATOR sets `KTSTR_CARGO_TEST_MODE=1` to mark a
     direct `cargo test` invocation that bypasses the `cargo ktstr`
     wrapper (e.g. `KTSTR_KERNEL=… KTSTR_CARGO_TEST_MODE=1 cargo
     test -- some_test`), `$PATH` is also consulted as part of the
     step-1 lookup — `which`-style, first match wins — so an
     `apt`-installed or `cargo install`-deployed scheduler binary
     resolves without requiring an in-tree build. The variable is
     NOT set automatically by `cargo ktstr test`; under the wrapper
     this step is just the literal `KTSTR_SCHEDULER` env var.
2. Sibling of the current executable (and, when the test binary
   lives under `target/{debug,release}/deps/`, the parent of
   `deps/` one level up — this covers the nextest / integration-
   test layout where the scheduler binary sits next to the test
   binary's parent).
3. `target/debug/`.
4. `target/release/`.
The ordering above (steps 1–4 first, on-demand build last) applies
only to a direct `cargo test` run marked with `KTSTR_CARGO_TEST_MODE=1`.
On the default `cargo ktstr test` (orchestrated) path the on-demand
build runs FIRST — right after the `KTSTR_SCHEDULER` env checks:

- `cargo build -p <scheduler>` runs before the sibling / target-dir
  probes, so an edited scheduler is never validated against a stale
  pre-built binary.
- If that build FAILS, ktstr REFUSES: the test hard-fails rather than
  falling back to a pre-built binary. Set
  `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK=1` to opt into the
  sibling → `target/{release,debug}/` fallback when the workspace
  build is momentarily broken.

So step 5 (the on-demand build after steps 2–4 miss) only runs on the
`KTSTR_CARGO_TEST_MODE` path; the orchestrated path builds up front.

**Fixes:**

- Build the scheduler first: `cargo build -p scx_mitosis` (on the
  default `cargo ktstr test` path this build already runs
  automatically before the pre-built probes, so pre-building only
  primes the cache to make the first test run faster).
- Set `KTSTR_SCHEDULER=/path/to/binary`.
- Use `SchedulerSpec::Path` for an explicit path in `#[ktstr_test]`.

## Scheduler died

```text
scheduler process died unexpectedly after completing step 2 of 5 (12.3s into test)
```

The scheduler process died while the scenario was running. This
is usually a crash. The exact message varies by when the crash was
detected (between steps, during workload, after completion).

The failure output contains diagnostic sections (each present only
when relevant):

- `--- scheduler log ---`: the scheduler's stdout and stderr,
  cycle-collapsed for readability.
- `--- diagnostics ---`: init stage classification, VM exit code,
  and the last 20 lines of kernel console output.
- `--- sched_ext dump ---`: `sched_ext_dump` trace lines from the
  guest kernel (present when a SysRq-D dump fired).

Set `RUST_BACKTRACE=1` to force `--- diagnostics ---` on all
failures, not just scheduler deaths.

**Next steps:**

- Check the `--- scheduler log ---` for the crash reason.
- Check `--- diagnostics ---` for BPF errors or kernel oops in
  the kernel console.
- Enable `auto_repro` in the test to capture the crash path with
  BPF probes. See [Auto-Repro](running-tests/auto-repro.md).
- Run with a longer duration and specific flags to narrow the
  reproducer.

See [Investigate a Crash](recipes/investigate-crash.md) for the
complete failure output format and auto-repro walkthrough.

## send_sys_rdy timeout

```text
WARN ktstr::vmm::rust_init: ktstr-init: send_sys_rdy failed within boot budget; see https://ktstr.dev/guide/troubleshooting.html#send_sys_rdy-timeout budget_ms=11200 vcpus=8 elapsed_ms=11342 port_exists=false kern_addrs_sent=false
```

The guest-side `ktstr-init` sends a `MSG_TYPE_SYS_RDY` TLV frame
through the virtio-console bulk port (`/dev/vport0p1`) after the
VM boots, signaling the host monitor that the guest is ready. The
retry loop waits for the port device via an inotify-evented watch
on `/dev` (guard-railed at a 1 s cadence cap, not a poll), bounded
by a wall-clock deadline; once the port appears, a failed send is
retried with a 100 ms throttle capped by the remaining budget. When the deadline expires the WARN above
fires, the guest continues running, and the host monitor falls
through to its `data_valid` gate to start sampling without the
SYS_RDY trigger; a late SYS_RDY arriving past the host's 5 s
pre-sample wait is harmless (the freeze coordinator's
`Option::take` makes the eventfd write fire-once). The test then
fails through the standard VM-teardown path once the harness gives
up — see [Scheduler died](#scheduler-died) for that failure
output.

### Diagnostic fields

- `port_exists=false` — the virtio-console multiport handshake
  never completed; `/dev/vport0p1` was never created by the
  kernel's virtio_console driver. **Common causes**: the guest
  kernel panicked before the driver probed (look for a kernel
  oops in the `--- diagnostics ---` console tail; see
  [Scheduler died](#scheduler-died) for the diagnostics-section
  layout); the host VMM never sent `PORT_ADD` on the control
  queue (host-side bug); or `VIRTIO_CONSOLE_F_MULTIPORT` was not
  negotiated, so the driver took the legacy single-console
  fallback and never created `/dev/vport0p1`.
- `port_exists=true, kern_addrs_sent=false` — the port device
  exists but writes did not go through. Either `writev` returned
  an error or 0, or `port_fops_write`'s `wait_port_writable`
  blocked on `host_connected` past the deadline. The kernel's
  wait has no timeout — a host that never sends `PORT_OPEN`
  blocks the loop indefinitely inside the syscall and this WARN
  never fires at all.
- `port_exists=true, kern_addrs_sent=true` — kern_addrs was
  delivered but the subsequent sys_rdy write failed or blocked
  past the deadline.
- `elapsed_ms` much larger than `budget_ms` — a blocking
  `writev` (kernel `wait_port_writable` on `host_connected`)
  consumed wall time beyond the budget. The deadline is checked
  only between iterations, so a blocked syscall can overshoot.

### Budget

The budget is a fixed 10 s base plus 150 ms per vCPU, capped at 90 s
— `10000 + vcpus*150`, clamped to 90000. The cap is first hit at 534
vCPUs, so every realistic topology (up to the 512-vCPU `MAX_VCPUS`)
gets its full additive budget. Worked examples:

| vCPUs   | budget_ms |
|---------|-----------|
| 1       | 10150 |
| 8       | 11200 |
| 67      | 20050 |
| 126     | 28900 |
| 256     | 48400 |
| 512     | 86800 |
| 534+    | 90000 (cap) |

On lightly-loaded hosts the 10 s base covers the boot path
comfortably. This is the GUEST-side `send_sys_rdy` budget and is
non-fatal — on exhaustion the guest WARNs and continues (the host
monitor's `data_valid` gate keeps reads safe). The authoritative
deadline is the HOST watchdog (`vm_timeout_from_entry`), which adds
this budget as boot headroom and additionally multiplies that headroom
by the host overcommit ratio (vCPUs / allowed host CPUs), so an
oversubscribed boot gets proportionally longer before the watchdog
fires.

### Fixes

**If `port_exists=false`** (slow / starved boot preventing the
driver from probing):

- Pass `--no-perf-mode` (or set `KTSTR_NO_PERF_MODE=1`) to
  disable RT scheduling and exclusive LLC reservation, which
  reduces the chance of host-side contention starving the
  guest's vCPU threads. See
  [Performance mode](concepts/performance-mode.md) for the full
  flag effect.
- Reduce the topology for the test (`llcs`, `cores`, `threads`)
  — fewer vCPUs means a shorter boot path.
- Reserve CPUs for ktstr via host-side isolation (`isolcpus=`)
  on the host kernel boot command line. See
  [Resource budget](concepts/resource-budget.md) for the
  host-side CPU isolation patterns ktstr expects.
- A KASAN / KCSAN / lockdep kernel build adds substantial
  boot-path overhead. Re-run on a non-instrumented kernel to
  rule out instrumentation cost vs. a real boot stall.

**If `port_exists=true`** (port device exists but the host-VMM
side is not accepting writes):

This is a host-VMM virtio-console state issue, not a guest CPU
contention issue — the `--no-perf-mode` / topology / isolcpus
fixes above do not apply. File a bug with the failure-output
dump (the `--- diagnostics ---` console tail + the WARN line +
the host-side log if available).

## Insufficient hugepages

```text
performance_mode: WARNING: no 2MB hugepages available, guest memory will use regular pages
```

```text
performance_mode: WARNING: need N 2MB hugepages, only K free — falling back to regular pages
```

[Performance mode](concepts/performance-mode.md) requests 2MB
hugepages for guest memory. The first form fires when no 2MB hugepages
are reserved on the host (`free == 0`); the second fires when some are
reserved but fewer than the run needs. In both cases the VM falls back
to regular pages and continues to boot.

**Fix:**

Allocate hugepages before the run:

```sh
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

## Worker assertion failures

```text
tid 2 stuck 4500ms on cpu2 at +3200ms (threshold 3000ms)
unfair cgroup: spread=42% (8-50%) 4 workers on 4 cpus (threshold 35%)
```

The `tid N` prefix on `stuck` names the thread the violation belongs
to so the operator can cross-reference with `--- timeline ---` and
`--- stats ---` sections, which key per-thread metrics by `tid`.
`unfair cgroup` is a per-cgroup assertion and has no `tid` prefix —
cross-reference with the per-cgroup spread / workers / cpus columns
in `--- stats ---` instead.

The Assert checks (`max_gap_ms`, `max_spread_pct`, etc.) detected a
worker metric outside the configured thresholds.

**Fixes:**

- Check whether the topology has enough CPUs for the scenario. Small
  topologies produce higher contention, larger gaps, and more spread.
- Use `execute_steps_with()` with a custom `Assert` to override
  thresholds for scenarios that need relaxed limits.
- Check the scheduler's behavior under the specific flag profile that
  triggered the failure.

## Cgroup name typos

A typo'd cgroup name surfaces only when an op tries to write to a
non-existent cgroup directory; the framework does not pre-validate
names against the tracking registry. The exact diagnostic depends on
which op references the typo:

- **`Op::RemoveCgroup` / `Op::StopCgroup` against a typo silently
  succeed** (rmdir / kill against a non-existent path are no-ops).
  The failure surfaces on the next op that touches the name.

- **`Op::SetCpuset` (writes `cpuset.cpus`)** falls through to the
  kernel's `ENOENT`, wrapped with `capture_cpuset_state` — a
  one-line snapshot of the parent's controller / subtree_control
  state and the child's listing:

  ```text
  cgroup-state-snapshot: parent=/sys/fs/cgroup/ktstr name=nonexistent parent.cgroup.controllers="cpuset cpu memory io pids" parent.cgroup.subtree_control="cpuset cpu memory" child.cgroup.controllers="<read failed: No such file or directory (os error 2)>" child.cpuset.cpus.exists=false child.listing=<read_dir failed: No such file or directory (os error 2)>: No such file or directory (os error 2)
  ```

  The `child.listing=<read_dir failed: ...>` segment is the operator
  tell: a typo'd name has no directory to list, distinguishing this
  from the "cgroup exists but the cpuset.cpus write was rejected"
  case (where the listing would enumerate the cgroupfs knobs).

- **Other setters (`cpu.max`, `memory.max`, `memory.swap.max`,
  `cpuset.mems`, etc.) against a typo** produce the wrapped
  `+controller in parent cgroup.subtree_control` form documented
  below in [Cgroup controller not enabled](#cgroup-controller-not-enabled).
  The same wrap fires for both the "cgroup directory exists but the
  parent's subtree_control lacks the controller" and the "cgroup
  directory does not exist" cases — distinguish by checking whether
  the directory itself is present.

- **`Op::AddCgroup` against a name that's already tracked** (a typo
  that happens to collide with an existing cgroup) bails with:

  ```text
  Op::AddCgroup 'cg_0' collides with a cgroup already tracked (by a prior Backdrop or step-local CgroupDef) — declare it in exactly one place; use a fresh name for the step-local cgroup
  ```

**Fixes:**

- Verify the cgroup name matches the `name` in `Op::AddCgroup`,
  `CgroupDef::named()`, or the `Backdrop.cgroups` declaration.
- When using dynamic cgroup names (e.g. `format!("cg_{i}")`), ensure
  the same formatting is used in all ops referencing that cgroup.
- For `Op::SetCpuset`, check the `child.listing` field of the
  snapshot to decide whether the cgroup directory exists at all (the
  `<read_dir failed: ENOENT>` shape rules out the controller-config
  hypothesis).

## Cgroup controller not enabled

```text
cgroup 'cg_0': set cpu.max='100000 100000' (requires +cpu in parent cgroup.subtree_control): No such file or directory (os error 2)
cgroup 'cg_0': set memory.max='4294967296' (requires +memory in parent cgroup.subtree_control): No such file or directory (os error 2)
cgroup 'cg_0': set memory.swap.max='1073741824' (requires +memory in parent cgroup.subtree_control; file absent on CONFIG_SWAP=n kernels): No such file or directory (os error 2)
cgroup 'cg_0': set cpuset.mems='0-1' (requires +cpuset in parent cgroup.subtree_control): No such file or directory (os error 2)
```

The cgroup exists but the controller knob is missing from
`/sys/fs/cgroup/<parent>/<name>/`. ktstr's `setup()` auto-enables
the controllers it detects on the scenario's `CgroupDef` / `Op` set,
so a missing controller means either:

- the scenario declared a knob (e.g. `memory_max`) but the
  framework's detection did not see it (file a bug — the detector
  walks every `CgroupDef` field);
- an outer parent (systemd `user.slice`, container runtime) stripped
  controllers from this subtree before ktstr ran;
- the kernel was built without `CONFIG_SWAP` (the `memory.swap.max`
  wrapping spells this out explicitly).

**Diagnostic command:**

```sh
cat /sys/fs/cgroup/<parent>/cgroup.subtree_control
```

The output enumerates the controllers the parent forwards to its
children. A controller present in the wrapped error must appear in
this list; if it does not, fix the parent first (`echo '+memory' >
.../cgroup.subtree_control` from a sufficiently-privileged shell) or
remove the knob from the scenario.

## CpusetSpec errors

```text
cgroup 'cg_0': CpusetSpec validation failed: not enough usable CPUs (4) for 8 partitions
cgroup 'cg_1': CpusetSpec validation failed: index 3 >= partition count 3
cgroup 'cg_2': CpusetSpec validation failed: Range fracs must lie in [0.0, 1.0]: start_frac=-1, end_frac=0.5
```

A `CpusetSpec` cannot produce a valid cpuset for the test topology.
`execute_steps` treats this as a hard error and aborts the step so the
downstream slicing/arithmetic in `CpusetSpec::resolve` is never reached
with inputs that would panic.

**Fixes:**

- Guard with a topology check before creating the step:
  `if ctx.topo.usable_cpus().len() < needed { return Ok(AssertResult::skip(...)); }`
- Call `CpusetSpec::validate(&ctx)` in your scenario builder so failures
  surface before `execute_steps` runs.
- Reduce the partition count or use `CpusetSpec::Llc` instead of
  `Disjoint` on topologies with fewer CPUs than partitions.
- For `Range`/`Overlap`, keep fractions finite and inside `[0.0, 1.0]`;
  `Range` additionally requires `start_frac < end_frac`.

## Worker count mismatches

```text
PipeIo (group 0) requires num_workers divisible by 2, got 3
```

Grouped work types (`PipeIo`, `FutexPingPong`, `CachePipe`,
`FutexFanOut`, `FanOutCompute`, plus the contention / waker
families documented at
[WorkloadHandle: spawning](architecture/workload-handle.md#spawning))
require `num_workers` divisible by their group size.
`WorkType::worker_group_size()` returns the divisor. The `(group N)`
segment names the composed[] entry the violation belongs to —
multi-group scenarios surface `(group 0)`, `(group 1)`, … so you
know which composed entry to fix.

**Fixes:**

- Set `CgroupDef::workers(n)` to a value divisible by the work
  type's group size (2 for pipe/futex pairs, `fan_out + 1` for
  FutexFanOut and FanOutCompute).
- Use an ungrouped work type (`SpinWait`, `Mixed`, `Bursty`,
  `IoSyncWrite`, `IoRandRead`, `IoConvoy`, `YieldHeavy`) if worker
  count flexibility is needed.

## Cache corruption

```text
  6.14.2-tarball-x86_64-kc...                 (corrupt: metadata.json malformed: ...)
warning: entries marked (corrupt) cannot be used — cached metadata is missing, malformed, or references a missing image. Inspect the entry directory under ~/.cache/ktstr/kernels to remove it manually, or run `kernel clean --corrupt-only --force` which removes ONLY corrupt entries and leaves valid ones intact. ...
```

A cached kernel entry has missing, unparseable, or
schema-drifted `metadata.json`, or metadata that references an
image file that is no longer present. This can happen after a
partial write (e.g. disk full, killed process), or after a ktstr
release that evolved the metadata schema in a
non-backward-compatible way. `cargo ktstr kernel list` surfaces
these as `(corrupt: ...)` rows; the trailing footer on stderr
summarizes the remediation options. `CacheDir::lookup` returns
`None` for corrupt entries so test runs at a specific cache key
fall through to the normal re-build path.

The JSON form (`cargo ktstr kernel list --json`) emits an
`error_kind` field on every corrupt entry — one of `"missing"`,
`"unreadable"`, `"schema_drift"`, `"malformed"`, `"truncated"`,
`"parse_error"`, `"image_missing"`, or `"unknown"` — so CI
scripts can dispatch on a stable token without parsing the
free-form `error` string.

**Fixes:**

- Remove ONLY corrupt entries (keeps valid ones intact):
  `cargo ktstr kernel clean --corrupt-only --force`
- Remove the corrupt entry along with everything else:
  `cargo ktstr kernel clean --force`
- Rebuild a specific version after cleanup: `cargo ktstr kernel build --force --kernel 6.14.2`
- Override the cache directory via `KTSTR_CACHE_DIR` if the default
  location is on a problematic filesystem.
- See [`cargo ktstr kernel clean`](running-tests/cargo-ktstr.md#kernel-clean)
  for all cleanup options, including `--keep N --force` to preserve
  the N newest entries.

## Stale `vmlinux.btf` or `default.profraw` in kernel source tree

After upgrading from an older ktstr version, you may notice extra
files in your kernel source directory:

- `<source>/vmlinux.btf` — a sidecar of the kernel's `.BTF`
  section bytes. Older ktstr versions wrote it next to whichever
  `vmlinux` they parsed, including source-tree builds. Current
  ktstr only writes the sidecar when the vmlinux path is inside
  the cache root (`~/.cache/ktstr/kernels/` or whatever
  `KTSTR_CACHE_DIR` points at) so source trees stay pristine.
- `<source>/default.profraw` — an LLVM coverage runtime artifact.
  Older ktstr versions could leave it in cwd when a
  coverage-instrumented `cargo ktstr test` was launched from
  inside the kernel tree. Current ktstr injects
  `LLVM_PROFILE_FILE=<cargo-ktstr-binary-parent>/llvm-cov-target/default-%p-%m.profraw`
  (LLVM's `%p` = process id, `%m` = module signature)
  for the bare `nextest` path so the profraw lands next to the
  cargo-ktstr binary regardless of cwd. See
  [profraw layout](running-tests/cargo-ktstr.md#profraw-layout)
  for the per-population directory map.

Both files are leftover state from prior runs and are safe to
remove:

```sh
rm -f /path/to/linux/vmlinux.btf
rm -f /path/to/linux/default.profraw
```

If you also see them turn up under a different ktstr-driven
source tree, check that you are running a current ktstr build
(re-run `cargo build` or `cargo install ktstr` to pick up the
fix) before deleting again — the guards live in the resolver,
not on disk, so an old binary will keep regenerating these
files.

## Cache directory not found

```text
HOME is unset; cannot resolve cache directory. The container init or login shell did not assign HOME — set it to an absolute path, or set KTSTR_CACHE_DIR to an absolute path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a cache location explicitly.
```

```text
HOME is set to the empty string; cannot resolve cache directory. An empty HOME usually means a Dockerfile or shell rc has `export HOME=` or `ENV HOME=` with no value. Either set HOME to a real absolute path, or set KTSTR_CACHE_DIR to an absolute path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a cache location explicitly.
```

The kernel image cache requires a writable directory. ktstr resolves
it as: `KTSTR_CACHE_DIR` > `$XDG_CACHE_HOME/ktstr/kernels/` >
`$HOME/.cache/ktstr/kernels/`. The first form fires when `HOME` is
absent from the environment (typical of bare container inits or
systemd units with no `Environment=HOME=...`); the second fires when
`HOME` is present but assigned to the empty string.

**Fix:** Set `KTSTR_CACHE_DIR` to an explicit path, or ensure `HOME`
is set to a real absolute path.

## Stale kconfig

```text
warning: entries marked (stale kconfig) were built against a different ktstr.kconfig. Rebuild with: kernel build --force --kernel <entry version> (add --extra-kconfig PATH if the entry also carries the (extra kconfig) tag).
```

`cargo ktstr kernel list` marks entries whose stored `ktstr_kconfig_hash`
differs from the current embedded `ktstr.kconfig` fragment. This
happens after updating ktstr (which may change the kconfig fragment).

**Fix:**

Rebuilds happen automatically on the next `cargo ktstr kernel build`
for stale entries. Use `--force` to override the cache for other
reasons. See [`cargo ktstr kernel list`](running-tests/cargo-ktstr.md#kernel-list)
for the full listing output.

## Kernel auto-download failures

```text
ktstr: no kernel found, downloading latest stable
fetch https://www.kernel.org/releases.json: <error>
```

ktstr auto-downloads a kernel when no `--kernel` is specified and no
kernel is found via the discovery chain (see
[Kernel discovery](getting-started.md#kernel-discovery)). The same
download path runs when `--kernel` specifies a version (e.g.
`--kernel 6.14.2`) that is not in the cache. The CLI label varies:
`ktstr:` for the standalone binary, `cargo ktstr:` for the cargo
subcommand.

The `<error>` above is the underlying reqwest error (DNS resolution,
connection refused, timeout, TLS handshake failure).

```text
fetch https://www.kernel.org/releases.json: HTTP 503
```

kernel.org returned a non-success status code.

```text
no stable kernel with patch >= 8 found in releases.json
```

ktstr requires a stable or longterm release with patch version >= 8
to avoid brand-new major versions that may have build issues. This
error means releases.json contained no qualifying version.

```text
download https://cdn.kernel.org/.../linux-6.14.10.tar.xz: <error>
```

Network failure during tarball download (same causes as above).

```text
extract tarball: <error>
```

Tarball extraction failed. Common causes: disk full, insufficient
permissions on the temp directory, or a truncated download.

```text
kernel built but cache store failed — cannot return image from temporary directory
```

The kernel built successfully but could not be stored in the cache.
Check disk space and permissions on the cache directory.

For version-specific download errors (HTTP 404, HTML responses), see
[Kernel download failures](#kernel-download-failures).

**Fixes:**

- Verify network connectivity: `curl -sI https://www.kernel.org/releases.json`
- Check DNS resolution for kernel.org and cdn.kernel.org.
- Check disk space — the download, extraction, and build require
  significant disk space.
- If behind a proxy, set `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`
  (reqwest respects these environment variables).
- Override the cache directory via `KTSTR_CACHE_DIR` if the default
  location has insufficient space or permissions.
- Pre-download a kernel explicitly: `cargo ktstr kernel build --kernel 6.14.10`
  to isolate whether the failure is in version resolution or download.

## Kernel download failures

These errors occur when `cargo ktstr kernel build` or `--kernel`
specifies an explicit version. For network and extraction errors
during auto-download, see
[Kernel auto-download failures](#kernel-auto-download-failures).

```text
version 6.14.22 not found. latest 6.14.x: 6.14.10
```

The requested version does not exist on kernel.org. When a version in
the same major.minor series is available in releases.json, the error
suggests it.

```text
version 5.4.99 not found
```

When the series is EOL or not in releases.json, only the "not found"
message appears (no suggestion).

```text
RC tarball not found: https://git.kernel.org/torvalds/t/linux-6.15-rc3.tar.gz
  RC releases are removed from git.kernel.org after the stable version ships.
```

RC tarballs are removed from git.kernel.org after the stable version
ships. Use `--kernel git+URL#tag=NAME` with a git.kernel.org URL to clone the tag instead.

```text
download ...: server returned HTML instead of tarball (URL may be invalid)
```

Some CDN error pages return HTTP 200 with `text/html` content type.
The download rejects these responses.

**Fixes:**

- Check the suggested version in the error message.
- Verify the version exists: check
  `https://www.kernel.org/releases.json` for available versions.
- For RC releases, use `--kernel git+URL#tag=NAME` with a
  git.kernel.org URL instead of a tarball download.
- Run `cargo ktstr kernel build` without a version to automatically
  fetch the latest stable.

## Shell mode issues

### stdin must be a terminal

```text
stdin must be a terminal for interactive shell mode
```

`cargo ktstr shell` requires a terminal for bidirectional I/O
forwarding. Piped or redirected stdin is rejected.

**Fix:** Run from an interactive terminal session.

### include file not found

```text
-i strace: not found in filesystem or PATH
```

Bare names (without `/`, `.`, or `..`) are searched in `PATH`. If the
binary is not in `PATH`, use an explicit path.

```text
--include-files path not found: ./missing-file
```

Explicit paths (containing `/` or starting with `.`) must exist on
disk.

**Fix:** Verify the file exists and use the correct path.

### include directory contains no files

```text
warning: -i ./empty-dir: directory contains no regular files
```

The directory passed to `--include-files` was walked recursively but
contained no regular files. FIFOs, device nodes, and sockets are
skipped during the walk.

**Fix:** Verify the directory contains the files you expect.

## Flock timeout / NFS rejection

```text
flock LOCK_EX on run-dir target/ktstr/6.14-abc1234 timed out after
30s (lockfile target/ktstr/.locks/6.14-abc1234.lock, holders:
  pid=12345 cmd=cargo-ktstr test --kernel 6.14). A peer cargo
ktstr test process is writing sidecars to the same
{kernel}-{project_commit} directory; wait for it to finish or kill
it, then retry.
```

A peer process is holding the per-run-key advisory `flock(2)`
that serializes sidecar writes; the helper polled for 30 s and
gave up. Run-dir locks live at
`{runs_root}/.locks/{kernel}-{project_commit}.lock` and serialize
the (pre-clear + write) cycle so two concurrent ktstr runs
sharing the same key can't tear partially-written sidecars.

```text
target/ktstr/.locks/6.14-abc1234.lock: filesystem NFS is not
supported for ktstr lockfiles (NFSv3 is advisory-only without
an NLM peer; NFSv4 byte-range locking does not cover flock(2)).
Move the lockfile path to a local filesystem (tmpfs, ext4, xfs,
btrfs, f2fs, bcachefs).
```

`try_flock` rejects NFS, CIFS, SMB2, CephFS, AFS, and FUSE mounts
because `flock(2)` semantics on those filesystems are unreliable
(see [Resource Budget — Filesystem requirement](concepts/resource-budget.md#filesystem-requirement)
for the per-filesystem rationale).

**Diagnose:**

- `cargo ktstr locks` (or `ktstr locks --watch 1s`) prints every
  ktstr flock currently held on the host with PID + cmdline,
  including per-run-key sidecar locks under the "Run-dir locks"
  section (see
  [`cargo ktstr locks`](running-tests/cargo-ktstr.md#locks)).
- `cat /proc/locks | grep '<lockfile-path-from-error>'` falls
  back to the kernel's own flock enumeration when the holder is
  outside ktstr.
- `stat -f -c '%T' <runs-root>` reports the filesystem type when
  the rejection error names NFS/CIFS/SMB/CephFS/AFS/FUSE.

**Fix:**

- For a peer-holder timeout: wait for the peer to finish, kill
  it (`kill <pid>` from the holder list), or retry with the peer
  done.
- For an NFS / remote-fs rejection: relocate the runs root to a
  local filesystem. Set `KTSTR_SIDECAR_DIR` to a local path
  (`/tmp/ktstr-sidecars`, a tmpfs mount) — note that this
  override path **also skips the cross-process flock**, so
  concurrent runs targeting the same `KTSTR_SIDECAR_DIR` have no
  serialization between them. Use the override only for a
  single-process run or per-process distinct paths.
- The kernel cache's lockfiles
  (`{cache_root}/.locks/*.lock`) face the same constraint —
  override `KTSTR_CACHE_DIR` to a local filesystem if the default
  resolves to NFS. See
  [Cache directory not found](#cache-directory-not-found).

## Tests pass locally but fail in CI

Common causes:

- **No KVM**: CI runners need hardware virtualization. Check for
  `/dev/kvm` access.
- **Fewer CPUs**: gauntlet topology presets up to 252 CPUs may
  exceed the runner's capacity. Use smaller topologies.
- **No kernel**: set `KTSTR_TEST_KERNEL` in the CI environment.
- **No CAP_SYS_NICE or rtprio**: performance-mode tests require
  `CAP_SYS_NICE` or an rtprio limit for RT scheduling, and enough
  host CPUs for exclusive LLC reservation. Pass `--no-perf-mode`
  (or set `KTSTR_NO_PERF_MODE=1`) to disable all performance mode
  features. Tests with `performance_mode=true` are skipped entirely
  under `--no-perf-mode`.
- **Debug thresholds**: CI often runs debug builds. Debug builds use
  relaxed thresholds (3000ms gap, 35% spread) but may still hit
  limits on slow runners. See
  [default thresholds](concepts/checking.md#default-thresholds).
