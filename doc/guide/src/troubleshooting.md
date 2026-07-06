# Troubleshooting

Most failures fall into one of three buckets: host setup, kernel or
scheduler discovery, and runtime verdicts from the VM. Match the exact
message first, then use the linked section for the smallest fix.

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Host setup</strong><p>Toolchain packages, <code>/dev/kvm</code>, hugepages, cgroups, flock, and cache state.</p></div>
<div class="kt-doc-card"><strong>Kernel + scheduler</strong><p>Kernel resolution, BTF mismatch, missing <code>scx_*</code> binaries, verifier rejection, and load errors.</p></div>
<div class="kt-doc-card"><strong>Runtime verdicts</strong><p>Worker checks, stuck-task observations, dumps, replay commands, and CI-only flakes.</p></div>
</div>

Find your error message, jump to its section:

| You see | Go to |
|---|---|
| `clang: No such file or directory` | [Build errors](#build-errors) |
| `pkg-config: command not found` | [Build errors](#build-errors) |
| `autoreconf: command not found` | [Build errors](#build-errors) |
| `busybox build requires 'make'` | [Build errors](#build-errors) |
| `no BTF source found` | [BTF errors](#btf-errors) |
| `failed to obtain busybox source` | [busybox download failure](#busybox-download-failure) |
| `/dev/kvm not found` / `permission denied` | [/dev/kvm not accessible](#devkvm-not-accessible) |
| `no kernel found` | [No kernel found](#no-kernel-found) |
| `scheduler 'NAME' not found` | [Scheduler not found](#scheduler-not-found) |
| `scheduler process died unexpectedly` | [Scheduler died](#scheduler-died) |
| `scheduler did not turn on` + verifier log | [Scheduler fails the BPF verifier](#scheduler-fails-the-bpf-verifier) |
| `libbpf: … func_proto … incompatible with vmlinux` | [Scheduler cannot load: kfunc BTF mismatch](#scheduler-cannot-load-kfunc-btf-mismatch) |
| `send_sys_rdy failed within boot budget` | [send_sys_rdy timeout](#send_sys_rdy-timeout) |
| `no 2MB hugepages available` | [Insufficient hugepages](#insufficient-hugepages) |
| `tid N stuck …` / `unfair cgroup: spread=…` | [Worker assertion failures](#worker-assertion-failures) |
| `cgroup-state-snapshot: …` | [Cgroup name typos](#cgroup-name-typos) |
| `requires +cpu in parent cgroup.subtree_control` | [Cgroup controller not enabled](#cgroup-controller-not-enabled) |
| `CpusetSpec validation failed` | [CpusetSpec errors](#cpusetspec-errors) |
| `requires num_workers divisible by` | [Worker count mismatches](#worker-count-mismatches) |
| `(corrupt: metadata.json malformed…)` | [Cache corruption](#cache-corruption) |
| `HOME is unset; cannot resolve cache directory` | [Cache directory not found](#cache-directory-not-found) |
| `entries marked (stale kconfig)` | [Stale kconfig](#stale-kconfig) |
| `fetch https://www.kernel.org/releases.json: …` | [Kernel auto-download failures](#kernel-auto-download-failures) |
| `version X not found` / `RC tarball not found` | [Kernel download failures](#kernel-download-failures) |
| `stdin must be a terminal` / `-i NAME: not found` | [Shell mode issues](#shell-mode-issues) |
| `flock LOCK_EX … timed out` / `filesystem NFS is not supported` | [Flock timeout / NFS rejection](#flock-timeout--nfs-rejection) |
| test marked SLOW, then killed by nextest | [Test hangs / nextest timeout](#test-hangs--nextest-timeout) |
| green locally, red in CI | [Tests pass locally but fail in CI](#tests-pass-locally-but-fail-in-ci) |

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

build.rs downloads the busybox tarball on first build (4 attempts
with backoff); subsequent builds use the cached binary. Follow the
remediation lines in the error itself — after one successful build,
no network access is needed unless `cargo clean` removes the cached
binary.

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

- `ls -l /dev/kvm` — typical output: `crw-rw---- 1 root kvm 10, 232 ...`.
- `getent group kvm` — confirm the group exists and see its members.

**Fixes:**

- Load the KVM module: `modprobe kvm_intel` or `modprobe kvm_amd`.
- Follow the group-membership hint in the error text (log out and
  back in afterward).
- On cloud VMs (GCP, AWS, Azure) or nested hypervisors, nested
  virtualization is typically off by default. Enable it per the
  provider's instructions (e.g. GCP `--enable-nested-virtualization`,
  AWS `.metal` instance types, Azure Dv3/Ev3+ with nested virt).
- In CI, ensure the runner has KVM access — see [CI](ci.md).

## No kernel found

```text
no kernel found — the test harness was likely invoked outside `cargo ktstr test` (which builds and injects a kernel automatically).
  hint: run `cargo ktstr test --kernel <path-or-version>` to drive this test, or set KTSTR_TEST_KERNEL=/path/to/{bzImage|Image} to point at a pre-built bootable image directly.
  hint: set KTSTR_KERNEL to one of: exact version (`6.14`), inclusive range (`6.14..7.0` or `6.14..=7.0`), git source (`git+URL#tag=NAME`, `git+URL#branch=NAME`, or `git+URL#sha=<40-hex>`), absolute or `~`-prefixed path, or cache key. List cached keys with `cargo ktstr kernel list`; build new ones with `cargo ktstr kernel build`
```

On aarch64 the first hint's image filename is `Image` instead of
`bzImage`. ktstr needs a bootable kernel image; see
[cargo ktstr kernel](running-tests/cargo-ktstr.md) for the discovery
chain. `ktstr shell` and `cargo ktstr shell` auto-download the latest
stable kernel when nothing is found — see
[Kernel auto-download failures](#kernel-auto-download-failures) for
download-specific errors.

**Fixes:**

- Download and cache a kernel: `cargo ktstr kernel build`
- Build from a local tree: `cargo ktstr kernel build --kernel ../linux`
- Set `KTSTR_TEST_KERNEL` to an explicit image path.
- The host's installed kernel works for basic testing.

## Scheduler not found

```text
scheduler 'scx_my_sched' not found. Set KTSTR_SCHEDULER or
place it next to the test binary or in target/{debug,release}/
```

`SchedulerSpec::Discover` resolves the scheduler binary entirely on
the host. The order depends on how the test was launched:

**Under `cargo ktstr test` (the normal path):**

1. `KTSTR_SCHEDULER_BIN_<NAME>`, then `KTSTR_SCHEDULER` env overrides.
2. `cargo build -p <scheduler>` — the build runs up front, so an
   edited scheduler is never validated against a stale pre-built
   binary. If that build fails, the test hard-fails rather than
   falling back; set `KTSTR_SCHEDULER_ALLOW_STALE_FALLBACK=1` to
   re-enable the sibling / `target/{debug,release}/` pre-built
   fallback while the workspace build is broken.

**Under bare `cargo test` / `cargo nextest run` (marked with
`KTSTR_CARGO_TEST_MODE=1`):**

1. The env overrides, with `$PATH` also consulted — so an installed
   scheduler binary resolves without an in-tree build.
2. Sibling of the test binary, then the `target/release/` and
   `target/debug/` build dirs — the scheduler's build profile
   (release by default) is probed first.
3. The on-demand `cargo build -p <scheduler>` runs last, only after
   the pre-built probes miss.

**Fixes:**

- `cargo build -p scx_my_sched` — on the orchestrated path this only
  primes the cache; on the bare path it makes the probe hit.
- Set `KTSTR_SCHEDULER=/path/to/binary` (or the per-name
  `KTSTR_SCHEDULER_BIN_<NAME>` variant).
- Use `SchedulerSpec::Path` for an explicit path.

## Scheduler died

```text
scheduler process died unexpectedly after completing step 2 of 5 (12.3s into test)
```

The scheduler process died while the scenario was running — usually a
crash. The exact message varies by when the crash was detected. The
failure output contains diagnostic sections (each present only when
relevant): `--- scheduler log ---` (the scheduler's own output,
cycle-collapsed), `--- diagnostics ---` (init stage, VM exit code,
kernel console tail), and `--- sched_ext dump ---` (when a SysRq-D
dump fired). Set `RUST_BACKTRACE=1` to force `--- diagnostics ---` on
all failures.

**Next steps:**

- Read the `--- scheduler log ---` for the crash reason; see
  [Reading Failure Output](running-tests/failures.md) for the full
  section-by-section anatomy.
- A second VM automatically reproduces the crash with BPF probes
  attached — see [Auto-Repro](running-tests/auto-repro.md).
- Follow [Investigate a Crash](recipes/investigate-crash.md) for the
  crash-to-pin workflow.

## Scheduler fails the BPF verifier

```text
verifier
  scheduler: NOT ATTACHED — scheduler process exited during BPF load/startup

verifier --- verifier stats ---
  processed=186  states=7/7

verifier --- scheduler log ---
Global function ktstr_dispatch() doesn't return scalar. Only those are supported.
0: R1=ctx() R10=fp0
; if (crash) @ main.bpf.c:423
0: (18) r1 = 0xff5d3bb3000f60dc       ; R1=map_value(map=bpf_bpf.bss,ks=4,vs=280,off=220)
...
; *p = (int)acc; @ main.bpf.c:464
191: (61) r2 = *(u32 *)(r10 -8)       ; R2=scalar(id=53,smin=0,smax=umax=0xffffffff,var_off=(0x0; 0xffffffff)) R10=fp0 fp-8=mmmmscalar(id=53,smin=0,smax=umax=0xffffffff,var_off=(0x0; 0xffffffff))
192: (63) *(u32 *)(r1 +0) = r2
R1 invalid mem access 'scalar'
processed 186 insns (limit 1000000) max_states_per_insn 0 total_states 7 peak_states 7 mark_read 0
```


<!-- captured: cargo ktstr verifier --kernel 7.0 (fixture scheduler with --verify-loop rejection knob) | ktstr 0.23.0 | kernel 7.0.14 -->
The in-guest BPF verifier rejected the program, so the scheduler
never attached. Read the log bottom-up: the last few lines name the
rejected instruction (`R1 invalid mem access 'scalar'`) and the
source line the C-line comments (`@ main.bpf.c:464`) map it to. The
first line is the verifier's summary of the top-level complaint.

Verifier acceptance depends on kernel version *and* topology —
values like `nr_cpus` bake into `.rodata`, so a program that
verifies on one CPU count can blow up on another. Sweep your
scheduler across kernels and topologies with
[`cargo ktstr verifier`](running-tests/verifier.md), which also
collapses repeated loop iterations (`--- N identical iterations
omitted ---`) so real rejections stay readable.

## Scheduler cannot load: kfunc BTF mismatch

```text
--- scheduler log ---
libbpf: extern (func ksym) 'scx_bpf_create_dsq': func_proto [755] incompatible with vmlinux [54769]
libbpf: failed to load BPF skeleton 'bpf_bpf': -EINVAL
Error: Failed to load BPF program
```


<!-- captured: cargo ktstr test (scx-ktstr, scenario_coverage) | ktstr 0.23.0 | kernel: local git build c5d2724 (7.1 merge window, reports 7.0.0) -->
ktstr surfaces this as `scheduler did not turn on — scheduler process
exited during BPF load/startup` in verifier cells, or as a scheduler
death / `no test result received from guest` in test runs — with the
libbpf lines above in the scheduler log.

The cause is the kernel image, not your scheduler. Newer kernels
(first released in v7.1) give scx kfuncs an implicit trailing
`struct bpf_prog_aux *aux` argument; kernel build tooling
(`resolve_btfids`, driven by pahole's `decl_tag_kfuncs` BTF feature)
is supposed to publish a BPF-facing twin of each kfunc with the
trimmed prototype so schedulers built against released scx headers
and libbpf still match. When the toolchain drops that tag for a
kfunc — observed with some pahole builds, and varying by config —
the plain-name prototype keeps the extra argument and no released
scheduler can load on that kernel.

Check any kernel in one command:

```sh
bpftool btf dump file <vmlinux> format raw | grep -E "FUNC 'scx_bpf_(create_dsq|error_bstr)"
# loadable: the plain name points at a trimmed proto (no 'aux' param)
# broken:   a single 4-arg entry — released libbpf/scx headers cannot match it
```

> [!WARNING]
> `expect_err = true` tests invert this load failure into a pass,
> and `post_vm` assertions skip when the scheduler never attached —
> so a suite can look green with zero schedulers ever loading. If a
> kernel's `expect_err` tests all "pass" while everything else
> reports the scheduler never turned on, check the kernel's BTF
> before trusting the run.

**Fixes:**

- Test against a kernel whose BTF passes the check above (kernels
  before the implicit-args change, e.g. `--kernel 7.0` or
  `--kernel 6.14`, are unaffected).
- Rebuild the kernel with a pahole/toolchain combination that
  preserves the kfunc tags, and re-run the check.

## send_sys_rdy timeout

```text
WARN ktstr::vmm::rust_init: ktstr-init: send_sys_rdy failed within boot budget; see https://ktstr.dev/guide/troubleshooting.html#send_sys_rdy-timeout budget_ms=11200 vcpus=8 elapsed_ms=11342 port_exists=false kern_addrs_sent=false
```

The guest init could not send its "ready" signal to the host within
the boot budget (10 s plus 150 ms per vCPU, capped at 90 s). The WARN
itself is non-fatal — the guest continues and the host starts
sampling anyway — but the test usually then fails through the normal
VM-teardown path (see [Scheduler died](#scheduler-died)); the
authoritative deadline is the host watchdog, which scales with host
overcommit.

The diagnostic fields split the cause in two:

- `port_exists=false` — the virtio-console port device never
  appeared in the guest. Almost always a slow or CPU-deprived boot (or an
  early guest panic — check the `--- diagnostics ---` console tail).
- `port_exists=true` — the port exists but writes did not complete.
  This is a host-side virtio-console issue, not guest CPU
  contention; file a bug with the failure dump.

**Fixes (for the `port_exists=false` case):**

- Pass `--no-perf-mode` (or `KTSTR_NO_PERF_MODE=1`) to reduce
  host-side contention that deprives the guest's vCPU threads.
- Reduce the test's topology — fewer vCPUs boot faster.
- KASAN / KCSAN / lockdep kernels add substantial boot overhead;
  re-run on a non-instrumented kernel to separate instrumentation
  cost from a real scheduler watchdog event.

## Insufficient hugepages

```text
performance_mode: WARNING: no 2MB hugepages available, guest memory will use regular pages
```

```text
performance_mode: WARNING: need N 2MB hugepages, only K free — falling back to regular pages
```

[Performance mode](concepts/performance-mode.md) requests 2MB
hugepages for guest memory. The first form fires when none are
reserved on the host; the second when fewer than the run needs. In
both cases the VM falls back to regular pages and continues to boot.

**Fix:**

```sh
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

## Worker assertion failures

```text
tid 2 stuck 4500ms on cpu2 at +3200ms (threshold 3000ms)
unfair cgroup: spread=42% (8-50%) 4 workers on 4 cpus (threshold 35%)
```

The Assert checks (`max_gap_ms`, `max_spread_pct`, etc.) detected a
worker metric outside the configured thresholds. The `tid N` prefix
names the thread so you can cross-reference the `--- timeline ---`
and `--- stats ---` sections, which key per-thread metrics by `tid`;
`unfair cgroup` is per-cgroup and cross-references the per-cgroup
spread / workers / cpus columns in `--- stats ---` instead.

**Fixes:**

- Check whether the topology has enough CPUs for the scenario —
  small topologies produce higher contention, larger gaps, and more
  spread.
- Override thresholds for scenarios that need relaxed limits — see
  [Customize Checking](recipes/custom-checking.md).
- Check the scheduler's behavior under the specific flag profile
  that triggered the failure.

## Cgroup name typos

A typo'd cgroup name surfaces only when an op tries to write to a
non-existent cgroup directory; names are not pre-validated. The
diagnostic depends on which op references the typo:

- `Op::RemoveCgroup` / `Op::StopCgroup` against a typo silently
  succeed (rmdir / kill against a non-existent path are no-ops); the
  failure surfaces on the next op that touches the name.
- `Op::SetCpuset` falls through to the kernel's `ENOENT`, wrapped
  with a one-line cgroup-state snapshot:

  ```text
  cgroup-state-snapshot: parent=/sys/fs/cgroup/ktstr name=nonexistent parent.cgroup.controllers="cpuset cpu memory io pids" parent.cgroup.subtree_control="cpuset cpu memory" child.cgroup.controllers="<read failed: No such file or directory (os error 2)>" child.cpuset.cpus.exists=false child.listing=<read_dir failed: No such file or directory (os error 2)>: No such file or directory (os error 2)
  ```

  The `child.listing=<read_dir failed: ...>` segment is the tell: a
  typo'd name has no directory to list, distinguishing this from
  "cgroup exists but the write was rejected" (where the listing
  would enumerate the cgroupfs knobs).
- Other setters (`cpu.max`, `memory.max`, `cpuset.mems`, …) against
  a typo produce the same wrapped form as
  [Cgroup controller not enabled](#cgroup-controller-not-enabled) —
  distinguish by checking whether the directory exists.
- `Op::AddCgroup` colliding with an already-tracked name bails:

  ```text
  Op::AddCgroup 'cg_0' collides with a cgroup already tracked (by a prior Backdrop or step-local CgroupDef) — declare it in exactly one place; use a fresh name for the step-local cgroup
  ```

**Fixes:** verify the name matches its `Op::AddCgroup` /
`CgroupDef::named()` / `Backdrop.cgroups` declaration, and that
dynamically formatted names (`format!("cg_{i}")`) use the same
formatting everywhere.

## Cgroup controller not enabled

```text
cgroup 'cg_0': set cpu.max='100000 100000' (requires +cpu in parent cgroup.subtree_control): No such file or directory (os error 2)
cgroup 'cg_0': set memory.max='4294967296' (requires +memory in parent cgroup.subtree_control): No such file or directory (os error 2)
cgroup 'cg_0': set memory.swap.max='1073741824' (requires +memory in parent cgroup.subtree_control; file absent on CONFIG_SWAP=n kernels): No such file or directory (os error 2)
cgroup 'cg_0': set cpuset.mems='0-1' (requires +cpuset in parent cgroup.subtree_control): No such file or directory (os error 2)
```

The cgroup exists but the controller knob is missing from its
directory. ktstr's setup auto-enables the controllers it detects on
the scenario's `CgroupDef` / `Op` set, so a missing controller means
either: the framework's detection did not see a declared knob (file
a bug); an outer parent (systemd `user.slice`, container runtime)
stripped controllers from the subtree before ktstr ran; or the
kernel was built without `CONFIG_SWAP` (the `memory.swap.max` wrap
spells this out).

**Diagnostic command:**

```sh
cat /sys/fs/cgroup/<parent>/cgroup.subtree_control
```

A controller named in the wrapped error must appear in this list; if
it does not, fix the parent first (`echo '+memory' >
.../cgroup.subtree_control` from a sufficiently-privileged shell) or
remove the knob from the scenario.

## CpusetSpec errors

```text
cgroup 'cg_0': CpusetSpec validation failed: not enough usable CPUs (4) for 8 partitions
cgroup 'cg_1': CpusetSpec validation failed: index 3 >= partition count 3
cgroup 'cg_2': CpusetSpec validation failed: Range fracs must lie in [0.0, 1.0]: start_frac=-1, end_frac=0.5
```

A `CpusetSpec` cannot produce a valid cpuset for the test topology;
the step aborts as a hard error before any downstream slicing runs.

**Fixes:**

- Guard with a topology check before creating the step:
  `if ctx.topo.usable_cpus().len() < needed { return Ok(AssertResult::skip(...)); }`
- Call `CpusetSpec::validate(&ctx)` in your scenario builder so
  failures surface before `execute_steps` runs.
- Reduce the partition count, or use `CpusetSpec::Llc` instead of
  `Disjoint` on topologies with fewer CPUs than partitions.
- For `Range`/`Overlap`, keep fractions finite and inside
  `[0.0, 1.0]`; `Range` additionally requires `start_frac < end_frac`.

## Worker count mismatches

```text
PipeIo (group 0) requires num_workers divisible by 2, got 3
```

Grouped work types (`PipeIo`, `FutexPingPong`, `CachePipe`,
`FutexFanOut`, `FanOutCompute`, and the contention / waker families —
see [Workers and Workloads](architecture/workers.md)) require
`num_workers` divisible by their group size. The `(group N)` segment
names the composed entry the violation belongs to, so multi-group
scenarios point at the entry to fix.

**Fixes:**

- Set `CgroupDef::workers(n)` to a multiple of the work type's group
  size (2 for pipe/futex pairs, `fan_out + 1` for FutexFanOut and
  FanOutCompute).
- Use an ungrouped work type (`SpinWait`, `Mixed`, `Bursty`,
  `IoSyncWrite`, `IoRandRead`, `IoConvoy`, `YieldHeavy`) if worker
  count flexibility is needed.

## Cache corruption

```text
  6.14.2-tarball-x86_64-kc...                 (corrupt: metadata.json malformed: ...)
warning: entries marked (corrupt) cannot be used — cached metadata is missing, malformed, or references a missing image. Inspect the entry directory under ~/.cache/ktstr/kernels to remove it manually, or run `kernel clean --corrupt-only --force` which removes ONLY corrupt entries and leaves valid ones intact. ...
```

A cached kernel entry has missing, unparseable, or schema-drifted
`metadata.json`, or references an image that is no longer present —
typically after a partial write (disk full, killed process) or a
ktstr upgrade that changed the metadata schema. Corrupt entries are
never used; runs fall through to a rebuild. The JSON listing
(`kernel list --json`) carries a stable `error_kind` token per
corrupt entry for CI scripts — see
[cargo ktstr kernel](running-tests/cargo-ktstr.md).

**Fixes:**

- Remove only corrupt entries:
  `cargo ktstr kernel clean --corrupt-only --force`
- Rebuild a specific version after cleanup:
  `cargo ktstr kernel build --force --kernel 6.14.2`
- Move the cache with `KTSTR_CACHE_DIR` if the default location is
  on a problematic filesystem.

## Stale `vmlinux.btf` or `default.profraw` in kernel source tree

Older ktstr versions could leave two files in a kernel source
directory: `<source>/vmlinux.btf` (a BTF sidecar, now written only
inside the cache root) and `<source>/default.profraw` (an LLVM
coverage artifact, now redirected next to the cargo-ktstr binary).
Both are leftover state and safe to remove:

```sh
rm -f /path/to/linux/vmlinux.btf /path/to/linux/default.profraw
```

If they keep reappearing, you are running an old ktstr binary —
rebuild or reinstall, then delete again. See
[profraw layout](running-tests/cargo-ktstr.md) for where coverage
artifacts land now.

## Cache directory not found

```text
HOME is unset; cannot resolve cache directory. The container init or login shell did not assign HOME — set it to an absolute path, or set KTSTR_CACHE_DIR to an absolute path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a cache location explicitly.
```

```text
HOME is set to the empty string; cannot resolve cache directory. An empty HOME usually means a Dockerfile or shell rc has `export HOME=` or `ENV HOME=` with no value. Either set HOME to a real absolute path, or set KTSTR_CACHE_DIR to an absolute path (e.g. /tmp/ktstr-cache) or XDG_CACHE_HOME to specify a cache location explicitly.
```

The kernel image cache requires a writable directory, resolved as
`KTSTR_CACHE_DIR` > `$XDG_CACHE_HOME/ktstr/` > `$HOME/.cache/ktstr/`.
The first form fires when `HOME` is absent (bare container inits,
systemd units without `Environment=HOME=…`); the second when `HOME`
is set to the empty string.

**Fix:** Set `KTSTR_CACHE_DIR` to an explicit path, or ensure `HOME`
is a real absolute path.

## Stale kconfig

```text
warning: entries marked (stale kconfig) were built against a different ktstr.kconfig. Rebuild with: kernel build --force --kernel <entry version> (add --extra-kconfig PATH if the entry also carries the (extra kconfig) tag).
```

`cargo ktstr kernel list` marks entries whose stored kconfig hash
differs from the current embedded `ktstr.kconfig` fragment — typical
after updating ktstr. Stale entries rebuild automatically on the next
`cargo ktstr kernel build`; `--force` overrides the cache for other
reasons.

## Kernel auto-download failures

```text
ktstr: no kernel found, downloading latest stable
fetch https://www.kernel.org/releases.json: <error>
```

ktstr auto-downloads a kernel when no `--kernel` is specified and
the discovery chain finds nothing; the same path runs when `--kernel`
names a version not in the cache. The `<error>` is the underlying
network error (DNS, connection refused, timeout, TLS). Variants:

```text
fetch https://www.kernel.org/releases.json: HTTP 503
```

kernel.org returned a non-success status.

```text
no stable kernel with patch >= 8 found in releases.json
```

ktstr requires a stable or longterm release with patch version >= 8
to avoid brand-new majors with build issues; releases.json contained
no qualifying version.

```text
extract tarball: <error>
```

Disk full, bad permissions on the temp directory, or a truncated
download.

**Fixes:**

- Verify connectivity: `curl -sI https://www.kernel.org/releases.json`
- If behind a proxy, set `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`.
- Check disk space; override the cache location with
  `KTSTR_CACHE_DIR` if needed.
- Pre-download explicitly — `cargo ktstr kernel build --kernel 6.14.10`
  isolates version resolution from download failures.

## Kernel download failures

These fire when an explicit version is requested:

```text
version 6.14.22 not found. latest 6.14.x: 6.14.10
```

The requested version does not exist; when a sibling in the same
series is available, the error suggests it. An EOL series gets only
the bare "not found".

```text
RC tarball not found: https://git.kernel.org/torvalds/t/linux-6.15-rc3.tar.gz
  RC releases are removed from git.kernel.org after the stable version ships.
```

Use `--kernel git+URL#tag=NAME` with a git.kernel.org URL to clone
the tag instead.

```text
download ...: server returned HTML instead of tarball (URL may be invalid)
```

Some CDN error pages return HTTP 200 with HTML; the download rejects
these responses. Check the URL / version against
`https://www.kernel.org/releases.json`.

## Shell mode issues

### stdin must be a terminal

```text
stdin must be a terminal for interactive shell mode
```

`cargo ktstr shell` requires a terminal for bidirectional I/O
forwarding; piped or redirected stdin is rejected.

### include file not found

```text
-i strace: not found in filesystem or PATH
```

Bare names (without `/`, `.`, or `..`) are searched in `PATH`; if the
binary is not there, use an explicit path.

```text
--include-files path not found: ./missing-file
```

Explicit paths must exist on disk.

### include directory contains no files

```text
warning: -i ./empty-dir: directory contains no regular files
```

The directory was walked recursively but contained no regular files
(FIFOs, device nodes, and sockets are skipped).

## Flock timeout / NFS rejection

```text
flock LOCK_EX on run-dir target/ktstr/6.14-abc1234 timed out after
30s (lockfile target/ktstr/.locks/6.14-abc1234.lock, holders:
  pid=12345 cmd=cargo-ktstr test --kernel 6.14). A peer cargo
ktstr test process is writing sidecars to the same
{kernel}-{project_commit} directory; wait for it to finish or kill
it, then retry.
```

A peer process is holding the per-run-key advisory `flock(2)` that
serializes sidecar writes; the helper polled for 30 s and gave up.
Run-dir locks live at `{runs_root}/.locks/{kernel}-{project_commit}.lock`
and serialize the pre-clear + write cycle so two concurrent runs
sharing a key cannot tear each other's sidecars.

```text
target/ktstr/.locks/6.14-abc1234.lock: filesystem NFS is not
supported for ktstr lockfiles (NFSv3 is advisory-only without
an NLM peer; NFSv4 byte-range locking does not cover flock(2)).
Move the lockfile path to a local filesystem (tmpfs, ext4, xfs,
btrfs, f2fs, bcachefs).
```

ktstr rejects NFS, CIFS, SMB2, CephFS, AFS, and FUSE mounts for
lockfiles because `flock(2)` semantics there are unreliable — see
[Resource Budget](concepts/resource-budget.md) for the rationale.

**Diagnose:**

- `cargo ktstr locks` (or `ktstr locks --watch 1s`) prints every
  ktstr flock currently held on the host with PID + cmdline — see
  [ktstr (standalone)](running-tests/ktstr.md).
- `cat /proc/locks | grep '<lockfile-path-from-error>'` falls back to
  the kernel's own flock enumeration when the holder is outside ktstr.
- `stat -f -c '%T' <runs-root>` reports the filesystem type.

**Fix:**

- Peer-holder timeout: wait for the peer, kill it (`kill <pid>` from
  the holder list), or retry.
- NFS / remote-fs rejection: relocate the runs root to a local
  filesystem via `KTSTR_SIDECAR_DIR` — noting that the override path
  also skips the cross-process flock, so give each concurrent run its
  own path. The kernel cache's lockfiles face the same constraint —
  override `KTSTR_CACHE_DIR` if the default resolves to NFS.

## Test hangs / nextest timeout

A VM test that stops making progress is eventually flagged `SLOW` by
nextest and then terminated when it exceeds the profile's
slow-timeout budget: 60 s × 2 periods on the default profile, 90 s ×
3 on the `ci` profile, with larger per-test overrides for heavy
classes (verifier sweeps 180 s, the wide-SMP boots up to 960 s — see
`.config/nextest.toml`).

ktstr's own per-VM watchdog is sized to fire *before* nextest's kill
so you get a failure dump instead of a blunt termination. If nextest
kills first, you lose the dump — so:

- Re-run just the failing test with its exact variant name and read
  the dump — see [Reading Failure Output](running-tests/failures.md).
- Check for peers holding CPU locks (`cargo ktstr locks`) — a
  contended host makes VM boots slow enough to blow timeouts.
- On a busy or small machine, pass `--no-perf-mode` and use
  `--profile ci` locally for the bigger budgets.
- If one test legitimately needs longer (huge topology), give it a
  per-test override in `.config/nextest.toml` rather than raising the
  profile-wide timeout.

## Tests pass locally but fail in CI

Common causes:

- **No KVM**: CI runners need hardware virtualization. Check for
  `/dev/kvm` access.
- **Fewer CPUs**: gauntlet topology presets up to 252 CPUs may exceed
  the runner's capacity. Use smaller topologies.
- **No kernel**: set `KTSTR_TEST_KERNEL` in the CI environment, or
  build and cache one per [CI](ci.md).
- **No CAP_SYS_NICE or rtprio**: performance-mode tests require
  `CAP_SYS_NICE` or an rtprio limit for RT scheduling, and enough
  host CPUs for exclusive LLC reservation. Pass `--no-perf-mode`
  (or set `KTSTR_NO_PERF_MODE=1`) to disable all performance mode
  features; tests with `performance_mode=true` are then skipped
  entirely.
- **Debug thresholds**: CI often runs debug builds. Debug builds use
  relaxed thresholds (3000ms gap, 35% spread) but may still hit
  limits on slow runners. See [Checking](concepts/checking.md).
