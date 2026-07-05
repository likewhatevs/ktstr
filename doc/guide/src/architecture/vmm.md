# VMM

ktstr includes a purpose-built VMM (virtual machine monitor) that boots
Linux kernels in KVM for testing.

## Why a purpose-built VMM

Three requirements rule out reusing a general-purpose VMM:

- **Direct guest-memory access.** The [monitor](monitor.md) reads
  scheduler state straight out of guest DRAM through a host-side
  pointer into the VM's memory mapping. Owning the VMM means owning
  that mapping — no guest agent, no hypercall surface, no negotiation
  with someone else's memory model.
- **Topology is the product.** Tests declare NUMA nodes, LLCs, cores,
  and SMT threads, and the guest must actually have that shape — down
  to asymmetric node sizes, inter-node distances, and CXL memory-only
  nodes. The VMM builds the ACPI tables to the declared shape rather
  than approximating it with generic knobs.
- **Boot cost is paid per test.** Every `#[ktstr_test]` boots a fresh
  VM, so setup has to be cheap. From a real run (2-vCPU guest, warm
  caches):

<!-- captured: cargo ktstr test --kernel 7.0 -E 'test(throughput_gate)' | ktstr 0.23.0 | kernel 7.0.14 -->
```text
  initramfs spawn: 55.583µs
  kvm+kernel: 867.005µs
  setup_memory (joins initramfs): 1.409360963s
  setup_vcpus: 1.409565321s
VM setup total: 1.409619773s
```

Creating the KVM VM and loading the kernel costs under a millisecond;
the dominant cost is populating guest memory, which joins the cached
initramfs build (below). After setup, the guest still has to boot the
kernel — total wall-clock per test is dominated by the scenario's own
duration.

## KtstrVm builder

```rust,ignore
let result = vmm::KtstrVm::builder()
    .kernel(&kernel_path)
    .init_binary(&ktstr_binary)
    .topology(Topology::new(numa_nodes, llcs, cores_per_llc, threads_per_core))
    .memory_mib(4096)
    .run_args(&["run".into(), "--ktstr-test-fn".into(), "my_test".into()])
    .build()?
    .run()?;
```

Test authors do not touch this directly — `#[ktstr_test]` drives it —
but every attribute on the macro (topology dims, memory, kargs) lands
here.

## Topology

The VM topology is specified as `(numa_nodes, llcs, cores_per_llc,
threads_per_core)`. On x86_64, the VMM creates ACPI tables (MADT,
SRAT, SLIT, and HMAT when `numa_nodes > 1`) and MP tables. On
aarch64, topology is expressed via FDT cpu nodes with MPIDR-derived
`reg` properties.

```rust,ignore
pub struct Topology {
    pub llcs: u32,
    pub cores_per_llc: u32,
    pub threads_per_core: u32,
    pub numa_nodes: u32,
    pub nodes: Option<&'static [NumaNode]>,
    pub distances: Option<&'static NumaDistance>,
}
```

`total_cpus()` = llcs × cores_per_llc × threads_per_core.

When `nodes` is `None` (the default), memory and LLCs are distributed
uniformly across NUMA nodes with default 10/20 distances. When
`Some`, each `NumaNode` specifies its LLC count, memory size, and
optional HMAT attributes (`latency_ns`, `bandwidth_mbs`,
`mem_side_cache`). A `NumaNode` with `llcs = 0` models a CXL
memory-only node.

`NumaDistance` is an NxN inter-node distance matrix. Diagonal entries
must be 10 and off-diagonal > 10 (ACPI SLIT requirements); ktstr
additionally requires the matrix to be symmetric.

Use `Topology::new(numa_nodes, llcs, cores, threads)` for uniform
topologies, or `Topology::with_nodes(cores, threads, &nodes)` for
explicit per-node configuration. The test-author view of all this is
[Topology](../concepts/topology.md).

## initramfs

The VMM builds a cpio initramfs containing:

- The test binary (as `/init`)
- Optional scheduler binary (as `/scheduler`)
- Shared library dependencies (resolved via ELF `DT_NEEDED` parsing)

The initramfs is split into a cached base plus a per-run suffix. The
base cache key is derived from the payload's shared-library set and
the content hashes of the packed scheduler/probe/worker binaries and
include files — not the test binary's own bytes, which ride the
per-run suffix. So recompiling your tests keeps the base cache warm,
while recompiling the scheduler invalidates it. The cached base lives
in a shared-memory segment that concurrent VMs map zero-copy, sharing
physical pages across parallel tests.

## Guest–host transports {#transports}

| Transport | Carries |
|-----------|---------|
| COM1 (serial) | Guest kernel console. Forwarded to stderr with `--dmesg`. |
| COM2 (serial) | Crash diagnostics only: the guest panic hook writes `PANIC: <info>` plus a backtrace here. |
| `/dev/hvc0` (console port 0) | Interactive console for `ktstr shell`. |
| Console port 1 | The primary guest-to-host data channel: test results, exit codes, scenario markers, payload metrics, coverage data, scheduler-exit notifications. |
| Console port 2 | Transparent byte relay for scx_stats requests/responses between the host and the in-guest scheduler. |

Two details worth internalizing:

- **COM2 is crash-only.** Ordinary guest stdout/stderr does not use
  COM2 — it travels over the port-1 stream as framed messages. COM2
  exists for diagnostics that must get out even when the framed
  transport can't be trusted (panics, fatal signals). The host parses
  the `PANIC:` header and surfaces the backtrace in test failure
  output.
- **Port 1 frames are integrity-checked.** Each frame on the port-1
  stream carries a CRC32, so a corrupted result is detected rather
  than mis-parsed.

## Performance mode

When performance mode is enabled, the VMM applies host-side isolation
(vCPU pinning, hugepages, NUMA mbind, RT scheduling), guest-visible
hints (`KVM_HINTS_REALTIME` CPUID), and KVM exit suppression.
Non-performance-mode VMs set the KVM halt-poll interval to 200µs;
overcommitted topologies set it to 0. See
[Performance Mode](../concepts/performance-mode.md).

## Dual-role dispatch

The same test binary is the host controller and the guest `/init` —
[Architecture Overview](../architecture.md) tells the story. The
mechanics: a constructor function runs before `main()` in every
ktstr-linked binary. Running as PID 1, it executes the guest init
path (mounts, scheduler start, test dispatch, reboot); given
`--ktstr-test-fn` plus a topology argument, it boots a VM as the host
side; given only `--ktstr-test-fn`, it runs the test function
directly because it is already inside a VM.

## Boot process

1. Load the kernel (bzImage on x86_64, Image on aarch64).
2. Create KVM vCPUs matching the declared topology. High vCPU counts
   add measurable boot latency — see
   [Performance Mode](../concepts/performance-mode.md) for sizing.
3. Build and load the initramfs.
4. Set up serial devices (COM1 kernel console, COM2 crash
   diagnostics), the virtio console, and virtio block/net devices for
   disk- and network-shaped workloads.
5. Boot the kernel.
6. The kernel starts `/init` (the test binary); PID 1 detection routes
   into the guest lifecycle: mount filesystems, start the scheduler,
   dispatch the test function, reboot.
