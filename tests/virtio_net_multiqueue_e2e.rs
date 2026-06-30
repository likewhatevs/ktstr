//! End-to-end: virtio-net multiqueue (`NetConfig::queue_pairs`).
//!
//! Boots a guest with more online CPUs (8) than the NIC's offered
//! queue-pair maximum (4) and asserts the interface comes up with exactly
//! 4 active queue-pairs. This exercises the whole multiqueue path
//! end-to-end:
//!
//!   * the device offers `VIRTIO_NET_F_MQ` + `VIRTIO_NET_F_CTRL_VQ` because
//!     `queue_pairs > 1` and the x86_64 PCI transport wired MSI-X (the gate
//!     in `device_features`);
//!   * `virtnet_probe` reads `max_virtqueue_pairs=4` from config space, sets
//!     `curr_queue_pairs = min(num_online_cpus, max) = min(8, 4) = 4`
//!     ("Enable multiqueue by default", drivers/net/virtio_net.c), and
//!     issues `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET(4)` over the control vq;
//!   * the VMM's `process_ctrl_queue` validates the command (`4 ∈ [1, 4]`),
//!     ACKs `VIRTIO_NET_OK`, and records `curr_queue_pairs = 4`;
//!   * the guest then calls `netif_set_real_num_{tx,rx}_queues(4)`, creating
//!     exactly four `/sys/class/net/<iface>/queues/{rx,tx}-N` dirs.
//!
//! Choosing 8 vCPUs against an offered max of 4 also proves the advertised
//! limit binds below the CPU count: a device that ignored its own
//! `max_virtqueue_pairs` would let the guest activate 8 pairs (and the
//! control vq would NAK the over-range request, leaving the guest at 1).
//! Exactly 4 is the only outcome consistent with the offered-max contract.
//!
//! The in-VMM loopback backend drives no traffic here — this checks the
//! queue-pair *setup* (config-space offer + control-vq negotiation), which is
//! what `queue_pairs` configures, AND the per-queue MSI-X layout: with
//! per-queue MSI-X each data virtqueue gets its OWN interrupt vector, so the
//! guest's EACH policy requests `config + 2*pairs` vectors and `/proc/interrupts`
//! shows one `{dev}-input.N` + one `{dev}-output.N` line per active pair (plus a
//! single `{dev}-config`). Distinct per-queue lines ARE the per-node IRQ
//! steering payload — a SHARED single queue vector would show just one data line
//! for every queue.
//!
//! x86_64-only: the multiqueue offer rides MSI-X (`msix.is_some()`), which
//! the PCI facade wires only on x86_64. On aarch64 the NIC uses the MMIO
//! transport (no MSI-X), so `F_MQ` is not offered and the guest stays
//! single-pair — a different code path covered by the unit tests.
//!
//! Run: cargo run --bin cargo-ktstr -- ktstr test --kernel ../linux \
//!        -- -E 'test(multiqueue_activates_offered_pairs)' \
//!        --success-output immediate
#![cfg(target_arch = "x86_64")]

use anyhow::{Result, ensure};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::NetConfig;
use ktstr::scenario::Ctx;

#[path = "common/wide_smp_irq.rs"]
mod wide_smp_irq;
use wide_smp_irq::virtio_net_iface;

/// virtio-net offering 4 queue-pairs with a deterministic locally-administered
/// MAC. Const because the `networks = [...]` macro arg needs a const-evaluable
/// path; `NetConfig::DEFAULT.mac(..).queue_pairs(..)` is the const-fn chain.
const KTSTR_NET: NetConfig =
    NetConfig::DEFAULT.mac([0x52, 0x54, 0x00, 0x12, 0x34, 0x58]).queue_pairs(4);

/// The NIC's offered queue-pair maximum (matches `KTSTR_NET.queue_pairs`).
const OFFERED_PAIRS: usize = 4;

#[ktstr_test(
    llcs = 2,
    cores = 2,
    threads = 2,
    networks = [KTSTR_NET],
    no_perf_mode,
    duration_s = 2
)]
fn multiqueue_activates_offered_pairs(_ctx: &Ctx) -> Result<AssertResult> {
    // 2 llcs x 2 cores x 2 threads = 8 online vCPUs > 4 offered pairs, so
    // curr_queue_pairs = min(8, 4) = 4 (the offered max binds).
    let iface = virtio_net_iface()?;

    // The kernel creates one `queues/rx-N` and one `queues/tx-N` sysfs dir
    // per active queue-pair via netif_set_real_num_{rx,tx}_queues, so the
    // dir counts ARE curr_queue_pairs.
    let qdir = format!("/sys/class/net/{iface}/queues");
    let mut rx = 0usize;
    let mut tx = 0usize;
    for entry in std::fs::read_dir(&qdir)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("rx-") {
            rx += 1;
        } else if name.starts_with("tx-") {
            tx += 1;
        }
    }

    eprintln!(
        "MQ iface={iface} rx_queues={rx} tx_queues={tx} \
         (offered max={OFFERED_PAIRS}, online vcpus=8)"
    );

    ensure!(
        rx == OFFERED_PAIRS,
        "expected {OFFERED_PAIRS} RX queues (the offered max binding below 8 \
         online CPUs), got {rx}; the control-vq VQ_PAIRS_SET the guest sent \
         at probe was not honored as a {OFFERED_PAIRS}-pair activation"
    );
    ensure!(
        tx == OFFERED_PAIRS,
        "expected {OFFERED_PAIRS} TX queues, got {tx}"
    );

    // Per-queue MSI-X: each data virtqueue gets its OWN interrupt vector under
    // the guest's EACH policy, so /proc/interrupts shows one `{dev}-input.N` +
    // one `{dev}-output.N` line per active pair, plus a single `{dev}-config`.
    // The control vq has no callback and so gets NO_VECTOR (no line). These
    // distinct per-queue lines ARE the per-node IRQ steering payload: a SHARED
    // single queue vector would show just one data line for every queue.
    let dev = std::fs::canonicalize(format!("/sys/class/net/{iface}/device"))?;
    let dev = dev
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("no device basename for {iface}"))?
        .to_string();
    let interrupts = std::fs::read_to_string("/proc/interrupts")?;
    let input_prefix = format!("{dev}-input.");
    let output_prefix = format!("{dev}-output.");
    let config_action = format!("{dev}-config");
    let (mut inputs, mut outputs, mut configs) = (0usize, 0usize, 0usize);
    for line in interrupts.lines() {
        let Some((_, rhs)) = line.split_once(':') else {
            continue;
        };
        let Some(action) = rhs.split_whitespace().last() else {
            continue;
        };
        if action.starts_with(&input_prefix) {
            inputs += 1;
        } else if action.starts_with(&output_prefix) {
            outputs += 1;
        } else if action == config_action {
            configs += 1;
        }
    }

    eprintln!(
        "MQ per-vq MSI-X dev={dev} input_irqs={inputs} output_irqs={outputs} \
         config_irqs={configs}"
    );

    ensure!(
        inputs == OFFERED_PAIRS,
        "expected {OFFERED_PAIRS} per-queue RX MSI-X lines ({dev}-input.N), got \
         {inputs}; per-queue MSI-X must give each RX virtqueue its own vector \
         (a SHARED single vector would show one data line for all queues)"
    );
    ensure!(
        outputs == OFFERED_PAIRS,
        "expected {OFFERED_PAIRS} per-queue TX MSI-X lines ({dev}-output.N), got \
         {outputs}"
    );
    ensure!(
        configs == 1,
        "expected exactly one {dev}-config MSI-X line, got {configs}"
    );

    Ok(AssertResult::pass())
}
