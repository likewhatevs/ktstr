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
//! queue-pair *setup* (config-space offer + control-vq negotiation), which
//! is what `queue_pairs` configures. Per-queue MSI-X/GSI steering (so each
//! pair raises a distinct interrupt vector) is the per-queue MSI-X follow-up.
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

    Ok(AssertResult::pass())
}
