// SPDX-License-Identifier: GPL-2.0-only
//! scx_stats userspace-protocol surface for the scx-ktstr fixture
//! scheduler.
//!
//! Mirrors the pattern used by stock scx_* schedulers (e.g.
//! `scx_bpfland::stats`): a `KtstrStats` struct deriving
//! `scx_stats_derive::Stats`, paired with a `server_data()` helper
//! that wires a `StatsServerData::open` callback to the Sender/
//! Receiver pair the main run loop owns. Each `stats` request the
//! socket receives drives one read of the BPF .bss counters
//! (`nr_dispatched`, `nr_enqueued`, `nr_select_cpu`, `nr_yielded`)
//! plus the `stats_magic` fidelity sentinel, and one response over
//! the channel.
//!
//! The default socket path is `/var/run/scx/root/stats` (set by
//! `StatsServer::new`'s defaults); ktstr's host-side
//! `SchedStatsClient` reaches it through the in-guest stats relay
//! that bridges `/dev/vport0p2` to the same Unix socket.

use scx_stats::prelude::*;
use scx_stats_derive::Stats;
use serde::Deserialize;
use serde::Serialize;

/// scx-ktstr scheduler counters plus a stats-bridge fidelity
/// sentinel. Wire shape matches the scx_stats line-delimited JSON
/// envelope: a successful "stats" request returns
/// `{"errno":0,"args":{"resp":{"nr_dispatched":N,"nr_enqueued":N,"nr_select_cpu":N,"nr_yielded":N,"stats_magic":N}}}\n`.
///
/// The four `nr_*` counters are monotonic and 64-bit (matching the
/// BPF `volatile u64` declarations in `main.bpf.c`). They are updated
/// atomically via `__sync_fetch_and_add` from the per-CPU ops
/// callbacks; userspace reads them under the BPF .bss accessor
/// without additional locking — the read is atomic with respect to
/// the increment because both sides operate on naturally aligned
/// 64-bit fields and the reader observes whatever value is current
/// at the read instant. Counter tests assert directionally (counter
/// increased) rather than against exact values; `stats_magic` is a
/// fixed sentinel — stamped once by `ktstr_init`, never incremented —
/// that the stats-bridge e2e asserts exactly.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Stats)]
#[stat(top)]
pub struct KtstrStats {
    /// Cumulative count of `scx_bpf_dsq_move_to_local` calls in
    /// `ktstr_dispatch`, bumped unconditionally after the call
    /// regardless of whether a task was moved (the kfunc's bool
    /// return is ignored). Increments after the move returns, so
    /// the `--stall`, `--slow`, and `--degrade` skip-paths do not
    /// bump the counter.
    #[stat(desc = "Number of dispatch moves attempted via SHARED_DSQ")]
    pub nr_dispatched: u64,
    /// Cumulative count of `ktstr_enqueue` invocations. Bumps on
    /// every callback regardless of which DSQ the task lands in
    /// (SHARED_DSQ vs. SCX_DSQ_LOCAL_ON | cpu under
    /// scattershot/degrade).
    #[stat(desc = "Number of enqueue callbacks observed")]
    pub nr_enqueued: u64,
    /// Cumulative count of `ktstr_select_cpu` invocations.
    #[stat(desc = "Number of select_cpu callbacks observed")]
    pub nr_select_cpu: u64,
    /// Cumulative count of `ktstr_yield` invocations -- bumped on every
    /// yield callback (undirected `sched_yield(2)` and directed
    /// `yield_to(2)` alike). The fixture handler treats both uniformly:
    /// zero the yielder's slice and return false (see `ktstr_yield` in
    /// main.bpf.c).
    #[stat(desc = "Number of yield callbacks observed")]
    pub nr_yielded: u64,
    /// Stats-bridge fidelity sentinel. `ktstr_init` stamps this with a
    /// fixed magic (`KTSTR_STATS_MAGIC` in `main.bpf.c`); unlike the
    /// counters above it is never incremented. The host stats-bridge
    /// e2e asserts the value it receives equals that magic exactly,
    /// proving the scx_stats relay delivers an emitted value
    /// byte-for-byte rather than a coincidental non-zero.
    #[stat(desc = "Fixed sentinel for stats-bridge round-trip fidelity")]
    pub stats_magic: u64,
}

/// Build the `StatsServerData` instance the scheduler hands to
/// `StatsServer::new`. The op registered under the "top" target
/// (the default target an untargeted "stats" request resolves to)
/// dispatches each incoming stats request through the channel pair
/// the main run loop owns:
/// the request triggers a fresh BPF .bss read on the userspace
/// thread, which sends the new `KtstrStats` instance back over
/// the response channel.
///
/// No primer / delta computation: tests want raw cumulative
/// counters so the host can assert "increased" rather than
/// "delta during the last interval".
pub fn server_data() -> StatsServerData<(), KtstrStats> {
    let open: Box<dyn StatsOpener<(), KtstrStats>> = Box::new(move |_| {
        let read: Box<dyn StatsReader<(), KtstrStats>> =
            Box::new(move |_args, (req_ch, res_ch)| {
                req_ch.send(())?;
                let cur = res_ch.recv()?;
                cur.to_json()
            });
        Ok(read)
    });

    StatsServerData::new()
        .add_meta(KtstrStats::meta())
        .add_ops("top", StatsOps { open, close: None })
}
