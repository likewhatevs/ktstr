/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Minimum-weight sched_ext scheduler for the verifier recursive-discovery
 * end-to-end fixture. Global FIFO through one shared DSQ: the default
 * `select_cpu` is left in place, `enqueue` inserts to the shared DSQ, and
 * `dispatch` moves the head to the local DSQ. That is the smallest ops table
 * that genuinely registers + enables sched_ext (so `/sys/kernel/sched_ext/
 * root/ops` goes non-empty and `state` reads `enabled`), which is what the
 * cell counts as a passing attach.
 *
 * Deliberately self-contained: it pulls only the CO-RE `vmlinux.h` (generated
 * from host BTF by build.rs) plus libbpf's bundled `bpf_helpers.h`/
 * `bpf_tracing.h`. The scx kfunc externs and the two struct-ops wrapper macros
 * are declared inline so the fixture needs no scx_utils/scx_cargo bundled BPF
 * headers. Built against the older (6.14) sched_ext_ops shape, which is a
 * subset of 7.1's, so one CO-RE object loads on both guest kernels.
 *
 * Safety (no watchdog kills): every runnable task is inserted with
 * SCX_SLICE_DFL and dispatched promptly from the shared DSQ, so no task is
 * starved and the scheduler never wedges.
 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

/*
 * A custom DSQ (id 0) rather than SCX_DSQ_GLOBAL: built-in DSQs are consumed
 * automatically and cannot be moved from with scx_bpf_dsq_move_to_local(),
 * which is the one dispatch primitive this scheduler exercises.
 */
#define SHARED_DSQ 0

/*
 * scx's BPF_STRUCT_OPS wrappers, inlined to avoid the scx bundled headers.
 * BPF_PROG (from bpf_tracing.h) generates the context-cast trampoline; the
 * SEC prefix selects the struct_ops program (`.s` marks the sleepable init).
 */
#define BPF_STRUCT_OPS(name, args...) SEC("struct_ops/" #name) BPF_PROG(name, ##args)
#define BPF_STRUCT_OPS_SLEEPABLE(name, args...) SEC("struct_ops.s/" #name) BPF_PROG(name, ##args)

/* sched_ext kfuncs. Names present in both 6.14 and 7.1 (the pre-6.13
 * scx_bpf_dispatch/scx_bpf_consume spellings are intentionally avoided). */
s32 scx_bpf_create_dsq(u64 dsq_id, s32 node) __ksym;
void scx_bpf_dsq_insert(struct task_struct *p, u64 dsq_id, u64 slice, u64 enq_flags) __ksym __weak;
bool scx_bpf_dsq_move_to_local(u64 dsq_id) __ksym __weak;

void BPF_STRUCT_OPS(standin_enqueue, struct task_struct *p, u64 enq_flags)
{
	scx_bpf_dsq_insert(p, SHARED_DSQ, SCX_SLICE_DFL, enq_flags);
}

void BPF_STRUCT_OPS(standin_dispatch, s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(SHARED_DSQ);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(standin_init)
{
	return scx_bpf_create_dsq(SHARED_DSQ, -1);
}

void BPF_STRUCT_OPS(standin_exit, struct scx_exit_info *ei)
{
}

SEC(".struct_ops.link")
struct sched_ext_ops standin_ops = {
	.enqueue = (void *)standin_enqueue,
	.dispatch = (void *)standin_dispatch,
	.init = (void *)standin_init,
	.exit = (void *)standin_exit,
	.flags = 0,
	.name = "scx_standin",
};
