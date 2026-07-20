//! Crash investigation via BPF kprobes, fentry, and fexit.
//!
//! Attaches kprobes and fentry probes to kernel and BPF functions from
//! a crash stack trace, selects the kernel's scheduler-exit hook
//! (`tp_btf/sched_ext_exit` on the newest kernels,
//! a raw `scx_vexit` entry/return pair on the preceding generation, or filtered
//! `fentry/scx_dump_state` on global-era kernels), captures argument state,
//! and formats annotated output with source locations.
//!
//! See the [Investigate a Crash](https://ktstr.dev/guide/recipes/investigate-crash.html)
//! recipe.

pub mod btf;
pub(crate) mod decode;
pub mod output;
pub mod process;
pub(crate) mod scx_defs;
pub mod stack;
