// Macro must match guest interpreter exactly. `exec_shell_line` at
// src/vmm/rust_init/dump.rs uses `split_once(" > ")` (literal space-
// greater-space substring). `echo 1>/path` (no spaces around `>`)
// contains no ` > ` substring, so the guest's `split_once(" > ")`
// returns None and `exec_shell_line` logs `unsupported command` and
// returns Err(()), which the caller counts as a partial-apply failure.
// Macro rejects at expand time to surface that as a compile error.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_BUILTIN_NO_SPACE_REDIRECT, {
    name = "kernel_builtin_no_space_redirect",
    kernel_builtin_enable = ["echo 1>/proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

fn main() {}
