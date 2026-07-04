// Mutual exclusion across {binary, binary_path, kernel_builtin_{enable,disable}}: each
// field selects a different `SchedulerSpec` variant and they cannot
// stack. The macro rejects with the pick-exactly-one diagnostic.
use ktstr::declare_scheduler;

declare_scheduler!(BINARY_AND_BINARY_PATH, {
    name = "binary_and_binary_path",
    binary = "scx_foo",
    binary_path = "/usr/local/bin/scx_foo",
});

fn main() {}
