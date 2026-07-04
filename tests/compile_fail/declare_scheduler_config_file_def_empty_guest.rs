// Pins config_file_def empty-string check on guest_path
// position (element 1). An empty guest_path leaves the config
// with no destination path inside the guest.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    config_file_def = ("--config {file}", ""),
});

fn main() {}
