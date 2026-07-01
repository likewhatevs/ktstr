// Pins the whitespace-only binary rejection. Previously the
// `lit.is_empty()` check accepted `" "` (single space) which flowed
// to runtime as `SchedulerSpec::Discover(" ")` and failed
// confusingly inside `build_and_find_binary(" ")`. The
// `check_visible_lit` guard (via `is_visibly_empty`) rejects empty,
// whitespace-only, AND invisible-only (ZWSP/BOM/bidi) binary literals
// at macro time.
use ktstr::declare_scheduler;

declare_scheduler!(WHITESPACE_BINARY, {
    name = "whitespace_binary",
    binary = "   ",
});

fn main() {}
