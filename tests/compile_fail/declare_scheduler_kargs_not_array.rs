// `declare_scheduler!`'s `kargs` field requires an array literal
// `[..]`. A scalar literal is rejected at macro-expansion time so the
// operator gets a typed error instead of a downstream "expected
// &[&str], got &str" deep in the guest cmdline assembly.
use ktstr::declare_scheduler;

declare_scheduler!(KARGS_NOT_ARRAY, {
    name = "kargs_not_array",
    binary = "scx_test",
    kargs = "not_an_array",
});

fn main() {}
