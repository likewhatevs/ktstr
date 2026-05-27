// `declare_scheduler!`'s `kargs` field elements must each be string
// literals (the runtime appends them verbatim to the guest kernel
// cmdline). A non-string element is rejected at macro-expansion time
// so the operator gets a typed error instead of a downstream
// "expected &str" deep in the cmdline assembly path.
use ktstr::declare_scheduler;

declare_scheduler!(KARGS_NON_STRING_ELEMENT, {
    name = "kargs_non_string_element",
    binary = "scx_test",
    kargs = ["nokaslr", 12345],
});

fn main() {}
