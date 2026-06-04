// Minimal prod-side bin for the devdep-isolation fixture. Body content
// is irrelevant — the verify script asserts on what cargo DOES NOT
// compile and what symbols ARE NOT present in the release binary. The
// single `println!` keeps the body non-empty so the linker has
// something to emit and `nm target/release/devdep-fixture` returns
// real symbols (rather than an empty binary that could falsely satisfy
// "no ktstr symbols" simply by having no symbols at all).
fn main() {
    println!("devdep-fixture: prod bin, no ktstr should be linked here");
}
