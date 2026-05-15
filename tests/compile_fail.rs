#[test]
#[ignore = "many trybuild fixtures, each spawning cargo build; run explicitly with --run-ignored=all"]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
