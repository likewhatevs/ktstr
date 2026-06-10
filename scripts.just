# rust-script recipes for ktstr. Run via `just scripts::<name>` or the
# thin wrappers in the main justfile. Each `[script("rust-script")]`
# body is a self-contained Rust program; values reach it ONLY through
# just's `{{ ... }}` template substitution, never argv/env.

# Verify ktstr stays out of a downstream consumer's release binary.
# Builds the dev-dep fixture (a standalone crate that takes ktstr as a
# [dev-dependencies]) and asserts the Cargo dev-dep isolation contract:
#   (1) `cargo build --release` compiles ZERO ktstr code,
#   (2) the release binary carries no `ktstr::` symbols.
# A regression that leaks ktstr into a prod build path trips one of the
# assertions and fails CI.
[script("rust-script")]
devdep-isolation:
    use std::path::Path;
    use std::process::{Command, exit};

    fn fail(msg: &str) -> ! {
        eprintln!("\nFAIL: {msg}");
        exit(1);
    }

    fn main() {
        // `[script]` recipes get values ONLY via just template
        // substitution, never argv/env -- inject the fixture dir here.
        let fixture = "{{justfile_directory()}}/tests/devdep-isolation";

        // Full clean so the "Compiling" check below sees this build's
        // real work, not stale artifacts. Scoped to the fixture's own
        // target dir (it is its own workspace), not the parent's.
        eprintln!("==> Running clean release build of devdep-fixture");
        let clean = Command::new("cargo")
            .args(["clean", "--quiet"])
            .current_dir(fixture)
            .status()
            .unwrap_or_else(|e| fail(&format!("spawn cargo clean: {e}")));
        if !clean.success() {
            fail(&format!("cargo clean exited {clean}"));
        }

        // Capture stdout+stderr (cargo emits "Compiling" on stderr),
        // then echo both so the build stays visible in the CI log.
        let build = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(fixture)
            .output()
            .unwrap_or_else(|e| fail(&format!("spawn cargo build: {e}")));
        let stdout = String::from_utf8_lossy(&build.stdout);
        let stderr = String::from_utf8_lossy(&build.stderr);
        print!("{stdout}");
        eprint!("{stderr}");
        if !build.status.success() {
            fail(&format!("cargo build --release exited {}", build.status));
        }

        // Assertion 1: cargo did not compile ktstr for the release
        // build. The trailing space pins the bare crate name (excludes
        // "ktstr-macros" / "ktstr-devdep-fixture"); the version token
        // (v or digit) anchors the match to a real "Compiling ktstr vX".
        let compiled = stderr.lines().chain(stdout.lines()).any(|line| {
            line.trim_start()
                .strip_prefix("Compiling ktstr ")
                .is_some_and(|r| r.starts_with(|c: char| c == 'v' || c.is_ascii_digit()))
        });
        if compiled {
            fail("`cargo build --release` compiled ktstr -- it is leaking \
                  into the prod build path of a dev-dep consumer.");
        }
        eprintln!("PASS: cargo did not compile ktstr for the release build");

        // Assertion 2: release binary carries no ktstr symbols. Demangle
        // (-C) so we match the `ktstr::` namespace rather than the
        // fixture's own `ktstr_devdep_fixture::` symbols.
        let bin = format!("{fixture}/target/release/devdep-fixture");
        // Confirm cargo produced the artifact. verify.sh checked the exec
        // bit; we check presence, which is what the nm step actually needs
        // -- nm reads the symbol table off the file without executing it (a
        // non-executable ELF analyzes fine), and a missing/unreadable file
        // still fails here or at nm.
        if !Path::new(&bin).is_file() {
            fail(&format!("expected release binary at {bin} -- cargo build did not produce it."));
        }
        let nm = Command::new("nm")
            .args(["-C", "--defined-only", &bin])
            .output()
            .unwrap_or_else(|e| fail(&format!("spawn nm: {e}")));
        if !nm.status.success() {
            fail(&format!("nm {bin} exited {}", nm.status));
        }
        let nm_out = String::from_utf8_lossy(&nm.stdout).into_owned();
        let hits: Vec<&str> = nm_out.lines().filter(|l| l.contains("ktstr::")).collect();
        if !hits.is_empty() {
            eprintln!("\nFAIL: release binary {bin} contains {} ktstr:: symbols:", hits.len());
            for h in hits.iter().take(20) {
                eprintln!("  {h}");
            }
            exit(1);
        }
        eprintln!("PASS: release binary {bin} contains no ktstr:: symbols");

        eprintln!("\nOK: ktstr stays out of the downstream consumer's release binary.");
    }
