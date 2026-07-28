//! Process-isolated environment fixtures shared by cargo-ktstr's unit tests.

use std::ffi::{OsStr, OsString};
use std::process::Command;

const REEXEC_CASE_ENV: &str = "KTSTR_TEST_REEXEC_CASE";

/// One environment edit applied only to a re-executed test process.
pub(crate) enum ChildEnv {
    Set { key: &'static str, value: OsString },
    Remove { key: &'static str },
}

impl ChildEnv {
    pub(crate) fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
        Self::Set {
            key,
            value: value.as_ref().to_os_string(),
        }
    }

    pub(crate) fn remove(key: &'static str) -> Self {
        Self::Remove { key }
    }
}

pub(crate) fn is_reexec_case(case: &str) -> bool {
    std::env::var_os(REEXEC_CASE_ENV).is_some_and(|value| value.as_os_str() == OsStr::new(case))
}

/// Run only the current libtest case in a child with an isolated environment.
///
/// The parent test never mutates its process-global environment, so unrelated
/// tests remain sound when libtest runs them concurrently.
pub(crate) fn reexec_current_test(
    case: &str,
    environment: impl IntoIterator<Item = ChildEnv>,
) -> String {
    let current_thread = std::thread::current();
    let test_name = current_thread
        .name()
        .expect("libtest must name test threads");
    let executable = std::env::current_exe().expect("resolve current test executable");
    let mut command = Command::new(executable);
    command
        .args(["--exact", test_name, "--nocapture"])
        .env(REEXEC_CASE_ENV, case);
    for edit in environment {
        match edit {
            ChildEnv::Set { key, value } => {
                command.env(key, value);
            }
            ChildEnv::Remove { key } => {
                command.env_remove(key);
            }
        }
    }

    let output = command.output().expect("re-execute current test");
    assert!(
        output.status.success(),
        "re-executed test case {case:?} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}
