//! Process preparation shared by every cargo-ktstr nextest frontend.
//!
//! Nextest keeps pipes and process handles open for admitted tests.  Its
//! effectively-unbounded admission mode must therefore not inherit a small
//! login-session `RLIMIT_NOFILE` soft limit (1024 is common even when the hard
//! limit is hundreds of thousands).  Raise only the child side of the fork to
//! the already-authorized hard limit immediately before exec.  The parent and
//! unrelated commands retain their original limits.

use std::ffi::{OsStr, OsString};
use std::io;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::process::CommandExt;
use std::process::{Command, ExitStatus};

/// Whether one inherited variable describes nextest's current test process.
///
/// Nextest deliberately exports an open-ended `NEXTEST_*` family (attempt,
/// run, test identity, and admission-slot coordinates) plus the exact
/// `NEXTEST` marker. None of those values can affect a cached Cargo build:
/// allowing them into a producer would give every retry a distinct cache
/// identity and expose a different environment to build scripts.
pub(crate) fn is_runtime_environment(name: &OsStr) -> bool {
    let name = name.as_bytes();
    name == b"NEXTEST" || name.starts_with(b"NEXTEST_")
}

/// Remove nextest's per-test runtime coordinates from one cached producer.
///
/// Inspect both the inherited process environment and explicit command
/// overrides. The latter keeps this correct when a caller constructed the
/// producer environment before normalization. `env_remove` is intentionally
/// command-local: the final nextest execution retains its complete runtime
/// environment for the tests it launches.
pub(crate) fn remove_runtime_environment(command: &mut Command) {
    let mut names = vec![OsString::from("NEXTEST")];
    names.extend(std::env::vars_os().map(|(name, _)| name));
    names.extend(command.get_envs().map(|(name, _)| name.to_os_string()));
    names.sort();
    names.dedup();
    for name in names {
        if is_runtime_environment(&name) {
            command.env_remove(name);
        }
    }
}

/// Install the child-only `RLIMIT_NOFILE` adjustment on a nextest command.
///
/// The closure runs after `fork` and before `exec`, so it deliberately does
/// nothing except the two rlimit syscalls and construction of an OS error.
/// Both Cargo's nextest plugin process and every test it subsequently spawns
/// inherit the raised soft limit.  Failure is a spawn failure rather than a
/// later, misleading `EMFILE` during the test run.
pub(crate) fn prepare(command: &mut Command) {
    // SAFETY: the pre-exec closure performs only async-signal-safe, allocation-
    // free libc calls. It captures no Rust state and returns before exec on
    // either syscall failure.
    unsafe {
        command.pre_exec(|| {
            let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
            if libc::getrlimit(libc::RLIMIT_NOFILE, limit.as_mut_ptr()) != 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: a successful getrlimit initialized the whole value.
            let mut limit = limit.assume_init();
            if limit.rlim_cur != limit.rlim_max {
                limit.rlim_cur = limit.rlim_max;
                if libc::setrlimit(libc::RLIMIT_NOFILE, &limit) != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
}

/// Run one nextest frontend through the shared signal-aware process owner.
pub(crate) fn run_status(mut command: Command) -> io::Result<ExitStatus> {
    prepare(&mut command);
    crate::interrupt::run_status(command)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const REEXEC_ENV: &str = "KTSTR_TEST_NEXTEST_NOFILE_REEXEC";

    #[test]
    fn cached_producer_removes_nextest_coordinates_but_keeps_build_inputs() {
        let mut command = Command::new("cargo");
        command
            .env("NEXTEST", "1")
            .env("NEXTEST_ATTEMPT", "7")
            .env("NEXTEST_TEST_GLOBAL_SLOT", "41")
            .env("NEXTEST_TEST_NAME", "ktstr::nested_retry")
            .env("SCHEDULER_FIXTURE_MODE", "semantic");

        remove_runtime_environment(&mut command);
        let environment = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(OsStr::to_owned)))
            .collect::<BTreeMap<_, _>>();

        for removed in [
            "NEXTEST",
            "NEXTEST_ATTEMPT",
            "NEXTEST_TEST_GLOBAL_SLOT",
            "NEXTEST_TEST_NAME",
        ] {
            assert_eq!(
                environment.get(OsStr::new(removed)),
                Some(&None),
                "{removed} must be explicitly absent from cached Cargo producers",
            );
        }
        assert_eq!(
            environment.get(OsStr::new("SCHEDULER_FIXTURE_MODE")),
            Some(&Some(OsString::from("semantic"))),
            "arbitrary build-script inputs must remain visible to the producer",
        );
    }

    fn nofile_limit() -> libc::rlimit {
        let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
        // SAFETY: getrlimit initializes `limit` on success.
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, limit.as_mut_ptr()) },
            0,
            "query RLIMIT_NOFILE: {}",
            io::Error::last_os_error(),
        );
        // SAFETY: the successful syscall above initialized the whole value.
        unsafe { limit.assume_init() }
    }

    #[test]
    fn child_soft_nofile_limit_is_raised_without_changing_parent() {
        if std::env::var_os(REEXEC_ENV).is_some() {
            let child = nofile_limit();
            assert_eq!(
                child.rlim_cur, child.rlim_max,
                "prepared nextest child must inherit its hard fd limit as soft",
            );
            return;
        }

        let before = nofile_limit();
        if before.rlim_max == 0 {
            // Such a process cannot exec a dynamically linked test child, and
            // already has no higher limit for the helper to install.
            return;
        }

        let test_name = std::thread::current()
            .name()
            .expect("libtest names the current test thread")
            .to_owned();
        let mut child = Command::new(std::env::current_exe().expect("current test executable"));
        child
            .args(["--exact", test_name.as_str(), "--nocapture"])
            .env(REEXEC_ENV, "1");

        // Force the forked child below its hard limit first, even on hosts
        // whose parent already has soft == hard. `prepare` is appended after
        // this hook and must restore soft == hard before exec. The parent is
        // never modified.
        let hard = before.rlim_max;
        // SAFETY: this test-only pre-exec hook calls setrlimit with a fully
        // initialized value and returns an OS error directly on failure.
        unsafe {
            child.pre_exec(move || {
                let lowered = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: hard,
                };
                if libc::setrlimit(libc::RLIMIT_NOFILE, &lowered) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        prepare(&mut child);

        let output = child.output().expect("run prepared nextest-limit child");
        assert!(
            output.status.success(),
            "prepared child failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let after = nofile_limit();
        assert_eq!(after.rlim_cur, before.rlim_cur, "parent soft limit changed");
        assert_eq!(after.rlim_max, before.rlim_max, "parent hard limit changed");
    }
}
