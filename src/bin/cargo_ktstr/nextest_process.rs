//! Process preparation shared by every cargo-ktstr nextest frontend.
//!
//! Nextest keeps pipes and process handles open for admitted tests.  Its
//! effectively-unbounded admission mode must therefore not inherit a small
//! login-session `RLIMIT_NOFILE` soft limit (1024 is common even when the hard
//! limit is hundreds of thousands).  Raise only the child side of the fork to
//! the already-authorized hard limit immediately before exec.  The parent and
//! unrelated commands retain their original limits.

use std::ffi::{OsStr, OsString};
use std::io::{self, IsTerminal as _};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::process::CommandExt;
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

const NEXTEST_PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

/// Keep the terminal tail of a nextest run visibly alive without interposing
/// on its stdout or stderr.
///
/// Nextest already streams every completed test directly to the inherited
/// terminal. A large VM run can nevertheless go quiet after the fast tests
/// finish because every remaining attempt is still running or waiting for
/// admission. Piping nextest through cargo-ktstr to detect that silence would
/// duplicate and retain a potentially huge log, so the existing child-owner
/// loop emits a low-frequency phase heartbeat instead.
struct NextestRunProgress {
    started: Instant,
    next_heartbeat: Instant,
    heartbeat_interval: Duration,
}

impl NextestRunProgress {
    fn new(heartbeat_interval: Duration) -> Self {
        Self::new_at(Instant::now(), heartbeat_interval)
    }

    fn new_at(started: Instant, heartbeat_interval: Duration) -> Self {
        assert!(
            !heartbeat_interval.is_zero(),
            "nextest heartbeat interval must be non-zero"
        );
        Self {
            started,
            next_heartbeat: started + heartbeat_interval,
            heartbeat_interval,
        }
    }

    fn tick_at(&mut self, now: Instant) -> Option<String> {
        if now >= self.next_heartbeat {
            self.next_heartbeat = now + self.heartbeat_interval;
            Some(nextest_heartbeat_line(
                now.saturating_duration_since(self.started),
            ))
        } else {
            None
        }
    }

    fn next_tick_in_at(&self, now: Instant) -> Duration {
        self.next_heartbeat.saturating_duration_since(now)
    }
}

impl crate::interrupt::StdoutObserver for NextestRunProgress {
    fn observe_stdout(&mut self, _bytes: &[u8]) {}

    fn tick(&mut self) {
        if let Some(line) = self.tick_at(Instant::now()) {
            ktstr::cli::print_status_line(&line);
        }
    }

    fn next_tick_in(&self) -> Duration {
        self.next_tick_in_at(Instant::now())
    }

    fn finished(&mut self, _status: &ExitStatus) {}

    fn failed(&mut self, _error: &io::Error) {}
}

fn nextest_heartbeat_line(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    let elapsed = if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{:.1}s", elapsed.as_secs_f64())
    };
    format!("cargo ktstr: nextest run still active; elapsed={elapsed}")
}

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

/// Stamp the production lock-dir reference onto a nextest run command.
///
/// Resolved in this cargo-ktstr parent process and inherited by every
/// nextest test process (and any child it re-execs), it is the sealed
/// reference the library's misclassification guard compares against:
/// only a test that resolves this exact shared namespace — not one
/// redirected into an isolated temp dir — is a misclassified resource
/// user. Every nextest frontend spawns through [`run_status`], so
/// stamping here covers the ordinary test, coverage, verifier, and
/// replay paths in one place.
pub(crate) fn stamp_production_lock_dir(command: &mut Command) {
    command.env(
        ktstr::KTSTR_PRODUCTION_LOCK_DIR_ENV,
        ktstr::resolve_production_lock_dir(),
    );
}

/// Run one nextest frontend through the shared signal-aware process owner.
pub(crate) fn run_status(mut command: Command) -> io::Result<ExitStatus> {
    stamp_production_lock_dir(&mut command);
    prepare(&mut command);
    // Nextest already owns an interactive progress bar. Adding periodic lines
    // underneath it would only make the terminal flicker. GitHub's self-hosted
    // runner can present stderr through a pseudo-terminal while its web log is
    // still line-oriented, so GITHUB_ACTIONS explicitly selects heartbeats.
    let github_actions = std::env::var_os("GITHUB_ACTIONS").is_some();
    if !should_emit_heartbeat(io::stderr().is_terminal(), github_actions) {
        crate::interrupt::run_status(command)
    } else {
        crate::interrupt::run_status_observed(
            command,
            NextestRunProgress::new(NEXTEST_PROGRESS_INTERVAL),
        )
    }
}

fn should_emit_heartbeat(stderr_is_terminal: bool, github_actions: bool) -> bool {
    !stderr_is_terminal || github_actions
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

    #[test]
    fn run_command_carries_production_lock_dir_reference() {
        let mut command = Command::new("cargo");
        stamp_production_lock_dir(&mut command);
        let stamped = command
            .get_envs()
            .find(|(name, _)| *name == OsStr::new(ktstr::KTSTR_PRODUCTION_LOCK_DIR_ENV))
            .and_then(|(_, value)| value)
            .map(OsStr::to_owned);
        assert_eq!(
            stamped,
            Some(ktstr::resolve_production_lock_dir().into_os_string()),
            "every nextest run must export the parent-resolved shared lock dir \
             so the misclassification guard can seal against it",
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

    #[test]
    fn nextest_progress_ticks_only_at_heartbeat_deadlines() {
        let started = Instant::now();
        let mut progress = NextestRunProgress::new_at(started, Duration::from_secs(10));

        assert_eq!(progress.next_tick_in_at(started), Duration::from_secs(10));
        assert_eq!(progress.tick_at(started + Duration::from_secs(9)), None);
        assert_eq!(
            progress.tick_at(started + Duration::from_secs(10)),
            Some("cargo ktstr: nextest run still active; elapsed=10.0s".to_string())
        );
        assert_eq!(
            progress.next_tick_in_at(started + Duration::from_secs(10)),
            Duration::from_secs(10),
        );

        assert_eq!(progress.tick_at(started + Duration::from_secs(19)), None);
        assert_eq!(
            progress.tick_at(started + Duration::from_secs(20)),
            Some("cargo ktstr: nextest run still active; elapsed=20.0s".to_string())
        );
    }

    #[test]
    fn nextest_progress_formats_minute_elapsed_time() {
        assert_eq!(
            nextest_heartbeat_line(Duration::from_secs(75)),
            "cargo ktstr: nextest run still active; elapsed=1m 15s",
        );
    }

    #[test]
    fn github_actions_keeps_heartbeats_on_a_pseudoterminal() {
        assert!(should_emit_heartbeat(false, false));
        assert!(!should_emit_heartbeat(true, false));
        assert!(should_emit_heartbeat(true, true));
    }
}
