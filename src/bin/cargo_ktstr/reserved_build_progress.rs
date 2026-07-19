//! Live progress for reserved Cargo JSON builds.
//!
//! Cargo's machine-readable output remains captured byte-for-byte by the
//! interruptible child runner. This module observes both streams because
//! wrappers such as `cargo llvm-cov` forward nested Cargo JSON on stderr. It
//! incrementally counts `compiler-artifact` messages and records Cargo's
//! `build-finished` phase without consuming or rewriting JSON.

use std::io::IsTerminal;
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::interrupt::StdoutObserver;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
const TTY_TICK_INTERVAL: Duration = Duration::from_millis(80);

type PlainEmitter = Box<dyn FnMut(&str) + Send>;
type ByteEmitter = Box<dyn FnMut(&[u8]) + Send>;

enum ProgressTarget {
    Tty(ProgressBar),
    Plain(PlainEmitter),
}

/// Synchronous progress for the admission wait that precedes a reserved build.
pub(crate) struct ReservationWaitProgress {
    label: String,
    started: Instant,
    next_heartbeat: Instant,
    heartbeat_interval: Duration,
    target: ProgressTarget,
    finished: bool,
}

impl ReservationWaitProgress {
    pub(crate) fn start(cli_label: &str) -> Self {
        let label = escape_free(cli_label);
        let started = Instant::now();
        if std::io::stderr().is_terminal() {
            let bar = ProgressBar::new_spinner();
            bar.set_draw_target(ProgressDrawTarget::stderr());
            bar.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {msg} [{elapsed_precise}]")
                    .expect("reservation-wait progress template is valid"),
            );
            if !bar.is_hidden() {
                bar.set_message(format!("{label}: acquiring reserved build capacity"));
                bar.tick();
                return Self {
                    label,
                    started,
                    next_heartbeat: started + HEARTBEAT_INTERVAL,
                    heartbeat_interval: HEARTBEAT_INTERVAL,
                    target: ProgressTarget::Tty(bar),
                    finished: false,
                };
            }
        }
        Self::plain(
            label,
            started,
            HEARTBEAT_INTERVAL,
            Box::new(|line| {
                let _ = writeln!(std::io::stderr(), "{line}");
            }),
        )
    }

    fn plain(
        label: String,
        started: Instant,
        heartbeat_interval: Duration,
        mut emit: PlainEmitter,
    ) -> Self {
        emit(&format!("{label}: acquiring reserved build capacity"));
        Self {
            label,
            started,
            next_heartbeat: started + heartbeat_interval,
            heartbeat_interval,
            target: ProgressTarget::Plain(emit),
            finished: false,
        }
    }

    pub(crate) fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    fn tick_at(&mut self, now: Instant) {
        match &mut self.target {
            ProgressTarget::Tty(bar) => bar.tick(),
            ProgressTarget::Plain(emit) if now >= self.next_heartbeat => {
                emit(&format!(
                    "{}: waiting for reserved build capacity; elapsed={}",
                    self.label,
                    format_elapsed(now.saturating_duration_since(self.started)),
                ));
                self.next_heartbeat = now + self.heartbeat_interval;
            }
            ProgressTarget::Plain(_) => {}
        }
    }

    pub(crate) fn acquired(&mut self) {
        self.finish(format!(
            "{}: acquired reserved build capacity in {}",
            self.label,
            format_elapsed(self.started.elapsed()),
        ));
    }

    pub(crate) fn failed(&mut self, error: &dyn std::fmt::Display) {
        self.finish(format!(
            "{}: failed to acquire reserved build capacity after {}; error={}",
            self.label,
            format_elapsed(self.started.elapsed()),
            escape_free(&error.to_string()),
        ));
    }

    fn finish(&mut self, message: String) {
        match &mut self.target {
            ProgressTarget::Tty(bar) => bar.finish_with_message(message),
            ProgressTarget::Plain(emit) => emit(&message),
        }
        self.finished = true;
    }

    #[cfg(test)]
    fn plain_for_test(
        label: &str,
        started: Instant,
        heartbeat_interval: Duration,
        emit: PlainEmitter,
    ) -> Self {
        Self::plain(escape_free(label), started, heartbeat_interval, emit)
    }
}

impl Drop for ReservationWaitProgress {
    fn drop(&mut self) {
        if !self.finished
            && let ProgressTarget::Tty(bar) = &self.target
        {
            bar.finish_and_clear();
        }
    }
}

/// An observer for the otherwise-silent reserved Cargo pre-builds.
pub(crate) struct ReservedBuildProgress {
    label: String,
    description: String,
    started: Instant,
    next_heartbeat: Instant,
    heartbeat_interval: Duration,
    artifacts: usize,
    cargo_finished: Option<bool>,
    partial_stdout_line: Vec<u8>,
    partial_stderr_line: Vec<u8>,
    stderr_tee: ByteEmitter,
    target: ProgressTarget,
    finished: bool,
}

impl ReservedBuildProgress {
    /// Start reporting immediately.
    ///
    /// A TTY gets an indicatif spinner whose message includes the current
    /// Cargo phase and artifact count. CI gets a plain start line immediately
    /// and one escape-free heartbeat at least every ten seconds thereafter.
    pub(crate) fn start(cli_label: &str, description: &str) -> Self {
        let label = escape_free(cli_label);
        let description = escape_free(description);
        let started = Instant::now();

        if std::io::stderr().is_terminal() {
            let bar = ProgressBar::new_spinner();
            bar.set_draw_target(ProgressDrawTarget::stderr());
            bar.set_style(
                ProgressStyle::with_template("{spinner:.cyan} {msg} [{elapsed_precise}]")
                    .expect("reserved-build progress template is valid"),
            );
            if !bar.is_hidden() {
                let stderr_bar = bar.clone();
                let mut progress = Self {
                    label,
                    description,
                    started,
                    next_heartbeat: started + HEARTBEAT_INTERVAL,
                    heartbeat_interval: HEARTBEAT_INTERVAL,
                    artifacts: 0,
                    cargo_finished: None,
                    partial_stdout_line: Vec::new(),
                    partial_stderr_line: Vec::new(),
                    stderr_tee: Box::new(move |bytes| {
                        stderr_bar.suspend(|| write_stderr(bytes));
                    }),
                    target: ProgressTarget::Tty(bar),
                    finished: false,
                };
                progress.refresh_tty();
                return progress;
            }
        }

        Self::plain(
            label,
            description,
            started,
            HEARTBEAT_INTERVAL,
            Box::new(|line| {
                let _ = writeln!(std::io::stderr(), "{line}");
            }),
            Box::new(write_stderr),
        )
    }

    fn plain(
        label: String,
        description: String,
        started: Instant,
        heartbeat_interval: Duration,
        mut emit: PlainEmitter,
        stderr_tee: ByteEmitter,
    ) -> Self {
        emit(&format!("{label}: starting {description}"));
        Self {
            label,
            description,
            started,
            next_heartbeat: started + heartbeat_interval,
            heartbeat_interval,
            artifacts: 0,
            cargo_finished: None,
            partial_stdout_line: Vec::new(),
            partial_stderr_line: Vec::new(),
            stderr_tee,
            target: ProgressTarget::Plain(emit),
            finished: false,
        }
    }

    fn phase(&self) -> &'static str {
        match self.cargo_finished {
            None => "building",
            Some(true) => "cargo build finished",
            Some(false) => "cargo reported failure",
        }
    }

    fn cargo_finished_label(&self) -> &'static str {
        match self.cargo_finished {
            None => "pending",
            Some(true) => "success",
            Some(false) => "failure",
        }
    }

    fn tty_message(&self) -> String {
        format!(
            "{}: {} — {}; {} compiler artifacts",
            self.label,
            self.description,
            self.phase(),
            self.artifacts,
        )
    }

    fn refresh_tty(&mut self) {
        let message = self.tty_message();
        if let ProgressTarget::Tty(bar) = &self.target {
            bar.set_message(message);
            bar.tick();
        }
    }

    fn observe_json_line(&mut self, line: &[u8]) {
        const ARTIFACT: &[u8] = br#""compiler-artifact""#;
        const FINISHED: &[u8] = br#""build-finished""#;
        if !contains_bytes(line, ARTIFACT) && !contains_bytes(line, FINISHED) {
            return;
        }
        let Ok(message) = serde_json::from_slice::<serde_json::Value>(line) else {
            return;
        };
        match message.get("reason").and_then(|reason| reason.as_str()) {
            Some("compiler-artifact") => {
                self.artifacts = self.artifacts.saturating_add(1);
            }
            Some("build-finished") => {
                self.cargo_finished = message.get("success").and_then(|value| value.as_bool());
            }
            _ => {}
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8], stderr: bool) {
        let complete = {
            let partial = if stderr {
                &mut self.partial_stderr_line
            } else {
                &mut self.partial_stdout_line
            };
            partial.extend_from_slice(bytes);
            take_complete_lines(partial)
        };
        let Some(complete) = complete else {
            return;
        };
        let before = (self.artifacts, self.cargo_finished);
        for line in complete.split(|byte| *byte == b'\n') {
            if !line.is_empty() {
                self.observe_json_line(line);
            }
        }
        if before != (self.artifacts, self.cargo_finished) {
            self.refresh_tty();
        }
    }

    fn consume_trailing_lines(&mut self) {
        let stdout = std::mem::take(&mut self.partial_stdout_line);
        let stderr = std::mem::take(&mut self.partial_stderr_line);
        if stdout.is_empty() && stderr.is_empty() {
            return;
        }
        let before = (self.artifacts, self.cargo_finished);
        if !stdout.is_empty() {
            self.observe_json_line(&stdout);
        }
        if !stderr.is_empty() {
            self.observe_json_line(&stderr);
        }
        if before != (self.artifacts, self.cargo_finished) {
            self.refresh_tty();
        }
    }

    fn heartbeat_line(&self, now: Instant) -> String {
        format!(
            "{}: {} — {}; elapsed={}; compiler-artifacts={}; build-finished={}",
            self.label,
            self.description,
            self.phase(),
            format_elapsed(now.saturating_duration_since(self.started)),
            self.artifacts,
            self.cargo_finished_label(),
        )
    }

    fn tick_at(&mut self, now: Instant) {
        let heartbeat = (now >= self.next_heartbeat).then(|| self.heartbeat_line(now));
        match &mut self.target {
            ProgressTarget::Tty(bar) => bar.tick(),
            ProgressTarget::Plain(emit) => {
                if let Some(line) = heartbeat {
                    emit(&line);
                    self.next_heartbeat = now + self.heartbeat_interval;
                }
            }
        }
    }

    fn next_tick_in_at(&self, now: Instant) -> Duration {
        match &self.target {
            ProgressTarget::Tty(_) => TTY_TICK_INTERVAL,
            ProgressTarget::Plain(_) => self.next_heartbeat.saturating_duration_since(now),
        }
    }

    fn finish_message(&self, status: &ExitStatus) -> String {
        let outcome = if status.success() {
            "completed"
        } else {
            "failed"
        };
        format!(
            "{}: {outcome} {} in {}; compiler-artifacts={}; \
             build-finished={}; exit={}",
            self.label,
            self.description,
            format_elapsed(self.started.elapsed()),
            self.artifacts,
            self.cargo_finished_label(),
            exit_label(status),
        )
    }

    fn finish_status(&mut self, status: &ExitStatus) {
        self.consume_trailing_lines();
        let message = self.finish_message(status);
        match &mut self.target {
            ProgressTarget::Tty(bar) => bar.finish_with_message(message),
            ProgressTarget::Plain(emit) => emit(&message),
        }
        self.finished = true;
    }

    fn finish_error(&mut self, error: &std::io::Error) {
        self.consume_trailing_lines();
        let message = format!(
            "{}: failed {} after {}; compiler-artifacts={}; \
             build-finished={}; error={}",
            self.label,
            self.description,
            format_elapsed(self.started.elapsed()),
            self.artifacts,
            self.cargo_finished_label(),
            escape_free(&error.to_string()),
        );
        match &mut self.target {
            ProgressTarget::Tty(bar) => bar.finish_with_message(message),
            ProgressTarget::Plain(emit) => emit(&message),
        }
        self.finished = true;
    }

    #[cfg(test)]
    fn plain_for_test(
        label: &str,
        description: &str,
        started: Instant,
        heartbeat_interval: Duration,
        emit: PlainEmitter,
    ) -> Self {
        Self::plain_for_test_with_stderr(
            label,
            description,
            started,
            heartbeat_interval,
            emit,
            Box::new(|_| {}),
        )
    }

    #[cfg(test)]
    fn plain_for_test_with_stderr(
        label: &str,
        description: &str,
        started: Instant,
        heartbeat_interval: Duration,
        emit: PlainEmitter,
        stderr_tee: ByteEmitter,
    ) -> Self {
        Self::plain(
            escape_free(label),
            escape_free(description),
            started,
            heartbeat_interval,
            emit,
            stderr_tee,
        )
    }
}

impl StdoutObserver for ReservedBuildProgress {
    fn observe_stdout(&mut self, bytes: &[u8]) {
        self.observe_bytes(bytes, false);
    }

    fn observe_stderr(&mut self, bytes: &[u8]) {
        (self.stderr_tee)(bytes);
        self.observe_bytes(bytes, true);
    }

    fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    fn next_tick_in(&self) -> Duration {
        self.next_tick_in_at(Instant::now())
    }

    fn finished(&mut self, status: &ExitStatus) {
        self.finish_status(status);
    }

    fn failed(&mut self, error: &std::io::Error) {
        self.finish_error(error);
    }
}

impl Drop for ReservedBuildProgress {
    fn drop(&mut self) {
        if !self.finished {
            if let ProgressTarget::Tty(bar) = &self.target {
                bar.finish_and_clear();
            }
        }
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{:.1}s", elapsed.as_secs_f64())
    }
}

fn exit_label(status: &ExitStatus) -> String {
    match (status.code(), status.signal()) {
        (Some(0), _) => "success".to_string(),
        (Some(code), _) => format!("code {code}"),
        (None, Some(signal)) => format!("signal {signal}"),
        (None, None) => "unknown".to_string(),
    }
}

fn escape_free(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|candidate| candidate == needle)
}

fn take_complete_lines(partial: &mut Vec<u8>) -> Option<Vec<u8>> {
    let last_newline = partial.iter().rposition(|byte| *byte == b'\n')?;
    let tail = partial.split_off(last_newline + 1);
    Some(std::mem::replace(partial, tail))
}

fn write_stderr(bytes: &[u8]) {
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    let _ = stderr.write_all(bytes);
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::{Arc, Mutex};

    fn capture(
        started: Instant,
        heartbeat_interval: Duration,
    ) -> (ReservedBuildProgress, Arc<Mutex<Vec<String>>>) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let output = Arc::clone(&lines);
        let progress = ReservedBuildProgress::plain_for_test(
            "cargo ktstr verifier",
            "reserved harness pre-build",
            started,
            heartbeat_interval,
            Box::new(move |line| output.lock().expect("capture lock").push(line.to_string())),
        );
        (progress, lines)
    }

    #[test]
    fn reservation_wait_reports_start_heartbeat_and_terminal_outcome() {
        let started = Instant::now();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let output = Arc::clone(&lines);
        let mut progress = ReservationWaitProgress::plain_for_test(
            "cargo ktstr\nverifier",
            started,
            Duration::from_secs(10),
            Box::new(move |line| output.lock().expect("capture lock").push(line.to_string())),
        );
        assert_eq!(
            lines.lock().expect("lines")[0],
            "cargo ktstr verifier: acquiring reserved build capacity",
        );
        progress.tick_at(started + Duration::from_secs(9));
        assert_eq!(lines.lock().expect("lines").len(), 1);
        progress.tick_at(started + Duration::from_secs(10));
        progress.acquired();

        let lines = lines.lock().expect("lines");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("waiting for reserved build capacity"));
        assert!(lines[1].contains("elapsed=10.0s"));
        assert!(lines[2].contains("acquired reserved build capacity"));
        assert!(lines.iter().all(|line| !line.contains('\u{1b}')));
    }

    #[test]
    fn plain_progress_starts_immediately_and_heartbeats_without_escapes() {
        let started = Instant::now();
        let (mut progress, lines) = capture(started, Duration::from_secs(10));
        assert_eq!(
            lines.lock().expect("lines")[0],
            "cargo ktstr verifier: starting reserved harness pre-build",
        );

        progress.tick_at(started + Duration::from_secs(9));
        assert_eq!(lines.lock().expect("lines").len(), 1);
        progress.tick_at(started + Duration::from_secs(10));

        let lines = lines.lock().expect("lines");
        assert_eq!(lines.len(), 2);
        assert!(lines[1].contains("elapsed=10.0s"));
        assert!(lines[1].contains("compiler-artifacts=0"));
        assert!(lines[1].contains("build-finished=pending"));
        assert!(
            lines.iter().all(|line| !line.contains('\u{1b}')),
            "CI progress is escape-free",
        );
    }

    #[test]
    fn fragmented_json_updates_artifacts_and_preserves_build_phase() {
        let started = Instant::now();
        let (mut progress, lines) = capture(started, Duration::from_secs(10));
        progress.observe_stdout(br#"{"reason":"compiler-art"#);
        progress.observe_stdout(
            b"ifact\"}\n{\"reason\":\"compiler-artifact\"}\n\
              {\"reason\":\"build-finished\",\"success\":true}",
        );
        progress.tick_at(started + Duration::from_secs(10));
        progress.finish_status(&ExitStatus::from_raw(0));

        let lines = lines.lock().expect("lines");
        assert!(lines[1].contains("compiler-artifacts=2"));
        assert!(lines[1].contains("build-finished=pending"));
        let completion = lines.last().expect("completion");
        assert!(completion.contains("completed reserved harness pre-build"));
        assert!(completion.contains("compiler-artifacts=2"));
        assert!(completion.contains("build-finished=success"));
        assert!(completion.contains("exit=success"));
    }

    #[test]
    fn stderr_is_teed_exactly_and_parsed_independently_from_stdout() {
        let started = Instant::now();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let line_output = Arc::clone(&lines);
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let stderr_output = Arc::clone(&stderr);
        let mut progress = ReservedBuildProgress::plain_for_test_with_stderr(
            "cargo ktstr verifier",
            "coverage harness pre-build",
            started,
            Duration::from_secs(10),
            Box::new(move |line| {
                line_output
                    .lock()
                    .expect("line capture lock")
                    .push(line.to_string())
            }),
            Box::new(move |bytes| {
                stderr_output
                    .lock()
                    .expect("stderr capture lock")
                    .extend_from_slice(bytes)
            }),
        );
        let stderr_first = b"warning: live diagnostic\n{\"reason\":\"compiler-art";
        let stderr_last =
            b"ifact\"}\n{\"reason\":\"build-finished\",\"success\":true}\ntrailing diagnostic";

        progress.observe_stdout(br#"{"reason":"compiler-art"#);
        progress.observe_stderr(stderr_first);
        progress.observe_stdout(b"ifact\"}\n");
        progress.observe_stderr(stderr_last);
        progress.finish_status(&ExitStatus::from_raw(0));

        let mut expected_stderr = stderr_first.to_vec();
        expected_stderr.extend_from_slice(stderr_last);
        assert_eq!(*stderr.lock().expect("stderr bytes"), expected_stderr);
        let lines = lines.lock().expect("lines");
        let completion = lines.last().expect("completion");
        assert!(completion.contains("compiler-artifacts=2"));
        assert!(completion.contains("build-finished=success"));
    }

    #[test]
    fn nonzero_and_observer_errors_emit_explicit_failures() {
        let started = Instant::now();
        let (mut nonzero, nonzero_lines) = capture(started, Duration::from_secs(10));
        nonzero.observe_stdout(b"{\"reason\":\"build-finished\",\"success\":false}\n");
        nonzero.finish_status(&ExitStatus::from_raw(7 << 8));
        let nonzero_lines = nonzero_lines.lock().expect("nonzero lines");
        let completion = nonzero_lines.last().expect("nonzero completion");
        assert!(completion.contains("failed reserved harness pre-build"));
        assert!(completion.contains("build-finished=failure"));
        assert!(completion.contains("exit=code 7"));

        let (mut io_failure, io_lines) = capture(started, Duration::from_secs(10));
        io_failure.finish_error(&std::io::Error::other("poll\nfailed\u{1b}[31m"));
        let io_lines = io_lines.lock().expect("io lines");
        let completion = io_lines.last().expect("I/O completion");
        assert!(completion.contains("error=poll failed [31m"));
        assert!(!completion.contains('\n'));
        assert!(!completion.contains('\u{1b}'));
    }
}
