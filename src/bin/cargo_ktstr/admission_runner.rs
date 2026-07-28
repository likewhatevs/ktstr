//! Hidden Cargo/nextest target runner for pre-exec VM admission.
//!
//! Nextest otherwise starts every heavyweight test binary before ktstr can
//! place the cell in its cross-process CPU/LLC queue. This runner reads the
//! generated cell's link-retained admission stamp, acquires the production
//! reservation while it is still the small cargo-ktstr process, and transfers
//! that reservation through a sealed same-PID exec handoff.

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::cli::KtstrCommand;
use crate::feature_discovery::effective_target_context;

const SUBCOMMAND: &str = "__ktstr_admission_runner";
const TARGET_ENV_KEY_ENV: &str = "KTSTR_ADMISSION_TARGET_ENV_KEY";
const CHAINED_RUNNER_ENV: &str = "KTSTR_ADMISSION_CHAINED_RUNNER";
const ORIGINAL_RUNNER_ENV: &str = "KTSTR_ADMISSION_ORIGINAL_RUNNER";

/// Cargo's resolved executable-with-arguments representation, encoded without
/// assuming Unix paths are UTF-8.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EncodedRunner {
    program: Vec<u8>,
    args: Vec<Vec<u8>>,
}

impl EncodedRunner {
    fn from_cargo(runner: cargo_config2::PathAndArgs) -> Result<Self, String> {
        let program = runner.path.as_os_str().as_bytes().to_vec();
        if program.is_empty() {
            return Err("Cargo resolved an empty target-runner program".to_string());
        }
        Ok(Self {
            program,
            args: runner
                .args
                .into_iter()
                .map(|argument| argument.as_bytes().to_vec())
                .collect(),
        })
    }

    fn command(&self) -> Result<Command, String> {
        if self.program.is_empty() {
            return Err("encoded Cargo target runner has an empty program".to_string());
        }
        let mut command = Command::new(OsString::from_vec(self.program.clone()));
        command.args(self.args.iter().cloned().map(OsString::from_vec));
        Ok(command)
    }
}

/// Dispatch the private target-runner argv shape before ordinary cargo-ktstr
/// startup. Returns normally when this is a user-facing invocation; a hidden
/// invocation either replaces itself with the test process or exits with a
/// diagnostic.
pub(crate) fn dispatch_if_requested() {
    let mut arguments = std::env::args_os();
    let _program = arguments.next();
    if arguments.next().as_deref() != Some(OsStr::new(SUBCOMMAND)) {
        return;
    }

    let code = match run(arguments.collect()) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: cargo ktstr admission runner: {error}");
            1
        }
    };
    std::process::exit(code);
}

/// Install the wrapper for every cargo-ktstr path that actually delegates test
/// execution to nextest. Report-only/dry-run/admin commands retain their exact
/// environment and do not pay Cargo-config resolution cost.
pub(crate) fn install_for_command(command: &KtstrCommand) -> Result<(), String> {
    let Some(arguments) = nextest_arguments(command) else {
        return Ok(());
    };
    reject_cli_config_override(arguments)?;

    let target = effective_target_context(arguments)
        .map_err(|error| format!("resolve effective Cargo target: {error}"))?;
    let target_name = target.target_name().to_string();
    let config = cargo_config2::Config::load()
        .map_err(|error| format!("load effective Cargo configuration: {error}"))?;
    let runner = config
        .runner(target.cargo_target())
        .map_err(|error| format!("resolve Cargo target runner for {target_name}: {error}"))?
        .map(EncodedRunner::from_cargo)
        .transpose()?;
    let env_key = target_runner_env_key(&target_name);
    let original_env = std::env::var_os(&env_key);
    let encoded_runner = runner
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| format!("encode existing Cargo target runner: {error}"))?;
    let wrapper = format!("/proc/{}/exe {SUBCOMMAND}", std::process::id(),);

    // SAFETY: this runs during cargo-ktstr's single-threaded startup, before
    // its interrupt guard or any command dispatcher can create a persistent
    // worker. Cargo-config resolution above joins every compiler subprocess.
    unsafe {
        std::env::set_var(TARGET_ENV_KEY_ENV, &env_key);
        match encoded_runner {
            Some(encoded) => std::env::set_var(CHAINED_RUNNER_ENV, encoded),
            None => std::env::remove_var(CHAINED_RUNNER_ENV),
        }
        match original_env {
            Some(value) => std::env::set_var(ORIGINAL_RUNNER_ENV, value),
            None => std::env::remove_var(ORIGINAL_RUNNER_ENV),
        }
        std::env::set_var(env_key, wrapper);
    }
    Ok(())
}

fn nextest_arguments(command: &KtstrCommand) -> Option<&[String]> {
    match command {
        KtstrCommand::Test { args, .. }
            if crate::run_cargo::cargo_sub_uses_nextest(crate::run_cargo::TEST_SUB_ARGV, args) =>
        {
            Some(args)
        }
        KtstrCommand::Coverage { args, .. }
            if crate::run_cargo::cargo_sub_uses_nextest(
                crate::run_cargo::COVERAGE_SUB_ARGV,
                args,
            ) =>
        {
            Some(args)
        }
        KtstrCommand::LlvmCov { args, .. }
            if crate::run_cargo::cargo_sub_uses_nextest(
                crate::run_cargo::LLVM_COV_SUB_ARGV,
                args,
            ) =>
        {
            Some(args)
        }
        KtstrCommand::Verifier { args, .. } => Some(args),
        KtstrCommand::Replay {
            exec: true, args, ..
        } => Some(args),
        // perf-delta's production path starts fresh cargo-ktstr test children;
        // each child installs a PID-stable wrapper for its own lifetime.
        KtstrCommand::PerfDelta { .. }
        | KtstrCommand::Stats { .. }
        | KtstrCommand::Replay { exec: false, .. }
        | KtstrCommand::Kernel { .. }
        | KtstrCommand::Completions { .. }
        | KtstrCommand::ShowHost
        | KtstrCommand::ShowThresholds { .. }
        | KtstrCommand::Affected { .. }
        | KtstrCommand::Export { .. }
        | KtstrCommand::Locks { .. }
        | KtstrCommand::Shell { .. } => None,
        // The guarded arms above already cover every Test/Coverage/LlvmCov
        // value which does not execute nextest.
        KtstrCommand::Test { .. }
        | KtstrCommand::Coverage { .. }
        | KtstrCommand::LlvmCov { .. } => None,
    }
}

fn reject_cli_config_override(arguments: &[String]) -> Result<(), String> {
    if arguments
        .iter()
        .take_while(|argument| argument.as_str() != "--")
        .any(|argument| argument == "--config" || argument.starts_with("--config="))
    {
        return Err(
            "cannot preserve an effective target runner supplied through Cargo `--config`: \
             move the runner to CARGO_TARGET_<TRIPLE>_RUNNER or a discovered \
             .cargo/config.toml before using ktstr pre-admission"
                .to_string(),
        );
    }
    Ok(())
}

fn target_runner_env_key(target: &str) -> String {
    let normalized = target
        .chars()
        .map(|character| match character {
            '-' | '.' => '_',
            other => other.to_ascii_uppercase(),
        })
        .collect::<String>();
    format!("CARGO_TARGET_{normalized}_RUNNER")
}

fn run(arguments: Vec<OsString>) -> Result<(), String> {
    let (binary, libtest_arguments) = arguments
        .split_first()
        .ok_or_else(|| "target runner received no test executable".to_string())?;
    let chained = decode_chained_runner()?;

    if is_listing(libtest_arguments) {
        return exec_passthrough(binary, libtest_arguments, chained.as_ref());
    }

    let exact_name = exact_test_name(libtest_arguments)?;
    let descriptor = ktstr::test_support::read_admission_cell_stamp(Path::new(binary), exact_name)?;
    let Some(descriptor) = descriptor else {
        return exec_passthrough(binary, libtest_arguments, chained.as_ref());
    };
    if descriptor.host_only || descriptor.kind == ktstr::test_support::AdmissionCellKind::Host {
        return exec_passthrough(binary, libtest_arguments, chained.as_ref());
    }
    if chained.is_some() {
        return Err(format!(
            "generated VM cell {:?} cannot compose the configured Cargo target runner: \
             pre-admission transfers lock ownership through a same-PID exec, while an \
             arbitrary runner may spawn the test. Remove the target runner for this native \
             ktstr run",
            descriptor.exact_name,
        ));
    }

    let mut command = Command::new(binary);
    command.args(libtest_arguments);
    restore_target_runner_environment(&mut command)?;
    let guard = ktstr::pre_admit_test_cell(descriptor)
        .map_err(|error| format!("acquire test-cell admission: {error:#}"))?;
    guard
        .exec(command)
        .map_err(|error| format!("exec pre-admitted test cell: {error:#}"))
}

fn is_listing(arguments: &[OsString]) -> bool {
    arguments
        .iter()
        .any(|argument| argument == OsStr::new("--list"))
}

fn exact_test_name(arguments: &[OsString]) -> Result<&str, String> {
    let mut exact = arguments
        .windows(2)
        .filter(|window| window[0] == OsStr::new("--exact"));
    let name = exact.next().ok_or_else(|| {
        "nextest execution omitted its required `--exact <test-name>` pair".to_string()
    })?[1]
        .to_str()
        .ok_or_else(|| "nextest exact test name is not valid UTF-8".to_string())?;
    if exact.next().is_some() {
        return Err("nextest execution supplied `--exact` more than once".to_string());
    }
    if name.is_empty() || name.starts_with('-') {
        return Err(format!("invalid nextest exact test name {name:?}"));
    }
    Ok(name)
}

fn decode_chained_runner() -> Result<Option<EncodedRunner>, String> {
    let Some(encoded) = std::env::var_os(CHAINED_RUNNER_ENV) else {
        return Ok(None);
    };
    let encoded = encoded
        .to_str()
        .ok_or_else(|| format!("{CHAINED_RUNNER_ENV} is not valid UTF-8"))?;
    serde_json::from_str(encoded)
        .map(Some)
        .map_err(|error| format!("decode {CHAINED_RUNNER_ENV}: {error}"))
}

fn exec_passthrough(
    binary: &OsStr,
    arguments: &[OsString],
    chained: Option<&EncodedRunner>,
) -> Result<(), String> {
    let mut command = match chained {
        Some(runner) => {
            let mut command = runner.command()?;
            command.arg(binary);
            command
        }
        None => Command::new(binary),
    };
    command.args(arguments);
    restore_target_runner_environment(&mut command)?;

    use std::os::unix::process::CommandExt;
    let description = format!("{command:?}");
    Err(format!(
        "exec pass-through test via {description}: {}",
        command.exec(),
    ))
}

/// Remove the private wrapper from the executed test's environment. An
/// inherited environment-defined runner is restored byte-for-byte; a runner
/// sourced from Cargo config is recovered naturally by nested Cargo after the
/// temporary environment override is removed.
fn restore_target_runner_environment(command: &mut Command) -> Result<(), String> {
    let env_key = std::env::var(TARGET_ENV_KEY_ENV)
        .map_err(|_| format!("hidden runner is missing {TARGET_ENV_KEY_ENV}"))?;
    match std::env::var_os(ORIGINAL_RUNNER_ENV) {
        Some(value) => {
            command.env(&env_key, value);
        }
        None => {
            command.env_remove(&env_key);
        }
    }
    command.env_remove(TARGET_ENV_KEY_ENV);
    command.env_remove(CHAINED_RUNNER_ENV);
    command.env_remove(ORIGINAL_RUNNER_ENV);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    fn command_uses_admission_runner(arguments: &[&str]) -> bool {
        let arguments = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        let rewritten = crate::argsplit::rewrite(&crate::cli::Cargo::command(), &arguments);
        let crate::cli::Cargo {
            command: crate::cli::CargoSub::Ktstr(command),
        } = crate::cli::Cargo::try_parse_from(rewritten).unwrap();
        nextest_arguments(&command.command).is_some()
    }

    fn os(arguments: &[&str]) -> Vec<OsString> {
        arguments.iter().map(OsString::from).collect()
    }

    #[test]
    fn target_runner_env_key_matches_cargo_normalization() {
        assert_eq!(
            target_runner_env_key("x86_64-unknown-linux-gnu"),
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
        );
    }

    #[test]
    fn exact_name_matches_nextest_argument_shape() {
        let arguments = os(&["--exact", "ktstr/cell/kernel", "--nocapture"]);
        assert_eq!(exact_test_name(&arguments).unwrap(), "ktstr/cell/kernel");
    }

    #[test]
    fn listing_is_never_treated_as_a_test_cell() {
        assert!(is_listing(&os(&["--list", "--format", "terse"])));
    }

    #[test]
    fn encoded_runner_roundtrips_non_utf8_arguments() {
        let encoded = EncodedRunner {
            program: b"runner".to_vec(),
            args: vec![vec![b'a', 0xff]],
        };
        let json = serde_json::to_string(&encoded).unwrap();
        assert_eq!(
            serde_json::from_str::<EncodedRunner>(&json).unwrap(),
            encoded
        );
    }

    #[test]
    fn cli_config_detection_stops_at_test_binary_separator() {
        assert!(
            reject_cli_config_override(&["--config".into(), "net.offline=true".into(),]).is_err()
        );
        assert!(
            reject_cli_config_override(&["--".into(), "--config=belongs-to-test".into(),]).is_ok()
        );
    }

    #[test]
    fn installs_only_for_nextest_execution_frontends() {
        assert!(command_uses_admission_runner(&["cargo", "ktstr", "test"]));
        assert!(command_uses_admission_runner(&[
            "cargo", "ktstr", "coverage"
        ]));
        assert!(!command_uses_admission_runner(&[
            "cargo", "ktstr", "coverage", "--no-run"
        ]));
        assert!(command_uses_admission_runner(&[
            "cargo", "ktstr", "llvm-cov", "nextest"
        ]));
        assert!(!command_uses_admission_runner(&[
            "cargo", "ktstr", "llvm-cov", "nextest", "--no-run"
        ]));
        assert!(!command_uses_admission_runner(&[
            "cargo", "ktstr", "llvm-cov", "report"
        ]));
        assert!(command_uses_admission_runner(&[
            "cargo", "ktstr", "verifier"
        ]));
        assert!(command_uses_admission_runner(&[
            "cargo", "ktstr", "replay", "--exec"
        ]));
        assert!(!command_uses_admission_runner(&[
            "cargo", "ktstr", "replay"
        ]));
    }
}
