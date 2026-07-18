//! Shared policy for every gix source-acquisition path.
//!
//! Keep this module build-script safe: `build.rs`, `scx-ktstr/build.rs`, and
//! the runtime fetcher all include it directly.

use std::path::PathBuf;

use super::gix;
use gix::bstr::ByteSlice;

const HTTP_CONNECT_TIMEOUT_MS: u64 = 20_000;
const HTTP_LOW_SPEED_LIMIT: u32 = 1024;
const HTTP_LOW_SPEED_TIME_SECONDS: u64 = 30;

/// A source whose implementation is known not to launch a helper process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InProcessSource {
    /// Smart HTTP handled by gix's in-process curl backend.
    Http,
    /// Encrypted smart HTTP handled by gix's in-process curl backend.
    Https,
    /// A repository opened directly through gix's object and ref APIs.
    Local(PathBuf),
}

/// Parse and classify a source before any repository, remote, or connection is
/// created.
///
/// gix's file, ssh, git, and external transports can invoke
/// `git-upload-pack`, `ssh`, or another configured helper. Local repositories
/// are therefore returned as paths and must be opened directly instead of
/// being handed to `remote_at*()`.
pub(crate) fn classify_source(url: &str) -> Result<InProcessSource, String> {
    let parsed = gix::Url::from_bytes(url.as_bytes().as_bstr())
        .map_err(|err| format!("parse source URL {url}: {err}"))?;
    let scheme = parsed.scheme.clone();
    match &scheme {
        gix::url::Scheme::Http => Ok(InProcessSource::Http),
        gix::url::Scheme::Https => Ok(InProcessSource::Https),
        gix::url::Scheme::File => {
            if parsed.user.is_some()
                || parsed.password.is_some()
                || parsed.port.is_some()
                || parsed
                    .host
                    .as_deref()
                    .is_some_and(|host| !host.eq_ignore_ascii_case("localhost"))
            {
                return Err(format!(
                    "local source {parsed} names a remote host or credentials; \
                     use a local path or file:///absolute/path"
                ));
            }
            Ok(InProcessSource::Local(gix::path::from_bstring(parsed.path)))
        }
        gix::url::Scheme::Ssh | gix::url::Scheme::Git | gix::url::Scheme::Ext(_) => Err(format!(
            "ktstr refuses {} transport for {parsed} because it can start an \
             external helper; use http(s):// for gix's in-process curl \
             transport or file:///path for a directly opened local repository",
            scheme
        )),
    }
}

/// Hermetic gix repository options shared by local and HTTP acquisition.
pub(crate) fn open_options() -> gix::open::Options {
    open_options_with_transport_limits(
        HTTP_CONNECT_TIMEOUT_MS,
        HTTP_LOW_SPEED_LIMIT,
        HTTP_LOW_SPEED_TIME_SECONDS,
        None,
    )
}

/// Test-only overrides are passed by the build-acquisition integration fixture
/// to exercise the real curl low-speed path without waiting 30 seconds.
pub(crate) fn open_options_with_transport_limits(
    connect_timeout_ms: u64,
    low_speed_limit: u32,
    low_speed_time_seconds: u64,
    no_proxy: Option<&str>,
) -> gix::open::Options {
    use gix::sec::trust::DefaultForLevel;

    let mut options = gix::open::Options::default_for_level(gix::sec::Trust::Full);
    options.permissions.config.system = false;
    options.permissions.config.git = false;
    options.permissions.config.user = false;
    options.permissions.config.env = false;
    options.permissions.config.git_binary = false;
    options.permissions.config.includes = false;
    options.permissions.attributes.system = false;
    options.permissions.attributes.git = false;
    options.permissions.attributes.git_binary = false;
    options.permissions.env.git_prefix = gix::sec::Permission::Deny;
    options.permissions.env.ssh_prefix = gix::sec::Permission::Deny;

    let mut overrides = vec![
        "core.logAllRefUpdates=false".to_string(),
        "credential.helper=".to_string(),
        "core.askPass=".to_string(),
        "gitoxide.credentials.terminalPrompt=false".to_string(),
        format!("gitoxide.http.connectTimeout={connect_timeout_ms}"),
        format!("http.lowSpeedLimit={low_speed_limit}"),
        format!("http.lowSpeedTime={low_speed_time_seconds}"),
    ];
    if let Some(no_proxy) = no_proxy {
        overrides.push(format!("gitoxide.http.noProxy={no_proxy}"));
    }
    options.config_overrides(overrides)
}

/// Refuse authentication uniformly for public source acquisition.
///
/// Installing this callback on every HTTP connection bypasses gix's
/// configured credential cascade entirely. The open policy above also clears
/// helpers and askpass so proxy authentication and policy inspection stay
/// hermetic.
pub(crate) fn reject_credentials(
    _action: gix::credentials::helper::Action,
) -> gix::credentials::protocol::Result {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{InProcessSource, classify_source};

    #[test]
    fn helper_capable_schemes_are_rejected_during_classification() {
        for url in [
            "ssh://example.invalid/repository.git",
            "git://example.invalid/repository.git",
            "git@example.invalid:repository.git",
            "hg://example.invalid/repository",
        ] {
            let err = classify_source(url).expect_err("helper-capable transport must be rejected");
            assert!(
                err.contains("can start an external helper"),
                "unexpected rejection for {url}: {err}"
            );
        }
    }

    #[test]
    fn only_curl_http_and_direct_local_sources_are_accepted() {
        assert_eq!(
            classify_source("http://example.invalid/repository.git").unwrap(),
            InProcessSource::Http
        );
        assert_eq!(
            classify_source("https://example.invalid/repository.git").unwrap(),
            InProcessSource::Https
        );
        assert!(matches!(
            classify_source("file:///tmp/repository.git").unwrap(),
            InProcessSource::Local(_)
        ));
        assert!(matches!(
            classify_source("/tmp/repository.git").unwrap(),
            InProcessSource::Local(_)
        ));
    }
}
