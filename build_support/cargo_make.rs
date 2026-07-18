//! Cargo-jobserver-aware child commands.
//!
//! `jobserver::Client::configure_make()` installs a `pre_exec` closure that
//! refers to the client's duplicated descriptors by number. The client must
//! therefore outlive `Command::spawn`/`status`; returning a bare configured
//! `Command` closes those descriptors before the closure runs and makes every
//! native build fail with `EBADF`.

use std::ops::{Deref, DerefMut};
use std::process::Command;

pub(crate) struct CargoCoordinatedMake {
    command: Command,
    _jobserver: Option<jobserver::Client>,
}

impl CargoCoordinatedMake {
    pub(crate) fn new() -> Self {
        // SAFETY: Cargo owns and exports the authenticated jobserver. The
        // returned client is retained by this wrapper through child spawn.
        let client = unsafe { jobserver::Client::from_env() };
        Self::with_client(Command::new("make"), client)
    }

    pub(crate) fn with_client(mut command: Command, client: Option<jobserver::Client>) -> Self {
        if let Some(client) = &client {
            client.configure_make(&mut command);
        } else {
            // Do not hand a child stale descriptor numbers when a caller left
            // make flags in the environment but no live jobserver exists.
            command.env_remove("MAKEFLAGS");
            command.env_remove("MFLAGS");
        }
        Self {
            command,
            _jobserver: client,
        }
    }
}

impl Deref for CargoCoordinatedMake {
    type Target = Command;

    fn deref(&self) -> &Self::Target {
        &self.command
    }
}

impl DerefMut for CargoCoordinatedMake {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.command
    }
}
