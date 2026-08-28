# Agent-assisted setup

Manual installation is usually enough. Agent assistance is useful when several
supported CLIs are installed, their home directories are customized, or skills
exist in more than one agent-specific location.

Give a coding agent with terminal access this bounded request:

```text
Install Agent Session Relay from
https://github.com/tomzhu0225/agent-session-relay.

Do not edit or delete any existing agent history or skill file. Install the
Python package, run `relay setup`, `relay doctor`, and
`relay skills sync --dry-run`. Report the detected CLI versions, history
locations, session counts, proposed skill symlinks, and any same-name skill
conflicts. Do not apply `relay skills sync` until I approve the dry-run.
Finally, demonstrate one `relay resume ... --dry-run --no-git-diff`; do not
launch a target agent or transmit recovered context during setup.
```

After reviewing the report, the user can approve:

```bash
relay skills sync
```

## What the setup agent should verify

- The expected `codex`, `grok`, `claude`, and/or `agy` executables are the ones found
  on `PATH`.
- `CODEX_HOME`, `GROK_HOME`, `CLAUDE_CONFIG_DIR`, and Relay's `AGY_HOME`
  discovery override are set when vendor data is stored outside the default
  home directories.
- Codex-compatible wrappers, such as provider-specific profiles, are configured with
  `relay agents add <name> --adapter codex --command <executable> --model-provider <provider>`
  rather than editing Relay's built-in agent list. The provider must match the
  profile's `model_provider` value (for example, `ZAI`).
- `relay sessions` finds the expected working directory.
- Same-name, different-content skills are left unresolved for the user.
- The recovery directory and files are private to the local user.

The setup agent is optional. Relay itself performs common-path discovery and
does not require an AI service to remain running.
