# Agent Session Relay

`relay` browses local coding-agent histories, shares applicable skills, and resumes a selected session with Codex, Grok, Claude Code, Antigravity CLI (`agy`), or a compatible custom target. It never requires the source agent to prepare a handoff.

The current MVP was built and tested against:

- Codex CLI 0.149.0
- Grok Build 1.0.4
- Claude Code 2.1.220
- Antigravity CLI (`agy`) 1.1.20
- Python 3.12 (Python 3.10 or newer is supported)

It has no third-party Python dependencies.

Agent Session Relay is an independent, unofficial project and is not affiliated
with OpenAI, xAI, Anthropic, or Google. CLI history formats are vendor-controlled and
may change; version compatibility is tested rather than guaranteed.

## Install

With `pipx`:

```bash
pipx install git+https://github.com/tomzhu0225/agent-session-relay.git
```

With `uv`:

```bash
uv tool install git+https://github.com/tomzhu0225/agent-session-relay.git
```

Or from a checkout:

```bash
git clone https://github.com/tomzhu0225/agent-session-relay.git
cd agent-session-relay
python3 -m pip install --user .
```

Relay is currently intended for POSIX environments, including Linux, WSL, and
macOS. Python 3.10 or newer is required.

## Quick start

```bash
relay setup
relay skills sync
relay agents
cd /path/to/project
relay sessions
relay resume
```

The setup is deliberately small enough to do manually. On a machine with
several agents, custom homes, or existing skill collections, using a coding
agent to inspect and configure Relay is convenient. See
[Agent-assisted setup](docs/agent-assisted-setup.md) for a bounded prompt that
does not let the installer modify histories or overwrite skill conflicts.

`relay resume` lists sessions for the current Git worktree, newest first. Select a source session and then select the target agent.

Codex-compatible deployments do not need a hardcoded Relay entry. Add their executable as a custom target and Relay will use the existing Codex history adapter:

```bash
relay agents add codex-glm --adapter codex --command codex-glm
relay resume --session codex:<session-id> --with codex-glm --dry-run
relay agents remove codex-glm
```

By default, a custom target is target-only. If it uses the same history root as its built-in adapter, Relay recognizes that they share history and delegates to the custom executable's native resume command instead of building a redundant recovery bundle. Use `--history-home` when a wrapper stores history elsewhere; use `--scan-history` only for a separate history root that should also appear as a source.

For scripted use:

```bash
relay sessions --json
relay resume --session codex:<session-id> --with grok --dry-run
relay resume --session <unique-id-prefix> --with claude
relay resume --session <selector> --with grok --no-git-diff
```

Use `--exact` to match only the literal working directory rather than every session in the same Git worktree. Use `--all` to search every indexed directory.

## Resume behavior

When the target is the source agent, Relay delegates to the vendor's native resume command:

```text
Codex  -> codex resume <id>
Grok   -> grok --resume <id>
Claude -> claude --resume <id>
AGY    -> agy --conversation <id>
```

When the target does not share the source's adapter and history root, Relay reads the selected source history and creates a normalized recovery bundle under:

```text
~/.local/state/agent-relay/recoveries/
```

Each bundle contains:

```text
manifest.json
brief.md
transcript.md
git-status.txt
git-diff.patch
skills.md
```

The target receives a short initial prompt directing it to read `brief.md`, verify the actual working tree, review `skills.md`, and consult the longer transcript only when necessary.

Custom targets using the same adapter and history root as the source (for example, Codex and a Codex-profile wrapper) also use native resume under the custom command.

## Shared skills

Relay maintains a neutral user-skill registry at:

```text
~/.local/share/agent-relay/skills/
```

Register existing personal skills without moving or copying their contents:

```bash
relay skills sync --dry-run
relay skills sync
```

The registry uses symlinks to the original skill folders and never overwrites a same-name variant. `relay skills list [directory]` shows the effective catalog and flags conflicts.

For each cross-agent handoff, Relay discovers:

```text
Shared    ~/.local/share/agent-relay/skills/
Codex     $CODEX_HOME/skills/
Grok      $GROK_HOME/skills/
Claude    $CLAUDE_CONFIG_DIR/skills/
AGY       $AGY_HOME/skills/ (default: ~/.gemini/antigravity-cli/skills/)
Project   .agents/skills/, .codex/skills/, .grok/skills/, .claude/skills/
```

Only the skill names, descriptions, scopes, and local paths are placed in `skills.md`. The receiving agent opens a `SKILL.md` only when its description matches the recovered task. Skill instructions do not expand permissions.

Relay intentionally ships no skills. `relay skills sync` only registers skills
that already exist in the user's agent homes, using symlinks; it does not copy,
download, publish, or merge their contents. Project-local skills remain in
their projects.

## Commands

```text
relay setup       Detect installed CLIs and their history homes
relay sessions    List sessions for a directory or Git worktree
relay resume      Select a source session and target agent
relay agents      List agents and manage custom targets
relay skills      List and register portable user/project skills
relay doctor      Check commands, history stores, and indexed counts
```

Useful options:

```text
--refresh         Rebuild cached session metadata
--dry-run         Build/print a cross-agent launch without starting it
--with AGENT      Select a built-in or configured custom target non-interactively
--session VALUE   Select by number, UUID, agent:UUID, prefix, or unique title
--json            Machine-readable setup/session output
--no-git-diff     Omit tracked/staged diffs from a cross-agent bundle
```

## Local history adapters

Relay currently recognizes:

```text
Codex   ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
Grok    ~/.grok/sessions/<encoded-cwd>/<session-id>/{summary,updates}.json*
Claude  ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl
AGY     ~/.gemini/antigravity-cli/{conversations/<id>.db,brain/<id>/.system_generated/logs/transcript.jsonl}
```

`CODEX_HOME`, `GROK_HOME`, `CLAUDE_CONFIG_DIR`, Relay's `AGY_HOME` discovery override, and the XDG config/state variables are respected. Discovery is saved to `~/.config/agent-relay/config.json`. The metadata cache stores paths, titles, timestamps, and IDs—not copied transcripts.

Custom target definitions are stored in the same file:

```json
{
  "custom_agents": {
    "codex-glm": {
      "adapter": "codex",
      "command": "/home/example/.local/bin/codex-glm"
    }
  }
}
```

`adapter` selects one of Relay's existing parsers and launch conventions (`codex`, `grok`, `claude`, or `agy`). `command` must be an executable; put provider/profile arguments in a small wrapper script. Optional `history_home` and `scan_history` fields support deployments with a separate store. Prefer `relay agents add` and `relay agents remove` to editing this JSON.

## Privacy and safety

- Vendor history files are read-only.
- Recovery directories use mode `0700`; recovery files use `0600`.
- Private reasoning and source system instructions are excluded from the normalized transcript; common credential forms are heuristically redacted.
- `git-diff.patch` is copied verbatim and can contain secrets. Use `--no-git-diff` for a sensitive working tree.
- Cross-agent transfer intentionally sends the recovered context to the selected target provider once that target starts.
- Stop or exit the original CLI before starting another writable agent in the same worktree. Relay does not yet prove that an arbitrary vendor process is inactive.
- A target with a separate history store is a recovered new session, not a byte-for-byte continuation of the source agent's internal state.

## Development

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Security reports should follow [SECURITY.md](SECURITY.md). Contributions and
new history adapters are welcome.
