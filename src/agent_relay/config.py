from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from .models import AgentInfo
from .util import atomic_write_json, command_version, config_dir


AGENT_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "codex": ("codex", "CODEX_HOME", ".codex"),
    "grok": ("grok", "GROK_HOME", ".grok"),
    "claude": ("claude", "CLAUDE_CONFIG_DIR", ".claude"),
    "agy": ("agy", "AGY_HOME", ".gemini/antigravity-cli"),
}
SUPPORTED_AGENTS = tuple(AGENT_DEFAULTS)


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_agents(use_saved: bool = True) -> dict[str, AgentInfo]:
    saved = load_config().get("agents", {}) if use_saved else {}
    result: dict[str, AgentInfo] = {}
    for name, (executable, environment_name, default_home) in AGENT_DEFAULTS.items():
        entry = saved.get(name, {}) if isinstance(saved, dict) else {}
        command = shutil.which(executable)
        saved_command = entry.get("command") if isinstance(entry, dict) else None
        if command is None and saved_command and Path(saved_command).is_file():
            command = saved_command

        environment_home = os.environ.get(environment_name)
        saved_home = entry.get("history_home") if isinstance(entry, dict) else None
        home = Path(
            environment_home or saved_home or (Path.home() / default_home)
        ).expanduser()
        version = command_version(command)
        result[name] = AgentInfo(
            name=name,
            command=command,
            version=version,
            history_root=home,
        )
    return result


def save_discovery(agents: dict[str, AgentInfo]) -> Path:
    path = config_path()
    data = {
        "schema_version": 1,
        "agents": {
            name: {
                "command": info.command,
                "version": info.version,
                "history_home": str(info.history_root),
            }
            for name, info in agents.items()
        },
    }
    atomic_write_json(path, data)
    return path
