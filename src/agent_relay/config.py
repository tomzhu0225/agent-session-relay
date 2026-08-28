from __future__ import annotations

import json
import os
from pathlib import Path
import re
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
BUILTIN_MODEL_PROVIDERS = {"codex": "openai"}
BUILTIN_AGENTS = tuple(AGENT_DEFAULTS)
SUPPORTED_AGENTS = BUILTIN_AGENTS
CUSTOM_AGENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_command(specified: object) -> str | None:
    if not isinstance(specified, str) or not specified.strip():
        return None
    value = specified.strip()
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value)


def _custom_definitions() -> dict[str, dict[str, Any]]:
    saved = load_config().get("custom_agents", {})
    if not isinstance(saved, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, definition in saved.items():
        if not isinstance(name, str) or not CUSTOM_AGENT_NAME_RE.fullmatch(name):
            continue
        if name in AGENT_DEFAULTS or not isinstance(definition, dict):
            continue
        adapter = definition.get("adapter", definition.get("kind"))
        if adapter not in AGENT_DEFAULTS:
            continue
        result[name] = definition
    return result


def configured_agent_names() -> tuple[str, ...]:
    """Return built-in agents followed by valid custom target names."""

    return (*BUILTIN_AGENTS, *_custom_definitions())


def add_custom_agent(
    name: str,
    adapter: str,
    command: str,
    history_home: str | None = None,
    scan_history: bool = False,
    model_provider: str | None = None,
) -> Path:
    if not CUSTOM_AGENT_NAME_RE.fullmatch(name):
        raise ValueError(
            "agent names must start with a letter or digit and contain only "
            "lowercase letters, digits, hyphens, and underscores"
        )
    if name in AGENT_DEFAULTS:
        raise ValueError(f"{name!r} is reserved for a built-in agent")
    if adapter not in AGENT_DEFAULTS:
        raise ValueError(f"unsupported adapter {adapter!r}")
    provider = model_provider.strip() if isinstance(model_provider, str) else ""
    if model_provider is not None and not provider:
        raise ValueError("model provider must not be empty")
    resolved_command = _find_command(command)
    if resolved_command is None:
        raise ValueError(f"command is not an executable file: {command}")

    config = load_config()
    saved = config.get("custom_agents", {})
    if not isinstance(saved, dict):
        saved = {}
    if name in saved:
        raise ValueError(f"custom agent {name!r} already exists")

    definition: dict[str, Any] = {
        "adapter": adapter,
        "command": resolved_command,
    }
    if history_home:
        definition["history_home"] = str(Path(history_home).expanduser())
    if scan_history:
        definition["scan_history"] = True
    if provider:
        definition["model_provider"] = provider

    saved[name] = definition
    config["schema_version"] = 1
    config["custom_agents"] = saved
    path = config_path()
    atomic_write_json(path, config)
    return path


def remove_custom_agent(name: str) -> Path:
    config = load_config()
    saved = config.get("custom_agents", {})
    if not isinstance(saved, dict) or name not in saved:
        raise ValueError(f"custom agent {name!r} does not exist")
    del saved[name]
    config["schema_version"] = 1
    config["custom_agents"] = saved
    path = config_path()
    atomic_write_json(path, config)
    return path


def update_custom_agent_provider(name: str, model_provider: str) -> Path:
    provider = model_provider.strip()
    if not provider:
        raise ValueError("model provider must not be empty")
    config = load_config()
    saved = config.get("custom_agents", {})
    if not isinstance(saved, dict) or name not in saved:
        raise ValueError(f"custom agent {name!r} does not exist")
    definition = saved[name]
    if not isinstance(definition, dict):
        raise ValueError(f"custom agent {name!r} has an invalid definition")
    definition["model_provider"] = provider
    config["schema_version"] = 1
    config["custom_agents"] = saved
    path = config_path()
    atomic_write_json(path, config)
    return path


def discover_agents(use_saved: bool = True) -> dict[str, AgentInfo]:
    loaded = load_config()
    saved = loaded.get("agents", {}) if use_saved else {}
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
            scan_history=True,
            model_provider=BUILTIN_MODEL_PROVIDERS.get(name, ""),
        )

    for name, definition in _custom_definitions().items():
        adapter = str(definition.get("adapter", definition.get("kind")))
        command = _find_command(definition.get("command"))
        home_value = definition.get("history_home")
        home = (
            Path(home_value).expanduser()
            if isinstance(home_value, str) and home_value
            else result[adapter].history_root
        )
        result[name] = AgentInfo(
            name=name,
            command=command,
            version=command_version(command),
            history_root=home,
            adapter_name=adapter,
            custom=True,
            scan_history=definition.get("scan_history") is True,
            model_provider=(
                str(definition["model_provider"]).strip()
                if isinstance(definition.get("model_provider"), str)
                else ""
            ),
        )
    return result


def save_discovery(agents: dict[str, AgentInfo]) -> Path:
    path = config_path()
    existing = load_config()
    data = {
        "schema_version": 1,
        "agents": {
            name: {
                "command": info.command,
                "version": info.version,
                "history_home": str(info.history_root),
            }
                for name, info in agents.items()
                if not info.custom
            },
        "custom_agents": _custom_definitions(),
    }
    # Preserve unrecognized keys so setup remains non-destructive for future
    # schema extensions and user-managed configuration.
    for key, value in existing.items():
        data.setdefault(key, value)
    atomic_write_json(path, data)
    return path
