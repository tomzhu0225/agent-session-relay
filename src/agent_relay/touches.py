from __future__ import annotations

import json
from pathlib import Path

from .models import Session
from .util import atomic_write_json, state_dir


TOUCHES_SCHEMA_VERSION = 1


def touches_path() -> Path:
    return state_dir() / "session-targets.json"


def load_session_targets() -> dict[str, str]:
    try:
        value = json.loads(touches_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != TOUCHES_SCHEMA_VERSION
        or not isinstance(value.get("targets"), dict)
    ):
        return {}
    return {
        str(path): target
        for path, target in value["targets"].items()
        if isinstance(path, str) and isinstance(target, str) and target
    }


def record_session_target(session: Session, target: str) -> Path:
    targets = load_session_targets()
    targets[str(session.source_path)] = target
    path = touches_path()
    atomic_write_json(
        path,
        {"schema_version": TOUCHES_SCHEMA_VERSION, "targets": targets},
    )
    return path
