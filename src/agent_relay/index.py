from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .models import Session
from .util import atomic_write_json, state_dir


Parser = Callable[[Path], Session | None]
INDEX_SCHEMA_VERSION = 3


class IndexCache:
    """Small metadata cache keyed by the original vendor history file."""

    def __init__(self, refresh: bool = False) -> None:
        self.path = state_dir() / "index.json"
        self.refresh = refresh
        self.entries: dict[str, dict[str, Any]] = {}
        self.seen: set[str] = set()
        self.dirty = False
        if not refresh:
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == INDEX_SCHEMA_VERSION
                    and isinstance(value.get("entries"), dict)
                ):
                    self.entries = value["entries"]
            except (OSError, json.JSONDecodeError):
                pass

    def session_for(self, path: Path, agent: str, parser: Parser) -> Session | None:
        key = str(path)
        self.seen.add(key)
        try:
            stat = path.stat()
        except OSError:
            return None
        cached = self.entries.get(key)
        if (
            not self.refresh
            and isinstance(cached, dict)
            and cached.get("agent") == agent
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and cached.get("size") == stat.st_size
            and isinstance(cached.get("session"), dict)
        ):
            try:
                return Session.from_cache_dict(cached["session"])
            except (KeyError, TypeError, ValueError):
                pass

        session = parser(path)
        if session is None:
            if key in self.entries:
                self.entries.pop(key, None)
                self.dirty = True
            return None
        self.entries[key] = {
            "agent": agent,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "session": session.to_cache_dict(),
        }
        self.dirty = True
        return session

    def save(self) -> None:
        missing = [key for key in self.entries if not Path(key).exists()]
        for key in missing:
            self.entries.pop(key, None)
            self.dirty = True
        if self.dirty or not self.path.exists():
            atomic_write_json(
                self.path,
                {"schema_version": INDEX_SCHEMA_VERSION, "entries": self.entries},
            )
