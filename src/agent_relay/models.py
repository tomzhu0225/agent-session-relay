from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentInfo:
    name: str
    command: str | None
    version: str | None
    history_root: Path
    adapter_name: str = ""
    custom: bool = False
    scan_history: bool = True
    model_provider: str = ""

    @property
    def history_adapter(self) -> str:
        """Name of the parser/launcher adapter used by this agent."""

        return self.adapter_name or self.name

    @property
    def installed(self) -> bool:
        return self.command is not None


@dataclass(frozen=True)
class Session:
    agent: str
    session_id: str
    title: str
    cwd: Path
    source_path: Path
    created_at: float
    updated_at: float
    status: str = "saved"
    branch: str = ""
    model: str = ""
    model_provider: str = ""
    last_touched_by: str = ""

    @property
    def selector(self) -> str:
        return f"{self.agent}:{self.session_id}"

    def to_cache_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cwd"] = str(self.cwd)
        data["source_path"] = str(self.source_path)
        return data

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "Session":
        values = dict(data)
        values["cwd"] = Path(values["cwd"])
        values["source_path"] = Path(values["source_path"])
        return cls(**values)


@dataclass(frozen=True)
class NormalizedEvent:
    kind: str
    text: str
    role: str = ""
    timestamp: str = ""


@dataclass
class NormalizedTranscript:
    first_requests: list[str] = field(default_factory=list)
    summary: str = ""
    events: list[NormalizedEvent] = field(default_factory=list)
    last_error: str = ""
