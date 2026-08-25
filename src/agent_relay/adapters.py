from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import unquote

from .index import IndexCache
from .models import AgentInfo, NormalizedEvent, NormalizedTranscript, Session
from .util import (
    clean_title,
    content_text,
    parse_time,
    redact,
    tail_json_lines,
    truncate,
    valid_json_lines,
    visible_user_text,
)


MAX_RECENT_EVENTS = 120
MAX_FIRST_REQUESTS = 3
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _head_objects(
    path: Path,
    max_lines: int = 120,
    max_bytes: int = 4_000_000,
) -> Iterable[dict[str, Any]]:
    consumed = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream):
                consumed += len(line)
                if number >= max_lines or consumed > max_bytes:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def _message_text(payload: dict[str, Any]) -> str:
    return content_text(payload.get("content"))


def _append_event(
    events: deque[NormalizedEvent],
    kind: str,
    text: Any,
    role: str = "",
    timestamp: str = "",
    limit: int = 4_000,
) -> None:
    cleaned = redact(text, limit=limit).strip()
    if cleaned:
        events.append(
            NormalizedEvent(
                kind=kind,
                text=cleaned,
                role=role,
                timestamp=timestamp,
            )
        )


class AgentAdapter:
    name = "agent"

    def __init__(self, info: AgentInfo) -> None:
        self.info = info

    @property
    def command(self) -> str | None:
        return self.info.command

    @property
    def history_home(self) -> Path:
        return self.info.history_root

    def metadata_files(self) -> Iterable[Path]:
        raise NotImplementedError

    def parse_session(self, path: Path) -> Session | None:
        raise NotImplementedError

    def normalize(self, session: Session) -> NormalizedTranscript:
        raise NotImplementedError

    def native_resume_command(self, session: Session) -> list[str]:
        raise NotImplementedError

    def cross_launch_command(self, cwd: Path, prompt: str) -> list[str]:
        raise NotImplementedError

    def scan(self, cache: IndexCache) -> list[Session]:
        sessions: list[Session] = []
        for path in self.metadata_files():
            session = cache.session_for(path, self.name, self.parse_session)
            if session is not None:
                sessions.append(session)
        return sessions


class CodexAdapter(AgentAdapter):
    name = "codex"

    def __init__(self, info: AgentInfo) -> None:
        super().__init__(info)
        self.titles: dict[str, tuple[str, float]] = {}

    def _load_titles(self) -> None:
        self.titles = {}
        index_path = self.history_home / "session_index.jsonl"
        for item in valid_json_lines(index_path):
            session_id = str(item.get("id", ""))
            if not session_id:
                continue
            self.titles[session_id] = (
                clean_title(item.get("thread_name", "")),
                parse_time(item.get("updated_at")),
            )

    def metadata_files(self) -> Iterable[Path]:
        self._load_titles()
        root = self.history_home / "sessions"
        if not root.is_dir():
            return []
        return root.glob("*/*/*/rollout-*.jsonl")

    def scan(self, cache: IndexCache) -> list[Session]:
        sessions = super().scan(cache)
        refreshed: list[Session] = []
        for session in sessions:
            title, updated = self.titles.get(session.session_id, ("", 0.0))
            refreshed.append(
                replace(
                    session,
                    title=title or session.title,
                    updated_at=updated or session.updated_at,
                )
            )
        return refreshed

    def parse_session(self, path: Path) -> Session | None:
        metadata: dict[str, Any] = {}
        first_user = ""
        model = ""
        for item in _head_objects(path):
            record_type = item.get("type")
            payload = item.get("payload")
            if (
                record_type == "session_meta"
                and isinstance(payload, dict)
                and not metadata
            ):
                metadata = payload
            elif record_type == "turn_context" and isinstance(payload, dict):
                model = str(payload.get("model", model))
            elif record_type == "response_item" and isinstance(payload, dict):
                if payload.get("type") == "message" and payload.get("role") == "user":
                    candidate = _message_text(payload).strip()
                    if candidate and visible_user_text(candidate):
                        first_user = candidate
            if metadata and first_user:
                break

        source = metadata.get("source")
        if metadata.get("thread_source") == "subagent" or (
            isinstance(source, dict) and "subagent" in source
        ):
            return None

        try:
            stat = path.stat()
        except OSError:
            return None
        match = UUID_RE.search(path.name)
        session_id = str(
            metadata.get("id")
            or (match.group(0) if match else None)
            or metadata.get("session_id")
            or path.stem
        )
        cwd_value = metadata.get("cwd")
        if not isinstance(cwd_value, str) or not cwd_value:
            return None
        indexed_title, indexed_updated = self.titles.get(session_id, ("", 0.0))
        title = indexed_title or clean_title(first_user)
        created = parse_time(metadata.get("timestamp"), stat.st_mtime)
        updated = indexed_updated or stat.st_mtime

        status = "saved"
        for item in tail_json_lines(path, max_bytes=256_000):
            if item.get("type") != "event_msg":
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "turn_aborted":
                status = "interrupted"
            elif payload.get("type") == "task_complete":
                status = "completed-turn"

        return Session(
            agent=self.name,
            session_id=session_id,
            title=title,
            cwd=Path(cwd_value).expanduser(),
            source_path=path,
            created_at=created,
            updated_at=updated,
            status=status,
            model=model,
        )

    def normalize(self, session: Session) -> NormalizedTranscript:
        events: deque[NormalizedEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        first_requests: list[str] = []
        summary = ""
        last_error = ""
        for item in valid_json_lines(session.source_path):
            record_type = item.get("type")
            payload = item.get("payload")
            timestamp = str(item.get("timestamp", ""))
            if record_type == "response_item" and isinstance(payload, dict):
                payload_type = payload.get("type")
                if payload_type == "message":
                    role = str(payload.get("role", ""))
                    if role not in {"user", "assistant"}:
                        continue
                    text = _message_text(payload).strip()
                    if role == "user" and not visible_user_text(text):
                        continue
                    if role == "user" and text and len(first_requests) < MAX_FIRST_REQUESTS:
                        first_requests.append(redact(text, limit=10_000))
                    _append_event(events, "message", text, role, timestamp, limit=8_000)
                elif payload_type in {"function_call", "custom_tool_call"}:
                    name = payload.get("name", "tool")
                    arguments = payload.get("arguments", payload.get("input", ""))
                    _append_event(
                        events,
                        "tool-call",
                        f"{name}: {truncate(arguments, 3_000)}",
                        timestamp=timestamp,
                    )
                elif payload_type in {"function_call_output", "custom_tool_call_output"}:
                    _append_event(
                        events,
                        "tool-result",
                        payload.get("output", ""),
                        timestamp=timestamp,
                    )
            elif record_type == "compacted" and isinstance(payload, dict):
                candidate = content_text(payload.get("message")) or content_text(
                    payload.get("replacement_history")
                )
                if candidate:
                    summary = redact(candidate, limit=20_000)
            elif record_type == "event_msg" and isinstance(payload, dict):
                if payload.get("type") == "turn_aborted":
                    reason = payload.get("reason", "turn aborted")
                    last_error = redact(reason, limit=4_000)
                    _append_event(events, "error", reason, timestamp=timestamp)

        return NormalizedTranscript(
            first_requests=first_requests,
            summary=summary,
            events=list(events),
            last_error=last_error,
        )

    def native_resume_command(self, session: Session) -> list[str]:
        assert self.command is not None
        return [self.command, "-C", str(session.cwd), "resume", session.session_id]

    def cross_launch_command(self, cwd: Path, prompt: str) -> list[str]:
        assert self.command is not None
        return [self.command, "-C", str(cwd), prompt]


class GrokAdapter(AgentAdapter):
    name = "grok"

    def metadata_files(self) -> Iterable[Path]:
        root = self.history_home / "sessions"
        if not root.is_dir():
            return []
        return root.glob("*/*/summary.json")

    def parse_session(self, path: Path) -> Session | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stat = path.stat()
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        info = data.get("info") if isinstance(data.get("info"), dict) else {}
        cwd_value = info.get("cwd")
        if not cwd_value:
            cwd_value = unquote(path.parent.parent.name)
        if not isinstance(cwd_value, str) or not cwd_value:
            return None
        session_id = str(info.get("id") or path.parent.name)
        title = clean_title(
            data.get("generated_title")
            or data.get("last_turn_summary")
            or data.get("session_summary")
        )
        return Session(
            agent=self.name,
            session_id=session_id,
            title=title,
            cwd=Path(cwd_value).expanduser(),
            source_path=path,
            created_at=parse_time(data.get("created_at"), stat.st_mtime),
            updated_at=parse_time(data.get("updated_at"), stat.st_mtime),
            status="saved",
            branch=str(data.get("head_branch", "") or ""),
            model=str(data.get("current_model_id", "") or ""),
        )

    def normalize(self, session: Session) -> NormalizedTranscript:
        try:
            summary_data = json.loads(session.source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary_data = {}
        summary = ""
        if isinstance(summary_data, dict):
            summary = redact(
                summary_data.get("session_summary")
                or summary_data.get("last_turn_summary")
                or "",
                limit=20_000,
            )

        updates_path = session.source_path.with_name("updates.jsonl")
        events: deque[NormalizedEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        first_requests: list[str] = []
        last_error = ""
        chunk_kind = ""
        chunk_text: list[str] = []
        chunk_timestamp = ""

        def flush_chunks() -> None:
            nonlocal chunk_kind, chunk_text, chunk_timestamp
            if not chunk_kind:
                return
            text = "".join(chunk_text).strip()
            role = "user" if chunk_kind == "user_message_chunk" else "assistant"
            if role == "user" and text and len(first_requests) < MAX_FIRST_REQUESTS:
                first_requests.append(redact(text, limit=10_000))
            _append_event(events, "message", text, role, chunk_timestamp, limit=8_000)
            chunk_kind = ""
            chunk_text = []
            chunk_timestamp = ""

        for item in valid_json_lines(updates_path):
            params = item.get("params")
            if not isinstance(params, dict):
                continue
            update = params.get("update")
            if not isinstance(update, dict):
                continue
            kind = str(update.get("sessionUpdate", update.get("type", "")))
            timestamp = str(item.get("timestamp", ""))
            if kind in {"user_message_chunk", "agent_message_chunk"}:
                if chunk_kind and chunk_kind != kind:
                    flush_chunks()
                chunk_kind = kind
                chunk_timestamp = chunk_timestamp or timestamp
                content = update.get("content")
                if isinstance(content, dict):
                    chunk_text.append(str(content.get("text", "")))
                continue

            flush_chunks()
            if kind == "agent_thought_chunk":
                continue
            if kind == "tool_call":
                title = update.get("title") or update.get("kind") or "tool"
                raw_input = update.get("rawInput", "")
                _append_event(
                    events,
                    "tool-call",
                    f"{title}: {truncate(raw_input, 3_000)}",
                    timestamp=timestamp,
                )
            elif kind == "tool_call_update":
                status = str(update.get("status", ""))
                raw_output = update.get("rawOutput")
                if raw_output is not None or status.lower() in {
                    "completed",
                    "failed",
                    "error",
                }:
                    title = update.get("title") or "tool"
                    body = raw_output if raw_output is not None else update.get("content", "")
                    _append_event(
                        events,
                        "tool-result",
                        f"{title} [{status or 'update'}]: {truncate(body, 4_000)}",
                        timestamp=timestamp,
                    )
            elif kind == "session_recap":
                candidate = update.get("summary")
                if candidate:
                    summary = redact(candidate, limit=20_000)
            elif kind == "retry_state" and update.get("message"):
                last_error = redact(update.get("message"), limit=4_000)
                _append_event(events, "error", last_error, timestamp=timestamp)
        flush_chunks()

        return NormalizedTranscript(
            first_requests=first_requests,
            summary=summary,
            events=list(events),
            last_error=last_error,
        )

    def native_resume_command(self, session: Session) -> list[str]:
        assert self.command is not None
        return [self.command, "--cwd", str(session.cwd), "--resume", session.session_id]

    def cross_launch_command(self, cwd: Path, prompt: str) -> list[str]:
        assert self.command is not None
        return [self.command, "--cwd", str(cwd), prompt]


class ClaudeAdapter(AgentAdapter):
    name = "claude"

    def metadata_files(self) -> Iterable[Path]:
        root = self.history_home / "projects"
        if not root.is_dir():
            return []
        return root.glob("*/*.jsonl")

    def parse_session(self, path: Path) -> Session | None:
        session_id = path.stem
        cwd_value = ""
        branch = ""
        first_user = ""
        created = 0.0
        model = ""
        title = ""
        for item in _head_objects(path, max_lines=200, max_bytes=4_000_000):
            session_id = str(item.get("sessionId") or item.get("session_id") or session_id)
            cwd_value = str(item.get("cwd") or cwd_value)
            branch = str(item.get("gitBranch") or branch)
            created = created or parse_time(item.get("timestamp"))
            message = item.get("message")
            if isinstance(message, dict):
                model = str(message.get("model") or model)
                if item.get("type") == "user" and not item.get("isMeta"):
                    candidate = content_text(message.get("content")).strip()
                    if candidate and visible_user_text(candidate):
                        first_user = candidate
            if cwd_value and first_user:
                break

        for item in tail_json_lines(path):
            if item.get("type") == "ai-title" and item.get("aiTitle"):
                title = str(item["aiTitle"])
        if not cwd_value:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        return Session(
            agent=self.name,
            session_id=session_id,
            title=clean_title(title or first_user),
            cwd=Path(cwd_value).expanduser(),
            source_path=path,
            created_at=created or stat.st_mtime,
            updated_at=stat.st_mtime,
            status="saved",
            branch=branch,
            model=model,
        )

    def normalize(self, session: Session) -> NormalizedTranscript:
        nodes: dict[str, tuple[str, list[NormalizedEvent], str]] = {}
        order: list[str] = []
        leaf = ""
        last_error = ""

        for item in valid_json_lines(session.source_path):
            record_type = item.get("type")
            if record_type == "last-prompt":
                leaf = str(item.get("leafUuid") or leaf)
                continue
            if record_type not in {"user", "assistant"} or item.get("isSidechain"):
                continue
            message = item.get("message")
            if not isinstance(message, dict):
                continue
            node_id = str(item.get("uuid") or "")
            if not node_id:
                continue
            parent = str(item.get("parentUuid") or "")
            timestamp = str(item.get("timestamp", ""))
            node_events: list[NormalizedEvent] = []
            content = message.get("content")
            if isinstance(content, str):
                if record_type == "user" and visible_user_text(content):
                    node_events.append(
                        NormalizedEvent("message", redact(content, 8_000), "user", timestamp)
                    )
                elif record_type == "assistant":
                    node_events.append(
                        NormalizedEvent(
                            "message", redact(content, 8_000), "assistant", timestamp
                        )
                    )
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind == "thinking":
                        continue
                    if kind == "text":
                        text = str(block.get("text", ""))
                        role = "user" if record_type == "user" else "assistant"
                        if role == "user" and not visible_user_text(text):
                            continue
                        cleaned = redact(text, limit=8_000).strip()
                        if cleaned:
                            node_events.append(
                                NormalizedEvent("message", cleaned, role, timestamp)
                            )
                    elif kind == "tool_use":
                        name = block.get("name", "tool")
                        value = f"{name}: {truncate(block.get('input', ''), 3_000)}"
                        node_events.append(
                            NormalizedEvent(
                                "tool-call", redact(value), timestamp=timestamp
                            )
                        )
                    elif kind == "tool_result":
                        value = content_text(block.get("content")) or block.get("content", "")
                        cleaned = redact(value, limit=4_000).strip()
                        if cleaned:
                            node_events.append(
                                NormalizedEvent(
                                    "tool-result", cleaned, timestamp=timestamp
                                )
                            )
                            if block.get("is_error"):
                                last_error = cleaned
            nodes[node_id] = (parent, node_events, timestamp)
            order.append(node_id)

        lineage: list[str] = []
        cursor = leaf if leaf in nodes else (order[-1] if order else "")
        visited: set[str] = set()
        while cursor and cursor in nodes and cursor not in visited:
            visited.add(cursor)
            lineage.append(cursor)
            cursor = nodes[cursor][0]
        lineage.reverse()
        selected = lineage if lineage else order

        events: deque[NormalizedEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        first_requests: list[str] = []
        for node_id in selected:
            for event in nodes[node_id][1]:
                if (
                    event.role == "user"
                    and len(first_requests) < MAX_FIRST_REQUESTS
                    and event.text
                ):
                    first_requests.append(event.text)
                events.append(event)

        return NormalizedTranscript(
            first_requests=first_requests,
            events=list(events),
            last_error=last_error,
        )

    def native_resume_command(self, session: Session) -> list[str]:
        assert self.command is not None
        return [self.command, "--resume", session.session_id]

    def cross_launch_command(self, cwd: Path, prompt: str) -> list[str]:
        assert self.command is not None
        return [self.command, prompt]


def build_adapters(infos: dict[str, AgentInfo]) -> dict[str, AgentAdapter]:
    classes = {
        "codex": CodexAdapter,
        "grok": GrokAdapter,
        "claude": ClaudeAdapter,
    }
    return {name: classes[name](info) for name, info in infos.items()}
