from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlparse

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
AGY_USER_REQUEST_RE = re.compile(
    r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>",
    re.IGNORECASE | re.DOTALL,
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


def _agy_user_text(content: Any) -> str:
    """Remove AGY's metadata envelope from an explicit user request."""

    text = content_text(content).strip()
    match = AGY_USER_REQUEST_RE.search(text)
    return match.group(1).strip() if match else text


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


class AgyAdapter(AgentAdapter):
    """Adapter for Google Antigravity CLI (``agy``) conversations."""

    name = "agy"

    def __init__(self, info: AgentInfo) -> None:
        super().__init__(info)
        self.summaries: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.direct_history: dict[str, list[dict[str, Any]]] = {}
        self.workspace_by_session: dict[str, Path] = {}
        self.transcript_metadata: dict[str, dict[str, Any]] = {}

    def _transcript_path(self, session_id: str) -> Path:
        return (
            self.history_home
            / "brain"
            / session_id
            / ".system_generated"
            / "logs"
            / "transcript.jsonl"
        )

    @staticmethod
    def _session_id_from_path(path: Path) -> str:
        if path.name == "transcript.jsonl" and len(path.parents) >= 3:
            return path.parents[2].name
        return path.stem

    @staticmethod
    def _workspace_path(value: Any) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if text.startswith("file:"):
            parsed = urlparse(text)
            decoded = unquote(parsed.path)
            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                decoded = f"//{parsed.netloc}{decoded}"
            text = decoded
        if not text:
            return None
        return Path(text).expanduser()

    @classmethod
    def _summary_workspaces(cls, value: Any) -> list[Path]:
        decoded = value
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
        if isinstance(decoded, dict):
            candidates = list(decoded.values())
        elif isinstance(decoded, list):
            candidates = decoded
        else:
            candidates = [decoded]
        paths: list[Path] = []
        for candidate in candidates:
            path = cls._workspace_path(candidate)
            if path is not None:
                paths.append(path)
        return paths

    def _source_files(self) -> list[Path]:
        sources: dict[str, Path] = {}
        conversations = self.history_home / "conversations"
        if conversations.is_dir():
            for path in conversations.glob("*.db"):
                sources[path.stem] = path
        brain = self.history_home / "brain"
        if brain.is_dir():
            for path in brain.glob(
                "*/.system_generated/logs/transcript.jsonl"
            ):
                sources[self._session_id_from_path(path)] = path
        return sorted(sources.values(), key=str)

    def _load_summaries(self) -> None:
        database = self.history_home / "conversation_summaries.db"
        if not database.is_file():
            return
        uri = f"file:{quote(str(database.resolve()), safe='/')}?mode=ro"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=1)
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM conversation_summaries")
            for row in rows:
                item = dict(row)
                session_id = str(
                    item.get("conversation_id")
                    or item.get("conversationId")
                    or ""
                )
                if not session_id:
                    continue
                current = self.summaries.get(session_id)
                current_time = parse_time(
                    current.get("last_modified_time") if current else None
                )
                item_time = parse_time(
                    item.get("last_modified_time")
                    or item.get("lastModifiedTime")
                )
                if current is None or item_time >= current_time:
                    self.summaries[session_id] = item
        except sqlite3.Error:
            return
        finally:
            if connection is not None:
                connection.close()

    def _transcript_details(self, session_id: str) -> dict[str, Any]:
        first_user = ""
        created = 0.0
        updated = 0.0
        has_user = False
        has_response = False
        for item in valid_json_lines(self._transcript_path(session_id)):
            source = str(item.get("source", "")).upper()
            kind = str(item.get("type", "")).upper()
            status = str(item.get("status", "")).upper()
            timestamp = parse_time(item.get("created_at") or item.get("timestamp"))
            if timestamp:
                created = created or timestamp
                updated = max(updated, timestamp)
            if source.startswith("USER") and kind == "USER_INPUT":
                text = _agy_user_text(item.get("content"))
                if text and visible_user_text(text):
                    has_user = True
                    first_user = first_user or text
            elif source == "MODEL" and kind in {
                "PLANNER_RESPONSE",
                "MODEL_RESPONSE",
                "RESPONSE",
            }:
                if status in {"", "DONE", "COMPLETED", "SUCCESS"}:
                    has_response = True
        return {
            "first_user": first_user,
            "created_at": created,
            "updated_at": updated,
            "has_user": has_user,
            "has_response": has_response,
        }

    def _load_auxiliary(self) -> None:
        self.summaries = {}
        self.history = list(valid_json_lines(self.history_home / "history.jsonl"))
        self.direct_history = {}
        self.workspace_by_session = {}
        self.transcript_metadata = {}
        self._load_summaries()

        for item in self.history:
            session_id = str(
                item.get("conversationId") or item.get("conversation_id") or ""
            )
            if not session_id:
                continue
            self.direct_history.setdefault(session_id, []).append(item)
            workspace = self._workspace_path(item.get("workspace"))
            if workspace is not None:
                self.workspace_by_session[session_id] = workspace

        cache_path = self.history_home / "cache" / "last_conversations.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if isinstance(cached, dict):
            for workspace_value, session_value in cached.items():
                session_id = str(session_value) if session_value is not None else ""
                workspace = self._workspace_path(workspace_value)
                if session_id and workspace is not None:
                    self.workspace_by_session.setdefault(session_id, workspace)

        sources = self._source_files()
        for source in sources:
            session_id = self._session_id_from_path(source)
            details = self._transcript_details(session_id)
            self.transcript_metadata[session_id] = details

        # Older AGY history rows do not always carry a conversation ID. Match
        # those rows to the first explicit user request, using time only to
        # disambiguate duplicate prompts. Each history row is assigned once.
        candidates: list[tuple[float, str, int, Path]] = []
        for session_id, details in self.transcript_metadata.items():
            if session_id in self.workspace_by_session:
                continue
            first_user = str(details.get("first_user", "")).strip()
            if not first_user:
                continue
            created = float(details.get("created_at", 0.0) or 0.0)
            for index, item in enumerate(self.history):
                if item.get("conversationId") or item.get("conversation_id"):
                    continue
                display = str(item.get("display", "")).strip()
                workspace = self._workspace_path(item.get("workspace"))
                if display != first_user or workspace is None:
                    continue
                timestamp = parse_time(item.get("timestamp"))
                distance = abs(timestamp - created) if timestamp and created else 0.0
                candidates.append((distance, session_id, index, workspace))

        assigned_sessions: set[str] = set()
        assigned_rows: set[int] = set()
        for _, session_id, index, workspace in sorted(candidates):
            if session_id in assigned_sessions or index in assigned_rows:
                continue
            self.workspace_by_session[session_id] = workspace
            assigned_sessions.add(session_id)
            assigned_rows.add(index)

    def metadata_files(self) -> Iterable[Path]:
        self._load_auxiliary()
        return self._source_files()

    def _summary_workspace(self, session_id: str) -> Path | None:
        summary = self.summaries.get(session_id, {})
        value = summary.get("workspace_uris") or summary.get("workspaceUris")
        workspaces = self._summary_workspaces(value)
        return workspaces[0] if workspaces else None

    def _session_workspace(self, session_id: str) -> Path | None:
        direct = self.direct_history.get(session_id, [])
        for item in reversed(direct):
            workspace = self._workspace_path(item.get("workspace"))
            if workspace is not None:
                return workspace
        return self._summary_workspace(session_id) or self.workspace_by_session.get(
            session_id
        )

    def parse_session(self, path: Path) -> Session | None:
        session_id = self._session_id_from_path(path)
        workspace = self._session_workspace(session_id)
        if workspace is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None

        details = self.transcript_metadata.get(session_id)
        if details is None:
            details = self._transcript_details(session_id)
        summary = self.summaries.get(session_id, {})
        direct = self.direct_history.get(session_id, [])
        first_history = str(direct[0].get("display", "")) if direct else ""
        title = clean_title(
            summary.get("title")
            or summary.get("preview")
            or details.get("first_user")
            or first_history
        )

        history_times = [parse_time(item.get("timestamp")) for item in direct]
        summary_updated = parse_time(
            summary.get("last_modified_time") or summary.get("lastModifiedTime")
        )
        transcript_created = float(details.get("created_at", 0.0) or 0.0)
        transcript_updated = float(details.get("updated_at", 0.0) or 0.0)
        created = transcript_created or next(
            (timestamp for timestamp in history_times if timestamp), stat.st_mtime
        )
        updated = max(
            [stat.st_mtime, transcript_updated, summary_updated]
            + [timestamp for timestamp in history_times if timestamp]
        )

        summary_status = str(summary.get("status", "")).strip().lower()
        if details.get("has_response"):
            status = "completed-turn"
        elif details.get("has_user"):
            status = "interrupted"
        else:
            status = summary_status.replace("_", "-") or "saved"

        return Session(
            agent=self.name,
            session_id=session_id,
            title=title,
            cwd=workspace,
            source_path=path,
            created_at=created,
            updated_at=updated,
            status=status,
            model=str(summary.get("agent_name", "") or ""),
        )

    def scan(self, cache: IndexCache) -> list[Session]:
        sessions = super().scan(cache)
        refreshed: list[Session] = []
        for session in sessions:
            workspace = self._session_workspace(session.session_id)
            summary = self.summaries.get(session.session_id, {})
            title_value = summary.get("title") or summary.get("preview")
            updated = parse_time(
                summary.get("last_modified_time")
                or summary.get("lastModifiedTime")
            )
            refreshed.append(
                replace(
                    session,
                    cwd=workspace or session.cwd,
                    title=clean_title(title_value) if title_value else session.title,
                    updated_at=max(session.updated_at, updated),
                )
            )
        return refreshed

    def normalize(self, session: Session) -> NormalizedTranscript:
        events: deque[NormalizedEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        first_requests: list[str] = []
        last_error = ""
        for item in valid_json_lines(self._transcript_path(session.session_id)):
            source = str(item.get("source", "")).upper()
            kind = str(item.get("type", "")).upper()
            timestamp = str(item.get("created_at") or item.get("timestamp") or "")
            if source == "SYSTEM" or any(
                marker in kind
                for marker in ("THOUGHT", "REASONING", "CHECKPOINT")
            ):
                continue
            text = (
                _agy_user_text(item.get("content"))
                if source.startswith("USER") and kind == "USER_INPUT"
                else content_text(item.get("content")).strip()
            )
            if source.startswith("USER") and kind == "USER_INPUT":
                if not text or not visible_user_text(text):
                    continue
                if len(first_requests) < MAX_FIRST_REQUESTS:
                    first_requests.append(redact(text, limit=10_000))
                _append_event(
                    events,
                    "message",
                    text,
                    role="user",
                    timestamp=timestamp,
                    limit=8_000,
                )
            elif source == "MODEL" and kind in {
                "PLANNER_RESPONSE",
                "MODEL_RESPONSE",
                "RESPONSE",
            }:
                _append_event(
                    events,
                    "message",
                    text,
                    role="assistant",
                    timestamp=timestamp,
                    limit=8_000,
                )
            elif source != "SYSTEM" and ("ERROR" in kind or "FAILED" in kind):
                last_error = redact(text, limit=4_000)
                _append_event(events, "error", text, timestamp=timestamp)

        return NormalizedTranscript(
            first_requests=first_requests,
            events=list(events),
            last_error=last_error,
        )

    def native_resume_command(self, session: Session) -> list[str]:
        assert self.command is not None
        return [self.command, "--conversation", session.session_id]

    def cross_launch_command(self, cwd: Path, prompt: str) -> list[str]:
        assert self.command is not None
        return [self.command, "--prompt-interactive", prompt]


def build_adapters(infos: dict[str, AgentInfo]) -> dict[str, AgentAdapter]:
    classes = {
        "codex": CodexAdapter,
        "grok": GrokAdapter,
        "claude": ClaudeAdapter,
        "agy": AgyAdapter,
    }
    result: dict[str, AgentAdapter] = {}
    for name, info in infos.items():
        adapter_class = classes.get(info.history_adapter)
        if adapter_class is None:
            continue
        adapter = adapter_class(info)
        # Custom agents can point at the same history format under a distinct
        # target name while retaining the parser/launcher behavior above.
        adapter.name = name
        result[name] = adapter
    return result
