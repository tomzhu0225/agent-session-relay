from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any, Iterable, Iterator


SECRET_PATTERNS = (
    re.compile(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;\]\}\"']+)",
    ),
    re.compile(
        r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|cookie)\s*[:=]\s*)([^\s,;\]\}\"']+)",
    ),
    re.compile(r"\b(?:sk|xai|ghp|github_pat)-[A-Za-z0-9_\-]{12,}\b"),
)

SKIP_USER_PREFIXES = (
    "<environment_context>",
    "<permissions instructions>",
    "<skills_instructions>",
    "<apps_instructions>",
    "<plugins_instructions>",
)


def config_dir() -> Path:
    override = os.environ.get("AGENT_RELAY_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "agent-relay"


def state_dir() -> Path:
    override = os.environ.get("AGENT_RELAY_STATE_DIR")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "agent-relay"


def data_dir() -> Path:
    override = os.environ.get("AGENT_RELAY_DATA_DIR")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "agent-relay"


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    ensure_private_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(mode)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def valid_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except (OSError, UnicodeError):
        return


def tail_json_lines(path: Path, max_bytes: int = 1_000_000) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            start = max(0, size - max_bytes)
            stream.seek(start)
            data = stream.read()
        if start:
            newline = data.find(b"\n")
            data = data[newline + 1 :] if newline >= 0 else b""
        result: list[dict[str, Any]] = []
        for raw_line in data.splitlines():
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                result.append(value)
        return result
    except OSError:
        return []


def parse_time(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        return number
    if not isinstance(value, str) or not value:
        return fallback
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return fallback


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative_time(timestamp: float, now: float | None = None) -> str:
    if not timestamp:
        return "unknown"
    current = time.time() if now is None else now
    seconds = max(0, int(current - timestamp))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    days = hours // 24
    if days < 60:
        return f"{days}d ago"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def truncate(text: Any, limit: int = 4_000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(text)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[relay truncated {omitted} characters]"


def redact(text: Any, limit: int = 4_000) -> str:
    result = truncate(text, limit=limit)
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            result = pattern.sub(r"\1[REDACTED]", result)
        else:
            result = pattern.sub("[REDACTED]", result)
    return result


def clean_title(text: Any, limit: int = 64) -> str:
    value = redact(text, limit=400).replace("\n", " ").strip()
    value = re.sub(r"\s+", " ", value)
    if not value:
        return "Untitled session"
    return value if len(value) <= limit else value[: limit - 1] + "…"


def visible_user_text(text: str) -> bool:
    stripped = text.lstrip().lower()
    return bool(stripped) and not any(
        stripped.startswith(prefix) for prefix in SKIP_USER_PREFIXES
    )


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in {"text", "input_text", "output_text"}:
                    pieces.append(str(item.get("text", "")))
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(content, dict):
        for key in ("text", "content", "message", "summary"):
            if key in content:
                return content_text(content[key])
    return ""


@lru_cache(maxsize=512)
def git_root(path_text: str) -> Path | None:
    path = Path(path_text)
    if not path.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return Path(result.stdout.strip()).resolve()


def canonical(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def belongs_to_scope(session_cwd: Path, target: Path, exact: bool = False) -> bool:
    source = canonical(session_cwd)
    destination = canonical(target)
    if exact:
        return source == destination
    destination_root = git_root(str(destination))
    if destination_root is None:
        return source == destination
    source_root = git_root(str(source))
    return source_root == destination_root


def command_version(command: str | None) -> str | None:
    if command is None:
        return None
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else None


def render_command(argv: Iterable[str]) -> str:
    return shlex.join(list(argv))


def bounded_command(argv: list[str], cwd: Path, limit: int = 2_000_000) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"[relay could not run {render_command(argv)}: {error}]\n"
    output = result.stdout
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    return truncate(output, limit=limit)
