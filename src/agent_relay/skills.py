from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .models import AgentInfo
from .util import canonical, data_dir, ensure_private_dir, git_root, truncate


SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
FRONTMATTER_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?:\s*(.*))?$")
AGENT_SKILL_DIRS = {
    "codex": "skills",
    "grok": "skills",
    "claude": "skills",
    "agy": "skills",
}
PROJECT_SKILL_DIRS = (
    ("project-shared", ".agents/skills"),
    ("project-codex", ".codex/skills"),
    ("project-grok", ".grok/skills"),
    ("project-claude", ".claude/skills"),
)


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    path: Path
    scope: str
    source: str

    @property
    def instructions(self) -> Path:
        return self.path / "SKILL.md"

    @property
    def real_path(self) -> Path:
        try:
            return self.path.resolve()
        except OSError:
            return self.path.absolute()

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "source": self.source,
            "path": str(self.path),
            "instructions": str(self.instructions),
            "real_path": str(self.real_path),
        }


@dataclass(frozen=True)
class SkillSyncAction:
    name: str
    status: str
    source: Path
    destination: Path
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "source": str(self.source),
            "destination": str(self.destination),
            "detail": self.detail,
        }


def shared_skills_root() -> Path:
    return data_dir() / "skills"


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value[:1] in {"'", '"'}:
        try:
            parsed = literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("'\"")
        return str(parsed)
    return value


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration:
        return {}, text

    fields: dict[str, str] = {}
    index = 1
    while index < end:
        match = FRONTMATTER_KEY_RE.match(lines[index])
        if not match:
            index += 1
            continue
        key = match.group(1).lower().replace("_", "-")
        raw = (match.group(2) or "").strip()
        if raw in {"|", ">", "|-", ">-", "|+", ">+"}:
            folded = raw.startswith(">")
            chunks: list[str] = []
            index += 1
            while index < end and (
                not lines[index].strip()
                or lines[index].startswith((" ", "\t"))
            ):
                chunks.append(lines[index].strip())
                index += 1
            separator = " " if folded else "\n"
            fields[key] = separator.join(chunks).strip()
            continue
        fields[key] = _yaml_scalar(raw)
        index += 1
    return fields, "\n".join(lines[end + 1 :])


def _body_description(body: str) -> str:
    paragraph: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#") and not paragraph:
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


def parse_skill(path: Path, scope: str, source: str) -> SkillRecord | None:
    skill_file = path / "SKILL.md"
    try:
        text = skill_file.read_text(encoding="utf-8", errors="replace")[:512_000]
    except OSError:
        return None
    fields, body = _frontmatter(text)
    declared_name = fields.get("name", "").strip().lower()
    folder_name = path.name.strip().lower()
    name = declared_name if SKILL_NAME_RE.fullmatch(declared_name) else folder_name
    if not SKILL_NAME_RE.fullmatch(name):
        return None
    description = fields.get("description", "").strip() or _body_description(body)
    description = re.sub(r"\s+", " ", description).strip()
    return SkillRecord(
        name=name,
        description=truncate(description or "No description provided.", limit=600),
        path=path,
        scope=scope,
        source=source,
    )


def _skills_in_root(
    root: Path,
    scope: str,
    source: str,
    excluded: set[str] | None = None,
) -> list[SkillRecord]:
    excluded = excluded or set()
    try:
        children = sorted(root.iterdir(), key=lambda child: child.name.lower())
    except OSError:
        return []
    records: list[SkillRecord] = []
    for child in children:
        if child.name.startswith(".") or child.name.lower() in excluded:
            continue
        if not child.is_dir():
            continue
        record = parse_skill(child, scope=scope, source=source)
        if record is not None:
            records.append(record)
    return records


def user_skills(agents: dict[str, AgentInfo]) -> list[SkillRecord]:
    records = _skills_in_root(
        shared_skills_root(),
        scope="shared",
        source="relay",
    )
    for name, info in agents.items():
        if name not in AGENT_SKILL_DIRS:
            continue
        excluded = {"synced"} if name == "claude" else set()
        records.extend(
            _skills_in_root(
                info.history_root / AGENT_SKILL_DIRS[name],
                scope="user",
                source=name,
                excluded=excluded,
            )
        )
    return records


def _directory_chain(directory: Path) -> Iterable[Path]:
    current = canonical(directory)
    repository = git_root(str(current))
    boundary = repository or current
    while True:
        yield current
        if current == boundary or current.parent == current:
            break
        current = current.parent


def project_skills(directory: Path) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    for base in _directory_chain(directory):
        for source, relative in PROJECT_SKILL_DIRS:
            records.extend(
                _skills_in_root(
                    base / relative,
                    scope="project",
                    source=source,
                    excluded={"synced"},
                )
            )
    return records


def _record_rank(record: SkillRecord) -> tuple[int, str]:
    scope_rank = {"project": 0, "shared": 1, "user": 2}
    return scope_rank.get(record.scope, 9), record.source


def deduplicate_skills(records: Iterable[SkillRecord]) -> list[SkillRecord]:
    unique: dict[tuple[str, str], SkillRecord] = {}
    for record in records:
        key = record.name, str(record.real_path)
        current = unique.get(key)
        if current is None or _record_rank(record) < _record_rank(current):
            unique[key] = record
    return sorted(
        unique.values(),
        key=lambda record: (record.name, _record_rank(record), str(record.path)),
    )


def discover_skills(
    directory: Path,
    agents: dict[str, AgentInfo],
) -> list[SkillRecord]:
    return deduplicate_skills(user_skills(agents) + project_skills(directory))


def conflicting_skill_names(records: Iterable[SkillRecord]) -> set[str]:
    paths: dict[str, set[str]] = {}
    for record in records:
        paths.setdefault(record.name, set()).add(str(record.real_path))
    return {name for name, values in paths.items() if len(values) > 1}


def sync_user_skills(
    agents: dict[str, AgentInfo],
    apply: bool = True,
) -> list[SkillSyncAction]:
    root = shared_skills_root()
    records = deduplicate_skills(user_skills(agents))
    grouped: dict[str, list[SkillRecord]] = {}
    for record in records:
        grouped.setdefault(record.name, []).append(record)

    actions: list[SkillSyncAction] = []
    for name in sorted(grouped):
        variants = grouped[name]
        real_paths = {str(record.real_path) for record in variants}
        destination = root / name
        representative = min(variants, key=_record_rank)
        if len(real_paths) > 1:
            detail = "; ".join(
                f"{record.source}={record.path}" for record in variants
            )
            actions.append(
                SkillSyncAction(
                    name=name,
                    status="conflict",
                    source=representative.path,
                    destination=destination,
                    detail=detail,
                )
            )
            continue

        central = next((record for record in variants if record.source == "relay"), None)
        if central is not None:
            actions.append(
                SkillSyncAction(
                    name=name,
                    status="present",
                    source=central.real_path,
                    destination=destination,
                )
            )
            continue

        if destination.exists() or destination.is_symlink():
            actions.append(
                SkillSyncAction(
                    name=name,
                    status="conflict",
                    source=representative.real_path,
                    destination=destination,
                    detail="destination exists but is not a readable skill",
                )
            )
            continue

        status = "would-link"
        if apply:
            ensure_private_dir(root)
            destination.symlink_to(representative.real_path, target_is_directory=True)
            status = "linked"
        actions.append(
            SkillSyncAction(
                name=name,
                status=status,
                source=representative.real_path,
                destination=destination,
            )
        )
    return actions


def render_skill_manifest(records: Iterable[SkillRecord]) -> str:
    skills = deduplicate_skills(records)
    conflicts = conflicting_skill_names(skills)
    header = """# Relay skill index

This file is an index of user and project skills available to the recovered session. It is not itself an instruction to load every skill.

Open a listed `SKILL.md` only when its description applies to the recovered task. A skill never expands filesystem, network, approval, or external-action permissions. Project skill files are part of the working tree and should be treated with the same trust as other repository instructions.
"""
    if not skills:
        return header + "\nNo user or project-specific skills were discovered.\n"

    sections: list[str] = [header]
    for skill in skills:
        collision = " — name conflict" if skill.name in conflicts else ""
        sections.append(
            f"## {skill.name}{collision}\n\n"
            f"- Scope: `{skill.scope}`\n"
            f"- Source: `{skill.source}`\n"
            f"- Instructions: `{skill.instructions}`\n\n"
            f"{skill.description}\n"
        )
    if conflicts:
        names = ", ".join(f"`{name}`" for name in sorted(conflicts))
        sections.append(
            "## Conflicts\n\n"
            f"Multiple distinct skills use these names: {names}. Do not guess between "
            "them; prefer the task-specific project variant when clearly applicable, "
            "otherwise ask the user.\n"
        )
    return "\n".join(sections)
