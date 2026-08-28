from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from . import __version__
from .adapters import AgentAdapter, build_adapters
from .config import (
    BUILTIN_AGENTS,
    add_custom_agent,
    configured_agent_names,
    config_path,
    discover_agents,
    remove_custom_agent,
    save_discovery,
    update_custom_agent_provider,
)
from .index import IndexCache
from .models import AgentInfo, Session
from .recovery import build_recovery_bundle
from .skills import (
    SkillRecord,
    conflicting_skill_names,
    discover_skills,
    shared_skills_root,
    sync_user_skills,
)
from .touches import load_session_targets, record_session_target
from .util import belongs_to_scope, canonical, relative_time, render_command, state_dir


def _scan(
    refresh: bool = False,
    use_saved: bool = True,
) -> tuple[dict[str, AgentInfo], dict[str, AgentAdapter], list[Session]]:
    infos = discover_agents(use_saved=use_saved)
    adapters = build_adapters(infos)
    cache = IndexCache(refresh=refresh)
    sessions: list[Session] = []
    for adapter in adapters.values():
        if adapter.info.scan_history:
            sessions.extend(adapter.scan(cache))
    cache.save()
    recorded_targets = load_session_targets()
    sessions = [
        _attribute_session(
            session,
            infos,
            adapters,
            recorded_targets.get(str(session.source_path), ""),
        )
        for session in sessions
    ]
    sessions.sort(key=lambda session: session.updated_at, reverse=True)
    return infos, adapters, sessions


def _scoped_sessions(
    sessions: list[Session],
    directory: Path,
    exact: bool,
    include_all: bool,
    agent: str | None = None,
) -> list[Session]:
    result = []
    for session in sessions:
        if agent and session.agent != agent:
            continue
        if include_all or belongs_to_scope(session.cwd, directory, exact=exact):
            result.append(session)
    return result


def _session_data(session: Session, number: int | None = None) -> dict[str, object]:
    data: dict[str, object] = {
        "agent": session.agent,
        "session_id": session.session_id,
        "selector": session.selector,
        "title": session.title,
        "cwd": str(session.cwd),
        "updated_at": session.updated_at,
        "created_at": session.created_at,
        "status": session.status,
        "branch": session.branch,
        "model": session.model,
        "model_provider": session.model_provider,
        "last_touched_by": session.last_touched_by or session.agent,
        "history_path": str(session.source_path),
    }
    if number is not None:
        data["number"] = number
    return data


def _print_sessions(sessions: list[Session], show_cwd: bool = False) -> None:
    if not sessions:
        print("No matching sessions found.")
        return
    rows: list[list[str]] = []
    for number, session in enumerate(sessions, start=1):
        row = [
            str(number),
            session.last_touched_by or session.agent,
            relative_time(session.updated_at),
            session.status,
            session.title,
        ]
        if show_cwd:
            row.append(str(session.cwd))
        rows.append(row)
    headers = ["#", "Last agent", "Updated", "State", "Title"]
    if show_cwd:
        headers.append("Directory")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = min(max(widths[index], len(value)), 70)

    def render(row: list[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            display = value
            if len(display) > widths[index]:
                display = display[: widths[index] - 1] + "…"
            cells.append(display.ljust(widths[index]))
        return "  ".join(cells).rstrip()

    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in rows:
        print(render(row))


def _print_skills(skills: list[SkillRecord]) -> None:
    if not skills:
        print("No user or project-specific skills found.")
        return
    conflicts = conflicting_skill_names(skills)
    rows = []
    for skill in skills:
        name = skill.name + (" !" if skill.name in conflicts else "")
        rows.append(
            [skill.scope, skill.source, name, skill.description, str(skill.path)]
        )
    headers = ["Scope", "Source", "Skill", "Description", "Path"]
    widths = [len(header) for header in headers]
    limits = [10, 16, 34, 72, 80]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = min(max(widths[index], len(value)), limits[index])

    def render(row: list[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            display = value
            if len(display) > widths[index]:
                display = display[: widths[index] - 1] + "…"
            cells.append(display.ljust(widths[index]))
        return "  ".join(cells).rstrip()

    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in rows:
        print(render(row))
    if conflicts:
        print("\n! Same-name variants differ; Relay will not choose or overwrite them.")


def _resolve_session(selector: str, sessions: list[Session]) -> Session:
    selector = selector.strip()
    if selector.isdigit():
        number = int(selector)
        if 1 <= number <= len(sessions):
            return sessions[number - 1]
        raise ValueError(f"session number {number} is outside the displayed list")

    lowered = selector.lower()
    exact = [
        session
        for session in sessions
        if lowered in {session.session_id.lower(), session.selector.lower()}
    ]
    if len(exact) == 1:
        return exact[0]
    prefixes = [
        session
        for session in sessions
        if session.session_id.lower().startswith(lowered)
        or session.selector.lower().startswith(lowered)
    ]
    if len(prefixes) == 1:
        return prefixes[0]
    titles = [session for session in sessions if lowered in session.title.lower()]
    if len(titles) == 1:
        return titles[0]
    matches = prefixes or titles
    if matches:
        raise ValueError(f"selector {selector!r} matches {len(matches)} sessions")
    raise ValueError(f"no session matches {selector!r}")


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)


def _choose_target(source: Session, infos: dict[str, AgentInfo]) -> str:
    installed = [name for name, info in infos.items() if info.installed]
    if not installed:
        raise ValueError("no supported target agents are installed")
    default = source.agent if source.agent in installed else installed[0]
    print("\nResume using:")
    for number, name in enumerate(installed, start=1):
        suffix = " (source/default)" if name == default else ""
        print(f"  {number}. {name}{suffix}")
    answer = _ask(f"Target [{default}]: ")
    if not answer:
        return default
    if answer.isdigit() and 1 <= int(answer) <= len(installed):
        return installed[int(answer) - 1]
    lowered = answer.lower()
    if lowered in installed:
        return lowered
    raise ValueError(f"unknown or unavailable target agent: {answer}")


def _same_history(left: AgentAdapter, right: AgentAdapter) -> bool:
    if left.info.history_adapter != right.info.history_adapter:
        return False
    try:
        return left.history_home.resolve() == right.history_home.resolve()
    except OSError:
        return False


def _normalized_provider(value: str) -> str:
    return value.strip().casefold()


def _target_label(name: str) -> str:
    return "codex-native" if name == "codex" else name


def _attribute_session(
    session: Session,
    infos: dict[str, AgentInfo],
    adapters: dict[str, AgentAdapter],
    recorded_target: str = "",
) -> Session:
    """Label the latest compatible target recorded by Relay or the provider."""

    provider = _normalized_provider(session.model_provider)
    source_adapter = adapters.get(session.agent)
    if source_adapter is None:
        return replace(session, last_touched_by=session.agent)

    if provider:
        if provider == _normalized_provider(source_adapter.info.model_provider):
            return replace(session, last_touched_by=_target_label(session.agent))
        matches = [
            name
            for name, info in infos.items()
            if name != session.agent
            and _normalized_provider(info.model_provider) == provider
            and name in adapters
            and _same_history(source_adapter, adapters[name])
        ]
        if len(matches) == 1:
            return replace(
                session,
                agent=matches[0],
                last_touched_by=_target_label(matches[0]),
            )

    recorded_adapter = adapters.get(recorded_target)
    if recorded_adapter is not None and _same_history(
        source_adapter, recorded_adapter
    ):
        return replace(
            session,
            agent=recorded_target,
            last_touched_by=_target_label(recorded_target),
        )

    if not provider:
        return replace(session, last_touched_by=_target_label(session.agent))
    fallback = (
        f"codex[{session.model_provider}]"
        if session.agent == "codex"
        else session.agent
    )
    return replace(session, last_touched_by=fallback)


def _can_resume_natively(
    source: Session,
    source_adapter: AgentAdapter,
    target_adapter: AgentAdapter,
    allow_unsafe_provider_switch: bool = False,
) -> bool:
    shares_native_history = source.agent == target_adapter.info.name or _same_history(
        source_adapter,
        target_adapter,
    )
    if not shares_native_history:
        return False
    return allow_unsafe_provider_switch or not target_adapter.native_resume_blocker(
        source,
        source_adapter,
    )


def command_agents_list(args: argparse.Namespace) -> int:
    infos = discover_agents(use_saved=True)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "config": str(config_path()),
                    "agents": [
                        {
                            "name": info.name,
                            "adapter": info.history_adapter,
                            "custom": info.custom,
                            "installed": info.installed,
                            "command": info.command,
                            "version": info.version,
                            "history_home": str(info.history_root),
                            "scans_history": info.scan_history,
                            "model_provider": info.model_provider,
                        }
                        for info in infos.values()
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"Relay agents: {config_path()}\n")
    for info in infos.values():
        marker = "✓" if info.installed else "–"
        kind = info.history_adapter + (" (custom)" if info.custom else "")
        mode = "source/target" if info.scan_history else "target-only"
        status = info.version or "not installed"
        provider = info.model_provider or "unspecified"
        print(
            f"{marker} {info.name:<12} adapter={kind:<14} mode={mode:<13} "
            f"provider={provider:<11} {status} — {info.history_root}"
        )
    return 0


def command_agents_add(args: argparse.Namespace) -> int:
    try:
        path = add_custom_agent(
            args.name,
            args.adapter,
            args.command,
            history_home=args.history_home,
            scan_history=args.scan_history,
            model_provider=args.model_provider,
        )
    except ValueError as error:
        print(f"relay: {error}", file=sys.stderr)
        return 2

    infos = discover_agents(use_saved=True)
    info = infos[args.name]
    status = info.version or "command found"
    print(f"Added custom target {info.name} ({info.history_adapter}) to {path}")
    print(f"  command: {info.command}")
    print(f"  provider: {info.model_provider or 'unspecified'}")
    print(f"  history: {info.history_root} ({status})")
    return 0


def command_agents_remove(args: argparse.Namespace) -> int:
    try:
        path = remove_custom_agent(args.name)
    except ValueError as error:
        print(f"relay: {error}", file=sys.stderr)
        return 2
    print(f"Removed custom target {args.name} from {path}")
    return 0


def command_agents_update(args: argparse.Namespace) -> int:
    try:
        path = update_custom_agent_provider(args.name, args.model_provider)
    except ValueError as error:
        print(f"relay: {error}", file=sys.stderr)
        return 2
    print(
        f"Updated custom target {args.name}: "
        f"provider={args.model_provider.strip()} in {path}"
    )
    return 0


def command_setup(args: argparse.Namespace) -> int:
    infos = discover_agents(use_saved=False)
    path = save_discovery(infos)
    _, _, sessions = _scan(refresh=True, use_saved=True)
    counts = {name: 0 for name in infos}
    for session in sessions:
        counts[session.agent] += 1

    if args.json:
        print(
            json.dumps(
                {
                    "config": str(path),
                    "agents": {
                        name: {
                            "installed": info.installed,
                            "command": info.command,
                            "version": info.version,
                            "history_home": str(info.history_root),
                            "model_provider": info.model_provider,
                            "sessions": counts[name],
                        }
                        for name, info in infos.items()
                    },
                },
                indent=2,
            )
        )
        return 0

    print(f"Relay configuration: {path}")
    for name, info in infos.items():
        marker = "✓" if info.installed else "–"
        status = info.version or "not installed"
        history = info.history_root
        mode = "source/target" if info.scan_history else "target-only"
        print(
            f"{marker} {name:<12} {status} — {counts[name]} sessions — "
            f"{mode} — {history}"
        )
    return 0


def command_sessions(args: argparse.Namespace) -> int:
    directory = canonical(args.directory)
    _, _, all_sessions = _scan(refresh=args.refresh)
    sessions = _scoped_sessions(
        all_sessions,
        directory,
        exact=args.exact,
        include_all=args.all,
        agent=args.agent,
    )
    sessions = sessions[: args.limit]
    if args.json:
        print(
            json.dumps(
                [_session_data(session, number) for number, session in enumerate(sessions, 1)],
                indent=2,
            )
        )
    else:
        scope = "all directories" if args.all else str(directory)
        print(f"Sessions for {scope}:\n")
        _print_sessions(sessions, show_cwd=args.all)
    return 0 if sessions else 1


def command_resume(args: argparse.Namespace) -> int:
    directory = canonical(args.directory)
    infos, adapters, all_sessions = _scan(refresh=args.refresh)
    matching = _scoped_sessions(
        all_sessions,
        directory,
        exact=args.exact,
        include_all=args.all,
    )
    if not matching:
        print(f"No sessions found for {directory}.", file=sys.stderr)
        return 1

    if args.session:
        try:
            source = _resolve_session(args.session, matching)
        except ValueError as error:
            print(f"relay: {error}", file=sys.stderr)
            return 2
    else:
        if not sys.stdin.isatty():
            print("relay: --session is required when stdin is not interactive", file=sys.stderr)
            return 2
        displayed = matching[: args.limit]
        print(f"Sessions for {directory}:\n")
        _print_sessions(displayed, show_cwd=args.all)
        answer = _ask("\nChoose session (number, ID, or title; q to cancel): ")
        if answer.lower() in {"q", "quit", "cancel"}:
            return 0
        try:
            source = _resolve_session(answer, displayed)
        except ValueError as error:
            print(f"relay: {error}", file=sys.stderr)
            return 2

    target = args.target.lower() if args.target else ""
    if not target:
        if sys.stdin.isatty():
            try:
                target = _choose_target(source, infos)
            except ValueError as error:
                print(f"relay: {error}", file=sys.stderr)
                return 2
        else:
            target = source.agent
    if target not in adapters:
        print(f"relay: unsupported target agent {target!r}", file=sys.stderr)
        return 2
    target_adapter = adapters[target]
    if target_adapter.command is None:
        print(f"relay: target agent {target!r} is not installed", file=sys.stderr)
        return 2
    if not source.cwd.is_dir():
        print(f"relay: source directory no longer exists: {source.cwd}", file=sys.stderr)
        return 2

    bundle = None
    source_adapter = adapters[source.agent]
    shares_native_history = source.agent == target_adapter.info.name or _same_history(
        source_adapter,
        target_adapter,
    )
    native_blocker = (
        target_adapter.native_resume_blocker(source, source_adapter)
        if shares_native_history
        else ""
    )
    unsafe_native = bool(
        getattr(args, "unsafe_native_provider_switch", False) and native_blocker
    )
    resume_natively = _can_resume_natively(
        source,
        source_adapter,
        target_adapter,
        allow_unsafe_provider_switch=unsafe_native,
    )
    if resume_natively:
        command = target_adapter.native_resume_command(source)
        mode = "unsafe native resume" if unsafe_native else "native resume"
    else:
        available_skills = discover_skills(source.cwd, infos)
        bundle = build_recovery_bundle(
            source,
            source_adapter,
            available_skills=available_skills,
            include_git_diff=not args.no_git_diff,
        )
        prompt = bundle.bootstrap_prompt(source)
        command = target_adapter.cross_launch_command(source.cwd, prompt)
        mode = "provider-safe recovery" if native_blocker else "cross-agent recovery"

    print(f"Source: {source.selector} — {source.title}")
    print(f"Target: {target} ({mode})")
    if native_blocker:
        print(f"Safety: {native_blocker}")
    if bundle:
        print(f"Recovery bundle: {bundle.root}")
    print(f"Command: {render_command(command)}")
    if args.dry_run:
        return 0

    if resume_natively:
        record_session_target(source, target)

    os.chdir(source.cwd)
    os.execv(command[0], command)
    return 0


def command_skills_list(args: argparse.Namespace) -> int:
    directory = canonical(getattr(args, "directory", "."))
    infos = discover_agents(use_saved=True)
    skills = discover_skills(directory, infos)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "registry": str(shared_skills_root()),
                    "directory": str(directory),
                    "conflicts": sorted(conflicting_skill_names(skills)),
                    "skills": [skill.to_dict() for skill in skills],
                },
                indent=2,
            )
        )
        return 0
    print(f"Relay skill registry: {shared_skills_root()}")
    print(f"Effective skills for:  {directory}\n")
    _print_skills(skills)
    return 0


def command_skills_sync(args: argparse.Namespace) -> int:
    infos = discover_agents(use_saved=True)
    apply = not args.dry_run
    actions = sync_user_skills(infos, apply=apply)
    conflicts = [action for action in actions if action.status == "conflict"]
    if args.json:
        print(
            json.dumps(
                {
                    "registry": str(shared_skills_root()),
                    "applied": apply,
                    "actions": [action.to_dict() for action in actions],
                },
                indent=2,
            )
        )
    else:
        print(f"Relay skill registry: {shared_skills_root()}")
        if not actions:
            print("No personal agent skills were found to register.")
        for action in actions:
            marker = {
                "linked": "+",
                "would-link": "+",
                "present": "=",
                "conflict": "!",
            }.get(action.status, "-")
            print(
                f"{marker} {action.name:<32} {action.status:<10} "
                f"{action.source} -> {action.destination}"
            )
            if action.detail:
                print(f"  {action.detail}")
    return 2 if conflicts else 0


def command_doctor(args: argparse.Namespace) -> int:
    infos, _, sessions = _scan(refresh=args.refresh)
    counts = {name: 0 for name in infos}
    for session in sessions:
        counts[session.agent] += 1
    healthy = any(info.installed for info in infos.values())
    print(f"Config: {config_path()} ({'present' if config_path().exists() else 'not initialized'})")
    print(f"Index:  {state_dir() / 'index.json'}")
    for name, info in infos.items():
        history_exists = info.history_root.is_dir()
        okay = not info.installed or history_exists
        healthy = healthy and okay
        marker = "✓" if info.installed and history_exists else ("!" if info.installed else "–")
        print(
            f"{marker} {name:<12} command={info.command or 'missing'} "
            f"history={'ok' if history_exists else 'missing'} sessions={counts[name]}"
        )
    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relay",
        description="Browse and resume local coding-agent sessions across CLI tools.",
    )
    parser.add_argument("--version", action="version", version=f"relay {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="detect agents and history stores")
    setup.add_argument("--json", action="store_true", help="emit machine-readable output")
    setup.set_defaults(handler=command_setup)

    agents = subparsers.add_parser(
        "agents",
        help="list built-in agents and manage custom relay targets",
    )
    agents.set_defaults(handler=command_agents_list, json=False)
    agent_commands = agents.add_subparsers(dest="agents_command")

    agents_list = agent_commands.add_parser("list", help="list configured agents")
    agents_list.add_argument("--json", action="store_true")
    agents_list.set_defaults(handler=command_agents_list)

    agents_add = agent_commands.add_parser(
        "add",
        help="add a custom target backed by an existing history adapter",
    )
    agents_add.add_argument("name", help="target name, for example codex-glm")
    agents_add.add_argument("--adapter", choices=BUILTIN_AGENTS, required=True)
    agents_add.add_argument(
        "--command",
        required=True,
        help="one executable command or path; put provider arguments in a wrapper script",
    )
    agents_add.add_argument(
        "--history-home",
        help="history root (defaults to the adapter's discovered history home)",
    )
    agents_add.add_argument(
        "--scan-history",
        action="store_true",
        help="also list this target's sessions (do not enable for a shared history)",
    )
    agents_add.add_argument(
        "--model-provider",
        help="provider identity stored in sessions, for example ZAI",
    )
    agents_add.set_defaults(handler=command_agents_add, json=False)

    agents_update = agent_commands.add_parser(
        "update",
        help="update metadata for a custom target",
    )
    agents_update.add_argument("name")
    agents_update.add_argument(
        "--model-provider",
        required=True,
        help="provider identity stored in sessions, for example ZAI",
    )
    agents_update.set_defaults(handler=command_agents_update, json=False)

    agents_remove = agent_commands.add_parser(
        "remove",
        help="remove a custom relay target",
    )
    agents_remove.add_argument("name")
    agents_remove.set_defaults(handler=command_agents_remove, json=False)

    sessions = subparsers.add_parser("sessions", help="list sessions for a directory")
    sessions.add_argument("directory", nargs="?", default=".")
    sessions.add_argument("--exact", action="store_true", help="match the exact directory instead of its Git worktree")
    sessions.add_argument("--all", action="store_true", help="include every directory")
    agent_choices = configured_agent_names()
    sessions.add_argument("--agent", choices=agent_choices)
    sessions.add_argument("--limit", type=int, default=30)
    sessions.add_argument("--refresh", action="store_true", help="rebuild cached metadata")
    sessions.add_argument("--json", action="store_true", help="emit machine-readable output")
    sessions.set_defaults(handler=command_sessions)

    resume = subparsers.add_parser("resume", help="select a session and resume it with any supported agent")
    resume.add_argument("directory", nargs="?", default=".")
    resume.add_argument("--session", help="session number, ID, agent:ID, or unique title text")
    resume.add_argument(
        "--with",
        dest="target",
        choices=agent_choices,
        help="target agent",
    )
    resume.add_argument("--exact", action="store_true", help="match the exact directory instead of its Git worktree")
    resume.add_argument("--all", action="store_true", help="select from every directory")
    resume.add_argument("--limit", type=int, default=30)
    resume.add_argument("--refresh", action="store_true", help="rebuild cached metadata")
    resume.add_argument("--dry-run", action="store_true", help="prepare and print the launch without starting an agent")
    resume.add_argument(
        "--unsafe-native-provider-switch",
        action="store_true",
        help=(
            "force a same-history Codex provider switch; provider-specific reasoning "
            "records may later fail compaction"
        ),
    )
    resume.add_argument(
        "--no-git-diff",
        action="store_true",
        help="omit tracked and staged Git diffs from a cross-agent recovery bundle",
    )
    resume.set_defaults(handler=command_resume)

    doctor = subparsers.add_parser("doctor", help="check agent commands and indexed histories")
    doctor.add_argument("--refresh", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    skills = subparsers.add_parser(
        "skills",
        help="share user and project skills across relay handoffs",
    )
    skills.set_defaults(handler=command_skills_list, directory=".", json=False)
    skill_commands = skills.add_subparsers(dest="skills_command")

    skills_list = skill_commands.add_parser(
        "list",
        help="list effective user and project skills",
    )
    skills_list.add_argument("directory", nargs="?", default=".")
    skills_list.add_argument("--json", action="store_true")
    skills_list.set_defaults(handler=command_skills_list)

    skills_sync = skill_commands.add_parser(
        "sync",
        help="register personal agent skills in Relay's shared library",
    )
    skills_sync.add_argument(
        "--dry-run",
        action="store_true",
        help="show links without creating them",
    )
    skills_sync.add_argument("--json", action="store_true")
    skills_sync.set_defaults(handler=command_skills_sync)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
