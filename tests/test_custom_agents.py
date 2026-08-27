from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_relay.adapters import CodexAdapter, build_adapters
from agent_relay.cli import (
    _can_resume_natively,
    _scan,
    build_parser,
    command_agents_add,
    command_agents_remove,
    command_resume,
)
from agent_relay.config import discover_agents, save_discovery
from agent_relay.models import AgentInfo, Session
from agent_relay.skills import user_skills


class CustomAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "agents": {},
                    "custom_agents": {
                        "codex-glm": {
                            "adapter": "codex",
                            "command": "/bin/true",
                        }
                    },
                    "favorite_color": "orange",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _discover(self) -> dict[str, AgentInfo]:
        environment = {
            "CODEX_HOME": str(self.root / ".codex"),
            "GROK_HOME": str(self.root / ".grok"),
            "CLAUDE_CONFIG_DIR": str(self.root / ".claude"),
            "AGY_HOME": str(self.root / ".antigravity"),
        }
        with patch(
            "agent_relay.config.config_path", return_value=self.config_path
        ), patch.dict(os.environ, environment), patch(
            "agent_relay.config.command_version", return_value=None
        ), patch(
            "agent_relay.config.shutil.which", return_value=None
        ):
            return discover_agents(use_saved=True)

    def test_custom_codex_target_uses_codex_history_without_rescanning(self) -> None:
        infos = self._discover()
        custom = infos["codex-glm"]

        self.assertTrue(custom.custom)
        self.assertEqual(custom.history_adapter, "codex")
        self.assertEqual(custom.history_root, infos["codex"].history_root)
        self.assertTrue(custom.installed)
        self.assertFalse(custom.scan_history)

        adapters = build_adapters(infos)
        self.assertIsInstance(adapters["codex-glm"], CodexAdapter)
        self.assertEqual(adapters["codex-glm"].name, "codex-glm")

    def test_shared_codex_history_supports_native_wrapper_resume(self) -> None:
        infos = self._discover()
        adapters = build_adapters(infos)
        source = Session(
            "codex",
            "11111111-1111-4111-8111-111111111111",
            "Custom target",
            self.root,
            self.root / "history.jsonl",
            1,
            2,
        )

        self.assertTrue(
            _can_resume_natively(source, adapters["codex"], adapters["codex-glm"])
        )

    def test_scan_does_not_duplicate_shared_custom_history(self) -> None:
        infos = self._discover()
        session_id = "22222222-2222-4222-8222-222222222222"
        history = (
            infos["codex"].history_root
            / "sessions/2026/08/27"
            / f"rollout-2026-08-27T00-00-00-{session_id}.jsonl"
        )
        history.parent.mkdir(parents=True)
        history.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "timestamp": "2026-08-27T00:00:00Z",
                    "payload": {
                        "id": session_id,
                        "cwd": str(self.root),
                        "timestamp": "2026-08-27T00:00:00Z",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with patch("agent_relay.cli.discover_agents", return_value=infos), patch(
            "agent_relay.cli.state_dir", return_value=self.root / "state"
        ):
            _, _, sessions = _scan(refresh=True)
        self.assertEqual(
            [(session.agent, session.session_id) for session in sessions],
            [("codex", session_id)],
        )

    def test_resume_dry_run_uses_custom_command_for_shared_history(self) -> None:
        infos = self._discover()
        adapters = build_adapters(infos)
        source = Session(
            "codex",
            "33333333-3333-4333-8333-333333333333",
            "Dry run custom target",
            self.root,
            self.root / "history.jsonl",
            1,
            2,
        )
        args = argparse.Namespace(
            directory=str(self.root),
            session=source.selector,
            target="codex-glm",
            exact=False,
            all=False,
            limit=30,
            refresh=False,
            dry_run=True,
            no_git_diff=True,
        )

        with patch(
            "agent_relay.cli._scan", return_value=(infos, adapters, [source])
        ), redirect_stdout(io.StringIO()) as output:
            result = command_resume(args)
        self.assertEqual(result, 0)
        self.assertIn("Target: codex-glm (native resume)", output.getvalue())
        self.assertIn("/bin/true -C", output.getvalue())

    def test_setup_preserves_custom_definitions_and_other_config_values(self) -> None:
        infos = self._discover()
        with patch("agent_relay.config.config_path", return_value=self.config_path):
            save_discovery(infos)
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            saved["custom_agents"]["codex-glm"],
            {"adapter": "codex", "command": "/bin/true"},
        )
        self.assertEqual(saved["favorite_color"], "orange")
        self.assertNotIn("codex-glm", saved["agents"])

    def test_custom_agent_choices_and_management_commands(self) -> None:
        with patch("agent_relay.config.config_path", return_value=self.config_path):
            sessions = build_parser().parse_args(["sessions", "--agent", "codex-glm"])
            resume = build_parser().parse_args(["resume", "--with", "codex-glm"])
        self.assertEqual(sessions.agent, "codex-glm")
        self.assertEqual(resume.target, "codex-glm")

        command = self.root / "wrapper"
        command.write_text("#!/bin/sh\nexec codex \"$@\"\n", encoding="utf-8")
        command.chmod(0o755)
        args = argparse.Namespace(
            name="claude-custom",
            adapter="claude",
            command=str(command),
            history_home=None,
            scan_history=False,
        )
        remove_args = argparse.Namespace(name="claude-custom")
        with patch(
            "agent_relay.config.config_path", return_value=self.config_path
        ), patch("agent_relay.config.command_version", return_value=None):
            with redirect_stdout(io.StringIO()):
                command_agents_add(args)
            with redirect_stdout(io.StringIO()):
                command_agents_remove(remove_args)
        self.assertNotIn("claude-custom", self._discover())

    def test_alias_skills_are_not_discovered_twice(self) -> None:
        infos = self._discover()
        skill = infos["codex"].history_root / "skills" / "relay-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: relay-skill\ndescription: One skill\n---\nUse it.\n",
            encoding="utf-8",
        )

        empty_shared_root = self.root / "shared-skills"
        with patch("agent_relay.skills.shared_skills_root", return_value=empty_shared_root):
            records = user_skills(infos)
        self.assertEqual([record.name for record in records], ["relay-skill"])
        self.assertEqual(records[0].source, "codex")


if __name__ == "__main__":
    unittest.main()
