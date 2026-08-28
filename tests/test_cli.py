from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_relay.cli import _print_sessions, _resolve_session, build_parser, command_doctor
from agent_relay.models import AgentInfo, Session


class CliTests(unittest.TestCase):
    def test_agent_qualified_exact_selector_wins_over_prefixes(self) -> None:
        root = Path("/tmp/relay-selector-test")
        exact_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        sessions = [
            Session("codex", exact_id, "exact", root, root / "exact", 1, 2),
            Session(
                "codex",
                exact_id + "-related",
                "prefix",
                root,
                root / "prefix",
                1,
                2,
            ),
        ]
        selected = _resolve_session(f"codex:{exact_id}", sessions)
        self.assertEqual(selected.session_id, exact_id)

    def test_resume_parser_accepts_no_git_diff(self) -> None:
        args = build_parser().parse_args(["resume", "--no-git-diff"])
        self.assertTrue(args.no_git_diff)

    def test_resume_parser_accepts_unsafe_native_provider_switch(self) -> None:
        args = build_parser().parse_args(
            ["resume", "--unsafe-native-provider-switch"]
        )
        self.assertTrue(args.unsafe_native_provider_switch)

    def test_session_list_labels_last_touch_target(self) -> None:
        root = Path("/tmp/relay-last-touch-test")
        session = Session(
            "codex",
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "Provider switch",
            root,
            root / "history.jsonl",
            1,
            2,
            model_provider="openai",
            last_touched_by="codex-native",
        )
        with redirect_stdout(io.StringIO()) as output:
            _print_sessions([session])
        self.assertIn("Last agent", output.getvalue())
        self.assertIn("codex-native", output.getvalue())

    def test_parser_accepts_agy_as_source_and_target(self) -> None:
        sessions = build_parser().parse_args(["sessions", "--agent", "agy"])
        resume = build_parser().parse_args(["resume", "--with", "agy"])
        self.assertEqual(sessions.agent, "agy")
        self.assertEqual(resume.target, "agy")

    def test_doctor_treats_uninstalled_agents_as_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex_home = root / ".codex"
            codex_home.mkdir()
            infos = {
                "codex": AgentInfo("codex", "/bin/true", "test", codex_home),
                "grok": AgentInfo("grok", None, None, root / ".grok"),
                "claude": AgentInfo("claude", None, None, root / ".claude"),
                "agy": AgentInfo(
                    "agy",
                    None,
                    None,
                    root / ".gemini" / "antigravity-cli",
                ),
            }
            with patch(
                "agent_relay.cli._scan", return_value=(infos, {}, [])
            ), patch(
                "agent_relay.cli.config_path", return_value=root / "config.json"
            ), patch("agent_relay.cli.state_dir", return_value=root / "state"):
                with redirect_stdout(io.StringIO()):
                    result = command_doctor(argparse.Namespace(refresh=False))
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
