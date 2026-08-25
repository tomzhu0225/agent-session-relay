from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_relay.adapters import AgyAdapter, ClaudeAdapter, CodexAdapter, GrokAdapter
from agent_relay.models import AgentInfo
from agent_relay.recovery import build_recovery_bundle
from agent_relay.skills import SkillRecord


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_codex_scan_normalize_and_commands(self) -> None:
        home = self.root / ".codex"
        session_id = "11111111-1111-4111-8111-111111111111"
        history = home / "sessions/2026/08/24" / f"rollout-2026-08-24T00-00-00-{session_id}.jsonl"
        records = [
            {
                "type": "session_meta",
                "timestamp": "2026-08-24T00:00:00Z",
                "payload": {
                    "id": session_id,
                    "cwd": str(self.workspace),
                    "timestamp": "2026-08-24T00:00:00Z",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-24T00:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix the parser"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-24T00:00:02Z",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": "API_KEY=very-secret-value pytest",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-24T00:00:03Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Parser updated"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-08-24T00:00:04Z",
                "payload": {"type": "turn_aborted", "reason": "connection lost"},
            },
        ]
        write_jsonl(history, records)
        write_jsonl(
            home / "session_index.jsonl",
            [{"id": session_id, "thread_name": "Parser repair", "updated_at": "2026-08-24T00:00:04Z"}],
        )
        adapter = CodexAdapter(AgentInfo("codex", "/bin/true", "test", home))
        adapter._load_titles()
        session = adapter.parse_session(history)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.title, "Parser repair")
        self.assertEqual(session.status, "interrupted")
        normalized = adapter.normalize(session)
        self.assertEqual(normalized.first_requests, ["Fix the parser"])
        rendered = "\n".join(event.text for event in normalized.events)
        self.assertIn("Parser updated", rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertNotIn("very-secret-value", rendered)
        self.assertEqual(normalized.last_error, "connection lost")
        self.assertEqual(
            adapter.native_resume_command(session),
            ["/bin/true", "-C", str(self.workspace), "resume", session_id],
        )

    def test_codex_skips_subagent_rollout_with_shared_parent_id(self) -> None:
        home = self.root / ".codex"
        parent_id = "11111111-1111-4111-8111-111111111111"
        child_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        history = (
            home
            / "sessions/2026/08/24"
            / f"rollout-2026-08-24T00-00-00-{child_id}.jsonl"
        )
        write_jsonl(
            history,
            [
                {
                    "type": "session_meta",
                    "timestamp": "2026-08-24T00:00:00Z",
                    "payload": {
                        "id": child_id,
                        "session_id": parent_id,
                        "thread_source": "subagent",
                        "source": {
                            "subagent": {
                                "thread_spawn": {"parent_thread_id": parent_id}
                            }
                        },
                        "cwd": str(self.workspace),
                    },
                },
                {
                    "type": "session_meta",
                    "timestamp": "2026-08-24T00:00:01Z",
                    "payload": {
                        "id": parent_id,
                        "session_id": parent_id,
                        "thread_source": "user",
                        "source": "cli",
                        "cwd": str(self.workspace),
                    },
                },
            ],
        )
        adapter = CodexAdapter(AgentInfo("codex", "/bin/true", "test", home))
        self.assertIsNone(adapter.parse_session(history))

    def test_grok_scan_and_normalize(self) -> None:
        home = self.root / ".grok"
        session_id = "22222222-2222-4222-8222-222222222222"
        session_dir = home / "sessions" / "%2Ftmp%2Fworkspace" / session_id
        session_dir.mkdir(parents=True)
        summary = {
            "info": {"id": session_id, "cwd": str(self.workspace)},
            "generated_title": "Repair API",
            "session_summary": "The API implementation is half complete.",
            "created_at": "2026-08-24T00:00:00Z",
            "updated_at": "2026-08-24T00:02:00Z",
            "current_model_id": "grok-test",
            "head_branch": "main",
        }
        (session_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        updates = [
            {
                "timestamp": "2026-08-24T00:00:01Z",
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "Finish the API"}}},
            },
            {
                "timestamp": "2026-08-24T00:00:02Z",
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "private thought"}}},
            },
            {
                "timestamp": "2026-08-24T00:00:03Z",
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "I changed the handler."}}},
            },
        ]
        write_jsonl(session_dir / "updates.jsonl", updates)
        adapter = GrokAdapter(AgentInfo("grok", "/bin/true", "test", home))
        session = adapter.parse_session(session_dir / "summary.json")
        self.assertIsNotNone(session)
        assert session is not None
        normalized = adapter.normalize(session)
        self.assertEqual(normalized.first_requests, ["Finish the API"])
        rendered = "\n".join(event.text for event in normalized.events)
        self.assertIn("I changed the handler", rendered)
        self.assertNotIn("private thought", rendered)
        self.assertIn("half complete", normalized.summary)

    def test_claude_uses_active_lineage_and_skips_thinking(self) -> None:
        home = self.root / ".claude"
        project = home / "projects/-tmp-workspace"
        history = project / "33333333-3333-4333-8333-333333333333.jsonl"
        records = [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "sessionId": history.stem,
                "cwd": str(self.workspace),
                "timestamp": "2026-08-24T00:00:00Z",
                "message": {"role": "user", "content": "Build the feature"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "sessionId": history.stem,
                "cwd": str(self.workspace),
                "timestamp": "2026-08-24T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": "Implemented the core."},
                    ],
                },
            },
            {
                "type": "assistant",
                "uuid": "side",
                "parentUuid": "u1",
                "sessionId": history.stem,
                "cwd": str(self.workspace),
                "isSidechain": True,
                "timestamp": "2026-08-24T00:00:02Z",
                "message": {"role": "assistant", "content": "sidechain output"},
            },
            {"type": "last-prompt", "sessionId": history.stem, "leafUuid": "a1"},
            {"type": "ai-title", "sessionId": history.stem, "aiTitle": "Feature build"},
        ]
        write_jsonl(history, records)
        adapter = ClaudeAdapter(AgentInfo("claude", "/bin/true", "test", home))
        session = adapter.parse_session(history)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.title, "Feature build")
        normalized = adapter.normalize(session)
        rendered = "\n".join(event.text for event in normalized.events)
        self.assertIn("Build the feature", rendered)
        self.assertIn("Implemented the core", rendered)
        self.assertNotIn("private reasoning", rendered)
        self.assertNotIn("sidechain output", rendered)

    def test_agy_matches_workspace_and_excludes_internal_transcript(self) -> None:
        home = self.root / ".gemini" / "antigravity-cli"
        session_id = "55555555-5555-4555-8555-555555555555"
        transcript = (
            home
            / "brain"
            / session_id
            / ".system_generated"
            / "logs"
            / "transcript.jsonl"
        )
        write_jsonl(
            transcript,
            [
                {
                    "step_index": 0,
                    "source": "USER_EXPLICIT",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-08-24T00:00:00Z",
                    "content": (
                        "<USER_REQUEST>\nFinish the AGY adapter\n</USER_REQUEST>\n"
                        "<ADDITIONAL_METADATA>private wrapper data</ADDITIONAL_METADATA>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "SYSTEM",
                    "type": "CHECKPOINT",
                    "status": "DONE",
                    "created_at": "2026-08-24T00:00:01Z",
                    "content": "private internal checkpoint",
                },
                {
                    "step_index": 2,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-08-24T00:00:02Z",
                    "content": "The AGY adapter is ready.",
                },
            ],
        )
        # Some AGY versions omit conversationId on the first history row.
        write_jsonl(
            home / "history.jsonl",
            [
                {
                    "display": "Finish the AGY adapter",
                    "timestamp": 1787529600000,
                    "workspace": str(self.workspace),
                }
            ],
        )

        adapter = AgyAdapter(AgentInfo("agy", "/bin/true", "test", home))
        sources = list(adapter.metadata_files())
        self.assertEqual(sources, [transcript])
        session = adapter.parse_session(transcript)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.cwd, self.workspace)
        self.assertEqual(session.title, "Finish the AGY adapter")
        self.assertEqual(session.status, "completed-turn")

        normalized = adapter.normalize(session)
        self.assertEqual(normalized.first_requests, ["Finish the AGY adapter"])
        rendered = "\n".join(event.text for event in normalized.events)
        self.assertIn("The AGY adapter is ready", rendered)
        self.assertNotIn("private internal checkpoint", rendered)
        self.assertNotIn("private wrapper data", rendered)
        self.assertEqual(
            adapter.native_resume_command(session),
            ["/bin/true", "--conversation", session_id],
        )
        self.assertEqual(
            adapter.cross_launch_command(self.workspace, "Read the relay bundle"),
            ["/bin/true", "--prompt-interactive", "Read the relay bundle"],
        )

    def test_agy_uses_summary_database_for_workspace(self) -> None:
        home = self.root / ".gemini" / "antigravity-cli"
        session_id = "66666666-6666-4666-8666-666666666666"
        conversation = home / "conversations" / f"{session_id}.db"
        conversation.parent.mkdir(parents=True)
        with sqlite3.connect(conversation) as connection:
            connection.execute("CREATE TABLE placeholder (value TEXT)")

        summary_database = home / "conversation_summaries.db"
        with sqlite3.connect(summary_database) as connection:
            connection.execute(
                """
                CREATE TABLE conversation_summaries (
                    conversation_id TEXT,
                    title TEXT,
                    preview TEXT,
                    last_modified_time INTEGER,
                    workspace_uris TEXT,
                    status TEXT,
                    agent_name TEXT
                )
                """
            )
            connection.execute(
                "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    "Summary-backed AGY session",
                    "",
                    1787529600000,
                    json.dumps([self.workspace.as_uri()]),
                    "SAVED",
                    "gemini-test",
                ),
            )

        adapter = AgyAdapter(AgentInfo("agy", "/bin/true", "test", home))
        sources = list(adapter.metadata_files())
        self.assertEqual(sources, [conversation])
        session = adapter.parse_session(conversation)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.cwd, self.workspace)
        self.assertEqual(session.title, "Summary-backed AGY session")
        self.assertEqual(session.status, "saved")
        self.assertEqual(session.model, "gemini-test")

    def test_recovery_bundle_is_private(self) -> None:
        home = self.root / ".grok"
        session_id = "44444444-4444-4444-8444-444444444444"
        session_dir = home / "sessions/x" / session_id
        session_dir.mkdir(parents=True)
        summary = {
            "info": {"id": session_id, "cwd": str(self.workspace)},
            "generated_title": "Bundle test",
            "session_summary": "Continue with tests.",
        }
        summary_path = session_dir / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        write_jsonl(session_dir / "updates.jsonl", [])
        adapter = GrokAdapter(AgentInfo("grok", "/bin/true", "test", home))
        session = adapter.parse_session(summary_path)
        assert session is not None
        skill = self.root / "skills" / "bundle-helper"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: bundle-helper\ndescription: Helps recover bundles.\n---\n\n"
            "PRIVATE_SKILL_BODY\n",
            encoding="utf-8",
        )
        state = self.root / "state"
        previous = os.environ.get("AGENT_RELAY_STATE_DIR")
        os.environ["AGENT_RELAY_STATE_DIR"] = str(state)
        try:
            bundle = build_recovery_bundle(
                session,
                adapter,
                available_skills=[
                    SkillRecord(
                        name="bundle-helper",
                        description="Helps recover bundles.",
                        path=skill,
                        scope="shared",
                        source="relay",
                    )
                ],
            )
            with patch(
                "agent_relay.recovery.git_root", return_value=self.workspace
            ), patch(
                "agent_relay.recovery.bounded_command", return_value="status\n"
            ) as command:
                bundle_without_diff = build_recovery_bundle(
                    session,
                    adapter,
                    include_git_diff=False,
                )
        finally:
            if previous is None:
                os.environ.pop("AGENT_RELAY_STATE_DIR", None)
            else:
                os.environ["AGENT_RELAY_STATE_DIR"] = previous
        self.assertEqual(bundle.root.stat().st_mode & 0o777, 0o700)
        for path in bundle.root.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["agent"], "grok")
        self.assertEqual(manifest["recovery"]["skill_count"], 1)
        self.assertIn("no Git worktree", bundle.git_status.read_text(encoding="utf-8"))
        skill_index = bundle.skills.read_text(encoding="utf-8")
        self.assertIn("bundle-helper", skill_index)
        self.assertIn(str(skill / "SKILL.md"), skill_index)
        self.assertNotIn("PRIVATE_SKILL_BODY", skill_index)
        self.assertIn(str(bundle.skills), bundle.bootstrap_prompt(session))
        self.assertEqual(
            bundle_without_diff.git_diff.read_text(encoding="utf-8"),
            "[relay: Git diff capture disabled by --no-git-diff]\n",
        )
        self.assertEqual(command.call_count, 1)
        no_diff_manifest = json.loads(
            bundle_without_diff.manifest.read_text(encoding="utf-8")
        )
        self.assertFalse(no_diff_manifest["recovery"]["git_diff_included"])


if __name__ == "__main__":
    unittest.main()
