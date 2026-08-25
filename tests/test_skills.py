from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from agent_relay.models import AgentInfo
from agent_relay.skills import (
    conflicting_skill_names,
    discover_skills,
    shared_skills_root,
    sync_user_skills,
)


def write_skill(path: Path, name: str, description: str, body: str = "Instructions") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: |\n  {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


class SkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.data = self.root / "relay-data"
        self.previous_data = os.environ.get("AGENT_RELAY_DATA_DIR")
        os.environ["AGENT_RELAY_DATA_DIR"] = str(self.data)
        self.agents = {
            name: AgentInfo(name, "/bin/true", "test", self.root / f".{name}")
            for name in ("codex", "grok", "claude", "agy")
        }

    def tearDown(self) -> None:
        if self.previous_data is None:
            os.environ.pop("AGENT_RELAY_DATA_DIR", None)
        else:
            os.environ["AGENT_RELAY_DATA_DIR"] = self.previous_data
        self.temporary.cleanup()

    def test_sync_registers_user_skill_and_discovers_project_skill(self) -> None:
        source = self.agents["codex"].history_root / "skills" / "repair-api"
        write_skill(source, "repair-api", "Repair this API safely.")
        project = self.workspace / ".agents" / "skills" / "workspace-checks"
        write_skill(project, "workspace-checks", "Run workspace-specific checks.")

        preview = sync_user_skills(self.agents, apply=False)
        self.assertEqual([action.status for action in preview], ["would-link"])
        self.assertFalse(shared_skills_root().exists())

        applied = sync_user_skills(self.agents, apply=True)
        self.assertEqual([action.status for action in applied], ["linked"])
        registered = shared_skills_root() / "repair-api"
        self.assertTrue(registered.is_symlink())
        self.assertEqual(registered.resolve(), source.resolve())

        skills = discover_skills(self.workspace, self.agents)
        by_name = {skill.name: skill for skill in skills}
        self.assertEqual(by_name["repair-api"].scope, "shared")
        self.assertEqual(by_name["repair-api"].source, "relay")
        self.assertEqual(by_name["workspace-checks"].scope, "project")
        self.assertIn("workspace-specific", by_name["workspace-checks"].description)

    def test_sync_refuses_same_name_variants(self) -> None:
        write_skill(
            self.agents["codex"].history_root / "skills" / "deploy",
            "deploy",
            "Codex deployment procedure.",
        )
        write_skill(
            self.agents["grok"].history_root / "skills" / "deploy",
            "deploy",
            "Different Grok deployment procedure.",
        )

        actions = sync_user_skills(self.agents, apply=True)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].status, "conflict")
        self.assertFalse((shared_skills_root() / "deploy").exists())
        self.assertEqual(
            conflicting_skill_names(discover_skills(self.workspace, self.agents)),
            {"deploy"},
        )


if __name__ == "__main__":
    unittest.main()
