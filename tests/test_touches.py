from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_relay.models import Session
from agent_relay.touches import load_session_targets, record_session_target


class TouchesTests(unittest.TestCase):
    def test_records_last_native_target_by_history_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = Session(
                "codex",
                "11111111-1111-4111-8111-111111111111",
                "Shared history",
                root,
                root / "rollout.jsonl",
                1,
                2,
            )
            with patch("agent_relay.touches.state_dir", return_value=root / "state"):
                path = record_session_target(session, "codex-glm")
                self.assertEqual(
                    load_session_targets(),
                    {str(session.source_path): "codex-glm"},
                )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
