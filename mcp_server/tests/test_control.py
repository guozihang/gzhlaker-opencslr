# -*- encoding: utf-8 -*-
"""训练中途热改的控制通道(MCP 侧):写控制文件、读 ack、判断是否有未确认改动。"""

import json
import unittest
from pathlib import Path

from mcp_server import control
from mcp_server.errors import McpToolError

from .helpers import make_temp_repo, remove_tree


class ControlChannelTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.control_file = control.control_path(self.layout.runs_dir, "20260101-000000-vac-abcd")
        self.control_file.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_revision_starts_at_one_and_increases(self):
        first = control.write_overrides(self.control_file, {"num_epoch": 5})
        self.assertEqual(first["revision"], 1)
        second = control.write_overrides(self.control_file, {"num_epoch": 6})
        self.assertEqual(second["revision"], 2)
        payload = json.loads(Path(self.control_file).read_text(encoding="utf-8"))
        self.assertEqual(payload["revision"], 2)
        self.assertEqual(payload["overrides"], {"num_epoch": 6})

    def test_state_reports_pending_until_ack_catches_up(self):
        control.write_overrides(self.control_file, {"num_epoch": 5})
        state = control.state(self.control_file)
        self.assertIsNone(state["ack"])
        self.assertTrue(state["pending"], "训练还没回执时应当标记为 pending")

        control.ack_path(self.control_file).write_text(
            json.dumps({"revision": 1, "applied": {"num_epoch": 5}, "ignored": {}}),
            encoding="utf-8",
        )
        self.assertFalse(control.state(self.control_file)["pending"])

        control.write_overrides(self.control_file, {"num_epoch": 7})
        self.assertTrue(control.state(self.control_file)["pending"])

    def test_state_without_control_file(self):
        state = control.state(self.control_file)
        self.assertIsNone(state["control"])
        self.assertIsNone(state["ack"])
        self.assertIsNone(state["pending"], "还没写过控制文件时不该声称 pending")

    def test_overrides_must_be_a_non_empty_mapping(self):
        for bad in ({}, None, [], "num_epoch=5"):
            with self.subTest(bad=bad):
                with self.assertRaises(McpToolError):
                    control.write_overrides(self.control_file, bad)

    def test_non_serialisable_override_is_rejected_before_writing(self):
        with self.assertRaises(McpToolError) as ctx:
            control.write_overrides(self.control_file, {"model_args": {1, 2, 3}})
        self.assertIn("JSON", str(ctx.exception))
        self.assertFalse(Path(self.control_file).exists(), "拒绝时不应留下文件")

    def test_corrupted_files_do_not_raise(self):
        Path(self.control_file).write_text("{ not json", encoding="utf-8")
        state = control.state(self.control_file)
        self.assertIsNone(state["control"])
        # 损坏的文件按「没有 revision」处理,下一次写入从 1 开始
        self.assertEqual(control.write_overrides(self.control_file, {"num_epoch": 1})["revision"], 1)

    def test_ack_lives_next_to_the_control_file(self):
        self.assertEqual(
            control.ack_path(self.control_file),
            Path(str(self.control_file) + ".ack.json"),
        )


if __name__ == "__main__":
    unittest.main()
