# -*- encoding: utf-8 -*-
"""闭环:训练进程真的轮询控制文件,MCP 能读回它写的 ack。

其他用例分别验证「MCP 写出了什么」和「控制面模块怎么应用」;这一个把两边接起来,
用假训练脚本加载仓库里真实的 ``core/utils/runtime_control.py``,证明「智能体在
训练进行中改一个超参数」这条链路整体是通的,而不是各自为政。
"""

import json
import time
import unittest
from pathlib import Path

from mcp_server import server

from .helpers import CONTROL_FAKE_MAIN, make_temp_repo, force_kill_record, remove_tree


class HotReloadLoopTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        # 换成会真的读控制文件的假训练脚本
        (self.layout.core_dir / "main.py").write_text(CONTROL_FAKE_MAIN, encoding="utf-8")
        server.configure(str(self.layout.root))
        self.record = None

    def tearDown(self):
        if self.record:
            try:
                server.stop_run(self.record["run_id"], force=True)
            except Exception:  # 环境不支持 ps 时退化成直接按 pid 清理
                force_kill_record(self.record)
        remove_tree(self.layout.root)

    # ---------------------------------------------------------------- 辅助

    def _log(self):
        try:
            return Path(self.record["log_path"]).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _wait_for_log(self, needle, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self._log():
                return True
            time.sleep(0.1)
        return False

    def _wait_for_ack(self, revision, timeout=20.0):
        deadline = time.monotonic() + timeout
        state = {}
        while time.monotonic() < deadline:
            state = server.get_control_state(self.record["run_id"])
            ack = state.get("ack") or {}
            if ack.get("revision") == revision:
                return state
            time.sleep(0.1)
        self.fail(
            "等不到 revision {} 的 ack;最后状态: {}".format(
                revision, json.dumps(state, ensure_ascii=False, default=str)[:500]
            )
        )

    # ---------------------------------------------------------------- 用例

    def test_hot_lr_change_takes_effect_and_is_acknowledged(self):
        self.record = server.launch_experiment(
            "vac", overrides={"num_epoch": 3}, extra_args=["--iterations", "300"]
        )
        run_id = self.record["run_id"]
        self.assertTrue(self._wait_for_log("[trainer] start"), self._log())
        self.assertTrue(self._wait_for_log("lr=0.001000"), self._log())

        written = server.set_hyperparameters(
            run_id, {"optimizer_args": {"base_lr": 0.002}}
        )
        self.assertEqual(written["revision"], 1)

        # 训练进程下一次轮询就应用了:日志里的学习率真的变了
        self.assertTrue(
            self._wait_for_log("lr=0.002000"),
            "训练进程没有应用热改后的学习率,日志:\n" + self._log()[-800:],
        )

        # 并且通过 ack 回到了 MCP
        state = self._wait_for_ack(1)
        self.assertFalse(state["pending"], state)
        self.assertIn(0.002, state["ack"]["applied"].values(), state["ack"])
        self.assertEqual(state["ack"]["ignored"], {}, state["ack"])

    def test_non_hot_key_is_rejected_with_a_reason_through_the_ack(self):
        self.record = server.launch_experiment(
            "vac", extra_args=["--iterations", "300"]
        )
        run_id = self.record["run_id"]
        self.assertTrue(self._wait_for_log("[trainer] start"), self._log())

        server.set_hyperparameters(run_id, {"batch_size": 8})
        state = self._wait_for_ack(1)
        self.assertEqual(state["ack"]["applied"], {}, state["ack"])
        self.assertIn("batch_size", state["ack"]["ignored"], state["ack"])
        self.assertIn("DataLoader", state["ack"]["ignored"]["batch_size"])

    def test_run_keeps_polling_after_a_change(self):
        """改完之后训练继续,后面的 revision 也能被处理(不是一次性通道)。"""
        self.record = server.launch_experiment(
            "vac", extra_args=["--iterations", "300"]
        )
        run_id = self.record["run_id"]
        self.assertTrue(self._wait_for_log("[trainer] start"), self._log())

        server.set_hyperparameters(run_id, {"optimizer_args": {"base_lr": 0.002}})
        self._wait_for_ack(1)
        server.set_hyperparameters(run_id, {"num_epoch": 42, "log_interval": 50})
        state = self._wait_for_ack(2)

        self.assertFalse(state["pending"], state)
        self.assertIn(42, state["ack"]["applied"].values(), state["ack"])
        self.assertTrue(self._wait_for_log("lr=0.002000"), self._log())


if __name__ == "__main__":
    unittest.main()
