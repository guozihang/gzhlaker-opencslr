# -*- encoding: utf-8 -*-
"""带超参数覆盖启动 + 训练中途热改的端到端链路。

用临时仓库里的假 main.py 覆盖进程生命周期,所以不需要 torch/GPU:验证的是
「MCP 把超参数送到了哪里、exp.yaml 有没有被动、控制文件写没写、临时配置有没有
留下垃圾」这条链路本身。
"""

import time
import unittest
from pathlib import Path

from mcp_server import server
from mcp_server.errors import McpToolError

from .helpers import make_temp_repo, force_kill_record, remove_tree

# 工具函数被 server.tool() 包装过:McpToolError 会翻译成 SDK 的 ToolError
# (这正是客户端看到的那一层),所以断言要同时接受两者。
try:
    from mcp.server.mcpserver.exceptions import ToolError as _SdkToolError
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp.exceptions import ToolError as _SdkToolError

TOOL_ERRORS = (McpToolError, _SdkToolError)


class LaunchOverrideTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.original_text = self.layout.exp_config.read_text(encoding="utf-8")
        server.configure(str(self.layout.root))
        self.started = []

    def tearDown(self):
        for record in self.started:
            try:
                server.stop_run(record["run_id"], force=True)
            except Exception:  # 清理失败不能掩盖用例本身的结论
                force_kill_record(record)
        remove_tree(self.layout.root)

    # ------------------------------------------------------------------ 工具

    def _launch(self, **kwargs):
        record = server.launch_experiment("vac", **kwargs)
        self.started.append(record)
        return record

    def _wait_status(self, run_id, expected, timeout=20.0):
        deadline = time.monotonic() + timeout
        record = server.get_run_status(run_id)
        while time.monotonic() < deadline and record["status"] != expected:
            time.sleep(0.2)
            record = server.get_run_status(run_id)
        self.assertEqual(record["status"], expected, record)
        return record

    def _configs_listing(self):
        return sorted(path.name for path in self.layout.configs_dir.iterdir())

    # ------------------------------------------------------- 临时配置与覆盖

    def test_overrides_go_into_a_temp_config_not_exp_yaml(self):
        record = self._launch(overrides={"num_epoch": 3, "model_args": {"use_bn": 0}})
        temp = Path(record["config_path"])
        self.assertEqual(temp.parent, self.layout.configs_dir)
        self.assertTrue(temp.name.startswith("_mcp_run_"))
        self.assertIn(str(temp), record["command_str"])
        self.assertIn("--control-file", record["command_str"])

        # 临时配置里是「深合并之后」的值:改了 use_bn,网络节里的兄弟键还在
        import yaml

        section = yaml.safe_load(temp.read_text(encoding="utf-8"))["vac"]
        self.assertEqual(section["model_args"]["use_bn"], 0)
        self.assertEqual(section["model_args"]["num_classes"], 1296)
        self.assertEqual(section["num_epoch"], 3)

        # exp.yaml 一个字节都没动
        self.assertEqual(self.layout.exp_config.read_text(encoding="utf-8"), self.original_text)
        # 快照留档,便于事后复现这次运行
        self.assertTrue(Path(record["config_snapshot"]).is_file())

        # 假 main.py 收到的就是临时配置(进程写日志有延迟,等它结束再读)
        self._wait_status(record["run_id"], "exited")
        log = Path(record["log_path"]).read_text(encoding="utf-8")
        self.assertIn(str(temp), log)

    def test_temp_config_is_cleaned_up_when_the_run_ends(self):
        record = self._launch(overrides={"num_epoch": 3})
        temp = Path(record["config_path"])
        snapshot = Path(record["config_snapshot"])
        self._wait_status(record["run_id"], "exited")
        self.assertFalse(temp.exists(), "进程结束后临时配置应被清理")
        self.assertTrue(snapshot.is_file(), "快照必须保留,否则这次运行无法复现")

    def test_launch_without_overrides_still_uses_exp_yaml(self):
        record = self._launch()
        self.assertEqual(Path(record["config_path"]), self.layout.exp_config)
        self.assertIsNone(record.get("config_snapshot"))
        self.assertIsNone(record.get("ephemeral_config"))

    def test_dry_run_writes_nothing(self):
        before = self._configs_listing()
        result = server.launch_experiment(
            "vac", overrides={"num_epoch": 3}, dry_run=True
        )
        self.assertTrue(result["dry_run"])
        self.assertIn("--config", result["command"])
        self.assertTrue(str(result["would_write_config"]).startswith(str(self.layout.configs_dir)))
        self.assertEqual(self._configs_listing(), before, "dry_run 不应落盘")
        self.assertEqual(
            self.layout.exp_config.read_text(encoding="utf-8"), self.original_text
        )

    def test_bad_override_leaves_no_temp_config_behind(self):
        before = self._configs_listing()
        with self.assertRaises(TOOL_ERRORS) as ctx:
            server.launch_experiment("vac", overrides={"num_epoch": "not-an-int"})
        self.assertIn("校验失败", str(ctx.exception))
        self.assertEqual(self._configs_listing(), before, "校验失败不应留下临时配置")

    def test_unknown_override_key_is_rejected(self):
        before = self._configs_listing()
        with self.assertRaises(TOOL_ERRORS):
            server.launch_experiment("vac", overrides={"no_such_key": 1})
        self.assertEqual(self._configs_listing(), before)

    # ---------------------------------------------------------------- 热改

    def test_hot_change_writes_control_file_and_reports_pending(self):
        record = self._launch(extra_args=["--sleep", "30"])
        run_id = record["run_id"]

        first = server.set_hyperparameters(run_id, {"optimizer_args": {"base_lr": 0.0002}})
        self.assertEqual(first["revision"], 1)

        state = server.get_control_state(run_id)
        self.assertTrue(state["pending"], "假 main.py 不轮询,应当一直是待处理")
        self.assertEqual(
            state["last_command"]["overrides"]["optimizer_args"]["base_lr"], 0.0002
        )
        self.assertIn("轮询", state["note"])

        second = server.set_hyperparameters(run_id, {"num_epoch": 5, "loss_weights": {"Dist": 0.5}})
        self.assertEqual(second["revision"], 2)
        self.assertEqual(server.get_control_state(run_id)["last_command"]["revision"], 2)

    def test_hot_change_rejected_once_the_run_is_over(self):
        record = self._launch()
        self._wait_status(record["run_id"], "exited")
        with self.assertRaises(TOOL_ERRORS) as ctx:
            server.set_hyperparameters(record["run_id"], {"num_epoch": 5})
        self.assertIn("进程已不在", str(ctx.exception))

    def test_hot_change_rejects_unknown_run(self):
        with self.assertRaises(TOOL_ERRORS):
            server.set_hyperparameters("20260101-000000-nonexistent", {"num_epoch": 5})

    def test_run_status_carries_the_control_state(self):
        record = self._launch(extra_args=["--sleep", "30"])
        server.set_hyperparameters(record["run_id"], {"num_epoch": 5})
        status = server.get_run_status(record["run_id"])
        self.assertIn("control", status)
        self.assertEqual(status["control"]["last_command"]["revision"], 1)
        self.assertTrue(status["control"]["pending"])

    # ------------------------------------------------------------ 超参数清单

    def test_get_hyperparameters_marks_what_can_be_hot_changed(self):
        info = server.get_hyperparameters("vac")
        arguments = {item["name"]: item for item in info["arguments"]}

        self.assertEqual(arguments["num_epoch"]["current"], 80)
        self.assertTrue(arguments["num_epoch"]["hot"])
        self.assertIn("set_hyperparameters", arguments["num_epoch"]["change_note"])

        self.assertFalse(arguments["batch_size"]["hot"])
        self.assertIn("DataLoader", arguments["batch_size"]["change_note"])
        self.assertIn("launch_experiment", arguments["batch_size"]["change_note"])

        # 字典类:整体不可热改,但有子键可以 —— 不能报成「只能在启动前设置」
        optimizer = arguments["optimizer_args"]
        self.assertEqual(optimizer["hot"], "partial")
        self.assertIn("optimizer_args.base_lr", optimizer["hot_keys"])
        self.assertIn("base_lr", optimizer["change_note"])
        self.assertIn("学习率", optimizer["hot_note"])

        nested = info["nested"]["model_args"]
        self.assertFalse(nested["hot"])
        self.assertIn("use_bn", nested["allowed"])
        self.assertEqual(nested["current"]["num_classes"], 1296)

        self.assertIn("optimizer_args.base_lr", info["hot_keys"])
        self.assertIn("set_hyperparameters", info["how_to_change"]["训练中途"])

    def test_get_hyperparameters_rejects_unknown_experiment(self):
        with self.assertRaises(TOOL_ERRORS):
            server.get_hyperparameters("not_an_experiment")


if __name__ == "__main__":
    unittest.main()
