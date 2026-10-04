# -*- encoding: utf-8 -*-
"""带超参数覆盖启动实验(上游没有运行时控制面,所以覆盖只能在启动前)。

用临时仓库里的假 main.py 覆盖进程生命周期,所以不需要 torch/GPU:验证的是
「覆盖被写到哪里、实验自己的配置文件有没有被动、dry_run 有没有落盘、临时配置
有没有留下垃圾」这条链路本身。
"""

import time
import unittest
from pathlib import Path

import yaml

from mcp_server import server
from mcp_server.config import EPHEMERAL_PREFIX
from mcp_server.errors import McpToolError
from mcp_server.runs import RunStore

from .helpers import force_kill_record, make_temp_repo, remove_tree

# 工具函数被 server.tool() 包装过:McpToolError 会翻译成 SDK 的 ToolError
# (这正是客户端看到的那一层),所以断言要同时接受两者。
try:
    from mcp.server.mcpserver.exceptions import ToolError as _SdkToolError
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp.exceptions import ToolError as _SdkToolError

TOOL_ERRORS = (McpToolError, _SdkToolError)

VAC_YAML = "vac.yaml"


class LaunchOverrideTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.experiment_file = self.layout.configs_dir / VAC_YAML
        self.original_text = self.experiment_file.read_text(encoding="utf-8")
        server.configure(str(self.layout.root))
        self.started = []

    def tearDown(self):
        for record in self.started:
            try:
                server.stop_run(record["run_id"], force=True)
            except Exception:  # 清理失败不能掩盖用例本身的结论
                # 沙箱拒绝 ps 时 stop_run 无法核对命令行,只能按 pid 清理;
                # 再刷新一次状态把退出码收回来,避免留下 ResourceWarning。
                force_kill_record(record)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if server.get_run_status(record["run_id"])["status"] != "running":
                        break
                    time.sleep(0.1)
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

    def _snapshots(self):
        return sorted(path.name for path in self.layout.runs_dir.glob("*.config.yaml"))

    def _stored(self, run_id):
        """读运行记录的原始 JSON(工具返回值只挑了一部分字段出来)。"""
        return RunStore(self.layout.runs_dir).load(run_id)

    # ------------------------------------------------------- 临时配置与覆盖

    def test_overrides_use_a_temp_config_and_spare_the_experiment_file(self):
        record = self._launch(overrides={"num_epoch": 3, "model_args": {"use_bn": 0}})
        temp = Path(record["config_path"])
        self.assertEqual(temp.parent, self.layout.configs_dir)
        self.assertTrue(temp.name.startswith("_mcp_run_"))
        self.assertIn(str(temp), record["command_str"])
        # 上游没有 --exp,也没有控制文件
        self.assertNotIn("--exp", record["command_str"])
        self.assertNotIn("--control-file", record["command_str"])

        # 临时配置是「深合并之后」的值:只改 use_bn,model_args 的兄弟键还在
        doc = yaml.safe_load(temp.read_text(encoding="utf-8"))
        self.assertEqual(doc["num_epoch"], 3)
        self.assertEqual(doc["model_args"]["use_bn"], 0)
        self.assertEqual(doc["model_args"]["num_classes"], 1296)
        self.assertEqual(doc["model"], "models.build_function.build_vac")
        self.assertEqual(doc["dataset"], "phoenix2014")

        # 实验自己的配置文件一个字节都没动
        self.assertEqual(self.experiment_file.read_text(encoding="utf-8"), self.original_text)
        # 快照留档,便于事后复现这次运行;记录里也标明了这次用的是临时配置
        snapshot = Path(record["config_snapshot"])
        self.assertTrue(snapshot.is_file())
        self.assertEqual(snapshot.read_text(encoding="utf-8"), temp.read_text(encoding="utf-8"))
        stored = self._stored(record["run_id"])
        self.assertEqual(stored["ephemeral_config"], str(temp))
        self.assertEqual(stored["config_snapshot"], str(snapshot))

        # 假 main.py 收到的就是临时配置,而不是实验文件
        self._wait_status(record["run_id"], "exited")
        log = Path(record["log_path"]).read_text(encoding="utf-8")
        self.assertIn(str(temp), log)
        self.assertNotIn(str(self.experiment_file), log)

    def test_temp_config_is_cleaned_up_when_the_run_ends(self):
        record = self._launch(overrides={"num_epoch": 3})
        temp = Path(record["config_path"])
        snapshot = Path(record["config_snapshot"])
        self.assertTrue(temp.is_file())
        self._wait_status(record["run_id"], "exited")
        self.assertFalse(temp.exists(), "进程结束后临时配置应被清理")
        self.assertTrue(snapshot.is_file(), "快照必须保留,否则这次运行无法复现")

    def test_launch_without_overrides_uses_the_experiment_config(self):
        before = self._configs_listing()
        record = self._launch()
        self.assertEqual(Path(record["config_path"]), self.experiment_file)
        self.assertIn(str(self.experiment_file), record["command_str"])
        self.assertIsNone(record["config_snapshot"])
        stored = self._stored(record["run_id"])
        self.assertIsNone(stored["ephemeral_config"])
        self.assertIsNone(stored["config_snapshot"])
        self.assertEqual(self._configs_listing(), before, "没有覆盖就不该写临时配置")

    def test_dry_run_writes_nothing(self):
        before = self._configs_listing()
        result = server.launch_experiment("vac", overrides={"num_epoch": 3}, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertIn("--config", result["command"])

        would_write = Path(result["would_write_config"])
        self.assertEqual(would_write.parent, self.layout.configs_dir)
        self.assertTrue(would_write.name.startswith(EPHEMERAL_PREFIX))
        self.assertFalse(would_write.exists(), "dry_run 不应落盘")

        self.assertEqual(self._configs_listing(), before)
        self.assertEqual(self._snapshots(), [])
        self.assertEqual(server.list_runs()["count"], 0, "dry_run 不应产生运行记录")
        self.assertEqual(self.experiment_file.read_text(encoding="utf-8"), self.original_text)

        # 不带覆盖的 dry_run:直接用实验自己的配置文件,也不会有临时配置
        plain = server.launch_experiment("vac", dry_run=True)
        self.assertIsNone(plain["would_write_config"])
        self.assertEqual(Path(plain["config_path"]), self.experiment_file)

    def test_failed_preflight_leaves_no_temp_config(self):
        before = self._configs_listing()
        with self.assertRaises(TOOL_ERRORS) as ctx:
            server.launch_experiment("vac", overrides={"num_epoch": "not-an-int"})
        self.assertIn("校验失败", str(ctx.exception))
        self.assertIn("not-an-int", str(ctx.exception))
        self.assertEqual(self._configs_listing(), before, "校验失败不应留下临时配置")
        self.assertEqual(self._snapshots(), [], "校验失败不应留下快照")
        self.assertEqual(server.list_runs()["count"], 0, "校验失败不应产生运行记录")

    def test_unknown_override_key_is_rejected(self):
        before = self._configs_listing()
        with self.assertRaises(TOOL_ERRORS) as ctx:
            server.launch_experiment("vac", overrides={"no_such_key": 1})
        self.assertIn("no_such_key", str(ctx.exception))
        self.assertEqual(self._configs_listing(), before)
        self.assertEqual(self._snapshots(), [])

    def test_unknown_experiment_and_bad_overrides_type_are_rejected(self):
        with self.assertRaises(TOOL_ERRORS) as ctx:
            server.launch_experiment("no_such_experiment")
        self.assertIn("不存在", str(ctx.exception))
        with self.assertRaises(TOOL_ERRORS) as ctx:
            server.launch_experiment("vac", overrides=["num_epoch"])
        self.assertIn("overrides", str(ctx.exception))


class HyperparameterTests(unittest.TestCase):
    """超参数清单:当前值来自真实解析链,而「能不能热改」如实为 null。"""

    def setUp(self):
        self.layout = make_temp_repo()
        server.configure(str(self.layout.root))

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_current_values_come_from_the_real_config_chain(self):
        info = server.get_hyperparameters("vac")
        self.assertTrue(info["resolution_ok"], info.get("resolution_error"))
        self.assertEqual(info["model"], "models.build_function.build_vac")
        self.assertEqual(info["dataset"], "phoenix2014")
        self.assertEqual(info["config_path"], str(self.layout.configs_dir / VAC_YAML))

        arguments = {item["name"]: item for item in info["arguments"]}
        self.assertEqual(arguments["num_epoch"]["current"], 80)
        self.assertEqual(arguments["dataset"]["current"], "phoenix2014")
        self.assertIn("config", arguments)

    def test_no_runtime_control_plane_is_reported_as_unknown(self):
        info = server.get_hyperparameters("vac")
        # 上游没有运行时控制面:hot_keys 必须是 null,而不是一张假表
        self.assertIsNone(info["hot_keys"])

        arguments = {item["name"]: item for item in info["arguments"]}
        self.assertIsNone(arguments["num_epoch"]["hot"])
        self.assertIn("运行时控制面", arguments["num_epoch"]["change_note"])

        nested = info["nested"]["model_args"]
        self.assertIsNone(nested["allowed"], "上游没有嵌套键白名单,不能假装有")
        self.assertIsNone(nested["hot"])
        self.assertEqual(nested["current"]["num_classes"], 1296)

        self.assertIn("没有运行时控制面", info["how_to_change"]["训练中途"])

    def test_unknown_experiment_is_rejected(self):
        with self.assertRaises(TOOL_ERRORS):
            server.get_hyperparameters("not_an_experiment")


if __name__ == "__main__":
    unittest.main()
