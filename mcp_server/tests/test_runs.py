# -*- encoding: utf-8 -*-
"""运行记录、命令构造与进程状态机。

用假 main.py(见 helpers.FAKE_MAIN)当作被启动的实验进程,覆盖启动、日志
重定向、正常结束、异常退出与主动停止这几条路径。命令构造按上游布局断言:
``python main.py --config <abs>/configs/<name>.yaml --phase ...``,没有 ``--exp``,
也不传 ``--model``(模型由配置文件里的点号路径决定)。
"""

import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from mcp_server.errors import McpToolError
from mcp_server.runs import (
    STATUS_EXITED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_UNKNOWN,
    RunLauncher,
    RunStore,
)

from .helpers import force_kill_record, make_temp_repo, remove_tree


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.launcher = RunLauncher(self.layout)
        self.config_path = self.layout.configs_dir / "vac.yaml"

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_training_command_matches_upstream_layout(self):
        command = self.launcher.build_training_command("vac", config_path=self.config_path)
        self.assertEqual(command[0], self.layout.python)
        self.assertEqual(command[1], "main.py")
        self.assertEqual(command[2:4], ["--config", str(self.config_path)])
        self.assertEqual(command[4:6], ["--phase", "train"])
        # 上游没有 --exp,模型由配置文件的 model: 点号路径决定,不必传 --model
        self.assertNotIn("--exp", command)
        self.assertNotIn("--model", command)
        self.assertNotIn("--work-dir", command)

    def test_training_command_with_overrides(self):
        command = self.launcher.build_training_command(
            "vac",
            phase="test",
            work_dir="./work_dir/vac/",
            device="0,1",
            extra_args=["--batch-size", "2", "--load-weights", "/tmp/x.pt"],
            config_path=self.config_path,
        )
        self.assertEqual(command[command.index("--phase") + 1], "test")
        self.assertEqual(command[command.index("--work-dir") + 1], "./work_dir/vac/")
        self.assertEqual(command[command.index("--device") + 1], "0,1")
        self.assertEqual(command[-2:], ["--load-weights", "/tmp/x.pt"])
        self.assertNotIn("--exp", command)

    def test_training_command_requires_a_config_path(self):
        with self.assertRaises(McpToolError) as ctx:
            self.launcher.build_training_command("vac")
        self.assertIn("config", str(ctx.exception))

    def test_preprocess_command_runs_in_preprocess_dir(self):
        command = self.launcher.build_preprocess_command(
            "phoenix2014", dataset_root="/data/phoenix", process_image=True
        )
        self.assertEqual(command[1], "dataset_preprocess.py")
        self.assertEqual(command[2:4], ["--dataset", "phoenix2014"])
        self.assertIn("--dataset-root", command)
        self.assertIn("/data/phoenix", command)
        self.assertIn("--process-image", command)
        self.assertEqual(self.launcher.build_preprocess_command("CSL", script="dataset_preprocess-CSL-Daily.py")[1],
                         "dataset_preprocess-CSL-Daily.py")

    def test_reserved_flags_cannot_be_smuggled_through_extra_args(self):
        for flag in ("--config", "--phase", "--work-dir", "--device"):
            with self.subTest(flag=flag):
                with self.assertRaises(McpToolError) as ctx:
                    self.launcher.build_training_command(
                        "vac", extra_args=[flag, "x"], config_path=self.config_path
                    )
                self.assertIn("extra_args", str(ctx.exception))

    def test_reserved_flags_cannot_be_smuggled_in_other_spellings(self):
        """argparse 还认 ``--config=other`` 和无歧义前缀,按整串比对会被绕过。"""
        attacks = (
            ["--config=other"],
            ["--config", "other"],
            ["--conf", "other"],
            ["--work-dir=/tmp/x"],
            ["--dev", "3"],
            ["--ph", "test"],
        )
        for extra in attacks:
            with self.subTest(extra=extra):
                with self.assertRaises(McpToolError):
                    self.launcher.build_training_command(
                        "vac", extra_args=extra, config_path=self.config_path
                    )

    def test_unrelated_flags_still_pass_through(self):
        extra_args = ["--batch-size=2", "--load-weights", "/tmp/best_model.pt",
                      "--num-epoch", "5"]
        command = self.launcher.build_training_command(
            "vac", extra_args=extra_args, config_path=self.config_path
        )
        self.assertEqual(command[-len(extra_args):], extra_args)

    def test_extra_args_must_be_a_list_of_strings(self):
        for bad in ("--batch-size 2", [1, 2], {"--batch-size": 2}, 5):
            with self.subTest(bad=bad):
                with self.assertRaises(McpToolError):
                    self.launcher.build_training_command(
                        "vac", extra_args=bad, config_path=self.config_path
                    )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.store = RunStore(self.layout.runs_dir)

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_save_load_and_sort(self):
        first = {"run_id": "20260101-000000-a", "started_at": "2026-01-01T00:00:00+00:00"}
        second = {"run_id": "20260102-000000-b", "started_at": "2026-01-02T00:00:00+00:00"}
        self.store.save(first)
        self.store.save(second)

        self.assertEqual(self.store.load("20260101-000000-a")["run_id"], first["run_id"])
        self.assertEqual([item["run_id"] for item in self.store.list()],
                         ["20260102-000000-b", "20260101-000000-a"])
        self.assertEqual(len(self.store.list(limit=1)), 1)

    def test_unknown_and_illegal_ids(self):
        with self.assertRaises(McpToolError):
            self.store.load("missing")
        with self.assertRaises(McpToolError):
            self.store.load("../../etc/passwd")


class LifecycleTests(unittest.TestCase):
    """真实地起进程、看状态、再停掉(用假 main.py,不碰 GPU)。

    ``stop_run`` 发信号前会用 ``ps`` 读 pid 的命令行,而本测试沙箱拒绝执行
    ``ps``(Operation not permitted)。所以这里只把「读命令行」这一步替换掉:
    真正的 SIGTERM/SIGKILL、等待退出与状态机全部跑生产代码。
    """

    def setUp(self):
        self.layout = make_temp_repo()
        self.launcher = RunLauncher(self.layout)
        self.config_path = self.layout.configs_dir / "vac.yaml"
        self.started = []

    def tearDown(self):
        for record in self.started:
            try:
                self.launcher.stop(record["run_id"], force=True)
            except McpToolError:
                # 沙箱拒绝 ps,stop_run 核对不了命令行,只能直接按 pid 清理;
                # refresh() 会把退出码收回来,免得留下僵尸进程与 ResourceWarning。
                force_kill_record(record)
                deadline = time.monotonic() + 5
                while record.get("status") == STATUS_RUNNING and time.monotonic() < deadline:
                    time.sleep(0.1)
                    self.launcher.refresh(record)
        remove_tree(self.layout.root)

    def _wait_for(self, run_id, expected, timeout=15.0):
        deadline = time.monotonic() + timeout
        record = self.launcher.status(run_id)
        while time.monotonic() < deadline and record["status"] == STATUS_RUNNING:
            time.sleep(0.2)
            record = self.launcher.status(run_id)
        self.assertEqual(record["status"], expected, record)
        return record

    def test_successful_run_is_recorded_with_log(self):
        record = self.launcher.launch_training("vac", config_path=self.config_path)
        self.started.append(record)
        self.assertEqual(record["status"], STATUS_RUNNING)
        self.assertEqual(record["cwd"], str(self.layout.core_dir))
        self.assertEqual(record["config_path"], str(self.config_path))

        finished = self._wait_for(record["run_id"], STATUS_EXITED)
        self.assertEqual(finished["exit_code"], 0)
        self.assertTrue(finished["finished_at"])

        log = Path(record["log_path"]).read_text(encoding="utf-8")
        self.assertIn("[fake-main] done", log)
        self.assertIn(str(self.config_path), log)
        self.assertIn("--phase train", log)
        self.assertNotIn("--exp", log)

    def test_failing_run_is_marked_failed(self):
        record = self.launcher.launch_training(
            "vac", extra_args=["--fail"], config_path=self.config_path
        )
        self.started.append(record)
        finished = self._wait_for(record["run_id"], STATUS_FAILED)
        self.assertEqual(finished["exit_code"], 3)
        self.assertIn("故意失败", Path(record["log_path"]).read_text(encoding="utf-8"))

    def test_stop_terminates_the_process(self):
        record = self.launcher.launch_training(
            "vac", extra_args=["--sleep", "30"], config_path=self.config_path
        )
        self.started.append(record)
        time.sleep(1.0)
        self.assertTrue(self.launcher.status(record["run_id"])["pid_alive"])

        with mock.patch("mcp_server.runs._process_command",
                        return_value=record["command_str"]):
            stopped = self.launcher.stop(record["run_id"])
        self.assertEqual(stopped["status"], STATUS_STOPPED)
        self.assertFalse(stopped["pid_alive"])
        self.assertIn("已停止", stopped["note"])
        self.assertEqual(stopped["exit_code"], -15, "SIGTERM 的退出码")

    def test_stop_refuses_when_the_command_line_cannot_be_read(self):
        """读不到命令行时宁可拒绝,也不能凭一个 pid 就发信号。"""
        record = self.launcher.launch_training(
            "vac", extra_args=["--sleep", "30"], config_path=self.config_path
        )
        self.started.append(record)
        time.sleep(1.0)

        with mock.patch("mcp_server.runs._process_command", return_value=None):
            with self.assertRaises(McpToolError) as ctx:
                self.launcher.stop(record["run_id"])
        self.assertIn("读不到", str(ctx.exception))
        self.assertTrue(self.launcher.status(record["run_id"])["pid_alive"])

    def test_stop_refuses_foreign_process(self):
        """只依据 pid 就发信号太危险:命令行对不上时必须拒绝。"""
        foreign = subprocess.Popen(  # noqa: S603 - 测试用的外来进程
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )

        def reap_foreign():
            foreign.kill()
            foreign.wait(timeout=5)

        self.addCleanup(reap_foreign)
        record = {
            "run_id": "20260101-000000-foreign",
            "kind": "train",
            "experiment": "vac",
            "command": ["python", "main.py", "--config", str(self.config_path)],
            "log_path": str(self.layout.runs_dir / "foreign.log"),
            "pid": foreign.pid,
            "status": STATUS_RUNNING,
            "started_at": "2026-01-01T00:00:00+00:00",
        }
        self.launcher.store.save(record)
        with mock.patch("mcp_server.runs._process_command",
                        return_value="python -c 'import time; time.sleep(30)'"):
            with self.assertRaises(McpToolError) as ctx:
                self.launcher.stop(record["run_id"])
        self.assertIn("拒绝停止", str(ctx.exception))
        self.assertIn("main.py", str(ctx.exception))
        self.assertIsNone(foreign.poll(), "拒绝之后外来进程必须还活着")

    def test_record_without_pid_handle_becomes_unknown(self):
        """服务重启后只剩记录:进程不在又没有退出码,标为 unknown。"""
        record = {
            "run_id": "20260101-000000-stale",
            "kind": "train",
            "experiment": "vac",
            "command": ["python", "main.py"],
            "pid": 99999999,
            "status": STATUS_RUNNING,
            "started_at": "2026-01-01T00:00:00+00:00",
        }
        self.launcher.store.save(record)
        refreshed = self.launcher.status(record["run_id"])
        self.assertEqual(refreshed["status"], STATUS_UNKNOWN)
        self.assertFalse(refreshed["pid_alive"])


if __name__ == "__main__":
    unittest.main()
