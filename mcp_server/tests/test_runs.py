# -*- encoding: utf-8 -*-
"""运行记录、命令构造与进程状态机。

用假 main.py(见 helpers.FAKE_MAIN)当作被启动的实验进程,覆盖启动、日志
重定向、正常结束、异常退出与主动停止这几条路径。
"""

import os
import time
import unittest
from pathlib import Path

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

from .helpers import make_temp_repo, remove_tree


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.launcher = RunLauncher(self.layout)

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_training_command_matches_repo_convention(self):
        command = self.launcher.build_training_command("vac")
        self.assertEqual(command[1], "main.py")
        self.assertEqual(command[2:4], ["--config", str(self.layout.exp_config)])
        self.assertEqual(command[4:6], ["--exp", "vac"])
        self.assertEqual(command[6:8], ["--phase", "train"])
        self.assertNotIn("--work-dir", command)

    def test_training_command_with_overrides(self):
        command = self.launcher.build_training_command(
            "vac",
            phase="test",
            work_dir="./work_dir/vac/",
            device="0,1",
            extra_args=["--batch-size", "2", "--load-weights", "/tmp/x.pt"],
        )
        self.assertIn("--work-dir", command)
        self.assertEqual(command[command.index("--work-dir") + 1], "./work_dir/vac/")
        self.assertEqual(command[command.index("--device") + 1], "0,1")
        self.assertEqual(command[command.index("--phase") + 1], "test")
        self.assertEqual(command[-2:], ["--load-weights", "/tmp/x.pt"])

    def test_preprocess_command_runs_in_preprocess_dir(self):
        command = self.launcher.build_preprocess_command(
            "phoenix2014", dataset_root="/data/phoenix", process_image=True
        )
        self.assertEqual(command[1], "dataset_preprocess.py")
        self.assertIn("--process-image", command)
        self.assertIn("/data/phoenix", command)

    def test_reserved_flags_cannot_be_smuggled_through_extra_args(self):
        for flag in ("--config", "--exp", "--work-dir", "--phase", "--device"):
            with self.assertRaises(McpToolError):
                self.launcher.build_training_command("vac", extra_args=[flag, "x"])

    def test_reserved_flags_cannot_be_smuggled_in_other_spellings(self):
        """argparse 还认 --exp=x 和无歧义前缀,按整串比对会被绕过。"""
        for flag in ("--exp=other", "--work-dir=/tmp/x", "--ex", "--dev"):
            with self.assertRaises(McpToolError):
                self.launcher.build_training_command("vac", extra_args=[flag])

    def test_unrelated_flags_still_pass_through(self):
        extra_args = ["--batch-size=2", "--load-weights", "/tmp/best_model.pt",
                      "--num-epoch", "5"]
        command = self.launcher.build_training_command("vac", extra_args=extra_args)
        self.assertEqual(command[-len(extra_args):], extra_args)

    def test_extra_args_must_be_string_list(self):
        with self.assertRaises(McpToolError):
            self.launcher.build_training_command("vac", extra_args="--batch-size 2")


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
    """真实地起进程、看状态、再停掉(用假 main.py,不碰 GPU)。"""

    def setUp(self):
        self.layout = make_temp_repo()
        self.launcher = RunLauncher(self.layout)
        self.started = []

    def tearDown(self):
        for record in self.started:
            try:
                self.launcher.stop(record["run_id"], force=True)
            except McpToolError:
                pass
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
        record = self.launcher.launch_training("vac")
        self.started.append(record)
        self.assertEqual(record["status"], STATUS_RUNNING)
        self.assertEqual(record["cwd"], str(self.layout.core_dir))

        finished = self._wait_for(record["run_id"], STATUS_EXITED)
        self.assertEqual(finished["exit_code"], 0)
        self.assertTrue(finished["finished_at"])

        log = Path(record["log_path"]).read_text(encoding="utf-8")
        self.assertIn("[fake-main] done", log)
        self.assertIn("--exp vac", log)

    def test_failing_run_is_marked_failed(self):
        record = self.launcher.launch_training("vac", extra_args=["--fail"])
        self.started.append(record)
        finished = self._wait_for(record["run_id"], STATUS_FAILED)
        self.assertEqual(finished["exit_code"], 3)
        self.assertIn("故意失败", Path(record["log_path"]).read_text(encoding="utf-8"))

    def test_stop_terminates_the_process(self):
        record = self.launcher.launch_training("vac", extra_args=["--sleep", "30"])
        self.started.append(record)
        time.sleep(1.0)
        self.assertTrue(self.launcher.status(record["run_id"])["pid_alive"])

        stopped = self.launcher.stop(record["run_id"])
        self.assertEqual(stopped["status"], STATUS_STOPPED)
        self.assertFalse(stopped["pid_alive"])
        self.assertIn("已停止", stopped["note"])

    def test_stop_refuses_foreign_process(self):
        """只依据 pid 就发信号太危险:命令行对不上时必须拒绝。"""
        record = {
            "run_id": "20260101-000000-foreign",
            "kind": "train",
            "experiment": "vac",
            "command": ["python", "main.py", "--exp", "vac"],
            "log_path": str(self.layout.runs_dir / "foreign.log"),
            "pid": os.getpid(),  # 本测试进程自己
            "status": STATUS_RUNNING,
            "started_at": "2026-01-01T00:00:00+00:00",
        }
        self.launcher.store = RunStore(self.layout.runs_dir)
        self.launcher.store.save(record)
        with self.assertRaises(McpToolError) as ctx:
            self.launcher.stop(record["run_id"])
        self.assertIn("拒绝停止", str(ctx.exception))
        self.assertTrue(os.getpid() > 0)  # 自己当然还活着

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
