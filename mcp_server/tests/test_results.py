# -*- encoding: utf-8 -*-
"""结果、日志与 checkpoint 的读取。"""

import unittest

from mcp_server.errors import McpToolError
from mcp_server.results import list_artifacts, list_work_dirs, read_results, tail_log
from mcp_server.runs import RunStore

from .helpers import make_temp_repo, remove_tree, touch, write_json


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.work_root = self.layout.core_dir / "work_dir"
        self.work_dir = self.work_root / "vac_smkd"
        self.work_dir.mkdir(parents=True)

        write_json(
            self.work_dir / "experiment_result.json",
            {
                "experiment_name": "vac_smkd",
                "seed": 0,
                "dataset": "phoenix2014",
                "split": "test",
                "model": "vac",
                "wer": 19.2,
                "total_samples": 642,
                "skip_rate": 0.0016,
                "status": "valid",
                "timestamp": "2026-09-20T10:00:00",
            },
        )
        write_json(
            self.work_dir / "experiment_result_dev.json",
            {"split": "dev", "wer": 18.4, "total_samples": 540, "status": "valid"},
        )
        write_json(
            self.work_dir / "sample_statistics_test.json",
            {"split": "test", "total_samples": 642, "num_successful": 641,
             "num_skipped": 1, "num_failed": 0, "skip_rate": 0.0016, "status": "valid"},
        )
        (self.work_dir / "log_20260920_100000.log").write_text(
            "".join(f"line {index}\n" for index in range(1, 51)), encoding="utf-8"
        )

        # checkpoint 的两种写法:目录内(_best)与文件名前缀形式
        touch(self.work_dir / "_best_model.pt")
        touch(self.work_dir / "dev_19.20_epoch5_model.pt")
        touch(self.work_root / "vac_smkd_best_model.pt")

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_read_results_summarises_key_metrics(self):
        payload = read_results(self.layout, "./work_dir/vac_smkd/")
        self.assertTrue(payload["exists"])
        self.assertEqual(payload["summary"]["experiment_result"]["wer"], 19.2)
        self.assertEqual(payload["summary"]["experiment_result_dev"]["split"], "dev")
        self.assertEqual(payload["summary"]["sample_statistics_test"]["num_skipped"], 1)
        self.assertEqual(
            payload["results"]["experiment_result"]["experiment_name"], "vac_smkd"
        )

    def test_read_results_on_missing_dir_explains_next_step(self):
        payload = read_results(self.layout, "./work_dir/never_ran/")
        self.assertFalse(payload["exists"])
        self.assertIn("launch_experiment", payload["note"])

    def test_list_artifacts_finds_both_checkpoint_forms(self):
        artifacts = list_artifacts(self.layout, "./work_dir/vac_smkd/")
        names = {item["name"] for item in artifacts["checkpoints"]}
        self.assertIn("_best_model.pt", names)
        self.assertIn("dev_19.20_epoch5_model.pt", names)
        self.assertIn("vac_smkd_best_model.pt", names)
        self.assertEqual(len(artifacts["result_files"]), 3)
        self.assertEqual(artifacts["latest_log"]["name"], "log_20260920_100000.log")

    def test_tail_log_reads_newest_log_in_work_dir(self):
        payload = tail_log(self.layout, work_dir="./work_dir/vac_smkd/", lines=5)
        self.assertIn("line 50", payload["content"])
        self.assertNotIn("line 45", payload["content"])

    def test_tail_log_by_run_id(self):
        store = RunStore(self.layout.runs_dir)
        log_path = self.layout.runs_dir / "20260101-000000-x.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("first\nsecond\nthird\n", encoding="utf-8")
        store.save(
            {"run_id": "20260101-000000-x", "log_path": str(log_path),
             "command_str": "python main.py --exp vac"}
        )
        payload = tail_log(self.layout, run_id="20260101-000000-x", store=store, lines=2)
        self.assertEqual(payload["content"], "second\nthird\n")

    def test_tail_log_requires_a_source(self):
        with self.assertRaises(McpToolError):
            tail_log(self.layout)

    def test_list_work_dirs_finds_runs_with_results(self):
        payload = list_work_dirs(self.layout, root="./work_dir/")
        self.assertEqual(payload["count"], 1)
        self.assertTrue(payload["work_dirs"][0]["path"].endswith("vac_smkd"))


if __name__ == "__main__":
    unittest.main()
