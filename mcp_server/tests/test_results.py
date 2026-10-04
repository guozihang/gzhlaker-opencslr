# -*- encoding: utf-8 -*-
"""结果、日志与 checkpoint 的读取。

上游**不落盘任何结果 JSON**:WER 只能从 ``<work_dir>/log_<ts>.log`` 里按文本
解析,``get_results`` 必须如实标明来源。checkpoint 的两种命名形式也都要能找到:
``<work_dir>/_best_model.pt`` 与 ``<work_dir>dev_19.20_epoch5_model.pt``
(work_dir 同时被当作目录和文件名前缀)。
"""

import unittest

from mcp_server.errors import McpToolError
from mcp_server.results import list_artifacts, list_work_dirs, read_results, tail_log
from mcp_server.runs import RunStore

from .helpers import make_temp_repo, remove_tree, touch, write_json

# 上游 LogManager 实际写出的几行(训练流程:dict repr / Best_dev / Epoch costs;
# test 阶段:Dev WER / Test WER)
UPSTREAM_LOG = """\
2026-09-20 10:00:00.123 | INFO     | manager.experiment_manager:run_train:116 - {'Dev': 21.34}
2026-09-20 10:00:01.456 | INFO     | manager.experiment_manager:run_train:117 - {'Test': 22.1}
2026-09-20 10:00:02.789 | INFO     | manager.experiment_manager:run_train:120 - Best_dev: 20.15, Epoch : 3
2026-09-20 10:05:00.000 | INFO     | manager.experiment_manager:run_train:143 - Dev WER: 19.90
2026-09-20 10:06:00.000 | INFO     | manager.experiment_manager:run_train:143 - Test WER: 20.30
2026-09-20 10:07:00.000 | INFO     | manager.experiment_manager:run_train:100 - Epoch 5 costs 123.4
"""


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.layout = make_temp_repo()
        self.work_root = self.layout.core_dir / "work_dir"
        self.work_dir = self.work_root / "vac_smkd"
        self.work_dir.mkdir(parents=True)
        self.log_file = self.work_dir / "log_20260920_100000.log"
        self.log_file.write_text(UPSTREAM_LOG, encoding="utf-8")

        # work_dir 以 "/" 结尾时,ExperimentManager 的两个 format 串都落在这个目录里
        touch(self.work_dir / "_best_model.pt")
        touch(self.work_dir / "dev_19.20_epoch5_model.pt")
        # work_dir 不带结尾斜杠时,checkpoint 会写成同级的「文件名前缀」形式
        touch(self.work_root / "vac_smkd_best_model.pt")

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_summary_is_parsed_from_the_upstream_log(self):
        payload = read_results(self.layout, "./work_dir/vac_smkd/")
        self.assertTrue(payload["exists"])
        self.assertEqual(payload["log_file"], str(self.log_file))
        # 没有任何结果 JSON,来源必须如实标成 log
        self.assertEqual(payload["source"], "log")
        self.assertEqual(payload["results"], {})

        metrics = payload["summary"]["from_log"]
        self.assertEqual(metrics["dev_wer"], 19.90, "后出现的 Dev WER 行应当覆盖 dict repr")
        self.assertEqual(metrics["test_wer"], 20.30)
        self.assertEqual(metrics["best_dev_wer"], 20.15)
        self.assertEqual(metrics["best_epoch"], 3)
        self.assertEqual(metrics["last_epoch"], 5)
        self.assertEqual(metrics["lines_scanned"], 6)
        self.assertIn("不落盘结果 JSON", payload["note"])

    def test_missing_work_dir_explains_the_next_step(self):
        payload = read_results(self.layout, "./work_dir/never_ran/")
        self.assertFalse(payload["exists"])
        self.assertIn("目录不存在", payload["note"])

    def test_legacy_result_json_is_read_alongside_the_log(self):
        write_json(
            self.work_dir / "experiment_result.json",
            {"experiment_name": "vac_smkd", "split": "test", "wer": 19.2, "status": "valid"},
        )
        payload = read_results(self.layout, "./work_dir/vac_smkd/")
        self.assertEqual(payload["source"], "log+json")
        self.assertEqual(payload["summary"]["experiment_result"]["wer"], 19.2)
        self.assertIn("from_log", payload["summary"])

    def test_list_artifacts_finds_both_checkpoint_naming_forms(self):
        artifacts = list_artifacts(self.layout, "./work_dir/vac_smkd/")
        names = {item["name"] for item in artifacts["checkpoints"]}
        self.assertIn("_best_model.pt", names)
        self.assertIn("dev_19.20_epoch5_model.pt", names)
        self.assertIn("vac_smkd_best_model.pt", names)
        self.assertEqual(artifacts["result_files"], [])
        self.assertEqual(artifacts["latest_log"]["name"], self.log_file.name)
        self.assertTrue(artifacts["logs"][0]["is_log"])

    def test_tail_log_reads_the_newest_log_in_work_dir(self):
        payload = tail_log(self.layout, work_dir="./work_dir/vac_smkd/", lines=3)
        self.assertEqual(payload["path"], str(self.log_file))
        self.assertIn("Epoch 5 costs", payload["content"])
        self.assertNotIn("{'Dev': 21.34}", payload["content"], "只返回末尾 N 行")

    def test_tail_log_by_run_id(self):
        store = RunStore(self.layout.runs_dir)
        log_path = self.layout.runs_dir / "20260101-000000-x.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("first\nsecond\nthird\n", encoding="utf-8")
        store.save(
            {"run_id": "20260101-000000-x", "log_path": str(log_path),
             "command_str": "python main.py --config configs/vac.yaml --phase train"}
        )
        payload = tail_log(self.layout, run_id="20260101-000000-x", store=store, lines=2)
        self.assertEqual(payload["source"], "run")
        self.assertEqual(payload["content"], "second\nthird\n")

    def test_tail_log_requires_a_source(self):
        with self.assertRaises(McpToolError) as ctx:
            tail_log(self.layout)
        self.assertIn("work_dir", str(ctx.exception))

    def test_list_work_dirs_finds_dirs_with_result_files(self):
        write_json(self.work_dir / "experiment_result.json", {"split": "test"})
        payload = list_work_dirs(self.layout, root="./work_dir/")
        self.assertEqual(payload["count"], 1)
        self.assertTrue(payload["work_dirs"][0]["path"].endswith("vac_smkd"))
        self.assertEqual(payload["work_dirs"][0]["result_files"], ["experiment_result.json"])


if __name__ == "__main__":
    unittest.main()
