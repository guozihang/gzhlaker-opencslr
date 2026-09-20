# -*- encoding: utf-8 -*-
"""实验配置的读取与创建。"""

import unittest
from pathlib import Path

import yaml

from mcp_server.config import ExperimentConfig, _section_line_range
from mcp_server.errors import McpToolError
from mcp_server.paths import RepoLayout

from .helpers import REPO_ROOT, make_temp_repo, remove_tree


class ReadOnlyTests(unittest.TestCase):
    """针对真实仓库配置的只读断言。"""

    @classmethod
    def setUpClass(cls):
        cls.layout = RepoLayout(REPO_ROOT)
        cls.config = ExperimentConfig(cls.layout)

    def test_experiment_names_skip_anchor_sections(self):
        names = self.config.experiment_names()
        self.assertIn("baseline", names)
        self.assertIn("vac", names)
        self.assertFalse([name for name in names if name.startswith("_")])

    def test_list_experiments_reports_references(self):
        experiments = {item["name"]: item for item in self.config.list_experiments()}
        self.assertEqual(experiments["vac"]["network"], "vac")
        self.assertEqual(experiments["vac"]["dataset"], "phoenix2014")
        self.assertEqual(experiments["vac"]["problems"], [])

    def test_experiment_config_uses_config_manager_merge_order(self):
        """网络节打底、exp 节覆盖,并去掉仅作引用的 network 键。"""
        payload = self.config.get_experiment_config("corrnet")
        arguments = payload["arguments"]
        # model 来自 network.yaml 的 corrnet 节
        self.assertEqual(arguments["model"], "corrnet")
        # num_epoch 由 exp 节的 corrnet 覆盖了公共节的 80
        self.assertEqual(arguments["num_epoch"], 40)
        # 公共节里的 feeder_args 通过 YAML anchor 继承下来
        self.assertEqual(arguments["feeder_args"]["datatype"], "memmap")
        self.assertNotIn("network", arguments)
        self.assertEqual(payload["network"]["name"], "corrnet")
        self.assertEqual(payload["dataset"]["name"], "phoenix2014")

    def test_unknown_experiment_lists_alternatives(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.get_experiment_config("not-an-experiment")
        self.assertIn("vac", str(ctx.exception))

    def test_allowed_argument_names_are_extracted_from_source(self):
        names = self.config.allowed_argument_names()
        for expected in ("work_dir", "config", "exp", "phase", "device",
                         "model_args", "decode_mode", "random_fix", "num_epoch"):
            self.assertIn(expected, names)
        self.assertNotIn("not_an_argument", names)

    def test_list_options(self):
        options = self.config.list_options()
        self.assertIn("vac", options["networks"])
        self.assertEqual(options["networks"]["vac"], "vac")
        self.assertIn("CSL-Daily", options["datasets"])
        self.assertIn("baseline", options["experiments"])


class CreateTests(unittest.TestCase):
    """在临时仓库里验证写入行为(不碰真实配置文件)。"""

    def setUp(self):
        self.layout = make_temp_repo()
        self.config = ExperimentConfig(self.layout)
        self.original_text = self.layout.exp_config.read_text(encoding="utf-8")
        self.original_doc = yaml.safe_load(self.original_text)

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_create_inherits_common_section_and_keeps_other_sections(self):
        result = self.config.create_experiment(
            name="vac_tune",
            network="vac",
            dataset="phoenix2014",
            overrides={"num_epoch": 12, "model_args": {"use_bn": 0}},
            work_dir="./work_dir/vac_tune/",
            device="1",
        )
        self.assertFalse(result["replaced_existing"])
        self.assertTrue(result["inherits_common"])

        text = self.layout.exp_config.read_text(encoding="utf-8")
        # anchor 名取自文件(`_common_experiment: &common_experiment`),不是键名
        self.assertIn("<<: *common_experiment", text)

        doc = yaml.safe_load(text)
        self.assertEqual(doc["vac"], self.original_doc["vac"], "其它实验节不应被改动")
        section = doc["vac_tune"]
        # 继承公共节
        self.assertEqual(section["batch_size"], 2)
        self.assertEqual(section["optimizer_args"]["base_lr"], 0.0001)
        # 自身设置生效
        self.assertEqual(section["network"], "vac")
        self.assertEqual(section["num_epoch"], 12)
        self.assertEqual(section["model_args"]["use_bn"], 0)
        self.assertEqual(section["device"], "1")
        self.assertIn("vac_tune", self.config.experiment_names())

    def test_created_experiment_passes_real_config_validation(self):
        """新建的实验要能通过 core 里真实的配置校验(会起子进程解析)。"""
        result = self.config.create_experiment(
            name="vac_check", network="vac", dataset="phoenix2014"
        )
        self.assertTrue(result["validation"]["ok"], result["validation"])
        verdict = self.config.resolve("vac_check")
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["effective"]["model"], "vac")

    def test_duplicate_name_requires_overwrite(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.create_experiment(name="vac", network="vac", dataset="phoenix2014")
        self.assertIn("overwrite", str(ctx.exception))

    def test_overwrite_replaces_only_target_section(self):
        self.config.create_experiment(
            name="vac",
            network="tlp",
            dataset="phoenix2014",
            overwrite=True,
            num_epoch=7,
        )
        doc = yaml.safe_load(self.layout.exp_config.read_text(encoding="utf-8"))
        self.assertEqual(doc["vac"]["network"], "tlp")
        self.assertEqual(doc["vac"]["num_epoch"], 7)
        self.assertEqual(doc["corrnet"], self.original_doc["corrnet"])

    def test_unknown_argument_key_is_rejected(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.create_experiment(
                name="bad_key",
                network="vac",
                dataset="phoenix2014",
                overrides={"num_epochs": 3},
            )
        self.assertIn("num_epochs", str(ctx.exception))
        # 拒绝后不应留下任何改动
        self.assertEqual(
            self.layout.exp_config.read_text(encoding="utf-8"), self.original_text
        )

    def test_unknown_network_or_dataset_is_rejected(self):
        with self.assertRaises(McpToolError):
            self.config.create_experiment(name="x", network="nope", dataset="phoenix2014")
        with self.assertRaises(McpToolError):
            self.config.create_experiment(name="x", network="vac", dataset="nope")

    def test_invalid_experiment_name_is_rejected(self):
        for name in ("_private", "1abc", "with space"):
            with self.assertRaises(McpToolError):
                self.config.create_experiment(name=name, network="vac", dataset="phoenix2014")


class SectionRangeTests(unittest.TestCase):
    """文本级节定位(overwrite 依赖它,不能误伤相邻节)。"""

    def test_range_covers_exactly_one_section(self):
        text = (
            "a:\n"
            "  x: 1\n"
            "\n"
            "# 注释行不应结束上一节\n"
            "b:\n"
            "  y: 2\n"
            "c:\n"
            "  z: 3\n"
        )
        start, end = _section_line_range(text, "b")
        self.assertEqual(text[start:end], "b:\n  y: 2\n")
        start, end = _section_line_range(text, "c")
        self.assertEqual(text[start:end], "c:\n  z: 3\n")


if __name__ == "__main__":
    unittest.main()
