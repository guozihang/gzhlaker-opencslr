# -*- encoding: utf-8 -*-
"""core_probe:复用 core/manager 的真实解析与校验。

这些用例跑的是仓库里真实的配置代码(不需要 torch),所以它们同时验证了
「MCP 看到的配置 = 训练时用的配置」这条前提。
"""

import unittest

from mcp_server.config import ExperimentConfig
from mcp_server.core_probe import probe
from mcp_server.paths import RepoLayout

from .helpers import REPO_ROOT


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = RepoLayout(REPO_ROOT)
        cls.config = ExperimentConfig(cls.layout)
        cls.request = {"config": str(cls.layout.exp_config)}

    def test_resolves_effective_config(self):
        response = probe(dict(self.request, exp="vac"))
        self.assertTrue(response["ok"], response)
        effective = response["effective"]
        self.assertEqual(effective["model"], "vac")
        self.assertEqual(effective["dataset"], "phoenix2014")
        self.assertEqual(effective["decode_mode"], "beam")
        # dataset.yaml 的内容会被并进 dataset_info
        self.assertIn("dataset_root", effective["dataset_info"])

    def test_unknown_experiment_reports_available_names(self):
        response = probe(dict(self.request, exp="does_not_exist"))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error_type"], "ValueError")
        self.assertIn("Available", response["error"])

    def test_scalar_override_wins_over_yaml(self):
        response = probe(dict(self.request, exp="vac", overrides={"num_epoch": 3, "device": "1"}))
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["effective"]["num_epoch"], 3)
        self.assertEqual(response["effective"]["device"], "1")

    def test_dict_override_deep_merges_instead_of_replacing(self):
        """字典类配置必须像 YAML 那样深合并,否则会丢掉继承来的兄弟键。"""
        response = probe(dict(self.request, exp="vac", overrides={"model_args": {"use_bn": 0}}))
        self.assertTrue(response["ok"], response)
        model_args = response["effective"]["model_args"]
        self.assertEqual(model_args["use_bn"], 0)
        self.assertEqual(model_args["c2d_type"], "resnet18")
        self.assertEqual(model_args["num_classes"], 1296)

    def test_unknown_override_key_is_rejected(self):
        response = probe(dict(self.request, exp="vac", overrides={"typo_key": 1}))
        self.assertFalse(response["ok"])
        self.assertIn("typo_key", response["error"])

    def test_invalid_nested_values_are_caught_by_real_validator(self):
        cases = [
            {"model_args": {"typo": 1}},                 # 未知嵌套键
            {"model_args": {"num_classes": "nope"}},     # 类型错误
            {"optimizer_args": {"optimizer": "ADAM"}},   # 非法取值
            {"optimizer_args": {"base_lr": -1}},         # 非正学习率
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                response = probe(dict(self.request, exp="vac", overrides=overrides))
                self.assertFalse(response["ok"], response)

    def test_config_defaults_are_merged_for_keys_missing_from_yaml(self):
        response = probe(dict(self.request, exp="vac"))
        effective = response["effective"]
        # exp.yaml 没写 random_seed,应当落到代码默认值 0
        self.assertEqual(effective["random_seed"], 0)
        self.assertEqual(effective["save_interval"], 5)

    def test_main_reads_request_from_stdin(self):
        """子进程协议:请求从 stdin 进来,响应以单行 JSON 写出。"""
        import io
        import json
        import sys

        from mcp_server import core_probe

        stdin, stdout = sys.stdin, sys.stdout
        sys.stdin = io.StringIO('{"exp": "vac"}')
        sys.stdout = io.StringIO()
        try:
            core_probe.main()
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdin, sys.stdout = stdin, stdout

        self.assertFalse(payload["ok"])
        self.assertIn("config", payload["error"])


if __name__ == "__main__":
    unittest.main()
