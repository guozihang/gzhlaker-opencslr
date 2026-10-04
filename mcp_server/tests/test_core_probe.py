# -*- encoding: utf-8 -*-
"""core_probe:在 core/ 里跑上游真实的解析链(不需要 torch)。

用例按**线上协议**调用探针子进程(请求走 stdin、响应是最后一行 JSON),拿到的
``effective`` 就是训练启动时 ``ArgumentManager`` 真正会用的那份配置。上游
``ConfigManager`` 没有嵌套键/取值校验,这里也如实反映,不假装有白名单。
"""

import unittest

import yaml

from mcp_server.paths import RepoLayout

from .helpers import REPO_ROOT, run_probe

VAC_CONFIG = REPO_ROOT / "core" / "configs" / "vac.yaml"
VAC_DOC = yaml.safe_load(VAC_CONFIG.read_text(encoding="utf-8"))


def request(overrides=None, mode=None, config=VAC_CONFIG):
    payload = {"config": str(config)}
    if overrides is not None:
        payload["overrides"] = overrides
    if mode:
        payload["mode"] = mode
    return payload


class ProbeTests(unittest.TestCase):
    """探针的解析结果必须等于配置文件 + 覆盖 + 上游默认值。"""

    def test_resolve_returns_the_effective_config(self):
        response = run_probe(request(overrides={}))
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["config_path"], str(VAC_CONFIG))
        effective = response["effective"]
        self.assertEqual(effective["model"], "models.build_function.build_vac")
        self.assertEqual(effective["dataset"], "phoenix2014")
        self.assertEqual(effective["decode_mode"], "beam")
        self.assertEqual(effective["num_epoch"], 80)
        # map() 把 configs/<dataset>.yaml 读进 dataset_info
        self.assertIn("dict_path", effective["dataset_info"])

    def test_scalar_overrides_win_over_the_config_file(self):
        response = run_probe(request(overrides={"num_epoch": 3, "device": "1"}))
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["effective"]["num_epoch"], 3)
        self.assertEqual(response["effective"]["device"], "1")
        self.assertEqual(response["overrides"], {"num_epoch": 3, "device": "1"})

    def test_nested_dict_override_deep_merges_so_siblings_survive(self):
        response = run_probe(request(overrides={"model_args": {"use_bn": 0}}))
        self.assertTrue(response["ok"], response)
        model_args = response["effective"]["model_args"]
        self.assertEqual(model_args["use_bn"], 0)
        # 只改一个嵌套键,同级的兄弟键必须来自配置文件
        for key, value in VAC_DOC["model_args"].items():
            if key == "use_bn":
                continue
            self.assertEqual(model_args[key], value, f"{key} 被整体替换丢掉了")

    def test_unknown_override_key_returns_a_clean_error(self):
        response = run_probe(request(overrides={"typo_key": 1}))
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error_type"], "ValueError")
        self.assertIn("typo_key", response["error"])
        self.assertNotIn("Traceback", response["error"])

    def test_invalid_value_reports_the_argparse_message(self):
        response = run_probe(request(overrides={"num_epoch": "abc"}))
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error_type"], "ValueError")
        self.assertIn("--num-epoch", response["error"])
        self.assertIn("invalid int value", response["error"])
        self.assertIn("abc", response["error"])

    def test_schema_mode_reports_real_values_and_no_whitelist(self):
        response = run_probe(request(mode="schema"))
        self.assertTrue(response["ok"], response)

        # 上游没有运行时控制面:不能报出一张假的「可热改键」表
        self.assertIsNone(response["hot"]["keys"])
        self.assertIn("没有运行时控制面", response["hot"]["note"])

        # 参数表来自 main.py 的 parser
        names = {item["name"] for item in response["arguments"]}
        for expected in ("config", "model", "dataset", "num_epoch", "model_args"):
            self.assertIn(expected, names)
        self.assertNotIn("exp", names)

        # 嵌套节只有当前值;allowed 一律为 null(没有白名单)
        nested = response["nested"]["model_args"]
        self.assertIsNone(nested["allowed"])
        self.assertEqual(nested["current"]["num_classes"], VAC_DOC["model_args"]["num_classes"])

        # schema 模式顺带回传一次生效配置
        self.assertTrue(response["resolution"]["ok"], response["resolution"])
        self.assertEqual(
            response["resolution"]["effective"]["model"],
            "models.build_function.build_vac",
        )

    def test_request_without_config_is_rejected(self):
        response = run_probe({})
        self.assertFalse(response["ok"], response)
        self.assertIn("config", response["error"])

    def test_missing_config_file_is_reported_not_raised(self):
        missing = REPO_ROOT / "core" / "configs" / "no_such_experiment.yaml"
        response = run_probe(request(config=missing))
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error_type"], "FileNotFoundError")
        self.assertIn("no_such_experiment.yaml", response["error"])


if __name__ == "__main__":
    unittest.main()
